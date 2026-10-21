#!/usr/bin/env python3
"""Replay saved TrajRisk models and derived checks without fitting any model.

Inputs are the author-supplied Repository Check 2 packet, not raw expression.
The packet contains trusted author-generated joblib models; do not substitute
untrusted pickle files. Scientific functions are loaded from explicitly named,
hash-verified source definitions without executing script-level training code.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
from pathlib import Path
import sys
import types
import zipfile
from typing import Dict, List, Tuple

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[1]
SCALES = ["Scale1_AcuteStress", "Scale2_PeakInjury", "Scale3_FateDecision", "Scale4_Resolution"]
log = logging.getLogger("trajrisk.replay")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_bundle(path, destination):
    path = path.expanduser().resolve()
    if path.is_file():
        release = json.loads((REPO / "metadata/frozen_release.json").read_text())
        if sha256(path) != release["archive_sha256"]:
            raise ValueError("Input ZIP differs from the verified frozen archive")
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                relative = Path(member.filename)
                target = (destination / relative).resolve()
                if relative.is_absolute() or not target.is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe archive path: " + member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Symbolic links are not supported in the input packet")
            archive.extractall(destination)
        path = destination
    if not (path / "PACKET_MANIFEST.json").is_file():
        matches = list(path.glob("*/PACKET_MANIFEST.json"))
        if len(matches) != 1:
            raise ValueError("Provide the Check 2 ZIP or its extracted packet directory")
        path = matches[0].parent
    manifest = json.loads((path / "PACKET_MANIFEST.json").read_text())
    expected_manifest = json.loads((REPO / "metadata/frozen_input_manifest.json").read_text())
    if manifest != expected_manifest:
        raise ValueError("Input packet differs from the frozen version recorded by this repository")
    for item in manifest["files"]:
        candidate = (path / item["path"]).resolve()
        if not candidate.is_relative_to(path.resolve()):
            raise ValueError("Unsafe manifest path")
        if candidate.stat().st_size != item["bytes"] or sha256(candidate) != item["sha256"]:
            raise ValueError("Input checksum mismatch: " + item["path"])
    return path, manifest


def source_module(filename, names):
    provenance = json.loads((REPO / "metadata/workflow_sources.json").read_text())
    relative = "scripts/" + filename
    record = next(row for row in provenance["files"] if row.get("repository_path") == relative)
    source = REPO / relative
    if sha256(source) != record["portable_sha256"]:
        raise ValueError("Source hash differs from the recorded replay source: " + relative)
    tree = ast.parse(source.read_text(), filename=str(source))
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in definitions} != set(names):
        raise ValueError("Missing required source definition")
    module = types.ModuleType("_trajrisk_replay_" + filename.removesuffix(".py"))
    module.__file__ = str(source)
    module.__dict__.update(
        torch=torch, nn=nn, MessagePassing=MessagePassing, np=np, pd=pd,
        NearestNeighbors=NearestNeighbors, XGBClassifier=XGBClassifier,
        Dict=Dict, List=List, Tuple=Tuple,
        sc=types.SimpleNamespace(AnnData=object), log=log,
        NODE_DIM=31, HIDDEN_DIM=64, N_SCALES=4, DROPOUT=0.3,
        SCALE_NAMES=SCALES, KNN_K=15, RANDOM_SEED=42, N_BOOTSTRAP=1000,
        mannwhitneyu=mannwhitneyu, roc_auc_score=roc_auc_score, spearmanr=spearmanr)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), module.__dict__)
    return module


def read_csv(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def compact(packet, key):
    folder = packet / "compact" / key
    obs = read_csv(folder / "observations.csv", index_col=0)
    metadata = json.loads((folder / "metadata.json").read_text())
    coordinates = folder / "coordinates.npz"
    arrays = {}
    if coordinates.exists():
        with np.load(coordinates, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    return obs, arrays, metadata


class Replay:
    def __init__(self, packet, manifest, output, bootstrap):
        self.packet, self.output, self.bootstrap = packet, output, bootstrap
        self.originals = packet / "originals"
        self.results = self.originals / "results"
        self.processed = self.originals / "Data/processed"
        self.checks = []
        self.notes = []
        self.manifest = manifest

    def check(self, name, passed, **evidence):
        row = {"check": name, "passed": bool(passed), **evidence}
        self.checks.append(row)
        print(("PASS: " if passed else "FAIL: ") + name, flush=True)
        if not passed:
            raise ValueError(name + ": " + json.dumps(evidence))

    def close(self, name, actual, expected, atol=2e-6, rtol=0):
        a, e = np.asarray(actual), np.asarray(expected)
        delta = float(np.max(np.abs(a.astype(float) - e.astype(float)))) if a.shape == e.shape and a.size else 0.
        self.check(name, a.shape == e.shape and np.allclose(a, e, atol=atol, rtol=rtol),
                   max_absolute_difference=delta, absolute_tolerance=atol, relative_tolerance=rtol)

    def table(self, filename):
        return read_csv(REPO / "manuscript/tables/data" / filename)

    def reference(self):
        self.mouse_source = source_module("validate_mouse_transfer.py", [
            "TrajRiskConv", "TrajRiskGNN", "get_scale_and_anchors_for_cell",
            "build_cell_specific_reference_edges", "bootstrap_auroc_ci"])
        self.human_source = source_module("validate_human_transfer.py", [
            "prepare_ortholog_gene_space", "_gaussian_knn_edges", "_assign_scale", "compute_human_risk_scores"])
        self.x = torch.load(self.processed / "week6_node_features.pt", map_location="cpu", weights_only=True)
        self.graphs = torch.load(self.processed / "multiscale_graphs.pt", map_location="cpu", weights_only=True)
        self.eis = [self.graphs[scale]["edge_index"] for scale in SCALES]
        self.ews = [self.graphs[scale]["edge_weight"] for scale in SCALES]
        self.ci = read_csv(self.processed / "walesky_cell_index.csv", index_col=0)
        self.saved = read_csv(self.processed / "week7_predictions.csv", index_col=0)
        self.check("Reference barcode order", self.saved.index.equals(self.ci.index))
        self.check("Reference features: 6,582 x 31", tuple(self.x.shape) == (6582, 31))
        self.check("Four graph edge counts", [int(e.shape[1]) for e in self.eis] == [167633, 63673, 117408, 72763])
        checkpoint = torch.load(self.originals / "models/week7_trajrisk_FINAL.pt", map_location="cpu", weights_only=True)
        self.model = self.mouse_source.TrajRiskGNN().eval()
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.check("87,233 trainable parameters", sum(p.numel() for p in self.model.parameters()) == 87233)
        captured = []
        hook = self.model.fusion_mlp.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        with torch.inference_mode():
            score = self.model(self.x, self.eis, self.ews).numpy()
        hook.remove()
        embeddings = captured[0].numpy().reshape(6582, 4, 64)
        self.close("Reference frozen scores", score, self.saved.risk_predicted.to_numpy())
        self.close("Four saved scale embeddings", embeddings, np.load(self.processed / "week7_scale_embeddings.npy"), atol=3e-5, rtol=1e-5)
        training = torch.load(self.processed / "week6_training_data.pt", map_location="cpu", weights_only=True)
        val = training["val_idx"].numpy()
        target = self.saved.pseudotime_norm.to_numpy()
        self.close("Validation MSE", mean_squared_error(target[val], score[val]), checkpoint["final_val_metrics"]["loss"], atol=2e-8)
        self.check("Validation AUROC rounds to 0.9071", round(roc_auc_score(target[val] > .10, score[val]), 4) == .9071)
        pd.DataFrame({"cell_id": self.saved.index, "risk_predicted": score}).to_csv(self.output / "reference_predictions.csv", index=False)
        np.save(self.output / "reference_scale_embeddings.npy", embeddings)

    def mouse(self):
        obs, arrays, _ = compact(self.packet, "mouse_integration")
        self.close("Reference latent axis in integrated object", arrays["X_scvi"][obs.index.get_indexer(self.ci.index)], self.x[:, :30].numpy(), atol=0)
        seeder = joblib.load(self.originals / "models/week6_xgboost_model.pkl")
        table = self.table("Table_S3A_external_mouse_validation_FINAL.csv")
        method_names = {"trajrisk_knn": "TrajRisk Reference-KNN", "trajrisk_wself": "TrajRisk W_self", "seeder": "Initial XGBoost seed"}
        expected_edges = {"Ben_Moshe": [29985, 0, 7980, 3585], "Umbaugh_2h": [321135, 0, 0, 0], "Umbaugh_48h72h": [34920, 0, 21615, 23055]}
        task_fragments = {"Ben_Moshe": "Ben-Moshe", "Umbaugh_2h": "Umbaugh 2h", "Umbaugh_48h72h": "Umbaugh 48/72h"}
        records = []
        score_frames = []
        for dataset in expected_edges:
            print("Replaying external cohort: " + dataset, flush=True)
            mask = obs.dataset.eq(dataset).to_numpy()
            current = obs.loc[mask].copy()
            latent = arrays["X_scvi"][mask].astype(np.float32)
            seed = seeder.predict_proba(latent)[:, 1].astype(np.float32)
            features = torch.from_numpy(np.vstack([self.x.numpy(), np.column_stack([latent, seed])]).astype(np.float32))
            with torch.inference_mode():
                wself = self.model(features, self.eis, self.ews).numpy()[6582:]
            ref_eis, ref_ews = self.mouse_source.build_cell_specific_reference_edges(
                {"x_walesky": self.x, "cell_index": self.ci}, latent, current, dataset, k=15)
            self.check(dataset + " reference-edge counts", [int(e.shape[1]) for e in ref_eis] == expected_edges[dataset])
            with torch.inference_mode():
                knn = self.model(features, [torch.cat([a, b], 1) for a, b in zip(self.eis, ref_eis)],
                                 [torch.cat([a, b]) for a, b in zip(self.ews, ref_ews)]).numpy()[6582:]
            scores = {"trajrisk_knn": knn, "trajrisk_wself": wself, "seeder": seed}
            score_frames.append(pd.DataFrame({"cell_id": current.index, "dataset": dataset, **scores}))
            y = current.treatment.eq("APAP").to_numpy().astype(int)
            candidates = table.loc[table.Dataset_task.str.startswith(task_fragments[dataset]) & ~table.Dataset_task.str.contains("Cdkn1a", case=False)]
            for key, values in scores.items():
                expected = candidates.loc[candidates.Method.eq(method_names[key])]
                self.check(dataset + " / " + key + " table match", len(expected) == 1)
                expected = expected.iloc[0]
                auc, ap = roc_auc_score(y, values), average_precision_score(y, values)
                self.check(dataset + " / " + key + " AUROC and AUPRC", round(auc, 4) == round(expected.AUROC, 4) and round(ap, 4) == round(expected.AUPRC, 4), auroc=float(auc), auprc=float(ap))
                self.check(dataset + " / " + key + " cohort denominator", len(y) == int(expected.N) and int(y.sum()) == int(expected.N_positive))
                row = {"dataset": dataset, "method": key, "n": len(y), "n_positive": int(y.sum()), "auroc": float(auc), "auprc": float(ap)}
                if self.bootstrap:
                    lo, hi = self.mouse_source.bootstrap_auroc_ci(y, values)
                    self.check(dataset + " / " + key + " 1,000-bootstrap AUROC CI", round(lo, 4) == round(expected.AUROC_95CI_lower, 4) and round(hi, 4) == round(expected.AUROC_95CI_upper, 4), lower=float(lo), upper=float(hi))
                    row.update(auroc_ci_lower=lo, auroc_ci_upper=hi)
                records.append(row)
        self.mouse_scores = pd.concat(score_frames, ignore_index=True)
        self.mouse_scores.to_csv(self.output / "external_mouse_predictions.csv", index=False)
        pd.DataFrame(records).to_csv(self.output / "external_mouse_metrics.csv", index=False)
        self.notes.append("Cdkn1a labels require expression values and were not regenerated from the compact packet. B1 is not refitted or assigned placeholder predictions.")
        self.notes.append("The historical manuscript_mouse_cell_scores.csv is not used as a corrected Figure 5B score source. Use the frozen-replay external_mouse_predictions.csv.")

    def human(self):
        obs, arrays, _ = compact(self.packet, "cross_species")
        self.check("Cross-species reference barcode order", obs.index[:6582].equals(self.ci.index.astype(str) + "-walesky"))
        mouse_genes = read_csv(self.packet / "compact/mouse_integration/gene_axis.csv").iloc[:, 0]
        human_genes = read_csv(self.packet / "compact/matchett_human_qc/gene_axis.csv").iloc[:, 0]
        orthologs = self.human_source.prepare_ortholog_gene_space({
            "orthologs": read_csv(self.processed / "mouse_human_orthologs_biomart.csv"),
            "adata_walesky": types.SimpleNamespace(var_names=mouse_genes),
            "human": types.SimpleNamespace(var_names=human_genes)})
        self.check("Strict shared mouse-human symbol pairs", len(orthologs["orth_usable"]) == 13122)
        combined_axis = read_csv(self.packet / "compact/cross_species/gene_axis.csv").iloc[:, 0].astype(str).tolist()
        self.check("Cross-species expression-axis gene order", combined_axis == orthologs["shared_human"])
        human = obs.iloc[6582:].copy()
        saved = read_csv(self.results / "week9b_cell_risk_scores.csv", index_col=0)
        self.check("Human barcode order", human.index.equals(saved.index.astype(str) + "-human"))
        model = self.mouse_source.TrajRiskGNN().eval()
        checkpoint = torch.load(self.originals / "models/week9b_trajrisk_cross_species.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        seeder = joblib.load(self.originals / "models/week9b_cross_species_seeder.pkl")
        risks = self.human_source.compute_human_risk_scores(
            types.SimpleNamespace(obsm=arrays),
            {"edge_indices": self.eis, "edge_weights": self.ews, "walesky_hours": self.ci.hours.to_numpy(), "human": types.SimpleNamespace(obs=human)},
            model, seeder, 6582)
        for key in ("risk_2a", "risk_2b"):
            self.close("Human frozen " + key, risks[key], saved[key].to_numpy())
        frame = pd.DataFrame({"cell_id": saved.index, "patient_id": human.patient_id.to_numpy(), "condition": human.condition.to_numpy(), "risk_2a": risks["risk_2a"], "risk_2b": risks["risk_2b"]})
        frame["patient_id"] = frame.patient_id.replace({"apap09A": "apap09", "apap09B": "apap09"})
        frame.to_csv(self.output / "human_predictions.csv", index=False)
        summary = json.loads((self.results / "week9b_summary.json").read_text())
        all_donors = frame.groupby("patient_id", sort=True).agg(condition=("condition", "first"), n_nuclei=("cell_id", "size"), risk_2a=("risk_2a", "mean"), risk_2b=("risk_2b", "mean"))
        for cohort in ("primary", "sensitivity"):
            donors = all_donors if cohort == "primary" else all_donors.loc[all_donors.n_nuclei.ge(300)]
            expected = summary[cohort]
            positive = donors.condition.eq("APAP_ALF").to_numpy()
            self.check(cohort + " donor counts", int(positive.sum()) == expected["n_alf"] and int((~positive).sum()) == expected["n_healthy"])
            for key in ("risk_2a", "risk_2b"):
                scores = donors[key].to_numpy()
                auc = roc_auc_score(positive, scores)
                p = mannwhitneyu(scores[positive], scores[~positive], alternative="greater").pvalue
                self.close(cohort + " " + key + " donor AUROC", auc, expected[key + "_auroc"], atol=1e-12)
                self.close(cohort + " " + key + " donor one-sided MW p", p, expected[key + "_mw_pval"], atol=1e-12)
                self.close(cohort + " " + key + " donor means", [scores[positive].mean(), scores[~positive].mean()], [expected[key + "_alf_mean"], expected[key + "_healthy_mean"]])
            donors.to_csv(self.output / ("human_donor_" + cohort + ".csv"))
        self.notes.append("Human signature gene selection and normalization require expression matrices; frozen risk scoring and donor statistics are replayed here, not signature discovery.")

    def graph_derived(self):
        edges = read_csv(self.results / "bottleneck_s5_all_edges_with_shift.csv")
        sources = edges.source_node.to_numpy(dtype=int)
        targets = edges.target_node.to_numpy(dtype=int)
        score = self.saved.risk_predicted.to_numpy(dtype=np.float32)
        expected_flux = edges.edge_weight.to_numpy(dtype=np.float32) * score[sources]
        expected_amp = score[targets] - score[sources]
        self.close("All-edge flux = weight x source score", edges.flux.to_numpy(), expected_flux, atol=1e-7, rtol=2e-6)
        self.close("All-edge signed amplification = target - source", edges.amplification.to_numpy(), expected_amp, atol=2e-7)
        counts = []
        for scale in SCALES:
            subset = edges.loc[edges.scale.eq(scale)]
            count = int((subset.flux.rank(method="average", pct=True) * 100 >= 99).sum())
            counts.append(count)
        self.check("Within-scale top-percentile bottleneck counts", counts == [1677, 637, 1175, 728], counts=counts)
        sources24 = np.flatnonzero(self.ci.hours.to_numpy() == 24)
        pt = self.saved.pseudotime_norm.to_numpy()[sources24]
        cutoffs = np.quantile(pt, [.25, .5, .75], method="linear")
        labels = np.searchsorted(cutoffs, pt, side="right")
        scale3 = self.eis[2][0].numpy()
        edge_counts = [int(np.isin(scale3, sources24[labels == q]).sum()) for q in range(4)]
        self.check("S6F higher-quartile boundary ties and edge counts", edge_counts == [30692, 30208, 30739, 25769], source_cells=[int((labels == q).sum()) for q in range(4)], cutoffs=cutoffs.tolist())
        self.check("S6F normalized source values do not exceed 1", bool(np.all((pt >= 0) & (pt <= 1))))
        mapping = read_csv(self.results / "week12_human_de_all_bottleneck_genes.csv")
        unique = mapping.drop_duplicates("human_gene")
        significant = unique.loc[unique.pval_human.le(.05)]
        self.check("Distinct human genes significant at nominal p <= .05", len(unique) == 2534 and len(significant) == 1023, fraction=float(len(significant) / len(unique)))
        records = []
        expected_tested, expected_significant = [161, 511, 1674, 906], [75, 238, 633, 380]
        for idx, scale in enumerate(SCALES):
            de = read_csv(self.results / ("bottleneck_s3_de_" + scale.lower() + ".csv"))
            selected = de.loc[de.pvalue.lt(.05) & de.log2fc.abs().gt(.20), "gene"]
            mapped = mapping.loc[mapping.mouse_gene.isin(selected)].drop_duplicates("human_gene")
            n_sig = int(mapped.pval_human.le(.05).sum())
            self.check(scale + " corrected distinct-human significance counts", len(mapped) == expected_tested[idx] and n_sig == expected_significant[idx])
            records.append({"scale": scale, "unique_human_genes": len(mapped), "significant_unique_human_genes": n_sig, "percentage": 100 * n_sig / len(mapped)})
        pd.DataFrame(records).to_csv(self.output / "figure6e_distinct_gene_counts.csv", index=False)
        self.notes.append("Per-scale conservation uses the previously audited mouse-selection rule; the original conservation/pathway generator is not recovered or rerun.")

    def write_report(self, error=None):
        packages = ["torch", "torch-geometric", "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib"]
        report = {
            "status": "passed" if error is None else "failed",
            "input_packet_files_verified": len(self.manifest["files"]),
            "input_packet_manifest_sha256": sha256(self.packet / "PACKET_MANIFEST.json"),
            "environment": {"python": platform.python_version(), "platform": platform.system(), **{name: importlib.metadata.version(name) for name in packages}},
            "bootstrap_ci_replayed": self.bootstrap,
            "checks": self.checks, "notes": self.notes,
            "error": error,
            "scope": "Frozen inference and saved-output numerical replay. No training, expression reconstruction, R rerun, or canonical figure replacement.",
            "remaining": ["Clean raw-expression preprocessing/training replay", "Final main-figure generator alignment", "Full training-environment installation including scVI; original RStudio versions remain user-confirmed"],
        }
        (self.output / "reproduction_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Author-supplied Check 2 ZIP or extracted folder")
    parser.add_argument("--output", type=Path, default=REPO / "build/frozen_replay")
    parser.add_argument("--bootstrap-ci", action="store_true", help="Also replay the original 1,000-bootstrap mouse AUROC CIs")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output == args.bundle.expanduser().resolve() or output.is_relative_to(REPO / "manuscript"):
        parser.error("Use a separate build output directory")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    packet, manifest = open_bundle(args.bundle, output / "inputs")
    replay = Replay(packet, manifest, output, args.bootstrap_ci)
    try:
        replay.reference()
        replay.mouse()
        replay.human()
        replay.graph_derived()
        changed = []
        for item in manifest["files"]:
            path = packet / item["path"]
            if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                changed.append(item["path"])
        replay.check("All input files unchanged after replay", not changed, files_checked=len(manifest["files"]), changed=changed)
    except Exception as error:
        replay.write_report(type(error).__name__ + ": " + str(error))
        raise
    report = replay.write_report()
    print("\nFrozen replay complete: %d checks passed" % len(report["checks"]))
    print("Outputs: " + str(output))


if __name__ == "__main__":
    main()
