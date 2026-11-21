"""Build the 20 supplementary table components from replay and study outputs.

Each component is checked against the manuscript CSV. The report identifies
archived training results and the precision-sensitive Matchett test fields.
This does not refit every upstream model.
"""
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import entropy, fisher_exact, mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
import torch

ROOT = Path(__file__).resolve().parents[1]
SCALES = ("Scale1_AcuteStress", "Scale2_PeakInjury", "Scale3_FateDecision", "Scale4_Resolution")
DISPLAY = ("Acute Stress", "Peak Injury", "Fate Decision", "Resolution")
TIMES = ((0, 6), (6, 24), (24, 48), (48, 96))
LABELS = [f"Scale {s} — {name}" for s, name in enumerate(DISPLAY, 1)]
METHODS = {
    "trajrisk_knn": "TrajRisk Reference-KNN", "trajrisk_wself": "TrajRisk W_self",
    "seeder": "Initial XGBoost seed", "b1_xgb": "Expression-only XGBoost",
}
MOUSE_TASKS = [
    ("Ben_Moshe", "T1:", "Ben-Moshe 48/72h — APAP vs Control"),
    ("Umbaugh_2h", "T2:", "Umbaugh 2h — APAP vs Control"),
    ("Umbaugh_48h72h", "T3b:", "Umbaugh 48/72h — APAP vs Control"),
]


def csv(path):
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False)


def flat_row(row, fields):
    return [getattr(row, key) for key in fields]


class Tables:
    def __init__(self, replay):
        self.replay = replay
        self.output = replay.output / "supplementary_tables"
        self.output.mkdir(exist_ok=True)
        self.source = ROOT / "manuscript/tables/data"
        self.results = replay.results
        self.archived = replay.originals / "Supplementary/tables"
        self.coverage = []
        self.hours = replay.ci.hours.to_numpy()
        self.pt = replay.saved.pseudotime_norm.to_numpy()
        self.risk = replay.saved.risk_predicted.to_numpy()
        self.treatment = replay.ci.treatment_str.to_numpy()

    def table_path(self, name):
        paths = list(self.source.glob(f"Table_{name}_*.csv"))
        if len(paths) != 1:
            raise ValueError(f"Expected one manuscript CSV for Table {name}, found {len(paths)}")
        return paths[0]

    def emit(self, name, rows, evidence, retained=None, *, rtol=2e-6, atol=1e-11):
        path = self.table_path(name)
        expected = csv(path)
        actual = pd.DataFrame(rows, columns=expected.columns)
        if actual.shape != expected.shape:
            raise ValueError(f"Table {name}: generated {actual.shape}, expected {expected.shape}")
        for col in actual:
            for i, (a, b) in enumerate(zip(actual[col], expected[col]), 1):
                try:
                    x, y = float(a), float(b)
                    # A large absolute tolerance would falsely pass small p values.
                    p_col = col.lower().endswith("_p") or "p_value" in col.lower() or "_p_" in col.lower()
                    equal = x == y or math.isclose(x, y, rel_tol=rtol, abs_tol=0 if p_col else atol)
                except (ValueError, TypeError):
                    equal = str(a) == str(b)
                if not equal:
                    raise ValueError(f"Table {name} row {i}, {col}: generated {a!r}, manuscript {b!r}")
        actual.to_csv(self.output / path.name, index=False)
        self.coverage.append({"component": name, "file": path.name, "rows": len(actual),
                              "compared_cells": actual.size, "generated_from": evidence,
                              "retained_original_results": retained or ""})
        print(f"PASS: Table {name} ({len(actual)} rows)", flush=True)

    def transitions(self):
        summary = json.loads((self.results / "week5_transition_summary.json").read_text())["scales"]
        rows = []
        n = len(self.hours)
        for j, scale in enumerate(SCALES):
            src, dst = TIMES[j]
            edge_index, weights = self.replay.eis[j].numpy(), self.replay.ews[j].numpy()
            ent = np.zeros(n)
            order = np.argsort(edge_index[0], kind="stable")
            sources, first = np.unique(edge_index[0][order], return_index=True)
            for cell, this_row in zip(sources, np.split(weights[order], first[1:])):
                ent[cell] = entropy(this_row)
            test = summary[scale]
            cv = np.asarray(test["cv_auroc_folds"])
            values = ent[self.hours == src]
            rows.append([
                f"Scale {j+1} ({DISPLAY[j]})", f"{src}h → {dst}h", dst-src, test["n_training_pairs"],
                f"{cv.mean():.4f} ± {cv.std():.4f}", round(test["oob_score"], 3), len(weights),
                round(100 * (1 - len(weights) / n**2), 3), f"{values.mean():.2f} ± {values.std():.2f}",
                round(test["gaussian_comparison"]["mean_row_pearson_vs_gaussian"], 3),
            ])
        self.emit("S1", rows, "Saved graph weights, training-pair counts and RF fold scores",
                  "Original RF fitting, out-of-bag scores and RF–Gaussian correlations")

    def reference_comparisons(self):
        val = self.replay.training["val_idx"].numpy()
        y = self.pt[val] > .10
        full = self.replay.reference_scores[val]
        seed = self.replay.x[val, 30].numpy()

        def metrics(target, score):
            return round(roc_auc_score(target, score), 4), round(average_precision_score(target, score), 4)

        def ci(target, score):
            return [round(x, 4) for x in self.replay.mouse_source.bootstrap_auroc_ci(target, score)]

        comparison = csv(self.results / "week8_comparison_table.csv")
        rows = []
        for item in comparison.itertuples(index=False):
            if item.Method == "TrajRisk (full)":
                auc, ap = metrics(y, full)
                rows.append([item.Method, item.Type, auc, *ci(y, full), ap,
                             round(spearmanr(self.pt[val], full).statistic, 4), 0, "—"])
                seed_auc, seed_ap = metrics(y, seed)
                rows.append(["Initial XGBoost seed (W6)", "Baseline", seed_auc, *ci(y, seed), seed_ap,
                             "", round(100 * (auc - seed_auc), 2), "—"])
            else:
                p = float(item.p_value_bootstrap)
                rows.append([*flat_row(item, ["Method", "Type", "AUROC", "AUROC_CI_lower", "AUROC_CI_upper",
                                              "AUPRC", "Spearman_rho", "Delta_vs_TrajRisk_pp"]),
                             "<0.001" if p < .001 else f"{p:.3f}"])
        self.emit("S2A", rows, "Saved validation split, reference/seed scores, and the original comparison output",
                  "Trained comparator and ablation results other than the seed")

        original = csv(self.results / "week8_singlescale_all_results.csv")
        rows = [
            [f"Scale {i+1} ({DISPLAY[i]}; {TIMES[i][0]}h→{TIMES[i][1]}h)",
             item.dev_auroc, item.val_auroc, item.selected]
            for i, item in enumerate(original.itertuples(index=False))
        ]
        self.emit("S2B", rows, "Single-scale training summary, with the manuscript scale names",
                  "Results of training the four single-scale models")

        original = csv(self.results / "week8_supplementary_table.csv")
        rows = []
        for item in original.loc[original.subset.isin(["Non-6h", "24h", "48h", "96h"])].itertuples(index=False):
            mask = self.hours[val] != 6 if item.subset == "Non-6h" else self.hours[val] == int(item.subset[:-1])
            truth = y[mask]
            auc, ap = (item.auroc, item.auprc) if item.method == "B1 XGBoost" else metrics(
                truth, (full if item.method == "TrajRisk" else seed)[mask])
            prev = round(float(truth.mean()), 4)
            rows.append([item.method, item.subset, len(truth), int(truth.sum()), prev, auc, ap, prev])
        self.emit("S2C", rows, "Saved validation labels, timepoint strata, reference and seed scores",
                  "Original expression-only XGBoost ranking on each reference stratum")

    def mouse_transfer(self):
        original = csv(self.results / "week9a_mouse_validation_table.csv")
        replayed = csv(self.replay.output / "external_mouse_metrics.csv").set_index(["dataset", "method"])
        rows = []
        for dataset, task, caption in MOUSE_TASKS:
            for method in METHODS:
                if (dataset, method) in replayed.index:
                    item = replayed.loc[(dataset, method)]
                    n, pos = int(item.n), int(item.n_positive)
                    auc, lo, hi, ap = (round(float(item[k]), 4) for k in ("auroc", "auroc_ci_lower", "auroc_ci_upper", "auprc"))
                    prevalence = round(pos / n, 4)
                else:
                    item = original.loc[original.test.str.startswith(task) & original.method.eq(method)].iloc[0]
                    n, pos, prevalence, auc, lo, hi, ap = flat_row(
                        item, ["n_total", "n_pos", "prevalence", "auroc", "auroc_ci_lo", "auroc_ci_hi", "auprc"])
                rows.append([caption, METHODS[method], n, pos, prevalence, auc, lo, hi, ap, prevalence])
        self.emit("S3A", rows, "Replayed predictions, treatment labels, and 1,000-resample AUROC intervals",
                  "Original B1 scores for the two cohorts outside the 2 h correction")

        p21 = csv(self.results / "week9a_mouse_p21_analysis.csv").set_index("method")
        rows = []
        for item in original.loc[original.test.str.startswith("T3a:")].itertuples(index=False):
            values = p21.loc[item.method]
            rows.append([METHODS[item.method], *flat_row(item, ["auroc", "auroc_ci_lo", "auroc_ci_hi", "auprc", "auprc_rand"]),
                         *flat_row(values, ["spearman_rho_cdkn1a", "spearman_p", "mean_cdkn1a_expressed", "mean_cdkn1a_zero", "mw_p"])])
        self.emit("S3B", rows, "Join of the original evaluation and Cdkn1a statistical outputs",
                  "Underlying Cdkn1a expression matrix and original B1 predictions are not in the input archive")

    def human_transfer(self):
        packet = self.replay.packet / "compact/cross_species"
        obs = csv(packet / "observations.csv")
        with np.load(packet / "coordinates.npz", allow_pickle=False) as archive:
            embedding = archive["X_scvi_cross"].astype(np.float32)
        idx = np.random.default_rng(42).choice(len(obs), min(10000, len(obs)), replace=False)
        saved_summary = json.loads((self.results / "week9b_summary.json").read_text())
        values = [len(csv(packet / "gene_axis.csv")), int(obs.species.eq("mouse").sum()),
                  int(obs.species.eq("human").sum()), len(obs),
                  round(float(silhouette_score(embedding[idx], obs.species.iloc[idx])), 4),
                  round(float(silhouette_score(embedding[idx], obs.condition.replace("", "unknown").iloc[idx])), 4),
                  round(float(saved_summary["cs_trajrisk_val_auroc"]), 4)]
        captions = csv(self.table_path("S4A"))[["Metric", "Note"]]
        self.emit("S4A", [[item.Metric, value, item.Note]
                           for item, value in zip(captions.itertuples(index=False), values, strict=True)],
                  "Joint latent space and saved gene axis with recomputed ASW",
                  "Cross-species training validation AUROC")

        validation, concordance = [], []
        titles = ["TrajRisk W_self", "TrajRisk Reference-KNN", "Net orthologous gene signature", "Raw risk gene signature"]
        for kind, label in (("primary", "Primary cohort"), ("sensitivity", "Sensitivity cohort (≥300 cells per ALF donor)")):
            donors = csv(self.replay.output / f"human_donor_{kind}.csv").set_index("patient_id")
            signatures = csv(self.results / f"week9b_donor_{kind}.csv").set_index("patient_id")
            donors = donors.join(signatures[["sig_net", "sig_risk"]], validate="one_to_one")
            positive = donors.condition.eq("APAP_ALF")
            for name, column in zip(titles, ("risk_2a", "risk_2b", "sig_net", "sig_risk"), strict=True):
                score = donors[column]
                alf, healthy = score[positive], score[~positive]
                validation.append([label, name, len(alf), len(healthy), alf.mean(), healthy.mean(),
                                   mannwhitneyu(alf, healthy, alternative="greater").pvalue,
                                   roc_auc_score(positive, score)])
            rho, p = spearmanr(donors.risk_2b, donors.sig_net)
            concordance.append([label, len(donors), "TrajRisk Reference-KNN vs net gene signature", rho, p])
        self.emit("S4B", validation, "Donor aggregates of replayed scores, supplied signature scores, rank tests and AUROC",
                  "Donor-level gene-signature inputs")
        self.emit("S4C", concordance, "Recomputed correlation of Reference-KNN donor means with supplied net signature",
                  "Donor-level net-signature inputs")

    def spatial(self):
        bm = csv(self.results / "week10_spatial_visium_benmoshe.csv")
        fibrogenic = bm.Fibrogenic_spot.astype(str).str.lower().isin(["true", "1"])
        hours = pd.to_numeric(bm.Time_point.astype(str).str.extract(r"(\d+)", expand=False))
        rows = []
        for h in (None, 24, 48, 72):
            group = np.ones(len(bm), dtype=bool) if h is None else hours.eq(h)
            a, b = bm.loc[group & fibrogenic, "sig_risk"], bm.loc[group & ~fibrogenic, "sig_risk"]
            u, p = mannwhitneyu(a, b, alternative="greater")
            rows.append(["All timepoints" if h is None else f"{h} h", len(a), len(b),
                         a.median(), b.median(), a.mean(), b.mean(), u, p])
        self.emit("S5A", rows, "Spot signature scores and deposited fibrogenic labels")

        for table, species, marker in (("S5B", "mouse", "Cyp2e1"), ("S5C", "human", "CYP2E1")):
            spots = csv(self.results / f"week10_spatial_matchett_{species}_visium.csv")
            pc, pp = spots.zone.eq("pericentral"), spots.zone.eq("periportal")
            if not pc.any() or not pp.any():
                raise ValueError(f"Missing portal zones in {species} metadata")
            recover = (50 * spots.sig_recovery - spots[marker]) / 49
            original = csv(self.archived / self.table_path(table).name)
            checks = []
            rows = []
            for i, (score, alternative) in enumerate(((spots.sig_risk, "greater"), (recover, "less"))):
                a, b = score[pc], score[pp]
                item = original.iloc[i]
                u, p = mannwhitneyu(a, b, alternative=alternative)
                rows.append([item.Dataset, item.Analysis, len(a), len(b), a.median(), b.median(), item.Statistic, item.P_value])
                checks.append([item.Analysis, u, p, item.Statistic, item.P_value])
            item = original.iloc[2]
            rho, p = spearmanr(spots[marker], spots.sig_risk)
            rows.append([item.Dataset, item.Analysis, "", "", "", "", item.Statistic, item.P_value])
            checks.append([item.Analysis, rho, p, item.Statistic, item.P_value])
            pd.DataFrame(checks, columns=["analysis","recomputed_statistic","recomputed_p","published_statistic","published_p"]).to_csv(
                self.output.parent / f"table_{table.lower()}_rank_test_audit.csv", index=False)
            self.emit(table, rows, "Archived spot CPM, zonation and corrected recovery-signature medians",
                      "Precision-sensitive rank-test statistics and p values from the archived manuscript; rechecks in a separate audit CSV")

        sections = csv(ROOT / "manuscript/data/Data_S5_human_spatial_section_zone_means.csv")
        rows = [[cond, zone, len(sub), sub.Mean_risk_signature.mean(), sub.Mean_risk_signature.std(ddof=1),
                 sub.Mean_recovery_signature.mean(), sub.Mean_net_signature.mean()]
                for (cond, zone), sub in sections.groupby(["condition", "zone"], sort=True)]
        self.emit("S5D", rows, "Human spatial section means (Data S5)")

    def perturb(self, scale, keep):
        indices, weights = list(self.replay.eis), list(self.replay.ews)
        indices[scale], weights[scale] = indices[scale][:, keep], weights[scale][keep]
        with torch.inference_mode():
            modified = self.replay.model(self.replay.x, indices, weights).numpy()
        return modified - self.replay.reference_scores

    def bottlenecks(self):
        arows, brows, crows, grows = [], [], [], []
        scale3 = None
        edge_outputs = csv(self.results / "bottleneck_s5_all_edges_with_shift.csv")
        for i, scale in enumerate(SCALES):
            frame = edge_outputs.loc[edge_outputs.scale.eq(scale)]
            a, b = frame.source_node.to_numpy(int), frame.target_node.to_numpy(int)
            flux, amplification = frame.flux.to_numpy(np.float32), frame.amplification.to_numpy(np.float32)
            shift = frame.embedding_shift.to_numpy()
            projected = self.replay.embeddings[:, i, :]
            # The table rank tests use the saved shifts; verify their source against the replayed embeddings.
            if not np.allclose(np.linalg.norm(projected[b] - projected[a], axis=1), shift, rtol=2e-5, atol=3e-5):
                raise ValueError(f"Embedding shift differs from the replayed graph: {scale}")
            rank = frame.flux.rank(method="average", pct=True).to_numpy() * 100
            top = rank >= 99
            src, dst = np.unique(a[top]), np.unique(b[top])
            h1, h2 = TIMES[i]
            arows.append([LABELS[i], f"{h1} h → {h2} h", len(a), flux.sum(), flux.mean(), int(top.sum()),
                          float(flux[top].sum()/flux.sum()*100), amplification.mean(),
                          100*np.mean(amplification>0), 100*np.mean(amplification<0)])
            delta = self.perturb(i, np.zeros(len(a), dtype=bool))
            change = abs(delta)
            brows.append([LABELS[i], change.sum(), change.mean(), change.max(),
                          100*np.mean(delta<-.01), 100*np.mean(delta>.01),
                          *[change[self.hours==h].mean() for h in (0,6,24,48,96)]])
            crows.append([LABELS[i], int(top.sum()), len(src), len(dst), h1, h2,
                          self.risk[src].mean(), self.risk[dst].mean(), self.pt[src].mean(), self.pt[dst].mean(),
                          100*np.mean(self.treatment[src]=="APAP"), 100*np.mean(self.treatment[dst]=="APAP")])
            u, p = mannwhitneyu(shift[top], shift[~top], alternative="greater")
            rho, q = spearmanr(frame.flux, shift)
            grows.append([LABELS[i], int(top.sum()), int((~top).sum()), shift[top].mean(), shift[~top].mean(),
                          np.median(shift[top]), np.median(shift[~top]), u, p, rho, q])
            if i == 2:
                scale3 = (a, rank, src, dst)
        self.emit("S6A", arows, "Verified edge weights, flux and signed amplification")
        self.emit("S6B", brows, "Frozen forward passes with each temporal graph removed",
                  rtol=5e-5, atol=3e-8)
        self.emit("S6C", crows, "Top percentile edge ranks, unique cells and reference metadata")
        self.emit("S6G", grows, "Verified embedding shifts and rank tests")

        source_edges, rank, sources, targets = scale3

        def enrichment(nodes, values, hours, alternative):
            universe = np.flatnonzero(self.hours == hours)
            threshold = float(np.quantile(values[universe], .75))
            complement = np.setdiff1d(universe, nodes)
            n_high = int((values[nodes] >= threshold).sum())
            remaining_high = int((values[complement] >= threshold).sum())
            odds, p = fisher_exact([[n_high, len(nodes)-n_high],
                                    [remaining_high, len(complement)-remaining_high]], alternative=alternative)
            return [len(nodes), n_high, 100*n_high/len(nodes), 100*(n_high+remaining_high)/len(universe), threshold, odds, alternative, p]

        rows = [
            ["Scale 3 source: high pseudotime", *enrichment(sources, self.pt, 24, "greater")],
            ["Scale 3 source: high predicted risk", *enrichment(sources, self.risk, 24, "greater")],
            ["Scale 3 target: high pseudotime", *enrichment(targets, self.pt, 48, "less")],
        ]
        self.emit("S6D", rows, "Scale 3 percentiles and Fisher exact tests")

        rows = []
        for pct in (.5, 1, 2, 5, 10):
            selected = rank >= 100-pct
            n, high, frac, bg, _, odds, _, p = enrichment(np.unique(source_edges[selected]), self.pt, 24, "greater")
            rows.append([pct, int(selected.sum()), n, high, frac, bg, odds, p])
        self.emit("S6E", rows, "Scale 3 top-flux threshold sweep and Fisher exact tests")

        universe = np.flatnonzero(self.hours == 24)
        cutoffs = np.quantile(self.pt[universe], [.25, .5, .75], method="linear")
        labels = np.searchsorted(cutoffs, self.pt[universe], side="right")
        ranges = [0., *cutoffs, 1.]
        rows = []
        for i,name in enumerate(("Q1_low_pt", "Q2_mid_lo_pt", "Q3_mid_hi_pt", "Q4_high_pt")):
            keep = ~np.isin(self.replay.eis[2][0].numpy(), universe[labels==i])
            change = abs(self.perturb(2, keep))
            x, y = change[self.hours == 48], change[self.hours == 96]
            rows.append([name, f"{ranges[i]:.3f}–{ranges[i+1]:.3f}", int((~keep).sum()), change.sum(),
                         x.mean(), y.mean(), x.max(), y.max()])
        self.emit("S6F", rows, "Frozen forward passes after removing edges from each source quartile",
                  rtol=5e-5, atol=3e-8)

    def run(self):
        self.transitions()
        self.reference_comparisons()
        self.mouse_transfer()
        self.human_transfer()
        self.spatial()
        self.bottlenecks()
        if len(self.coverage) != 20 or {item["file"] for item in self.coverage} != {p.name for p in self.source.glob("*.csv")}:
            raise ValueError("The 20 published table components were not all built")
        report = {"status": "passed", "components": self.coverage}
        (self.output.parent / "table_reproduction_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
