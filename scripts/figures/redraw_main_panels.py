"""Redraw data panels from the frozen replay and rebuilt tables.

These plots reproduce the plotted analyses. The manuscript PDFs retain the
approved layouts, including schematic panels and original training curves.
"""
from pathlib import Path
import argparse
import ast
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

ROOT = Path(__file__).resolve().parents[2]
HOURS = (0, 6, 24, 48, 96)
SCALES = ["Scale1_AcuteStress", "Scale2_PeakInjury", "Scale3_FateDecision", "Scale4_Resolution"]
SHORT = ["S1", "S2", "S3", "S4"]
PALETTE = ["#C0392B", "#E67E22", "#27AE60", "#2980B9"]
BLUE = "#246CA7"

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300})


def panel(ax, letter, title):
    ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold", fontsize=9)


def iqrs(ax, values, positions, color="black"):
    for x, y in zip(positions, values, strict=True):
        q1, center, q3 = np.quantile(y, [.25, .5, .75])
        ax.vlines(x, q1, q3, color=color, linewidth=2.1, zorder=5)
        ax.scatter(x, center, color="white", edgecolor=color, s=16, linewidth=.65, zorder=6)


class Figures:
    def __init__(self, replay, packet, output):
        self.replay, self.output = replay, output
        self.results = packet / "originals/results"
        self.tables = replay / "supplementary_tables"
        self.ref = pd.read_csv(packet / "originals/Data/processed/week7_predictions.csv", index_col=0)
        prediction = pd.read_csv(replay / "reference_predictions.csv").set_index("cell_id")
        if not self.ref.index.equals(prediction.index):
            raise ValueError("Reference figure inputs use different cell indices")
        self.ref["score"] = prediction.risk_predicted
        self.output.mkdir(parents=True, exist_ok=True)
        self.completed = []

    def table(self, component):
        path, = self.tables.glob(f"Table_{component}_*.csv")
        return pd.read_csv(path, float_precision="round_trip")

    def save(self, fig, number, panels):
        file = f"figure_{number}_panels.pdf"
        fig.savefig(self.output / file, metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
        self.completed.append({"figure": number, "panels": panels, "file": file})
        print(f"Redrew Figure {number} {panels} from study outputs", flush=True)

    def figure2(self):
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.9), layout="constrained")
        a, b, c, d = axes.ravel()
        pseudo = [self.ref.loc[self.ref.hours.eq(h), "pseudotime_norm"] for h in HOURS]
        parts = a.violinplot(pseudo, np.arange(5), showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(BLUE)
            body.set_alpha(.43)
        iqrs(a, pseudo, np.arange(5))
        a.set(xticks=range(5), xticklabels=[f"{h} h\n(n={len(x):,})" for h,x in zip(HOURS,pseudo)],
              ylabel="Normalized pseudotime", ylim=(-.02, 1.02))
        panel(a,"A","Reference pseudotime")
        summary = json.loads((self.results / "week5_transition_summary.json").read_text())["scales"]
        folds = np.asarray([summary[s]["cv_auroc_folds"] for s in SCALES])
        b.bar(SHORT, folds.mean(axis=1), color=PALETTE, alpha=.35)
        b.errorbar(range(4), folds.mean(axis=1), yerr=1.96*folds.std(axis=1)/np.sqrt(5), fmt="none", color="black", capsize=2)
        b.scatter(np.repeat(np.arange(4), 5), folds.ravel(), color="black", s=9, label="CV fold")
        b.scatter(range(4), [summary[s]["oob_score"] for s in SCALES], marker="D", color=BLUE, s=18, label="OOB accuracy")
        b.set(ylim=(.925,1.005), ylabel="AUROC / accuracy")
        b.legend(loc="lower left", fontsize=7, frameon=False)
        panel(b,"B","Transition classifier performance")
        c.bar(SHORT,[summary[s]["gaussian_comparison"]["mean_row_pearson_vs_gaussian"] for s in SCALES],color=PALETTE)
        c.set(ylabel="Mean row-wise Pearson r")
        panel(c,"C","Learned versus Gaussian transitions")
        d.bar(SHORT,[summary[s]["entropy"]["entropy_mean"] for s in SCALES],
              yerr=[summary[s]["entropy"]["entropy_std"] for s in SCALES], color=PALETTE, capsize=3)
        d.set(ylabel="Outgoing Shannon entropy (nats)")
        panel(d,"D","Transition entropy")
        self.save(fig, 2, "A–D")

    def figure3(self):
        fig, (a,b) = plt.subplots(1,2,figsize=(7.8,3.0),layout="constrained")
        risk = [self.ref.loc[self.ref.hours.eq(h),"score"] for h in HOURS]
        vol = a.violinplot(risk,np.arange(5),showextrema=False)
        for body in vol["bodies"]:
            body.set_facecolor(BLUE)
        iqrs(a,risk,np.arange(5))
        for i,r in enumerate(risk):
            a.text(i, .97, f"{r.mean():.3f}", ha="center", va="top", fontsize=7)
        a.set(xticks=range(5),xticklabels=[f"{h} h\n(n={len(s):,})" for h,s in zip(HOURS,risk)],
              ylabel="TrajRisk score",ylim=(-.025,1.02))
        panel(a,"E","Score by sampling time")
        latent = np.load(self.replay / "reference_scale_embeddings.npy",allow_pickle=False)
        effects = [float(spearmanr(self.ref.score, np.linalg.norm(latent[:,i,:],axis=1)).statistic)
                   for i in range(4)]
        b.bar(SHORT,effects,color=PALETTE)
        b.axhline(0,color="black",lw=.7)
        b.set(ylabel="Spearman ρ")
        panel(b,"F","Score and scale embedding norm")
        self.save(fig,3,"E–F")

    def figure4(self):
        overall, within = self.table("S2A"), self.table("S2C")
        full = overall.iloc[0]
        fig = plt.figure(figsize=(8.1,6.0),layout="constrained")
        grid = fig.add_gridspec(2,2,height_ratios=[1,1.15])
        for j, codes in enumerate((("B1","B2","B3","B4"),("A1","A3","A4"))):
            ax = fig.add_subplot(grid[0,j])
            others = [overall.loc[overall.Method.str.startswith(code)].iloc[0] for code in codes]
            short = [r.Method.split(":",1)[-1].strip().replace(" (CV-tuned)","") for r in others]
            ax.axvspan(full.AUROC_95CI_lower,full.AUROC_95CI_upper,color=BLUE,alpha=.14)
            ax.axvline(full.AUROC,color=BLUE,lw=1.2,label="Full TrajRisk")
            for i,row in enumerate(others):
                v = row.AUROC
                ax.errorbar(v,i,xerr=[[v-row.AUROC_95CI_lower],[row.AUROC_95CI_upper-v]],fmt="o",
                            color="#374151",ms=4,capsize=2)
            ax.set(xlim=(.74,1.0),yticks=range(len(short)),yticklabels=short,xlabel="AUROC (validation)")
            ax.invert_yaxis()
            panel(ax,"AB"[j],("Baseline comparisons","Ablation comparisons")[j])
            if j==1:
                ax.legend(loc="lower right",fontsize=7,frameon=False)
        ax = fig.add_subplot(grid[1,:])
        groups = list(within.Subset.drop_duplicates())
        methods = ["TrajRisk","Seeder (W6)","B1 XGBoost"]
        colors = [BLUE,"#E89D47","#778599"]
        vals = within.pivot(index="Subset", columns="Method", values="AUPRC").loc[groups,methods]
        for i,label in enumerate(methods):
            rects = ax.bar(np.arange(len(groups))+(i-1)*.23,vals[label],width=.21,color=colors[i],label=label)
            ax.bar_label(rects,fmt="%.2f",padding=2,fontsize=7)
        first = within.drop_duplicates("Subset").set_index("Subset").loc[groups]
        ax.hlines(first.Random_AUPRC,np.arange(len(groups))-.34,np.arange(len(groups))+.34,
                  color="#242424",ls="--",lw=.9,label="Prevalence")
        ax.set(xticks=np.arange(len(groups)),xticklabels=[f"{g}\n(n={n:,})" for g,n in zip(groups,first.N)],
               ylim=(0,.84),ylabel="Average precision")
        panel(ax,"C","Within-timepoint discrimination")
        ax.legend(loc="upper left",frameon=False,ncol=4,fontsize=7)
        self.save(fig,4,"A–C")

    def figure6(self):
        fig, axes = plt.subplots(2,2,figsize=(7.5,6.4),layout="constrained")
        a,b,c,d = axes.ravel()
        validation = self.table("S4B")
        primary = None
        for i, (cohort, ax, letter) in enumerate((("primary",a,"B"),("sensitivity",b,"C"))):
            plot = pd.read_csv(self.replay / f"human_donor_{cohort}.csv").set_index("patient_id")
            if primary is None:
                primary = plot.join(pd.read_csv(self.results / "week9b_donor_primary.csv").set_index("patient_id").sig_net)
            groups = [plot.loc[plot.condition.eq(cond),"risk_2a"] for cond in ("Healthy","APAP_ALF")]
            color = ["#078189","#B0485E"]
            for j,x in enumerate(groups):
                ax.scatter(np.linspace(j-.075,j+.075,len(x)),x,color=color[j],s=25,alpha=.9)
            iqrs(ax,groups,[0,1])
            stats = validation.loc[validation.Method.eq("TrajRisk W_self")].iloc[i]
            ax.text(.04,.94,f"AUROC = {stats.Donor_level_AUROC:.3f}\np = {stats.Mann_Whitney_one_sided_p:.3f}",
                    transform=ax.transAxes,va="top",fontsize=8)
            ax.set(xticks=[0,1],xticklabels=[f"Healthy\n(n={len(groups[0])})",f"APAP-ALF\n(n={len(groups[1])})"],
                   ylabel="Donor mean TrajRisk score",ylim=(.035,.35))
            panel(ax,letter,cohort.capitalize()+" analysis")
        for cond,label,color in (("Healthy","Healthy","#078189"),("APAP_ALF","APAP-ALF","#B0485E")):
            u = primary.loc[primary.condition.eq(cond)]
            c.scatter(u.sig_net,u.risk_2b,color=color,s=27,label=label)
        c.set(xlabel="Net gene signature",ylabel="Reference-KNN donor mean score")
        c.legend(frameon=False,fontsize=7)
        rho,p = spearmanr(primary.sig_net,primary.risk_2b)
        c.text(.04,.96,f"ρ = {rho:.3f}; p = {p:.2g}",transform=c.transAxes,va="top",fontsize=8)
        panel(c,"D","Score and orthologous gene signature")
        counts = pd.read_csv(self.replay/"figure6e_distinct_gene_counts.csv")
        rects=d.bar(SHORT,counts.percentage,color=PALETTE)
        d.bar_label(rects,[f"n={n:,}" for n in counts.unique_human_genes],padding=3,fontsize=7)
        distinct = pd.read_csv(self.results/"week12_human_de_all_bottleneck_genes.csv").drop_duplicates("human_gene")
        whole = distinct.pval_human.le(.05).mean()*100
        d.axhline(whole,color="black",ls="--",lw=.8,label=f"{whole:.1f}% overall")
        d.set(ylim=(0,64),ylabel="Distinct mapped genes significant in human (%)")
        d.legend(frameon=False,fontsize=7)
        panel(d,"E","Cross-species gene overlap")
        self.save(fig,6,"B–E")

    def figure7(self):
        e = pd.read_csv(self.results / "bottleneck_s5_all_edges_with_shift.csv", float_precision="round_trip")
        groups = [e.loc[e.scale.eq(s)] for s in SCALES]
        top = [g.flux.rank(method="average",pct=True).ge(.99) for g in groups]
        fig, axes = plt.subplots(3,2,figsize=(8.4,9.4),layout="constrained")
        a,b,c,d,f,g = axes.ravel()
        grid = np.linspace(0.,1.,200)
        for i,(part,mask,color) in enumerate(zip(groups,top,PALETTE,strict=True)):
            a.hist(np.log10(part.loc[part.flux.gt(0),"flux"]),bins=80,density=True,histtype="step",
                   color=color,label=SHORT[i],lw=1)
            density = gaussian_kde(part.loc[mask,"pt_source"],bw_method=.30)(grid)
            b.plot(grid,density/density.max(),color=color,label=SHORT[i])
        a.set(xlabel="log₁₀ risk flux",ylabel="Density")
        a.legend(frameon=False,fontsize=7,ncol=2)
        panel(a,"A","Risk flux distribution")
        b.set(xlabel="Source normalized pseudotime",ylabel="Relative density")
        b.legend(frameon=False,fontsize=7)
        panel(b,"B","Top 1% flux edges")
        e3 = self.table("S6D")
        src,tgt = e3.iloc[0],e3.iloc[2]
        c.bar([0,1],[src.Bottleneck_high_pct,tgt.Bottleneck_high_pct],color=[PALETTE[2],BLUE],width=.62)
        c.hlines([src.Background_high_pct,tgt.Background_high_pct],[-.42,.58],[.42,1.42],
                 ls="--",color="#374151",label="Timepoint background")
        c.set(xticks=[0,1],xticklabels=[f"Sources at 24 h (n={src.N_bottleneck_cells})",
                                       f"Targets at 48 h (n={tgt.N_bottleneck_cells})"],
              ylim=(0,100),ylabel="High-pseudotime cells (%)")
        c.legend(frameon=False,loc="center right",fontsize=7)
        c.text(.99,.97,f"Source OR = {src.Odds_ratio:.1f}\np = {src.Fisher_p:.1e}",transform=c.transAxes,
               ha="right",va="top",fontsize=8)
        panel(c,"C","Scale 3 source and target states")
        ablated=self.table("S6B")
        d.barh(SHORT,ablated.total_abs_risk_change,color=PALETTE)
        d.set(xlabel="Total absolute score change")
        view = d.inset_axes([.48,.08,.47,.44])
        view.imshow(ablated[[f"tp_{h}h_mean_abs" for h in HOURS]],aspect="auto",cmap="Reds",interpolation="nearest")
        view.set(xticks=range(5),xticklabels=[str(h) for h in HOURS],yticks=range(4),yticklabels=SHORT)
        view.tick_params(labelsize=5,length=1)
        view.set_title("Mean change by hour",fontsize=6)
        panel(d,"D","Frozen-model scale removal")
        for i,(part,mask) in enumerate(zip(groups,top,strict=True)):
            vals = [part.loc[~mask,"embedding_shift"],part.loc[mask,"embedding_shift"]]
            at = [i-.16,i+.16]
            parts = f.violinplot(vals,at,widths=.28,showextrema=False)
            for body,color in zip(parts["bodies"],["#B5B7C0",PALETTE[i]]):
                body.set_facecolor(color)
                body.set_alpha(.72)
            iqrs(f,vals,at)
        f.set(xticks=range(4),xticklabels=SHORT,ylabel="Embedding shift (L₂)")
        f.legend(handles=[Patch(color="#B5B7C0",label="Background"),Patch(color=PALETTE[2],label="Top 1% (scale color)")],
                 frameon=False,fontsize=7)
        panel(f,"E","Edge embedding shift")
        # Use the pathway-display rules recorded with the original component.
        tree = ast.parse((ROOT/"archive/figure_components/generate_bottleneck_panels.py").read_text())
        rules = {}
        for item in tree.body:
            if isinstance(item,ast.Assign) and len(item.targets)==1 and isinstance(item.targets[0],ast.Name):
                if item.targets[0].id in {"EXCLUDE","TOP_N"}:
                    rules[item.targets[0].id] = ast.literal_eval(item.value)
        dots = []
        for i,scale in enumerate(SCALES):
            for direction in ("up","down"):
                tab = pd.read_csv(self.results/f"bottleneck_s4_{scale.lower()}_{direction}_kegg_2019_mouse.csv",
                                  dtype={"Adjusted P-value": float})
                tab = tab.loc[~tab.Term.str.contains("|".join(rules["EXCLUDE"]),case=False,na=False)]
                for _,row in tab.nsmallest(rules["TOP_N"],"Adjusted P-value").iterrows():
                    term = row.Term.split(" - ")[0]
                    dots.append((i,direction,term,float(row["Adjusted P-value"])))
        names = sorted({(direction, term) for _,direction,term,_ in dots})
        for i,direction,term,p in dots:
            g.scatter(i,names.index((direction,term)),s=np.clip(-12*np.log10(max(p,1e-30)),8,100),
                      marker="^" if direction=="up" else "v",color=PALETTE[i],alpha=.8)
        g.set(xticks=range(4),xticklabels=SHORT,yticks=range(len(names)),
              yticklabels=[(s[:27]+"…" if len(s)>28 else s) for _,s in names])
        g.tick_params(axis="y",labelsize=6)
        g.legend(handles=[Line2D([],[],marker=s,ls="none",color="#424650",label=text)
                          for s,text in [("^","Upregulated"),("v","Downregulated")]],
                 frameon=False,fontsize=6)
        panel(g,"F","Selected KEGG enrichments")
        self.save(fig,7,"A–F")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Verified extracted model-input bundle")
    parser.add_argument("--replay", type=Path, default=ROOT/"build/frozen_replay")
    parser.add_argument("--output", type=Path, default=ROOT/"build/figures/redrawn")
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT/"manuscript"):
        parser.error("Write redrawn panels to a separate build directory")
    figs=Figures(args.replay,args.bundle_dir,args.output)
    for fn in (figs.figure2,figs.figure3,figs.figure4,figs.figure6,figs.figure7):
        fn()
    (args.output/"redrawn_panels.json").write_text(json.dumps(figs.completed,indent=2)+"\n")


if __name__ == "__main__":
    main()
