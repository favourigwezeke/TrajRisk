#!/usr/bin/env python3
"""Redraw only the manuscript panels affected by the GSE200771 metadata correction.

Use the published rc4 figure templates and the current replay outputs.
"""
import argparse
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

REPO = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--replay", type=Path, default=REPO / "build/frozen_replay")
parser.add_argument("--bundle-dir", type=Path, required=True, help="Verified extracted frozen-input directory")
parser.add_argument("--output", type=Path, default=REPO / "build/figures/corrected")
args = parser.parse_args()
analysis = args.replay.resolve()
target = args.output.resolve()
if target.is_relative_to(REPO / "manuscript"):
    parser.error("Write regenerated figures to a separate build directory")
target.mkdir(parents=True, exist_ok=True)


def original_pdf(name):
    # Preserve the published layout and panels outside the corrected cohort.
    return fitz.open(REPO / "manuscript/figures" / name)


audit = json.loads((analysis / "umbaugh_2h_reanalysis.json").read_text())
scores = pd.read_csv(analysis / "umbaugh_2h_scores.csv", index_col="cell_id")
metrics = audit["after"]
counts = scores["treatment"].value_counts()
assert scores.index.is_unique and len(scores) == metrics["knn"]["n"]
assert counts["APAP"] == metrics["knn"]["n_positive"]
white, black = (1.,1.,1.), (0.,0.,0.)


def erase(page, box, *, images=False, fill=white):
    page.add_redact_annot(fitz.Rect(box), fill=fill, cross_out=False)
    page.apply_redactions(images=2 if images else 0, graphics=0, text=0)


def center_text(page, x, y, text, size, color=black, font="helv"):
    width = fitz.get_text_length(text, fontname=font, fontsize=size)
    page.insert_text((x-width/2,y),text,fontname=font,fontsize=size,color=color)


def save(doc, name):
    doc.save(target / name, garbage=4, deflate=True)


def figure_5():
    if any(m["lower"] is None for m in metrics.values()):
        raise ValueError("Regenerating Figure 5 requires the bootstrap confidence intervals")
    name = "figure_5.pdf"
    with original_pdf(name) as doc:
        p = doc[0]
        # Keep the original panel A axes, legend and remaining four benchmark groups.
        erase(p, (241.0,122.,306.,187.87))
        auroc_y = lambda value: 60.682556 - (value-1)*282.708008
        blue = (26/255,110/255,175/255)
        for key, center, color in [
            ("knn",253.245,blue),
            ("seed",273.97,(224/255,123/255,57/255)),
            ("b1",294.695,(.6195,.6196,.6195)),
        ]:
            m = metrics[key]
            p.draw_rect((center-8.806,auroc_y(m["auroc"]),center+8.806,187.9),
                        color=None,fill=color)
            top,bottom = map(auroc_y,[m["upper"],m["lower"]])
            p.draw_line((center,top),(center,bottom),color=black,width=.65)
            for h in (top,bottom):
                p.draw_line((center-2.45,h),(center+2.45,h),color=black,width=.65)
        p.draw_line((241.0,187.9),(306.0,187.9),color=black,width=.62)

        # Panel B retains its original axes. KDE and markers use the new cell scores.
        erase(p, (51.,253.82,232.31,370.90))
        score_y = lambda value: 366.503 - 107.924*value
        for treatment, center, color in [
            ("Control",88.265,(46/255,134/255,193/255)),
            ("APAP",178.325,(192/255,57/255,43/255)),
        ]:
            data = scores.loc[scores["treatment"].eq(treatment),"Reference-KNN"].to_numpy()
            assert len(data) == counts[treatment] and (0 <= data.min() <= data.max() <= 1)
            points = np.linspace(data.min(),data.max(),160)
            pdf = gaussian_kde(data)(points)
            width = 23.415 * pdf / pdf.max()
            xy = [(x,y) for x,y in zip(center-width, score_y(points))]
            xy.extend(zip((center+width)[::-1],score_y(points)[::-1]))
            p.draw_polyline(xy,width=.2,color=black,fill=color,closePath=True,fill_opacity=.60)
            q1,median,q3 = np.quantile(data,[.25,.5,.75])
            p.draw_line((center,score_y(q1)),(center,score_y(q3)),color=black,width=1.65)
            p.draw_circle((center,score_y(median)),2.20,width=.7,color=black,fill=white)
        center_text(p,132,261.46,f"AUROC = {metrics['knn']['auroc']:.3f}",7.2,blue)
        assert not p.search_for("0.652")
        save(doc,name)


def figure_s1():
    name = "figure_S1.pdf"
    with original_pdf(name) as doc:
        p = doc[0]
        # Update panel B's treatment-composition chart; QC distributions are unchanged.
        erase(p,(600.,51.5,831.2,186.5))
        y = lambda count: 181.28-0.00752*count
        for label,cx,c in [("Control",675.605,(76/255,120/255,168/255)),
                            ("APAP",785.44,(228/255,87/255,86/255))]:
            n = int(counts[label])
            p.draw_rect((cx-35.698,y(n),cx+35.698,y(0)),color=None,fill=c)
            center_text(p,cx,y(n)-1.6,f"{n:,}",6.5)
            p.draw_line((cx,y(0)),(cx,y(0)+3.50),width=.65,color=black)
        p.draw_line((630.85,52.98),(630.85,y(0)),width=.65,color=black)
        p.draw_line((630.85,y(0)),(830.20,y(0)),width=.65,color=black)
        for n in range(0,16001,2000):
            p.draw_line((627.35,y(n)),(630.85,y(n)),width=.65,color=black)
            txt = f"{n:,}"
            fw = fitz.get_text_length(txt,fontname="helv",fontsize=7.)
            p.insert_text((623.85-fw,y(n)+2.5),txt,fontname="helv",fontsize=7.)
        assert not p.search_for("7,907") and not p.search_for("13,502")
        assert p.search_for("6,077") and p.search_for("15,332")
        save(doc,name)


def umap(out_pdf):
    obj = args.bundle_dir / "compact" / "mouse_integration"
    obs = pd.read_csv(obj/"observations.csv",index_col="cell_id")
    with np.load(obj/"coordinates.npz") as compact:
        xy = compact["X_umap"]
    assert len(xy) == len(obs) == 36067 and obs.index.is_unique
    early = obs["dataset"].eq("Umbaugh_2h")
    assert set(obs.index[early]) == set(scores.index)
    obs.loc[scores.index,"treatment"] = scores["treatment"]
    obs.loc[scores.index,"timepoint_str"] = "2h"

    plt.rcParams.update({"font.family":"Liberation Sans","pdf.fonttype":42,
                         "axes.spines.right":False,"axes.spines.top":False,
                         "axes.linewidth":.35,"font.size":4.5})
    fig = plt.figure(figsize=(332.64/72,166.32/72),dpi=400,facecolor="white")
    fig.text(.011,.946,"C", fontsize=9.1, fontweight="bold", va="top")
    panels = [
        ("Time point", "timepoint_str",
         dict(zip(("0h","2h","6h","24h","48h","72h","96h"),
                  ("#dc6255","#c79813","#779901","#1b9a74","#3999d9","#8364dd","#dd69bd")))),
        ("Treatment", "treatment", {"Control":"#00a4af","APAP":"#e76f63"}),
        ("Cohort", "dataset", {"Walesky":"#9162d1","Umbaugh_2h":"#7e9f12",
                              "Ben_Moshe":"#d3795d","Umbaugh_48h72h":"#3bbfa7"}),
    ]
    labels = {"Umbaugh_2h":"Umbaugh 2 h","Umbaugh_48h72h":"Umbaugh 48/72 h",
              "Ben_Moshe":"Ben-Moshe"}
    rng = np.random.default_rng(42)
    order = rng.permutation(len(obs))
    for i,(title,col,colors) in enumerate(panels):
        ax = fig.add_axes([[.057,.371,.684][i],.135,[.21,.21,.195][i],.74])
        assert set(obs[col]) == colors.keys(), (col, set(obs[col]))
        c = obs[col].map(colors).to_numpy()
        ax.scatter(xy[order,0],xy[order,1],s=.57,alpha=.73,c=c[order],
                   linewidths=0,rasterized=True)
        ax.set_title(title,fontsize=5.7,pad=2.8)
        ax.set(xlabel="UMAP 1",ylabel="UMAP 2",
               xlim=(xy[:,0].min()-1,xy[:,0].max()+1),ylim=(-17,16))
        ax.xaxis.labelpad = 1.5
        ax.yaxis.labelpad = 1.
        ax.tick_params(length=1.4,width=.35,pad=1.2,labelsize=4.2)
        ax.set_xticks([-10,-5,0,5,10])
        ax.set_yticks([-15,-10,-5,0,5,10,15])
        handles = [Line2D([0],[0],marker="o",markersize=2.8,linestyle="none",
                          markeredgewidth=0,markerfacecolor=color,
                          label=labels.get(k,k.replace("h"," h")) if col=="timepoint_str"
                          else labels.get(k,k)) for k,color in colors.items()]
        ax.legend(handles=handles,loc="center left",bbox_to_anchor=(1.01,.52),
                  frameon=False,borderaxespad=0,handlelength=.70,
                  handletextpad=.35,fontsize=4.0,labelspacing=.40)
    fig.savefig(out_pdf)
    plt.close(fig)


def figure_1():
    name = "figure_1.pdf"
    new_umap = target / "figure_1c_corrected.pdf"
    umap(new_umap)
    with original_pdf(name) as doc, fitz.open(new_umap) as panel:
        p = doc[0]
        # Panel A records the collection time, which is shared by all six samples.
        erase(p,(114.1,90.7,126.1,96.0),images=True,fill=(.973,.973,.973))
        p.insert_text((114.7,95.42),"2 h",fontname="helv",fontsize=4.55,color=black)
        # The vehicle mice and APAP mice share the 2 h collection time.
        erase(p,(133.9,378.9,182.6,398.2))
        center_y = 388.525
        grey = (.949,.949,.949)
        green = (.1843,.7020,.4157)
        p.draw_circle((142.56,center_y),5.063,color=None,fill=grey)
        area = lambda n: 42.0 + 360.0*np.sqrt(n/35737)
        radius = 7.76*np.sqrt(area(len(scores))/area(13502))
        p.draw_circle((172.30,center_y),radius,color=white,fill=green,width=.38)
        center_text(p,172.30,center_y+1.30,f"{len(scores)/1000:.1f}k",4.,white,"hebo")

        # Replace the old time/treatment overlays, retaining the stored UMAP coordinates.
        frame = fitz.Rect(6.406,498.318,339.046,664.639)
        erase(p,frame,images=True)
        p.show_pdf_page(frame,panel)
        save(doc,name)


figure_1()
figure_5()
figure_s1()
print("Updated Figure 1A–C, Figure 5A/B, and Figure S1B.")
