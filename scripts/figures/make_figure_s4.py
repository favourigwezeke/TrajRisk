from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = Path(os.environ.get("TRAJRISK_ROOT", REPO_ROOT)).expanduser().resolve()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from matplotlib.lines import Line2D


# PATHS


RES = INPUT_ROOT / "results"
SUPP = REPO_ROOT / "manuscript"
TABLES = REPO_ROOT / "manuscript" / "tables" / "data"
DATA = REPO_ROOT / "manuscript" / "data"
OUT = REPO_ROOT / "build" / "figures" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

BM_FILE = RES / "week10_spatial_visium_benmoshe.csv"
MM_FILE = RES / "week10_spatial_matchett_mouse_visium.csv"
MH_FILE = RES / "week10_spatial_matchett_human_visium.csv"

S5A_FILE = TABLES / "Table_S5A_BenMoshe_fibrogenic_enrichment_FINAL.csv"
S5B_FILE = TABLES / "Table_S5B_Matchett_mouse_zonation_FINAL.csv"
S5C_FILE = TABLES / "Table_S5C_Matchett_human_zonation_FINAL.csv"
S5D_FILE = TABLES / "Table_S5D_human_condition_descriptive_FINAL.csv"

DATA_S5_FILE = DATA / "Data_S5_human_spatial_section_zone_means.csv"

PNG_OUT = OUT / "Figure_S4_spatial_validation_FINAL.png"
PDF_OUT = OUT / "Figure_S4_spatial_validation_FINAL.pdf"


# CONSTANTS


N_RECOVERY_ORIGINAL = 50
N_RECOVERY_CORRECTED = 49

ZONE_ORDER = [
    "pericentral",
    "midlobular",
    "periportal",
]

ZONE_LABELS = [
    "Pericentral",
    "Midlobular",
    "Periportal",
]

FIB_COLOR = "#D95F59"
NONFIB_COLOR = "#7A8DA6"

PC_COLOR = "#D95F59"
PP_COLOR = "#4F9D69"

APAP_COLOR = "#D95F59"
HEALTHY_COLOR = "#4C78A8"


# STYLE


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# HELPERS


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(path)


def require_columns(df, cols, label):
    missing = [
        c
        for c in cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label} missing required columns: "
            f"{missing}"
        )


def assert_close(
    observed,
    expected,
    label,
    rtol=1e-6,
    atol=1e-3,
):
    if not np.isclose(
        observed,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    ):
        raise ValueError(
            f"{label} mismatch: "
            f"observed={observed}, "
            f"expected={expected}"
        )


def assert_p_close(
    observed,
    expected,
    label,
):
    observed = float(observed)
    expected = float(expected)

    if expected == 0.0:

        if not (
            observed == 0.0
            or observed <= 1e-300
        ):
            raise ValueError(
                f"{label} mismatch: "
                f"observed={observed:.4e}, "
                "expected underflow / <=1e-300"
            )

        return

    if expected == 1.0:

        if not np.isclose(
            observed,
            1.0,
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{label} mismatch: "
                f"observed={observed:.8g}, "
                "expected=1.0"
            )

        return

    if observed <= 0:
        raise ValueError(
            f"{label}: observed p-value <= 0 "
            "while expected p-value is non-zero."
        )

    log_diff = abs(
        np.log10(observed)
        - np.log10(expected)
    )

    if log_diff > 1e-5:
        raise ValueError(
            f"{label} mismatch: "
            f"observed={observed:.8e}, "
            f"expected={expected:.8e}"
        )


def assert_statistical_consistency(
    observed_stat,
    expected_stat,
    observed_p,
    expected_p,
    label,
):
    observed_stat = float(observed_stat)
    expected_stat = float(expected_stat)

    observed_p = float(observed_p)
    expected_p = float(expected_p)

    stat_scale = max(
        abs(expected_stat),
        1.0,
    )

    stat_rel_diff = (
        abs(
            observed_stat
            - expected_stat
        )
        / stat_scale
    )

    if stat_rel_diff > 5e-6:
        raise ValueError(
            f"{label} statistic mismatch: "
            f"observed={observed_stat}, "
            f"expected={expected_stat}, "
            f"relative difference="
            f"{stat_rel_diff:.3e}"
        )

    if expected_p == 0.0:

        if not (
            observed_p == 0.0
            or observed_p <= 1e-300
        ):
            raise ValueError(
                f"{label} p-value mismatch: "
                f"observed={observed_p:.4e}, "
                "expected underflow / <=1e-300"
            )

        return

    if expected_p > 0.999999:

        if abs(
            observed_p
            - expected_p
        ) > 1e-10:
            raise ValueError(
                f"{label} p-value mismatch: "
                f"observed={observed_p:.16g}, "
                f"expected={expected_p:.16g}"
            )

        return

    if expected_p >= 1e-6:

        if abs(
            observed_p
            - expected_p
        ) > 5e-5:
            raise ValueError(
                f"{label} p-value mismatch: "
                f"observed={observed_p:.8g}, "
                f"expected={expected_p:.8g}"
            )

        return

    if observed_p <= 0:
        raise ValueError(
            f"{label}: observed p-value <= 0 "
            "while reported p-value is non-zero."
        )

    log10_diff = abs(
        np.log10(observed_p)
        - np.log10(expected_p)
    )

    if log10_diff > 0.01:
        raise ValueError(
            f"{label} p-value mismatch: "
            f"observed={observed_p:.8e}, "
            f"expected={expected_p:.8e}, "
            f"log10 difference={log10_diff:.4g}"
        )


def format_locked_p(
    p,
    digits=2,
):
    p = float(p)

    if p == 0.0:
        return r"$p<10^{-300}$"

    if p >= 0.9995:
        return rf"$p={p:.3f}$"

    if p >= 0.001:
        return rf"$p={p:.3f}$"

    exponent = int(
        np.floor(
            np.log10(p)
        )
    )

    mantissa = (
        p
        / (
            10 ** exponent
        )
    )

    return (
        rf"$p={mantissa:.{digits}f}"
        rf"\times10^{{{exponent}}}$"
    )


def style_boxplot(
    ax,
    data,
    positions,
    colors,
    widths=0.34,
):
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=widths,
        showfliers=False,
        patch_artist=True,
        medianprops={
            "color": "black",
            "linewidth": 1.1,
        },
        whiskerprops={
            "linewidth": 0.8,
        },
        capprops={
            "linewidth": 0.8,
        },
        boxprops={
            "linewidth": 0.8,
        },
    )

    for patch, color in zip(
        bp["boxes"],
        colors,
    ):
        patch.set_facecolor(
            color
        )

        patch.set_alpha(
            0.78
        )

    return bp


def add_panel_label(
    ax,
    label,
    x=-0.15,
    y=1.08,
):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="top",
        ha="left",
    )



# LOAD


print(
    "===== FIGURE S4 INPUT VALIDATION ====="
)

for path in [
    BM_FILE,
    MM_FILE,
    MH_FILE,
    S5A_FILE,
    S5B_FILE,
    S5C_FILE,
    S5D_FILE,
    DATA_S5_FILE,
]:
    require_file(
        path
    )

bm = pd.read_csv(
    BM_FILE
)

mm = pd.read_csv(
    MM_FILE
)

mh = pd.read_csv(
    MH_FILE
)

s5a = pd.read_csv(
    S5A_FILE
)

s5b = pd.read_csv(
    S5B_FILE
)

s5c = pd.read_csv(
    S5C_FILE
)

s5d = pd.read_csv(
    S5D_FILE
)

data_s5 = pd.read_csv(
    DATA_S5_FILE
)

print(
    f"Ben-Moshe spots: "
    f"{len(bm):,}"
)

print(
    f"Matchett mouse spots: "
    f"{len(mm):,}"
)

print(
    f"Matchett human spots: "
    f"{len(mh):,}"
)


# REQUIRED COLUMNS


require_columns(
    bm,
    [
        "sig_risk",
        "Time_point",
        "Fibrogenic_spot",
    ],
    "Ben-Moshe spatial file",
)

require_columns(
    mm,
    [
        "sig_risk",
        "sig_recovery",
        "Cyp2e1",
        "zone",
        "sample",
        "timepoint",
        "condition",
    ],
    "Matchett mouse spatial file",
)

require_columns(
    mh,
    [
        "sig_risk",
        "sig_recovery",
        "CYP2E1",
        "zone",
        "sample",
        "condition",
    ],
    "Matchett human spatial file",
)

require_columns(
    data_s5,
    [
        "sample",
        "condition",
        "zone",
        "Mean_risk_signature",
        "Mean_recovery_signature",
        "Mean_net_signature",
    ],
    "Data S5",
)


# CORRECT THE MATCHETT RECOVERY SIGNATURE EXACTLY


mm["sig_recovery_corrected"] = (
    N_RECOVERY_ORIGINAL
    * mm["sig_recovery"].astype(float)
    - mm["Cyp2e1"].astype(float)
) / N_RECOVERY_CORRECTED

mh["sig_recovery_corrected"] = (
    N_RECOVERY_ORIGINAL
    * mh["sig_recovery"].astype(float)
    - mh["CYP2E1"].astype(float)
) / N_RECOVERY_CORRECTED

mm["sig_net_corrected"] = (
    mm["sig_risk"].astype(float)
    - mm["sig_recovery_corrected"]
)

mh["sig_net_corrected"] = (
    mh["sig_risk"].astype(float)
    - mh["sig_recovery_corrected"]
)

print(
    "Corrected 49-gene recovery "
    "signatures reconstructed."
)


# VALIDATION 1 — BEN-MOSHE AGAINST reported S5A


print(
    "\n===== BEN-MOSHE reported-TABLE VALIDATION ====="
)

bm_expected_map = {
    "All timepoints": None,
    "24 h": 24,
    "48 h": 48,
    "72 h": 72,
}

for label, tp in bm_expected_map.items():

    expected = s5a.loc[
        s5a["Timepoint"]
        .astype(str)
        .eq(label)
    ]

    if len(expected) != 1:
        raise ValueError(
            f"Could not uniquely locate "
            f"S5A row: {label}"
        )

    expected = expected.iloc[0]

    if tp is None:
        sub = bm.copy()

    else:
        sub = bm[
            pd.to_numeric(
                bm["Time_point"],
                errors="coerce",
            ).eq(tp)
        ].copy()

    fib = sub.loc[
        sub[
            "Fibrogenic_spot"
        ].eq(1),
        "sig_risk",
    ].astype(float)

    non = sub.loc[
        sub[
            "Fibrogenic_spot"
        ].eq(0),
        "sig_risk",
    ].astype(float)

    bm_res = mannwhitneyu(
        fib,
        non,
        alternative="greater",
    )

    obs_fib_med = float(
        np.median(
            fib
        )
    )

    obs_non_med = float(
        np.median(
            non
        )
    )

    if int(
        len(fib)
    ) != int(
        expected[
            "N_fibrogenic"
        ]
    ):
        raise ValueError(
            f"Ben-Moshe {label}: "
            "fibrogenic count mismatch."
        )

    if int(
        len(non)
    ) != int(
        expected[
            "N_nonfibrogenic"
        ]
    ):
        raise ValueError(
            f"Ben-Moshe {label}: "
            "non-fibrogenic count mismatch."
        )

    assert_close(
        obs_fib_med,
        float(
            expected[
                "Median_risk_fibrogenic"
            ]
        ),
        f"Ben-Moshe {label} "
        "fibrogenic median",
    )

    assert_close(
        obs_non_med,
        float(
            expected[
                "Median_risk_nonfibrogenic"
            ]
        ),
        f"Ben-Moshe {label} "
        "non-fibrogenic median",
    )

    assert_p_close(
        float(
            bm_res.pvalue
        ),
        float(
            expected[
                "One_sided_p_fibrogenic_gt_nonfibrogenic"
            ]
        ),
        f"Ben-Moshe {label} p-value",
    )

    print(
        f"{label:<14} PASS | "
        f"fib median="
        f"{obs_fib_med:.3f} | "
        f"non-fib median="
        f"{obs_non_med:.3f} | "
        f"p="
        f"{bm_res.pvalue:.3e}"
    )


# VALIDATION 2 — MATCHETT MOUSE AGAINST reported S5B


print(
    "\n===== MATCHETT MOUSE reported-TABLE VALIDATION ====="
)

mm_pc = mm[
    mm["zone"]
    .astype(str)
    .eq("pericentral")
].copy()

mm_pp = mm[
    mm["zone"]
    .astype(str)
    .eq("periportal")
].copy()

mouse_risk_row = s5b[
    s5b["Analysis"]
    .astype(str)
    .str.startswith(
        "Risk signature:"
    )
].iloc[0]

mouse_rec_row = s5b[
    s5b["Analysis"]
    .astype(str)
    .str.startswith(
        "Recovery signature"
    )
].iloc[0]

if len(
    mm_pc
) != int(
    mouse_risk_row[
        "N_pericentral"
    ]
):
    raise ValueError(
        "Mouse pericentral count mismatch."
    )

if len(
    mm_pp
) != int(
    mouse_risk_row[
        "N_periportal"
    ]
):
    raise ValueError(
        "Mouse periportal count mismatch."
    )

mm_risk_pc = float(
    np.median(
        mm_pc[
            "sig_risk"
        ]
    )
)

mm_risk_pp = float(
    np.median(
        mm_pp[
            "sig_risk"
        ]
    )
)

mm_risk_res = mannwhitneyu(
    mm_pc["sig_risk"],
    mm_pp["sig_risk"],
    alternative="greater",
)

assert_close(
    mm_risk_pc,
    float(
        mouse_risk_row[
            "Pericentral_summary"
        ]
    ),
    "Mouse risk PC median",
)

assert_close(
    mm_risk_pp,
    float(
        mouse_risk_row[
            "Periportal_summary"
        ]
    ),
    "Mouse risk PP median",
)

assert_statistical_consistency(
    observed_stat=float(
        mm_risk_res.statistic
    ),
    expected_stat=float(
        mouse_risk_row[
            "Statistic"
        ]
    ),
    observed_p=float(
        mm_risk_res.pvalue
    ),
    expected_p=float(
        mouse_risk_row[
            "P_value"
        ]
    ),
    label="Mouse risk",
)

if len(
    mm_pc
) != int(
    mouse_rec_row[
        "N_pericentral"
    ]
):
    raise ValueError(
        "Mouse recovery pericentral "
        "count mismatch."
    )

if len(
    mm_pp
) != int(
    mouse_rec_row[
        "N_periportal"
    ]
):
    raise ValueError(
        "Mouse recovery periportal "
        "count mismatch."
    )

mm_rec_pc = float(
    np.median(
        mm_pc[
            "sig_recovery_corrected"
        ]
    )
)

mm_rec_pp = float(
    np.median(
        mm_pp[
            "sig_recovery_corrected"
        ]
    )
)

mm_rec_res = mannwhitneyu(
    mm_pc[
        "sig_recovery_corrected"
    ],
    mm_pp[
        "sig_recovery_corrected"
    ],
    alternative="less",
)

assert_close(
    mm_rec_pc,
    float(
        mouse_rec_row[
            "Pericentral_summary"
        ]
    ),
    "Mouse corrected recovery "
    "PC median",
)

assert_close(
    mm_rec_pp,
    float(
        mouse_rec_row[
            "Periportal_summary"
        ]
    ),
    "Mouse corrected recovery "
    "PP median",
)

assert_statistical_consistency(
    observed_stat=float(
        mm_rec_res.statistic
    ),
    expected_stat=float(
        mouse_rec_row[
            "Statistic"
        ]
    ),
    observed_p=float(
        mm_rec_res.pvalue
    ),
    expected_p=float(
        mouse_rec_row[
            "P_value"
        ]
    ),
    label="Mouse corrected recovery",
)

print(
    "Mouse risk PASS | "
    f"PC={mm_risk_pc:.3f}, "
    f"PP={mm_risk_pp:.3f} | "
    f"current p={mm_risk_res.pvalue:.8g} | "
    f"reported p="
    f"{float(mouse_risk_row['P_value']):.8g}"
)

print(
    "Mouse corrected recovery PASS | "
    f"PC={mm_rec_pc:.3f}, "
    f"PP={mm_rec_pp:.3f} | "
    f"current p={mm_rec_res.pvalue:.3e} | "
    f"reported p="
    f"{float(mouse_rec_row['P_value']):.3e}"
)


# VALIDATION 3 — MATCHETT HUMAN AGAINST reported S5C


print(
    "\n===== MATCHETT HUMAN reported-TABLE VALIDATION ====="
)

mh_pc = mh[
    mh["zone"]
    .astype(str)
    .eq("pericentral")
].copy()

mh_pp = mh[
    mh["zone"]
    .astype(str)
    .eq("periportal")
].copy()

human_risk_row = s5c[
    s5c["Analysis"]
    .astype(str)
    .str.startswith(
        "Risk signature:"
    )
].iloc[0]

human_rec_row = s5c[
    s5c["Analysis"]
    .astype(str)
    .str.startswith(
        "Recovery signature"
    )
].iloc[0]

if len(
    mh_pc
) != int(
    human_risk_row[
        "N_pericentral"
    ]
):
    raise ValueError(
        "Human pericentral count mismatch."
    )

if len(
    mh_pp
) != int(
    human_risk_row[
        "N_periportal"
    ]
):
    raise ValueError(
        "Human periportal count mismatch."
    )

mh_risk_pc = float(
    np.median(
        mh_pc[
            "sig_risk"
        ]
    )
)

mh_risk_pp = float(
    np.median(
        mh_pp[
            "sig_risk"
        ]
    )
)

mh_risk_res = mannwhitneyu(
    mh_pc["sig_risk"],
    mh_pp["sig_risk"],
    alternative="greater",
)

assert_close(
    mh_risk_pc,
    float(
        human_risk_row[
            "Pericentral_summary"
        ]
    ),
    "Human risk PC median",
)

assert_close(
    mh_risk_pp,
    float(
        human_risk_row[
            "Periportal_summary"
        ]
    ),
    "Human risk PP median",
)

assert_statistical_consistency(
    observed_stat=float(
        mh_risk_res.statistic
    ),
    expected_stat=float(
        human_risk_row[
            "Statistic"
        ]
    ),
    observed_p=float(
        mh_risk_res.pvalue
    ),
    expected_p=float(
        human_risk_row[
            "P_value"
        ]
    ),
    label="Human risk",
)

if len(
    mh_pc
) != int(
    human_rec_row[
        "N_pericentral"
    ]
):
    raise ValueError(
        "Human recovery pericentral "
        "count mismatch."
    )

if len(
    mh_pp
) != int(
    human_rec_row[
        "N_periportal"
    ]
):
    raise ValueError(
        "Human recovery periportal "
        "count mismatch."
    )

mh_rec_pc = float(
    np.median(
        mh_pc[
            "sig_recovery_corrected"
        ]
    )
)

mh_rec_pp = float(
    np.median(
        mh_pp[
            "sig_recovery_corrected"
        ]
    )
)

mh_rec_res = mannwhitneyu(
    mh_pc[
        "sig_recovery_corrected"
    ],
    mh_pp[
        "sig_recovery_corrected"
    ],
    alternative="less",
)

assert_close(
    mh_rec_pc,
    float(
        human_rec_row[
            "Pericentral_summary"
        ]
    ),
    "Human corrected recovery "
    "PC median",
)

assert_close(
    mh_rec_pp,
    float(
        human_rec_row[
            "Periportal_summary"
        ]
    ),
    "Human corrected recovery "
    "PP median",
)

assert_statistical_consistency(
    observed_stat=float(
        mh_rec_res.statistic
    ),
    expected_stat=float(
        human_rec_row[
            "Statistic"
        ]
    ),
    observed_p=float(
        mh_rec_res.pvalue
    ),
    expected_p=float(
        human_rec_row[
            "P_value"
        ]
    ),
    label="Human corrected recovery",
)

print(
    "Human risk PASS | "
    f"PC={mh_risk_pc:.3f}, "
    f"PP={mh_risk_pp:.3f} | "
    f"current p={mh_risk_res.pvalue:.8g} | "
    f"reported p="
    f"{float(human_risk_row['P_value']):.8g}"
)

print(
    "Human corrected recovery PASS | "
    f"PC={mh_rec_pc:.3f}, "
    f"PP={mh_rec_pp:.3f} | "
    f"current p={mh_rec_res.pvalue:.3e} | "
    f"reported p="
    f"{float(human_rec_row['P_value']):.3e}"
)


# VALIDATION 4 — DATA S5 AGAINST reported S5D


print(
    "\n===== HUMAN SECTION-LEVEL DESCRIPTIVE VALIDATION ====="
)

section_summary = (
    data_s5
    .groupby(
        [
            "condition",
            "zone",
        ],
        observed=True,
    )
    .agg(
        N_sections=(
            "sample",
            "nunique",
        ),
        Mean_section_risk_signature=(
            "Mean_risk_signature",
            "mean",
        ),
        SD_section_risk_signature=(
            "Mean_risk_signature",
            "std",
        ),
        Mean_section_recovery_signature=(
            "Mean_recovery_signature",
            "mean",
        ),
        Mean_section_net_signature=(
            "Mean_net_signature",
            "mean",
        ),
    )
    .reset_index()
)

check_cols = [
    "N_sections",
    "Mean_section_risk_signature",
    "SD_section_risk_signature",
    "Mean_section_recovery_signature",
    "Mean_section_net_signature",
]

s5d_check = s5d.merge(
    section_summary,
    on=[
        "condition",
        "zone",
    ],
    suffixes=(
        "_locked",
        "_recomputed",
    ),
)

if len(
    s5d_check
) != 6:
    raise ValueError(
        "Unexpected S5D merge size."
    )

for _, row in s5d_check.iterrows():

    label = (
        f"{row['condition']} "
        f"{row['zone']}"
    )

    if int(
        row[
            "N_sections_locked"
        ]
    ) != int(
        row[
            "N_sections_recomputed"
        ]
    ):
        raise ValueError(
            f"{label}: "
            "N_sections mismatch."
        )

    for col in check_cols[1:]:

        assert_close(
            float(
                row[
                    f"{col}_recomputed"
                ]
            ),
            float(
                row[
                    f"{col}_locked"
                ]
            ),
            f"{label}: {col}",
            rtol=1e-8,
            atol=1e-6,
        )

print(
    "Human section-level summaries: PASS"
)

print(
    "\nALL FIGURE S4 VALIDATIONS PASSED."
)


# FIGURE


fig = plt.figure(
    figsize=(15.2, 8.2)
)

gs = fig.add_gridspec(
    2,
    3,
    left=0.065,
    right=0.985,
    bottom=0.09,
    top=0.93,
    wspace=0.38,
    hspace=0.58,
)

axA = fig.add_subplot(
    gs[0, 0]
)

axB = fig.add_subplot(
    gs[0, 1]
)

axC = fig.add_subplot(
    gs[0, 2]
)

axD = fig.add_subplot(
    gs[1, 0]
)

axE = fig.add_subplot(
    gs[1, 1]
)

axF = fig.add_subplot(
    gs[1, 2]
)


# PANEL A — BEN-MOSHE FIBROGENIC ENRICHMENT


panel_a_groups = [
    (
        "All",
        bm,
        "All timepoints",
    ),
    (
        "24 h",
        bm[
            pd.to_numeric(
                bm["Time_point"],
                errors="coerce",
            ).eq(24)
        ],
        "24 h",
    ),
    (
        "48 h",
        bm[
            pd.to_numeric(
                bm["Time_point"],
                errors="coerce",
            ).eq(48)
        ],
        "48 h",
    ),
    (
        "72 h",
        bm[
            pd.to_numeric(
                bm["Time_point"],
                errors="coerce",
            ).eq(72)
        ],
        "72 h",
    ),
]

base_positions = (
    np.arange(
        len(
            panel_a_groups
        )
    )
    * 1.35
    + 1.0
)

offset = 0.22

non_data = []
fib_data = []

for _, sub, _ in panel_a_groups:

    non_data.append(
        sub.loc[
            sub[
                "Fibrogenic_spot"
            ].eq(0),
            "sig_risk",
        ]
        .astype(float)
        .values
    )

    fib_data.append(
        sub.loc[
            sub[
                "Fibrogenic_spot"
            ].eq(1),
            "sig_risk",
        ]
        .astype(float)
        .values
    )

style_boxplot(
    axA,
    non_data,
    base_positions
    - offset,
    [
        NONFIB_COLOR
    ]
    * len(
        non_data
    ),
)

style_boxplot(
    axA,
    fib_data,
    base_positions
    + offset,
    [
        FIB_COLOR
    ]
    * len(
        fib_data
    ),
)

axA.set_xticks(
    base_positions
)

axA.set_xticklabels(
    [
        x[0]
        for x
        in panel_a_groups
    ]
)

axA.set_xlabel(
    "Sampling time",
    labelpad=6,
)

axA.set_ylabel(
    "Risk signature (CPM)"
)

axA.set_title(
    "Ben-Moshe fibrogenic enrichment"
)

legend_a = [
    Line2D(
        [0],
        [0],
        color=NONFIB_COLOR,
        lw=7,
        label="Non-fibrogenic",
    ),
    Line2D(
        [0],
        [0],
        color=FIB_COLOR,
        lw=7,
        label="Fibrogenic",
    ),
]

axA.legend(
    handles=legend_a,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=2,
    handlelength=2.0,
    columnspacing=1.2,
)

for x, (_, _, locked_label) in zip(
    base_positions,
    panel_a_groups,
):

    row = s5a[
        s5a[
            "Timepoint"
        ]
        .astype(str)
        .eq(
            locked_label
        )
    ].iloc[0]

    p_locked = float(
        row[
            "One_sided_p_fibrogenic_gt_nonfibrogenic"
        ]
    )

    axA.text(
        x,
        0.97,
        format_locked_p(
            p_locked
        ),
        transform=axA.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=6.5,
    )


# PANEL B — MATCHETT MOUSE RISK ZONATION


style_boxplot(
    axB,
    [
        mm_pc[
            "sig_risk"
        ].values,
        mm_pp[
            "sig_risk"
        ].values,
    ],
    [
        1,
        2,
    ],
    [
        PC_COLOR,
        PP_COLOR,
    ],
    widths=0.52,
)

axB.set_xticks(
    [
        1,
        2,
    ]
)

axB.set_xticklabels(
    [
        "Pericentral",
        "Periportal",
    ]
)

axB.set_ylabel(
    "Risk signature (CPM)"
)

axB.set_title(
    "Matchett mouse: risk signature"
)

axB.text(
    0.5,
    0.96,
    format_locked_p(
        mouse_risk_row[
            "P_value"
        ]
    ),
    transform=axB.transAxes,
    ha="center",
    va="top",
    fontsize=7.5,
)


# PANEL C — MATCHETT MOUSE  RECOVERY


style_boxplot(
    axC,
    [
        mm_pc[
            "sig_recovery_corrected"
        ].values,
        mm_pp[
            "sig_recovery_corrected"
        ].values,
    ],
    [
        1,
        2,
    ],
    [
        PC_COLOR,
        PP_COLOR,
    ],
    widths=0.52,
)

axC.set_xticks(
    [
        1,
        2,
    ]
)

axC.set_xticklabels(
    [
        "Pericentral",
        "Periportal",
    ]
)

axC.set_ylabel(
    "Recovery signature (CPM)"
)

axC.set_title(
    r"Matchett mouse: recovery ($Cyp2e1$ excluded)"
)

axC.text(
    0.5,
    0.96,
    format_locked_p(
        mouse_rec_row[
            "P_value"
        ]
    ),
    transform=axC.transAxes,
    ha="center",
    va="top",
    fontsize=7.5,
)


# PANEL D — MATCHETT HUMAN RISK ZONATION


style_boxplot(
    axD,
    [
        mh_pc[
            "sig_risk"
        ].values,
        mh_pp[
            "sig_risk"
        ].values,
    ],
    [
        1,
        2,
    ],
    [
        PC_COLOR,
        PP_COLOR,
    ],
    widths=0.52,
)

axD.set_xticks(
    [
        1,
        2,
    ]
)

axD.set_xticklabels(
    [
        "Pericentral",
        "Periportal",
    ]
)

axD.set_ylabel(
    "Risk signature (CPM)"
)

axD.set_title(
    "Matchett human: risk signature"
)

axD.text(
    0.5,
    0.96,
    format_locked_p(
        human_risk_row[
            "P_value"
        ]
    ),
    transform=axD.transAxes,
    ha="center",
    va="top",
    fontsize=7.5,
)


# PANEL E — MATCHETT HUMAN  RECOVERY


style_boxplot(
    axE,
    [
        mh_pc[
            "sig_recovery_corrected"
        ].values,
        mh_pp[
            "sig_recovery_corrected"
        ].values,
    ],
    [
        1,
        2,
    ],
    [
        PC_COLOR,
        PP_COLOR,
    ],
    widths=0.52,
)

axE.set_xticks(
    [
        1,
        2,
    ]
)

axE.set_xticklabels(
    [
        "Pericentral",
        "Periportal",
    ]
)

axE.set_ylabel(
    "Recovery signature (CPM)"
)

axE.set_title(
    r"Matchett human: recovery ($CYP2E1$ excluded)"
)

axE.text(
    0.5,
    0.96,
    format_locked_p(
        human_rec_row[
            "P_value"
        ]
    ),
    transform=axE.transAxes,
    ha="center",
    va="top",
    fontsize=7.5,
)


# PANEL F — HUMAN SECTION-LEVEL DESCRIPTIVE RISK


x_zone = np.arange(
    len(
        ZONE_ORDER
    )
)

for condition, display_label, color in [
    (
        "Healthy",
        "Healthy",
        HEALTHY_COLOR,
    ),
    (
        "APAP",
        "APAP-ALF",
        APAP_COLOR,
    ),
]:

    sub = data_s5[
        data_s5[
            "condition"
        ]
        .astype(str)
        .eq(
            condition
        )
    ].copy()

    for sample, sample_df in sub.groupby(
        "sample",
        observed=True,
    ):

        sample_df = (
            sample_df
            .set_index(
                "zone"
            )
            .reindex(
                ZONE_ORDER
            )
        )

        axF.plot(
            x_zone,
            sample_df[
                "Mean_risk_signature"
            ].values,
            color=color,
            alpha=0.25,
            linewidth=1.0,
            marker="o",
            markersize=3,
        )

    mean_profile = (
        sub
        .groupby(
            "zone",
            observed=True,
        )[
            "Mean_risk_signature"
        ]
        .mean()
        .reindex(
            ZONE_ORDER
        )
    )

    axF.plot(
        x_zone,
        mean_profile.values,
        color=color,
        linewidth=2.4,
        marker="o",
        markersize=5,
        label=display_label,
    )

axF.set_xticks(
    x_zone
)

axF.set_xticklabels(
    ZONE_LABELS,
    rotation=15,
    ha="right",
)

axF.set_ylabel(
    "Section mean risk signature (CPM)"
)

axF.set_xlabel(
    "Spatial zone"
)

axF.set_title(
    "Human section-level risk profiles"
)

axF.legend(
    frameon=False,
    title=None,
)

axF.text(
    0.02,
    0.02,
    "Descriptive only; no zone-level\n"
    "case-control significance test",
    transform=axF.transAxes,
    ha="left",
    va="bottom",
    fontsize=6.5,
)


# PANEL LABELS


for label, ax in zip(
    [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ],
    [
        axA,
        axB,
        axC,
        axD,
        axE,
        axF,
    ],
):
    add_panel_label(
        ax,
        label,
    )


# LIGHT GRID


for ax in [
    axA,
    axB,
    axC,
    axD,
    axE,
    axF,
]:
    ax.grid(
        axis="y",
        alpha=0.12,
        linewidth=0.5,
    )


# SAVE


fig.savefig(
    PNG_OUT,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    PDF_OUT,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(
    fig
)

print(
    "\nCreated:"
)

print(
    " ",
    PNG_OUT,
)

print(
    " ",
    PDF_OUT,
)
