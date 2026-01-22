import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
import math

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="SPC Color Dashboard",
    page_icon="🎨",
    layout="wide"
)

# =========================
# STYLE
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            270deg,
            #ffffff,
            #f0f9ff,
            #e0f2fe,
            #fef3c7,
            #ecfeff
        );
        background-size: 800% 800%;
        animation: gradientBG 20s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# REFRESH
# =========================
if st.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

# =========================
# DATA URL
# =========================
DATA_URL = "https://docs.google.com/spreadsheets/d/1lqsLKSoDTbtvAsHzJaEri8tPo5pA3vqJ__LVHp2R534/export?format=csv"
LIMIT_URL = "https://docs.google.com/spreadsheets/d/1jbP8puBraQ5Xgs9oIpJ7PlLpjIK3sltrgbrgKUcJ-Qo/export?format=csv"

@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(DATA_URL)
    df["Time"] = pd.to_datetime(df["Time"])
    return df

@st.cache_data(ttl=300)
def load_limit():
    return pd.read_csv(LIMIT_URL)

df = load_data()
limit_df = load_limit()

# =========================
# CLEAN COLUMN NAMES
# =========================
df.columns = (
    df.columns
    .str.replace("\r\n", " ", regex=False)
    .str.replace("\n", " ", regex=False)
    .str.replace("　", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.title("🎨 Filter")

color = st.sidebar.selectbox(
    "Color code",
    sorted(df["塗料編號"].dropna().unique())
)

df = df[df["塗料編號"] == color]

year = st.sidebar.selectbox(
    "Year",
    sorted(df["Time"].dt.year.unique())
)

df = df[df["Time"].dt.year == year]

# =========================
# LIMIT FUNCTION (FIXED)
# =========================
def get_limit(color, factor, prefix):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty:
        return None, None

    lcl_col = f"{factor} {prefix} LCL"
    ucl_col = f"{factor} {prefix} UCL"

    lcl = row[lcl_col].iloc[0] if lcl_col in row.columns else None
    ucl = row[ucl_col].iloc[0] if ucl_col in row.columns else None

    return lcl, ucl

# =========================
# PREP SPC DATA
# =========================
def prep_spc(df, north, south):
    tmp = df.copy()
    tmp["value"] = tmp[[north, south]].mean(axis=1)
    return tmp.groupby("製造批號", as_index=False).agg(
        Time=("Time", "min"),
        value=("value", "mean")
    )

def prep_lab(df, col):
    return df.groupby("製造批號", as_index=False).agg(
        Time=("Time", "min"),
        value=(col, "mean")
    )

spc = {
    "ΔL": {
        "lab": prep_lab(df, "入料檢測 ΔL 正面"),
        "line": prep_spc(df, "正-北 ΔL", "正-南 ΔL")
    },
    "Δa": {
        "lab": prep_lab(df, "入料檢測 Δa 正面"),
        "line": prep_spc(df, "正-北 Δa", "正-南 Δa")
    },
    "Δb": {
        "lab": prep_lab(df, "入料檢測 Δb 正面"),
        "line": prep_spc(df, "正-北 Δb", "正-南 Δb")
    }
}

# =========================
# SPC SUMMARY TABLES
# =========================
def build_summary(spc_part, prefix):
    rows = []
    for k in spc:
        values = spc[k][spc_part]["value"].dropna()
        mean = values.mean()
        std = values.std()
        n = values.count()
        vmin = values.min()
        vmax = values.max()

        lcl, ucl = get_limit(color, k, prefix)

        ca = cp = cpk = None
        if std > 0 and lcl is not None and ucl is not None:
            cp = (ucl - lcl) / (6 * std)
            cpk = min(
                (ucl - mean) / (3 * std),
                (mean - lcl) / (3 * std)
            )
            ca = abs(mean - (ucl + lcl) / 2) / ((ucl - lcl) / 2)

        rows.append({
            "Factor": k,
            "Min": round(vmin, 2),
            "Max": round(vmax, 2),
            "Mean": round(mean, 2),
            "Std Dev": round(std, 2),
            "Ca": round(ca, 2) if ca is not None else "",
            "Cp": round(cp, 2) if cp is not None else "",
            "Cpk": round(cpk, 2) if cpk is not None else "",
            "n (batches)": n
        })
    return pd.DataFrame(rows)

# =========================
# PLOT FUNCTIONS
# =========================
def spc_combined(lab, line, title, lab_lim, line_lim):
    fig, ax = plt.subplots(figsize=(12, 4))

    mean = line["value"].mean()
    std = line["value"].std()

    ax.plot(lab["製造批號"], lab["value"],
            marker="o", linestyle="-",
            label="LAB", color="#1f77b4")

    ax.plot(line["製造批號"], line["value"],
            marker="s", linestyle="-",
            label="LINE", color="#2ca02c")

    ax.axhline(mean + 3 * std, color="orange", linestyle="--", label="+3σ")
    ax.axhline(mean - 3 * std, color="orange", linestyle="--", label="-3σ")

    if lab_lim[0] is not None:
        ax.axhline(lab_lim[0], color="#1f77b4", linestyle=":", label="LAB LCL")
        ax.axhline(lab_lim[1], color="#1f77b4", linestyle=":", label="LAB UCL")

    if line_lim[0] is not None:
        ax.axhline(line_lim[0], color="red", label="LINE LCL")
        ax.axhline(line_lim[1], color="red", label="LINE UCL")

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True)
    ax.tick_params(axis="x", rotation=45)
    fig.subplots_adjust(right=0.78)
    return fig

def spc_single(df_spc, title, limit, color, marker):
    fig, ax = plt.subplots(figsize=(12, 4))

    mean = df_spc["value"].mean()
    std = df_spc["value"].std()

    ax.plot(df_spc["製造批號"], df_spc["value"],
            marker=marker, linestyle="-",
            color=color)

    ax.axhline(mean + 3 * std, color="orange", linestyle="--", label="+3σ")
    ax.axhline(mean - 3 * std, color="orange", linestyle="--", label="-3σ")

    if limit[0] is not None:
        ax.axhline(limit[0], color="red", label="LCL")
        ax.axhline(limit[1], color="red", label="UCL")

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True)
    ax.tick_params(axis="x", rotation=45)
    fig.subplots_adjust(right=0.78)
    return fig

# =========================
# DASHBOARD
# =========================
st.title(f"🎨 SPC Color Dashboard — {color}")

st.markdown("### 📋 SPC Summary Statistics (LINE)")
st.dataframe(build_summary("line", "LINE"), use_container_width=True, hide_index=True)

st.markdown("### 📋 SPC Summary Statistics (LAB)")
st.dataframe(build_summary("lab", "LAB"), use_container_width=True, hide_index=True)

st.markdown("### 📊 COMBINED SPC")
for k in spc:
    st.pyplot(
        spc_combined(
            spc[k]["lab"],
            spc[k]["line"],
            f"COMBINED {k}",
            get_limit(color, k, "LAB"),
            get_limit(color, k, "LINE")
        )
    )

st.markdown("### 🧪 LAB SPC")
for k in spc:
    st.pyplot(
        spc_single(
            spc[k]["lab"],
            f"LAB {k}",
            get_limit(color, k, "LAB"),
            "#1f77b4",
            marker="o"
        )
    )

st.markdown("### 🏭 LINE SPC")
for k in spc:
    st.pyplot(
        spc_single(
            spc[k]["line"],
            f"LINE {k}",
            get_limit(color, k, "LINE"),
            "#2ca02c",
            marker="s"
        )
    )
