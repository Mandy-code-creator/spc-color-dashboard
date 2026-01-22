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

    .stDataFrame th, .stDataFrame td {
        text-align: center !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# REFRESH BUTTON
# =========================
if st.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

# =========================
# GOOGLE SHEET LINKS
# =========================
DATA_URL = "https://docs.google.com/spreadsheets/d/1lqsLKSoDTbtvAsHzJaEri8tPo5pA3vqJ__LVHp2R534/export?format=csv"
LIMIT_URL = "https://docs.google.com/spreadsheets/d/1jbP8puBraQ5Xgs9oIpJ7PlLpjIK3sltrgbrgKUcJ-Qo/export?format=csv"

# =========================
# LOAD DATA
# =========================
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
# FIX COLUMN NAMES
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
# SIDEBAR – FILTER
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

month = st.sidebar.multiselect(
    "Month (optional)",
    sorted(df["Time"].dt.month.unique())
)

df = df[df["Time"].dt.year == year]
if month:
    df = df[df["Time"].dt.month.isin(month)]

# =========================
# LIMIT FUNCTION
# =========================
def get_limit(color, prefix, factor):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty:
        return None, None
    return (
        row.get(f"{factor} {prefix} LCL", [None]).values[0],
        row.get(f"{factor} {prefix} UCL", [None]).values[0]
    )

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

# =========================
# SPC DATA
# =========================
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
# DOWNLOAD FUNCTION
# =========================
def download(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button(
        f"⬇ Download {filename}",
        data=buf.getvalue(),
        file_name=filename,
        mime="image/png"
    )

# =========================
# SPC PLOT FUNCTIONS
# =========================
def spc_single(df, title, limits, color):
    lcl, ucl = limits
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df["Time"], df["value"], marker="o", color=color)
    ax.axhline(0, linestyle=":", color="gray")

    if lcl is not None:
        ax.axhline(lcl, color="red", linestyle="--")
        ax.text(1.01, lcl, f"LCL {lcl:.2f}",
                transform=ax.get_yaxis_transform(), va="center", color="red")

    if ucl is not None:
        ax.axhline(ucl, color="red", linestyle="--")
        ax.text(1.01, ucl, f"UCL {ucl:.2f}",
                transform=ax.get_yaxis_transform(), va="center", color="red")

    ax.text(1.01, 0, "Target 0",
            transform=ax.get_yaxis_transform(), va="center", color="gray")

    ax.set_title(title)
    ax.grid(alpha=0.3)

    plt.subplots_adjust(right=0.82)
    return fig


def spc_combined(lab_df, line_df, title, lab_limits, line_limits):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(lab_df["Time"], lab_df["value"], marker="o",
            label="LAB", color="#1f77b4")
    ax.plot(line_df["Time"], line_df["value"], marker="s",
            label="LINE", color="#2ca02c")

    lab_lcl, lab_ucl = lab_limits
    line_lcl, line_ucl = line_limits

    if lab_lcl is not None:
        ax.axhline(lab_lcl, color="#1f77b4", linestyle="--", alpha=0.6)
        ax.text(1.01, lab_lcl, f"LAB LCL {lab_lcl:.2f}",
                transform=ax.get_yaxis_transform(), va="center",
                color="#1f77b4")

    if lab_ucl is not None:
        ax.axhline(lab_ucl, color="#1f77b4", linestyle="--", alpha=0.6)
        ax.text(1.01, lab_ucl, f"LAB UCL {lab_ucl:.2f}",
                transform=ax.get_yaxis_transform(), va="center",
                color="#1f77b4")

    if line_lcl is not None:
        ax.axhline(line_lcl, color="#2ca02c", linestyle="--", alpha=0.6)
        ax.text(1.01, line_lcl, f"LINE LCL {line_lcl:.2f}",
                transform=ax.get_yaxis_transform(), va="center",
                color="#2ca02c")

    if line_ucl is not None:
        ax.axhline(line_ucl, color="#2ca02c", linestyle="--", alpha=0.6)
        ax.text(1.01, line_ucl, f"LINE UCL {line_ucl:.2f}",
                transform=ax.get_yaxis_transform(), va="center",
                color="#2ca02c")

    ax.axhline(0, linestyle=":", color="gray")
    ax.text(1.01, 0, "Target 0",
            transform=ax.get_yaxis_transform(), va="center", color="gray")

    ax.legend()
    ax.set_title(title)
    ax.grid(alpha=0.3)

    plt.subplots_adjust(right=0.82)
    return fig

# =========================
# MAIN DASHBOARD
# =========================
st.title(f"🎨 SPC Color Dashboard — {color}")

st.markdown("### 📊 COMBINED SPC")
for k in spc:
    fig = spc_combined(
        spc[k]["lab"],
        spc[k]["line"],
        f"COMBINED {k}",
        get_limit(color, k, "LAB"),
        get_limit(color, k, "LINE")
    )
    st.pyplot(fig)
    download(fig, f"COMBINED_{color}_{k}.png")

st.markdown("### 🧪 LAB SPC")
for k in spc:
    fig = spc_single(
        spc[k]["lab"],
        f"LAB {k}",
        get_limit(color, k, "LAB"),
        "#1f77b4"
    )
    st.pyplot(fig)

st.markdown("### 🏭 LINE SPC")
for k in spc:
    fig = spc_single(
        spc[k]["line"],
        f"LINE {k}",
        get_limit(color, k, "LINE"),
        "#2ca02c"
    )
    st.pyplot(fig)
