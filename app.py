import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
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

df_raw = load_data()
limit_df = load_limit()

# =========================
# FIX COLUMN NAMES
# =========================
df_raw.columns = (
    df_raw.columns
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
    sorted(df_raw["塗料編號"].dropna().unique())
)

df = df_raw[df_raw["塗料編號"] == color]

latest_year = int(df["Time"].dt.year.max())
year = st.sidebar.selectbox(
    "Year",
    sorted(df["Time"].dt.year.unique()),
    index=list(sorted(df["Time"].dt.year.unique())).index(latest_year)
)

month = st.sidebar.multiselect(
    "Month (optional)",
    sorted(df["Time"].dt.month.unique())
)

df = df[df["Time"].dt.year == year]
if month:
    df = df[df["Time"].dt.month.isin(month)]

# =========================
# GLOBAL HEADER (ALWAYS VISIBLE)
# =========================
if not df.empty:
    t_min = df["Time"].min().strftime("%Y-%m-%d")
    t_max = df["Time"].max().strftime("%Y-%m-%d")
    n_batches = df["製造批號"].nunique()
else:
    t_min, t_max, n_batches = "N/A", "N/A", 0

month_text = "All" if not month else ", ".join(map(str, month))

st.markdown(f"""
<h2 style="margin-bottom:0;">
🎨 SPC Color Dashboard — {color}
</h2>

<p style="font-size:18px; margin-top:4px; color:#444;">
⏱ {t_min} → {t_max} | n = {n_batches} batches | Year: {year} | Month: {month_text}
</p>
<hr style="margin-top:6px; margin-bottom:16px;">
""", unsafe_allow_html=True)

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
# SPC PLOT FUNCTIONS
# =========================
def spc_single(spc, title, limit, color):
    fig, ax = plt.subplots(figsize=(12, 4))
    mean = spc["value"].mean()
    std = spc["value"].std()

    ax.plot(spc["製造批號"], spc["value"], "o-", color=color)
    ax.axhline(mean + 3 * std, linestyle="--")
    ax.axhline(mean - 3 * std, linestyle="--")

    if limit[0] is not None:
        ax.axhline(limit[0], linestyle=":")
        ax.axhline(limit[1], linestyle=":")

    ax.set_title(title)
    ax.grid(True)
    ax.tick_params(axis="x", rotation=45)
    return fig

# =========================
# MAIN DASHBOARD
# =========================
st.markdown("### 🧪 LAB SPC")
for k in spc:
    st.pyplot(
        spc_single(
            spc[k]["lab"],
            f"LAB {k}",
            get_limit(color, k, "LAB"),
            "#1f77b4"
        )
    )

st.markdown("---")
st.markdown("### 🏭 LINE SPC")
for k in spc:
    st.pyplot(
        spc_single(
            spc[k]["line"],
            f"LINE {k}",
            get_limit(color, k, "LINE"),
            "#2ca02c"
        )
    )
