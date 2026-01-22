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
# STYLE
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(270deg,#ffffff,#f0f9ff,#e0f2fe,#fef3c7,#ecfeff);
    background-size: 800% 800%;
    animation: gradientBG 20s ease infinite;
}
@keyframes gradientBG {
    0% {background-position:0% 50%}
    50% {background-position:100% 50%}
    100% {background-position:0% 50%}
}
[data-testid="stSidebar"] {background-color:#f6f8fa;}
</style>
""", unsafe_allow_html=True)

# =========================
# DATA URL
# =========================
DATA_URL = "https://docs.google.com/spreadsheets/d/1lqsLKSoDTbtvAsHzJaEri8tPo5pA3vqJ__LVHp2R534/export?format=csv"
LIMIT_URL = "https://docs.google.com/spreadsheets/d/1jbP8puBraQ5Xgs9oIpJ7PlLpjIK3sltrgbrgKUcJ-Qo/export?format=csv"

# =========================
# LOAD DATA
# =========================
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv(DATA_URL)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    return df

@st.cache_data(ttl=300)
def load_limit():
    return pd.read_csv(LIMIT_URL)

df = load_data()
limit_df = load_limit()

# =========================
# CLEAN COLUMN NAMES
# =========================
for d in [df, limit_df]:
    d.columns = (
        d.columns.str.replace("\r\n", " ", regex=False)
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
    sorted(df["Time"].dt.year.dropna().unique())
)

month = st.sidebar.multiselect(
    "Month (optional)",
    sorted(df["Time"].dt.month.dropna().unique())
)

df = df[df["Time"].dt.year == year]
if month:
    df = df[df["Time"].dt.month.isin(month)]

if df.empty:
    st.warning("No data available")
    st.stop()

# =========================
# SIDEBAR LIMIT TABLE
# =========================
def show_limits(source):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty:
        return
    cols = [c for c in row.columns if source in c]
    table = row[cols].copy()
    for c in table.columns:
        table[c] = table[c].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    st.sidebar.markdown(f"**{source} Control Limits**")
    st.sidebar.dataframe(table, use_container_width=True, hide_index=True)

show_limits("LAB")
show_limits("LINE")

# =========================
# LIMIT FUNCTION
# =========================
def get_limit(color, factor, source):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty:
        return None, None
    lcl = row.get(f"{factor} {source} LCL", pd.Series([None])).iloc[0]
    ucl = row.get(f"{factor} {source} UCL", pd.Series([None])).iloc[0]
    return lcl, ucl

# =========================
# PREP DATA
# =========================
def prep_line(df, n, s):
    return (
        df.assign(value=df[[n, s]].mean(axis=1))
        .groupby("製造批號", as_index=False)
        .agg(value=("value", "mean"))
    )

def prep_lab(df, col):
    return df.groupby("製造批號", as_index=False).agg(value=(col, "mean"))

spc = {
    "ΔL": {
        "line": prep_line(df, "正-北 ΔL", "正-南 ΔL"),
        "lab": prep_lab(df, "入料檢測 ΔL 正面"),
    },
    "Δa": {
        "line": prep_line(df, "正-北 Δa", "正-南 Δa"),
        "lab": prep_lab(df, "入料檢測 Δa 正面"),
    },
    "Δb": {
        "line": prep_line(df, "正-北 Δb", "正-南 Δb"),
        "lab": prep_lab(df, "入料檢測 Δb 正面"),
    },
}

# =========================
# TITLE
# =========================
st.title(f"🎨 SPC Color Dashboard — {color}")

# =========================
# COMBINED SPC (BẢN ĐÚNG)
# =========================
st.markdown("### 📊 COMBINED SPC")

for k in spc:
    fig, ax = plt.subplots(figsize=(12,4))

    line = spc[k]["line"]
    lab = spc[k]["lab"]

    ax.plot(line["製造批號"], line["value"], "o-", label="LINE", color="#2ca02c")
    ax.plot(lab["製造批號"], lab["value"], "s--", label="LAB", color="#1f77b4")

    lcl_l, ucl_l = get_limit(color, k, "LINE")
    lcl_b, ucl_b = get_limit(color, k, "LAB")

    if lcl_l is not None:
        ax.axhline(lcl_l, color="#2ca02c", linestyle=":", label="LINE LCL")
        ax.axhline(ucl_l, color="#2ca02c", linestyle=":")

    if lcl_b is not None:
        ax.axhline(lcl_b, color="#1f77b4", linestyle=":", label="LAB LCL")
        ax.axhline(ucl_b, color="#1f77b4", linestyle=":")

    ax.set_title(f"COMBINED {k}")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# =========================
# DISTRIBUTION
# =========================
st.markdown("## 📈 LINE Distribution")

cols = st.columns(3)
for i, k in enumerate(spc):
    with cols[i]:
        values = spc[k]["line"]["value"].dropna()
        fig, ax = plt.subplots(figsize=(4,3))
        ax.hist(values, bins=10, edgecolor="white")
        ax.set_title(k)
        st.pyplot(fig)

st.markdown("## 📈 LAB Distribution")

cols = st.columns(3)
for i, k in enumerate(spc):
    with cols[i]:
        values = spc[k]["lab"]["value"].dropna()
        fig, ax = plt.subplots(figsize=(4,3))
        ax.hist(values, bins=10, edgecolor="white")
        ax.set_title(f"{k} (LAB)")
        st.pyplot(fig)
