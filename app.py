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
# SIDEBAR STYLE
# =========================
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #f6f8fa;
    }
    </style>
    """,
    unsafe_allow_html=True
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

latest_year = df["Time"].dt.year.max()
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

st.sidebar.divider()

# =========================
# LIMIT DISPLAY
# =========================
def show_limits(factor):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty:
        return
    table = row.filter(like=factor).copy()
    for c in table.columns:
        table[c] = table[c].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    st.sidebar.markdown(f"**{factor} Control Limits**")
    st.sidebar.dataframe(table, use_container_width=True, hide_index=True)

show_limits("LAB")
show_limits("LINE")

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
# MAIN DASHBOARD
# =========================
st.title(f"🎨 SPC Color Dashboard — {color}")

# ======================================================
# 📋 SPC SUMMARY STATISTICS (LINE + LAB)
# ======================================================
summary_rows = []

for k in spc:
    line_values = spc[k]["line"]["value"].dropna()
    lab_values = spc[k]["lab"]["value"].dropna()

    line_mean = line_values.mean()
    line_std = line_values.std()
    line_min = line_values.min()
    line_max = line_values.max()
    n = line_values.count()

    lab_mean = lab_values.mean()
    lab_std = lab_values.std()
    lab_min = lab_values.min()
    lab_max = lab_values.max()

    lcl, ucl = get_limit(color, k, "LINE")

    ca = cp = cpk = None
    if line_std > 0 and lcl is not None and ucl is not None:
        cp = (ucl - lcl) / (6 * line_std)
        cpk = min(
            (ucl - line_mean) / (3 * line_std),
            (line_mean - lcl) / (3 * line_std)
        )
        ca = abs(line_mean - (ucl + lcl) / 2) / ((ucl - lcl) / 2)

    summary_rows.append({
        "Factor": k,

        "Line Min": round(line_min, 2),
        "Line Max": round(line_max, 2),
        "Line Mean": round(line_mean, 2),
        "Line Std": round(line_std, 2),

        "LAB Min": round(lab_min, 2),
        "LAB Max": round(lab_max, 2),
        "LAB Mean": round(lab_mean, 2),
        "LAB Std": round(lab_std, 2),

        "Ca (LINE)": round(ca, 2) if ca is not None else "",
        "Cp (LINE)": round(cp, 2) if cp is not None else "",
        "Cpk (LINE)": round(cpk, 2) if cpk is not None else "",

        "n (batches)": n
    })

summary_df = pd.DataFrame(summary_rows)

st.markdown("### 📋 SPC Summary Statistics (LINE + LAB)")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# =========================
# (PHẦN BIỂU ĐỒ & DISTRIBUTION GIỮ NGUYÊN – KHÔNG ĐỔI)
# =========================
# 👉 phần này của bạn đã đúng nên KHÔNG ĐỤNG NỮA
