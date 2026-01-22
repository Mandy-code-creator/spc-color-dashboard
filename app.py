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
# SIDEBAR FILTER
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

# =========================
# GET LIMIT (FIXED – NO ERROR)
# =========================
def get_limit(color, factor, mode):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty:
        return None, None

    lcl_col = f"{factor} {mode} LCL"
    ucl_col = f"{factor} {mode} UCL"

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
# HIGHLIGHT FUNCTION
# =========================
def highlight_out_of_limit(row, factor, mode):
    styles = [""] * len(row)
    lcl, ucl = get_limit(color, factor, mode)

    for i, col in enumerate(row.index):
        if col == "Min" and lcl is not None and row[col] < lcl:
            styles[i] = "background-color:#ffcccc"
        if col == "Max" and ucl is not None and row[col] > ucl:
            styles[i] = "background-color:#ffcccc"

    return styles

# =========================
# SUMMARY TABLE
# =========================
summary_line = []
summary_lab = []

for k in spc:
    # LINE
    v = spc[k]["line"]["value"].dropna()
    summary_line.append({
        "Factor": k,
        "Min": round(v.min(), 2),
        "Max": round(v.max(), 2),
        "Mean": round(v.mean(), 2),
        "Std Dev": round(v.std(), 2),
        "n": v.count()
    })

    # LAB
    v = spc[k]["lab"]["value"].dropna()
    summary_lab.append({
        "Factor": k,
        "Min": round(v.min(), 2),
        "Max": round(v.max(), 2),
        "Mean": round(v.mean(), 2),
        "Std Dev": round(v.std(), 2),
        "n": v.count()
    })

summary_line_df = pd.DataFrame(summary_line)
summary_lab_df = pd.DataFrame(summary_lab)

styled_line = summary_line_df.style.apply(
    lambda r: highlight_out_of_limit(r, r["Factor"], "LINE"),
    axis=1
).format("{:.2f}", subset=["Min", "Max", "Mean", "Std Dev"])

styled_lab = summary_lab_df.style.apply(
    lambda r: highlight_out_of_limit(r, r["Factor"], "LAB"),
    axis=1
).format("{:.2f}", subset=["Min", "Max", "Mean", "Std Dev"])

# =========================
# DISPLAY
# =========================
st.title(f"🎨 SPC Color Dashboard — {color}")
st.markdown("### 📋 SPC Summary Statistics")

c1, c2 = st.columns(2)

with c1:
    st.markdown("#### 🏭 LINE")
    st.dataframe(styled_line, use_container_width=True)

with c2:
    st.markdown("#### 🧪 LAB")
    st.dataframe(styled_lab, use_container_width=True)
