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
    df.columns.str.replace("\n", " ")
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
# LIMIT FUNCTION
# =========================
def get_limit(color, factor, mode):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty:
        return None, None
    return (
        row.get(f"{factor} {mode} LCL", [None]).values[0],
        row.get(f"{factor} {mode} UCL", [None]).values[0]
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
    mean, std = v.mean(), v.std()
    lcl, ucl = get_limit(color, k, "LINE")

    summary_line.append({
        "Factor": k,
        "Min": round(v.min(), 2),
        "Max": round(v.max(), 2),
        "Mean": round(mean, 2),
        "Std Dev": round(std, 2),
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
