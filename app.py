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
# GOOGLE SHEET LINKS
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
# FILTER
# =========================
st.sidebar.title("🎨 Filter")
color = st.sidebar.selectbox("Color code", sorted(df["塗料編號"].dropna().unique()))
df = df[df["塗料編號"] == color]

year = st.sidebar.selectbox("Year", sorted(df["Time"].dt.year.unique()))
df = df[df["Time"].dt.year == year]

# =========================
# LIMIT FUNCTION
# =========================
def get_limit(color, factor, mode):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty:
        return None, None
    return (
        row.get(f"{mode} {factor} LCL", [None]).values[0],
        row.get(f"{mode} {factor} UCL", [None]).values[0]
    )

# =========================
# PREP DATA
# =========================
def prep_spc(df, north, south):
    tmp = df.copy()
    tmp["value"] = tmp[[north, south]].mean(axis=1)
    return tmp.groupby("製造批號", as_index=False).agg(value=("value", "mean"))

def prep_lab(df, col):
    return df.groupby("製造批號", as_index=False).agg(value=(col, "mean"))

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
# CORE PLOT FUNCTION (OOS HIGHLIGHT)
# =========================
def plot_series(ax, df, lcl, ucl, color, label, marker):
    for _, r in df.iterrows():
        v = r["value"]
        batch = r["製造批號"]
        oos = (lcl is not None and (v < lcl or v > ucl))
        ax.scatter(
            batch, v,
            color="red" if oos else color,
            marker=marker,
            s=80,
            zorder=3
        )
        if oos:
            ax.text(batch, v, f"{batch}\n{v:.2f}",
                    fontsize=8, color="red",
                    ha="center", va="bottom")

# =========================
# COMBINED SPC
# =========================
st.markdown("### 📊 COMBINED SPC")

for k in spc:
    lab = spc[k]["lab"]
    line = spc[k]["line"]
    lab_lcl, lab_ucl = get_limit(color, k, "LAB")
    line_lcl, line_ucl = get_limit(color, k, "LINE")

    fig, ax = plt.subplots(figsize=(12, 4))

    plot_series(ax, lab, lab_lcl, lab_ucl, "#1f77b4", "LAB", "o")
    plot_series(ax, line, line_lcl, line_ucl, "#2ca02c", "LINE", "s")

    if lab_lcl is not None:
        ax.axhline(lab_lcl, color="#1f77b4", linestyle=":")
        ax.axhline(lab_ucl, color="#1f77b4", linestyle=":")

    if line_lcl is not None:
        ax.axhline(line_lcl, color="red")
        ax.axhline(line_ucl, color="red")

    ax.set_title(f"COMBINED {k}")
    ax.grid(True)
    st.pyplot(fig)

# =========================
# LAB SPC
# =========================
st.markdown("### 🧪 LAB SPC")

for k in spc:
    lab = spc[k]["lab"]
    lcl, ucl = get_limit(color, k, "LAB")

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_series(ax, lab, lcl, ucl, "#1f77b4", "LAB", "o")

    if lcl is not None:
        ax.axhline(lcl, color="red")
        ax.axhline(ucl, color="red")

    ax.set_title(f"LAB {k}")
    ax.grid(True)
    st.pyplot(fig)

# =========================
# LINE SPC
# =========================
st.markdown("### 🏭 LINE SPC")

for k in spc:
    line = spc[k]["line"]
    lcl, ucl = get_limit(color, k, "LINE")

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_series(ax, line, lcl, ucl, "#2ca02c", "LINE", "s")

    if lcl is not None:
        ax.axhline(lcl, color="red")
        ax.axhline(ucl, color="red")

    ax.set_title(f"LINE {k}")
    ax.grid(True)
    st.pyplot(fig)
