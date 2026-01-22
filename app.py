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
df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.title("🎨 Filter")
color = st.sidebar.selectbox("Color code", sorted(df["塗料編號"].dropna().unique()))
df = df[df["塗料編號"] == color]

year = st.sidebar.selectbox("Year", sorted(df["Time"].dt.year.unique()))
df = df[df["Time"].dt.year == year]

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
    return tmp.groupby("製造批號", as_index=False)["value"].mean()

def prep_lab(df, col):
    return df.groupby("製造批號", as_index=False)[col].mean().rename(columns={col: "value"})

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
# SUMMARY TABLES
# =========================
line_rows = []
lab_rows = []

for k in spc:
    line_vals = spc[k]["line"]["value"].dropna()
    lab_vals = spc[k]["lab"]["value"].dropna()

    lcl, ucl = get_limit(color, k, "LINE")
    mean = line_vals.mean()
    std = line_vals.std()

    cp = cpk = ca = None
    if std > 0 and lcl is not None and ucl is not None:
        cp = (ucl - lcl) / (6 * std)
        cpk = min((ucl - mean), (mean - lcl)) / (3 * std)
        ca = abs(mean - (ucl + lcl) / 2) / ((ucl - lcl) / 2)

    line_rows.append([
        k,
        round(line_vals.min(), 2),
        round(line_vals.max(), 2),
        round(std, 2),
        round(ca, 2) if ca else "",
        round(cp, 2) if cp else "",
        round(cpk, 2) if cpk else ""
    ])

    lab_rows.append([
        k,
        round(lab_vals.min(), 2),
        round(lab_vals.max(), 2),
        round(lab_vals.std(), 2)
    ])

line_df = pd.DataFrame(
    line_rows,
    columns=["Factor", "Min", "Max", "Std Dev", "Ca", "Cp", "Cpk"]
)

lab_df = pd.DataFrame(
    lab_rows,
    columns=["Factor", "Min", "Max", "Std Dev"]
)

# =========================
# DISPLAY SUMMARY TABLES
# =========================
st.markdown("## 📋 SPC Summary")

c1, c2 = st.columns(2)

with c1:
    st.markdown("### 🏭 LINE")
    st.dataframe(line_df, use_container_width=True)

with c2:
    st.markdown("### 🧪 LAB")
    st.dataframe(lab_df, use_container_width=True)

# =========================
# SPC CHART FUNCTIONS
# =========================
def spc_single(df, title, limit, color):
    fig, ax = plt.subplots(figsize=(12, 4))
    mean = df["value"].mean()
    std = df["value"].std()
    ax.plot(df.index, df["value"], "o-", color=color)
    ax.axhline(mean + 3 * std, linestyle="--", color="orange")
    ax.axhline(mean - 3 * std, linestyle="--", color="orange")
    if limit[0] is not None:
        ax.axhline(limit[0], color="red")
        ax.axhline(limit[1], color="red")
    ax.set_title(title)
    ax.grid(True)
    return fig

def download(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    buf.seek(0)
    st.download_button("📥 Download PNG", buf, name, "image/png")

# =========================
# DASHBOARD
# =========================
st.markdown("---")
st.markdown("### 🏭 LINE SPC")

for k in spc:
    fig = spc_single(
        spc[k]["line"],
        f"LINE {k}",
        get_limit(color, k, "LINE"),
        "#2ca02c"
    )
    st.pyplot(fig)
    download(fig, f"LINE_{color}_{k}.png")

st.markdown("---")
st.markdown("### 🧪 LAB SPC")

for k in spc:
    fig = spc_single(
        spc[k]["lab"],
        f"LAB {k}",
        get_limit(color, k, "LAB"),
        "#1f77b4"
    )
    st.pyplot(fig)
    download(fig, f"LAB_{color}_{k}.png")
