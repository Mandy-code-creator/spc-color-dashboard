import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="SPC Color Dashboard",
    page_icon="🎨",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://docs.google.com/spreadsheets/d/1lqsLKSoDTbtvAsHzJaEri8tPo5pA3vqJ__LVHp2R534/export?format=csv"
    )
    df["Time"] = pd.to_datetime(df["Time"])
    return df

@st.cache_data
def load_limit():
    return pd.read_csv(
        "https://docs.google.com/spreadsheets/d/1jbP8puBraQ5Xgs9oIpJ7PlLpjIK3sltrgbrgKUcJ-Qo/export?format=csv"
    )

df = load_data()
limit_df = load_limit()

# =========================
# CLEAN COLUMN NAMES
# =========================
df.columns = (
    df.columns
    .str.replace("\r\n", " ", regex=False)
    .str.replace("\n", " ", regex=False)
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
def get_limit(color, factor):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty:
        return None, None
    lcl = row[f"{factor} LINE LCL"].iloc[0]
    ucl = row[f"{factor} LINE UCL"].iloc[0]
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
    return (
        df.groupby("製造批號", as_index=False)
        .agg(value=(col, "mean"))
    )

# =========================
# SPC DATA
# =========================
spc = {
    "ΔL": {
        "line": prep_line(df, "正-北 ΔL", "正-南 ΔL"),
        "lab": prep_lab(df, "入料檢測 ΔL 正面")
    },
    "Δa": {
        "line": prep_line(df, "正-北 Δa", "正-南 Δa"),
        "lab": prep_lab(df, "入料檢測 Δa 正面")
    },
    "Δb": {
        "line": prep_line(df, "正-北 Δb", "正-南 Δb"),
        "lab": prep_lab(df, "入料檢測 Δb 正面")
    }
}

# =========================
# SPC SUMMARY
# =========================
summary = []

for k in spc:
    line = spc[k]["line"]["value"].dropna()
    lab = spc[k]["lab"]["value"].dropna()

    lcl, ucl = get_limit(color, k)

    mean = line.mean()
    std = line.std()

    cp = cpk = ca = None
    if std > 0:
        cp = (ucl - lcl) / (6 * std)
        cpk = min((ucl - mean), (mean - lcl)) / (3 * std)
        ca = abs(mean - (ucl + lcl) / 2) / ((ucl - lcl) / 2)

    summary.append({
        "Factor": k,

        "Line Min": round(line.min(), 2),
        "Line Max": round(line.max(), 2),
        "Line Mean": round(mean, 2),
        "Line Std": round(std, 2),

        "LAB Min": round(lab.min(), 2),
        "LAB Max": round(lab.max(), 2),
        "LAB Mean": round(lab.mean(), 2),
        "LAB Std": round(lab.std(), 2),

        "Ca (LINE)": round(ca, 2),
        "Cp (LINE)": round(cp, 2),
        "Cpk (LINE)": round(cpk, 2)
    })

summary_df = pd.DataFrame(summary)

# =========================
# DISPLAY SUMMARY
# =========================
st.title(f"🎨 SPC Color Dashboard — {color}")

st.markdown("### 📋 SPC Summary Statistics")

c1, c2 = st.columns(2)

with c1:
    st.markdown("#### 🏭 LINE")
    st.dataframe(
        summary_df[
            ["Factor", "Line Min", "Line Max", "Line Mean", "Line Std",
             "Ca (LINE)", "Cp (LINE)", "Cpk (LINE)"]
        ],
        use_container_width=True,
        hide_index=True
    )

with c2:
    st.markdown("#### 🧪 LAB")
    st.dataframe(
        summary_df[
            ["Factor", "LAB Min", "LAB Max", "LAB Mean", "LAB Std"]
        ],
        use_container_width=True,
        hide_index=True
    )
