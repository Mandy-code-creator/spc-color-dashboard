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
    df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
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

month = st.sidebar.multiselect(
    "Month (optional)",
    sorted(df["Time"].dt.month.unique())
)

df = df[df["Time"].dt.year == year]
if month:
    df = df[df["Time"].dt.month.isin(month)]

# =========================
# GET LIMIT (FIXED)
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
# SPC SINGLE (LAB / LINE)
# =========================
def spc_single(spc_df, title, limit, base_color, marker):
    fig, ax = plt.subplots(figsize=(12, 4))

    mean = spc_df["value"].mean()
    std = spc_df["value"].std()

    x = spc_df["製造批號"]
    y = spc_df["value"]
    lcl, ucl = limit

    for i in range(len(y)):
        out = (
            lcl is not None and ucl is not None and
            (y.iloc[i] < lcl or y.iloc[i] > ucl)
        )

        ax.plot(
            x.iloc[i],
            y.iloc[i],
            marker=marker,
            linestyle="None",
            color="red" if out else base_color
        )

        if out:
            ax.text(
                x.iloc[i],
                y.iloc[i],
                f"{y.iloc[i]:.2f}",
                color="red",
                fontsize=9,
                ha="center",
                va="bottom"
            )

    ax.plot(x, y, "-", color=base_color, alpha=0.5)

    ax.axhline(mean + 3 * std, color="orange", linestyle="--")
    ax.axhline(mean - 3 * std, color="orange", linestyle="--")

    if lcl is not None:
        ax.axhline(lcl, color="red")
        ax.axhline(ucl, color="red")

    ax.set_title(title)
    ax.grid(True)
    ax.tick_params(axis="x", rotation=45)

    return fig

# =========================
# SPC COMBINED
# =========================
def spc_combined(lab, line, title, lab_lim, line_lim):
    fig, ax = plt.subplots(figsize=(12, 4))

    mean = line["value"].mean()
    std = line["value"].std()

    # LAB
    for x, y in zip(lab["製造批號"], lab["value"]):
        out = lab_lim[0] is not None and (y < lab_lim[0] or y > lab_lim[1])
        ax.plot(x, y, marker="o", linestyle="None",
                color="red" if out else "#1f77b4")
        if out:
            ax.text(x, y, f"{y:.2f}", color="red", fontsize=9)

    ax.plot(lab["製造批號"], lab["value"], "-", color="#1f77b4", alpha=0.5, label="LAB")

    # LINE
    for x, y in zip(line["製造批號"], line["value"]):
        out = line_lim[0] is not None and (y < line_lim[0] or y > line_lim[1])
        ax.plot(x, y, marker="s", linestyle="None",
                color="red" if out else "#2ca02c")
        if out:
            ax.text(x, y, f"{y:.2f}", color="red", fontsize=9)

    ax.plot(line["製造批號"], line["value"], "-", color="#2ca02c", alpha=0.5, label="LINE")

    ax.axhline(mean + 3 * std, color="orange", linestyle="--")
    ax.axhline(mean - 3 * std, color="orange", linestyle="--")

    ax.legend()
    ax.grid(True)
    ax.tick_params(axis="x", rotation=45)

    return fig

# =========================
# DASHBOARD
# =========================
st.title(f"🎨 SPC Color Dashboard — {color}")

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

    st.pyplot(
        spc_single(
            spc[k]["lab"],
            f"LAB {k}",
            get_limit(color, k, "LAB"),
            "#1f77b4",
            "o"
        )
    )

    st.pyplot(
        spc_single(
            spc[k]["line"],
            f"LINE {k}",
            get_limit(color, k, "LINE"),
            "#2ca02c",
            "s"
        )
    )
