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

st.markdown("""
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
[data-testid="stSidebar"] {
    background-color: #f6f8fa;
}
</style>
""", unsafe_allow_html=True)

# =========================
# REFRESH
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
# TITLE
# =========================
st.title(f"🎨 SPC Color Dashboard — {color}")

# =========================
# SPC CHART FUNCTIONS
# =========================
def spc_single(spc_df, title, limit, color):
    fig, ax = plt.subplots(figsize=(12, 4))
    mean = spc_df["value"].mean()
    std = spc_df["value"].std()

    ax.plot(spc_df["製造批號"], spc_df["value"], "o-", color=color)
    ax.axhline(mean + 3 * std, color="orange", linestyle="--", label="+3σ")
    ax.axhline(mean - 3 * std, color="orange", linestyle="--", label="-3σ")

    if limit[0] is not None:
        ax.axhline(limit[0], color="red", label="LCL")
        ax.axhline(limit[1], color="red", label="UCL")

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1))
    ax.grid(True)
    ax.tick_params(axis="x", rotation=45)
    fig.subplots_adjust(right=0.78)
    return fig

def download(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.download_button("📥 Download PNG", buf, name, "image/png")

# =========================
# SPC CHARTS
# =========================
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

# =========================
# SPC SUMMARY TABLES
# =========================
def calc_cp(std, lcl, ucl):
    if std == 0 or lcl is None or ucl is None:
        return None
    return (ucl - lcl) / (6 * std)

def calc_cpk(mean, std, lcl, ucl):
    if std == 0 or lcl is None or ucl is None:
        return None
    return min(
        (ucl - mean) / (3 * std),
        (mean - lcl) / (3 * std)
    )

def calc_ca(mean, lcl, ucl):
    if lcl is None or ucl is None:
        return None
    target = (ucl + lcl) / 2
    return abs(mean - target) / ((ucl - lcl) / 2)

def build_line_summary(spc, color):
    rows = []
    for k in spc:
        v = spc[k]["line"]["value"].dropna()
        mean = v.mean()
        std = v.std()
        lcl, ucl = get_limit(color, k, "LINE")

        rows.append({
            "Item": k,
            "Max": v.max(),
            "Min": v.min(),
            "Std": std,
            "Ca": calc_ca(mean, lcl, ucl),
            "Cp": calc_cp(std, lcl, ucl),
            "Cpk": calc_cpk(mean, std, lcl, ucl)
        })
    return pd.DataFrame(rows).set_index("Item").round(2)

def build_lab_summary(spc):
    rows = []
    for k in spc:
        v = spc[k]["lab"]["value"].dropna()
        rows.append({
            "Item": k,
            "Max": v.max(),
            "Min": v.min(),
            "Std": v.std()
        })
    return pd.DataFrame(rows).set_index("Item").round(2)

st.markdown("---")
st.markdown("## 📋 SPC Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏭 LINE Summary")
    st.dataframe(
        build_line_summary(spc, color)
        .style.format("{:.2f}")
        .set_properties(**{"text-align": "center"}),
        use_container_width=True
    )

with col2:
    st.markdown("### 🧪 LAB Summary")
    st.dataframe(
        build_lab_summary(spc)
        .style.format("{:.2f}")
        .set_properties(**{"text-align": "center"}),
        use_container_width=True
    )

# =========================
# DISTRIBUTION DASHBOARD
# =========================
st.markdown("---")
st.markdown("## 📈 Process Distribution Dashboard")

def normal_pdf(x, mean, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * np.exp(
        -0.5 * ((x - mean) / std) ** 2
    )

cols = st.columns(3)
for i, k in enumerate(spc):
    with cols[i]:
        values = spc[k]["line"]["value"].dropna()
        mean = values.mean()
        std = values.std()
        lcl, ucl = get_limit(color, k, "LINE")

        fig, ax = plt.subplots(figsize=(4, 3))
        bins = np.histogram_bin_edges(values, bins=10)
        ax.hist(values, bins=bins, color="#4dabf7", edgecolor="white")

        if std > 0:
            x = np.linspace(mean - 3 * std, mean + 3 * std, 300)
            ax.plot(x, normal_pdf(x, mean, std) * len(values) * (bins[1] - bins[0]), color="black")

        ax.set_title(k)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
