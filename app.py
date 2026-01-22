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
df.columns = (
    df.columns
    .str.replace("\r\n", " ", regex=False)
    .str.replace("\n", " ", regex=False)
    .str.replace("　", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

limit_df.columns = (
    limit_df.columns
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

if df.empty:
    st.warning("No data available for this selection.")
    st.stop()

# =========================
# SAFE LIMIT FUNCTION (FIXED)
# =========================
def get_limit(color, factor, source):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty:
        return None, None

    lcl_col = f"{factor} {source} LCL"
    ucl_col = f"{factor} {source} UCL"

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
# TITLE
# =========================
st.title(f"🎨 SPC Color Dashboard — {color}")

# =========================
# 📋 SPC SUMMARY (LINE + LAB)
# =========================
summary = []

for k in spc:
    line = spc[k]["line"]["value"].dropna()
    lab = spc[k]["lab"]["value"].dropna()

    lcl, ucl = get_limit(color, k, "LINE")

    line_min, line_max = line.min(), line.max()
    line_mean, line_std = line.mean(), line.std()

    lab_min, lab_max = lab.min(), lab.max()
    lab_mean, lab_std = lab.mean(), lab.std()

    cp = cpk = ca = None
    if line_std > 0 and lcl is not None and ucl is not None:
        cp = (ucl - lcl) / (6 * line_std)
        cpk = min(
            (ucl - line_mean) / (3 * line_std),
            (line_mean - lcl) / (3 * line_std),
        )
        ca = abs(line_mean - (ucl + lcl) / 2) / ((ucl - lcl) / 2)

    summary.append({
        "Factor": k,

        "Line Min": round(line_min, 2),
        "Line Max": round(line_max, 2),
        "Line Mean": round(line_mean, 2),
        "Line Std": round(line_std, 2),

        "LAB Min": round(lab_min, 2),
        "LAB Max": round(lab_max, 2),
        "LAB Mean": round(lab_mean, 2),
        "LAB Std": round(lab_std, 2),

        "Ca (LINE)": round(ca, 2) if ca is not None else "-",
        "Cp (LINE)": round(cp, 2) if cp is not None else "-",
        "Cpk (LINE)": round(cpk, 2) if cpk is not None else "-",
    })

summary_df = pd.DataFrame(summary)

st.markdown("### 📋 SPC Summary Statistics")

c1, c2 = st.columns(2)

with c1:
    st.markdown("#### 🏭 LINE")
    st.dataframe(
        summary_df[
            [
                "Factor",
                "Line Min",
                "Line Max",
                "Line Mean",
                "Line Std",
                "Ca (LINE)",
                "Cp (LINE)",
                "Cpk (LINE)",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with c2:
    st.markdown("#### 🧪 LAB")
    st.dataframe(
        summary_df[
            [
                "Factor",
                "LAB Min",
                "LAB Max",
                "LAB Mean",
                "LAB Std",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
# =========================
# SPC CHARTS
# =========================
def spc_combined(lab, line, title, lab_lim, line_lim):
    fig, ax = plt.subplots(figsize=(12, 4))

    mean = line["value"].mean()
    std = line["value"].std()

    ax.plot(lab["製造批號"], lab["value"], "o-", label="LAB", color="#1f77b4")
    ax.plot(line["製造批號"], line["value"], "o-", label="LINE", color="#2ca02c")

    ax.axhline(mean + 3 * std, color="orange", linestyle="--", label="+3σ")
    ax.axhline(mean - 3 * std, color="orange", linestyle="--", label="-3σ")

    if lab_lim[0] is not None:
        ax.axhline(lab_lim[0], color="#1f77b4", linestyle=":", label="LAB LCL")
        ax.axhline(lab_lim[1], color="#1f77b4", linestyle=":", label="LAB UCL")

    if line_lim[0] is not None:
        ax.axhline(line_lim[0], color="red", label="LINE LCL")
        ax.axhline(line_lim[1], color="red", label="LINE UCL")

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True)
    ax.tick_params(axis="x", rotation=45)
    fig.subplots_adjust(right=0.78)
    return fig


def spc_single(spc, title, limit, color):
    fig, ax = plt.subplots(figsize=(12, 4))

    mean = spc["value"].mean()
    std = spc["value"].std()

    ax.plot(spc["製造批號"], spc["value"], "o-", color=color)
    ax.axhline(mean + 3 * std, color="orange", linestyle="--", label="+3σ")
    ax.axhline(mean - 3 * std, color="orange", linestyle="--", label="-3σ")

    if limit[0] is not None:
        ax.axhline(limit[0], color="red", label="LCL")
        ax.axhline(limit[1], color="red", label="UCL")

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
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
# DASHBOARD
# =========================
st.markdown("### 📊 COMBINED SPC")
for k in spc:
    fig = spc_combined(
        spc[k]["lab"],
        spc[k]["line"],
        f"COMBINED {k}",
        get_limit(color, k, "LAB"),
        get_limit(color, k, "LINE"),
    )
    st.pyplot(fig)
    download(fig, f"COMBINED_{color}_{k}.png")

st.markdown("---")

st.markdown("### 🧪 LAB SPC")
for k in spc:
    fig = spc_single(
        spc[k]["lab"],
        f"LAB {k}",
        get_limit(color, k, "LAB"),
        "#1f77b4",
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
        "#2ca02c",
    )
    st.pyplot(fig)
    download(fig, f"LINE_{color}_{k}.png")
# =========================
# DISTRIBUTION DASHBOARD
# =========================
st.markdown("---")
st.markdown("## 📈 Line Process Distribution Dashboard")

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
        counts, _, patches = ax.hist(
            values,
            bins=bins,
            edgecolor="white",
            color="#4dabf7"
        )

        # Highlight out-of-spec bins
        for p, l, r in zip(patches, bins[:-1], bins[1:]):
            center = (l + r) / 2
            if lcl is not None and ucl is not None:
                if center < lcl or center > ucl:
                    p.set_facecolor("red")

        # Normal curve
        if std > 0:
            x = np.linspace(mean - 3 * std, mean + 3 * std, 300)
            pdf = normal_pdf(x, mean, std)
            ax.plot(
                x,
                pdf * len(values) * (bins[1] - bins[0]),
                color="black"
            )

        ax.set_title(k)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)

# =========================
# LAB DISTRIBUTION DASHBOARD
# =========================
st.markdown("---")
st.markdown("## 📈 LAB Process Distribution Dashboard")

cols = st.columns(3)

for i, k in enumerate(spc):
    with cols[i]:
        values = spc[k]["lab"]["value"].dropna()
        mean = values.mean()
        std = values.std()
        lcl, ucl = get_limit(color, k, "LAB")

        fig, ax = plt.subplots(figsize=(4, 3))

        bins = np.histogram_bin_edges(values, bins=10)
        counts, _, patches = ax.hist(
            values,
            bins=bins,
            edgecolor="white",
            color="#1f77b4"
        )

        # Highlight out-of-spec bins
        for p, l, r in zip(patches, bins[:-1], bins[1:]):
            center = (l + r) / 2
            if lcl is not None and ucl is not None:
                if center < lcl or center > ucl:
                    p.set_facecolor("red")

        # Normal curve
        if std > 0:
            x = np.linspace(mean - 3 * std, mean + 3 * std, 300)
            pdf = normal_pdf(x, mean, std)
            ax.plot(
                x,
                pdf * len(values) * (bins[1] - bins[0]),
                color="black"
            )

        ax.set_title(f"{k} (LAB)")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)

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


