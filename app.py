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

if not df.empty:
    t_min = df["Time"].min().strftime("%Y-%m-%d")
    t_max = df["Time"].max().strftime("%Y-%m-%d")
    n_batch = df["製造批號"].nunique()
else:
    t_min = t_max = "N/A"
    n_batch = 0

st.markdown(
    f"⏱ **{t_min} → {t_max} | n = {n_batch} batches | Year: {year} | Month: {'All' if not month else month}**"
)

# ======================================================
# =========================
# SUMMARY TABLES
# =========================

def calc_line_summary(spc_dict, color):
    rows = []
    for k in spc_dict:
        values = spc_dict[k]["line"]["value"].dropna()
        mean = values.mean()
        std = values.std()

        lcl, ucl = get_limit(color, k, "LINE")

        Ca = Cp = Cpk = np.nan
        if std > 0 and lcl is not None and ucl is not None:
            Ca = abs((mean - (ucl + lcl) / 2) / ((ucl - lcl) / 2))
            Cp = (ucl - lcl) / (6 * std)
            Cpk = min((ucl - mean) / (3 * std), (mean - lcl) / (3 * std))

        rows.append({
            "Item": k,
            "Max": values.max(),
            "Min": values.min(),
            "Stdev": std,
            "Ca": Ca,
            "Cp": Cp,
            "Cpk": Cpk
        })

    df = pd.DataFrame(rows)
    num_cols = df.columns.drop("Item")
    df[num_cols] = df[num_cols].round(2)
    return df


def calc_lab_summary(spc_dict):
    rows = []
    for k in spc_dict:
        values = spc_dict[k]["lab"]["value"].dropna()
        rows.append({
            "Item": k,
            "Max": values.max(),
            "Min": values.min(),
            "Stdev": values.std()
        })

    df = pd.DataFrame(rows)
    num_cols = df.columns.drop("Item")
    df[num_cols] = df[num_cols].round(2)
    return df


line_df = calc_line_summary(spc, color)
lab_df = calc_lab_summary(spc)

st.markdown("### 📋 SPC Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏭 LINE Summary")
    st.dataframe(
        line_df,
        use_container_width=True,
        hide_index=True
    )

with col2:
    st.markdown("#### 🧪 LAB Summary")
    st.dataframe(
        lab_df,
        use_container_width=True,
        hide_index=True
    )

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
        get_limit(color, k, "LINE")
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

        for p, l, r in zip(patches, bins[:-1], bins[1:]):
            center = (l + r) / 2
            if lcl is not None and ucl is not None:
                if center < lcl or center > ucl:
                    p.set_facecolor("red")

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


