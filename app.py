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
    page_icon="📊",
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
# ===== CHỌN NĂM =====
df["date"] = pd.to_datetime(df["Time"])
df["year"] = df["date"].dt.year

all_years = sorted(df["year"].unique())
latest_year = max(all_years)

selected_years = st.sidebar.multiselect(
    "📅  Select Year(s)",
    options=all_years,
    default=[latest_year]
)

df = df[df["year"].isin(selected_years)]


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
# OUT-OF-CONTROL DETECTION
# =========================
def detect_out_of_control(spc_df, lcl, ucl):
    """
    spc_df: DataFrame có cột ['製造批號', 'value']
    """
    mean = spc_df["value"].mean()
    std = spc_df["value"].std()

    result = spc_df.copy()

    result["Rule_CL"] = False
    result["Rule_3Sigma"] = False

    if lcl is not None and ucl is not None:
        result["Rule_CL"] = (
            (result["value"] < lcl) |
            (result["value"] > ucl)
        )

    if std > 0:
        result["Rule_3Sigma"] = (
            (result["value"] > mean + 3 * std) |
            (result["value"] < mean - 3 * std)
        )

    result["Out_of_Control"] = (
        result["Rule_CL"] | result["Rule_3Sigma"]
    )

    return result[result["Out_of_Control"]]

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
st.title(f"📊 SPC Color Dashboard — {color}")

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
# ======================================================
# 📋 SUMMARY TABLE (LAB & LINE)
# ======================================================
summary_line = []
summary_lab = []

for k in spc:
    # ===== LINE =====
    line_values = spc[k]["line"]["value"].dropna()
    line_mean = line_values.mean()
    line_std = line_values.std()
    line_n = line_values.count()

    line_min = line_values.min()
    line_max = line_values.max()

    lcl, ucl = get_limit(color, k, "LINE")
    summary_line.append({
        "Factor": k,
        "Min": round(line_min, 2),
        "Max": round(line_max, 2),
        "Mean": round(line_mean, 2),
        "Std Dev": round(line_std, 2),
        "n": line_n
    })

    # ===== LAB =====
    lab_values = spc[k]["lab"]["value"].dropna()
    lab_mean = lab_values.mean()
    lab_std = lab_values.std()
    lab_n = lab_values.count()

    lab_min = lab_values.min()
    lab_max = lab_values.max()

    summary_lab.append({
        "Factor": k,
        "Min": round(lab_min, 2),
        "Max": round(lab_max, 2),
        "Mean": round(lab_mean, 2),
        "Std Dev": round(lab_std, 2),
        "n": lab_n
    })

summary_line_df = pd.DataFrame(summary_line)
summary_lab_df = pd.DataFrame(summary_lab)

# =========================
# DISPLAY SIDE BY SIDE
# =========================
st.markdown("### 📋 Summary Statistics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏭 LINE")
    st.dataframe(summary_line_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("#### 🧪 LAB")
    st.dataframe(summary_lab_df, use_container_width=True, hide_index=True)

# =========================
# SPC CHARTS (GIỮ NGUYÊN)
# =========================
def spc_combined(lab, line, title, lab_lim, line_lim):
    fig, ax = plt.subplots(figsize=(12, 4))

    mean = line["value"].mean()
    std = line["value"].std()

    # ===== original lines (GIỮ NGUYÊN) =====
    ax.plot(lab["製造批號"], lab["value"], "o-", label="LAB", color="#1f77b4")
    ax.plot(line["製造批號"], line["value"], "o-", label="LINE", color="#2ca02c")

    # ===== highlight LAB out-of-limit =====
    x_lab = lab["製造批號"]
    y_lab = lab["value"]
    LCL_lab, UCL_lab = lab_lim

    if LCL_lab is not None and UCL_lab is not None:
        out_lab = (y_lab > UCL_lab) | (y_lab < LCL_lab)
        ax.scatter(x_lab[out_lab], y_lab[out_lab], color="red", s=80, zorder=5)

    # ===== highlight LINE out-of-limit =====
    x_line = line["製造批號"]
    y_line = line["value"]
    LCL_line, UCL_line = line_lim

    if LCL_line is not None and UCL_line is not None:
        out_line = (y_line > UCL_line) | (y_line < LCL_line)
        ax.scatter(x_line[out_line], y_line[out_line], color="red", s=80, zorder=5)

    
    # ===== control limits =====
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

    # original line
    ax.plot(spc["製造批號"], spc["value"], "o-", color=color)

    # highlight out-of-limit
    x = spc["製造批號"]
    y = spc["value"]
    LCL, UCL = limit

    if LCL is not None and UCL is not None:
        out = (y > UCL) | (y < LCL)
        ax.scatter(x[out], y[out], color="red", s=80, zorder=5)

    ax.axhline(mean + 3 * std, color="orange", linestyle="--", label="+3σ")
    ax.axhline(mean - 3 * std, color="orange", linestyle="--", label="-3σ")

    if LCL is not None:
        ax.axhline(LCL, color="red", label="LCL")
        ax.axhline(UCL, color="red", label="UCL")

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
st.markdown("### 📊 CONTROL CHART: LAB-LINE")
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



# =========================
# =========================
# =========================
# DISTRIBUTION DASHBOARD
# =========================

def calc_capability(values, lcl, ucl):
    if lcl is None or ucl is None:
        return None, None, None

    mean = values.mean()
    std = values.std()

    if std == 0 or np.isnan(std):
        return None, None, None

    cp = (ucl - lcl) / (6 * std)
    cpk = min(
        (ucl - mean) / (3 * std),
        (mean - lcl) / (3 * std)
    )
    ca = abs(mean - (ucl + lcl) / 2) / ((ucl - lcl) / 2)

    return round(ca, 2), round(cp, 2), round(cpk, 2)


def normal_pdf(x, mean, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * np.exp(
        -0.5 * ((x - mean) / std) ** 2
    )


# =========================
# =========================
# LINE PROCESS DISTRIBUTION
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

        if len(values) < 3:
            st.warning("Not enough data")
            continue

        mean = values.mean()
        std = values.std()
        lcl, ucl = get_limit(color, k, "LINE")

        # ===== capability (BẮT BUỘC PHẢI Ở ĐÂY) =====
        ca, cp, cpk = calc_capability(values, lcl, ucl)

        fig, ax = plt.subplots(figsize=(5, 4))

        # ===== Histogram =====
        bins = np.histogram_bin_edges(values, bins=10)
        counts, _, patches = ax.hist(
            values,
            bins=bins,
            edgecolor="white",
            color="#4dabf7",
            alpha=0.85
        )

        # ===== Highlight out-of-spec bins =====
        for p, l, r in zip(patches, bins[:-1], bins[1:]):
            center = (l + r) / 2
            if lcl is not None and ucl is not None:
                if center < lcl or center > ucl:
                    p.set_facecolor("#ff6b6b")

        # ===== Normal curve (long tail) =====
        if std > 0:
            x = np.linspace(mean - 4 * std, mean + 4 * std, 500)
            pdf = normal_pdf(x, mean, std)
            ax.plot(
                x,
                pdf * len(values) * (bins[1] - bins[0]),
                color="black",
                linewidth=2
            )

        # ===== USL / LSL =====
        if lcl is not None:
            ax.axvline(lcl, color="red", linestyle="--", linewidth=1.5, label="LSL")
        if ucl is not None:
            ax.axvline(ucl, color="red", linestyle="--", linewidth=1.5, label="USL")

        # ===== Capability box =====
        if cp is not None:
            ax.text(
                0.98, 0.95,
                f"Ca  = {ca}\nCp  = {cp}\nCpk = {cpk}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.9)
            )

        # ===== Info box =====
        ax.text(
            0.02, 0.95,
            f"N = {len(values)}\n"
            f"Mean = {mean:.3f}\n"
            f"Std = {std:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.9)
        )

        ax.set_title(f"{k} (LINE)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

        # ===== SHOW FIG =====
        st.pyplot(fig)

        # =========================
        # DOWNLOAD IMAGE
        # =========================
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            label="⬇ Download chart image",
            data=buf,
            file_name=f"{k}_line_distribution.png",
            mime="image/png"
        )

        # =========================
        # BIN SUMMARY TABLE
        # =========================
        bin_edges = np.histogram_bin_edges(values, bins=10)
        counts, _ = np.histogram(values, bins=bin_edges)
        bin_width = bin_edges[1] - bin_edges[0]

        bin_df = pd.DataFrame({
            "Bin Range": [
                f"{bin_edges[j]:.3f} ~ {bin_edges[j+1]:.3f}"
                for j in range(len(bin_edges) - 1)
            ],
            "Count": counts,
            "Density": (counts / (len(values) * bin_width)).round(4)
        })

        with st.expander("📊 Distribution bin details"):
            st.dataframe(bin_df, use_container_width=True, hide_index=True)


# =========================
# =========================
# LAB PROCESS DISTRIBUTION
# =========================
st.markdown("---")
st.markdown("## 🧪 LAB Process Distribution Dashboard")

def normal_pdf(x, mean, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * np.exp(
        -0.5 * ((x - mean) / std) ** 2
    )

cols = st.columns(3)

for i, k in enumerate(spc):
    with cols[i]:
        values = spc[k]["lab"]["value"].dropna()

        if len(values) < 3:
            st.warning("Not enough data")
            continue

        mean = values.mean()
        std = values.std()
        lcl, ucl = get_limit(color, k, "LAB")

        # ===== capability =====
        ca, cp, cpk = calc_capability(values, lcl, ucl)

        fig, ax = plt.subplots(figsize=(5, 4))

        # ===== Histogram =====
        bins = np.histogram_bin_edges(values, bins=10)
        counts, _, patches = ax.hist(
            values,
            bins=bins,
            edgecolor="white",
            color="#1f77b4",
            alpha=0.85
        )

        # ===== Highlight out-of-spec bins =====
        for p, l, r in zip(patches, bins[:-1], bins[1:]):
            center = (l + r) / 2
            if lcl is not None and ucl is not None:
                if center < lcl or center > ucl:
                    p.set_facecolor("#ff6b6b")

        # ===== Normal curve (long tail) =====
        if std > 0:
            x = np.linspace(mean - 4 * std, mean + 4 * std, 500)
            pdf = normal_pdf(x, mean, std)
            ax.plot(
                x,
                pdf * len(values) * (bins[1] - bins[0]),
                color="black",
                linewidth=2
            )

        # ===== USL / LSL =====
        if lcl is not None:
            ax.axvline(lcl, color="red", linestyle="--", linewidth=1.5, label="LSL")
        if ucl is not None:
            ax.axvline(ucl, color="red", linestyle="--", linewidth=1.5, label="USL")

        # ===== Capability box =====
        if cp is not None:
            ax.text(
                0.98, 0.95,
                f"Ca  = {ca}\nCp  = {cp}\nCpk = {cpk}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.9)
            )

        # ===== Info box =====
        ax.text(
            0.02, 0.95,
            f"N = {len(values)}\n"
            f"Mean = {mean:.3f}\n"
            f"Std = {std:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.9)
        )

        ax.set_title(f"{k} (LAB)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

        # ===== SHOW FIG =====
        st.pyplot(fig)

        # =========================
        # DOWNLOAD IMAGE
        # =========================
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            label="⬇ Download chart image",
            data=buf,
            file_name=f"{k}_lab_distribution.png",
            mime="image/png"
        )

        # =========================
        # BIN SUMMARY TABLE
        # =========================
        bin_edges = np.histogram_bin_edges(values, bins=10)
        counts, _ = np.histogram(values, bins=bin_edges)
        bin_width = bin_edges[1] - bin_edges[0]

        bin_df = pd.DataFrame({
            "Bin Range": [
                f"{bin_edges[j]:.3f} ~ {bin_edges[j+1]:.3f}"
                for j in range(len(bin_edges) - 1)
            ],
            "Count": counts,
            "Density": (counts / (len(values) * bin_width)).round(4)
        })

        with st.expander("📊 Distribution bin details"):
            st.dataframe(bin_df, use_container_width=True, hide_index=True)

# =========================
# 🚨 OUT-OF-CONTROL BATCH TABLE
# =========================
st.markdown("## 🚨 Out-of-Control Batches")

ooc_rows = []

for k in spc:
    # ===== LINE =====
    lcl, ucl = get_limit(color, k, "LINE")
    ooc_line = detect_out_of_control(spc[k]["line"], lcl, ucl)

    for _, r in ooc_line.iterrows():
        ooc_rows.append({
            "Factor": k,
            "Type": "LINE",
            "製造批號": r["製造批號"],
            "Value": round(r["value"], 2),
            "Rule_CL": r["Rule_CL"],
            "Rule_3Sigma": r["Rule_3Sigma"]
        })

    # ===== LAB =====
    lcl, ucl = get_limit(color, k, "LAB")
    ooc_lab = detect_out_of_control(spc[k]["lab"], lcl, ucl)

    for _, r in ooc_lab.iterrows():
        ooc_rows.append({
            "Factor": k,
            "Type": "LAB",
            "製造批號": r["製造批號"],
            "Value": round(r["value"], 2),
            "Rule_CL": r["Rule_CL"],
            "Rule_3Sigma": r["Rule_3Sigma"]
        })

if ooc_rows:
    ooc_df = pd.DataFrame(ooc_rows)
    st.dataframe(ooc_df, use_container_width=True)
else:
    st.success("✅ No out-of-control batches detected")



























