import streamlit as st
import pandas as pd
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
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(
        "https://docs.google.com/spreadsheets/d/1lqsLKSoDTbtvAsHzJaEri8tPo5pA3vqJ__LVHp2R534/export?format=csv"
    )
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_limit():
    return pd.read_csv(
        "https://docs.google.com/spreadsheets/d/1jbP8puBraQ5Xgs9oIpJ7PlLpjIK3sltrgbrgKUcJ-Qo/export?format=csv"
    )


df_all = load_data()
limit_df = load_limit()

# =========================
# CLEAN COLUMN NAMES
# =========================
df_all.columns = (
    df_all.columns
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
    sorted(df_all["塗料編號"].dropna().unique())
)

df_color = df_all[df_all["塗料編號"] == color]

year = st.sidebar.selectbox(
    "Year",
    sorted(df_color["Time"].dropna().dt.year.unique())
)

df = df_color[df_color["Time"].dt.year == year]

if df.empty:
    st.warning("No data available for this selection.")
    st.stop()

# =========================
# LIMIT FUNCTION
# =========================
def get_limit(color, factor):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty:
        return None, None
    lcl = row.get(f"{factor} LINE LCL", pd.Series([None])).iloc[0]
    ucl = row.get(f"{factor} LINE UCL", pd.Series([None])).iloc[0]
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
# SPC SUMMARY (FIXED)
# =========================
summary = []

for k in spc:
    line = spc[k]["line"]["value"].dropna()
    lab = spc[k]["lab"]["value"].dropna()

    lcl, ucl = get_limit(color, k)

    line_min = line.min() if not line.empty else None
    line_max = line.max() if not line.empty else None
    line_mean = line.mean() if not line.empty else None
    line_std = line.std() if not line.empty else None

    lab_min = lab.min() if not lab.empty else None
    lab_max = lab.max() if not lab.empty else None
    lab_mean = lab.mean() if not lab.empty else None
    lab_std = lab.std() if not lab.empty else None

    cp = cpk = ca = None
    if (
        line_std is not None
        and line_std > 0
        and lcl is not None
        and ucl is not None
    ):
        cp = (ucl - lcl) / (6 * line_std)
        cpk = min((ucl - line_mean), (line_mean - lcl)) / (3 * line_std)
        ca = abs(line_mean - (ucl + lcl) / 2) / ((ucl - lcl) / 2)

    summary.append({
        "Factor": k,

        "Line Min": round(line_min, 2) if line_min is not None else None,
        "Line Max": round(line_max, 2) if line_max is not None else None,
        "Line Mean": round(line_mean, 2) if line_mean is not None else None,
        "Line Std": round(line_std, 2) if line_std is not None else None,

        "LAB Min": round(lab_min, 2) if lab_min is not None else None,
        "LAB Max": round(lab_max, 2) if lab_max is not None else None,
        "LAB Mean": round(lab_mean, 2) if lab_mean is not None else None,
        "LAB Std": round(lab_std, 2) if lab_std is not None else None,

        "Ca (LINE)": round(ca, 2) if ca is not None else None,
        "Cp (LINE)": round(cp, 2) if cp is not None else None,
        "Cpk (LINE)": round(cpk, 2) if cpk is not None else None,
    })

summary_df = pd.DataFrame(summary)

# ⭐ QUAN TRỌNG: để LAB luôn hiển thị
summary_df = summary_df.fillna("-")

# =========================
# DISPLAY
# =========================
st.title(f"🎨 SPC Color Dashboard — {color}")

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
