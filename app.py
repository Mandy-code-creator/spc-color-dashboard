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
st.title("室內隔間用途－塗料入料管控專案")
st.caption("Incoming Paint SPC · LAB / LINE · Phase II Monitoring")

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
# FIX COLUMN NAMES & TYPES
# =========================
df.columns = (
    df.columns
    .str.replace("\r\n", " ", regex=False)
    .str.replace("\n", " ", regex=False)
    .str.replace("　", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

numeric_columns = [
    "入料檢測 ΔL 正面", "入料檢測 Δa 正面", "入料檢測 Δb 正面",
    "正-北 ΔL", "正-南 ΔL", "正-北 Δa", "正-南 Δa", "正-北 Δb", "正-南 Δb",
    "Avergage Thickness", "Average value ΔE 正面", "Average value ΔL 正面", 
    "Average value Δa 正面", "Average value Δb 正面"
]
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# =========================
# LIMIT FUNCTIONS
# =========================
def get_limit(color, prefix, factor):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty:
        return None, None
    return (
        row.get(f"{factor} {prefix} LCL", [None]).values[0],
        row.get(f"{factor} {prefix} UCL", [None]).values[0]
    )

def get_active_limit(color, prefix, factor, mode, series):
    """Lấy giới hạn cố định hoặc tự động tính toán (±3σ) dựa trên View Mode"""
    sheet_lcl, sheet_ucl = get_limit(color, prefix, factor)
    
    if mode == "Calculated (±3σ from data)":
        clean_series = series.dropna()
        if len(clean_series) >= 2:
            mean = clean_series.mean()
            std = clean_series.std()
            return (mean - 3 * std), (mean + 3 * std)
            
    return sheet_lcl, sheet_ucl

# =========================
# CONTROL BATCH FUNCTION  
# =========================
def get_control_batch(color):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty: return None
    value = row["Control_batch"].values[0]
    if pd.isna(value): return None
    if isinstance(value, str):
        import re
        m = re.search(r"\d+", value)
        if m: return int(m.group())
    try: return int(float(value))
    except: return None

def get_control_batch_code(df, control_batch):
    if control_batch is None or df.empty: return None
    batch_order = df.sort_values("Time").groupby("製造批號", as_index=False).first().reset_index(drop=True)
    if 1 <= control_batch <= len(batch_order):
        return batch_order.loc[control_batch - 1, "製造批號"]
    return None

# =========================
# SIDEBAR – FILTER & VIEW MODE
# =========================
st.sidebar.title("🎨 Filter & View")

color = st.sidebar.selectbox("Color code", sorted(df["塗料編號"].dropna().unique()), key="sidebar_color")
df = df[df["塗料編號"] == color]

# --- LỌC THỜI GIAN ---
all_years = sorted(df["Time"].dt.year.dropna().astype(int).unique())
selected_years = st.sidebar.multiselect("📅 Year (Leave empty for ALL)", options=all_years, default=[], key="sidebar_year")

all_months = sorted(df["Time"].dt.month.dropna().astype(int).unique())
selected_months = st.sidebar.multiselect("📅 Month (optional)", options=all_months, default=[], key="sidebar_month")

if len(selected_years) > 0: df = df[df["Time"].dt.year.isin(selected_years)]
if len(selected_months) > 0: df = df[df["Time"].dt.month.isin(selected_months)]

st.sidebar.divider()

# --- TÍNH NĂNG VIEW MODE MỚI ---
limit_mode = st.sidebar.radio(
    "⚙️ Control Limit Mode",
    ["Standard (Google Sheet)", "Calculated (±3σ from data)"],
    key="limit_mode",
    help="Chọn 'Calculated' để app tự động tính giới hạn LCL/UCL dựa trên ±3 độ lệch chuẩn của dữ liệu đang hiển thị."
)

st.sidebar.divider()

# =========================
# CONTROL BATCH INFO
# =========================
control_batch = get_control_batch(color)
control_batch_code = get_control_batch_code(df, control_batch)

if control_batch is not None and not df.empty:
    batch_order = df.sort_values("Time").groupby("製造批號", as_index=False).first().reset_index(drop=True)
    if 1 <= control_batch <= len(batch_order):
        control_batch_code = batch_order.loc[control_batch - 1, "製造批號"]
        st.sidebar.info(f"🔔 **Control batch**\n\nBatch #{control_batch} → **{control_batch_code}**")
    else:
        st.sidebar.warning(f"⚠ Control batch #{control_batch} vượt quá số batch hiện có")

st.sidebar.divider()

# =========================
# LIMIT DISPLAY
# =========================
def show_limits(factor):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty: return
    table = row.filter(like=factor).copy()
    for c in table.columns:
        table[c] = table[c].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
    st.sidebar.markdown(f"**{factor} Std. Limits (Sheet)**")
    st.sidebar.dataframe(table, use_container_width=True, hide_index=True)

show_limits("LAB")
show_limits("LINE")

if limit_mode == "Calculated (±3σ from data)":
    st.sidebar.info("💡 Biểu đồ đang sử dụng giới hạn ±3σ tự động tính toán (Không dùng bảng trên).")

# =========================
# OUT-OF-CONTROL DETECTION
# =========================
def detect_out_of_control(spc_df, lcl, ucl):
    mean = spc_df["value"].mean()
    std = spc_df["value"].std()
    result = spc_df.copy()
    result["Rule_CL"] = False
    result["Rule_3Sigma"] = False

    if lcl is not None and ucl is not None:
        result["Rule_CL"] = ((result["value"] < lcl) | (result["value"] > ucl))
    if std > 0:
        result["Rule_3Sigma"] = ((result["value"] > mean + 3 * std) | (result["value"] < mean - 3 * std))

    result["Out_of_Control"] = (result["Rule_CL"] | result["Rule_3Sigma"])
    return result[result["Out_of_Control"]]

# =========================
# PREP SPC DATA
# =========================
def prep_spc(df, north, south):
    tmp = df.copy()
    tmp[north] = pd.to_numeric(tmp[north], errors='coerce')
    tmp[south] = pd.to_numeric(tmp[south], errors='coerce')
    tmp["value"] = tmp[[north, south]].mean(axis=1)
    return tmp.groupby("製造批號", as_index=False).agg(Time=("Time", "min"), value=("value", "mean"))

def prep_lab(df, col):
    tmp = df.copy()
    tmp[col] = pd.to_numeric(tmp[col], errors='coerce')
    return tmp.groupby("製造批號", as_index=False).agg(Time=("Time", "min"), value=(col, "mean"))

spc = {
    "ΔL": {"lab": prep_lab(df, "入料檢測 ΔL 正面"), "line": prep_spc(df, "正-北 ΔL", "正-南 ΔL")},
    "Δa": {"lab": prep_lab(df, "入料檢測 Δa 正面"), "line": prep_spc(df, "正-北 Δa", "正-南 Δa")},
    "Δb": {"lab": prep_lab(df, "入料檢測 Δb 正面"), "line": prep_spc(df, "正-北 Δb", "正-南 Δb")}
}

# =========================
# MAIN DASHBOARD
# =========================
st.title(f"📊 SPC Color Dashboard — {color}")

if not df.empty:
    t_min = df["Time"].min().strftime("%Y-%m-%d")
    t_max = df["Time"].max().strftime("%Y-%m-%d")
    n_batch = df["製造批號"].nunique()
    display_year = "ALL" if len(selected_years) == 0 else ", ".join(map(str, selected_years))
    display_month = "ALL" if len(selected_months) == 0 else ", ".join(map(str, selected_months))
else:
    t_min = t_max = "N/A"; n_batch = 0; display_year = "N/A"; display_month = "N/A"

st.markdown(f"⏱ **{t_min} → {t_max} | n = {n_batch} batches | Year: {display_year} | Month: {display_month}**")

# ======================================================
# 📋 SUMMARY TABLE
# ======================================================
summary_line = []
summary_lab = []

for k in spc:
    line_values = spc[k]["line"]["value"].dropna()
    if not line_values.empty:
        summary_line.append({
            "Factor": k, "Min": round(line_values.min(), 2), "Max": round(line_values.max(), 2),
            "Mean": round(line_values.mean(), 2), "Std Dev": round(line_values.std(), 2), "n": line_values.count()
        })

    lab_values = spc[k]["lab"]["value"].dropna()
    if not lab_values.empty:
        summary_lab.append({
            "Factor": k, "Min": round(lab_values.min(), 2), "Max": round(lab_values.max(), 2),
            "Mean": round(lab_values.mean(), 2), "Std Dev": round(lab_values.std(), 2), "n": lab_values.count()
        })

summary_line_df = pd.DataFrame(summary_line)
summary_lab_df = pd.DataFrame(summary_lab)

st.markdown("### 📋 Summary Statistics")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🏭 LINE")
    st.dataframe(summary_line_df, use_container_width=True, hide_index=True)
with col2:
    st.markdown("#### 🧪 LAB")
    st.dataframe(summary_lab_df, use_container_width=True, hide_index=True)

# =========================
# SPC CHARTS
# =========================
def spc_combined(lab, line, title, lab_lim, line_lim, control_batch_code):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(lab["製造批號"], lab["value"], "o-", label="LAB", color="#1f77b4")
    ax.plot(line["製造批號"], line["value"], "o-", label="LINE", color="#2ca02c")
    
    if control_batch_code is not None:
        ax.axvline(x=control_batch_code, color="#b22222", linestyle="--", linewidth=1.5)
        ax.text(control_batch_code, ax.get_ylim()[1] * 0.97, "Phase II", color="#b22222", fontsize=9, ha="center", va="top")

    if lab_lim[0] is not None and lab_lim[1] is not None:
        out_lab = (lab["value"] > lab_lim[1]) | (lab["value"] < lab_lim[0])
        ax.scatter(lab["製造批號"][out_lab], lab["value"][out_lab], color="red", s=80, zorder=5)

    if line_lim[0] is not None and line_lim[1] is not None:
        out_line = (line["value"] > line_lim[1]) | (line["value"] < line_lim[0])
        ax.scatter(line["製造批號"][out_line], line["value"][out_line], color="red", s=80, zorder=5)
    
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

def spc_combined_phase2(lab, line, title, lab_lim, line_lim, control_batch_code):
    if control_batch_code is None: return None
    lab2 = lab[lab["製造批號"] >= control_batch_code]
    line2 = line[line["製造批號"] >= control_batch_code]
    if lab2.empty and line2.empty: return None

    fig, ax = plt.subplots(figsize=(12, 4))
    if not lab2.empty: ax.plot(lab2["製造批號"], lab2["value"], "o-", label="LAB", color="#1f77b4")
    if not line2.empty: ax.plot(line2["製造批號"], line2["value"], "o-", label="LINE", color="#2ca02c")

    if not lab2.empty and lab_lim[0] is not None:
        out = (lab2["value"] < lab_lim[0]) | (lab2["value"] > lab_lim[1])
        ax.scatter(lab2["製造批號"][out], lab2["value"][out], color="red", s=90, zorder=6)

    if not line2.empty and line_lim[0] is not None:
        out = (line2["value"] < line_lim[0]) | (line2["value"] > line_lim[1])
        ax.scatter(line2["製造批號"][out], line2["value"][out], color="red", s=90, zorder=6)

    ax.axvline(x=control_batch_code, color="#b22222", linestyle="--", linewidth=1.5, label="Phase II start")

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

def download(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.download_button("📥 Download PNG", buf, name, "image/png", key=f"dl_{name}")

st.markdown("### 📊 CONTROL CHART: LAB-LINE")
for k in spc:
    lab_lim = get_active_limit(color, "LAB", k, limit_mode, spc[k]["lab"]["value"])
    line_lim = get_active_limit(color, "LINE", k, limit_mode, spc[k]["line"]["value"])
    
    fig = spc_combined(spc[k]["lab"], spc[k]["line"], f"COMBINED {k}", lab_lim, line_lim, control_batch_code)
    st.pyplot(fig)
    download(fig, f"COMBINED_{color}_{k}.png")

st.markdown("---")
st.subheader("📊 SPC Combined Chart (LAB + LINE) – Phase II")
for k in ["ΔL", "Δa", "Δb"]:
    lab_s = spc[k]["lab"]
    line_s = spc[k]["line"]
    
    if control_batch_code is not None:
        lab2 = lab_s[lab_s["製造批號"] >= control_batch_code]
        line2 = line_s[line_s["製造批號"] >= control_batch_code]
        lab_lim = get_active_limit(color, "LAB", k, limit_mode, lab2["value"])
        line_lim = get_active_limit(color, "LINE", k, limit_mode, line2["value"])
    else:
        lab_lim = (None, None); line_lim = (None, None)

    fig = spc_combined_phase2(lab_s, line_s, f"{k} – LAB + LINE (Phase II)", lab_lim, line_lim, control_batch_code)
    if fig is not None:
        st.pyplot(fig)
        download(fig, f"COMBINED_PHASE2_{color}_{k}.png")
    else:
        st.info(f"{k}: Not enough Phase II data")

# =========================
# DISTRIBUTION DASHBOARD
# =========================
st.markdown("---")
st.markdown("## 📈 Line Process Distribution Dashboard")

def normal_pdf(x, mean, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

cols = st.columns(3)
for i, k in enumerate(spc):
    with cols[i]:
        values = spc[k]["line"]["value"].dropna()
        if len(values) < 3:
            st.warning("Not enough data")
            continue
        mean, std = values.mean(), values.std()
        lcl, ucl = get_active_limit(color, "LINE", k, limit_mode, values)

        fig, ax = plt.subplots(figsize=(5, 4))
        bins = np.histogram_bin_edges(values, bins=10)
        counts, _, patches = ax.hist(values, bins=bins, edgecolor="white", color="#4dabf7", alpha=0.85)

        for p, l, r in zip(patches, bins[:-1], bins[1:]):
            center = (l + r) / 2
            if lcl is not None and ucl is not None and (center < lcl or center > ucl):
                p.set_facecolor("#ff6b6b")

        if std > 0:
            x = np.linspace(mean - 4 * std, mean + 4 * std, 500)
            ax.plot(x, normal_pdf(x, mean, std) * len(values) * (bins[1] - bins[0]), color="black", linewidth=2)

        if lcl is not None: ax.axvline(lcl, color="red", linestyle="--", linewidth=1.5, label="LSL")
        if ucl is not None: ax.axvline(ucl, color="red", linestyle="--", linewidth=1.5, label="USL")

        ax.text(0.02, 0.95, f"N = {len(values)}\nMean = {mean:.3f}\nStd = {std:.3f}", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.9))
        ax.set_title(f"{k} (LINE)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        st.pyplot(fig)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        st.download_button("⬇ Download", data=buf, file_name=f"{k}_line_dist.png", mime="image/png", key=f"dl_line_dist_{k}")

st.markdown("---")
st.markdown("## 🧪 LAB Process Distribution Dashboard")
cols = st.columns(3)
for i, k in enumerate(spc):
    with cols[i]:
        values = spc[k]["lab"]["value"].dropna()
        if len(values) < 3:
            st.warning("Not enough data")
            continue
        mean, std = values.mean(), values.std()
        lcl, ucl = get_active_limit(color, "LAB", k, limit_mode, values)

        fig, ax = plt.subplots(figsize=(5, 4))
        bins = np.histogram_bin_edges(values, bins=10)
        counts, _, patches = ax.hist(values, bins=bins, edgecolor="white", color="#1f77b4", alpha=0.85)

        for p, l, r in zip(patches, bins[:-1], bins[1:]):
            center = (l + r) / 2
            if lcl is not None and ucl is not None and (center < lcl or center > ucl):
                p.set_facecolor("#ff6b6b")

        if std > 0:
            x = np.linspace(mean - 4 * std, mean + 4 * std, 500)
            ax.plot(x, normal_pdf(x, mean, std) * len(values) * (bins[1] - bins[0]), color="black", linewidth=2)

        if lcl is not None: ax.axvline(lcl, color="red", linestyle="--", linewidth=1.5, label="LSL")
        if ucl is not None: ax.axvline(ucl, color="red", linestyle="--", linewidth=1.5, label="USL")

        ax.text(0.02, 0.95, f"N = {len(values)}\nMean = {mean:.3f}\nStd = {std:.3f}", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.9))
        ax.set_title(f"{k} (LAB)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        st.download_button("⬇ Download", data=buf, file_name=f"{k}_lab_dist.png", mime="image/png", key=f"dl_lab_dist_{k}")

# =========================
# 🚨 OUT-OF-CONTROL BATCH TABLE
# =========================
st.markdown("## 🚨 Out-of-Control Batches")
ooc_rows = []
for k in spc:
    if control_batch_code is not None:
        line_phase2 = spc[k]["line"][spc[k]["line"]["製造批號"] >= control_batch_code]
        lcl, ucl = get_active_limit(color, "LINE", k, limit_mode, line_phase2["value"])
        ooc_line = detect_out_of_control(line_phase2, lcl, ucl)
        for _, r in ooc_line.iterrows():
            ooc_rows.append({"Factor": k, "Type": "LINE", "製造批號": r["製造批號"], "Value": round(r["value"], 2), "Rule_CL": r["Rule_CL"], "Rule_3Sigma": r["Rule_3Sigma"]})

    if control_batch_code is not None:
        lab_phase2 = spc[k]["lab"][spc[k]["lab"]["製造批號"] >= control_batch_code]
        lcl, ucl = get_active_limit(color, "LAB", k, limit_mode, lab_phase2["value"])
        ooc_lab = detect_out_of_control(lab_phase2, lcl, ucl)
        for _, r in ooc_lab.iterrows():
            ooc_rows.append({"Factor": k, "Type": "LAB", "製造批號": r["製造批號"], "Value": round(r["value"], 2), "Rule_CL": r["Rule_CL"], "Rule_3Sigma": r["Rule_3Sigma"]})

if ooc_rows:
    st.dataframe(pd.DataFrame(ooc_rows), use_container_width=True)
else:
    st.success("✅ No out-of-control batches detected")


# =========================================================
# 🎨 THICKNESS - COLOR ANALYSIS
# =========================================================
st.markdown("---")
st.header("🎨 Thickness – Color Analysis (Per Coil)")

coil_col, time_col, thickness_col = "Coil No.", "Time", "Avergage Thickness"
dE_col, dL_col, da_col, db_col = "Average value ΔE 正面", "Average value ΔL 正面", "Average value Δa 正面", "Average value Δb 正面"

required_cols = [coil_col, time_col, thickness_col, dE_col, dL_col, da_col, db_col]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"❌ Missing required columns: {missing}")
    st.stop()

df_plot = df.copy()
df_plot[time_col] = pd.to_datetime(df_plot[time_col], errors="coerce")
df_plot = df_plot.dropna(subset=[time_col])
df_plot["Year"] = df_plot[time_col].dt.year
df_plot["Month"] = df_plot[time_col].dt.to_period("M").astype(str)

st.subheader("⏱ Time Filter")
col1, col2 = st.columns(2)
with col1:
    filter_mode = st.radio("Filter by", ["Month", "Year"], horizontal=True, key="bottom_filter_mode")

with col2:
    if filter_mode == "Month":
        month_sel = st.multiselect("Select month(s)", sorted(df_plot["Month"].unique()), default=[sorted(df_plot["Month"].unique())[-1]], key="bottom_month_sel")
        df_plot = df_plot[df_plot["Month"].isin(month_sel)]
    else:
        year_sel = st.multiselect("Select year(s)", sorted(df_plot["Year"].unique()), default=[df_plot["Year"].max()], key="bottom_year_sel")
        df_plot = df_plot[df_plot["Year"].isin(year_sel)]

if df_plot.empty:
    st.warning("⚠️ No data after time filtering")
    st.stop()

st.subheader("📊 Average Thickness vs ΔE (Each Point = 1 Coil)")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df_plot[thickness_col], df_plot[dE_col], alpha=0.75)
if len(df_plot) > 0:
    ax.axhline(df_plot[dE_col].mean(), linestyle="--", linewidth=2, label=f"Mean ΔE = {df_plot[dE_col].mean():.2f}")
ax.set_xlabel("Average Thickness"); ax.set_ylabel("ΔE"); ax.set_title("Thickness – Color Relationship per Coil")
ax.legend(); ax.grid(True, linestyle="--", alpha=0.4); st.pyplot(fig)

st.subheader("📈 ΔE Distribution (Per Coil)")
fig2, ax2 = plt.subplots(figsize=(10, 4))
data_de = df_plot[dE_col].dropna()
if len(data_de) > 0:
    mean_de, std_de = data_de.mean(), data_de.std()
    ax2.hist(data_de, bins=20, density=True, alpha=0.7, edgecolor="black", label="ΔE Histogram")
    if std_de > 0:
        x_de = np.linspace(mean_de - 5*std_de, mean_de + 5*std_de, 1000)
        ax2.plot(x_de, normal_pdf(x_de, mean_de, std_de), linewidth=3, label="Normal Distribution")
    ax2.axvline(mean_de, linestyle="--", linewidth=2, label=f"Mean = {mean_de:.2f}")
    ax2.set_xlabel("ΔE"); ax2.set_ylabel("Density"); ax2.set_title("ΔE Distribution")
    ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.4); st.pyplot(fig2)

st.subheader("📊 Average Thickness Distribution")
data = df_plot[thickness_col].dropna()
if len(data) > 0:
    mean, std = data.mean(), data.std()
    col1, col2 = st.columns(2)
    with col1: LSL = st.number_input("LSL", value=float(mean - 3 * std), key="bottom_lsl")
    with col2: USL = st.number_input("USL", value=float(mean + 3 * std), key="bottom_usl")
    if LSL >= USL: st.error("❌ LSL must be smaller than USL"); st.stop()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(data, bins=20, density=True, alpha=0.7, edgecolor="black", label="Thickness")
    x = np.linspace(mean - 5 * std, mean + 5 * std, 1000)
    ax.plot(x, normal_pdf(x, mean, std), linewidth=3, label="Normal Distribution")
    ax.axvline(mean, linestyle="--", linewidth=2, color="red", label=f"Mean = {mean:.2f}")
    ax.axvline(LSL, linestyle="--", linewidth=2, color="green", label=f"LSL = {LSL:.2f}")
    ax.axvline(USL, linestyle="--", linewidth=2, color="green", label=f"USL = {USL:.2f}")
    ax.axvspan(LSL, USL, alpha=0.15, label="Spec Zone")
    ax.set_xlim(mean - 5 * std, mean + 5 * std)
    ax.set_xlabel("Average Thickness"); ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.4); ax.legend(); st.pyplot(fig)

st.subheader("📋 Coil Summary")
st.dataframe(df_plot[[coil_col, thickness_col, dE_col, dL_col, da_col, db_col, time_col]].sort_values(by=dE_col, ascending=False), use_container_width=True)

# ==========================================================
# 🔬 PHASE II – THICKNESS CORRELATION
# ==========================================================
st.markdown("---")
st.header("🔬 PHASE II – THICKNESS CORRELATION (INDEPENDENT MODULE) (Phase II – Per Coil)")

if control_batch_code is None:
    st.warning("⚠ Control batch not defined. Phase II cannot be determined.")
    st.stop()

df_p2 = df[(df["製造批號"] >= control_batch_code) & (df["塗料編號"] == color)].copy()
if df_p2.empty:
    st.warning("⚠ No Phase II data after filtering")
    st.stop()

COLOR_FACTORS = {"ΔL": ["入料檢測 ΔL 正面", "Average value ΔL 正面"], "Δa": ["入料檢測 Δa 正面", "Average value Δa 正面"], "Δb": ["入料檢測 Δb 正面", "Average value Δb 正面"], "ΔE": ["Average value ΔE 正面"]}
available_factors = {k: c for k, cols in COLOR_FACTORS.items() for c in cols if c in df_p2.columns}

if not available_factors:
    st.warning("⚠ No color factor columns found")
    st.stop()

factor_label = st.selectbox("🎯 Select Color Factor", list(available_factors.keys()), index=0, key="bottom_color_factor")
factor_col = available_factors[factor_label]

coil_df = df_p2.groupby("Coil No.", as_index=False).agg({"Avergage Thickness": "mean", factor_col: "mean", "製造批號": "min"}).dropna()
if coil_df.empty:
    st.warning("⚠ No valid coil-level data")
    st.stop()

# Lấy giới hạn từ View Mode cho phần Correlation
lcl, ucl = get_active_limit(color, "LINE", factor_label, limit_mode, coil_df[factor_col])

ooc_mask = (coil_df[factor_col] < lcl) | (coil_df[factor_col] > ucl) if lcl is not None and ucl is not None else np.zeros(len(coil_df), dtype=bool)

x, y = coil_df["Avergage Thickness"].values, coil_df[factor_col].values
r2 = None
if len(x) >= 2:
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - np.mean(y)) ** 2) if np.sum((y - np.mean(y)) ** 2) != 0 else 0

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(coil_df[~ooc_mask]["Avergage Thickness"], coil_df[~ooc_mask][factor_col], alpha=0.7, label="Normal Coil")
if ooc_mask.any(): ax.scatter(coil_df[ooc_mask]["Avergage Thickness"], coil_df[ooc_mask][factor_col], color="red", s=80, label="OOC Coil")
if r2 is not None: ax.plot(np.linspace(x.min(), x.max(), 100), slope * np.linspace(x.min(), x.max(), 100) + intercept, linestyle="--", linewidth=2, label=f"Regression line (R² = {r2:.3f})")

ax.set_title(f"Phase II – Per Coil Analysis\nThickness vs {factor_label}" + (f" | r = {coil_df['Avergage Thickness'].corr(coil_df[factor_col]):.3f}, R² = {r2:.3f}" if r2 is not None else ""))
ax.set_xlabel("Average Thickness (per Coil)"); ax.set_ylabel(factor_label); ax.legend(); ax.grid(True, linestyle="--", alpha=0.4); st.pyplot(fig)

st.markdown("### 🧠 Interpretation")
if r2 is not None:
    if r2 >= 0.6: st.error("🔴 Thickness strongly explains color variation (High R²)")
    elif r2 >= 0.3: st.warning("🟠 Thickness may contribute to color drift (Moderate R²)")
    else: st.success("🟢 Thickness unlikely main driver (Low R²)")
else: st.info("ℹ Not enough data for regression analysis")

with st.expander("📋 Phase II – Coil level data"): st.dataframe(coil_df.sort_values("製造批號"), use_container_width=True)

st.markdown("### 📐 Correlation Criteria (|R|)")
st.dataframe(pd.DataFrame({"|R| Range": ["≥ 0.70", "0.40 – 0.69", "0.20 – 0.39", "< 0.20"], "Strength": ["Strong", "Moderate", "Weak", "Negligible"], "Interpretation": ["Likely primary driver", "Contributing factor", "Minor influence", "No practical relationship"]}), use_container_width=True)
