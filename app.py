import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
import math
import re

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
        background: linear-gradient(270deg, #ffffff, #f0f9ff, #e0f2fe, #fef3c7, #ecfeff);
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

if st.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

st.markdown("""<style>[data-testid="stSidebar"] {background-color: #f6f8fa;}</style>""", unsafe_allow_html=True)

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

df_raw = load_data()
limit_df = load_limit()

# Làm sạch tên cột
df_raw.columns = df_raw.columns.str.replace("\r\n", " ", regex=False).str.replace("\n", " ", regex=False).str.replace("　", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
limit_df.columns = limit_df.columns.str.replace("\r\n", " ", regex=False).str.replace("\n", " ", regex=False).str.replace("　", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()

numeric_columns = [
    "入料檢測 ΔL 正面", "入料檢測 Δa 正面", "入料檢測 Δb 正面",
    "正-北 ΔL", "正-南 ΔL", "正-北 Δa", "正-南 Δa", "正-北 Δb", "正-南 Δb",
    "Avergage Thickness", "Average value ΔE 正面", "Average value ΔL 正面", 
    "Average value Δa 正面", "Average value Δb 正面"
]
for col in numeric_columns:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')


# =========================
# HELPER FUNCTIONS
# =========================
def safe_get_limit(c_code, src, fac):
    """Hàm dò tìm thông minh, chống lỗi khoảng trắng và thứ tự chữ."""
    try:
        c_search = str(c_code).strip().upper()
        match_idx = -1
        if "Color_code" in limit_df.columns:
            for idx, val in limit_df["Color_code"].items():
                if str(val).strip().upper() == c_search:
                    match_idx = idx
                    break
        if match_idx == -1: return None, None
        
        row = limit_df.iloc[[match_idx]]
        src_up = str(src).strip().upper()
        fac1 = str(fac).strip().upper()
        fac2 = fac1.replace("Δ", "DELTA ")
        
        lcl, ucl = None, None
        for col in row.columns:
            cup = str(col).strip().upper()
            if src_up in cup and (fac1 in cup or fac2 in cup):
                val = row[col].values[0]
                try:
                    num = float(val)
                    if not pd.isna(num):
                        if "LCL" in cup: lcl = num
                        if "UCL" in cup: ucl = num
                except: pass
        return lcl, ucl
    except:
        return None, None

def get_control_batch(color):
    c_search = str(color).strip().upper()
    mask = limit_df["Color_code"].astype(str).str.strip().str.upper() == c_search
    row = limit_df[mask]
    if row.empty: return None
    value = row["Control_batch"].values[0]
    if pd.isna(value): return None
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        if m: return int(m.group())
    try: return int(float(value))
    except: return None

def get_control_batch_code(df_unfiltered, control_batch):
    if control_batch is None or df_unfiltered.empty: return None
    batch_order = df_unfiltered.sort_values("Time").groupby("製造批號", as_index=False).first().reset_index(drop=True)
    if 1 <= control_batch <= len(batch_order): return batch_order.loc[control_batch - 1, "製造批號"]
    return None

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

def calculate_batch_averages(df_filtered_color):
    res = {}
    for f in ["ΔL", "Δa", "Δb"]:
        tmp = df_filtered_color.copy()
        # LINE
        col_n, col_s = f"正-北 {f}", f"正-南 {f}"
        tmp[col_n] = pd.to_numeric(tmp[col_n], errors='coerce')
        tmp[col_s] = pd.to_numeric(tmp[col_s], errors='coerce')
        tmp["row_avg"] = tmp[[col_n, col_s]].mean(axis=1)
        line_b = tmp.groupby("製造批號", as_index=False).agg({"Time": "min", "row_avg": "mean"}).rename(columns={"row_avg": "value"}).dropna()
        # LAB
        col_lab = f"入料檢測 {f} 正面"
        tmp[col_lab] = pd.to_numeric(tmp[col_lab], errors='coerce')
        lab_b = tmp.groupby("製造批號", as_index=False).agg({"Time": "min", col_lab: "mean"}).rename(columns={col_lab: "value"}).dropna()
        res[f] = {"line": line_b, "lab": lab_b}
    return res


# =========================
# SIDEBAR – NAVIGATION
# =========================
st.sidebar.markdown("### 📊 View Mode")
app_mode = st.sidebar.radio(
    "Select View Mode",
    ["🚀 Main Dashboard", "📋 Limit Status Summary", "🎛️ Control Limit Calculator"],
    label_visibility="collapsed"
)

st.sidebar.divider()
st.sidebar.title("🎨 Filter")
color = st.sidebar.selectbox("Color code", sorted(df_raw["塗料編號"].dropna().unique()), key="sidebar_color")

df_color = df_raw[df_raw["塗料編號"] == color].copy()
control_batch = get_control_batch(color)
control_batch_code = get_control_batch_code(df_color, control_batch)

all_years = sorted(df_color["Time"].dt.year.dropna().astype(int).unique())
selected_years = st.sidebar.multiselect("📅 Year (Leave empty for ALL)", options=all_years, default=[], key="sidebar_year")

all_months = sorted(df_color["Time"].dt.month.dropna().astype(int).unique())
selected_months = st.sidebar.multiselect("📅 Month (optional)", options=all_months, default=[], key="sidebar_month")

df = df_color.copy()
if len(selected_years) > 0: df = df[df["Time"].dt.year.isin(selected_years)]
if len(selected_months) > 0: df = df[df["Time"].dt.month.isin(selected_months)]

# Tính toán chuẩn SPC dùng chung
spc_data = calculate_batch_averages(df)


# =========================================================
# VIEW 1: MAIN DASHBOARD (BẢN FULL GỐC)
# =========================================================
if app_mode == "🚀 Main Dashboard":

    # --- 1. SIDEBAR ELEMENTS ---
    st.sidebar.divider()
    if control_batch_code is not None:
        st.sidebar.info(f"🔔 **Control batch**\n\nBatch #{control_batch} → **{control_batch_code}**")
    elif control_batch is not None:
        st.sidebar.warning(f"⚠ Control batch #{control_batch} exceeds available batches")
    st.sidebar.divider()

    # Bảng Sheet Limits ở thanh Sidebar
    def show_limits(factor):
        c_search = str(color).strip().upper()
        mask = limit_df["Color_code"].astype(str).str.strip().str.upper() == c_search
        row = limit_df[mask]
        if row.empty: return
        table = row.filter(like=factor).copy()
        for c in table.columns: table[c] = table[c].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
        st.sidebar.markdown(f"**{factor} Control Limits (Sheet)**")
        st.sidebar.dataframe(table, use_container_width=True, hide_index=True)

    show_limits("LAB")
    show_limits("LINE")

    # --- 2. MAIN TITLES & HEADER INFO ---
    st.title("室內隔間用途－塗料入料管控專案")
    st.caption("Incoming Paint SPC · LAB / LINE · Phase II Monitoring")
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

    # --- 3. COLLAPSIBLE BATCH SUMMARY ---
    with st.expander("🔎 Batch Summary (Before SPC Aggregation)"):
        if not df.empty:
            batch_summary = (
                df.groupby("製造批號", as_index=False)
                .agg(
                    First_Time=("Time", "min"),
                    LAB_ΔL=("入料檢測 ΔL 正面", "mean"), LAB_Δa=("入料檢測 Δa 正面", "mean"), LAB_Δb=("入料檢測 Δb 正面", "mean"),
                    LINE_ΔL_N=("正-北 ΔL", "mean"), LINE_ΔL_S=("正-南 ΔL", "mean"),
                    LINE_Δa_N=("正-北 Δa", "mean"), LINE_Δa_S=("正-南 Δa", "mean"),
                    LINE_Δb_N=("正-北 Δb", "mean"), LINE_Δb_S=("正-南 Δb", "mean"),
                    Rows_in_Batch=("製造批號", "count")
                ).sort_values("First_Time")
            )
            batch_summary["LINE_ΔL"] = batch_summary[["LINE_ΔL_N", "LINE_ΔL_S"]].mean(axis=1)
            batch_summary["LINE_Δa"] = batch_summary[["LINE_Δa_N", "LINE_Δa_S"]].mean(axis=1)
            batch_summary["LINE_Δb"] = batch_summary[["LINE_Δb_N", "LINE_Δb_S"]].mean(axis=1)
            display_cols = ["製造批號", "First_Time", "LAB_ΔL", "LAB_Δa", "LAB_Δb", "LINE_ΔL", "LINE_Δa", "LINE_Δb", "Rows_in_Batch"]
            st.dataframe(batch_summary[display_cols], use_container_width=True, hide_index=True)
        else:
            st.warning("No data after filtering.")

    # --- 4. SUMMARY STATISTICS (Phần tiếp theo của code) ---
    summary_line, summary_lab = [], []
    for k in ["ΔL", "Δa", "Δb"]:
        line_values = spc_data[k]["line"]["value"].dropna()
        if not line_values.empty:
            summary_line.append({"Factor": k, "Min": round(line_values.min(), 2), "Max": round(line_values.max(), 2), "Mean": round(line_values.mean(), 2), "Std Dev": round(line_values.std(), 2), "n": line_values.count()})
        lab_values = spc_data[k]["lab"]["value"].dropna()
        if not lab_values.empty:
            summary_lab.append({"Factor": k, "Min": round(lab_values.min(), 2), "Max": round(lab_values.max(), 2), "Mean": round(lab_values.mean(), 2), "Std Dev": round(lab_values.std(), 2), "n": lab_values.count()})
    st.markdown("### 📋 Summary Statistics")
    col1, col2 = st.columns(2)
    with col1: st.markdown("#### 🏭 LINE"); st.dataframe(pd.DataFrame(summary_line), use_container_width=True, hide_index=True)
    with col2: st.markdown("#### 🧪 LAB"); st.dataframe(pd.DataFrame(summary_lab), use_container_width=True, hide_index=True)

    # Các hàm vẽ Chart
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
        if lab_lim[0] is not None: ax.axhline(lab_lim[0], color="#1f77b4", linestyle=":", label="LAB LCL"); ax.axhline(lab_lim[1], color="#1f77b4", linestyle=":", label="LAB UCL")
        if line_lim[0] is not None: ax.axhline(line_lim[0], color="red", label="LINE LCL"); ax.axhline(line_lim[1], color="red", label="LINE UCL")
        ax.set_title(title); ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left"); ax.grid(True); ax.tick_params(axis="x", rotation=45); fig.subplots_adjust(right=0.78)
        return fig

    def spc_combined_phase2(lab, line, title, lab_lim, line_lim, control_batch_code):
        if control_batch_code is None: return None
        lab2 = lab[lab["製造批號"] >= control_batch_code]; line2 = line[line["製造批號"] >= control_batch_code]
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
        if lab_lim[0] is not None: ax.axhline(lab_lim[0], color="#1f77b4", linestyle=":", label="LAB LCL"); ax.axhline(lab_lim[1], color="#1f77b4", linestyle=":", label="LAB UCL")
        if line_lim[0] is not None: ax.axhline(line_lim[0], color="red", label="LINE LCL"); ax.axhline(line_lim[1], color="red", label="LINE UCL")
        ax.set_title(title); ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left"); ax.grid(True); ax.tick_params(axis="x", rotation=45); fig.subplots_adjust(right=0.78)
        return fig

    def download(fig, name):
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=200, bbox_inches="tight"); buf.seek(0)
        st.download_button("📥 Download PNG", buf, name, "image/png", key=f"dl_{name}")

    # Section: CONTROL CHART LAB-LINE
    st.markdown("### 📊 CONTROL CHART: LAB-LINE")
    for k in ["ΔL", "Δa", "Δb"]:
        lab_lim = safe_get_limit(color, "LAB", k)
        line_lim = safe_get_limit(color, "LINE", k)
        fig = spc_combined(spc_data[k]["lab"], spc_data[k]["line"], f"COMBINED {k}", lab_lim, line_lim, control_batch_code)
        st.pyplot(fig); download(fig, f"COMBINED_{color}_{k}.png")

    # Section: PHASE 2 CHARTS
    st.markdown("---")
    st.subheader("📊 SPC Combined Chart (LAB + LINE) – Phase II")
    for k in ["ΔL", "Δa", "Δb"]:
        lab_lim = safe_get_limit(color, "LAB", k)
        line_lim = safe_get_limit(color, "LINE", k)
        fig = spc_combined_phase2(spc_data[k]["lab"], spc_data[k]["line"], f"{k} – LAB + LINE (Phase II)", lab_lim, line_lim, control_batch_code)
        if fig is not None: st.pyplot(fig); download(fig, f"COMBINED_PHASE2_{color}_{k}.png")
        else: st.info(f"{k}: Not enough Phase II data")

    # Section: DISTRIBUTIONS DASHBOARD
    st.markdown("---")
    st.markdown("## 📈 Line Process Distribution Dashboard")
    def normal_pdf(x, mean, std): return (1 / (std * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    cols = st.columns(3)
    for i, k in enumerate(["ΔL", "Δa", "Δb"]):
        with cols[i]:
            values = spc_data[k]["line"]["value"].dropna()
            if len(values) < 3: st.warning("Not enough data"); continue
            mean, std = values.mean(), values.std(); lcl, ucl = safe_get_limit(color, "LINE", k)
            fig, ax = plt.subplots(figsize=(5, 4))
            bins = np.histogram_bin_edges(values, bins=10)
            counts, _, patches = ax.hist(values, bins=bins, edgecolor="white", color="#4dabf7", alpha=0.85)
            for p, l, r in zip(patches, bins[:-1], bins[1:]):
                center = (l + r) / 2
                if lcl is not None and ucl is not None and (center < lcl or center > ucl): p.set_facecolor("#ff6b6b")
            if std > 0:
                x = np.linspace(mean - 4 * std, mean + 4 * std, 500)
                ax.plot(x, normal_pdf(x, mean, std) * len(values) * (bins[1] - bins[0]), color="black", linewidth=2)
            if lcl is not None: ax.axvline(lcl, color="red", linestyle="--", linewidth=1.5, label="LSL")
            if ucl is not None: ax.axvline(ucl, color="red", linestyle="--", linewidth=1.5, label="USL")
            ax.text(0.02, 0.95, f"N = {len(values)}\nMean = {mean:.3f}\nStd = {std:.3f}", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.9))
            ax.set_title(f"{k} (LINE)"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8); st.pyplot(fig)
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight"); buf.seek(0)
            st.download_button("⬇ Download", data=buf, file_name=f"{k}_line_dist.png", mime="image/png", key=f"dl_line_dist_{k}")

    st.markdown("---")
    st.markdown("## 🧪 LAB Process Distribution Dashboard")
    cols = st.columns(3)
    for i, k in enumerate(["ΔL", "Δa", "Δb"]):
        with cols[i]:
            values = spc_data[k]["lab"]["value"].dropna()
            if len(values) < 3: st.warning("Not enough data"); continue
            mean, std = values.mean(), values.std(); lcl, ucl = safe_get_limit(color, "LAB", k)
            fig, ax = plt.subplots(figsize=(5, 4))
            bins = np.histogram_bin_edges(values, bins=10)
            counts, _, patches = ax.hist(values, bins=bins, edgecolor="white", color="#1f77b4", alpha=0.85)
            for p, l, r in zip(patches, bins[:-1], bins[1:]):
                center = (l + r) / 2
                if lcl is not None and ucl is not None and (center < lcl or center > ucl): p.set_facecolor("#ff6b6b")
            if std > 0:
                x = np.linspace(mean - 4 * std, mean + 4 * std, 500)
                ax.plot(x, normal_pdf(x, mean, std) * len(values) * (bins[1] - bins[0]), color="black", linewidth=2)
            if lcl is not None: ax.axvline(lcl, color="red", linestyle="--", linewidth=1.5, label="LSL")
            if ucl is not None: ax.axvline(ucl, color="red", linestyle="--", linewidth=1.5, label="USL")
            ax.text(0.02, 0.95, f"N = {len(values)}\nMean = {mean:.3f}\nStd = {std:.3f}", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.9))
            ax.set_title(f"{k} (LAB)"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8); st.pyplot(fig)
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight"); buf.seek(0)
            st.download_button("⬇ Download", data=buf, file_name=f"{k}_lab_dist.png", mime="image/png", key=f"dl_lab_dist_{k}")

    # Section: OOC TABLE
    st.markdown("## 🚨 Out-of-Control Batches")
    ooc_rows = []
    for k in ["ΔL", "Δa", "Δb"]:
        lcl, ucl = safe_get_limit(color, "LINE", k)
        if control_batch_code is not None:
            line_phase2 = spc_data[k]["line"][spc_data[k]["line"]["製造批號"] >= control_batch_code]
            ooc_line = detect_out_of_control(line_phase2, lcl, ucl)
            for _, r in ooc_line.iterrows(): ooc_rows.append({"Factor": k, "Type": "LINE", "製造批號": r["製造批號"], "Value": round(r["value"], 2), "Rule_CL": r["Rule_CL"], "Rule_3Sigma": r["Rule_3Sigma"]})
        lcl, ucl = safe_get_limit(color, "LAB", k)
        if control_batch_code is not None:
            lab_phase2 = spc_data[k]["lab"][spc_data[k]["lab"]["製造批號"] >= control_batch_code]
            ooc_lab = detect_out_of_control(lab_phase2, lcl, ucl)
            for _, r in ooc_lab.iterrows(): ooc_rows.append({"Factor": k, "Type": "LAB", "製造批號": r["製造批號"], "Value": round(r["value"], 2), "Rule_CL": r["Rule_CL"], "Rule_3Sigma": r["Rule_3Sigma"]})
    if ooc_rows: st.dataframe(pd.DataFrame(ooc_rows), use_container_width=True)
    else: st.success("✅ No out-of-control batches detected")

   # Section: THICKNESS CORRELATION
    st.markdown("---")
    st.header("🎨 Thickness – Color Analysis (Per Coil)")
    
    # Lưu ý: Giữ nguyên tên cột gốc trong DataFrame của bạn (kể cả có lỗi chính tả như "Avergage Thickness")
    coil_col, time_col, thickness_col = "Coil No.", "Time", "Avergage Thickness"
    dE_col, dL_col, da_col, db_col = "Average value ΔE 正面", "Average value ΔL 正面", "Average value Δa 正面", "Average value Δb 正面"
    required_cols = [coil_col, time_col, thickness_col, dE_col, dL_col, da_col, db_col]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"❌ Missing required columns: {missing}")
    else:
        df_plot = df.copy()
        df_plot[time_col] = pd.to_datetime(df_plot[time_col], errors="coerce")
        df_plot = df_plot.dropna(subset=[time_col])
        df_plot["Year"] = df_plot[time_col].dt.year
        df_plot["Month"] = df_plot[time_col].dt.to_period("M").astype(str)

        st.subheader("⏱ Time Filter")
        col1, col2 = st.columns(2)
        
        with col1: 
            filter_mode_bottom = st.radio("Filter by", ["Month", "Year"], horizontal=True, key="bottom_filter_mode")
        
        with col2:
            # --- CẬP NHẬT LOGIC: KHÔNG CHỌN = HIỂN THỊ TẤT CẢ ---
            if filter_mode_bottom == "Month":
                all_months = sorted(df_plot["Month"].unique())
                month_sel = st.multiselect("Select month(s) [Leave empty to show all]", all_months, default=[], key="bottom_month_sel")
                if month_sel:  # Chỉ lọc khi có tháng được chọn
                    df_plot = df_plot[df_plot["Month"].isin(month_sel)]
            else:
                all_years = sorted(df_plot["Year"].unique())
                year_sel = st.multiselect("Select year(s) [Leave empty to show all]", all_years, default=[], key="bottom_year_sel")
                if year_sel:   # Chỉ lọc khi có năm được chọn
                    df_plot = df_plot[df_plot["Year"].isin(year_sel)]

        if df_plot.empty:
            st.warning("⚠️ No data available for the selected time period.")
        else:
            st.subheader("📊 Average Thickness vs ΔE (Each Point = 1 Coil)")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_plot[thickness_col], df_plot[dE_col], alpha=0.75)
            if len(df_plot) > 0: 
                ax.axhline(df_plot[dE_col].mean(), linestyle="--", linewidth=2, label=f"Mean ΔE = {df_plot[dE_col].mean():.2f}")
            ax.set_xlabel("Average Thickness")
            ax.set_ylabel("ΔE")
            ax.set_title("Thickness – Color Relationship per Coil")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig)

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
                ax2.set_xlabel("ΔE")
                ax2.set_ylabel("Density")
                ax2.set_title("ΔE Distribution")
                ax2.legend()
                ax2.grid(True, linestyle="--", alpha=0.4)
                st.pyplot(fig2)

            st.subheader("📊 Average Thickness Distribution")
            data = df_plot[thickness_col].dropna()
            if len(data) > 0:
                mean, std = data.mean(), data.std()
                col1, col2 = st.columns(2)
                with col1: LSL = st.number_input("LSL", value=float(mean - 3 * std), key="bottom_lsl")
                with col2: USL = st.number_input("USL", value=float(mean + 3 * std), key="bottom_usl")
                
                if LSL >= USL: 
                    st.error("❌ LSL must be strictly smaller than USL")
                else:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(data, bins=20, density=True, alpha=0.7, edgecolor="black", label="Thickness Histogram")
                    x = np.linspace(mean - 5 * std, mean + 5 * std, 1000)
                    ax.plot(x, normal_pdf(x, mean, std), linewidth=3, label="Normal Distribution")
                    ax.axvline(mean, linestyle="--", linewidth=2, color="red", label=f"Mean = {mean:.2f}")
                    ax.axvline(LSL, linestyle="--", linewidth=2, color="green", label=f"LSL = {LSL:.2f}")
                    ax.axvline(USL, linestyle="--", linewidth=2, color="green", label=f"USL = {USL:.2f}")
                    ax.axvspan(LSL, USL, alpha=0.15, label="Spec Zone")
                    ax.set_xlim(mean - 5 * std, mean + 5 * std)
                    ax.set_xlabel("Average Thickness")
                    ax.set_ylabel("Density")
                    ax.grid(True, linestyle="--", alpha=0.4)
                    ax.legend()
                    st.pyplot(fig)

            with st.expander("📋 Coil Summary Data"):
                st.dataframe(df_plot[[coil_col, thickness_col, dE_col, dL_col, da_col, db_col, time_col]].sort_values(by=dE_col, ascending=False), use_container_width=True)

            st.markdown("---")
            st.header("🔬 PHASE II – THICKNESS CORRELATION")
            if control_batch_code is None:
                st.warning("⚠ Control batch not defined. Phase II cannot be determined.")
            else:
                df_p2 = df[(df["製造批號"] >= control_batch_code) & (df["塗料編號"] == color)].copy()
                if df_p2.empty:
                    st.warning("⚠ No Phase II data after filtering.")
                else:
                    COLOR_FACTORS = {"ΔL": ["入料檢測 ΔL 正面", "Average value ΔL 正面"], "Δa": ["入料檢測 Δa 正面", "Average value Δa 正面"], "Δb": ["入料檢測 Δb 正面", "Average value Δb 正面"], "ΔE": ["Average value ΔE 正面"]}
                    available_factors = {k: c for k, cols in COLOR_FACTORS.items() for c in cols if c in df_p2.columns}
                    
                    if not available_factors:
                        st.warning("⚠ No color factor columns found in dataset.")
                    else:
                        factor_label = st.selectbox("🎯 Select Color Factor", list(available_factors.keys()), index=0, key="bottom_color_factor")
                        factor_col = available_factors[factor_label]
                        coil_df = df_p2.groupby("Coil No.", as_index=False).agg({"Avergage Thickness": "mean", factor_col: "mean", "製造批號": "min"}).dropna()
                        
                        if coil_df.empty:
                            st.warning("⚠ No valid coil-level data.")
                        else:
                            lcl, ucl = safe_get_limit(color, "LINE", factor_label)
                            ooc_mask = (coil_df[factor_col] < lcl) | (coil_df[factor_col] > ucl) if lcl is not None and ucl is not None else np.zeros(len(coil_df), dtype=bool)
                            x, y = coil_df["Avergage Thickness"].values, coil_df[factor_col].values
                            r2 = None
                            
                            if len(x) >= 2:
                                slope, intercept = np.polyfit(x, y, 1)
                                r2 = 1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - np.mean(y)) ** 2) if np.sum((y - np.mean(y)) ** 2) != 0 else 0
                            
                            fig, ax = plt.subplots(figsize=(9, 6))
                            ax.scatter(coil_df[~ooc_mask]["Avergage Thickness"], coil_df[~ooc_mask][factor_col], alpha=0.7, label="Normal Coil")
                            if ooc_mask.any(): 
                                ax.scatter(coil_df[ooc_mask]["Avergage Thickness"], coil_df[ooc_mask][factor_col], color="red", s=80, label="OOC Coil")
                            if r2 is not None: 
                                ax.plot(np.linspace(x.min(), x.max(), 100), slope * np.linspace(x.min(), x.max(), 100) + intercept, linestyle="--", linewidth=2, label=f"Regression Line (R² = {r2:.3f})")
                            
                            ax.set_title(f"Phase II – Per Coil Analysis\nThickness vs {factor_label}" + (f" | r = {coil_df['Avergage Thickness'].corr(coil_df[factor_col]):.3f}, R² = {r2:.3f}" if r2 is not None else ""))
                            ax.set_xlabel("Average Thickness (per Coil)")
                            ax.set_ylabel(factor_label)
                            ax.legend()
                            ax.grid(True, linestyle="--", alpha=0.4)
                            st.pyplot(fig)

                            st.markdown("### 🧠 Interpretation")
                            # --- CORRELATION REFERENCE TABLE ---
                            st.markdown("""
                            **📊 Correlation Levels Reference:**
                            
                            | Level | Correlation Coefficient (\|R\|) | Coefficient of Determination (R²) | Interpretation |
                            | :--- | :--- | :--- | :--- |
                            | 🔴 **Strong** | \|R\| ≥ 0.77 | R² ≥ 0.60 | Thickness strongly affects and explains most of the color variation. |
                            | 🟠 **Moderate**| 0.55 ≤ \|R\| < 0.77 | 0.30 ≤ R² < 0.60 | Thickness partially contributes to color drift. |
                            | 🟢 **Weak/Low** | \|R\| < 0.55 | R² < 0.30 | Thickness is unlikely the main driver of color variation. |
                            """)
                            # -----------------------------------

                            if r2 is not None:
                                if r2 >= 0.6: 
                                    st.error("🔴 Thickness strongly explains color variation (High R²)")
                                elif r2 >= 0.3: 
                                    st.warning("🟠 Thickness may contribute to color drift (Moderate R²)")
                                else: 
                                    st.success("🟢 Thickness unlikely main driver (Low R²)")
                            else: 
                                st.info("ℹ Not enough data for regression analysis.")
                            # --- AUTOMATED RISK ALERT (OOC CLUSTERING) ---
                            if ooc_mask.any() and (~ooc_mask).any():
                                thick_col = "Avergage Thickness"
                                ooc_thick = coil_df[ooc_mask][thick_col]
                                norm_thick = coil_df[~ooc_mask][thick_col]
                                
                                mean_ooc = ooc_thick.mean()
                                q1_norm = norm_thick.quantile(0.25)
                                q3_norm = norm_thick.quantile(0.75)
                                
                                if mean_ooc > q3_norm:
                                    st.warning(f"🚨 **Automated Risk Alert:** OOC coils are noticeably clustered in the **HIGH** thickness zone (Mean OOC Thickness: {mean_ooc:.2f} > Normal Q3: {q3_norm:.2f}). Consider tightening the **Upper Specification Limit (USL)** for thickness to mitigate color drift risks.")
                                elif mean_ooc < q1_norm:
                                    st.warning(f"🚨 **Automated Risk Alert:** OOC coils are noticeably clustered in the **LOW** thickness zone (Mean OOC Thickness: {mean_ooc:.2f} < Normal Q1: {q1_norm:.2f}). Consider tightening the **Lower Specification Limit (LSL)** for thickness to mitigate color drift risks.")
                            # ---------------------------------------------
                            with st.expander("📋 Phase II – Coil Level Data"): 
                                st.dataframe(coil_df.sort_values("製造批號"), use_container_width=True)
# =========================================================
# VIEW 2: LIMIT STATUS SUMMARY
# =========================================================
elif app_mode == "📋 Limit Status Summary":
    st.title("📋 Limit Status Summary")
    st.markdown("Global overview of all color codes, identifying stable processes and those requiring control limit recalculation based on SPC rules.")

    st.markdown("### ⚙️ Alert Settings")
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        c_th = st.number_input("Consecutive OOC threshold (Rule 4):", min_value=1, max_value=10, value=2, step=1)
    with col_set2:
        t_th = st.number_input("Total (non-consecutive) OOC threshold:", min_value=1, max_value=20, value=5, step=1)

    all_colors = sorted(df_raw["塗料編號"].dropna().unique())
    summary_data = []

    def max_consecutive_true(s):
        if s.empty: return 0
        return (s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)).max()

    for c in all_colors:
        c_clean = str(c).strip()
        mask = limit_df["Color_code"].astype(str).str.strip() == c_clean
        row = limit_df[mask]
        
        status = "❌ No"
        if not row.empty:
            limit_cols = [col for col in row.columns if "LCL" in col or "UCL" in col]
            if not row[limit_cols].isna().all().all():
                status = "✅ Yes"

        df_c = df_raw[df_raw["塗料編號"] == c].sort_values("Time")
        total_batches = df_c["製造批號"].nunique()
        can_calc_initial = "✅ Yes" if total_batches >= 3 else "❌ No"
        
        cb = get_control_batch(c)
        cb_code = get_control_batch_code(df_c, cb)
        recalc_status = "❌ Not Enough Data"
        phase2_batches = 0
        
        if cb_code is not None:
            df_p2 = df_c[df_c["製造批號"] >= cb_code]
            phase2_batches = df_p2["製造批號"].nunique()
            
            if phase2_batches >= 3:
                if status == "✅ Yes":
                    max_consec_any_chart, max_total_any_chart = 0, 0
                    spc_p2 = calculate_batch_averages(df_p2)
                    
                    for f in ["ΔL", "Δa", "Δb"]:
                        for source in ["line", "lab"]:
                            lcl, ucl = safe_get_limit(c, source, f)
                            if lcl is not None and ucl is not None:
                                vals = spc_p2[f][source]["value"]
                                mask_val = (vals < lcl) | (vals > ucl)
                                consec, total_ooc = max_consecutive_true(mask_val), mask_val.sum()
                                if consec > max_consec_any_chart: max_consec_any_chart = consec
                                if total_ooc > max_total_any_chart: max_total_any_chart = total_ooc
                    
                    if max_consec_any_chart >= c_th: 
                        recalc_status = f"⚠️ Propose Recalc ({max_consec_any_chart} Consec. OOCs)"
                    elif max_total_any_chart >= t_th: 
                        recalc_status = f"⚠️ Propose Recalc ({max_total_any_chart} Total OOCs)"
                    else: 
                        recalc_status = f"✅ Stable (Max Consec: {max_consec_any_chart}, Total: {max_total_any_chart})"
                else: 
                    recalc_status = "❌ Missing Current Limits"
            
        summary_data.append({
            "Color Code": c, 
            "Total Batches": total_batches, 
            "Phase II Batches": phase2_batches,
            "Current Limits": status, 
            "Ready for Calc (Total)": can_calc_initial, 
            "Recommend Recalc (Phase II)": recalc_status
        })

    summary_df = pd.DataFrame(summary_data)
    total_c = len(summary_df)
    has_limit_c = len(summary_df[summary_df["Current Limits"] == "✅ Yes"])
    ready_initial_c = len(summary_df[summary_df["Ready for Calc (Total)"] == "✅ Yes"])
    needs_recalc_c = sum(1 for d in summary_data if "⚠️ Propose Recalc" in d["Recommend Recalc (Phase II)"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Colors", total_c)
    col2.metric("Colors Configured", has_limit_c)
    col3.metric("Ready to Calc (Initial)", ready_initial_c)
    col4.metric("Needs Recalculation", needs_recalc_c, delta="Process Shift Alert", delta_color="inverse")

    st.markdown("---")
    st.markdown("### 📊 Comprehensive Status Table")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # =========================================================
    # =========================================================
    # NEW SECTION: ACTION REQUIRED (MISSING LIMITS)
    # =========================================================
    st.markdown("---")
    st.markdown("### 🚨 Action Required: Missing Limits")
    st.markdown("The following colors do not have configured control limits but have enough data (≥ 3 batches). Please navigate to **🎛️ Control Limit Calculator** (View 3) to configure them.")
    
    # Lọc ra những màu chưa có Limit VÀ đã đủ dữ liệu để tính
    pending_colors = summary_df[
        (summary_df["Current Limits"] == "❌ No") & 
        (summary_df["Ready for Calc (Total)"] == "✅ Yes")
    ]
    
    if not pending_colors.empty:
        st.warning(f"Found **{len(pending_colors)}** color(s) waiting for limit calculation:")
        
        # Tạo cột tỷ lệ 1:2 để bảng chỉ chiếm 1/3 màn hình bên trái, nhìn sẽ cân đối và sang hơn
        col_table, col_empty = st.columns([1, 2])
        with col_table:
            st.dataframe(
                pending_colors[["Color Code", "Total Batches"]], 
                hide_index=True,           # Ẩn cột index 0, 1, 2...
                use_container_width=True   # Trải đều độ rộng lấp đầy không gian cột 1
            )
    else:
        st.success("🎉 All colors with sufficient data already have their control limits configured!")

# =========================================================
# VIEW 3: CONTROL LIMIT CALCULATOR
# =========================================================
elif app_mode == "🎛️ Control Limit Calculator":
    
    st.title("🎛️ Control Limits Analysis & Derived ΔE")
    
    with st.expander("⚙️ Data Source Settings", expanded=True):
        st.markdown("**Select Data Source:**")
        calc_source = st.radio("Data Source", ["LINE", "LAB"], horizontal=True)
        
    # Placeholder để đẩy bảng so sánh 2 phương pháp lên đầu trang
    result_placeholder = st.empty()
    st.markdown("---")

    factors = ["ΔL", "Δa", "Δb"]
    calc_res = {}
    
    # Khởi tạo 2 biến để tính tổng bình phương cho 2 phương pháp
    dE_std_sq, dE_iqr_sq = 0, 0

    for f in factors:
        st.markdown(f"### 📊 Analysis: **{f}** ({calc_source})")
        
        col_sig, col_iqr = st.columns(2)
        with col_sig:
            sig = st.number_input(f"🔸 Sigma (K) for {f}", value=3.0, step=0.1, key=f"sig_{f}")
        with col_iqr:
            iqr_k = st.number_input(f"🔸 IQR Sens. for {f}", value=1.5, step=0.1, key=f"iqr_{f}")
        
        d = spc_data[f][calc_source.lower()]["value"]
        
        if len(d) >= 3:
            m, s = d.mean(), d.std()
            q1, q3 = d.quantile(0.25), d.quantile(0.75)
            olcl, oucl = safe_get_limit(color, calc_source, f)
            
            std_lcl, std_ucl = m - sig*s, m + sig*s
            iqr_lcl, iqr_ucl = q1 - iqr_k*(q3-q1), q3 + iqr_k*(q3-q1)
            
            calc_res[f] = {
                "data": d, "batch": spc_data[f][calc_source.lower()]["製造批號"], "m": m, "s": s, "median": d.median(),
                "sig": sig, "iqr_k": iqr_k, "olcl": olcl, "oucl": oucl, "std_lcl": std_lcl, "std_ucl": std_ucl, "iqr_lcl": iqr_lcl, "iqr_ucl": iqr_ucl
            }
            
            # Tính max bình phương cho từng phương pháp độc lập
            dE_std_sq += max(abs(std_lcl), abs(std_ucl))**2
            dE_iqr_sq += max(abs(iqr_lcl), abs(iqr_ucl))**2

            col_chart, col_table = st.columns([2.2, 1])
            res = calc_res[f]
            
            with col_table:
                olcl_str = f"{res['olcl']:.3f}" if pd.notnull(res['olcl']) else "None"
                oucl_str = f"{res['oucl']:.3f}" if pd.notnull(res['oucl']) else "None"
                
                df_table = pd.DataFrame([
                    {"Method": "0. Spec (Sheet)", "Min": olcl_str, "Max": oucl_str, "Center": "-", "Note": "Current Target"},
                    {"Method": f"1. Standard ({res['sig']}σ)", "Min": f"{res['std_lcl']:.3f}", "Max": f"{res['std_ucl']:.3f}", "Center": f"{res['m']:.3f}", "Note": "Basic Stats"},
                    {"Method": f"2. IQR (k={res['iqr_k']})", "Min": f"{res['iqr_lcl']:.3f}", "Max": f"{res['iqr_ucl']:.3f}", "Center": f"{res['median']:.3f}", "Note": "Filtered"}
                ])
                st.dataframe(df_table, hide_index=True, use_container_width=True)
                st.info(f"**Stats:** μ={res['m']:.3f} | σ={res['s']:.3f} | n={len(res['data'])}")

            with col_chart:
                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax.plot(res["batch"], res["data"], "o-", color="#808080", alpha=0.5, label="Process Data")
                if pd.notnull(res["olcl"]):
                    ax.axhline(res["olcl"], color="black", linestyle="-", linewidth=1.5, label="0. Spec")
                    ax.axhline(res["oucl"], color="black", linestyle="-", linewidth=1.5)
                ax.axhline(res["std_lcl"], color="#d62728", linestyle="--", linewidth=1.5, label=f"1. Std (±{res['sig']}σ)")
                ax.axhline(res["std_ucl"], color="#d62728", linestyle="--", linewidth=1.5)
                ax.axhline(res["iqr_lcl"], color="#1f77b4", linestyle=":", linewidth=2, label=f"2. IQR (k={res['iqr_k']})")
                ax.axhline(res["iqr_ucl"], color="#1f77b4", linestyle=":", linewidth=2)
                ax.axhline(res["m"], color="#2ca02c", linestyle="-.", alpha=0.5, label="Mean")
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                fig.subplots_adjust(right=0.75, bottom=0.2)
                st.pyplot(fig)
                plt.close(fig)
            st.markdown("---")
        else:
            st.warning(f"Not enough data for {f} (min 3 batches).")

    # --- ĐIỀN KẾT QUẢ VÀO PLACEHOLDER Ở ĐẦU TRANG CHO CẢ 2 PHƯƠNG PHÁP ---
    if len(calc_res) == 3:
        # Căn bậc 2 để ra giá trị ΔE cuối cùng
        dE_std = math.sqrt(dE_std_sq)
        dE_iqr = math.sqrt(dE_iqr_sq)
        
        # Mốc đánh giá tùy theo LINE hay LAB
        limit_threshold = 1.0 if calc_source.upper() == "LINE" else 0.5
        
        with result_placeholder.container():
            st.markdown("### 🎯 Derived ΔE UCL Comparison")
            col_res1, col_res2 = st.columns(2)
            
            # Hiển thị kết quả Method 1 (Standard)
            with col_res1:
                if dE_std <= limit_threshold: 
                    st.success(f"**Method 1 (Standard)** ΔE UCL: **{dE_std:.3f}** (✅ ≤ {limit_threshold})")
                else: 
                    st.error(f"**Method 1 (Standard)** ΔE UCL: **{dE_std:.3f}** (⚠️ > {limit_threshold})")
                    
            # Hiển thị kết quả Method 2 (IQR)
            with col_res2:
                if dE_iqr <= limit_threshold: 
                    st.success(f"**Method 2 (IQR)** ΔE UCL: **{dE_iqr:.3f}** (✅ ≤ {limit_threshold})")
                else: 
                    st.error(f"**Method 2 (IQR)** ΔE UCL: **{dE_iqr:.3f}** (⚠️ > {limit_threshold})")

# =========================================================
    # MANUAL ΔE CALCULATOR (Bottom Section)
    # =========================================================
    st.markdown("---")
    st.subheader("🧮 Manual ΔE Calculator")
    st.markdown("Enter custom values for ΔL, Δa, and Δb to calculate the resulting overall color difference (ΔE).")
    
    # Tạo 3 cột để nhập liệu cho gọn gàng
    col_ml, col_ma, col_mb = st.columns(3)
    with col_ml:
        man_L = st.number_input("Input ΔL value:", value=0.000, step=0.100, format="%.3f")
    with col_ma:
        man_a = st.number_input("Input Δa value:", value=0.000, step=0.100, format="%.3f")
    with col_mb:
        man_b = st.number_input("Input Δb value:", value=0.000, step=0.100, format="%.3f")
        
    # Tính toán ΔE bằng công thức hình học không gian
    manual_dE = math.sqrt(man_L**2 + man_a**2 + man_b**2)
    
    # Lấy lại mốc limit_threshold đã xác định ở trên (dựa vào LINE hoặc LAB)
    limit_threshold = 1.0 if calc_source.upper() == "LINE" else 0.5
    
    # Hiển thị kết quả với cảnh báo trực quan
    st.markdown("#### **Calculation Result**")
    if manual_dE <= limit_threshold:
        st.success(f"### 🎯 Calculated ΔE: **{manual_dE:.3f}** (✅ Meets **{calc_source}** standard ≤ {limit_threshold})")
    else:
        st.error(f"### 🎯 Calculated ΔE: **{manual_dE:.3f}** (⚠️ Exceeds **{calc_source}** limit > {limit_threshold})")
    
    # Hiển thị công thức minh hoạ (Tùy chọn)
    st.latex(r"\Delta E = \sqrt{\Delta L^2 + \Delta a^2 + \Delta b^2}")













