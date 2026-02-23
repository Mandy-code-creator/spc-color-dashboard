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

# =========================
# REFRESH BUTTON
# =========================
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
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    return df

@st.cache_data(ttl=300)
def load_limit():
    df = pd.read_csv(LIMIT_URL)
    df.columns = df.columns.str.strip()
    return df

df_raw = load_data()
limit_df = load_limit()

# =========================
# HELPER FUNCTIONS
# =========================
def get_limit(color, prefix, factor):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty: return None, None
    col_lcl = f"{factor} {prefix} LCL"
    col_ucl = f"{factor} {prefix} UCL"
    lcl = row[col_lcl].values[0] if col_lcl in row.columns else None
    ucl = row[col_ucl].values[0] if col_ucl in row.columns else None
    return lcl, ucl

def get_control_batch(color):
    row = limit_df[limit_df["Color_code"] == color]
    if row.empty: return None
    value = row["Control_batch"].values[0]
    if pd.isna(value): return None
    try: return int(float(re.search(r"\d+", str(value)).group()))
    except: return None

def get_control_batch_code(df_unfiltered, control_batch):
    if control_batch is None or df_unfiltered.empty: return None
    batch_order = df_unfiltered.sort_values("Time").groupby("製造批號", as_index=False).first().reset_index(drop=True)
    if 1 <= control_batch <= len(batch_order): return batch_order.loc[control_batch - 1, "製造批號"]
    return None

def detect_out_of_control(spc_df, lcl, ucl):
    mean, std = spc_df["value"].mean(), spc_df["value"].std()
    res = spc_df.copy()
    res["Rule_CL"] = ((res["value"] < lcl) | (res["value"] > ucl)) if lcl is not None and ucl is not None else False
    res["Rule_3Sigma"] = ((res["value"] > mean + 3*std) | (res["value"] < mean - 3*std)) if std > 0 else False
    res["Out_of_Control"] = (res["Rule_CL"] | res["Rule_3Sigma"])
    return res[res["Out_of_Control"]]

# HÀM TÍNH TRUNG BÌNH LÔ CHUẨN
def calculate_batch_averages(df_filtered_color):
    res = {}
    for f in ["ΔL", "Δa", "Δb"]:
        tmp = df_filtered_color.copy()
        
        # LINE: Tính trung bình Bắc/Nam của từng dòng, sau đó trung bình theo lô
        col_n, col_s = f"正-北 {f}", f"正-南 {f}"
        tmp[col_n] = pd.to_numeric(tmp[col_n], errors='coerce')
        tmp[col_s] = pd.to_numeric(tmp[col_s], errors='coerce')
        tmp["row_avg"] = tmp[[col_n, col_s]].mean(axis=1)
        line_b = tmp.groupby("製造批號", as_index=False).agg({"Time": "min", "row_avg": "mean"}).rename(columns={"row_avg": "value"}).dropna()
        
        # LAB: Tính trung bình Lab theo lô
        col_lab = f"入料檢測 {f} 正面"
        tmp[col_lab] = pd.to_numeric(tmp[col_lab], errors='coerce')
        lab_b = tmp.groupby("製造批號", as_index=False).agg({"Time": "min", col_lab: "mean"}).rename(columns={col_lab: "value"}).dropna()
        
        res[f] = {"line": line_b, "lab": lab_b}
    return res

# =========================
# SIDEBAR – NAVIGATION & FILTERS
# =========================
st.sidebar.markdown("### 📊 View Mode")
app_mode = st.sidebar.radio(
    "Select View Mode",
    ["🚀 Main Dashboard", "📋 Limit Status Summary", "🎛️ Control Limit Calculator"],
    label_visibility="collapsed"
)

st.sidebar.divider()
color = st.sidebar.selectbox("Color code", sorted(df_raw["塗料編號"].dropna().unique()), key="sidebar_color")

df_color = df_raw[df_raw["塗料編號"] == color].copy()
control_batch = get_control_batch(color)
control_batch_code = get_control_batch_code(df_color, control_batch)

all_years = sorted(df_color["Time"].dt.year.dropna().astype(int).unique())
sel_years = st.sidebar.multiselect("📅 Year", options=all_years, default=[])
all_months = sorted(df_color["Time"].dt.month.dropna().astype(int).unique())
sel_months = st.sidebar.multiselect("📅 Month", options=all_months, default=[])

df_filtered = df_color.copy()
if sel_years: df_filtered = df_filtered[df_filtered["Time"].dt.year.isin(sel_years)]
if sel_months: df_filtered = df_filtered[df_filtered["Time"].dt.month.isin(sel_months)]

# DỮ LIỆU SPC DÙNG CHUNG CHO TẤT CẢ CÁC VIEW
spc_data = calculate_batch_averages(df_filtered)

# =========================================================
# VIEW 1: MAIN DASHBOARD
# =========================================================
if app_mode == "🚀 Main Dashboard":
    st.title(f"📊 SPC Color Dashboard — {color}")
    
    # Validation Table
    st.markdown("### 🔎 Batch Summary (Before SPC Aggregation)")
    if not df_filtered.empty:
        batch_summary = (
            df_filtered.groupby("製造批號", as_index=False)
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
        st.dataframe(batch_summary[["製造批號", "First_Time", "LAB_ΔL", "LAB_Δa", "LAB_Δb", "LINE_ΔL", "LINE_Δa", "LINE_Δb", "Rows_in_Batch"]], use_container_width=True, hide_index=True)
    
    st.markdown("---")
    if control_batch_code: st.sidebar.info(f"🔔 Control batch: #{control_batch} → {control_batch_code}")

    # Statistics
    sum_line, sum_lab = [], []
    for k in ["ΔL", "Δa", "Δb"]:
        vl = spc_data[k]["line"]["value"]
        if not vl.empty: sum_line.append({"Factor": k, "Min": round(vl.min(), 3), "Max": round(vl.max(), 3), "Mean": round(vl.mean(), 3), "Std": round(vl.std(), 3), "n": len(vl)})
        vb = spc_data[k]["lab"]["value"]
        if not vb.empty: sum_lab.append({"Factor": k, "Min": round(vb.min(), 3), "Max": round(vb.max(), 3), "Mean": round(vb.mean(), 3), "Std": round(vb.std(), 3), "n": len(vb)})
    
    c1, c2 = st.columns(2)
    with c1: st.markdown("#### 🏭 LINE"); st.dataframe(pd.DataFrame(sum_line), hide_index=True)
    with c2: st.markdown("#### 🧪 LAB"); st.dataframe(pd.DataFrame(sum_lab), hide_index=True)

    # Charts
    for k in ["ΔL", "Δa", "Δb"]:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(spc_data[k]["lab"]["製造批號"], spc_data[k]["lab"]["value"], "o-", label="LAB", color="#1f77b4")
        ax.plot(spc_data[k]["line"]["製造批號"], spc_data[k]["line"]["value"], "o-", label="LINE", color="#2ca02c")
        
        lcl, ucl = get_limit(color, "LINE", k)
        if lcl is not None: ax.axhline(lcl, color="red", linestyle="--"); ax.axhline(ucl, color="red", linestyle="--")
        if control_batch_code: ax.axvline(control_batch_code, color="brown", linestyle=":", label="Phase II")
        
        ax.set_title(f"{k} Trend Analysis"); ax.legend(loc="upper left"); ax.grid(True, alpha=0.3); plt.xticks(rotation=45); st.pyplot(fig)


# =========================================================
# VIEW 2: LIMIT STATUS SUMMARY
# =========================================================
elif app_mode == "📋 Limit Status Summary":
    st.title("📋 Limit Status Summary")
    st.markdown("Global overview of all color codes, identifying stable processes and those requiring control limit recalculation.")

    col_set1, col_set2 = st.columns(2)
    with col_set1: c_th = st.number_input("Consecutive OOC threshold (Rule 4):", 1, 10, 2)
    with col_set2: t_th = st.number_input("Total OOC threshold:", 1, 20, 5)

    def max_consecutive_true(s):
        return (s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)).max() if not s.empty else 0

    summary_data = []
    for c in sorted(df_raw["塗料編號"].dropna().unique()):
        row = limit_df[limit_df["Color_code"] == c]
        status = "✅ Yes" if not row.empty and not row.filter(like="LCL").isna().all().all() else "❌ No"
        df_c = df_raw[df_raw["塗料編號"] == c].sort_values("Time")
        cb_code = get_control_batch_code(df_c, get_control_batch(c))
        recalc = "❌ Insufficient Data"
        
        if cb_code and status == "✅ Yes":
            df_p2 = df_c[df_c["製造批號"] >= cb_code]
            if df_p2["製造批號"].nunique() >= 3:
                m_con, m_tot = 0, 0
                p2_data = calculate_batch_averages(df_p2)
                for f in ["ΔL", "Δa", "Δb"]:
                    for src in ["LINE", "LAB"]:
                        lcl, ucl = get_limit(c, src, f)
                        if lcl is not None:
                            mask = (p2_data[f][src.lower()]["value"] < lcl) | (p2_data[f][src.lower()]["value"] > ucl)
                            m_con, m_tot = max(m_con, max_consecutive_true(mask)), max(m_tot, mask.sum())
                recalc = "⚠️ Propose Recalc" if (m_con >= c_th or m_tot >= t_th) else "✅ Stable"
        
        summary_data.append({"Color Code": c, "Total Batches": df_c["製造批號"].nunique(), "Limits Config": status, "Recalc Status": recalc})

    summary_df = pd.DataFrame(summary_data)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Colors", len(summary_df))
    c2.metric("Configured", len(summary_df[summary_df["Limits Config"] == "✅ Yes"]))
    c4.metric("Needs Recalc", sum(1 for d in summary_data if "⚠️" in d["Recalc Status"]), delta="Shift detected", delta_color="inverse")
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# =========================================================
# VIEW 3: LIMIT SIMULATOR
# =========================================================
elif app_mode == "🎛️ Control Limit Calculator":
    st.title("🎛️ Control Limit Calculator & Derived ΔE")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        calc_source = st.radio("Data Source", ["LINE", "LAB"], horizontal=True)
        sig = float(st.text_input("Sigma (σ)", "3.0"))
        iqr = float(st.text_input("IQR (k)", "1.5"))
        method = st.radio("Method:", ["Std Dev (σ)", "IQR"], horizontal=True)
        
        calc_res, dE_max_sq, valid_old = {}, 0, True
        factors = ["ΔL", "Δa", "Δb"]
        for f in factors:
            d = spc_data[f][calc_source.lower()]["value"]
            if len(d) >= 3:
                m, s = d.mean(), d.std()
                q1, q3 = d.quantile(0.25), d.quantile(0.75)
                olcl, oucl = get_limit(color, calc_source, f)
                
                clcl, cucl = (m - sig*s, m + sig*s) if method == "Std Dev (σ)" else (q1 - iqr*(q3-q1), q3 + iqr*(q3-q1))
                cent = m if method == "Std Dev (σ)" else d.median()
                
                calc_res[f] = {"data": d, "b": spc_data[f][calc_source.lower()]["製造批號"], "olcl": olcl, "oucl": oucl, "clcl": clcl, "cucl": cucl, "cent": cent}
                dE_max_sq += max(abs(clcl), abs(cucl))**2
                if pd.notnull(olcl) and pd.notnull(oucl): valid_old &= True
                else: valid_old = False
                
        if len(calc_res) == 3:
            st.markdown("**Limits Table**")
            rows = [{"Factor": f, "Sheet LCL": calc_res[f]["olcl"], "Sheet UCL": calc_res[f]["oucl"], "Calc LCL": calc_res[f]["clcl"], "Calc Center": calc_res[f]["cent"], "Calc UCL": calc_res[f]["cucl"]} for f in factors]
            rows.append({"Factor": "Derived ΔE", "Sheet LCL": "-", "Sheet UCL": "-", "Calc LCL": 0, "Calc Center": "-", "Calc UCL": math.sqrt(dE_max_sq)})
            st.dataframe(pd.DataFrame(rows).style.format(precision=3, na_rep="-"), hide_index=True)
            
            sel_f = st.selectbox("Visualize:", factors)
        else:
            st.warning("Not enough data to calculate limits.")

    with c2:
        if len(calc_res) == 3:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(calc_res[sel_f]["b"], calc_res[sel_f]["data"], "o-", color="gray", alpha=0.5)
            
            if pd.notnull(calc_res[sel_f]["olcl"]):
                ax.axhline(calc_res[sel_f]["olcl"], color="red", linestyle="--", label="Sheet Limit")
                ax.axhline(calc_res[sel_f]["oucl"], color="red", linestyle="--")
                
            ax.axhline(calc_res[sel_f]["clcl"], color="blue", linewidth=2, label="Calc Limit")
            ax.axhline(calc_res[sel_f]["cucl"], color="blue", linewidth=2)
            ax.axhline(calc_res[sel_f]["cent"], color="green", linestyle=":", label="Center")
            
            ax.set_title(f"Simulation: {calc_source} - {sel_f}"); ax.legend(); ax.grid(True); plt.xticks(rotation=45)
            st.pyplot(fig)
