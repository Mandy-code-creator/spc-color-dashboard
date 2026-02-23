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
# VIEW 3: CONTROL LIMIT CALCULATOR (NEW LAYOUT)
# VIEW 3: CONTROL LIMIT CALCULATOR (NEW LAYOUT + dE RULE)
# =========================================================
elif app_mode == "🎛️ Control Limit Calculator":
    st.title("🎛️ Control Limits Analysis & Derived ΔE")
    
    # KHUNG CÀI ĐẶT
    with st.expander("⚙️ 設定參數 (Settings)", expanded=True):
        st.markdown("**Chọn nguồn dữ liệu & Tùy chỉnh tham số riêng:**")
        calc_source = st.radio("Data Source", ["LINE", "LAB"], horizontal=True)
        st.divider()
        cols = st.columns(3)
        factors = ["ΔL", "Δa", "Δb"]
        sig_dict, iqr_dict = {}, {}
        
        for i, f in enumerate(factors):
            with cols[i]:
                st.markdown(f"🔸 **{f}**")
                sig_dict[f] = st.number_input(f"Sigma (K)", value=3.0, step=0.1, key=f"sig_{f}")
                iqr_dict[f] = st.number_input(f"IQR Sens.", value=1.5, step=0.1, key=f"iqr_{f}")

    # TÍNH TOÁN
    calc_res, dE_max_sq = {}, 0
    for f in factors:
        d = spc_data[f][calc_source.lower()]["value"]
        if len(d) >= 3:
            m, s = d.mean(), d.std()
            q1, q3 = d.quantile(0.25), d.quantile(0.75)
            olcl, oucl = get_limit(color, calc_source, f)
            sig, iqr_k = sig_dict[f], iqr_dict[f]
            
            std_lcl, std_ucl = m - sig*s, m + sig*s
            iqr_lcl, iqr_ucl = q1 - iqr_k*(q3-q1), q3 + iqr_k*(q3-q1)
            
            calc_res[f] = {
                "data": d, "batch": spc_data[f][calc_source.lower()]["製造批號"], "m": m, "s": s, "median": d.median(),
                "sig": sig, "iqr_k": iqr_k, "olcl": olcl, "oucl": oucl, "std_lcl": std_lcl, "std_ucl": std_ucl, "iqr_lcl": iqr_lcl, "iqr_ucl": iqr_ucl
            }
            dE_max_sq += max(abs(std_lcl), abs(std_ucl))**2

    # HIỂN THỊ
    if len(calc_res) == 3:
        dE_ucl = math.sqrt(dE_max_sq)
        
        # LOGIC KIỂM TRA dE <= 1.0
        if dE_ucl <= 1.0:
            st.success(f"### 🎯 Target Derived ΔE UCL: **{dE_ucl:.3f}** (✅ Đạt tiêu chuẩn ≤ 1.0)")
        else:
            st.error(f"### 🎯 Target Derived ΔE UCL: **{dE_ucl:.3f}** (⚠️ Vượt giới hạn > 1.0)")
            
        st.markdown("---")

        for f in factors:
            st.markdown(f"### 📊 Analysis: **{f}** ({calc_source})")
            col_chart, col_table = st.columns([2.2, 1])
            res = calc_res[f]
            
            with col_table:
                df_table = pd.DataFrame([
                    {"Method": "0. Spec (Sheet)", "Min": res["olcl"], "Max": res["oucl"], "Center": "-", "Note": "Current Target"},
                    {"Method": f"1. Standard ({res['sig']}σ)", "Min": round(res["std_lcl"], 3), "Max": round(res["std_ucl"], 3), "Center": round(res["m"], 3), "Note": "Basic Stats"},
                    {"Method": f"2. IQR (k={res['iqr_k']})", "Min": round(res["iqr_lcl"], 3), "Max": round(res["iqr_ucl"], 3), "Center": round(res["median"], 3), "Note": "Filtered"}
                ])
                st.dataframe(df_table.style.format({"Min": "{:.3f}", "Max": "{:.3f}", "Center": "{}"}, na_rep="-"), hide_index=True, use_container_width=True)
                st.info(f"**Stats:** μ={res['m']:.3f} | σ={res['s']:.3f} | n={len(res['data'])}")

            with col_chart:
                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax.plot(res["batch"], res["data"], "o-", color="#808080", alpha=0.5, label="Process Data")
                if pd.notnull(res["olcl"]):
                    ax.axhline(res["olcl"], color="black", linestyle="-", linewidth=1.5, label="0. Spec")
                    ax.axhline(res["oucl"], color="black", linestyle="-", linewidth=1.5)
                ax.axhline(res["std_lcl"], color="#d62728", linestyle="--", linewidth=1.5, label=f"1. Std (±{res['sig']}σ)")
                ax.axhline(res["std_ucl"], color="#d62728", linestyle="--", linewidth=1.5)
                ax.axhline(res["iqr_lcl"], color="#1f77b4", linestyle=":", linewidth=2, label=f"2. IQR")
                ax.axhline(res["iqr_ucl"], color="#1f77b4", linestyle=":", linewidth=2)
                ax.axhline(res["m"], color="#2ca02c", linestyle="-.", alpha=0.5, label="Mean")
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9); ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45); fig.subplots_adjust(right=0.75, bottom=0.2)
                st.pyplot(fig); plt.close(fig)
            st.markdown("---")
    else:
        st.warning("Not enough data (min 3 batches).")
