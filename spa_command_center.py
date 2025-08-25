import io
import os
from pathlib import Path
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------- CONFIG ----------
APP_TITLE = "SPA Command Center"
DATA_DIR = Path(__file__).parent / "data"
FIELD_VARS_XLSX = DATA_DIR / "Field variables.xlsx"
PROCESS_FLOW_XLSX = DATA_DIR / "Process Flow.xlsx"
DUBAI_HOLIDAYS_CSV = DATA_DIR / "dubai_holidays.csv"  # optional

# Codes specified (resolved to field names using Field variables.xlsx)
PANEL1_CODES = ['H','I','M','Q','Y','AB','AK','AZ','BB']
PANEL2_DATE_CODES = ['J','K','L','BD','AA','AE','AF','AL']
TIMELINE_CODES = ['J','K','L','BD','AA','AE','AF','AL']
CRM_TRAIL_CODES = ['AX','AY','BA']
FINANCE_CODES = ['O','P','R']   # O,P currency AED; R is %
SALES_CODES = ['S','T','U','W','X']
LOGISTICS_CODES = ['AC','AH','AI','AJ','BC']
TABLE_CODES = ['A','B','C','D','E','F','N']

BUSINESS_WEEKMASK = "1111100"  # Mon-Fri
DEFAULT_ANCHOR_LABELS_PRIORITY = ["SPA Sent Date"]
DEFAULT_ANCHOR_WINDOW_MONTHS = 6

# ---------- UTILS ----------
@st.cache_data(show_spinner=False)
def load_reference_files():
    # Field variables (code -> field mapping)
    field_vars = pd.read_excel(FIELD_VARS_XLSX)
    field_vars.columns = [str(c).strip() for c in field_vars.columns]
    code_col = next(c for c in field_vars.columns if "Col" in c and "Code" in c)
    field_col = next(c for c in field_vars.columns if c.lower().strip() == "field")
    fv = field_vars[[code_col, field_col]].copy()
    fv.columns = ["Code", "Field"]
    fv["Code"] = fv["Code"].astype(str).str.strip()
    fv["Field"] = fv["Field"].astype(str).str.strip()
    code_to_field = dict(zip(fv["Code"], fv["Field"]))
    field_to_code = dict(zip(fv["Field"], fv["Code"]))

    # Process flow (stages & TATs)
    process_flow = pd.read_excel(PROCESS_FLOW_XLSX)
    process_flow.columns = [str(c).strip() for c in process_flow.columns]
    expected_pf_cols = {
        "From Field (Date)": None, "To Field (Date)": None,
        "Flow direction (Field to Field)": None, "Stage": None, "Defined TAT": None
    }
    for k in expected_pf_cols:
        for c in process_flow.columns:
            if k.lower() in c.lower():
                expected_pf_cols[k] = c
                break
        if expected_pf_cols[k] is None:
            raise ValueError(f"Process Flow is missing expected column like: {k}")
    stage_col = expected_pf_cols["Stage"]
    process_flow = process_flow.sort_values(stage_col).reset_index(drop=True)

    # Dubai holidays (optional)
    if DUBAI_HOLIDAYS_CSV.exists():
        holidays = pd.read_csv(DUBAI_HOLIDAYS_CSV)
        if 'date' in holidays.columns:
            hol = pd.to_datetime(holidays['date']).dt.date.values
        else:
            hol = pd.to_datetime(holidays.iloc[:,0]).dt.date.values
        holidays_arr = np.array(hol, dtype='datetime64[D]')
    else:
        holidays_arr = np.array([], dtype='datetime64[D]')

    return code_to_field, field_to_code, process_flow, expected_pf_cols, holidays_arr, field_vars

def _coerce_date(s: pd.Series):
    return pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)

def to_business_days(d1: pd.Series, d2: pd.Series, holidays_arr):
    a = pd.to_datetime(d1).dt.date.values
    b = pd.to_datetime(d2).dt.date.values
    mask_valid = ~pd.isna(a) & ~pd.isna(b)
    out = np.full(len(d1), np.nan, dtype='float')
    if mask_valid.any():
        a_valid = a[mask_valid].astype('datetime64[D]')
        b_valid = b[mask_valid].astype('datetime64[D]')
        calc = np.busday_count(a_valid, b_valid, weekmask=BUSINESS_WEEKMASK, holidays=holidays_arr).astype(float)
        out[mask_valid] = calc
    return out

def _friendly_currency(x):
    if pd.isna(x):
        return ""
    try:
        return f"AED {float(x):,.2f}"
    except:
        return str(x)

def _friendly_percent(x):
    if pd.isna(x):
        return ""
    try:
        val = float(x)
        if val <= 1:
            val *= 100.0
        return f"{val:,.2f}%"
    except:
        return str(x)

def _band_for_delay(d):
    if pd.isna(d):
        return None
    v = float(d)
    if v <= 3: return "≤3"
    if v <= 7: return "3–7"
    if v <= 15: return "7–15"
    if v <= 30: return "15–30"
    if v <= 60: return "30–60"
    if v <= 180: return "60–180"
    if v <= 360: return "180–360"
    return ">360"

def _winsorize(s: pd.Series, p=0.01):
    if s.empty: return s
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s.clip(lower=lo, upper=hi)

def _trim_outliers(s: pd.Series, p=0.01):
    if s.empty: return s
    lo, hi = s.quantile(p), s.quantile(1-p)
    return s[(s>=lo) & (s<=hi)]

# ---------- COMPUTE STAGES ----------
def compute_stage_metrics(df: pd.DataFrame,
                          pf: pd.DataFrame,
                          pf_cols: dict,
                          holidays_arr,
                          tat_overrides=None):
    df = df.copy()
    from_col = pf_cols["From Field (Date)"]
    to_col = pf_cols["To Field (Date)"]
    stage_col = pf_cols["Stage"]
    tat_col = pf_cols["Defined TAT"]

    stages = pf[[stage_col, from_col, to_col, tat_col]].copy()
    stages.columns = ["Stage", "From", "To", "TAT"]
    stages["Stage"] = stages["Stage"].astype(int)

    # Ensure date fields exist & coerce
    for fld in pd.unique(stages[["From","To"]].values.ravel()):
        if fld not in df.columns:
            df[fld] = pd.NaT
        else:
            df[fld] = _coerce_date(df[fld])

    # TAT dict (with overrides)
    tat_dict = {}
    for _, r in stages.iterrows():
        sN = f"S{int(r['Stage'])}"
        tat_dict[sN] = float(r["TAT"]) if pd.notna(r["TAT"]) else np.nan
    if tat_overrides:
        for k,v in tat_overrides.items():
            if k in tat_dict and v is not None:
                tat_dict[k] = float(v)

    # Logs
    log_negative, log_edge_missing, log_single_side = [], [], []

    # Raw durations
    stage_order = []
    for _, r in stages.iterrows():
        sN = f"S{int(r['Stage'])}"
        stage_order.append(sN)
        fr, to = r["From"], r["To"]
        raw = to_business_days(df[fr], df[to], holidays_arr)
        df[f"{sN}_duration_bd_raw"] = raw

    # Build timeline t0..t7
    timeline_fields = []
    if not stages.empty:
        timeline_fields.append(stages.iloc[0]["From"])
        for _, r in stages.iterrows():
            timeline_fields.append(r["To"])
    TL, n = timeline_fields, len(timeline_fields)
    TL_vals = [df[c] if c in df.columns else pd.Series([pd.NaT]*len(df)) for c in TL]

    # Initialize final durations
    for sN in stage_order:
        df[f"{sN}_duration_bd"] = df[f"{sN}_duration_bd_raw"]

    # Row-wise imputation & exclusions
    for idx in range(len(df)):
        row_dates = [TL_vals[i].iloc[idx] if i < len(TL_vals) else pd.NaT for i in range(n)]

        # Edge missing logs
        if n >= 2:
            if pd.isna(row_dates[0]):
                log_edge_missing.append({"Index": idx, "Issue": "Edge-missing", "Stage": "S1", "Edge": "first", "From": TL[0], "To": TL[1]})
            if pd.isna(row_dates[-1]):
                log_edge_missing.append({"Index": idx, "Issue": "Edge-missing", "Stage": f"S{n-1}", "Edge": "last", "From": TL[-2], "To": TL[-1]})

        # Interior equal-split with known bookends
        i = 0
        while i < n-1:
            if pd.notna(row_dates[i]):
                j = i+1
                while j < n and pd.isna(row_dates[j]):
                    j += 1
                if j < n and pd.notna(row_dates[j]) and (j-i) >= 2:
                    # Split total business days across the (j - i) stages
                    start = np.datetime64(pd.to_datetime(row_dates[i]).date())
                    end   = np.datetime64(pd.to_datetime(row_dates[j]).date())
                    bd_total = np.busday_count(start, end, weekmask=BUSINESS_WEEKMASK, holidays=holidays_arr)
                    k_stages = j - i
                    per_stage = float(bd_total) / k_stages if k_stages > 0 else np.nan
                    for s_num in range(i+1, j+1):
                        df.at[idx, f"S{s_num}_duration_bd"] = per_stage
                i = j
            else:
                i += 1

        # Single-sided boundaries → omit stage
        for s_num in range(1, n):
            sN = f"S{s_num}"
            d_from = row_dates[s_num-1]
            d_to = row_dates[s_num]
            if pd.isna(d_from) ^ pd.isna(d_to):
                df.at[idx, f"{sN}_duration_bd"] = np.nan
                log_single_side.append({"Index": idx, "Issue": "Single-sided boundary", "Stage": sN})

        # Negative durations → omit stage
        for s_num in range(1, n):
            sN = f"S{s_num}"
            d_from = row_dates[s_num-1]
            d_to = row_dates[s_num]
            if pd.notna(d_from) and pd.notna(d_to) and (pd.to_datetime(d_to) < pd.to_datetime(d_from)):
                df.at[idx, f"{sN}_duration_bd"] = np.nan
                log_negative.append({"Index": idx, "Issue": "Negative duration", "Stage": sN})

    # Delays & totals
    for sN in stage_order:
        tat = tat_dict.get(sN, np.nan)
        dur = df[f"{sN}_duration_bd"]
        delay = dur - tat
        delay = delay.where(delay > 0, 0)
        df[f"{sN}_delay_bd"] = delay
        df[f"delay_band_{sN}"] = df[f"{sN}_delay_bd"].apply(_band_for_delay)

    dur_cols = [f"{sN}_duration_bd" for sN in stage_order]
    tat_cols = [f"{sN}_tat_tmp" for sN in stage_order]
    tmp = pd.DataFrame({c: df[c] for c in dur_cols})
    for sN in stage_order:
        df[f"{sN}_tat_tmp"] = tat_dict.get(sN, np.nan)
    ttat = pd.DataFrame({c: df[c] for c in tat_cols})

    valid_mask = ~tmp.isna()
    df["total_duration_bd"] = (tmp.where(valid_mask, 0)).sum(axis=1)
    df["total_tat_bd"] = (ttat.where(valid_mask, 0)).sum(axis=1)
    df["total_delay_bd"] = (df["total_duration_bd"] - df["total_tat_bd"]).clip(lower=0)
    df.drop(columns=tat_cols, inplace=True)

    logs = {
        "negative": pd.DataFrame(log_negative),
        "edge_missing": pd.DataFrame(log_edge_missing),
        "single_sided": pd.DataFrame(log_single_side),
    }
    return df, logs, stage_order, tat_dict

# ---------- APP ----------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("SPA process tracking with Dubai business days, SLA monitoring, and per-stage delay analytics.")

    # Load references
    try:
        code_to_field, field_to_code, process_flow, pf_cols, holidays_arr, field_vars_full = load_reference_files()
    except Exception as e:
        st.error(f"Unable to load reference files in /data. {e}")
        st.stop()

    # Uploaders
    left, right = st.columns([1,1])
    with left:
        report_file = st.file_uploader("Upload Salesforce report (report*.xlsx)", type=["xlsx"])
    with right:
        crm_file = st.file_uploader("Upload Booking ↔ CRM Executive map (…CRM Executive…xlsx)", type=["xlsx"])

    if not report_file or not crm_file:
        st.info("Please upload both files to proceed.")
        st.stop()

    # Read uploads
    report = pd.read_excel(report_file)
    report.columns = [str(c).strip() for c in report.columns]
    crm_map = pd.read_excel(crm_file)
    crm_map.columns = [str(c).strip() for c in crm_map.columns]

    # Majority join: Booking Name -> CRM Executive
    rep_bn_col = next((c for c in report.columns if c.lower().strip() == "booking: booking name"), None)
    crm_bn_col = next((c for c in crm_map.columns if c.lower().strip() == "booking name"), None)
    crm_exec_col = next((c for c in crm_map.columns if "crm" in c.lower() and "executive" in c.lower()), None)
    if rep_bn_col is None or crm_bn_col is None or crm_exec_col is None:
        st.error("Could not find required Booking/CRM columns in the uploaded files.")
        st.stop()

    crm_grouped = crm_map.copy()
    crm_grouped[crm_bn_col] = crm_grouped[crm_bn_col].astype(str).str.strip().str.lower()
    crm_grouped[crm_exec_col] = crm_grouped[crm_exec_col].astype(str).stripped if False else crm_grouped[crm_exec_col].astype(str).str.strip()  # guard
    majority_map = (
        crm_grouped.groupby([crm_bn_col, crm_exec_col]).size()
        .reset_index(name='count')
        .sort_values(['count'], ascending=False)
    )
    majority_pick = majority_map.loc[majority_map.groupby(crm_bn_col)['count'].idxmax(), [crm_bn_col, crm_exec_col]]
    majority_dict = dict(zip(majority_pick[crm_bn_col], majority_pick[crm_exec_col]))

    report["_bn_key_"] = report[rep_bn_col].astype(str).str.strip().str.lower()
    report["CRM Executive: Full Name (joined)"] = report["_bn_key_"].map(majority_dict)

    dup_map_log = (
        majority_map.merge(majority_pick, on=crm_bn_col, how="left", suffixes=("", "_chosen"))
    )
    dup_map_log["chosen?"] = dup_map_log[crm_exec_col] == dup_map_log[f"{crm_exec_col}_chosen"]

    # ---------- Panel 1 (sidebar) ----------
    with st.sidebar:
        st.subheader("Filters")
        cat_filters = {}
        for code in PANEL1_CODES:
            fld = code_to_field.get(code)
            if fld and fld in report.columns:
                vals = pd.Series(report[fld].astype(str)).replace({"nan": ""}).unique().tolist()
                vals = sorted([v for v in vals if v != ""])
                sel = st.multiselect(f"{fld}", options=vals, default=[])
                if sel: cat_filters[fld] = set(sel)
            else:
                st.warning(f"{code} not found/mapped; skipping.")

        crm_vals = pd.Series(report["CRM Executive: Full Name (joined)"].astype(str)).replace({"nan": ""}).unique().tolist()
        crm_vals = sorted([v for v in crm_vals if v != ""])
        sel_crm = st.multiselect("CRM Executive: Full Name", options=crm_vals, default=[])
        if sel_crm: cat_filters["CRM Executive: Full Name (joined)"] = set(sel_crm)

        st.markdown("---")
        outlier_mode = st.selectbox("CRM delay outlier handling", ["Trim 1% (default)", "Winsorize 1%", "None"], index=0)

        with st.expander("Edit Stage TATs (business days)"):
            tat_overrides_inputs = {}
            pf = process_flow.sort_values(pf_cols["Stage"])
            for _, r in pf.iterrows():
                sN = f"S{int(r[pf_cols['Stage']])}"
                default_tat = r[pf_cols["Defined TAT"]]
                tat_overrides_inputs[sN] = st.number_input(f"{sN} TAT", min_value=0.0, value=float(default_tat) if pd.notna(default_tat) else 0.0, step=1.0)

    # Apply Panel 1 filters
    df_work = report.copy()
    for fld, allowed in cat_filters.items():
        if fld in df_work.columns:
            df_work = df_work[df_work[fld].astype(str).isin(allowed)]

    # ---------- Panel 2 (top bar) ----------
    st.markdown("### Date Filters")
    panel2_fields = [(c, code_to_field.get(c)) for c in PANEL2_DATE_CODES]
    avail_names = [fld for _, fld in panel2_fields if fld in df_work.columns]

    default_anchor = None
    for pref in DEFAULT_ANCHOR_LABELS_PRIORITY:
        for _, fld in panel2_fields:
            if fld and fld in df_work.columns and pref.lower() in fld.lower():
                default_anchor = fld; break
        if default_anchor: break
    if not default_anchor and avail_names: default_anchor = avail_names[0]

    ccols = st.columns([1.2, 1, 1, 1, 1, 1, 1, 1, 1])
    with ccols[0]:
        anchor_field = st.selectbox("Anchor date (default window applies here)", options=avail_names, index=avail_names.index(default_anchor) if default_anchor in avail_names else 0)

    today = date.today()
    anchor_start_default = today - timedelta(days=int(30.44 * DEFAULT_ANCHOR_WINDOW_MONTHS))
    date_ranges = {}
    for i, (code, fld) in enumerate(panel2_fields, start=1):
        with ccols[i if i < len(ccols) else -1]:
            if fld and fld in df_work.columns:
                active = st.checkbox(f"Filter {fld}", value=(fld == anchor_field), key=f"chk_{fld}")
                if active:
                    c1, c2 = st.columns(2)
                    with c1:
                        start = st.date_input(f"{fld} from", value=anchor_start_default if fld == anchor_field else None, key=f"{fld}_from")
                    with c2:
                        end = st.date_input(f"{fld} to", value=today if fld == anchor_field else None, key=f"{fld}_to")
                    date_ranges[fld] = (start, end)

    for fld, (dfrom, dto) in date_ranges.items():
        s = pd.to_datetime(df_work[fld], errors="coerce")
        mask = pd.Series(True, index=df_work.index)
        if dfrom is not None: mask &= (s.dt.date >= dfrom)
        if dto is not None:   mask &= (s.dt.date <= dto)
        df_work = df_work[mask]

    # ---------- Compute stages ----------
    df_stages, logs, stage_order, tat_used = compute_stage_metrics(
        df_work, process_flow, pf_cols, holidays_arr, tat_overrides=tat_overrides_inputs
    )

    # ---------- Panel 3 (right): delay bands ----------
    st.markdown("---")
    cols_main = st.columns([4, 1.6])
    main_col, right_panel = cols_main[0], cols_main[1]
    with right_panel:
        st.subheader("Delay Bands")
        stage_band_filters = {}
        bands = ["≤3","3–7","7–15","15–30","30–60","60–180","180–360",">360"]
        for sN in stage_order:
            sel = st.multiselect(f"{sN} delay", options=bands, default=[], key=f"band_{sN}")
            if sel: stage_band_filters[sN] = set(sel)
    df_filtered = df_stages.copy()
    for sN, allowed in stage_band_filters.items():
        df_filtered = df_filtered[df_filtered[f"delay_band_{sN}"].isin(allowed)]

    # ---------- Chart ----------
    with main_col:
        st.subheader("Stage Performance (Business Days)")
        means, medians, p90s, sla_hits, means_delay, p90_delay, n_valid, tat_list = [],[],[],[],[],[],[],[]
        for sN in stage_order:
            dur = df_filtered[f"{sN}_duration_bd"]; valid = dur.dropna()
            means.append(valid.mean() if not valid.empty else 0)
            medians.append(valid.median() if not valid.empty else 0)
            p90s.append(valid.quantile(0.90) if not valid.empty else 0)
            tat = tat_used.get(sN, np.nan); tat_list.append(tat if pd.notna(tat) else 0)
            sla_hits.append((valid <= tat).mean()*100 if (not valid.empty and pd.notna(tat)) else np.nan)
            dly = df_filtered[f"{sN}_delay_bd"].dropna()
            means_delay.append(dly.mean() if not dly.empty else 0)
            p90_delay.append(dly.quantile(0.90) if not dly.empty else 0)
            n_valid.append(int(valid.shape[0]))
        fig = go.Figure()
        fig.add_bar(x=stage_order, y=means, name="Mean Duration (bdays)")
        fig.add_bar(x=stage_order, y=tat_list, name="Defined TAT (bdays)")
        fig.update_layout(barmode="group", xaxis_title="Stage", yaxis_title="Business Days", legend_title="Metric")
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Summary ----------
    with main_col:
        st.markdown("### Summary Statistics")
        summary_stage = pd.DataFrame({
            "Stage": stage_order,
            "N (valid)": n_valid,
            "Mean Duration (bdays)": np.round(means,2),
            "Median Duration (bdays)": np.round(medians,2),
            "P90 Duration (bdays)": np.round(p90s,2),
            "SLA Hit %": [np.round(x,2) if pd.notna(x) else np.nan for x in sla_hits],
            "Mean Delay (bdays)": np.round(means_delay,2),
            "P90 Delay (bdays)": np.round(p90_delay,2),
            "TAT (bdays)": np.round(tat_list,2),
        })
        st.dataframe(summary_stage, use_container_width=True)

        total_bands = df_filtered["total_delay_bd"].apply(_band_for_delay)
        band_counts = total_bands.value_counts(dropna=True).reindex(
            ["≤3","3–7","7–15","15–30","30–60","60–180","180–360",">360"], fill_value=0
        )
        st.write("**Total Delay (bdays) – band distribution**")
        st.bar_chart(band_counts)

    # ---------- CRM & Bottlenecks ----------
    with main_col:
        st.markdown("### Bottlenecks & CRM Performance")
        max_p90_delay = max([x for x in p90_delay if x is not None], default=1) or 1
        max_mean_delay = max([x for x in means_delay if x is not None], default=1) or 1
        rows = []
        for i, sN in enumerate(stage_order):
            sla_comp = 0.5*(1 - (sla_hits[i]/100.0 if sla_hits[i]==sla_hits[i] else 0))
            p90_comp = 0.3*((p90_delay[i]/max_p90_delay) if max_p90_delay>0 else 0)
            mean_comp = 0.2*((means_delay[i]/max_mean_delay) if max_mean_delay>0 else 0)
            rows.append({"Stage": sN, "Score": sla_comp + p90_comp + mean_comp})
        st.write("**Top 5 Bottleneck Stages (composite score)**")
        st.dataframe(pd.DataFrame(rows).sort_values("Score", ascending=False).head(5), use_container_width=True)

        # CRM delay leaderboard (outlier handling)
        crm = df_filtered.groupby("CRM Executive: Full Name (joined)", dropna=False)["total_delay_bd"].agg(list).reset_index()
        perf_rows = []
        for _, r in crm.iterrows():
            name = r["CRM Executive: Full Name (joined)"] or ""
            series = pd.Series(r["total_delay_bd"]).dropna()
            if st.session_state.get("outlier_mode", "Trim 1% (default)") == "Winsorize 1%":
                series2 = _winsorize(series, 0.01)
            elif st.session_state.get("outlier_mode", "Trim 1% (default)") == "None":
                series2 = series
            else:
                series2 = _trim_outliers(series, 0.01)
            perf_rows.append({
                "CRM Executive": name,
                "Cases": int(series.shape[0]),
                "Avg Total Delay (bdays)": round(series2.mean(),2) if not series2.empty else np.nan,
                "Median Total Delay (bdays)": round(series2.median(),2) if not series2.empty else np.nan,
            })
        st.write("**CRM Delay Leaderboard (outlier-adjusted)**")
        st.dataframe(pd.DataFrame(perf_rows).sort_values(["Avg Total Delay (bdays)"], ascending=False).head(10), use_container_width=True)

    # ---------- Table + Case modal ----------
    with main_col:
        st.markdown("### Cases")
        table_fields = []
        for c in TABLE_CODES:
            fld = code_to_field.get(c)
            if fld in df_filtered.columns: table_fields.append(fld)
        table_df = df_filtered[table_fields].copy() if table_fields else pd.DataFrame()
        if not table_df.empty:
            table_df = table_df.reset_index(drop=True)
            table_df.insert(0, "Select", False)
            edited = st.data_editor(table_df, key="case_table", use_container_width=True, num_rows="fixed")
            selected_rows = edited.index[edited["Select"] == True].tolist()
            if st.button("Open selected case details", use_container_width=False):
                if not selected_rows:
                    st.warning("Select a row first.")
                else:
                    irow = selected_rows[0]
                    with st.modal("Case Details", width=1000):
                        row = df_filtered.iloc[irow]
                        st.subheader("Timeline")
                        tcols = st.columns(2)
                        timeline_fields = [code_to_field.get(c) for c in TIMELINE_CODES if code_to_field.get(c) in df_filtered.columns]
                        with tcols[0]:
                            for fld in timeline_fields:
                                d = pd.to_datetime(row.get(fld), errors="coerce")
                                st.write(f"**{fld}:** {d.strftime('%Y-%m-%d') if pd.notna(d) else ''}")
                        with tcols[1]:
                            for sidx in range(1, len(timeline_fields)):
                                prev = pd.to_datetime(row.get(timeline_fields[sidx-1]), errors='coerce')
                                curr = pd.to_datetime(row.get(timeline_fields[sidx]), errors='coerce')
                                if pd.notna(prev) and pd.notna(curr):
                                    bd = np.busday_count(np.datetime64(prev.date()), np.datetime64(curr.date()), weekmask=BUSINESS_WEEKMASK, holidays=holidays_arr)
                                    st.write(f"**S{sidx} (bdays):** {bd}")
                                else:
                                    st.write(f"**S{sidx} (bdays):** ")

                        st.subheader("CRM Trail")
                        for code in CRM_TRAIL_CODES:
                            fld = code_to_field.get(code)
                            if fld in df_filtered.columns:
                                d = pd.to_datetime(row.get(fld), errors="coerce")
                                st.write(f"**{fld}:** {d.strftime('%Y-%m-%d') if pd.notna(d) else ''}")

                        c1,c2 = st.columns(2)
                        with c1:
                            st.subheader("Finances")
                            for code, fmt in [('O',"AED"),('P',"AED"),('R',"%")]:
                                fld = code_to_field.get(code)
                                if fld in df_filtered.columns:
                                    val = row.get(fld)
                                    if fmt == "AED":
                                        st.write(f"**{fld}:** {_friendly_currency(val)}")
                                    else:
                                        st.write(f"**{fld}:** {_friendly_percent(val)}")
                            st.subheader("Sales Personnel")
                            for code in SALES_CODES:
                                fld = code_to_field.get(code)
                                if fld in df_filtered.columns:
                                    val = str(row.get(fld, "")).strip()
                                    if val: st.write(f"- {fld}: {val}")

                        with c2:
                            st.subheader("Logistic Information")
                            for code in LOGISTICS_CODES:
                                fld = code_to_field.get(code)
                                if fld in df_filtered.columns:
                                    val = str(row.get(fld, "")).strip()
                                    if val: st.write(f"**{fld}:** {val}")

                            st.subheader("Per-Case Stage vs TAT (bdays)")
                            s_means = [row.get(f"S{i}_duration_bd", np.nan) or 0 for i in range(1, 1+len(timeline_fields)-1)]
                            s_tats  = [tat_used.get(f"S{i}", 0) for i in range(1, 1+len(timeline_fields)-1)]
                            x_labels = [f"S{i}" for i in range(1, 1+len(s_means))]
                            fig2 = go.Figure()
                            fig2.add_bar(x=x_labels, y=s_means, name="Duration")
                            fig2.add_bar(x=x_labels, y=s_tats, name="TAT")
                            fig2.update_layout(barmode="group", xaxis_title="Stage", yaxis_title="Business Days", legend_title="Metric")
                            st.plotly_chart(fig2, use_container_width=True)

    # ---------- Data Issues & Audit ----------
    st.markdown("---")
    tabs = st.tabs(["Data Issues", "Audit"])
    with tabs[0]:
        st.write("**Negative durations (omitted):**")
        st.dataframe(logs["negative"], use_container_width=True)
        st.download_button("Download negative durations log", data=logs["negative"].to_csv(index=False), file_name="negative_durations_log.csv")

        st.write("**Edge-missing stages (not imputed):**")
        st.dataframe(logs["edge_missing"], use_container_width=True)
        st.download_button("Download edge-missing log", data=logs["edge_missing"].to_csv(index=False), file_name="edge_missing_log.csv")

        st.write("**Single-sided boundaries (omitted):**")
        st.dataframe(logs["single_sided"], use_container_width=True)
        st.download_button("Download single-sided boundary log", data=logs["single_sided"].to_csv(index=False), file_name="single_sided_boundary_log.csv")

        st.write("**Duplicate CRM map candidates (majority chosen):**")
        st.dataframe(dup_map_log, use_container_width=True)
        st.download_button("Download CRM duplicate mapping log", data=dup_map_log.to_csv(index=False), file_name="crm_duplicate_mapping_log.csv")

    with tabs[1]:
        st.write("**Audit Trail**")
        st.write(f"Salesforce report file: `{report_file.name}`")
        st.write(f"CRM map file: `{crm_file.name}`")
        st.write(f"Data last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
