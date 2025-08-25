import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title='SPA Command Center', layout='wide')

DATA_DIR = Path(__file__).parent / 'data'
FIELD_VARS_PATH = DATA_DIR / 'Field variables.xlsx'
PROCESS_FLOW_PATH = DATA_DIR / 'Process Flow.xlsx'
HOLIDAYS_PATH = DATA_DIR / 'dubai_holidays.csv'

def read_excel_safe(path, **kwargs):
    try:
        return pd.read_excel(path, engine='openpyxl', **kwargs)
    except Exception as e:
        st.error(f'Failed to read {path.name}: {e}')
        return pd.DataFrame()

with st.sidebar:
    st.header('Reference Files (from /data)')
    st.write(f"**Field variables:** {'✅ Found' if FIELD_VARS_PATH.exists() else '❌ Missing'}")
    st.write(f"**Process Flow:** {'✅ Found' if PROCESS_FLOW_PATH.exists() else '❌ Missing'}")
    st.write(f"**Holidays (optional):** {'✅ Found' if HOLIDAYS_PATH.exists() else '—'}")

field_vars = read_excel_safe(FIELD_VARS_PATH)
process_flow = read_excel_safe(PROCESS_FLOW_PATH)
code_to_field = {}
if not field_vars.empty:
    cols = {c.strip().lower(): c for c in field_vars.columns}
    code_col = cols.get('col. code no.') or cols.get('col code no.') or cols.get('code')
    field_col = cols.get('field') or cols.get('field name') or cols.get('field (true name)')
    if code_col and field_col:
        code_to_field = dict(zip(field_vars[code_col].astype(str).str.strip(), field_vars[field_col].astype(str).str.strip()))

st.title('SPA Command Center — Deployment Check')
st.write('This is a minimal working app to verify deployment. Upload the two runtime files below each session. The full algorithm can be layered on top.')

col1, col2 = st.columns(2)
with col1:
    report_file = st.file_uploader('Upload Salesforce export (report*.xlsx)', type=['xlsx'], key='report')
with col2:
    crm_file = st.file_uploader('Upload Booking Name & CRM Executive mapping (2 cols)', type=['xlsx'], key='crm')

if not report_file or not crm_file:
    st.info('Please upload both files to proceed.')
    st.stop()

try:
    DF = pd.read_excel(report_file, engine='openpyxl')
    DF.columns = DF.columns.str.strip()
except Exception as e:
    st.error(f'Could not read Salesforce report: {e}')
    st.stop()

try:
    CRM = pd.read_excel(crm_file, engine='openpyxl')
    CRM.columns = CRM.columns.str.strip()
except Exception as e:
    st.error(f'Could not read CRM mapping file: {e}')
    st.stop()

def norm_booking_name(x):
    if pd.isna(x):
        return ''
    return str(x).strip().lower()

def get_col(df, wanted):
    if wanted in df.columns:
        return wanted
    lower_map = {c.lower(): c for c in df.columns}
    return lower_map.get(wanted.lower())

bn_col_report = get_col(DF, 'Booking: Booking Name') or get_col(DF, 'Booking Name')
bn_col_crm = get_col(CRM, 'Booking Name')
crm_fullname_col = get_col(CRM, 'CRM Executive: Full Name') or get_col(CRM, 'CRM Executive')

if not bn_col_report or not bn_col_crm or not crm_fullname_col:
    st.error("Missing expected columns. Ensure the report has 'Booking: Booking Name' and the CRM mapping has 'Booking Name' and 'CRM Executive: Full Name'.")
    st.stop()

DF['_bn_key_'] = DF[bn_col_report].apply(norm_booking_name)
CRM['_bn_key_'] = CRM[bn_col_crm].apply(norm_booking_name)
CRM['_crm_exec_'] = CRM[crm_fullname_col].astype(str).str.strip()

counts = CRM.groupby(['_bn_key_', '_crm_exec_']).size().reset_index(name='count')
idx = counts.groupby('_bn_key_')['count'].idxmax()
majority = counts.loc[idx, ['_bn_key_', '_crm_exec_']].rename(columns={'_crm_exec_':'crm_majority'})
crm_map = dict(zip(majority['_bn_key_'], majority['crm_majority']))

DF['CRM Executive: Full Name (joined)'] = DF['_bn_key_'].map(crm_map)
st.success('Files loaded and CRM majority rule applied (preview below).')
st.dataframe(DF.head(20))
st.caption('This skeleton confirms deployment. Next, add the stage timeline, business-day imputation, KPIs, and visuals per your algorithm.')
