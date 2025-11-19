

# File Ingestion Dashboard — strict duplicates (core + channel + date)
# -------------------------------------------------------------------
# Requirements: streamlit, pandas, plotly, sqlalchemy, mysql-connector-python, numpy

import os
import re
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime

# --------------------------
# Page config (MUST be first Streamlit call)
# --------------------------
st.set_page_config(page_title="File Ingestion Dashboard", layout="wide")

# --------------------------
# DB Connection (simple: plain variables)
# --------------------------
DB_USER = "report_user"        # <- change me
DB_PASS = "A<<ess"        # <- change me
DB_HOST = "10.20.200.144"   # <- change me
DB_NAME = "reports_db"  # <- change me

engine = create_engine(
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}",
    pool_pre_ping=True,
)
print(engine)



# --------------------------
# Data Load
# --------------------------
@st.cache_data(ttl=300, show_spinner="Loading reports…")
def load_data() -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM reports_2", engine, parse_dates=["DateTime"])

    # Guard against missing columns/types we depend on
    for col in ["filename", "Source", "DB_Status"]:
        if col not in df.columns:
            df[col] = pd.NA

    bool_cols = ["ManuallyDownloaded", "AutoRenamed", "AutoDownloaded", "ManuallyRenamed"]
    for col in bool_cols:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)

    return df

df = load_data().copy()

# --------------------------
# Normalization (STRICT: keep channel + date)
# --------------------------
_MONTHS = {
    "jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
    "jul":"07","aug":"08","sep":"09","sept":"09","oct":"10","nov":"11","dec":"12"
}

# Capture a tail date token
_DATE_TAIL_RE = re.compile(
    r'(?:[_\-\s])('
    r'\d{8}'                              # 07092025 or 20250907
    r'|\d{4}[-_/]\d{2}[-_/]\d{2}'         # 2025-09-07
    r'|\d{2}[-_/]\d{2}[-_/]\d{4}'         # 07-09-2025
    r'|\d{1,2}[\s_-]?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[\s_-]?\d{2,4}'
    r')$',
    re.IGNORECASE
)

# Channel tokens → canonical names (edit this list to fit your data)
_CHANNEL_MAP = {
    "zoho":"zoho", "telegram":"telegram", "email":"email", "gmail":"email",
    "whatsapp":"whatsapp", "drive":"drive", "gdrive":"drive",
    "onedrive":"onedrive", "dropbox":"dropbox", "sharepoint":"sharepoint",
    "portal":"portal", "website":"website", "download":"download", "upload":"upload", "workdrive":"workdrive",
    "direct":"direct"
}
_CHANNEL_TAIL_RE = re.compile(
    r'(?:[_\-\s])(' + "|".join(map(re.escape,_CHANNEL_MAP.keys())) + r')$',
    re.IGNORECASE
)

def _canon_date(raw: str) -> str:
    """Return YYYYMMDD from many common tokens; fallback 'nodate'."""
    t = raw.lower().replace("/", "-").replace("_", "-").replace(" ", "-")

    # 8 digits: yyyyMMdd or ddMMyyyy (heuristic)
    if re.fullmatch(r"\d{8}", t):
        if int(t[:4]) >= 1900:
            y, m, d = t[:4], t[4:6], t[6:8]
        else:
            d, m, y = t[:2], t[2:4], t[4:8]
        return f"{y}{m}{d}"

    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", t)  # yyyy-mm-dd
    if m:
        y, m_, d = m.groups()
        return f"{y}{m_}{d}"

    m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{4})", t)  # dd-mm-yyyy
    if m:
        d, m_, y = m.groups()
        return f"{y}{m_}{d}"

    m = re.fullmatch(r"(\d{1,2})-?(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)-?(\d{2,4})", t)
    if m:
        d, mon, y = m.groups()
        mm = _MONTHS[mon]
        if len(y) == 2:
            y = "20" + y  # assume 2000s for 2-digit years
        return f"{y}{mm}{int(d):02d}"

    digits = re.sub(r"[^0-9]", "", t)
    return digits if digits else "nodate"

# ---------------------------
# Normalize function
# ---------------------------

def normalize_keep_date_channel(file_name: str):
    """
    Normalize filenames for deduplication:
    Returns: (core, channel, date, filename_normalized_strict)
    """
    if not isinstance(file_name, str) or not file_name.strip():
        return ("", "nochan", "nodate", "no_core_nochan_nodate")

    fname = file_name.strip()
    
    # Replace company_update → Analyst_Report (case-insensitive)
    # fname = re.sub(r'company_update', 'Analyst_Report', fname, flags=re.IGNORECASE)

    base = fname.rsplit(".", 1)[0]  # remove extension
    base = re.sub(r"[\s\-]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    base = re.sub(r"(?:_v\d+|_rev\d+|_r\d+)$", "", base)

    # ---------------------------
    # Extract date at the end
    # Matches YYYYMMDD or DDMMYYYY or YYYY-MM-DD
    # ---------------------------
    norm_date = "nodate"
    m = re.search(r'(\d{8}|\d{4}-\d{2}-\d{2})$', base)
    if m:
        raw_date = m.group(1)
        # heuristic: first 4 digits >= 1900 → YYYYMMDD, else DDMMYYYY
        if "-" not in raw_date and int(raw_date[:2]) < 1900:
            # DDMMYYYY
            d, mo, y = raw_date[:2], raw_date[2:4], raw_date[4:8]
            norm_date = f"{y}-{mo}-{d}"
        else:
            # YYYYMMDD or YYYY-MM-DD
            if "-" in raw_date:
                norm_date = raw_date
            else:
                y, mo, d = raw_date[:4], raw_date[4:6], raw_date[6:8]
                norm_date = f"{y}-{mo}-{d}"
        base = base[:m.start()].rstrip("_- ")

    # ---------------------------
    # Extract channel
    # ---------------------------
    norm_channel = "nochan"
    m = _CHANNEL_TAIL_RE.search(base)
    if m:
        raw_chan = m.group(1).lower()
        norm_channel = _CHANNEL_MAP.get(raw_chan, raw_chan)
        base = base[:m.start()].rstrip("_- ")
    else:
        parts = base.split('_')
        if len(parts) >= 2:
            potential_channel = parts[-1].lower()
            if potential_channel in _CHANNEL_MAP:
                norm_channel = _CHANNEL_MAP[potential_channel]
                base = '_'.join(parts[:-1])
            elif potential_channel in _CHANNEL_MAP.values():
                norm_channel = potential_channel
                base = '_'.join(parts[:-1])

    # Core
    norm_core = base.lower()

    # Strict normalized string
    norm_strict = f"{norm_core}_{norm_channel}_{norm_date}"

    return (norm_core, norm_channel, norm_date, norm_strict)

# ---------------------------
# Apply normalization
# ---------------------------

norm_cols = pd.DataFrame(df["filename"].apply(normalize_keep_date_channel).tolist(),
                         columns=["filename_norm_core", "norm_channel", "norm_date", "filename_normalized_strict"])

df = pd.concat([df, norm_cols], axis=1)
df["filename_normalized"] = df["filename_normalized_strict"]

# ---------------------------
# Duplicate detection
# ---------------------------
df['DB_Status'] = ~df['filename_normalized'].duplicated(keep='first')
df['DB_Status'] = df['DB_Status'].map({True: "No", False: "Yes"})   # first occurrence = No, duplicates = Yes

# ---------------------------
# Parse date
# ---------------------------
def parse_norm_date(d):
    if d in ("nodate", "", None):
        return None
    for fmt in ("%Y-%m-%d", "%d%m%Y", "%Y%m%d"):
        try:
            return datetime.strptime(d, fmt)
        except:
            continue
    return None

df['norm_date_dt'] = df['norm_date'].apply(parse_norm_date)

# ---------------------------
# Fiscal year & quarter logic
# ---------------------------

def is_before_fy_cutoff(dt, fy_cutoff=2026, q_cutoff=2):
    if dt is None:
        return False

    month = dt.month
    year = dt.year

    if 1 <= month <= 3:       # Jan–Mar
        q = 3
        fy = year
    elif 4 <= month <= 6:     # Apr–Jun
        q = 4
        fy = year
    elif 7 <= month <= 9:     # Jul–Sep
        q = 1
        fy = year + 1
    else:                     # Oct–Dec
        q = 2
        fy = year + 1

    return fy < fy_cutoff or (fy == fy_cutoff and q < q_cutoff)

cutoff_date = datetime.strptime("02-10-2025", "%d-%m-%Y")
# Mark Analyst Reports before FY2026 Q2 as already in DB
mask = (
    df['filename_norm_core'].str.contains("analyst_report", case=False)
    & df['norm_date_dt'].notna()
    # & df['norm_date_dt'].apply(is_before_fy_cutoff)
    & (df['norm_date_dt'] < cutoff_date)
    
)

df.loc[mask, 'DB_Status'] = "Yes"

update_mask = df['norm_date_dt'].notna() & (df['norm_date_dt'] >= cutoff_date)
# df.loc[df['DB_Status'] != "None", 'DB_Status'] = ~df.loc[df['DB_Status'] != "None", 'filename_normalized'].duplicated(keep='first')
# df['DB_Status'] = df['DB_Status'].map({True: "No", False: "Yes"}).fillna("None")

update_mask = df['norm_date_dt'].notna() & (df['norm_date_dt'] >= cutoff_date)
df.loc[update_mask, 'DB_Status'] = ~df.loc[update_mask, 'filename_normalized'].duplicated(keep='first')
df.loc[update_mask, 'DB_Status'] = df.loc[update_mask, 'DB_Status'].map({True: "No", False: "Yes"})
# ---------------------------
# Cleanup
# ---------------------------
df.drop(columns=['norm_date_dt'], inplace=True)



# Build normalization columns
norm_cols = df["filename"].apply(normalize_keep_date_channel).apply(pd.Series)
norm_cols.columns = ["filename_norm_core", "norm_channel", "norm_date", "filename_normalized_strict"]

# Use strict key downstream
df[["filename_norm_core","norm_channel","norm_date","filename_normalized_strict"]] = norm_cols
df["filename_normalized"] = df["filename_normalized_strict"] # alias for the rest of the app

# --------------------------
# Other derived fields
# --------------------------
def classify(row: pd.Series) -> str:
    md = bool(row.get("ManuallyDownloaded"))
    ad = bool(row.get("AutoDownloaded"))
    mr = bool(row.get("ManuallyRenamed"))
    ar = bool(row.get("AutoRenamed"))
    if md and ar: return "MD_AR"
    if ad and mr: return "AD_MR"
    if md and mr: return "MD_MR"
    if ad and ar: return "AD_AR"
    return "Other"

df["AutomationCategory"] = df.apply(classify, axis=1)
df["Hour"] = df["DateTime"].dt.hour
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["Weekday"] = pd.Categorical(df["DateTime"].dt.day_name(), categories=weekday_order, ordered=True)

def _ext_safe(x):
    try:
        s = str(x)
        if "." in s:
            return s.rsplit(".", 1)[-1].lower()
    except Exception:
        pass
    return "unknown"

df["FileType"] = df.get("filename", pd.Series(index=df.index)).apply(_ext_safe)
df["DB_Status"] = df["DB_Status"].fillna("Unknown")
df["Source"] = df["Source"].fillna("Unknown")

# --------------------------
# Global Filters
# --------------------------
all_sources = ["All"] + sorted(df["Source"].dropna().unique().tolist())
selected_source = st.selectbox("🔎 Select Source:", all_sources)
filtered = df if selected_source == "All" else df[df["Source"] == selected_source]

all_status = ["All"] + sorted(filtered["DB_Status"].dropna().unique().tolist())
selected_status = st.selectbox("📂 Select DB Status:", all_status)
filtered = filtered if selected_status == "All" else filtered[filtered["DB_Status"] == selected_status]

# --------------------------
# Mode Switch
# --------------------------
mode = st.radio("📊 Select Analysis Mode:", ["Today", "Historical", "Custom Range"], horizontal=True)
today = pd.Timestamp.today().normalize()

if mode == "Today":
    mask = filtered["DateTime"].dt.date == today.date()
    df_selected = filtered.loc[mask].copy()
    st.header(f"📅 Spotlight Analysis — {today.date()}")
elif mode == "Custom Range":
    min_d = filtered["DateTime"].dropna().min().date()
    max_d = filtered["DateTime"].dropna().max().date()
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", min_d, min_value=min_d, max_value=max_d)
    end_date = col2.date_input("End Date", max_d, min_value=min_d, max_value=max_d)
    if start_date > end_date:
        st.error("Start Date must be on or before End Date.")
        df_selected = filtered.iloc[0:0].copy()
    else:
        mask = (filtered["DateTime"].dt.date >= start_date) & (filtered["DateTime"].dt.date <= end_date)
        df_selected = filtered.loc[mask].copy()
    st.header(f"📅 Custom Range Analysis — {start_date} → {end_date}")
else:
    df_selected = filtered.copy()
    st.header("📈 Historical Analysis (All Time)")

# --------------------------
# Analysis
# --------------------------
if df_selected.empty:
    st.warning("No data available for the selected period.")
else:
    # --- KPIs (STRICT key)
    total_files = len(df_selected)
    KEY_COL = "filename_normalized"  # strict = core + channel + datenormalize_filename


    # changing here datetime 
    # Ensure date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_selected['DateTime']):
        df_selected['DateTime'] = pd.to_datetime(df_selected['DateTime'], errors='coerce')

    names = df_selected[KEY_COL].fillna("")
    valid = names.ne("")

    # Get first occurrence dates for each CORE content (ignoring source/channel for true uniqueness)
    # Group by core filename + date (excluding channel) for source-level deduplication
    core_key = df_selected['filename_norm_core'] + '_' + df_selected['norm_date']
    
    first_appearances = (
        df_selected
        .assign(core_key=core_key)
        .sort_values('DateTime')  # Sort by date to get earliest occurrence
        .groupby('core_key')
        .agg({
            'DateTime': 'first',
            'DB_Status': 'first',  # Get the initial DB status
            'Source': 'first',     # Get the source of first occurrence
            'filename': 'first',   # Get the filename of first occurrence
            'filename_normalized': 'first'  # Get the full normalized name of first occurrence
        })
    )

    # Get DB status counts for display (ROW-LEVEL counts)
    db_status_counts = df_selected["DB_Status"].value_counts(dropna=False).to_dict()
    
    # Calculate duplicates based on core content (ignoring source for true deduplication)
    df_selected_with_core = df_selected.assign(core_key=core_key)
    core_names = df_selected_with_core['core_key'].fillna("")
    core_valid = core_names.ne("")
    
    # Count unique core content (CORE-LEVEL count)
    unique_core_content = core_names[core_valid].nunique()
    
    # CORRECTED: Calculate proper counts for math validation
    # Files that exist in DB (ROW-LEVEL)
    files_already_in_db = int(db_status_counts.get("Yes", 0))
    
    # Files newly ingested (ROW-LEVEL) 
    files_newly_ingested = int(db_status_counts.get("No", 0))
    
    # CORE-LEVEL unique ingestion count (first occurrence with status "No")
    # unique_files_mask = (first_appearances['DB_Status'] == 'No')
    # unique_files_count_core_level = unique_files_mask.sum()
    unique_files_count_core_level = (df_selected_with_core.groupby('core_key')['DB_Status'].apply(lambda x: 'No' in x.values).sum())

    
    
    # Calculate duplicates SEPARATELY for each DB_Status to get accurate math
    
    # ALL FILES - Calculate duplicate groups and extra copies (ROW-LEVEL logic)
    core_name_counts = core_names.groupby(core_names).transform("size")
    dups_any_mask = core_valid & (core_name_counts >= 2)
    
    # Duplicate groups (unique core content that appears more than once)
    dup_groups = int(core_names.loc[dups_any_mask].nunique())

    
    # Total rows that are part of duplicate groups
    total_rows_in_groups = int(dups_any_mask.sum())
    
    # Extra copies beyond the first occurrence (ALL FILES)
    extra_copies_all = total_rows_in_groups - dup_groups
    
    # NEW FILES ONLY - Calculate extra copies for "No" status files
    new_files_df = df_selected_with_core[df_selected_with_core['DB_Status'] == 'No'].copy()
    if len(new_files_df) > 0:
        new_core_names = new_files_df['core_key'].fillna("")
        new_core_valid = new_core_names.ne("")
        new_core_name_counts = new_core_names.groupby(new_core_names).transform("size")
        new_dups_mask = new_core_valid & (new_core_name_counts >= 2)
        
        new_dup_groups = int(new_core_names.loc[new_dups_mask].nunique())
        new_total_rows_in_groups = int(new_dups_mask.sum())
        extra_copies_new_only = new_total_rows_in_groups - new_dup_groups
    else:
        extra_copies_new_only = 0
        new_dup_groups = 0
        new_total_rows_in_groups = 0
    
    # ROW-LEVEL breakdown for math validation: 
    # Files_Newly_Ingested = Unique_New_Content + Extra_Copies_of_New_Content
    rows_new_content_no_duplicates = files_newly_ingested - extra_copies_new_only
    
    # For backward compatibility, use total extra copies for overall stats
    extra_copies = extra_copies_all
    rows_after_dedupe = total_files - extra_copies_all  # files after deduplication
    
    # Verify: rows_new_content_no_duplicates should equal unique_files_count_core_level
    
    # Calculate download method counts (handle overlaps properly)
    manual_only = (df_selected['ManuallyDownloaded'] == True) & (df_selected['AutoDownloaded'] != True)
    auto_only = (df_selected['AutoDownloaded'] == True) & (df_selected['ManuallyDownloaded'] != True)
    both_methods = (df_selected['ManuallyDownloaded'] == True) & (df_selected['AutoDownloaded'] == True)
    neither_method = (df_selected['ManuallyDownloaded'] != True) & (df_selected['AutoDownloaded'] != True)
    
    manual_only_count = int(manual_only.sum())
    auto_only_count = int(auto_only.sum())
    both_methods_count = int(both_methods.sum())
    neither_count = int(neither_method.sum())
    
    # For backward compatibility, keep original counts
    manual_downloads = int((df_selected['ManuallyDownloaded'] == True).sum())
    auto_downloads = int((df_selected['AutoDownloaded'] == True).sum())
    
    manual_pct = (manual_downloads / total_files * 100) if total_files > 0 else 0
    auto_pct = (auto_downloads / total_files * 100) if total_files > 0 else 0
    
    ingestion_pct = (files_newly_ingested / total_files * 100) if total_files > 0 else 0
    existing_pct = (files_already_in_db / total_files * 100) if total_files > 0 else 0
    duplicate_pct_new = (extra_copies_new_only / total_files * 100) if total_files > 0 else 0
    duplicate_pct_all = (extra_copies / total_files * 100) if total_files > 0 else 0

    # analyst report calculation for db_status = None
    unique_files_none = (first_appearances['DB_Status'] == 'No')
    unique_files_count_core_none = unique_files_none.sum()
    
    # counts all the old_files for db_status = None (with duplicates)
    all_files_count = 0


    # final unique calculation after subratcting the none values

    # Display metrics with updated calculations and percentages
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Files", f"{total_files:,}")
    c2.metric("Duplicate Groups", f"{dup_groups:,}", f"{(dup_groups/unique_core_content*100):.1f}% of unique content")
    c3.metric("Already in DB", f"{files_already_in_db:,}", f"{existing_pct:.1f}%")
    c4.metric("Analyst Files with Date",f"{unique_files_count_core_none}")
    c5.metric("Unique Ingested", f"{unique_files_count_core_level:,}", f"{(unique_files_count_core_level/total_files*100):.1f}%")

    
    # Add a summary overview table
    st.subheader("📈 File Processing Summary")
    summary_data = {
        "Category": [
            "🆕 Unique Files Ingested", 
            "📁 Already in Database", 
            "🔄 Duplicate", 
            "📋 Extra Copies", 
            "📥 Manual Downloads", 
            "🤖 Auto Downloads"
        ],
        "Count": [
            f"{unique_files_count_core_level:,}", 
            f"{files_already_in_db:,}", 
            f"{dup_groups:,}", 
            f"{extra_copies_new_only:,}", 
            f"{manual_downloads:,}", 
            f"{auto_downloads:,}"
        ],
        "% of Total": [
            f"{(unique_files_count_core_level/total_files*100):.1f}%", 
            f"{existing_pct:.1f}%", 
            f"{(dup_groups/total_files*100):.1f}%", 
            f"{duplicate_pct_new:.1f}%", 
            f"{manual_pct:.1f}%", 
            f"{auto_pct:.1f}%"
        ],
        "Status": [
            "✅ New Content", 
            "⏭️ Skip", 
            "🔄 Has Duplicates", 
            "📋 Extra Copies", 
            "👤 Manual", 
            "🤖 Automated"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Add detailed metrics with proper math validation
    with st.expander("📊 Detailed Breakdown & Math Validation"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("File Breakdown (Row-Level Math)")
            breakdown_data = {
                "Category": ["Files Newly Ingested", "Files Already in DB", "Analyst count before Q2","**TOTAL**"],
                "Count": [files_newly_ingested, files_already_in_db, all_files_count, total_files],
                "Percentage": [f"{(files_newly_ingested/total_files*100):.1f}%", 
                              f"{existing_pct:.1f}%",
                              f"{(all_files_count/total_files*100):.1f}%", 
                              "100.0%"]
            }
            st.table(pd.DataFrame(breakdown_data))
            
            st.subheader("New Files Breakdown")
            new_files_breakdown = {
                "Category": ["Unique New Content", "Extra Copies of New", "**TOTAL NEW**"],
                "Count": [rows_new_content_no_duplicates, extra_copies_new_only, files_newly_ingested],
                "Percentage": [f"{(rows_new_content_no_duplicates/total_files*100):.1f}%", 
                              f"{(extra_copies_new_only/total_files*100):.1f}%",
                              f"{(files_newly_ingested/total_files*100):.1f}%"]
            }
            st.table(pd.DataFrame(new_files_breakdown))
            
            # Math validation
            calculated_total = files_newly_ingested + files_already_in_db +  all_files_count
            if calculated_total == total_files:
                st.success(f"✅ Basic Math: {files_newly_ingested} + {files_already_in_db} + {all_files_count} = {total_files}")
            else:
                st.error(f"❌ Basic Math Error: {calculated_total} ≠ {total_files}")
                
            # Validate new files breakdown
            new_calculated = rows_new_content_no_duplicates + extra_copies_new_only
            if new_calculated == files_newly_ingested:
                st.success(f"✅ New Files Math: {rows_new_content_no_duplicates} + {extra_copies_new_only} = {files_newly_ingested}")
            else:
                st.error(f"❌ New Files Math Error: {new_calculated} ≠ {files_newly_ingested}")
                
            # Core-level validation
            if rows_new_content_no_duplicates == unique_files_count_core_level:
                st.success(f"✅ Core-Level Consistency: {unique_files_count_core_level} unique content matches row-level calculation")
            else:
                st.error(f"❌ Core-Level Mismatch: {unique_files_count_core_level} (core) ≠ {rows_new_content_no_duplicates} (row)")
                
            st.info(f"🔍 All Files Extra Copies: {extra_copies} (includes both 'Yes' and 'No' status duplicates)")
            st.info(f"🔍 New Files Only Extra Copies: {extra_copies_new_only} (only 'No' status duplicates)")
        
        with col2:
            st.subheader("Duplication Analysis")
            dup_analysis_data = {
                "Metric": [
                    "Unique Core Content", 
                    "Files with Duplicates (All)", 
                    "Files without Duplicates", 
                    "Extra Copies (All Files)",
                    "Extra Copies (New Files Only)",
                    "Avg Copies per Duplicate Group"
                ],
                "Value": [
                    f"{unique_core_content:,}", 
                    f"{dup_groups:,}", 
                    f"{unique_core_content - dup_groups:,}", 
                    f"{extra_copies:,}",
                    f"{extra_copies_new_only:,}",
                    f"{(total_rows_in_groups/dup_groups):.1f}" if dup_groups > 0 else "0"
                ]
            }
            st.table(pd.DataFrame(dup_analysis_data))
            
            st.subheader("Download Method Breakdown (Non-Overlapping)")
            download_data = {
                "Method": ["Manual Only", "Auto Only", "Both Methods", "Neither/Unknown"],
                "Count": [manual_only_count, auto_only_count, both_methods_count, neither_count],
                "Percentage": [f"{(manual_only_count/total_files*100):.1f}%", 
                              f"{(auto_only_count/total_files*100):.1f}%", 
                              f"{(both_methods_count/total_files*100):.1f}%",
                              f"{(neither_count/total_files*100):.1f}%"]
            }
            st.table(pd.DataFrame(download_data))
            
            # Validation for download methods
            download_total = manual_only_count + auto_only_count + both_methods_count + neither_count
            if download_total == total_files:
                st.success(f"✅ Download Methods Math: {manual_only_count} + {auto_only_count} + {both_methods_count} + {neither_count} = {total_files}")
            else:
                st.error(f"❌ Download Methods Error: {download_total} ≠ {total_files}")
    with st.expander("Dedup math (strict key)"):
        st.write("All Files Deduplication:")
        st.write(
            {
                "rows_in_duplicate_groups_all": total_rows_in_groups,
                "duplicate_groups_all": dup_groups,
                "extra_copies_all_files": extra_copies,
                "rows_after_dedupe_keep_1_per_group": rows_after_dedupe,
            }
        )
        st.write("New Files Only Deduplication:")
        st.write(
            {
                "new_files_total": files_newly_ingested,
                "new_dup_groups": new_dup_groups,
                "new_total_rows_in_groups": new_total_rows_in_groups,
                "extra_copies_new_only": extra_copies_new_only,
                "rows_new_content_no_duplicates": rows_new_content_no_duplicates,
                "unique_files_count_core_level": unique_files_count_core_level,
            }
        )



    # --- Auto vs Manual Downloads (long format)
    download_type = pd.Series(pd.NA, index=df_selected.index, dtype="object")
    download_type = download_type.mask(df_selected["AutoDownloaded"], "Auto")
    download_type = download_type.mask(df_selected["ManuallyDownloaded"], "Manual")
    df_selected = df_selected.assign(DownloadType=download_type.fillna("Other"))

    dl_summary = (
        df_selected.assign(Date=df_selected["DateTime"].dt.date)
        .groupby(["Date", "DownloadType"])
        .size()
        .reset_index(name="count")
        .sort_values("Date")
    )
    fig_dl = px.line(
        dl_summary, x="Date", y="count", color="DownloadType",
        title="Auto vs Manual Downloads Over Time"
    )
    st.plotly_chart(fig_dl, use_container_width=True)




    # --- DB Status Trend with Deduplication (by core content)
    trend_data = (
        df_selected_with_core
        .assign(Date=lambda x: x['DateTime'].dt.date)  # Convert to date type for grouping
        .groupby(['core_key', 'Date'])
        .agg({
            'DB_Status': 'first'  # Take the first status for each unique core content on each day
        })
        .reset_index()
    )
    
    # Create the trend analysis
    db_trend = (
        trend_data
        .groupby(['Date', 'DB_Status'])
        .size()
        .reset_index(name='count')
        .sort_values('Date')
    )
    
    if not db_trend.empty:
        fig2 = px.line(
            db_trend, 
            x="Date", 
            y="count", 
            color="DB_Status", 
            title="DB Status Trend (Deduplicated by Core Content)"
        )
        fig2.update_layout(yaxis_title="Number of Unique Files (by content)")
        st.plotly_chart(fig2, use_container_width=True)

    # --- Automation Summary
    auto_df = df_selected.groupby("AutomationCategory").size().reset_index(name="count")
    fig_auto = px.bar(auto_df, x="AutomationCategory", y="count", text="count", title="Automation Categories")
    st.plotly_chart(fig_auto, use_container_width=True)

    # --- Automation → Source Breakdown (stacked)
    auto_source = (
        df_selected.groupby(["AutomationCategory", "Source"]).size().reset_index(name="count")
    )
    fig_auto_src = px.bar(
        auto_source,
        x="AutomationCategory",
        y="count",
        color="Source",
        barmode="stack",
        title="Automation Category → Source Breakdown",
    )
    st.plotly_chart(fig_auto_src, use_container_width=True)

    st.dataframe(auto_source)
    # ======================
    # Duplicates Breakdown (by Core Content)
    # ======================
    st.subheader("🌀 Duplicates Analysis (Source-Level Deduplication)")

    dups = df_selected_with_core.loc[dups_any_mask].copy()
    if dups.empty:
        st.info("No duplicates found.")
    else:
        # Add first occurrence information
        dups = dups.merge(
            first_appearances[['DateTime']], 
            left_on='core_key', 
            right_index=True, 
            suffixes=('', '_first')
        )
        
        # Group by core content and source to show source-level duplicates
        dup_table = (
            dups.groupby(['core_key', 'Source'])
            .agg({
                'DateTime': ['min', 'max', 'count'],  # Get first, last, and count of occurrences
                'DB_Status': list,  # List all statuses for this file from this source
                'filename': 'first'  # Get example filename
            })
            .reset_index()
        )
        
        # Flatten column names
        dup_table.columns = [
            'core_key', 'Source', 
            'first_seen', 'last_seen', 'occurrences',
            'status_history', 'example_filename'
        ]
        
        # Calculate time span
        dup_table['days_span'] = (dup_table['last_seen'] - dup_table['first_seen']).dt.days
        
        # Sort by number of occurrences and first seen date
        dup_table = dup_table.sort_values(
            by=['occurrences', 'first_seen'],
            ascending=[False, True]
        )
        st.dataframe(dup_table, use_container_width=True)

        # -------- View A: ROWS in duplicate groups by Source
        rows_by_source = (
            dups.groupby("Source")
            .size()
            .reset_index(name="rows_in_duplicate_groups")
            .sort_values("rows_in_duplicate_groups", ascending=False)
        )

        # -------- View B: GROUPS by Source (attributed, sums to KPI)
        tmp = dup_table.copy()
        tmp["max_occurrences"] = tmp.groupby('core_key')["occurrences"].transform("max")
        tmp["is_top"] = tmp["occurrences"].eq(tmp["max_occurrences"])

        top_candidates = tmp.loc[tmp["is_top"], ['core_key', "Source", "occurrences"]]
        tie_counts = top_candidates.groupby('core_key').size().rename("num_top_sources")
        top_candidates = top_candidates.merge(tie_counts, on='core_key')

        top_candidates["primary_source"] = np.where(
            top_candidates["num_top_sources"] > 1, "Multiple", top_candidates["Source"]
        )

        primary_assignments = (
            top_candidates.sort_values(['core_key', "primary_source"])
            .drop_duplicates('core_key')
            [['core_key', "primary_source"]]
        )

        groups_by_primary_source = (
            primary_assignments.groupby("primary_source")
            .size()
            .reset_index(name="duplicate_groups_attributed")
            .sort_values("duplicate_groups_attributed", ascending=False)
        )

        st.subheader("📊 Aggregated Duplicates by Source")

        colA, colB = st.columns(2)
        with colA:
            st.caption("View A: **Rows** in duplicate groups (row counts).")
            st.dataframe(rows_by_source, use_container_width=True)
            fig_rows = px.bar(
                rows_by_source,
                x="Source",
                y="rows_in_duplicate_groups",
                text="rows_in_duplicate_groups",
                title="Rows in Duplicate Groups by Source"
            )
            fig_rows.update_traces(textposition="outside")
            st.plotly_chart(fig_rows, use_container_width=True)

        with colB:
            st.caption("View B: **Groups** attributed to a source (STRICT; sums to KPI; ties → Multiple).")
            st.dataframe(groups_by_primary_source, use_container_width=True)
            fig_groups = px.bar(
                groups_by_primary_source,
                x="primary_source",
                y="duplicate_groups_attributed",
                text="duplicate_groups_attributed",
                title="Duplicate Groups by Source (Strict, Attributed)"
            )
            fig_groups.update_traces(textposition="outside")
            st.plotly_chart(fig_groups, use_container_width=True)



    # --- Files Table + Download
    st.subheader("📂 Files")
    st.dataframe(df_selected.sort_values("DateTime", ascending=False), use_container_width=True)
    
    # Create properly deduplicated unique files list using CORE CONTENT logic
    # This ensures no source-level duplicates - only one file per core content
    unique_files_by_core = (
        df_selected[df_selected["DB_Status"] == "No"]
        .assign(core_key=df_selected[df_selected["DB_Status"] == "No"]['filename_norm_core'] + '_' + df_selected[df_selected["DB_Status"] == "No"]['norm_date'])
        .sort_values('DateTime')  # Sort by datetime to get earliest occurrence
        .drop_duplicates(['core_key'], keep='first')  # Keep only first occurrence per core content
        [["filename", "DateTime", "Source", "DB_Status", "filename_norm_core", "norm_date", "norm_channel"]]
    )
    
    st.markdown("### Unique Files Ingested (DB_Status='No') - TRUE Source-Level Deduplication")
    st.markdown("**Policy: Only one file per content, regardless of source/channel**")
    st.markdown(f"**Examples that would be deduplicated:**")
    st.markdown("- `ACME_Report_EMAIL_12092025.pdf` and `ACME_Report_WHATSAPP_12092025.pdf` → Only first one counted")
    st.dataframe(unique_files_by_core.sort_values("DateTime", ascending=False), use_container_width=True)
    st.write(f"Total Unique Files Ingested (by core content): {len(unique_files_by_core)}")
    
    # Validate consistency between different unique counts
    if len(unique_files_by_core) == unique_files_count_core_level:
        st.success(f"✅ Consistency Check: Both methods show {unique_files_count_core_level} unique files ingested")
    else:
        st.error(f"❌ Inconsistency: Display shows {len(unique_files_by_core)}, KPI shows {unique_files_count_core_level}")
    
    # Show source-level duplicate analysis
    with st.expander("🔍 Source-Level Duplicate Analysis"):
        # Find all files with DB_Status='No' that have the same core content
        potential_source_dups = df_selected[df_selected["DB_Status"] == "No"].copy()
        potential_source_dups = potential_source_dups.assign(
            core_key=potential_source_dups['filename_norm_core'] + '_' + potential_source_dups['norm_date']
        )
        
        # Count how many sources each core content appears in
        source_counts = (
            potential_source_dups
            .groupby('core_key')['Source']
            .nunique()
            .reset_index()
            .rename(columns={'Source': 'source_count'})
        )
        
        # Find files that appear in multiple sources
        multi_source_files = source_counts[source_counts['source_count'] > 1]
        
        if len(multi_source_files) > 0:
            st.warning(f"⚠️ Found {len(multi_source_files)} files that appear in multiple sources!")
            
            # Show detailed breakdown
            multi_source_details = (
                potential_source_dups
                .merge(multi_source_files[['core_key']], on='core_key')
                .groupby(['core_key', 'Source'])
                .agg({
                    'DateTime': ['min', 'count'],
                    'filename': 'first',
                    'filename_norm_core': 'first',
                    'norm_date': 'first',
                    'norm_channel': 'first'
                })
                .reset_index()
            )
            
            # Flatten column names
            multi_source_details.columns = ['core_key', 'Source', 'first_seen', 'count', 'example_filename', 'core_content', 'date', 'channel']
            multi_source_details = multi_source_details.sort_values(['core_key', 'first_seen'])
            
            st.dataframe(multi_source_details, use_container_width=True)
            
            st.info("💡 **Current Policy**: Only the first occurrence (by datetime) is counted as 'ingested', others are considered source-level duplicates.")
            
        else:
            st.success("✅ No source-level duplicates detected. Each unique file content appears from only one source.")
    
    csv = df_selected.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data (CSV)", data=csv, file_name="filtered_reports.csv", mime="text/csv")