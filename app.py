import os
import difflib
from io import BytesIO
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="MATA (Supabase) vs PGK Data Verification",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 MATA (Supabase `YgnTBPro`) vs. PGK Excel Data Verification")
st.caption(
    "Automatically loads `YgnTBPro` from Supabase, pairs columns against your uploaded "
    "PGK Excel file (`Presumptive` & `CI` sheets), displays discrepancies in an interactive table, "
    "and exports the verification report to Excel."
)


# ==============================================================================
# 2. HELPER FUNCTIONS: SUPABASE & EXCEL LOADING
# ==============================================================================
def get_default_supabase_credentials():
    """Retrieve Supabase URL and Key from Streamlit secrets or environment variables."""
    url = ""
    key = ""

    # 1. Check st.secrets safely
    try:
        if "SUPABASE_URL" in st.secrets:
            url = st.secrets["SUPABASE_URL"]
        if "SUPABASE_KEY" in st.secrets:
            key = st.secrets["SUPABASE_KEY"]
        if (not url or not key) and "supabase" in st.secrets:
            supa_sec = st.secrets["supabase"]
            url = url or supa_sec.get("url", "") or supa_sec.get("SUPABASE_URL", "")
            key = key or supa_sec.get("key", "") or supa_sec.get("SUPABASE_KEY", "")
        if (not url or not key) and "connections" in st.secrets and "supabase" in st.secrets["connections"]:
            conn_sec = st.secrets["connections"]["supabase"]
            url = url or conn_sec.get("SUPABASE_URL", "") or conn_sec.get("url", "")
            key = key or conn_sec.get("SUPABASE_KEY", "") or conn_sec.get("key", "")
    except Exception:
        pass

    # 2. Fallback to environment variables
    url = url or os.environ.get("SUPABASE_URL", "")
    key = key or os.environ.get("SUPABASE_KEY", "")
    return url, key


@st.cache_data(ttl=600, show_spinner=False)
def fetch_supabase_ygntbpro(supabase_url: str, supabase_key: str, table_name: str = "YgnTBPro", batch_size: int = 1000) -> pd.DataFrame:
    """Fetch all rows from Supabase table `YgnTBPro` using pagination to bypass the 1,000-row limit."""
    from supabase import create_client

    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL and Key are required to load data from Supabase.")

    client = create_client(supabase_url.strip(), supabase_key.strip())
    all_rows = []
    start = 0

    while True:
        response = (
            client.table(table_name)
            .select("*")
            .range(start, start + batch_size - 1)
            .execute()
        )
        batch = response.data or []
        all_rows.extend(batch)
        if len(batch) < batch_size:
            break
        start += batch_size

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["SheetName"] = table_name
    return df


def clean_mata_supabase_df(df_raw: pd.DataFrame, filter_vol_only: bool = True) -> pd.DataFrame:
    """Clean and standardize the Supabase `YgnTBPro` DataFrame."""
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    dfmata = df_raw.copy()

    # Filter VOL == 1 safely across numeric/string types
    if filter_vol_only and "VOL" in dfmata.columns:
        dfmata = dfmata[pd.to_numeric(dfmata["VOL"], errors="coerce") == 1].copy()

    expected_columns = [
        "PatientID", "Name", "Age", "Sex", "VOL", "Referralfor",
        "VolunteerName", "Organization", "TPTregno", "RegnoofindexTBcase",
        "TPTregistrationdate", "TPTregimen", "Date", "TownshipTBRegNumber",
        "TypeofTBTreatment", "Team", "Tsp", "Approach", "Visit_no", "Sr_No"
    ]

    for col in expected_columns:
        if col not in dfmata.columns:
            dfmata[col] = pd.NA

    # Keep expected columns first, plus any additional columns present in Supabase for custom pairing
    extra_cols = [c for c in dfmata.columns if c not in expected_columns]
    dfmata = dfmata[expected_columns + extra_cols].copy()

    # Decode numeric codes into standard labels (supporting both int and string codes)
    dfmata["Sex"] = dfmata["Sex"].replace({1: "M", 2: "F", "1": "M", "2": "F"})
    dfmata["Referralfor"] = dfmata["Referralfor"].replace({
        1: "Presumptive", 2: "CI",
        "1": "Presumptive", "2": "CI"
    })
    dfmata["TypeofTBTreatment"] = dfmata["TypeofTBTreatment"].replace({
        1: "DS-TB", 2: "DR-TB", 3: "TPT",
        "1": "DS-TB", "2": "DR-TB", "3": "TPT"
    })

    # Keep only rows with a valid non-empty PatientID
    dfmata = dfmata[
        dfmata["PatientID"].notna() & (dfmata["PatientID"].astype(str).str.strip() != "")
    ].copy()

    return dfmata


@st.cache_data(show_spinner=False)
def load_and_clean_pgk_excel(file_bytes: bytes) -> pd.DataFrame:
    """Read `Presumptive` and `CI` sheets from PGK Excel bytes and standardize column names."""
    required_sheets = ["Presumptive", "CI"]

    with pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl") as xf:
        missing_sheets = [s for s in required_sheets if s not in xf.sheet_names]
        if missing_sheets:
            raise ValueError(
                f"Missing required sheet(s) {missing_sheets}. Available sheets: {xf.sheet_names}"
            )
        dfs = pd.read_excel(xf, sheet_name=required_sheets)

    dfpgk_p = dfs["Presumptive"].copy()
    dfpgk_c = dfs["CI"].copy()

    # Standardize Presumptive sheet columns
    presumptive_rename_map = {
        "Name of presumptive TB patient": "Name",
        "TB Registeration No.": "TownshipTBRegNumber",
        "Diagnosis Result ": "TypeofTBTreatment",
        "Diagnosis Result": "TypeofTBTreatment",
        "Patients ID": "PatientID"
    }
    if "VL Name" in dfpgk_p.columns:
        presumptive_rename_map["VL Name"] = "VolunteerName"
    elif "Referred Volunteer Name" in dfpgk_p.columns:
        presumptive_rename_map["Referred Volunteer Name"] = "VolunteerName"

    dfpgk_p.rename(columns=presumptive_rename_map, inplace=True)

    # Standardize CI sheet columns in a single dictionary pass
    ci_rename_map = {
        "Sex": "PatientSex",
        "Age": "PatientAge",
        "Name of Contact": "Name",
        "Sex2": "Sex",
        "Age3": "Age",
        "Name": "VolunteerName",
        "Result ": "TypeofTBTreatment",
        "Result": "TypeofTBTreatment",
        "Patients ID": "PatientID"
    }
    dfpgk_c.rename(columns=ci_rename_map, inplace=True)

    dfpgk_c["SheetName"] = "CI"
    dfpgk_p["SheetName"] = "Presumptive"

    # Remove any duplicate column headers before concatenation
    dfpgk_c = dfpgk_c.loc[:, ~dfpgk_c.columns.duplicated()].copy()
    dfpgk_p = dfpgk_p.loc[:, ~dfpgk_p.columns.duplicated()].copy()

    # Combine CI and Presumptive sheets
    dfpgk = pd.concat([dfpgk_c, dfpgk_p], ignore_index=True, sort=False)

    # Normalize 'No active TB' to NA
    if "TypeofTBTreatment" in dfpgk.columns:
        dfpgk["TypeofTBTreatment"] = (
            dfpgk["TypeofTBTreatment"]
            .astype("string")
            .str.strip()
            .replace({"No active TB": pd.NA, "": pd.NA})
        )

    # Filter out empty PatientID rows
    if "PatientID" in dfpgk.columns:
        dfpgk = dfpgk[
            dfpgk["PatientID"].notna() & (dfpgk["PatientID"].astype(str).str.strip() != "")
        ].copy()

    return dfpgk


# ==============================================================================
# 3. HELPER FUNCTIONS: NORMALIZATION, COMPARISON & EXCEL EXPORT
# ==============================================================================
def normalize_compact(val) -> str:
    """Lowercase string with all whitespace stripped for space-insensitive equality checks."""
    if pd.isna(val):
        return ""
    s = "".join(str(val).split()).lower()
    return "" if s in ("nan", "none", "<na>") else s


def normalize_tokens(val) -> str:
    """Lowercase string with single spaces preserved for token-based fuzzy matching."""
    if pd.isna(val):
        return ""
    s = " ".join(str(val).split()).lower()
    return "" if s in ("nan", "none", "<na>") else s


def compute_similarity(val_a, val_b) -> float:
    """Compute similarity score (0-100) using rapidfuzz (or difflib fallback)."""
    compact_a, compact_b = normalize_compact(val_a), normalize_compact(val_b)
    token_a, token_b = normalize_tokens(val_a), normalize_tokens(val_b)
    if HAS_RAPIDFUZZ:
        score = max(
            fuzz.ratio(compact_a, compact_b),
            fuzz.token_sort_ratio(token_a, token_b)
        )
    else:
        score = difflib.SequenceMatcher(None, compact_a, compact_b).ratio() * 100
    return round(float(score), 1)


def values_differ_numeric(val_a, val_b) -> bool:
    """Compare two values numerically when possible (e.g., 48 == 48.0 == '48')."""
    na_a, na_b = pd.isna(val_a), pd.isna(val_b)
    if na_a and na_b:
        return False
    if na_a != na_b:
        return True
    num_a = pd.to_numeric(val_a, errors="coerce")
    num_b = pd.to_numeric(val_b, errors="coerce")
    if pd.notna(num_a) and pd.notna(num_b):
        return float(num_a) != float(num_b)
    return normalize_compact(val_a) != normalize_compact(val_b)


def run_pairwise_verification(
    df_mata: pd.DataFrame,
    df_pgk: pd.DataFrame,
    id_col_mata: str,
    id_col_pgk: str,
    column_pairs: list[dict],
    check_missing_ids: bool = True
) -> pd.DataFrame:
    """
    Merge Supabase (MATA) and PGK DataFrames on the selected ID column pair
    and compare all user-configured column pairs.
    """
    mata = df_mata.copy()
    pgk = df_pgk.copy()

    mata["_join_id"] = mata[id_col_mata].astype(str).str.strip()
    pgk["_join_id"] = pgk[id_col_pgk].astype(str).str.strip()

    merged = mata.merge(
        pgk,
        on="_join_id",
        how="outer",
        suffixes=("_MATA", "_PGK"),
        indicator=True
    )

    def resolve_col(col_name: str, side: str) -> str:
        """Return actual column name in `merged` after suffix resolution."""
        suffixed = f"{col_name}_{side}"
        if suffixed in merged.columns:
            return suffixed
        return col_name

    results = []

    for _, row in merged.iterrows():
        pid = row["_join_id"]

        if row["_merge"] == "left_only":
            if check_missing_ids:
                results.append({
                    "PatientID": pid,
                    "Difference": "PatientID missing in PGK",
                    "PGK": "",
                    "MATA": pid,
                    "Score": None
                })
            continue

        if row["_merge"] == "right_only":
            if check_missing_ids:
                results.append({
                    "PatientID": pid,
                    "Difference": "PatientID missing in MATA",
                    "PGK": pid,
                    "MATA": "",
                    "Score": None
                })
            continue

        # Compare each enabled column pair
        for pair in column_pairs:
            if not pair.get("Enabled", True):
                continue

            m_col_raw = pair["Supabase_Column"]
            p_col_raw = pair["PGK_Column"]
            diff_label = pair.get("Difference_Label") or f"{m_col_raw} vs {p_col_raw}"
            comp_mode = pair.get("Comparison_Type", "Fuzzy Text (with Score)")

            m_col = resolve_col(m_col_raw, "MATA")
            p_col = resolve_col(p_col_raw, "PGK")

            val_mata = row[m_col] if m_col in merged.columns else pd.NA
            val_pgk = row[p_col] if p_col in merged.columns else pd.NA

            if comp_mode == "Numeric / Age":
                if values_differ_numeric(val_mata, val_pgk):
                    results.append({
                        "PatientID": pid,
                        "Difference": diff_label,
                        "PGK": val_pgk,
                        "MATA": val_mata,
                        "Score": None
                    })
            elif comp_mode == "Exact Text":
                if normalize_compact(val_mata) != normalize_compact(val_pgk):
                    results.append({
                        "PatientID": pid,
                        "Difference": diff_label,
                        "PGK": val_pgk,
                        "MATA": val_mata,
                        "Score": None
                    })
            else:  # Fuzzy Text (with Score)
                if normalize_compact(val_mata) != normalize_compact(val_pgk):
                    results.append({
                        "PatientID": pid,
                        "Difference": diff_label,
                        "PGK": val_pgk,
                        "MATA": val_mata,
                        "Score": compute_similarity(val_mata, val_pgk)
                    })

    return pd.DataFrame(results, columns=["PatientID", "Difference", "PGK", "MATA", "Score"])


def generate_excel_bytes(df_diff: pd.DataFrame, sheet_name: str = "CheckData") -> bytes:
    """Serialize the verification DataFrame into an in-memory .xlsx file for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if df_diff is not None and not df_diff.empty:
            df_diff.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            pd.DataFrame(columns=["PatientID", "Difference", "PGK", "MATA", "Score"]).to_excel(
                writer, sheet_name=sheet_name, index=False
            )
    return output.getvalue()


# ==============================================================================
# 4. SIDEBAR: SUPABASE CONNECTION (`YgnTBPro` AUTOMATICALLY SELECTED)
# ==============================================================================
default_url, default_key = get_default_supabase_credentials()

with st.sidebar:
    st.header("🗄️ Supabase Connection")
    st.info("MATA data is automatically fetched from Supabase table **`YgnTBPro`**.")

    supabase_url = st.text_input(
        "Supabase URL",
        value=default_url,
        placeholder="https://your-project.supabase.co",
        help="Loaded automatically from st.secrets['SUPABASE_URL'] or environment variable SUPABASE_URL."
    )
    supabase_key = st.text_input(
        "Supabase Key",
        value=default_key,
        type="password",
        placeholder="Enter Supabase anon / service_role key",
        help="Loaded automatically from st.secrets['SUPABASE_KEY'] or environment variable SUPABASE_KEY."
    )
    table_name = "YgnTBPro"
    st.text_input("Supabase Table", value=table_name, disabled=True)

    filter_vol_only = st.checkbox("Filter `VOL == 1` only", value=True)

    if st.button("🔄 Refresh Supabase Data", use_container_width=True):
        fetch_supabase_ygntbpro.clear()
        st.rerun()


# ==============================================================================
# 5. STEP 1 & 2: LOAD SUPABASE `YgnTBPro` & UPLOAD PGK EXCEL FILE
# ==============================================================================
col_left, col_right = st.columns(2)

dfmata = pd.DataFrame()
dfpgk = pd.DataFrame()

with col_left:
    st.subheader("1️⃣ MATA Data: Supabase (`YgnTBPro`)")
    if not supabase_url or not supabase_key:
        st.warning(
            "Please provide `SUPABASE_URL` and `SUPABASE_KEY` in `.streamlit/secrets.toml`, "
            "environment variables, or the sidebar to automatically load `YgnTBPro`."
        )
    else:
        try:
            with st.spinner("Fetching `YgnTBPro` records from Supabase..."):
                df_supa_raw = fetch_supabase_ygntbpro(supabase_url, supabase_key, table_name="YgnTBPro")
                dfmata = clean_mata_supabase_df(df_supa_raw, filter_vol_only=filter_vol_only)
            st.success(
                f"Loaded **{len(dfmata):,}** valid rows from Supabase `YgnTBPro` "
                f"(Total raw rows fetched: {len(df_supa_raw):,})."
            )
            with st.expander("Preview Supabase `YgnTBPro` Data", expanded=False):
                st.dataframe(dfmata.head(50), use_container_width=True)
        except Exception as exc:
            st.error(f"Failed to load `YgnTBPro` from Supabase: {exc}")

with col_right:
    st.subheader("2️⃣ PGK Data: Upload Excel File")
    uploaded_pgk_file = st.file_uploader(
        "Upload PGK Excel Workbook (must contain `Presumptive` and `CI` sheets)",
        type=["xlsx", "xls", "xlsm"]
    )
    if uploaded_pgk_file is not None:
        try:
            with st.spinner("Reading and standardizing PGK Excel sheets (`Presumptive` & `CI`)..."):
                dfpgk = load_and_clean_pgk_excel(uploaded_pgk_file.getvalue())
            st.success(f"Loaded **{len(dfpgk):,}** valid rows from PGK Excel (`Presumptive` + `CI`).")
            with st.expander("Preview Cleaned PGK Excel Data", expanded=False):
                st.dataframe(dfpgk.head(50), use_container_width=True)
        except Exception as exc:
            st.error(f"Error reading PGK Excel file: {exc}")


# ==============================================================================
# 6. STEP 3: PAIR-WISE COLUMN SELECTION FOR DATA CHECKING
# ==============================================================================
st.subheader("3️⃣ Pair-Wise Column Mapping for Data Checking")

if dfmata.empty or dfpgk.empty:
    st.info("Once both Supabase `YgnTBPro` data and the PGK Excel file are loaded above, configure your column pairs and view the verification results here.")
else:
    mata_cols = list(dfmata.columns)
    pgk_cols = list(dfpgk.columns)

    # --- Join Key Selection ---
    key_col1, key_col2, key_col3 = st.columns([2, 2, 2])
    with key_col1:
        id_col_mata = st.selectbox(
            "🔑 Supabase (`YgnTBPro`) Primary ID Column",
            options=mata_cols,
            index=mata_cols.index("PatientID") if "PatientID" in mata_cols else 0
        )
    with key_col2:
        id_col_pgk = st.selectbox(
            "🔑 PGK Excel Primary ID Column",
            options=pgk_cols,
            index=pgk_cols.index("PatientID") if "PatientID" in pgk_cols else 0
        )
    with key_col3:
        st.write("")
        st.write("")
        check_missing_ids = st.checkbox(
            "Report missing PatientIDs in either source",
            value=True
        )

    # --- Default Pair-Wise Mapping Configuration ---
    default_pair_specs = [
        ("Referralfor", "SheetName", "PresumptiveCI", "Exact Text"),
        ("Name", "Name", "Name", "Fuzzy Text (with Score)"),
        ("Sex", "Sex", "Sex", "Exact Text"),
        ("Age", "Age", "Age", "Numeric / Age"),
        ("VolunteerName", "VolunteerName", "VolunteerName", "Fuzzy Text (with Score)"),
        ("TypeofTBTreatment", "TypeofTBTreatment", "TypeofTBTreatment", "Exact Text"),
        ("TownshipTBRegNumber", "TownshipTBRegNumber", "TownshipTBRegNumber", "Fuzzy Text (with Score)"),
    ]

    if "column_pairs_df" not in st.session_state:
        initial_rows = []
        for m_c, p_c, label, c_type in default_pair_specs:
            if m_c in mata_cols and p_c in pgk_cols:
                initial_rows.append({
                    "Enabled": True,
                    "Supabase_Column": m_c,
                    "PGK_Column": p_c,
                    "Difference_Label": label,
                    "Comparison_Type": c_type
                })
        st.session_state.column_pairs_df = pd.DataFrame(initial_rows)

    st.markdown(
        "Select, modify, or add **pair-wise columns** below to compare each Supabase (`YgnTBPro`) column "
        "against its corresponding PGK Excel column:"
    )

    # Quick Add Pair form
    with st.expander("➕ Add Custom Column Pair", expanded=False):
        add_c1, add_c2, add_c3, add_c4, add_c5 = st.columns([2, 2, 2, 2, 1])
        with add_c1:
            new_supa_col = st.selectbox("Supabase (`YgnTBPro`) Column", options=mata_cols, key="new_supa_col")
        with add_c2:
            new_pgk_col = st.selectbox(
                "PGK Excel Column",
                options=pgk_cols,
                index=pgk_cols.index(new_supa_col) if new_supa_col in pgk_cols else 0,
                key="new_pgk_col"
            )
        with add_c3:
            new_label = st.text_input("Difference Label", value=new_supa_col, key="new_diff_label")
        with add_c4:
            new_comp_type = st.selectbox(
                "Comparison Type",
                options=["Fuzzy Text (with Score)", "Exact Text", "Numeric / Age"],
                key="new_comp_type"
            )
        with add_c5:
            st.write("")
            st.write("")
            if st.button("Add Pair", use_container_width=True):
                new_row = pd.DataFrame([{
                    "Enabled": True,
                    "Supabase_Column": new_supa_col,
                    "PGK_Column": new_pgk_col,
                    "Difference_Label": new_label.strip() or f"{new_supa_col} vs {new_pgk_col}",
                    "Comparison_Type": new_comp_type
                }])
                st.session_state.column_pairs_df = pd.concat(
                    [st.session_state.column_pairs_df, new_row],
                    ignore_index=True
                )
                st.rerun()

    # Interactive Pair-Wise Table Editor
    edited_pairs_df = st.data_editor(
        st.session_state.column_pairs_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Enabled": st.column_config.CheckboxColumn(
                "Check?",
                help="Enable or disable checking this column pair",
                default=True
            ),
            "Supabase_Column": st.column_config.SelectboxColumn(
                "Supabase (YgnTBPro) Column",
                help="Column from Supabase table YgnTBPro",
                options=mata_cols,
                required=True
            ),
            "PGK_Column": st.column_config.SelectboxColumn(
                "PGK Excel Column",
                help="Column from uploaded PGK Excel file",
                options=pgk_cols,
                required=True
            ),
            "Difference_Label": st.column_config.TextColumn(
                "Difference Label",
                help="Name shown in the 'Difference' column of the output report",
                required=True
            ),
            "Comparison_Type": st.column_config.SelectboxColumn(
                "Comparison Type",
                help="How values in this pair should be compared",
                options=["Fuzzy Text (with Score)", "Exact Text", "Numeric / Age"],
                default="Fuzzy Text (with Score)",
                required=True
            ),
        },
        key="pairs_editor"
    )

    # ==============================================================================
    # 7. STEP 4: RUN DATA CHECKING, DISPLAY OUTPUT TABLE & EXPORT TO EXCEL
    # ==============================================================================
    st.subheader("4️⃣ Verification Output Table & Excel Export")

    pair_records = edited_pairs_df.dropna(subset=["Supabase_Column", "PGK_Column"]).to_dict(orient="records")

    with st.spinner("Comparing paired columns between Supabase `YgnTBPro` and PGK Excel..."):
        df_difference = run_pairwise_verification(
            df_mata=dfmata,
            df_pgk=dfpgk,
            id_col_mata=id_col_mata,
            id_col_pgk=id_col_pgk,
            column_pairs=pair_records,
            check_missing_ids=check_missing_ids
        )

    # Summary Metrics
    missing_in_pgk_count = int((df_difference["Difference"] == "PatientID missing in PGK").sum()) if not df_difference.empty else 0
    missing_in_mata_count = int((df_difference["Difference"] == "PatientID missing in MATA").sum()) if not df_difference.empty else 0
    field_mismatch_count = int(len(df_difference) - missing_in_pgk_count - missing_in_mata_count)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Discrepancies", f"{len(df_difference):,}")
    m2.metric("Column Value Mismatches", f"{field_mismatch_count:,}")
    m3.metric("Missing in PGK", f"{missing_in_pgk_count:,}")
    m4.metric("Missing in Supabase (MATA)", f"{missing_in_mata_count:,}")

    # Filter & Search Controls for the Output Table
    filter_col1, filter_col2, export_col = st.columns([2, 2, 2])

    diff_types = sorted(df_difference["Difference"].dropna().unique().tolist()) if not df_difference.empty else []
    with filter_col1:
        selected_diff_types = st.multiselect(
            "Filter by Difference Type",
            options=diff_types,
            default=diff_types
        )
    with filter_col2:
        search_query = st.text_input(
            "Search by PatientID or Value",
            value="",
            placeholder="e.g., KMD26 or Aye..."
        )

    # Apply filters to displayed table
    df_filtered = df_difference.copy()
    if not df_filtered.empty:
        if selected_diff_types:
            df_filtered = df_filtered[df_filtered["Difference"].isin(selected_diff_types)]
        else:
            df_filtered = df_filtered.iloc[0:0]

        if search_query.strip():
            q = search_query.strip().lower()
            mask = (
                df_filtered["PatientID"].astype(str).str.lower().str.contains(q, na=False)
                | df_filtered["PGK"].astype(str).str.lower().str.contains(q, na=False)
                | df_filtered["MATA"].astype(str).str.lower().str.contains(q, na=False)
            )
            df_filtered = df_filtered[mask]

    with export_col:
        timestamp_create = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        excel_filename = f"vsoutput_PythonDataVerification_{timestamp_create}.xlsx"
        export_scope = st.radio(
            "Excel Export Scope",
            options=["All Discrepancies", "Filtered Table Only"],
            horizontal=True
        )
        df_to_export = df_difference if export_scope == "All Discrepancies" else df_filtered
        excel_bytes = generate_excel_bytes(df_to_export, sheet_name="CheckData")

        st.download_button(
            label=f"📥 Generate & Download Excel ({len(df_to_export):,} rows)",
            data=excel_bytes,
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )

    # Breakdown summary expander
    if not df_difference.empty:
        with st.expander("📊 Breakdown by Difference Category", expanded=False):
            summary_counts = (
                df_difference["Difference"]
                .value_counts()
                .rename_axis("Difference Category")
                .reset_index(name="Count")
            )
            st.dataframe(summary_counts, use_container_width=True, hide_index=True)

    # Display Interactive Output Table
    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
        height=500
    )