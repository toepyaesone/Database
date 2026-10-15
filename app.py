import os
from datetime import timedelta
import pandas as pd
import streamlit as st
from supabase import Client, create_client

from functions import (
    PRIMARY_COLUMNS,
    DATE_COLUMNS,
    functionGetDataFromTable,
    align_dataframe_columns,
    clean_and_assign_dtypes,
    compare_data_fast,
    serialize_dataframe_for_supabase,
)

# ============================================================
# 1. CONFIGURATION & PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="YgnTBPro Database Management System",
    page_icon="🩺",
    layout="wide",
)

SUPABASE_URL = st.secrets.get(
    "SUPABASE_URL_ygntbpro",
    st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL_ygntbpro", os.getenv("SUPABASE_URL", ""))),
)
SUPABASE_KEY = st.secrets.get(
    "SUPABASE_KEY_ygntbpro",
    st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY_ygntbpro", os.getenv("SUPABASE_KEY", ""))),
)

TABLE_NAME = "ygntbpro"
TARGET_TABLE = "target"
MAX_ROWS = 1000

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error(
        "Supabase credentials missing! Please configure `SUPABASE_URL_ygntbpro` "
        "and `SUPABASE_KEY_ygntbpro` in `.streamlit/secrets.toml` or Streamlit Cloud Secrets."
    )
    st.stop()


# ============================================================
# 2. SUPABASE CLIENTS & SESSION STATE
# ============================================================

@st.cache_resource
def get_base_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

base_supabase = get_base_client()

def get_user_client() -> Client:
    session = st.session_state.get("session")
    if not session:
        return base_supabase

    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    access_token = getattr(session, "access_token", None)
    refresh_token = getattr(session, "refresh_token", None)

    if access_token and refresh_token:
        try:
            client.auth.set_session(access_token, refresh_token)
            return client
        except Exception:
            pass
    if access_token:
        client.postgrest.auth(access_token)
    return client


DEFAULTS = {
    "session": None,
    "user_role": None,
    "grid_version": 0,
    "filter_version": 0,
    "pending_updates": {},
    "pending_inserts": [],
    "pending_deletes": set(),
    "editor_df": None,
    "editor_source_df": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 3. AUTHENTICATION & RBAC
# ============================================================

def login_user(email: str, password: str):
    try:
        response = base_supabase.auth.sign_in_with_password(
            {"email": email.strip(), "password": password}
        )
        if not response.session:
            return False, "Login failed: No session returned."

        st.session_state.session = response.session
        user_client = get_user_client()

        role_result = (
            user_client.table("user_roles")
            .select("role")
            .eq("user_id", response.session.user.id)
            .limit(1)
            .execute()
        )
        st.session_state.user_role = (
            role_result.data[0].get("role", "viewer")
            if role_result.data
            else "viewer"
        )
        return True, "Login successful."
    except Exception as exc:
        return False, str(exc)


def logout_user():
    try:
        base_supabase.auth.sign_out()
    except Exception:
        pass
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.rerun()


if not st.session_state.session:
    st.title("🔑 YgnTBPro Database Login")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        login_clicked = st.form_submit_button("Login", use_container_width=True, type="primary")

    if login_clicked:
        success, message = login_user(email, password)
        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)
    st.stop()


user_role = st.session_state.user_role or "viewer"
can_edit = user_role in {"editor", "admin"}
can_add = user_role == "admin"
can_delete = user_role == "admin"

st.sidebar.title("🩺 YgnTBPro System")
st.sidebar.write(f"**User Role:** `{user_role}`")
if st.sidebar.button("Logout", use_container_width=True):
    logout_user()


# ============================================================
# 4. SHARED HELPERS & CACHING
# ============================================================

def resolve_column(columns, *candidates):
    lookup = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        found = lookup.get(str(candidate).lower())
        if found is not None:
            return found
    return None

def values_equal(left, right) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return str(left).strip() == str(right).strip()

def clean_value(value):
    return None if pd.isna(value) else value

@st.cache_data(ttl=600, show_spinner=False)
def fetch_full_table(table_name: str, access_token: str = None) -> pd.DataFrame:
    try:
        df = functionGetDataFromTable(table_name, SUPABASE_URL, SUPABASE_KEY, access_token=access_token, page_size=1000)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    response = get_user_client().table(table_name).select("*").limit(MAX_ROWS).execute()
    return pd.DataFrame(response.data or [])

@st.cache_data(ttl=60, show_spinner=False)
def get_unique_values(table_name: str, column: str):
    try:
        result = get_user_client().table(table_name).select(column).limit(10000).execute()
        values = {
            str(row[column]).strip()
            for row in (result.data or [])
            if row.get(column) is not None and str(row[column]).strip()
        }
        return sorted(values, key=str.lower)
    except Exception:
        return []

def clear_pending_changes():
    st.session_state.pending_updates = {}
    st.session_state.pending_inserts = []
    st.session_state.pending_deletes = set()
    st.session_state.editor_source_df = None
    st.session_state.editor_df = None
    st.session_state.grid_version += 1

def capture_editor_changes(source_df: pd.DataFrame, edited_df: pd.DataFrame, primary_key: str):
    if source_df is None or edited_df is None:
        return
    source = source_df.reset_index(drop=True).copy()
    edited = edited_df.reset_index(drop=True).copy()

    if primary_key not in source.columns or primary_key not in edited.columns:
        return

    source_ids = {str(v).strip(): i for i, v in enumerate(source[primary_key]) if pd.notna(v) and str(v).strip()}
    edited_ids = {str(v).strip() for v in edited[primary_key] if pd.notna(v) and str(v).strip()}

    # 1. Track Deletes
    for pid in source_ids:
        if pid not in edited_ids:
            st.session_state.pending_deletes.add(pid)
            st.session_state.pending_updates.pop(pid, None)
        else:
            st.session_state.pending_deletes.discard(pid)

    # 2. Track Updates
    comparable_cols = [c for c in source.columns if c in edited.columns and c not in {primary_key, "updated_at"}]
    for pid, source_idx in source_ids.items():
        if pid not in edited_ids:
            continue
        edited_idx = next((i for i, v in enumerate(edited[primary_key]) if pd.notna(v) and str(v).strip() == pid), None)
        if edited_idx is None:
            continue

        current_pending = st.session_state.pending_updates.get(pid, {}).copy()
        for col in comparable_cols:
            old_val = source.at[source_idx, col]
            new_val = edited.at[edited_idx, col]
            if values_equal(old_val, new_val):
                current_pending.pop(col, None)
            else:
                current_pending[col] = clean_value(new_val)

        if current_pending:
            st.session_state.pending_updates[pid] = current_pending
        else:
            st.session_state.pending_updates.pop(pid, None)

    # 3. Track Inserts (New rows in editor)
    st.session_state.pending_inserts = []
    for _, row in edited.iterrows():
        pid = row.get(primary_key)
        if pd.notna(pid) and str(pid).strip() in source_ids:
            continue
        clean_row = {col: clean_value(val) for col, val in row.items() if col != "updated_at"}
        if any(v not in (None, "") for v in clean_row.values()):
            st.session_state.pending_inserts.append(clean_row)


# ============================================================
# 5. MAIN APPLICATION TABS
# ============================================================

tabs = st.tabs([
    "💾 Explorer & Search",
    "✏️ Data Editor & Sync",
    "📤 Bulk Upload & Reconcile (Excel)",
])

token = getattr(st.session_state.session, "access_token", None)

# ------------------------------------------------------------
# TAB 1: EXPLORER & SEARCH
# ------------------------------------------------------------
with tabs[0]:
    st.subheader("💾 Database Explorer & Query Engine")
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        table_choice = st.selectbox("Select Database Table", [TABLE_NAME, TARGET_TABLE], index=0)
    with col_t2:
        st.write("")
        if st.button("🔄 Refresh Cache", use_container_width=True):
            fetch_full_table.clear()
            st.rerun()

    with st.spinner(f"Loading '{table_choice}' from Supabase..."):
        df_raw = fetch_full_table(table_choice, access_token=token)

    if df_raw.empty:
        st.warning(f"No records returned for `{table_choice}`.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records", f"{len(df_raw):,}")
        m2.metric("Total Columns", f"{len(df_raw.columns):,}")
        m3.metric("Memory Footprint", f"{df_raw.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")

        st.divider()
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            search_query = st.text_input("🔍 Global Keyword Search", placeholder="Search any field...")
        with c2:
            selected_cols = st.multiselect(
                "Columns to Display",
                options=list(df_raw.columns),
                default=list(df_raw.columns)[:15] if len(df_raw.columns) > 15 else list(df_raw.columns),
            )
        with c3:
            max_display = st.number_input("Max Rows to Render", min_value=10, max_value=10000, value=500, step=50)

        df_display = df_raw.copy()
        if search_query:
            mask = df_display.astype(str).apply(
                lambda col: col.str.contains(search_query, case=False, na=False, regex=False)
            ).any(axis=1)
            df_display = df_display[mask]

        if selected_cols:
            df_display = df_display[selected_cols]

        st.dataframe(df_display.head(int(max_display)), use_container_width=True)

        csv_data = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📄 Download Filtered CSV",
            data=csv_data,
            file_name=f"{table_choice}_export.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------
# TAB 2: INTERACTIVE DATA EDITOR & SYNC
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("🩺 Interactive Record Editor")
    schema_sample = get_user_client().table(TABLE_NAME).select("*").limit(1).execute()
    schema_df = pd.DataFrame(schema_sample.data or [])

    if schema_df.empty:
        st.warning("Table is empty or inaccessible.")
    else:
        cols = list(schema_df.columns)
        primary_key = resolve_column(cols, "PatientID", "patientid", "patient_id")
        date_column = resolve_column(cols, "Date", "date")

        filter_columns = {
            "Visit No": resolve_column(cols, "visitno", "Visit_no"),
            "SR No": resolve_column(cols, "srno", "Sr_No"),
            "Patient ID": primary_key,
            "Team": resolve_column(cols, "team", "Team"),
            "TSP": resolve_column(cols, "tsp", "Tsp", "TSP"),
            "Approach": resolve_column(cols, "approach", "Approach"),
            "Case": resolve_column(cols, "case", "Case"),
        }

        fv = st.session_state.filter_version
        fc1, fc2, fc3 = st.columns(3)
        filter_values = {}

        with fc1:
            for lbl in ["Visit No", "SR No", "Patient ID"]:
                col_name = filter_columns[lbl]
                if col_name:
                    filter_values[col_name] = st.multiselect(lbl, get_unique_values(TABLE_NAME, col_name), key=f"f_{lbl}_{fv}")
        with fc2:
            for lbl in ["Team", "TSP", "Approach"]:
                col_name = filter_columns[lbl]
                if col_name:
                    filter_values[col_name] = st.multiselect(lbl, get_unique_values(TABLE_NAME, col_name), key=f"f_{lbl}_{fv}")
        with fc3:
            col_case = filter_columns["Case"]
            if col_case:
                filter_values[col_case] = st.multiselect("Case", get_unique_values(TABLE_NAME, col_case), key=f"f_Case_{fv}")
            date_from = st.date_input("Date From", value=None, key=f"df_{fv}")
            date_to = st.date_input("Date To", value=None, key=f"dt_{fv}")

        if st.button("🔄 Reset Filters"):
            st.session_state.filter_version += 1
            st.session_state.grid_version += 1
            st.rerun()

        client = get_user_client()
        query = client.table(TABLE_NAME).select("*")
        for c_name, vals in filter_values.items():
            if vals:
                query = query.in_(c_name, vals)
        if date_column and date_from:
            query = query.gte(date_column, date_from.isoformat())
        if date_column and date_to:
            query = query.lt(date_column, (date_to + timedelta(days=1)).isoformat())

        res = query.order(primary_key, desc=True).limit(MAX_ROWS).execute()
        df_edit_view = pd.DataFrame(res.data or [])

        if not df_edit_view.empty:
            st.caption(f"Showing {len(df_edit_view):,} row(s). Note: To add a new record, scroll to the bottom of the grid.")
            st.session_state.editor_source_df = df_edit_view.copy()

            # Allow primary_key input when adding rows, but prevent changing existing IDs via diff logic
            disabled_cols = ["updated_at"] if "updated_at" in df_edit_view.columns else []

            edited_df = st.data_editor(
                df_edit_view,
                key=f"editor_grid_{st.session_state.grid_version}",
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic" if can_add else "fixed",
                disabled=disabled_cols,
            )
            if can_edit:
                capture_editor_changes(st.session_state.editor_source_df, edited_df, primary_key)

        # Render Pending Queue
        pending_rows = []
        for pid, chg in st.session_state.pending_updates.items():
            pending_rows.append({"Action": "UPDATE", primary_key: pid, **chg})
        for r in st.session_state.pending_inserts:
            pending_rows.append({"Action": "INSERT", **r})
        for pid in sorted(st.session_state.pending_deletes):
            pending_rows.append({"Action": "DELETE", primary_key: pid})

        if pending_rows:
            st.subheader(f"📝 Pending Changes ({len(pending_rows)})")
            st.dataframe(pd.DataFrame(pending_rows), use_container_width=True)
            b_sync, b_disc = st.columns(2)
            with b_sync:
                if st.button("💾 Commit Changes to Supabase", type="primary", disabled=not can_edit, use_container_width=True):
                    sync_client = get_user_client()
                    errors, success_cnt = [], 0

                    # 1. Execute Updates
                    for pid, u_data in list(st.session_state.pending_updates.items()):
                        try:
                            payload = serialize_dataframe_for_supabase(pd.DataFrame([u_data]))[0]
                            sync_client.table(TABLE_NAME).update(payload).eq(primary_key, pid).execute()
                            success_cnt += 1
                            st.session_state.pending_updates.pop(pid, None)
                        except Exception as e:
                            errors.append(f"UPDATE {pid}: {e}")

                    # 2. Execute Inserts
                    if st.session_state.pending_inserts and can_add:
                        try:
                            payloads = serialize_dataframe_for_supabase(pd.DataFrame(st.session_state.pending_inserts))
                            sync_client.table(TABLE_NAME).insert(payloads).execute()
                            success_cnt += len(payloads)
                            st.session_state.pending_inserts = []
                        except Exception as e:
                            errors.append(f"INSERT Error: {e}")

                    # 3. Execute Batch Deletes
                    if st.session_state.pending_deletes and can_delete:
                        try:
                            del_list = list(st.session_state.pending_deletes)
                            sync_client.table(TABLE_NAME).delete().in_(primary_key, del_list).execute()
                            success_cnt += len(del_list)
                            st.session_state.pending_deletes.clear()
                        except Exception as e:
                            errors.append(f"DELETE Error: {e}")

                    if success_cnt:
                        fetch_full_table.clear()
                        get_unique_values.clear()
                        st.session_state.grid_version += 1
                        st.success(f"Successfully synced {success_cnt} operation(s)!")
                        if not errors:
                            st.rerun()
                    for err in errors:
                        st.error(err)

            with b_disc:
                if st.button("↩️ Discard Pending Changes", use_container_width=True):
                    clear_pending_changes()
                    st.rerun()


# ------------------------------------------------------------
# TAB 3: BULK EXCEL UPLOAD, ALIGNMENT & RECONCILIATION
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("📤 Bulk Excel Upload & Database Synchronization")
    st.caption("Upload an Excel workbook, automatically align columns, validate data types, compare differences, and batch-upsert to Supabase.")

    if not can_edit:
        st.warning("You need `editor` or `admin` permissions to run bulk synchronizations.")

    uploaded_file = st.file_uploader("Upload Excel File (`.xlsx`, `.xlsm`)", type=["xlsx", "xlsm"])

    if uploaded_file:
        excel_obj = pd.ExcelFile(uploaded_file)
        sheet_names = excel_obj.sheet_names
        default_idx = sheet_names.index("YgnTBPro") if "YgnTBPro" in sheet_names else 0

        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            selected_sheet = st.selectbox("Select Worksheet", sheet_names, index=default_idx)
        with col_s2:
            sim_threshold = st.slider("Fuzzy Column Match Threshold (%)", min_value=10.0, max_value=100.0, value=80.0, step=5.0)

        @st.cache_data(show_spinner=False)
        def load_and_prep_excel(file_bytes, sheet, threshold):
            df_raw_xl = pd.read_excel(file_bytes, sheet_name=sheet)
            df_aligned, df_matching, df_unmatching = align_dataframe_columns(df_raw_xl, PRIMARY_COLUMNS, threshold=threshold)
            df_clean, df_error = clean_and_assign_dtypes(df_aligned)
            return df_raw_xl, df_clean, df_matching, df_unmatching, df_error

        with st.spinner("Aligning schema and validating data types..."):
            df_raw_xl, df_clean, df_matching, df_unmatching, df_error = load_and_prep_excel(
                uploaded_file, selected_sheet, sim_threshold
            )

        # 1. Schema Alignment & Type Diagnostics
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Excel Rows", f"{len(df_clean):,}")
        p2.metric("Matched Columns", f"{len(df_matching)} / {len(PRIMARY_COLUMNS)}")
        dup_xl = df_clean["PatientID"].dropna().duplicated().sum() if "PatientID" in df_clean.columns else 0
        p3.metric("Duplicate PatientIDs (Excel)", f"{dup_xl}")
        p4.metric("Type Conversion Errors", f"{len(df_error.columns)}")

        with st.expander("🔍 View Column Alignment & Unparseable Type Errors", expanded=not df_error.empty):
            ac1, ac2 = st.columns(2)
            with ac1:
                st.write("**Matched Columns (Sorted by Lowest Similarity)**")
                st.dataframe(df_matching, use_container_width=True, height=220)
            with ac2:
                st.write("**Unmatched Primary vs. Dropped Excel Columns**")
                st.dataframe(df_unmatching, use_container_width=True, height=220)
            if not df_error.empty:
                st.error("⚠️ Unparseable values were coerced to `None`/`NaT`. Please review below:")
                st.dataframe(df_error, use_container_width=True)

        st.divider()

        # 2. Database Reconciliation Diff
        if st.button("⚖️ Compare Uploaded Excel with Live Supabase Table", use_container_width=True):
            with st.spinner("Fetching full Supabase table and computing vectorized diff..."):
                df_supa_all = fetch_full_table(TABLE_NAME, access_token=token)
                st.session_state["reconcile_log"] = compare_data_fast(df_clean, df_supa_all, id_col="PatientID")

        if "reconcile_log" in st.session_state:
            log_df = st.session_state["reconcile_log"]
            if log_df.empty:
                st.success("🎉 Excel file and Supabase table are 100% identical!")
            else:
                st.subheader(f"📊 Reconciliation Summary ({len(log_df):,} Differences)")
                summary_counts = log_df["ColumnName"].value_counts().reset_index()
                summary_counts.columns = ["Discrepancy / Column", "Count"]

                rc1, rc2 = st.columns([1, 2])
                with rc1:
                    st.dataframe(summary_counts, use_container_width=True, height=300)
                with rc2:
                    st.dataframe(log_df, use_container_width=True, height=300)

                # Extract IDs present in Supabase but missing in Excel
                not_in_excel_ids = log_df.loc[log_df["ColumnName"] == "_Not in Excel", "PatientID"].dropna().unique().tolist()

                st.divider()
                st.subheader("🚀 Execute Bulk Database Actions")
                act_col1, act_col2 = st.columns(2)

                with act_col1:
                    if st.button(
                        f"📦 Batch Upsert All {len(df_clean):,} Excel Rows to Supabase",
                        type="primary",
                        disabled=not can_edit,
                        use_container_width=True,
                    ):
                        sync_client = get_user_client()
                        records = serialize_dataframe_for_supabase(df_clean, DATE_COLUMNS)
                        batch_size = 500
                        total_batches = (len(records) + batch_size - 1) // batch_size
                        prog = st.progress(0, text="Starting batch upsert...")

                        try:
                            for idx, i in enumerate(range(0, len(records), batch_size)):
                                chunk = records[i : i + batch_size]
                                sync_client.table(TABLE_NAME).upsert(chunk, on_conflict="PatientID").execute()
                                prog.progress((idx + 1) / total_batches, text=f"Upserted batch {idx + 1}/{total_batches}...")

                            fetch_full_table.clear()
                            get_unique_values.clear()
                            st.success(f"✅ Successfully upserted {len(records):,} records into `{TABLE_NAME}`!")
                        except Exception as e:
                            st.error(f"❌ Batch Upsert Failed: {e}")

                with act_col2:
                    if not_in_excel_ids:
                        if st.button(
                            f"🗑️ Delete {len(not_in_excel_ids)} Orphaned Row(s) (`_Not in Excel`)",
                            disabled=not can_delete,
                            use_container_width=True,
                        ):
                            sync_client = get_user_client()
                            try:
                                for i in range(0, len(not_in_excel_ids), 200):
                                    batch_ids = not_in_excel_ids[i : i + 200]
                                    sync_client.table(TABLE_NAME).delete().in_("PatientID", batch_ids).execute()
                                fetch_full_table.clear()
                                get_unique_values.clear()
                                st.success(f"✅ Deleted {len(not_in_excel_ids)} record(s) from `{TABLE_NAME}`.")
                            except Exception as e:
                                st.error(f"❌ Delete Failed: {e}")
                    else:
                        st.info("No orphaned records (`_Not in Excel`) to delete.")
