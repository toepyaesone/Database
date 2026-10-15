import os
import math
from datetime import date, timedelta
import pandas as pd
import streamlit as st
from supabase import Client, create_client

from functions import (
    PRIMARY_COLUMNS,
    DATE_COLUMNS,
    functionGetDataFromTable,
    fetch_supabase_by_ids,
    fetch_all_patient_ids_only,
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

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabase credentials missing! Configure `SUPABASE_URL_ygntbpro` and `SUPABASE_KEY_ygntbpro` in Secrets.")
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
    "assigned_tsp": None,
    "assigned_team": None,
    "grid_version": 0,
    "filter_version": 0,
    "pending_updates": {},
    "pending_inserts": [],
    "pending_deletes": set(),
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
            .select("*")
            .eq("user_id", response.session.user.id)
            .limit(1)
            .execute()
        )
        if role_result.data:
            row_info = role_result.data[0]
            st.session_state.user_role = row_info.get("role", "viewer")
            st.session_state.assigned_tsp = row_info.get("assigned_tsp")
            st.session_state.assigned_team = row_info.get("assigned_team")
        else:
            st.session_state.user_role = "viewer"

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
st.sidebar.markdown(f"**Role:** `{user_role.upper()}`")
if st.session_state.assigned_tsp:
    st.sidebar.markdown(f"**Assigned Tsp:** `{st.session_state.assigned_tsp}`")
if st.session_state.assigned_team:
    st.sidebar.markdown(f"**Assigned Team:** `{st.session_state.assigned_team}`")
if st.sidebar.button("Logout", use_container_width=True):
    logout_user()


# ============================================================
# 4. CACHED SCHEMA & FAST FILTER VIEW FETCHERS
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

@st.cache_data(ttl=600, show_spinner=False)
def get_table_columns(table_name: str) -> list:
    try:
        res = get_user_client().table(table_name).select("*").limit(1).execute()
        if res.data:
            return list(res.data[0].keys())
    except Exception:
        pass
    return PRIMARY_COLUMNS

@st.cache_data(ttl=300, show_spinner=False)
def get_all_filter_options() -> dict:
    """Single <1KB query to the PostgreSQL View `ygntbpro_filter_options`."""
    default_opts = {
        "visit_no_list": [],
        "sr_no_list": [],
        "team_list": [],
        "tsp_list": [],
        "approach_list": [],
        "case_list": [],
    }
    try:
        res = get_user_client().table("ygntbpro_filter_options").select("*").limit(1).execute()
        if res.data:
            row = res.data[0]
            return {k: (row.get(k) or []) for k in default_opts}
    except Exception:
        pass
    return default_opts


# ============================================================
# 5. VALIDATED MODAL USERFORM (@st.dialog)
# ============================================================

def safe_date(val):
    if pd.isna(val) or val in (None, "", "None", "NaT", "nan"):
        return None
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None

def safe_int(val, default=0):
    if pd.isna(val) or val in (None, "", "None", "nan", "<NA>"):
        return default
    try:
        return int(float(val))
    except Exception:
        return default

def safe_float(val, default=0.0):
    if pd.isna(val) or val in (None, "", "None", "nan", "<NA>"):
        return default
    try:
        return float(val)
    except Exception:
        return default

def safe_str(val):
    if pd.isna(val) or val in (None, "None", "nan", "<NA>"):
        return ""
    return str(val).strip()


@st.dialog("🩺 Patient Clinical Record UserForm", width="large")
def open_patient_userform(mode: str, patient_row: dict, primary_key: str):
    is_new = (mode == "create")
    pid_val = safe_str(patient_row.get(primary_key))

    if is_new:
        st.info("➕ **Creating New Patient Record** (Admin Only)")
    else:
        st.caption(f"Editing Record — **{primary_key}: `{pid_val}`** | **Name:** `{safe_str(patient_row.get('Name'))}`")

    with st.form("clinical_userform", clear_on_submit=False):
        ftabs = st.tabs([
            "1. ID & Demographics",
            "2. Symptoms & Risk",
            "3. Diagnostics & Lab",
            "4. Regimen & Outcome",
            "5. TPT & Disability",
        ])

        with ftabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            f_pid = c1.text_input("Patient ID *", value=pid_val, disabled=(not is_new and user_role != "admin"))
            f_date = c2.date_input("Screening Date *", value=safe_date(patient_row.get("Date")) or date.today())
            f_reg_date = c3.date_input("Registrar Date", value=safe_date(patient_row.get("RegistrarDate")))
            f_team = c4.number_input("Team No", min_value=0, max_value=99, value=safe_int(patient_row.get("Team")))

            d1, d2, d3, d4 = st.columns(4)
            f_name = d1.text_input("Patient Name *", value=safe_str(patient_row.get("Name")))
            f_age = d2.number_input("Age (Years) *", min_value=0, max_value=120, value=safe_int(patient_row.get("Age")))
            current_sex = safe_int(patient_row.get("Sex"), default=1)
            f_sex = d3.selectbox("Sex *", options=[1, 2], index=0 if current_sex == 1 else 1, format_func=lambda x: "1 - Male" if x == 1 else "2 - Female")
            f_phone = d4.text_input("Phone No", value=safe_str(patient_row.get("PhoneNo")))

            loc1, loc2, loc3, loc4 = st.columns(4)
            f_sr = loc1.text_input("State/Region", value=safe_str(patient_row.get("StateRegionName")) or "Yangon")
            f_tsp = loc2.text_input("Township (Tsp) *", value=safe_str(patient_row.get("Tsp")))
            f_ward = loc3.text_input("Ward/Village", value=safe_str(patient_row.get("WardVillage")))
            f_approach = loc4.text_input("Approach", value=safe_str(patient_row.get("Approach")))

            f_address = st.text_input("Full Address", value=safe_str(patient_row.get("Address")))

        with ftabs[1]:
            st.markdown("**Clinical Symptoms (0 = No, 1 = Yes)**")
            s1, s2, s3, s4 = st.columns(4)
            f_cough = s1.selectbox("Cough", [0, 1], index=1 if safe_int(patient_row.get("Cough")) == 1 else 0)
            f_fever = s2.selectbox("Fever", [0, 1], index=1 if safe_int(patient_row.get("Fever")) == 1 else 0)
            f_wtloss = s3.selectbox("Weight Loss", [0, 1], index=1 if safe_int(patient_row.get("Wtloss")) == 1 else 0)
            f_nightsweat = s4.selectbox("Night Sweat", [0, 1], index=1 if safe_int(patient_row.get("Nightsweat")) == 1 else 0)

            s5, s6, s7, s8 = st.columns(4)
            f_haemo = s5.selectbox("Haemoptysis", [0, 1], index=1 if safe_int(patient_row.get("Haemoptysis")) == 1 else 0)
            f_chest = s6.selectbox("Chest Pain", [0, 1], index=1 if safe_int(patient_row.get("Chestpain")) == 1 else 0)
            f_tbcontact = s7.selectbox("TB Contact", [0, 1], index=1 if safe_int(patient_row.get("TBcontact")) == 1 else 0)
            f_smoking = s8.selectbox("Smoking", [0, 1], index=1 if safe_int(patient_row.get("Smoking")) == 1 else 0)

            st.divider()
            st.markdown("**Comorbidities & Vitals**")
            m1, m2, m3, m4 = st.columns(4)
            f_rbs = m1.number_input("RBS (mg/dL)", min_value=0.0, max_value=1000.0, step=1.0, value=safe_float(patient_row.get("RBS")))
            f_hba1c = m2.number_input("HbA1c (%)", min_value=0.0, max_value=25.0, step=0.1, value=safe_float(patient_row.get("HbA1c")))
            f_sys = m3.number_input("Systolic BP", min_value=0.0, max_value=300.0, step=1.0, value=safe_float(patient_row.get("Systolic")))
            f_dia = m4.number_input("Diastolic BP", min_value=0.0, max_value=200.0, step=1.0, value=safe_float(patient_row.get("Diastolic")))

        with ftabs[2]:
            l1, l2, l3 = st.columns(3)
            f_cxrr = l1.number_input("CXR Requested (Code)", min_value=0, max_value=9, value=safe_int(patient_row.get("Cxrr")))
            f_cxr_res = l2.number_input("CXR Result (Code)", min_value=0, max_value=9, value=safe_int(patient_row.get("CXRresult")))
            f_hiv = l3.text_input("HIV Status", value=safe_str(patient_row.get("HIVStatus")))

            l4, l5, l6 = st.columns(3)
            f_sputum_req = l4.number_input("Sputum Requested", min_value=0, max_value=9, value=safe_int(patient_row.get("Sputum_request")))
            f_sputum_res = l5.number_input("Microscopy Result", min_value=0, max_value=9, value=safe_int(patient_row.get("Sputummicroscopyresult")))
            f_xpert_res = l6.number_input("GeneXpert Result", min_value=0, max_value=9, value=safe_int(patient_row.get("GeneXpertresult")))

            l7, l8, l9 = st.columns(3)
            f_bact = l7.number_input("Bacteriological Status", min_value=0, max_value=9, value=safe_int(patient_row.get("Bact_status")))
            f_case = l8.number_input("Case Classification", min_value=0, max_value=9, value=safe_int(patient_row.get("Case")))
            f_type_dis = l9.text_input("Type of Disease", value=safe_str(patient_row.get("TypeofDisease")))

        with ftabs[3]:
            r1, r2, r3 = st.columns(3)
            f_tb_reg_no = r1.text_input("Township TB Reg Number", value=safe_str(patient_row.get("TownshipTBRegNumber")))
            f_regimen = r2.number_input("Treatment Regimen (Code)", min_value=0, max_value=99, value=safe_int(patient_row.get("TreatmentRegimen")))
            f_init_date = r3.date_input("Initial Started Date", value=safe_date(patient_row.get("InitialStartedDate")))

            r4, r5, r6 = st.columns(3)
            f_outcome = r4.text_input("Treatment Outcome", value=safe_str(patient_row.get("TreatmentOutcome1211")))
            f_outcome_date = r5.date_input("Tx Outcome Date", value=safe_date(patient_row.get("Tx_Outcome_Date")))
            f_dot_vol = r6.text_input("DOT Volunteer Name", value=safe_str(patient_row.get("DOTvolName111")))

            f_remark = st.text_area("Clinical Remark", value=safe_str(patient_row.get("Remark")))

        with ftabs[4]:
            t1, t2, t3 = st.columns(3)
            f_tpt_regno = t1.text_input("TPT Reg No", value=safe_str(patient_row.get("TPTregno")))
            f_tpt_date = t2.date_input("TPT Registration Date", value=safe_date(patient_row.get("TPTregistrationdate")))
            f_tpt_regimen = t3.text_input("TPT Regimen", value=safe_str(patient_row.get("TPTregimen")))

            t4, t5, t6 = st.columns(3)
            f_tpt_outcome = t4.text_input("TPT Treatment Outcome", value=safe_str(patient_row.get("TPTTreatmentOutcome")))
            f_tpt_out_date = t5.date_input("TPT Outcome Date", value=safe_date(patient_row.get("OutcomeDate")))
            f_disability = t6.text_input("Disability Status", value=safe_str(patient_row.get("Disability1")))

        submitted = st.form_submit_button(
            "💾 Validate & Commit to Supabase",
            type="primary",
            disabled=not (can_add if is_new else can_edit),
            use_container_width=True,
        )

        if submitted:
            val_errors = []
            if not f_pid.strip():
                val_errors.append("Patient ID is mandatory.")
            if not f_name.strip():
                val_errors.append("Patient Name is mandatory.")
            if not f_tsp.strip():
                val_errors.append("Township (Tsp) is mandatory.")
            if f_outcome_date and f_date and f_outcome_date < f_date:
                val_errors.append("Tx Outcome Date cannot be earlier than Screening Date.")
            if f_tpt_out_date and f_tpt_date and f_tpt_out_date < f_tpt_date:
                val_errors.append("TPT Outcome Date cannot be earlier than TPT Registration Date.")
            if f_sys > 0 and f_dia > 0 and f_dia >= f_sys:
                val_errors.append("Diastolic BP must be lower than Systolic BP.")

            if val_errors:
                for err in val_errors:
                    st.error(f"⚠️ Validation Error: {err}")
            else:
                raw_payload = {
                    primary_key: f_pid.strip(),
                    "Date": f_date,
                    "RegistrarDate": f_reg_date,
                    "Team": f_team,
                    "Name": f_name.strip(),
                    "Age": f_age,
                    "Sex": f_sex,
                    "PhoneNo": f_phone.strip() or None,
                    "StateRegionName": f_sr.strip() or None,
                    "Tsp": f_tsp.strip(),
                    "WardVillage": f_ward.strip() or None,
                    "Approach": f_approach.strip() or None,
                    "Address": f_address.strip() or None,
                    "Cough": f_cough,
                    "Fever": f_fever,
                    "Wtloss": f_wtloss,
                    "Nightsweat": f_nightsweat,
                    "Haemoptysis": f_haemo,
                    "Chestpain": f_chest,
                    "TBcontact": f_tbcontact,
                    "Smoking": f_smoking,
                    "RBS": f_rbs if f_rbs > 0 else None,
                    "HbA1c": f_hba1c if f_hba1c > 0 else None,
                    "Systolic": f_sys if f_sys > 0 else None,
                    "Diastolic": f_dia if f_dia > 0 else None,
                    "Cxrr": f_cxrr,
                    "CXRresult": f_cxr_res,
                    "HIVStatus": f_hiv.strip() or None,
                    "Sputum_request": f_sputum_req,
                    "Sputummicroscopyresult": f_sputum_res,
                    "GeneXpertresult": f_xpert_res,
                    "Bact_status": f_bact,
                    "Case": f_case,
                    "TypeofDisease": f_type_dis.strip() or None,
                    "TownshipTBRegNumber": f_tb_reg_no.strip() or None,
                    "TreatmentRegimen": f_regimen,
                    "InitialStartedDate": f_init_date,
                    "TreatmentOutcome1211": f_outcome.strip() or None,
                    "Tx_Outcome_Date": f_outcome_date,
                    "DOTvolName111": f_dot_vol.strip() or None,
                    "Remark": f_remark.strip() or None,
                    "TPTregno": f_tpt_regno.strip() or None,
                    "TPTregistrationdate": f_tpt_date,
                    "TPTregimen": f_tpt_regimen.strip() or None,
                    "TPTTreatmentOutcome": f_tpt_outcome.strip() or None,
                    "OutcomeDate": f_tpt_out_date,
                    "Disability1": f_disability.strip() or None,
                }

                clean_payload = serialize_dataframe_for_supabase(pd.DataFrame([raw_payload]), DATE_COLUMNS)[0]
                sync_client = get_user_client()

                try:
                    if is_new:
                        sync_client.table(TABLE_NAME).insert(clean_payload).execute()
                    else:
                        sync_client.table(TABLE_NAME).update(clean_payload).eq(primary_key, pid_val).execute()

                    get_all_filter_options.clear()
                    st.session_state.grid_version += 1
                    st.success("✅ Patient record validated and saved to Supabase!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"❌ Database Error: {ex}")


# ============================================================
# 6. GRID DIFF HELPERS
# ============================================================

def values_equal(left, right) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return str(left).strip() == str(right).strip()

def clean_value(value):
    return None if pd.isna(value) else value

def clear_pending_changes():
    st.session_state.pending_updates = {}
    st.session_state.pending_inserts = []
    st.session_state.pending_deletes = set()
    st.session_state.editor_source_df = None
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

    for pid in source_ids:
        if pid not in edited_ids:
            st.session_state.pending_deletes.add(pid)
            st.session_state.pending_updates.pop(pid, None)
        else:
            st.session_state.pending_deletes.discard(pid)

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

    st.session_state.pending_inserts = []
    for _, row in edited.iterrows():
        pid = row.get(primary_key)
        if pd.notna(pid) and str(pid).strip() in source_ids:
            continue
        clean_row = {col: clean_value(val) for col, val in row.items() if col != "updated_at"}
        if any(v not in (None, "") for v in clean_row.values()):
            st.session_state.pending_inserts.append(clean_row)


# ============================================================
# 7. FRAGMENT-ISOLATED APPLICATION TABS
# ============================================================

@st.fragment
def render_explorer_tab():
    st.subheader("💾 Server-Side Paginated Database Explorer")
    c_top1, c_top2, c_top3, c_top4 = st.columns([2, 2, 1, 1])

    with c_top1:
        table_choice = st.selectbox("Select Database Table", [TABLE_NAME, TARGET_TABLE], index=0)
    cols = get_table_columns(table_choice)
    pk_col = resolve_column(cols, "PatientID", "patientid", "id") or cols[0]
    name_col = resolve_column(cols, "Name", "name")
    tsp_col = resolve_column(cols, "Tsp", "tsp")

    with c_top2:
        search_term = st.text_input("🔍 Server-Side Search (PatientID / Name / Tsp)", placeholder="Type to search in PostgreSQL...")
    with c_top3:
        page_size = st.selectbox("Rows / Page", [50, 100, 250, 500], index=1, key="exp_page_size")
    with c_top4:
        page_num = st.number_input("Page #", min_value=1, value=1, step=1, key="exp_page_num")

    selected_cols = st.multiselect(
        "Columns to Fetch & Display (Reduces Bandwidth)",
        options=cols,
        default=cols[:15] if len(cols) > 15 else cols,
    )
    select_str = ",".join(f'"{c}"' for c in selected_cols) if selected_cols else "*"

    client = get_user_client()
    query = client.table(table_choice).select(select_str, count="exact")

    if search_term.strip():
        s = search_term.strip()
        or_clauses = [f"{pk_col}.ilike.%{s}%"]
        if name_col:
            or_clauses.append(f"{name_col}.ilike.%{s}%")
        if tsp_col:
            or_clauses.append(f"{tsp_col}.ilike.%{s}%")
        query = query.or_(",".join(or_clauses))

    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size - 1

    try:
        res = query.order(pk_col, desc=True).range(start_idx, end_idx).execute()
        df_page = pd.DataFrame(res.data or [])
        total_count = res.count or len(df_page)
    except Exception as exc:
        st.error(f"Query Error: {exc}")
        df_page, total_count = pd.DataFrame(), 0

    total_pages = max(1, math.ceil(total_count / page_size))
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Matching Records (DB)", f"{total_count:,}")
    m2.metric("Current Page", f"{page_num} of {total_pages:,}")
    m3.metric("Page RAM Footprint", f"{df_page.memory_usage(deep=True).sum() / 1024:.1f} KB")

    st.dataframe(df_page, use_container_width=True, hide_index=False)

    if not df_page.empty:
        csv_bytes = df_page.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📄 Download Current Page CSV",
            data=csv_bytes,
            file_name=f"{table_choice}_page_{page_num}.csv",
            mime="text/csv",
        )


@st.fragment
def render_editor_tab():
    st.subheader("🩺 Paginated Record Editor & Validated UserForm")
    cols = get_table_columns(TABLE_NAME)
    primary_key = resolve_column(cols, "PatientID", "patientid", "patient_id") or "PatientID"
    date_column = resolve_column(cols, "Date", "date")

    filter_columns = {
        "Visit No": (resolve_column(cols, "visitno", "Visit_no"), "visit_no_list"),
        "SR No": (resolve_column(cols, "srno", "Sr_No"), "sr_no_list"),
        "Team": (resolve_column(cols, "team", "Team"), "team_list"),
        "TSP": (resolve_column(cols, "tsp", "Tsp", "TSP"), "tsp_list"),
        "Approach": (resolve_column(cols, "approach", "Approach"), "approach_list"),
        "Case": (resolve_column(cols, "case", "Case"), "case_list"),
    }

    # Instant 1-query filter options from SQL View
    view_opts = get_all_filter_options()
    fv = st.session_state.filter_version

    fc1, fc2, fc3 = st.columns(3)
    active_filters = {}

    with fc1:
        pid_search = st.text_input("🔎 Search Patient ID (Exact or Prefix)", key=f"pid_s_{fv}")
        for lbl in ["Visit No", "SR No"]:
            col_name, view_key = filter_columns[lbl]
            if col_name:
                active_filters[col_name] = st.multiselect(lbl, view_opts.get(view_key, []), key=f"f_{lbl}_{fv}")

    with fc2:
        for lbl in ["Team", "TSP", "Approach"]:
            col_name, view_key = filter_columns[lbl]
            if col_name:
                active_filters[col_name] = st.multiselect(lbl, view_opts.get(view_key, []), key=f"f_{lbl}_{fv}")

    with fc3:
        col_case, case_key = filter_columns["Case"]
        if col_case:
            active_filters[col_case] = st.multiselect("Case", view_opts.get(case_key, []), key=f"f_Case_{fv}")
        d_col1, d_col2 = st.columns(2)
        date_from = d_col1.date_input("Date From", value=None, key=f"df_{fv}")
        date_to = d_col2.date_input("Date To", value=None, key=f"dt_{fv}")

    # Pagination Bar + Reset
    p_col1, p_col2, p_col3 = st.columns([1, 1, 2])
    with p_col1:
        ed_page_size = st.selectbox("Editor Page Size", [50, 100, 250], index=1, key="ed_ps")
    with p_col2:
        ed_page_num = st.number_input("Editor Page #", min_value=1, value=1, step=1, key="ed_pn")
    with p_col3:
        st.write("")
        if st.button("🔄 Reset Filters & Reload", use_container_width=True):
            st.session_state.filter_version += 1
            st.session_state.grid_version += 1
            st.rerun()

    # Build Paginated Server-Side Query
    client = get_user_client()
    query = client.table(TABLE_NAME).select("*", count="exact")

    if pid_search.strip():
        query = query.ilike(primary_key, f"%{pid_search.strip()}%")
    for c_name, vals in active_filters.items():
        if vals:
            query = query.in_(c_name, vals)
    if date_column and date_from:
        query = query.gte(date_column, date_from.isoformat())
    if date_column and date_to:
        query = query.lt(date_column, (date_to + timedelta(days=1)).isoformat())

    start_idx = (ed_page_num - 1) * ed_page_size
    end_idx = start_idx + ed_page_size - 1

    res = query.order(primary_key, desc=True).range(start_idx, end_idx).execute()
    df_edit_view = pd.DataFrame(res.data or [])
    total_matches = res.count or len(df_edit_view)
    total_ed_pages = max(1, math.ceil(total_matches / ed_page_size))

    # --- USERFORM LAUNCHER BAR ---
    st.divider()
    uf_col1, uf_col2, uf_col3 = st.columns([2, 1, 1])
    with uf_col1:
        pid_options = df_edit_view[primary_key].dropna().astype(str).tolist() if not df_edit_view.empty else []
        selected_pid_for_form = st.selectbox(
            "Select Patient ID (from current page) for Validated UserForm",
            options=pid_options,
            index=0 if pid_options else None,
        )
    with uf_col2:
        st.write("")
        if st.button("📋 Open Patient UserForm", type="primary", disabled=(not selected_pid_for_form or not can_edit), use_container_width=True):
            matched_row = df_edit_view[df_edit_view[primary_key].astype(str) == str(selected_pid_for_form)].iloc[0].to_dict()
            open_patient_userform(mode="edit", patient_row=matched_row, primary_key=primary_key)
    with uf_col3:
        st.write("")
        if st.button("➕ New Patient UserForm", disabled=not can_add, use_container_width=True):
            open_patient_userform(mode="create", patient_row={}, primary_key=primary_key)

    st.divider()

    # --- SPREADSHEET GRID EDITOR ---
    if not df_edit_view.empty:
        st.caption(f"Showing Page **{ed_page_num} of {total_ed_pages:,}** ({len(df_edit_view)} rows loaded out of **{total_matches:,}** total matching records).")
        st.session_state.editor_source_df = df_edit_view.copy()

        protected_for_editor = [primary_key, "UniqueKey", "Sr_No", "updated_at"]
        if user_role == "admin":
            disabled_cols = ["updated_at"] if "updated_at" in df_edit_view.columns else []
        elif user_role == "editor":
            disabled_cols = [c for c in protected_for_editor if c in df_edit_view.columns]
        else:
            disabled_cols = list(df_edit_view.columns)

        edited_df = st.data_editor(
            df_edit_view,
            key=f"editor_grid_{st.session_state.grid_version}_{ed_page_num}",
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic" if can_add else "fixed",
            disabled=disabled_cols,
        )
        if can_edit:
            capture_editor_changes(st.session_state.editor_source_df, edited_df, primary_key)

    # Render Pending Grid Queue
    pending_rows = []
    for pid, chg in st.session_state.pending_updates.items():
        pending_rows.append({"Action": "UPDATE", primary_key: pid, **chg})
    for r in st.session_state.pending_inserts:
        pending_rows.append({"Action": "INSERT", **r})
    for pid in sorted(st.session_state.pending_deletes):
        pending_rows.append({"Action": "DELETE", primary_key: pid})

    if pending_rows:
        st.subheader(f"📝 Pending Grid Changes ({len(pending_rows)})")
        st.dataframe(pd.DataFrame(pending_rows), use_container_width=True)
        b_sync, b_disc = st.columns(2)
        with b_sync:
            if st.button("💾 Commit Grid Changes to Supabase", type="primary", disabled=not can_edit, use_container_width=True):
                sync_client = get_user_client()
                errors, success_cnt = [], 0

                for pid, u_data in list(st.session_state.pending_updates.items()):
                    try:
                        payload = serialize_dataframe_for_supabase(pd.DataFrame([u_data]))[0]
                        sync_client.table(TABLE_NAME).update(payload).eq(primary_key, pid).execute()
                        success_cnt += 1
                        st.session_state.pending_updates.pop(pid, None)
                    except Exception as e:
                        errors.append(f"UPDATE {pid}: {e}")

                if st.session_state.pending_inserts and can_add:
                    try:
                        payloads = serialize_dataframe_for_supabase(pd.DataFrame(st.session_state.pending_inserts))
                        sync_client.table(TABLE_NAME).insert(payloads).execute()
                        success_cnt += len(payloads)
                        st.session_state.pending_inserts = []
                    except Exception as e:
                        errors.append(f"INSERT Error: {e}")

                if st.session_state.pending_deletes and can_delete:
                    try:
                        del_list = list(st.session_state.pending_deletes)
                        sync_client.table(TABLE_NAME).delete().in_(primary_key, del_list).execute()
                        success_cnt += len(del_list)
                        st.session_state.pending_deletes.clear()
                    except Exception as e:
                        errors.append(f"DELETE Error: {e}")

                if success_cnt:
                    get_all_filter_options.clear()
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


@st.fragment
def render_bulk_upload_tab():
    st.subheader("📤 Bulk Excel Upload & Low-Memory Reconciliation")
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
            return df_clean, df_matching, df_unmatching, df_error

        with st.spinner("Aligning schema and validating data types..."):
            df_clean, df_matching, df_unmatching, df_error = load_and_prep_excel(
                uploaded_file, selected_sheet, sim_threshold
            )

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Excel Rows", f"{len(df_clean):,}")
        p2.metric("Matched Columns", f"{len(df_matching)} / {len(PRIMARY_COLUMNS)}")
        dup_xl = df_clean["PatientID"].dropna().duplicated().sum() if "PatientID" in df_clean.columns else 0
        p3.metric("Duplicate PatientIDs (Excel)", f"{dup_xl}")
        p4.metric("Type Conversion Errors", f"{len(df_error.columns)}")

        with st.expander("🔍 View Column Alignment & Unparseable Type Errors", expanded=not df_error.empty):
            ac1, ac2 = st.columns(2)
            with ac1:
                st.write("**Matched Columns**")
                st.dataframe(df_matching, use_container_width=True, height=200)
            with ac2:
                st.write("**Unmatched Primary vs. Dropped Excel Columns**")
                st.dataframe(df_unmatching, use_container_width=True, height=200)
            if not df_error.empty:
                st.error("⚠️ Unparseable values coerced to `None`/`NaT`:")
                st.dataframe(df_error, use_container_width=True)

        st.divider()

        token = getattr(st.session_state.session, "access_token", None)

        if st.button("⚖️ Run Smart Memory-Optimized Reconciliation", use_container_width=True):
            with st.spinner("Fetching matching records and ID index from Supabase..."):
                excel_pids = df_clean["PatientID"].dropna().astype(str).str.strip().unique().tolist()
                # 1. Fetch full rows ONLY for IDs present in the Excel file
                df_supa_matched = fetch_supabase_by_ids(TABLE_NAME, excel_pids, SUPABASE_URL, SUPABASE_KEY, access_token=token)
                # 2. Fetch lightweight ID set for '_Not in Excel' detection
                all_supa_ids = fetch_all_patient_ids_only(TABLE_NAME, SUPABASE_URL, SUPABASE_KEY, access_token=token)

                diff_log = compare_data_fast(df_clean, df_supa_matched, id_col="PatientID")
                # Append any orphaned IDs in Supabase that were not in the Excel upload
                missing_in_xl = sorted(list(all_supa_ids - set(excel_pids)))
                if missing_in_xl:
                    orphan_df = pd.DataFrame(
                        [[pid, "_Not in Excel", None, pid] for pid in missing_in_xl],
                        columns=["PatientID", "ColumnName", "Value_Excel", "Value_Supabase"],
                    )
                    diff_log = pd.concat([diff_log, orphan_df], ignore_index=True)

                st.session_state["reconcile_log"] = diff_log

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
                    st.dataframe(summary_counts, use_container_width=True, height=280)
                with rc2:
                    st.dataframe(log_df, use_container_width=True, height=280)

                # Delta-only upsert list: only rows that are new or have at least 1 changed cell!
                changed_pids = set(
                    log_df.loc[log_df["ColumnName"] != "_Not in Excel", "PatientID"].dropna().astype(str).unique()
                )
                df_delta = df_clean[df_clean["PatientID"].astype(str).str.strip().isin(changed_pids)]
                not_in_excel_ids = log_df.loc[log_df["ColumnName"] == "_Not in Excel", "PatientID"].dropna().unique().tolist()

                st.divider()
                st.subheader("🚀 Execute Bulk Database Actions")
                act_col1, act_col2 = st.columns(2)

                with act_col1:
                    if st.button(
                        f"⚡ Delta Upsert Only Changed/New Rows ({len(df_delta):,} of {len(df_clean):,})",
                        type="primary",
                        disabled=(not can_edit or df_delta.empty),
                        use_container_width=True,
                    ):
                        sync_client = get_user_client()
                        records = serialize_dataframe_for_supabase(df_delta, DATE_COLUMNS)
                        batch_size = 500
                        total_batches = max(1, (len(records) + batch_size - 1) // batch_size)
                        prog = st.progress(0, text="Starting delta batch upsert...")

                        try:
                            for idx, i in enumerate(range(0, len(records), batch_size)):
                                chunk = records[i : i + batch_size]
                                sync_client.table(TABLE_NAME).upsert(chunk, on_conflict="PatientID").execute()
                                prog.progress((idx + 1) / total_batches, text=f"Upserted batch {idx + 1}/{total_batches}...")

                            get_all_filter_options.clear()
                            st.success(f"✅ Synced {len(records):,} modified/new records in seconds!")
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
                                get_all_filter_options.clear()
                                st.success(f"✅ Deleted {len(not_in_excel_ids)} record(s) from `{TABLE_NAME}`.")
                            except Exception as e:
                                st.error(f"❌ Delete Failed: {e}")
                    else:
                        st.info("No orphaned records (`_Not in Excel`) to delete.")


# ============================================================
# 8. MAIN TAB ROUTER
# ============================================================

tabs = st.tabs([
    "💾 Explorer & Search",
    "✏️ Data Editor & UserForm",
    "📤 Bulk Upload & Reconcile (Excel)",
])

with tabs[0]:
    render_explorer_tab()

with tabs[1]:
    render_editor_tab()

with tabs[2]:
    render_bulk_upload_tab()
