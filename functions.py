import io
import math
import json
import requests
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import requests.packages.urllib3.util.connection as urllib3_conn

# ============================================================
# SCHEMA & COLUMN DEFINITIONS
# ============================================================

ALT_CLOUDFLARE_IPS = ["104.16.132.229", "104.18.32.7", "172.67.74.135"]

PRIMARY_COLUMNS = list(dict.fromkeys([
    'Date', 'Team', 'StateRegionName', 'Tsp', 'WardVillage', 'WardCode', 
    'Approach', 'Visit_no', 'Sr_No', 'ReportingPeriod', 'RegistrarDate', 
    'TownshipTBRegNumber', 'PatientID', 'Name', 'Nationality', 'Age', 
    'Sex', 'Address', 'Ptstsp', 'PhoneNo', 'Origin', 'VOL', 'Referralfor', 
    'VolunteerName', 'Organization', 'Cough', 'Fever', 'Wtloss', 'Nightsweat', 
    'Haemoptysis', 'Chestpain', 'Fatigue', 'Neckglands', 'TBcontact', 
    'MDRTBcontact', 'TBTreatmenthistory', 'Transferin', 'TypeofPatients', 
    'TypeofDisease', 'SpecifyifEPSiteofEPTB', 'Smoking', 'Reasonforexamination', 
    'TypeofPatient', 'PublicHealthCare1', 'TypeofPatient1', 'DM1', 'RBS', 
    'HbA1c', 'HT1', 'Systolic', 'Diastolic', 'DMHT1', 'RTIAVI1', 
    'Generalweakness1', 'Other1', 'HIVStatus', 'CPT', 'ART', 'Cxrr', 
    'CXRresult', 'Cxr_others', 'Sputum_request', 'Micror', 
    'Sputummicroscopyresult', 'Genexpertrequested', 'GeneXpertresult', 
    'XDRResult', 'Bact_status', 'Case', 'Treatmentreferral', 
    'OLDTYPEOFTBTREATMENT', 'TreatmentRegimen', 'Modifiedregimen', 
    'Placeforreferral', 'Ref1', 'CultureDSTresult', 'Microscope_Result_2nd_mnth', 
    'Xpert_Result_2nd_mnth', 'Microscope_Result_3rd_mnth', 'Xpert_Result_3rd_mnth', 
    'Microscope_Result_5th_mnth', 'Xpert_Result_5th_mnth', 'EndofTxMicroscopeResult', 
    'EndofTxXRayResult', 'EndofTxXpertResult', 'TreatmentOutcome1211', 
    'Tx_Outcome_Date', 'DOTvolName111', 'VolunteerGender111', 
    'VolunteerOrganization111', 'ContactInvestigation111', 'DOTSupervision111', 
    'DOTStartedDate111', 'DOTsupervisiontillTreatmentComp111', 'Seeing1', 
    'Hearing1', 'Walking1', 'Cognition1', 'Selfcare1', 'Communication1', 
    'Disability1', 'Remark', 'MonthDiagnosis11', 'Location11', 'EPI11', 
    'TB11', 'Xray2ndReading11', 'XrayRemark11', 'ReportingTsp11', 
    'CXRresult211', 'XrayEQA11', 'ReportingYear11', 'TPTregno', 
    'RegnoofindexTBcase', 'TPTregistrationdate', 'TPTregimen', 'DrugDose', 
    'Drugpickup1', 'Date_drug_pickup2', 'Date_drug_pickup_3', 
    'Date_drug_pickup_4', 'Date_drug_pickup_5', 'Date_drug_pickup_6', 
    'Date_drug_pickup_7', 'Date_drug_pickup_8', 'Date_drug_pickup_9', 
    'TPTdiscontinuationdate', 'TPTTreatmentOutcome', 'OutcomeDate', 
    'Recstatus', 'UniqueKey', 'TypeofTBTreatment', 'InitialStartedDate'
]))

DATE_COLUMNS = [
    'Date', 'RegistrarDate', 'Tx_Outcome_Date', 'DOTStartedDate111', 
    'TPTregistrationdate', 'Drugpickup1', 'Date_drug_pickup2', 
    'Date_drug_pickup_3', 'Date_drug_pickup_4', 'Date_drug_pickup_5', 
    'Date_drug_pickup_6', 'Date_drug_pickup_7', 'Date_drug_pickup_8', 
    'Date_drug_pickup_9', 'TPTdiscontinuationdate', 'OutcomeDate', 
    'CPT', 'ART', 'InitialStartedDate'
]

STRING_COLUMNS = [
    'StateRegionName', 'Tsp', 'WardVillage', 'WardCode', 'Approach', 'Ptstsp', 
    'Nationality', 'Organization', 'Transferin', 'TypeofPatients', 'TypeofDisease', 
    'HIVStatus', 'XDRResult', 'OLDTYPEOFTBTREATMENT', 'Modifiedregimen', 
    'CultureDSTresult', 'Microscope_Result_2nd_mnth', 'Xpert_Result_2nd_mnth', 
    'Microscope_Result_3rd_mnth', 'Xpert_Result_3rd_mnth', 'Microscope_Result_5th_mnth', 
    'Xpert_Result_5th_mnth', 'EndofTxMicroscopeResult', 'EndofTxXRayResult', 
    'EndofTxXpertResult', 'TreatmentOutcome1211', 'VolunteerGender111', 
    'VolunteerOrganization111', 'ContactInvestigation111', 'DOTSupervision111', 
    'DOTsupervisiontillTreatmentComp111', 'MonthDiagnosis11', 'Location11', 
    'EPI11', 'TB11', 'XrayRemark11', 'ReportingTsp11', 'XrayEQA11', 'TPTregimen', 
    'TPTTreatmentOutcome', 'ReportingPeriod', 'TownshipTBRegNumber', 'PatientID', 
    'Name', 'Address', 'PhoneNo', 'VolunteerName', 'SpecifyifEPSiteofEPTB', 
    'Other1', 'Cxr_others', 'Ref1', 'DOTvolName111', 'Remark', 'TPTregno', 
    'RegnoofindexTBcase', 'DrugDose', 'Disability1', 'Recstatus'
]

INT_COLUMNS = [
    'Team', 'Visit_no', 'Sr_No', 'Age', 'Sex', 'Cough', 'Fever', 'Wtloss', 
    'Nightsweat', 'Haemoptysis', 'Chestpain', 'Fatigue', 'Neckglands', 
    'TBcontact', 'MDRTBcontact', 'TBTreatmenthistory', 'Smoking', 'VOL', 
    'PublicHealthCare1', 'ReportingYear11', 'UniqueKey', 'Referralfor', 
    'Reasonforexamination', 'TypeofPatient', 'TypeofPatient1', 'DM1', 'HT1', 
    'DMHT1', 'RTIAVI1', 'Generalweakness1', 'Cxrr', 'CXRresult', 'Sputum_request', 
    'Micror', 'Sputummicroscopyresult', 'Genexpertrequested', 'GeneXpertresult', 
    'Bact_status', 'Case', 'Treatmentreferral', 'TreatmentRegimen', 
    'Placeforreferral', 'Seeing1', 'Hearing1', 'Walking1', 'Cognition1', 
    'Selfcare1', 'Communication1', 'Xray2ndReading11', 'CXRresult211', 
    'TypeofTBTreatment', 'Origin'
]

FLOAT_COLUMNS = ['RBS', 'HbA1c', 'Systolic', 'Diastolic']


# ============================================================
# NETWORK & SUPABASE PAGINATED FETCH/UPSERT
# ============================================================

def _create_censorship_resistant_session(host_domain: str, alt_ip: str) -> requests.Session:
    orig_create_connection = urllib3_conn.create_connection
    def patched_create_connection(address, *args, **kwargs):
        host, port = address
        if host == host_domain:
            host = alt_ip
        return orig_create_connection((host, port), *args, **kwargs)
    session = requests.Session()
    urllib3_conn.create_connection = patched_create_connection
    return session


def get_working_session(url: str, headers: dict) -> requests.Session:
    host_domain = url.replace("https://", "").replace("http://", "").split("/")[0]
    for alt_ip in [None] + ALT_CLOUDFLARE_IPS:
        try:
            test_session = requests.Session() if alt_ip is None else _create_censorship_resistant_session(host_domain, alt_ip)
            check_res = test_session.get(f"{url.rstrip('/')}/rest/v1/", headers=headers, timeout=5)
            if check_res.status_code < 500:
                return test_session
        except requests.exceptions.RequestException:
            continue
    return requests.Session()


def functionGetDataFromTable(tableName: str, url: str, key: str, access_token: str = None, page_size: int = 1000) -> pd.DataFrame:
    endpoint = f"{url.rstrip('/')}/rest/v1/{tableName}?select=*"
    auth_token = access_token if access_token else key.strip()
    headers = {
        "apikey": key.strip(),
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }
    session = get_working_session(url, headers)
    all_data = []
    start_index = 0

    while True:
        page_headers = headers.copy()
        page_headers["Range"] = f"{start_index}-{start_index + page_size - 1}"
        response = session.get(endpoint, headers=page_headers, timeout=20)
        response.raise_for_status()
        chunk = response.json()
        if not chunk:
            break
        all_data.extend(chunk)
        if len(chunk) < page_size:
            break
        start_index += page_size

    return pd.DataFrame(all_data)


def serialize_dataframe_for_supabase(df: pd.DataFrame, date_columns: list = DATE_COLUMNS) -> list[dict]:
    df_out = df.copy()
    for col in date_columns:
        if col in df_out.columns:
            df_out[col] = pd.to_datetime(df_out[col], errors="coerce").dt.strftime("%Y-%m-%d")

    records = df_out.to_dict(orient="records")
    clean_records = []
    for record in records:
        clean_record = {}
        for k, v in record.items():
            if pd.isna(v) or v is None or str(v).strip() in ("", "nan", "NaT", "None", "<NA>"):
                clean_record[k] = None
            elif isinstance(v, (pd.Timestamp, pd.Timedelta)):
                clean_record[k] = v.strftime("%Y-%m-%d")
            elif isinstance(v, (float, np.floating)) and float(v).is_integer():
                clean_record[k] = int(v)
            elif isinstance(v, (np.integer,)):
                clean_record[k] = int(v)
            else:
                clean_record[k] = v
        clean_records.append(clean_record)
    return clean_records


# ============================================================
# ALIGNMENT, CLEANING, & FAST VECTORIZED DIFF
# ============================================================

def similarity_score(str1: str, str2: str) -> float:
    return SequenceMatcher(None, str1, str2).ratio() * 100


def align_dataframe_columns(df_recipient: pd.DataFrame, primary_cols: list = PRIMARY_COLUMNS, threshold: float = 80.0):
    df_recipient_aligned = df_recipient.copy()
    recipient_cols = list(df_recipient_aligned.columns)
    
    all_pairs = [
        {'ColumnNamePrimary': p_col, 'ColumnNameRecipient': r_col, 'SimilarityScore': similarity_score(p_col, r_col)}
        for p_col in primary_cols for r_col in recipient_cols
    ]
    all_pairs.sort(key=lambda x: x['SimilarityScore'], reverse=True)
    
    column_mapping, matched_primary, matched_recipient, matching_records = {}, set(), set(), []
    
    for pair in all_pairs:
        p_col, r_col, score = pair['ColumnNamePrimary'], pair['ColumnNameRecipient'], pair['SimilarityScore']
        if score < threshold:
            break
        if p_col not in matched_primary and r_col not in matched_recipient:
            matched_primary.add(p_col)
            matched_recipient.add(r_col)
            column_mapping[r_col] = p_col
            matching_records.append({
                'ColumnNamePrimary': p_col,
                'ColumnNameRecipient': r_col,
                'SimilarityScore': round(score, 2)
            })

    df_matching = pd.DataFrame(matching_records, columns=['ColumnNamePrimary', 'ColumnNameRecipient', 'SimilarityScore'])
    if not df_matching.empty:
        df_matching = df_matching.sort_values(by='SimilarityScore', ascending=True).reset_index(drop=True)
    
    unmatched_recipient_cols = [c for c in recipient_cols if c not in matched_recipient]
    missing_primary_cols = [c for c in primary_cols if c not in matched_primary]
    
    df_unmatching = pd.DataFrame({
        'ColumnNamePrimary': pd.Series(missing_primary_cols, dtype='object'),
        'ColumnNameRecipient': pd.Series(unmatched_recipient_cols, dtype='object')
    })
    
    df_recipient_aligned.rename(columns=column_mapping, inplace=True)
    df_recipient_aligned = df_recipient_aligned.loc[:, ~df_recipient_aligned.columns.duplicated()].copy()
    
    for m_col in missing_primary_cols:
        df_recipient_aligned[m_col] = None
        
    df_recipient_aligned = df_recipient_aligned.reindex(columns=primary_cols)
    return df_recipient_aligned, df_matching, df_unmatching


def clean_and_assign_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    errors_dict = {}

    for col in DATE_COLUMNS:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors='coerce')
            invalid_mask = df[col].notna() & parsed.isna()
            if invalid_mask.any():
                errors_dict[f"{col} (date)"] = df.loc[invalid_mask, col].unique().tolist()
            df[col] = parsed

    for col in STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    for col in INT_COLUMNS:
        if col in df.columns:
            parsed = pd.to_numeric(df[col], errors='coerce')
            invalid_mask = df[col].notna() & parsed.isna()
            if invalid_mask.any():
                errors_dict[f"{col} (int)"] = df.loc[invalid_mask, col].unique().tolist()
            df[col] = parsed.astype('Int64')

    for col in FLOAT_COLUMNS:
        if col in df.columns:
            parsed = pd.to_numeric(df[col], errors='coerce')
            invalid_mask = df[col].notna() & parsed.isna()
            if invalid_mask.any():
                errors_dict[f"{col} (float)"] = df.loc[invalid_mask, col].unique().tolist()
            df[col] = parsed.astype('float64')

    df_error = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in errors_dict.items()])) if errors_dict else pd.DataFrame()
    return df, df_error


def _vectorized_normalize_series(s: pd.Series) -> pd.Series:
    """Fast vectorized string normalization to replace slow row-by-row loops."""
    s_str = s.astype(str).str.strip()
    null_mask = s.isna() | s_str.str.lower().isin(["", "nan", "nat", "none", "null", "<na>"])
    s_clean = s_str.str.replace(r"\.0$", "", regex=True)
    return s_clean.mask(null_mask, None)


def compare_data_fast(dfExcel: pd.DataFrame, dfSupabase: pd.DataFrame, id_col: str = "PatientID") -> pd.DataFrame:
    records = []
    dfE = dfExcel.copy()
    dfS = dfSupabase.copy()

    if id_col not in dfE.columns or id_col not in dfS.columns:
        return pd.DataFrame(columns=["PatientID", "ColumnName", "Value_Excel", "Value_Supabase"])

    dfE[id_col] = dfE[id_col].astype(str).str.strip()
    dfS[id_col] = dfS[id_col].astype(str).str.strip()

    for col in DATE_COLUMNS:
        if col in dfE.columns:
            dfE[col] = pd.to_datetime(dfE[col], errors="coerce").dt.strftime("%Y-%m-%d")
        if col in dfS.columns:
            dfS[col] = pd.to_datetime(dfS[col], errors="coerce").dt.strftime("%Y-%m-%d")

    ids_E = set(dfE[id_col].dropna().unique()) - {"", "nan", "None", "<NA>"}
    ids_S = set(dfS[id_col].dropna().unique()) - {"", "nan", "None", "<NA>"}

    for pid in sorted(ids_E - ids_S):
        records.append([pid, "_Not in Supabase", pid, None])
    for pid in sorted(ids_S - ids_E):
        records.append([pid, "_Not in Excel", None, pid])

    common_cols = [c for c in dfE.columns if c in dfS.columns and c not in {id_col, "updated_at"}]

    # Deduplicate on ID before merging for cell comparison
    dfE_unique = dfE.drop_duplicates(subset=[id_col]).set_index(id_col)
    dfS_unique = dfS.drop_duplicates(subset=[id_col]).set_index(id_col)
    common_ids = sorted(list(ids_E & ids_S))

    if common_ids and common_cols:
        subE = dfE_unique.loc[common_ids, common_cols]
        subS = dfS_unique.loc[common_ids, common_cols]

        for col in common_cols:
            normE = _vectorized_normalize_series(subE[col])
            normS = _vectorized_normalize_series(subS[col])
            diff_mask = (normE != normS) & ~(normE.isna() & normS.isna())
            if diff_mask.any():
                diff_ids = diff_mask[diff_mask].index
                for pid in diff_ids:
                    records.append([pid, col, subE.at[pid, col], subS.at[pid, col]])

    return pd.DataFrame(records, columns=["PatientID", "ColumnName", "Value_Excel", "Value_Supabase"])

# Add to the bottom of functions.py

def fetch_supabase_by_ids(
    tableName: str,
    id_list: list,
    url: str,
    key: str,
    access_token: str = None,
    id_col: str = "PatientID",
    chunk_size: int = 300,
) -> pd.DataFrame:
    """Fetches only the rows matching the PatientIDs in the uploaded Excel file."""
    if not id_list:
        return pd.DataFrame()

    auth_token = access_token if access_token else key.strip()
    headers = {
        "apikey": key.strip(),
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }
    session = get_working_session(url, headers)
    all_rows = []
    clean_ids = [str(x).strip() for x in id_list if pd.notna(x) and str(x).strip()]

    for i in range(0, len(clean_ids), chunk_size):
        batch = clean_ids[i : i + chunk_size]
        formatted_ids = ",".join(batch)
        endpoint = f"{url.rstrip('/')}/rest/v1/{tableName}?{id_col}=in.({formatted_ids})&select=*"
        res = session.get(endpoint, headers=headers, timeout=20)
        res.raise_for_status()
        data = res.json()
        if data:
            all_rows.extend(data)

    return pd.DataFrame(all_rows)


def fetch_all_patient_ids_only(
    tableName: str, url: str, key: str, access_token: str = None, id_col: str = "PatientID", page_size: int = 5000
) -> set:
    """Lightweight fetch of ONLY the PatientID column (~98% less RAM than SELECT *)."""
    endpoint = f"{url.rstrip('/')}/rest/v1/{tableName}?select={id_col}"
    auth_token = access_token if access_token else key.strip()
    headers = {
        "apikey": key.strip(),
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }
    session = get_working_session(url, headers)
    all_ids = set()
    start_index = 0

    while True:
        page_headers = headers.copy()
        page_headers["Range"] = f"{start_index}-{start_index + page_size - 1}"
        res = session.get(endpoint, headers=page_headers, timeout=15)
        res.raise_for_status()
        chunk = res.json()
        if not chunk:
            break
        for row in chunk:
            val = row.get(id_col)
            if val is not None and str(val).strip():
                all_ids.add(str(val).strip())
        if len(chunk) < page_size:
            break
        start_index += page_size

    return all_ids
