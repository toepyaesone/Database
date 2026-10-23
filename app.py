import os
import re
import gc
import difflib
import tempfile
import warnings
from io import BytesIO
from pathlib import Path
from datetime import datetime, date

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

try:
    from rapidfuzz import fuzz, process as rf_process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


# ==============================================================================
# 1. CROSS-PLATFORM CONFIGURATION (WINDOWS, MACOS, IPAD, IPHONE)
# ==============================================================================
st.set_page_config(
    page_title="MATA TB Data Verification Suite (PPM & CBTBC)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="auto",
)

SUPABASE_DEFAULT_TABLE = "ygntbpro"
APP_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Cross-platform month abbreviations (independent of OS C-locale on Windows/macOS/iOS)
MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# Epoch-zero placeholder dates across Unix (1970), Windows Excel (1899/1900), and macOS Excel (1904)
EPOCH_ZERO_DATES = {
    date(1970, 1, 1),
    date(1899, 12, 30),
    date(1899, 12, 31),
    date(1900, 1, 1),
    date(1904, 1, 1),
}

INVALID_DATE_TOKENS = {
    "", "0", "0.0", "nan", "nat", "none", "<na>", "null", "nil", "n/a", "na", "-", "--",
    "1970-01-01", "1970-01-01 00:00:00", "01-jan-1970", "1900-01-00", "1900-01-01",
    "1899-12-30", "1904-01-01", "00/00/0000", "00-00-0000",
}


def parse_valid_datetime(val) -> pd.Timestamp | type(pd.NaT):
    """
    Cross-platform safe datetime parser (Windows, macOS, Linux, iPadOS, iOS).
    Returns `pd.NaT` for blank values, numeric zeros (`0`/`0.0`), unparseable strings,
    and epoch-zero artifacts (`01-Jan-1970`, `1899-12-30`, `1900-01-01`, `1904-01-01`).
    """
    if val is None or pd.isna(val):
        return pd.NaT

    # Prevent raw numbers (like 0 or 0.0) from being parsed as nanoseconds since 01-Jan-1970
    if isinstance(val, (int, float, np.number, bool)):
        return pd.NaT

    if isinstance(val, (pd.Timestamp, datetime, date)):
        dt_val = pd.Timestamp(val)
        if pd.isna(dt_val) or dt_val.date() in EPOCH_ZERO_DATES or dt_val.year <= 1970:
            return pd.NaT
        return dt_val

    s = str(val).strip()
    if not s or s.lower() in INVALID_DATE_TOKENS:
        return pd.NaT

    # Reject purely numeric strings (e.g. "0", "0.0", "123") so they never become 1970-01-01
    try:
        float(s)
        return pd.NaT
    except ValueError:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt = pd.to_datetime(s, errors="coerce")

    if pd.isna(dt) or dt.date() in EPOCH_ZERO_DATES or dt.year <= 1970:
        return pd.NaT
    return dt


def clean_datetime_series(series: pd.Series) -> pd.Series:
    """
    Vectorized, cross-platform datetime cleaner for DataFrame columns.
    Masks blanks, numeric zeros, and 1970/1900/1904 epoch placeholders to `pd.NaT`.
    """
    if series is None or series.empty:
        return pd.to_datetime(series, errors="coerce")

    if pd.api.types.is_numeric_dtype(series):
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    s_str = series.astype("string").str.strip()
    lower_mask = s_str.str.lower().isin(INVALID_DATE_TOKENS)
    numeric_mask = pd.to_numeric(s_str, errors="coerce").notna()
    s_clean = s_str.mask(lower_mask | numeric_mask, pd.NA)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt_series = pd.to_datetime(s_clean, errors="coerce")

    epoch_mask = dt_series.dt.date.isin(EPOCH_ZERO_DATES) | (dt_series.dt.year <= 1970)
    return dt_series.mask(epoch_mask, pd.NaT)


def format_date_cross_platform(dt_obj: pd.Timestamp) -> str:
    """Format a Timestamp as DD-Mmm-YYYY identically across Windows, macOS, iPad, and iPhone."""
    if pd.isna(dt_obj):
        return "Missing"
    return f"{dt_obj.day:02d}-{MONTH_ABBR.get(dt_obj.month, 'Jan')}-{dt_obj.year}"


# ==============================================================================
# 2. GENDER PREDICTION FEATURE EXTRACTOR (TOP-LEVEL FOR PICKLE COMPATIBILITY)
# ==============================================================================
def extract_features(name):
    """Extract character and token level features from a person's name."""
    name_str = "" if pd.isna(name) else str(name).strip().lower()
    words = name_str.split()
    return {
        "length": len(name_str),
        "vowel_count": len(re.findall(r"[aeiou]", name_str)),
        "has_repetition": 1 if len(set(words)) < len(words) else 0,
        "first_two": name_str[:2] if len(name_str) >= 2 else name_str,
        "last_two": name_str[-2:] if len(name_str) >= 2 else name_str,
        "consonant_count": len(re.findall(r"[bcdfghjklmnpqrstvwxyz]", name_str)),
        "space_count": name_str.count(" "),
    }


class NameFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for custom name features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = [extract_features(name) for name in X]
        return np.array(
            [
                [
                    f["length"],
                    f["vowel_count"],
                    f["has_repetition"],
                    f["consonant_count"],
                    f["space_count"],
                ]
                for f in features
            ],
            dtype=float,
        )

    @staticmethod
    def predict_gender(name, model=None, model_file="gender_predictor_model.pkl"):
        if model is None:
            model_path = Path(model_file)
            if not model_path.is_absolute():
                candidate = APP_DIR / model_path
                model_path = candidate if candidate.exists() else model_path
            model = joblib.load(model_path)
        prediction = model.predict([str(name)])[0]
        probability = model.predict_proba([str(name)])[0]
        return prediction, float(max(probability))


def build_gender_pipeline(n_estimators: int = 200, random_state: int = 42) -> Pipeline:
    """
    Create a scikit-learn Pipeline combining character TF-IDF and custom name features.
    Uses n_jobs=2 instead of -1 to prevent CPU/RAM thread exhaustion on Streamlit Cloud.
    """
    combined_features = FeatureUnion(
        transformer_list=[
            (
                "char_tfidf",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True),
            ),
            ("word_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True)),
            ("name_stats", NameFeatureExtractor()),
        ]
    )
    return Pipeline(
        steps=[
            ("features", combined_features),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=2,
                ),
            ),
        ]
    )


def filter_gender_results(df_input: pd.DataFrame, min_confidence: float = 0.5) -> pd.DataFrame:
    """Filter gender discrepancy predictions by minimum confidence threshold."""
    if df_input is None or df_input.empty:
        return pd.DataFrame(columns=["PatientID", "name", "gender", "predicted_gender", "confidence"])
    df_out = df_input.copy()
    df_out["confidence"] = pd.to_numeric(df_out["confidence"], errors="coerce")
    return df_out[df_out["confidence"] >= min_confidence].copy()


def run_gender_prediction_check(
    df_source: pd.DataFrame,
    model,
    id_col: str = "PatientID",
    name_col: str = "Name_x",
    sex_col: str = "Sex_x",
    min_confidence: float = 0.5,
) -> pd.DataFrame:
    """Run gender prediction against recorded sex and return high-confidence mismatches."""
    if model is None or df_source is None or df_source.empty:
        return pd.DataFrame(columns=["PatientID", "name", "gender", "predicted_gender", "confidence"])

    actual_name_col = name_col if name_col in df_source.columns else ("Name" if "Name" in df_source.columns else None)
    actual_sex_col = sex_col if sex_col in df_source.columns else ("Sex" if "Sex" in df_source.columns else None)
    actual_id_col = id_col if id_col in df_source.columns else None

    if not actual_name_col or not actual_sex_col:
        return pd.DataFrame(columns=["PatientID", "name", "gender", "predicted_gender", "confidence"])

    df_gender = pd.DataFrame()
    df_gender["PatientID"] = df_source[actual_id_col].astype(str) if actual_id_col else ""
    df_gender["name"] = df_source[actual_name_col].fillna("").astype(str).str.strip()
    df_gender["gender"] = (
        df_source[actual_sex_col]
        .astype(str)
        .str.strip()
        .replace({"1": "M", "2": "F", "1.0": "M", "2.0": "F", "Male": "M", "Female": "F"})
    )

    df_predict = df_gender[
        df_gender["name"].ne("") & ~df_gender["name"].str.lower().isin(["nan", "none", "<na>"])
    ].copy()
    if df_predict.empty:
        return pd.DataFrame(columns=["PatientID", "name", "gender", "predicted_gender", "confidence"])

    predictions = model.predict(df_predict["name"])
    probabilities = model.predict_proba(df_predict["name"])
    df_predict["predicted_gender"] = predictions
    df_predict["confidence"] = np.max(probabilities, axis=1).round(4)

    valid_recorded = df_predict["gender"].isin(["M", "F"])
    mismatches = df_predict[valid_recorded & (df_predict["gender"] != df_predict["predicted_gender"])].copy()
    mismatches = mismatches.sort_values(by="confidence", ascending=False).reset_index(drop=True)
    return filter_gender_results(mismatches, min_confidence=min_confidence)


# ==============================================================================
# 3. SUPABASE & EXCEL LOADING HELPERS
# ==============================================================================
def get_default_supabase_credentials():
    """Retrieve Supabase URL and Key from Streamlit secrets, environment variables, or project defaults."""
    url = "https://kocihpxevlowqbguhstf.supabase.co"
    key = "sb_publishable_JtrNLjMNSvZ5LzvXKbv2xw_mj-hl5MD"
    try:
        if "SUPABASE_URL" in st.secrets:
            url = st.secrets["SUPABASE_URL"]
        if "SUPABASE_KEY" in st.secrets:
            key = st.secrets["SUPABASE_KEY"]
        if "supabase" in st.secrets:
            supa_sec = st.secrets["supabase"]
            url = supa_sec.get("url", "") or supa_sec.get("SUPABASE_URL", "") or url
            key = supa_sec.get("key", "") or supa_sec.get("SUPABASE_KEY", "") or key
        if "connections" in st.secrets and "supabase" in st.secrets["connections"]:
            conn_sec = st.secrets["connections"]["supabase"]
            url = conn_sec.get("SUPABASE_URL", "") or conn_sec.get("url", "") or url
            key = conn_sec.get("SUPABASE_KEY", "") or conn_sec.get("key", "") or key
    except Exception:
        pass

    url = os.environ.get("SUPABASE_URL", "") or url
    key = os.environ.get("SUPABASE_KEY", "") or key
    return url, key


@st.cache_resource(show_spinner=False)
def get_supabase_client(supabase_url: str, supabase_key: str):
    """Create and cache a single Supabase client instance across reruns."""
    from supabase import create_client
    return create_client(supabase_url.strip(), supabase_key.strip())


@st.cache_data(ttl=600, max_entries=3, show_spinner=False)
def fetch_supabase_table(
    supabase_url: str,
    supabase_key: str,
    table_name: str = SUPABASE_DEFAULT_TABLE,
    batch_size: int = 1000,
) -> pd.DataFrame:
    """
    Fetch all rows from Supabase table `ygntbpro` (which stores the merged data of
    `YgnTBPro` and `Notified TB Register`) using pagination to bypass the 1,000-row limit.
    """
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL and Key are required.")

    client = get_supabase_client(supabase_url, supabase_key)
    all_rows = []
    start = 0

    while True:
        response = (
            client.table(table_name.strip())
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


@st.cache_data(max_entries=2, show_spinner=False)
def read_excel_workbook_sheets(
    file_bytes: bytes, candidate_sheets: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Read available sheets from an uploaded Excel workbook in a single pass."""
    with pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl") as xf:
        available = xf.sheet_names
        if candidate_sheets is None:
            sheets_to_load = available
        else:
            sheets_to_load = [s for s in candidate_sheets if s in available]
            if not sheets_to_load:
                raise ValueError(
                    f"None of the expected sheets {candidate_sheets} were found. Available sheets: {available}"
                )

        dfs = pd.read_excel(xf, sheet_name=sheets_to_load)
        for name, df in dfs.items():
            df["SheetName"] = name
        return dfs


def generate_multisheet_excel_bytes(sheets_dict: dict[str, pd.DataFrame]) -> bytes:
    """Write multiple non-empty verification DataFrames into a single Excel file in memory."""
    output = BytesIO()
    valid_sheets = {
        name: df for name, df in sheets_dict.items() if isinstance(df, pd.DataFrame) and not df.empty
    }
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if valid_sheets:
            for sheet_name, df in valid_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        else:
            pd.DataFrame({"Status": ["No discrepancies or records found."]}).to_excel(
                writer, sheet_name="Summary", index=False
            )
    return output.getvalue()


# ==============================================================================
# 4. PPM (DATABASE VERIFICATION) ENGINE & HELPERS
# ==============================================================================
def clean_column_name(col_name: str) -> str:
    """Clean column names and standardize known header variations."""
    cleaned = re.sub(r"[\\/,()\n]|_x000D_|\b(BC/CD|DS_TB/DR_TB)\b", "", str(col_name)).strip()
    rename_dict = {
        "Type of TB Treatment DS_TBDR_TB": "Type of TB Treatment",
        "Treatment Registration YN": "Treatment Registration",
        "Treatment Registrar Dateplease fill in date format: 1-Jan-2021": "Treatment Registration Date",
        "NationalityN  NN": "Nationality",
        "Specify ______ if EPSite of EPTB": "Specify if EPTB",
        "CPT please fill in date format: 1-Jan-2021": "CPT Date",
        "ARTplease fill in date format: 1-Jan-2021": "ART Date",
        "Xpert Result N= MTB not detected I = Invalid No result  T= MTB detected RR= Rif resistant TI= MTB +  Rif resistant is invalid or No result": "Gene Xpert Result",
        "Initial Regimen Started Dateplease fill in date format: 1-Jan-2021": "Initial Regimen Started Date",
        "Type of Patient's": "Type Of Patient",
    }
    return rename_dict.get(cleaned, cleaned)


DTYPE_CHANGE_MAP = {
    "Date_x": "datetime64[ns]", "Team": "Int64", "Tsp": "string", "WardVillage": "string",
    "WardCode": "string", "Approach": "string", "Visit_no": "Int64", "Sr_No": "Int64",
    "PatientID": "string", "Name_x": "string", "Age_x": "float32", "Sex_x": "Int64",
    "Address_x": "string", "Ptstsp_x": "string", "PhoneNo_x": "string", "Origin": "Int64",
    "VOL": "Int64", "Referralfor": "Int64", "VolunteerName": "string", "Organization": "string",
    "Cough": "Int64", "Fever": "Int64", "Wtloss": "Int64", "Nightsweat": "Int64", "Haemoptysis": "Int64",
    "Chestpain": "Int64", "Fatigue": "Int64", "Neckglands": "Int64", "TBcontact": "Int64",
    "MDRTBcontact": "Int64", "TBTreatmenthistory": "Int64",
    "Smoking": "Int64", "Reasonforexamination": "Int64",
    "TypeofPatient": "Int64", "PublicHealthCare1": "Int64", "TypeofPatient1": "Int64",
    "DM1": "Int64", "HT1": "Int64", "RBS": "Int64", "HbA1c": "Int64", "Systolic": "Int64", "Diastolic": "Int64",
    "DMHT1": "Int64", "RTIAVI1": "Int64", "Generalweakness1": "Int64",
    "Other1": "string", "Cxrr": "Int64", "CXRresult": "Int64", "Cxr_others": "string",
    "Sputum_request": "Int64", "Micror": "Int64", "Sputummicroscopyresult": "Int64",
    "Genexpertrequested": "Int64", "GeneXpertresult": "Int64", "XDRResult": "string", "CultureDSTresult": "string",
    "Bact_status": "Int64", "Case": "Int64", "Treatmentreferral": "Int64", "OLDTYPEOFTBTREATMENT": "string",
    "TreatmentRegimen": "Int64", "Modifiedregimen": "string", "Placeforreferral": "Int64", "Ref1": "string",
    "TreatmentOutcome1211": "Int64", "Tx_Outcome_Date": "datetime64[ns]",
    "DOTvolName111": "string", "VolunteerGender111": "string", "VolunteerOrganization111": "string",
    "ContactInvestigation111": "string", "DOTSupervision111": "string", "DOTStartedDate111": "string",
    "DOTsupervisiontillTreatmentComp111": "string",
    "Seeing1": "Int64", "Hearing1": "Int64", "Walking1": "Int64", "Cognition1": "Int64",
    "Selfcare1": "Int64", "Communication1": "Int64", "Disability1": "Int64", "Remark_x": "string",
    "MonthDiagnosis11": "string", "Location11": "string", "EPI11": "string",
    "TB11": "string", "CXRresult211": "Int64", "Xray2ndReading11": "Int64",
    "XrayRemark11": "string", "XrayEQA11": "string", "TownshipTBRegNumber": "string",
    "ReportingTsp11": "string", "ReportingYear11": "Int64", "Recstatus": "string",
    "UniqueKey": "string", "Month Diagnosis": "string", "Place of Notification": "string",
    "Name_y": "string", "Address_y": "string", "Ptstsp_y": "string", "Ward": "string",
    "PhoneNo_y": "string", "Volunteer Referral": "string", "Bacteriological status": "string",
    "Type of TB Treatment": "string", "Treatment Registration": "string",
    "StateRegionName": "string", "StateRegion Name": "string", "Township Name": "string",
    "Facility Name Place of Registration": "string", "ReportingPeriod": "string", "RegistrarDate": "datetime64[ns]",
    "Reporting Period": "string", "Treatment Registrar Date": "datetime64[ns]", "Township TB Reg Number": "string",
    "Nationality_x": "string", "Nationality_y": "string", "Sex_y": "string",
    "Age_y": "float32", "Refered from": "string", "Transferin": "string", "Transfer in": "string",
    "Type of Patient": "string", "TB Site": "string", "Specify Site of EPTB if EP": "string",
    "Treatment Regimens": "string", "TypeofPatients": "string", "TypeofDisease": "string",
    "SpecifyifEPSiteofEPTB": "string", "Smoking Status": "string", "DM Status": "string",
    "HIVStatus": "string", "CPT": "datetime64[ns]", "ART": "datetime64[ns]", "HIV Status": "string",
    "CPT Date": "datetime64[ns]", "ART Date": "datetime64[ns]",
    "Microscope Result": "string", "X-Ray Result": "Int64", "Xpert Result": "string",
    "Culture Result": "string", "2nd month Microscope Result": "string",
    "2nd month Xpert Result": "string", "3rd month Microscope Result": "string",
    "3rd month Xpert Result": "string", "5th month Microscope Result": "string",
    "5th month Xpert Result": "string", "End of Tx Microscope Result": "string",
    "End of Tx X-Ray Result": "string", "End of Tx Xpert Result": "string",
    "Treatment Outcome": "string", "Initial Regimen Started Date": "datetime64[ns]",
    "Treatment Outcome date": "datetime64[ns]", "Volunteer Name": "string",
    "Volunteer Gender": "string", "Volunteer Organization": "string",
    "Contact Investigation": "string", "DOT  Supervision": "string",
    "DOT Started Date": "datetime64[ns]",
    "DOT supervision till Treatment Completion": "string", "Seeing": "Int64",
    "Hearing": "Int64", "Walking": "Int64", "Cognition": "Int64", "Selfcare": "Int64",
    "Communication": "Int64", "Disability": "Int64", "Remark_y": "string",
    "SDG_EPI": "string", "SDG_TB": "string", "Location": "string", "Date_y": "datetime64[ns]",
    "Reporting Year": "Int64", "SheetName_y": "string", "TPTregno": "string",
    "RegnoofindexTBcase": "string", "TPTregistrationdate": "datetime64[ns]",
    "TPTregimen": "string", "DrugDose": "string", "Drugpickup1": "datetime64[ns]",
    "Date_drug_pickup2": "datetime64[ns]", "Date_drug_pickup_3": "datetime64[ns]",
    "Date_drug_pickup_4": "datetime64[ns]", "Date_drug_pickup_5": "datetime64[ns]",
    "Date_drug_pickup_6": "datetime64[ns]", "Date_drug_pickup_7": "datetime64[ns]",
    "Date_drug_pickup_8": "datetime64[ns]", "Date_drug_pickup_9": "datetime64[ns]",
    "TPTdiscontinuationdate": "datetime64[ns]", "TPTTreatmentOutcome": "string",
    "OutcomeDate": "datetime64[ns]", "TypeofTBTreatment": "Int64", "SheetName_x": "string",
}

CANONICAL_COL_LOOKUP = {k.lower(): k for k in DTYPE_CHANGE_MAP.keys()}
CANONICAL_COL_LOOKUP.update(
    {
        "patientid": "PatientID",
        "name": "Name",
        "age": "Age",
        "sex": "Sex",
        "date": "Date",
        "address": "Address",
        "ptstsp": "Ptstsp",
        "phoneno": "PhoneNo",
        "remark": "Remark",
        "nationality": "Nationality",
        "sheetname": "SheetName",
    }
)


def align_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Align column casing with expected schema names (handles lowercase Postgres headers)."""
    df_out = df.copy()
    new_cols = []
    for c in df_out.columns:
        cleaned = clean_column_name(c)
        canonical = CANONICAL_COL_LOOKUP.get(cleaned.lower(), cleaned)
        new_cols.append(canonical)
    df_out.columns = new_cols
    return df_out


def changeDtype(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame columns to their standardized nullable types safely across platforms."""
    df_out = df.copy()
    for col, dtype in DTYPE_CHANGE_MAP.items():
        if col in df_out.columns:
            if dtype.startswith("datetime64"):
                df_out[col] = clean_datetime_series(df_out[col])
            elif dtype == "Int64":
                df_out[col] = pd.to_numeric(df_out[col], errors="coerce").astype("Int64")
            elif dtype == "float32":
                df_out[col] = pd.to_numeric(df_out[col], errors="coerce").astype("float32")
            elif dtype == "string":
                df_out[col] = df_out[col].astype("string")
    return df_out


def prepare_combined_dataframe(ygntbpro_df: pd.DataFrame, tb_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Prepare the combined DataFrame for PPM validation:
    - When loading from Excel (`YgnTBPro` + `Notified TB Register`), merges both sheets on `PatientID`.
    - When loading from Supabase `ygntbpro` (which is already the merged table of `YgnTBPro` and
      `Notified TB Register`), aligns column names and ensures `_x` / `_y` columns are ready.
    """
    ygn = align_canonical_columns(ygntbpro_df)

    if tb_df is not None and not tb_df.empty:
        tb = align_canonical_columns(tb_df)
        df_combined = pd.merge(ygn, tb, on="PatientID", how="outer", suffixes=("_x", "_y"))
    else:
        df_combined = ygn.copy()
        standalone_suffix_map = {
            "Date": "Date_x",
            "Name": "Name_x",
            "Age": "Age_x",
            "Sex": "Sex_x",
            "Address": "Address_x",
            "Ptstsp": "Ptstsp_x",
            "PhoneNo": "PhoneNo_x",
            "Remark": "Remark_x",
            "Nationality": "Nationality_x",
            "SheetName": "SheetName_x",
        }
        rename_map = {
            orig_c: suff_c
            for orig_c, suff_c in standalone_suffix_map.items()
            if orig_c in df_combined.columns and suff_c not in df_combined.columns
        }
        if rename_map:
            df_combined = df_combined.rename(columns=rename_map)

    if not df_combined.empty:
        df_combined = align_canonical_columns(df_combined)
        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()].copy()
        df_combined = changeDtype(df_combined)
        for col in ["EPI11", "TB11", "SDG_EPI", "SDG_TB"]:
            if col in df_combined.columns:
                df_combined[col] = (
                    df_combined[col]
                    .astype(str)
                    .str.replace(r"\.0$", "", regex=True)
                    .replace({"nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
                )
    return df_combined


class TBDataValidator:
    """Rule-based data validator for combined YgnTBPro & Notified TB Register records."""

    def __init__(self, target_year: int = 2026):
        self.invalid_rows = []
        self.visit_log = {}
        self.target_year = target_year
        self.existing_tb_regs = set()
        self.seen_pids = set()

        self.mappings = {
            "sex": {1: "M", 2: "F"},
            "yes_no": {1: "Yes", 2: "No"},
            "y_n": {1: "Y", 2: "N"},
            "tx_regime": {1: "I", 2: "R", 3: "C", 4: "MDR", 5: "M"},
            "bact_status": {1: "BC", 2: "CD"},
            "facility": {1: "1. NTP", 2: "2. MMA-PPM", 3: "3. PSI-PPM", 4: "4. MATA-PPM", 5: "5. Other"},
            "smoking": {0: "N", 1: "C", 2: "P"},
            "dm": {0: "U", 1: "Y", 2: "N", 3: "Y"},
            "dsdr": {1: "DS-TB", 2: "DR-TB"},
            "gene": {0: "N", 1: "I", 2: "T", 3: "RR", 4: "TI", 5: "denied", 6: "missing", 7: "TT"},
        }
        self.date_columns_to_check = [
            "Date_y", "Date_x", "RegistrarDate", "CPT", "ART", "Tx_Outcome_Date",
            "DOTStartedDate111", "TPTregistrationdate", "Drugpickup1",
            "Date_drug_pickup2", "Date_drug_pickup_3", "Date_drug_pickup_4",
            "Date_drug_pickup_5", "Date_drug_pickup_6", "Date_drug_pickup_7",
            "Date_drug_pickup_8", "Date_drug_pickup_9", "TPTdiscontinuationdate",
            "OutcomeDate", "Treatment Registrar Date", "CPT Date", "ART Date",
            "Initial Regimen Started Date", "Treatment Outcome date", "DOT Started Date",
        ]

    @staticmethod
    def is_missing(data) -> bool:
        if data is None or pd.isna(data):
            return True
        if str(data).strip().lower() in INVALID_DATE_TOKENS:
            return True
        return False

    @staticmethod
    def is_in(val, options) -> bool:
        if pd.isna(val):
            return False
        return bool(val in options)

    @staticmethod
    def is_equal(val, target) -> bool:
        if pd.isna(val):
            return False
        return bool(val == target)

    def add_error(self, row, message: str):
        self.invalid_rows.append(
            {
                "InvalidData": message,
                "PatientID": row.get("PatientID", ""),
                "Approach": row.get("Approach", ""),
                "Tsp": row.get("Tsp", ""),
            }
        )

    def check_year_consistency(self, row):
        for col in self.date_columns_to_check:
            date_val = row.get(col)
            dt_obj = parse_valid_datetime(date_val)
            if not pd.isna(dt_obj) and dt_obj.year != self.target_year:
                self.add_error(
                    row,
                    f"checkYear in {col}: Year is {format_date_cross_platform(dt_obj)}",
                )

    def check_date(self, row):
        date_field = row.get("Date_x")
        visit_no = row.get("Visit_no")
        approach = row.get("Approach")

        if self.is_missing(date_field) or self.is_missing(visit_no) or self.is_missing(approach):
            return

        visit_date = parse_valid_datetime(date_field)
        if pd.isna(visit_date):
            return

        if self.is_in(approach, ["PPM", "Diagnostic Center"]):
            try:
                visit_month = int(float(visit_no))
                if 1 <= visit_month <= 12:
                    curr_year = visit_date.year
                    if visit_month == 1:
                        start_date = pd.Timestamp(year=curr_year - 1, month=12, day=26)
                        end_date = pd.Timestamp(year=curr_year, month=1, day=25)
                    else:
                        start_date = pd.Timestamp(year=curr_year, month=visit_month - 1, day=26)
                        end_date = pd.Timestamp(year=curr_year, month=visit_month, day=25)
                    if not (start_date <= visit_date <= end_date):
                        self.add_error(
                            row,
                            f"Date {visit_date.date()} outside PPM date range ({start_date.date()} to {end_date.date()})",
                        )
                else:
                    self.add_error(row, f"Invalid Month: {visit_no}")
            except (ValueError, TypeError):
                pass
        else:
            if visit_no in self.visit_log:
                last_date = parse_valid_datetime(self.visit_log[visit_no])
                if not pd.isna(last_date):
                    date_diff = abs((visit_date - last_date).days)
                    if date_diff > 30:
                        self.add_error(row, f"check Date or Visit {visit_no} with discrepancy: {date_diff} days difference from last recorded date {last_date.date()}")
            else:
                self.visit_log[visit_no] = visit_date

    def check_patientid(self, row):
        pid = str(row.get("PatientID", "")).strip()
        if self.is_missing(pid) or len(pid) < 12:
            self.add_error(row, f"check_PatientID: Invalid PatientID {pid}")
            return

        if pid in self.seen_pids:
            self.add_error(row, f"check_PatientID: Duplicate PatientID detected: {pid}")
        else:
            self.seen_pids.add(pid)

        raw_date = row.get("Date_x")
        dt_parsed = parse_valid_datetime(raw_date)

        if (
            self.is_missing(row.get("Visit_no"))
            or pd.isna(dt_parsed)
            or self.is_missing(row.get("Sr_No"))
            or self.is_missing(row.get("Team"))
            or self.is_missing(row.get("Tsp"))
            or self.is_missing(row.get("Approach"))
        ):
            date_str = dt_parsed.date() if not pd.isna(dt_parsed) else "Missing"
            self.add_error(
                row,
                f"missing: Year={date_str}, VisitNo={row.get('Visit_no')}, "
                f"SerialNo={row.get('Sr_No')}, Team={row.get('Team')}, "
                f"Approach={row.get('Approach')}, Tsp={row.get('Tsp')}",
            )

        team = str(row.get("Team", "")).strip()
        tsp = str(row.get("Tsp", "")).strip()
        app = str(row.get("Approach", "")).strip()

        raw_vn = row.get("Visit_no")
        vn = str(int(float(raw_vn))).zfill(2) if not self.is_missing(raw_vn) else ""

        raw_sr = row.get("Sr_No")
        sr = str(int(float(raw_sr))).zfill(4) if not self.is_missing(raw_sr) else ""

        yr = f"{dt_parsed.year % 100:02d}" if not pd.isna(dt_parsed) else ""

        if self.is_equal(team, "5") and team != pid[0]:
            self.add_error(row, f"check_PatientID: Organization {pid[0]} mismatch with Team {team}")

        if self.is_in(tsp, ["SDG", "KMD", "HLG", "HTY"]):
            if tsp != pid[1:4]:
                self.add_error(row, f"check_PatientID: Tsp {pid[1:4]} mismatch with {tsp}")

        app_map = {
            "PPM": "P",
            "Diagnostic Center": "D",
            "Mobile Visit": "M",
            "Elderly Care": "E",
            "Touring": "T",
        }
        if self.is_in(app, app_map.keys()):
            app_char = app_map[app]
            if pid[8] != app_char:
                self.add_error(row, f"check_PatientID: Approach {pid[8]} mismatch with {app}")

        if not self.is_missing(raw_vn) and vn != pid[6:8]:
            self.add_error(row, f"check_PatientID: Visit {pid[6:8]} vs {vn}")

        if not pd.isna(dt_parsed) and yr != pid[4:6]:
            self.add_error(row, f"check_PatientID: Year {pid[4:6]} vs {yr}")

        if not self.is_missing(raw_sr) and pid.endswith("S"):
            serial_part = pid[9:-1] if len(pid) > 13 else pid[9:]
            if sr != serial_part:
                self.add_error(row, f"check_PatientID: Serial {pid[9:]} vs {sr}")

    def check_CXR_GXP(self, row):
        cxr_req, cxr_res = row.get("Cxrr"), row.get("CXRresult")
        gxp_req, gxp_res = row.get("Genexpertrequested"), row.get("GeneXpertresult")
        sputum_req = row.get("Sputum_request")
        micro_req, micro_res = row.get("Micror"), row.get("Sputummicroscopyresult")

        if self.is_equal(cxr_req, 1) and (self.is_missing(cxr_res) or not (1 <= cxr_res <= 5)):
            if "still on tx" not in str(row.get("Remark_x", "")).lower():
                self.add_error(row, "check_CXR: missing CXR result although requested")

        if self.is_equal(sputum_req, 1) and self.is_missing(micro_req) and self.is_missing(gxp_req):
            self.add_error(row, "check_sputum: Sputum requested but no Microscope or GXP request")

        if self.is_equal(gxp_req, 1) and (self.is_missing(gxp_res) or not (0 <= gxp_res <= 7)):
            if "waiting result" not in str(row.get("Remark_x", "")).lower():
                self.add_error(row, "check_GeneXpert: missing GXP result although requested")

        if self.is_equal(micro_req, 1) and (self.is_missing(micro_res) or not (0 <= micro_res <= 7)):
            if "waiting result" not in str(row.get("Remark_x", "")).lower():
                self.add_error(row, "check_Microscope: missing Microscope result although requested")

    def check_bact_status(self, row):
        bact = row.get("Bact_status")
        sputum = row.get("Sputummicroscopyresult")
        gene = row.get("GeneXpertresult")

        if self.is_equal(bact, 1):  # BC
            if (self.is_missing(sputum) and self.is_missing(gene)) or (
                self.is_in(sputum, [0, 5, 6]) and self.is_in(gene, [0, 1, 5, 6])
            ):
                self.add_error(row, "check_bactstatus: BC but results are negative/missing")
        elif self.is_equal(bact, 2):  # CD
            if self.is_in(sputum, [1, 2, 3, 4]) or self.is_in(gene, [2, 3, 4, 7]):
                self.add_error(row, "check_bactstatus: CD but found positive results")
        else:
            if self.is_in(sputum, [1, 2, 3, 4]) or self.is_in(gene, [2, 3, 4, 7]):
                self.add_error(row, "check_bactstatus: Missing Bact_status but found positive results")

    def check_treatment_regime(self, row):
        tbcase = row.get("Case")
        reg = row.get("Treatment Registration")
        dsdrtb = row.get("Type of TB Treatment")
        regime = row.get("Treatment Regimens")
        age = row.get("Age_y")
        p_type = row.get("Type of Patient")

        age_val = float(age) if not self.is_missing(age) else None
        reg_norm = str(reg).strip().upper() if not self.is_missing(reg) else ""
        is_reg_yes = reg_norm in ("YES", "Y")
        is_reg_no = reg_norm in ("NO", "N")

        if self.is_equal(tbcase, 1) and is_reg_yes:
            if age_val is not None:
                if age_val < 15:
                    if not self.is_missing(regime) and regime != "C":
                        self.add_error(row, f"check_regime: Child regime (C) expected for age {age_val}")
                else:
                    if not self.is_missing(regime) and regime == "C":
                        self.add_error(row, f"check_regime: Adult regime expected for age {age_val}")

            if not self.is_missing(regime):
                if self.is_equal(dsdrtb, "DS-TB"):
                    if self.is_in(regime, ["I", "C"]) and not self.is_in(p_type, ["N"]):
                        self.add_error(row, f"check_regime: Regime {regime} but patient {p_type} for DS-TB")
                    if self.is_in(regime, ["R"]) and self.is_in(p_type, ["N"]):
                        self.add_error(row, f"check_regime: Regime {regime} but patient {p_type} for DS-TB")
                    if self.is_in(regime, ["MDR"]):
                        self.add_error(row, f"check_regime: Regime {regime} for DS-TB")

                if self.is_equal(dsdrtb, "DR-TB"):
                    if self.is_in(regime, ["I", "R", "C"]):
                        self.add_error(row, f"check_regime: Regime {regime} for DR-TB")
        else:
            if is_reg_yes and self.is_missing(regime):
                self.add_error(row, "check_regime: Treatment Registration is Yes but missing regime")
            if is_reg_no and not self.is_missing(regime):
                self.add_error(row, f"check_regime: Treatment Registration is No but found regime {regime}")
            if self.is_equal(tbcase, 1):
                if is_reg_yes and self.is_missing(regime):
                    self.add_error(
                        row, f"check_regime: TB Case is 1 but Treatment Registration is {reg} and regime is {regime}"
                    )
                elif is_reg_no and not self.is_missing(regime):
                    self.add_error(
                        row, f"check_regime: TB Case is 1 but Treatment Registration is {reg} and regime is {regime}"
                    )

    def check_completeness(self, row):
        tbcase = row.get("Case")
        reg = row.get("Treatment Registration")
        reason = row.get("Reasonforexamination")

        if self.is_missing(reason):
            return

        required = []
        try:
            reason_int = int(float(reason))
        except (ValueError, TypeError):
            return

        if self.is_in(reason_int, [1, 2]):
            required.extend(
                [
                    "Date_x", "MonthDiagnosis11", "Team", "Tsp", "Approach",
                    "Visit_no", "Sr_No", "Name_x", "Age_x", "Case", "VOL", "Reasonforexamination",
                ]
            )
        if self.is_in(tbcase, [1, 2]):
            required.extend(
                [
                    "Month Diagnosis", "Bacteriological status", "Type of TB Treatment",
                    "StateRegion Name", "Township Name", "Volunteer Referral",
                ]
            )
            reg_norm = str(reg).strip().upper() if not self.is_missing(reg) else ""
            if reg_norm in ("YES", "Y"):
                required.extend(
                    [
                        "Township TB Reg Number", "Type of Patient", "TB Site",
                        "Treatment Regimens", "Treatment Registrar Date",
                        "Initial Regimen Started Date", "HIV Status",
                        "Facility Name Place of Registration", "Reporting Period",
                    ]
                )

        if required:
            missing_fields = [f for f in required if f in row.index and self.is_missing(row.get(f))]
            if missing_fields:
                self.add_error(row, f"check_completeness: Missing {', '.join(missing_fields)}")

    def check_sdgclinic(self, row):
        if self.is_missing(row.get("TypeofTBTreatment")) or self.is_missing(row.get("Sr_No")):
            return

        tsp = row.get("Tsp")
        approach = row.get("Approach")
        if not (self.is_equal(tsp, "SDG") and self.is_equal(approach, "PPM")):
            return

        sdg_epi = str(row.get("EPI11", "")).strip()
        sdg_tb = str(row.get("TB11", "")).strip()
        try:
            sr = int(float(row.get("Sr_No")))
        except (ValueError, TypeError):
            return
        case = row.get("Case")

        if not self.is_in(sdg_epi, ["19", "107", "LDK"]):
            self.add_error(row, "check_sdgclinic: missing SDG_EPI for SDG PPM")
            if self.is_equal(case, 1) and not self.is_in(sdg_tb, ["19", "107", "LDK"]):
                self.add_error(row, "check_sdgclinic: missing SDG_TB for TB case in SDG PPM")
        if 1 <= sr <= 1000 and not self.is_in(sdg_epi, ["19"]):
            self.add_error(row, f"check_sdgclinic: SDG_EPI {sdg_epi} should be 19 for Sr_No {sr}")
            if self.is_equal(case, 1) and not self.is_in(sdg_tb, ["19"]):
                self.add_error(row, f"check_sdgclinic: SDG_TB {sdg_tb} should be 19 for TB case with Sr_No {sr}")
        if 1001 <= sr <= 2000 and not self.is_in(sdg_epi, ["107"]):
            self.add_error(row, f"check_sdgclinic: SDG_EPI {sdg_epi} should be 107 for Sr_No {sr}")
            if self.is_equal(case, 1) and not self.is_in(sdg_tb, ["107"]):
                self.add_error(row, f"check_sdgclinic: SDG_TB {sdg_tb} should be 107 for TB case with Sr_No {sr}")
        if sr > 2000 and not self.is_in(sdg_epi, ["LDK"]):
            self.add_error(row, f"check_sdgclinic: SDG_EPI {sdg_epi} should be LDK for Sr_No {sr}")
            if self.is_equal(case, 1) and not self.is_in(sdg_tb, ["LDK"]):
                self.add_error(row, f"check_sdgclinic: SDG_TB {sdg_tb} should be LDK for TB case with Sr_No {sr}")

    def check_case(self, row):
        if self.is_missing(row.get("Case")):
            return
        try:
            case = int(float(row.get("Case")))
        except (ValueError, TypeError):
            return
        cxr = row.get("CXRresult211")
        bc = row.get("Bact_status")
        rfe = row.get("Reasonforexamination")
        age = row.get("Age_y")
        if self.is_equal(case, 1):
            if self.is_missing(bc):
                self.add_error(row, "check_case: Case is 1 but missing Bact_status")
            if self.is_missing(cxr):
                self.add_error(row, "check_case: Case is 1 but missing CXR result")
            if not self.is_in(cxr, [2, 3, 4]) and not self.is_equal(bc, 1):
                self.add_error(
                    row,
                    f"check_case: Case is {case} but found Bact_status = {bc}, "
                    f"CXR result = {cxr}, Age = {age} and Reasonforexamination = {rfe}",
                )
        else:
            if not self.is_missing(bc) and self.is_equal(bc, 1):
                self.add_error(
                    row,
                    f"check_case: Case is {case} but found Bact_status = {bc}, "
                    f"Age = {age} and Reasonforexamination = {rfe}",
                )
            if not self.is_missing(cxr) and self.is_equal(cxr, 2):
                self.add_error(
                    row,
                    f"check_case: Case is {case} but found CXR result = {cxr}, "
                    f"Age = {age} and Reasonforexamination = {rfe}",
                )

    def check_outcome(self, row):
        outcome_raw = row.get("Treatment Outcome")
        outcome_dt = parse_valid_datetime(row.get("Treatment Outcome date"))
        if self.is_missing(outcome_raw):
            if not pd.isna(outcome_dt):
                self.add_error(row, "check_outcome: Missing Treatment Outcome but found date")
        else:
            outcome = str(outcome_raw).strip().lower()
            bact = row.get("Bacteriological status")
            if pd.isna(outcome_dt):
                self.add_error(row, "check_outcome date: Missing Outcome date but found Outcome")
            if self.is_equal(outcome, "c") and not self.is_equal(bact, "BC"):
                self.add_error(row, f"check_outcome: Outcome {outcome} but not BC status")

    def check_volunteer(self, row):
        vol = row.get("VOL")
        pci = row.get("Referralfor")
        vol_name = row.get("VolunteerName")
        org = row.get("Organization")
        reason = row.get("Reasonforexamination")

        if self.is_equal(vol, 1):
            if self.is_missing(pci) or self.is_missing(vol_name) or self.is_missing(org):
                self.add_error(
                    row,
                    f"check_volunteerReferral: VOL is 1 but Referralfor {pci}, "
                    f"VolunteerName {vol_name}, or Organization {org}",
                )
            if self.is_missing(reason) or not self.is_in(reason, [1, 2]):
                self.add_error(row, f"check_volunteerReferral: VOL is 1 but Reasonforexamination {reason}")
        else:
            if not self.is_missing(pci) or not self.is_missing(vol_name) or not self.is_missing(org):
                self.add_error(
                    row,
                    f"check_volunteerReferral: VOL is not 1 but found Referralfor {pci}, "
                    f"VolunteerName {vol_name}, or Organization {org}",
                )

    def check_duplicate_tb_reg(self, row):
        tb_reg_num = row.get("TownshipTBRegNumber")
        township = row.get("Facility Name Place of Registration")
        facility = row.get("Placeforreferral")
        if self.is_missing(tb_reg_num):
            return
        composite_key = (str(tb_reg_num).strip(), str(township).strip(), str(facility).strip())
        if composite_key in self.existing_tb_regs:
            self.add_error(
                row,
                f"check_Duplicate TownshipTBRegNumber '{tb_reg_num}' "
                f"found for Township '{township}' and Facility '{facility}'",
            )
        else:
            self.existing_tb_regs.add(composite_key)

    def check_uncoding(self, row):
        if not self.is_equal(row.get("Case"), 1):
            return

        checks = [
            ("Sex_y", "Sex_x", "sex"),
            ("Treatment Registration", "Treatmentreferral", "yes_no"),
            ("Bacteriological status", "Bact_status", "bact_status"),
            ("Smoking Status", "Smoking", "smoking"),
            ("DM Status", "DM1", "dm"),
            ("Treatment Regimens", "TreatmentRegimen", "tx_regime"),
            ("Type of TB Treatment", "TypeofTBTreatment", "dsdr"),
            ("Xpert Result", "GeneXpertresult", "gene"),
            ("Facility Name Place of Registration", "Placeforreferral", "facility"),
        ]

        issues = []
        for t_col, p_col, map_key in checks:
            t_val = row.get(t_col)
            p_val = row.get(p_col)
            if not self.is_missing(t_val) and not self.is_missing(p_val):
                try:
                    p_key = int(float(p_val))
                except (ValueError, TypeError):
                    p_key = p_val
                expected = self.mappings[map_key].get(p_key)
                if str(t_val).strip() != str(expected):
                    issues.append(f"{map_key.capitalize()}: {p_val} vs {t_val}")

        if issues:
            self.add_error(row, f"check_uncoding: {', '.join(issues)}")

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main entry point to validate the combined DataFrame."""
        for _, row in df.iterrows():
            self.check_date(row)
            self.check_patientid(row)
            self.check_CXR_GXP(row)
            self.check_bact_status(row)
            self.check_treatment_regime(row)
            self.check_completeness(row)
            self.check_sdgclinic(row)
            self.check_case(row)
            self.check_outcome(row)
            self.check_volunteer(row)
            self.check_uncoding(row)
            self.check_year_consistency(row)
            self.check_duplicate_tb_reg(row)

        if not self.invalid_rows:
            return pd.DataFrame(columns=["InvalidData", "PatientID", "Approach", "Tsp"])

        df_val = pd.DataFrame(self.invalid_rows)
        df_val = df_val.sort_values(by="InvalidData", ascending=True, ignore_index=True)
        return df_val


def fuzzyMatchingData(dfmata: pd.DataFrame, score_threshold: int = 85, chunk_size: int = 500) -> pd.DataFrame:
    """
    Memory-safe, chunked fuzzy duplicate detection.
    Processes `chunk_size` rows at a time with uint8 scores instead of allocating a giant
    N x N matrix and `np.triu_indices(N)` in memory.
    """
    empty_cols = [
        "Score", "ID", "SimilarID", "Patient", "SimilarPatient", "ScoreAddress",
        "Address", "SimilarAddress", "Visit1", "Visit2", "Duration",
        "Name", "Age", "Sex", "Tsp", "ReasonForExamination", "Analysis",
    ]
    if dfmata is None or dfmata.empty:
        return pd.DataFrame(columns=empty_cols)

    columns_to_check = ["Name_x", "Age_x", "Sex_x", "Tsp", "Reasonforexamination"]
    required_columns = columns_to_check + ["PatientID", "Address_x", "Visit_no"]
    missing_cols = [col for col in required_columns if col not in dfmata.columns]
    if missing_cols:
        return pd.DataFrame(columns=empty_cols)

    df_sub = dfmata[pd.to_numeric(dfmata["Reasonforexamination"], errors="coerce") == 1].copy()
    n_rows = len(df_sub)
    if n_rows < 2:
        return pd.DataFrame(columns=empty_cols)

    df_sub = df_sub.reset_index(drop=True)
    for c in columns_to_check:
        df_sub[c] = df_sub[c].fillna("").astype(str).str.strip()

    df_sub["DataCheck"] = (
        df_sub["Name_x"]
        + "_"
        + df_sub["Age_x"]
        + "_"
        + df_sub["Sex_x"]
        + "_"
        + df_sub["Tsp"]
        + "_"
        + df_sub["Reasonforexamination"]
    )
    df_sub["Visit_no_int"] = pd.to_numeric(df_sub["Visit_no"], errors="coerce").fillna(0).astype(int)

    datacheck_values = df_sub["DataCheck"].tolist()
    pids = df_sub["PatientID"].astype(str).tolist()
    addresses = df_sub["Address_x"].fillna("").astype(str).tolist()
    visits = df_sub["Visit_no_int"].tolist()
    names = df_sub["Name_x"].tolist()
    ages = df_sub["Age_x"].tolist()
    sexes = df_sub["Sex_x"].tolist()
    tsps = df_sub["Tsp"].tolist()
    rfes = df_sub["Reasonforexamination"].tolist()

    results = []

    for start_idx in range(0, n_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, n_rows)
        chunk_queries = datacheck_values[start_idx:end_idx]

        if HAS_RAPIDFUZZ:
            scores_chunk = rf_process.cdist(
                chunk_queries,
                datacheck_values,
                scorer=fuzz.ratio,
                score_cutoff=score_threshold,
                dtype=np.uint8,
                workers=1,
            )
            rel_is, js = np.where(scores_chunk >= score_threshold)
            for rel_i, j in zip(rel_is, js):
                i = start_idx + int(rel_i)
                j = int(j)
                if j <= i:
                    continue
                score = float(scores_chunk[rel_i, j])
                addr_i, addr_j = addresses[i], addresses[j]
                score_addr = float(fuzz.ratio(addr_i, addr_j))

                match_flags = {
                    "Name": names[i] == names[j],
                    "Age": ages[i] == ages[j],
                    "Sex": sexes[i] == sexes[j],
                    "Tsp": tsps[i] == tsps[j],
                    "ReasonForExamination": rfes[i] == rfes[j],
                }
                matched_attrs = [k for k, is_match in match_flags.items() if is_match]
                results.append(
                    {
                        "Score": round(score, 1),
                        "ID": pids[i],
                        "SimilarID": pids[j],
                        "Patient": datacheck_values[i],
                        "SimilarPatient": datacheck_values[j],
                        "ScoreAddress": round(score_addr, 1),
                        "Address": addr_i,
                        "SimilarAddress": addr_j,
                        "Visit1": visits[i],
                        "Visit2": visits[j],
                        "Duration": abs(visits[j] - visits[i]),
                        **match_flags,
                        "Analysis": matched_attrs,
                    }
                )
        else:
            for rel_i, q_val in enumerate(chunk_queries):
                i = start_idx + rel_i
                for j in range(i + 1, n_rows):
                    score = difflib.SequenceMatcher(None, q_val, datacheck_values[j]).ratio() * 100
                    if score >= score_threshold:
                        addr_i, addr_j = addresses[i], addresses[j]
                        score_addr = difflib.SequenceMatcher(None, addr_i, addr_j).ratio() * 100
                        match_flags = {
                            "Name": names[i] == names[j],
                            "Age": ages[i] == ages[j],
                            "Sex": sexes[i] == sexes[j],
                            "Tsp": tsps[i] == tsps[j],
                            "ReasonForExamination": rfes[i] == rfes[j],
                        }
                        matched_attrs = [k for k, is_match in match_flags.items() if is_match]
                        results.append(
                            {
                                "Score": round(score, 1),
                                "ID": pids[i],
                                "SimilarID": pids[j],
                                "Patient": datacheck_values[i],
                                "SimilarPatient": datacheck_values[j],
                                "ScoreAddress": round(score_addr, 1),
                                "Address": addr_i,
                                "SimilarAddress": addr_j,
                                "Visit1": visits[i],
                                "Visit2": visits[j],
                                "Duration": abs(visits[j] - visits[i]),
                                **match_flags,
                                "Analysis": matched_attrs,
                            }
                        )

    if not results:
        return pd.DataFrame(columns=empty_cols)

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values(by="Score", ascending=False, ignore_index=True)
    return df_result


UNIQUE_CHECK_COLUMNS = [
    "Team", "StateRegionName", "Tsp", "WardVillage", "Approach", "Visit_no", "ReportingPeriod",
    "Nationality_x", "Sex_x", "Origin", "VOL", "Referralfor", "Organization", "Cough", "Fever",
    "Wtloss", "Nightsweat", "Haemoptysis", "Chestpain", "Fatigue", "Neckglands", "TBcontact",
    "MDRTBcontact", "TBTreatmenthistory", "Transferin", "TypeofPatients", "TypeofDisease",
    "SpecifyifEPSiteofEPTB", "Smoking", "Reasonforexamination", "TypeofPatient", "PublicHealthCare1",
    "TypeofPatient1", "DM1", "RBS", "HbA1c", "HT1", "Systolic", "Diastolic", "DMHT1", "RTIAVI1",
    "Generalweakness1", "HIVStatus", "Cxrr", "CXRresult", "Cxr_others", "Sputum_request", "Micror",
    "Sputummicroscopyresult", "Genexpertrequested", "GeneXpertresult", "XDRResult", "Bact_status",
    "Case", "Treatmentreferral", "TreatmentRegimen", "Modifiedregimen", "Placeforreferral", "Ref1",
    "CultureDSTresult", "Microscope_Result_2nd_mnth", "Xpert_Result_2nd_mnth", "Microscope_Result_3rd_mnth",
    "Xpert_Result_3rd_mnth", "Microscope_Result_5th_mnth", "Xpert_Result_5th_mnth", "EndofTxMicroscopeResult",
    "EndofTxXRayResult", "EndofTxXpertResult", "TreatmentOutcome1211", "VolunteerGender111",
    "VolunteerOrganization111", "ContactInvestigation111", "DOTSupervision111",
    "DOTsupervisiontillTreatmentComp111", "Seeing1", "Hearing1", "Walking1", "Cognition1", "Selfcare1",
    "Communication1", "Disability1", "MonthDiagnosis11", "Location11", "EPI11", "TB11", "Xray2ndReading11",
    "ReportingTsp11", "CXRresult211", "XrayEQA11", "ReportingYear11", "TPTregimen", "DrugDose",
    "TPTTreatmentOutcome", "TypeofTBTreatment", "SheetName_x", "Month Diagnosis", "Place of Notification",
    "Ward", "Volunteer Referral", "Bacteriological status", "Type of TB Treatment", "Treatment Registration",
    "StateRegion Name", "Township Name", "Facility Name Place of Registration", "Reporting Period",
    "Nationality_y", "Sex_y", "Refered from", "Transfer in", "Type of Patient", "TB Site",
    "Specify Site of EPTB if EP", "Treatment Regimens", "Smoking Status", "DM Status", "HIV Status",
    "Microscope Result", "X-Ray Result", "Xpert Result", "Culture Result", "2nd month Microscope Result",
    "2nd month Xpert Result", "3rd month Microscope Result", "3rd month Xpert Result",
    "5th month Microscope Result", "5th month Xpert Result", "End of Tx Microscope Result",
    "End of Tx X-Ray Result", "End of Tx Xpert Result", "Treatment Outcome", "Volunteer Gender",
    "Volunteer Organization", "Contact Investigation", "DOT  Supervision",
    "DOT supervision till Treatment Completion", "Seeing", "Hearing", "Walking", "Cognition", "Selfcare",
    "Communication", "Disability", "SDG_EPI", "SDG_TB", "Location", "Reporting Year", "SheetName_y",
]


def compute_unique_values_table(df_combined: pd.DataFrame) -> pd.DataFrame:
    """Build a transposed summary table of sorted unique values for all present categorical columns."""
    present_cols = [c for c in UNIQUE_CHECK_COLUMNS if c in df_combined.columns]
    if not present_cols:
        return pd.DataFrame()

    unique_series = {}
    for col in present_cols:
        u_values = df_combined[col].dropna().unique()
        s = pd.Series([str(v).strip() for v in u_values if str(v).strip() not in ("", "nan", "<NA>", "None")])
        s = s.drop_duplicates().sort_values().reset_index(drop=True)
        unique_series[col] = s

    df_unique = pd.concat(unique_series, axis=1).T
    df_unique = df_unique.rename_axis("ColumnNames").reset_index()
    df_unique.columns = [str(c) for c in df_unique.columns]
    return df_unique


DATE_COLUMNS = [
    "Date", "Date_x", "Date_y", "RegistrarDate", "CPT", "ART", "Tx_Outcome_Date",
    "DOTStartedDate111", "TPTregistrationdate", "Drugpickup1", "Date_drug_pickup2",
    "Date_drug_pickup_3", "Date_drug_pickup_4", "Date_drug_pickup_5", "Date_drug_pickup_6",
    "Date_drug_pickup_7", "Date_drug_pickup_8", "Date_drug_pickup_9",
    "TPTdiscontinuationdate", "OutcomeDate",
]


def normalize_value(val):
    if pd.isna(val) or val is None or val is np.nan:
        return None
    s_val = str(val).strip()
    if s_val == "" or s_val.lower() in ("nan", "nat", "none", "null", "<na>"):
        return None
    try:
        f_val = float(s_val)
        if f_val.is_integer():
            return str(int(f_val))
        return str(f_val)
    except (ValueError, TypeError):
        pass
    return s_val


def ensure_tpt_patientid(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `PatientID` exists for `TPT Register` sheets."""
    df_out = df.copy()
    if "PatientID" not in df_out.columns:
        required_columns = {"Township", "TPT Registration Number", "Name"}
        if required_columns.issubset(set(df_out.columns)):
            df_out["PatientID"] = (
                df_out["Township"].astype(str).str.strip()
                + "_"
                + df_out["TPT Registration Number"].astype(str).str.strip()
                + "_"
                + df_out["Name"].astype(str).str.strip()
            )
    return df_out


def compare_data(
    dfCurrent: pd.DataFrame,
    dfPrevious: pd.DataFrame,
    id_col: str = "PatientID",
    date_cols: list = DATE_COLUMNS,
) -> pd.DataFrame:
    """Fast comparison between Current and Previous DataFrames for the same sheet."""
    out_cols = ["SheetName", "PatientID", "ColumnName", "Value_Current", "Value_Previous"]
    dfC = ensure_tpt_patientid(dfCurrent)
    dfP = ensure_tpt_patientid(dfPrevious)

    if id_col not in dfC.columns or id_col not in dfP.columns:
        return pd.DataFrame(columns=out_cols)

    if "SheetName" in dfC.columns:
        shname = ", ".join(dfC["SheetName"].astype(str).fillna("").unique())
    elif "SheetName" in dfP.columns:
        shname = ", ".join(dfP["SheetName"].astype(str).fillna("").unique())
    else:
        shname = "Unknown"

    dfC[id_col] = dfC[id_col].astype(str).str.strip()
    dfP[id_col] = dfP[id_col].astype(str).str.strip()

    for col in date_cols:
        if col in dfC.columns:
            dfC[col] = clean_datetime_series(dfC[col]).dt.strftime("%Y-%m-%d")
        if col in dfP.columns:
            dfP[col] = clean_datetime_series(dfP[col]).dt.strftime("%Y-%m-%d")

    records = []
    ids_current = set(dfC[id_col].dropna().unique()) - {"", "nan", "None", "<NA>"}
    ids_previous = set(dfP[id_col].dropna().unique()) - {"", "nan", "None", "<NA>"}

    for pid in sorted(ids_current - ids_previous):
        records.append([shname, pid, "_Not in Previous", pid, None])
    for pid in sorted(ids_previous - ids_current):
        records.append([shname, pid, "_Not in Current", None, pid])

    cols_current = set(dfC.columns) - {id_col, "SheetName"}
    cols_previous = set(dfP.columns) - {id_col, "SheetName"}

    for col in sorted(cols_current - cols_previous):
        records.append([shname, col, "Missing Column in dfPrevious", "Exists", "Missing"])
    for col in sorted(cols_previous - cols_current):
        records.append([shname, col, "Missing Column in dfCurrent", "Missing", "Exists"])

    common_cols = [c for c in dfC.columns if c in cols_previous and c != id_col]

    if common_cols:
        merged = pd.merge(
            dfC[[id_col] + common_cols],
            dfP[[id_col] + common_cols],
            on=id_col,
            how="inner",
            suffixes=("_Current", "_Previous"),
        )
        if not merged.empty:
            pids_arr = merged[id_col].tolist()
            for col in common_cols:
                col_c = f"{col}_Current"
                col_p = f"{col}_Previous"
                vals_c = merged[col_c].tolist()
                vals_p = merged[col_p].tolist()
                for pid, vc, vp in zip(pids_arr, vals_c, vals_p):
                    nc = normalize_value(vc)
                    np_val = normalize_value(vp)
                    if nc is None and np_val is None:
                        continue
                    if nc != np_val:
                        records.append([shname, pid, col, vc, vp])

    df_res = pd.DataFrame(records, columns=out_cols)
    return df_res.replace({"nan": "", "NaT": "", "<NA>": ""})


def compare_workbook_dicts(df_dict_curr: dict[str, pd.DataFrame], df_dict_prev: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare all matching sheets between Current and Previous report dictionaries."""
    out_cols = ["SheetName", "PatientID", "ColumnName", "Value_Current", "Value_Previous"]
    diffs = []
    for sheet_name, df1 in df_dict_curr.items():
        if sheet_name in df_dict_prev:
            df2 = df_dict_prev[sheet_name]
            if df1 is not None and not df1.empty and df2 is not None and not df2.empty:
                df_sheet_diff = compare_data(df1, df2)
                if not df_sheet_diff.empty:
                    diffs.append(df_sheet_diff)
    if not diffs:
        return pd.DataFrame(columns=out_cols)
    return pd.concat(diffs, axis=0, ignore_index=True)


# ==============================================================================
# 5. CBTBC (PGK DATA VERIFICATION) HELPERS
# ==============================================================================
def clean_mata_supabase_df(df_raw: pd.DataFrame, filter_vol_only: bool = True) -> pd.DataFrame:
    """
    Clean and standardize the Supabase `ygntbpro` DataFrame (which merges `YgnTBPro` and
    `Notified TB Register`) for CBTBC (PGK) comparison.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    dfmata = align_canonical_columns(df_raw)

    merged_to_base_map = {
        "Name_x": "Name",
        "Age_x": "Age",
        "Sex_x": "Sex",
        "Date_x": "Date",
        "Address_x": "Address",
        "Ptstsp_x": "Ptstsp",
        "PhoneNo_x": "PhoneNo",
        "Remark_x": "Remark",
    }
    for suff_col, base_col in merged_to_base_map.items():
        if suff_col in dfmata.columns and base_col not in dfmata.columns:
            dfmata[base_col] = dfmata[suff_col]

    if filter_vol_only and "VOL" in dfmata.columns:
        dfmata = dfmata[pd.to_numeric(dfmata["VOL"], errors="coerce") == 1].copy()

    expected_columns = [
        "PatientID", "Name", "Age", "Sex", "VOL", "Referralfor",
        "VolunteerName", "Organization", "TPTregno", "RegnoofindexTBcase",
        "TPTregistrationdate", "TPTregimen", "Date", "TownshipTBRegNumber",
        "TypeofTBTreatment", "Team", "Tsp", "Approach", "Visit_no", "Sr_No",
    ]
    for col in expected_columns:
        if col not in dfmata.columns:
            dfmata[col] = pd.NA

    extra_cols = [c for c in dfmata.columns if c not in expected_columns]
    dfmata = dfmata[expected_columns + extra_cols].copy()
    dfmata = dfmata.loc[:, ~dfmata.columns.duplicated()].copy()

    dfmata["Sex"] = dfmata["Sex"].replace({1: "M", 2: "F", "1": "M", "2": "F", "1.0": "M", "2.0": "F"})
    dfmata["Referralfor"] = dfmata["Referralfor"].replace(
        {1: "Presumptive", 2: "CI", "1": "Presumptive", "2": "CI", "1.0": "Presumptive", "2.0": "CI"}
    )
    dfmata["TypeofTBTreatment"] = dfmata["TypeofTBTreatment"].replace(
        {
            1: "DS-TB", 2: "DR-TB", 3: "TPT",
            "1": "DS-TB", "2": "DR-TB", "3": "TPT",
            "1.0": "DS-TB", "2.0": "DR-TB", "3.0": "TPT",
        }
    )

    dfmata = dfmata[
        dfmata["PatientID"].notna() & (dfmata["PatientID"].astype(str).str.strip() != "")
    ].copy()
    return dfmata


@st.cache_data(max_entries=2, show_spinner=False)
def load_and_clean_pgk_excel(file_bytes: bytes) -> pd.DataFrame:
    """Read `Presumptive` and `CI` sheets from CBTBC / PGK Excel bytes and standardize column names."""
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

    presumptive_rename_map = {
        "Name of presumptive TB patient": "Name",
        "TB Registeration No.": "TownshipTBRegNumber",
        "Diagnosis Result ": "TypeofTBTreatment",
        "Diagnosis Result": "TypeofTBTreatment",
        "Patients ID": "PatientID",
    }
    if "VL Name" in dfpgk_p.columns:
        presumptive_rename_map["VL Name"] = "VolunteerName"
    elif "Referred Volunteer Name" in dfpgk_p.columns:
        presumptive_rename_map["Referred Volunteer Name"] = "VolunteerName"

    dfpgk_p = dfpgk_p.rename(columns=presumptive_rename_map)

    ci_rename_map = {
        "Sex": "PatientSex",
        "Age": "PatientAge",
        "Name of Contact": "Name",
        "Sex2": "Sex",
        "Age3": "Age",
        "Name": "VolunteerName",
        "Result ": "TypeofTBTreatment",
        "Result": "TypeofTBTreatment",
        "Patients ID": "PatientID",
    }
    dfpgk_c = dfpgk_c.rename(columns=ci_rename_map)

    dfpgk_c["SheetName"] = "CI"
    dfpgk_p["SheetName"] = "Presumptive"

    dfpgk_c = dfpgk_c.loc[:, ~dfpgk_c.columns.duplicated()].copy()
    dfpgk_p = dfpgk_p.loc[:, ~dfpgk_p.columns.duplicated()].copy()

    dfpgk = pd.concat([dfpgk_c, dfpgk_p], ignore_index=True, sort=False)

    if "TypeofTBTreatment" in dfpgk.columns:
        dfpgk["TypeofTBTreatment"] = (
            dfpgk["TypeofTBTreatment"]
            .astype("string")
            .str.strip()
            .replace({"No active TB": pd.NA, "": pd.NA})
        )

    if "PatientID" in dfpgk.columns:
        dfpgk = dfpgk[
            dfpgk["PatientID"].notna() & (dfpgk["PatientID"].astype(str).str.strip() != "")
        ].copy()

    return dfpgk


def normalize_compact(val) -> str:
    if pd.isna(val):
        return ""
    s = "".join(str(val).split()).lower()
    return "" if s in ("nan", "none", "<na>") else s


def normalize_tokens(val) -> str:
    if pd.isna(val):
        return ""
    s = " ".join(str(val).split()).lower()
    return "" if s in ("nan", "none", "<na>") else s


def compute_similarity(val_a, val_b) -> float:
    compact_a, compact_b = normalize_compact(val_a), normalize_compact(val_b)
    token_a, token_b = normalize_tokens(val_a), normalize_tokens(val_b)
    if HAS_RAPIDFUZZ:
        score = max(
            fuzz.ratio(compact_a, compact_b),
            fuzz.token_sort_ratio(token_a, token_b),
        )
    else:
        score = difflib.SequenceMatcher(None, compact_a, compact_b).ratio() * 100
    return round(float(score), 1)


def values_differ_numeric(val_a, val_b) -> bool:
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
    check_missing_ids: bool = True,
) -> pd.DataFrame:
    mata = df_mata.copy()
    pgk = df_pgk.copy()

    mata["_join_id"] = mata[id_col_mata].astype(str).str.strip()
    pgk["_join_id"] = pgk[id_col_pgk].astype(str).str.strip()

    merged = mata.merge(
        pgk,
        on="_join_id",
        how="outer",
        suffixes=("_MATA", "_PGK"),
        indicator=True,
    )

    def resolve_col(col_name: str, side: str) -> str:
        suffixed = f"{col_name}_{side}"
        return suffixed if suffixed in merged.columns else col_name

    results = []
    for _, row in merged.iterrows():
        pid = row["_join_id"]

        if row["_merge"] == "left_only":
            if check_missing_ids:
                results.append(
                    {
                        "PatientID": pid,
                        "Difference": "PatientID missing in PGK",
                        "PGK": "",
                        "MATA": pid,
                        "Score": None,
                    }
                )
            continue

        if row["_merge"] == "right_only":
            if check_missing_ids:
                results.append(
                    {
                        "PatientID": pid,
                        "Difference": "PatientID missing in MATA",
                        "PGK": pid,
                        "MATA": "",
                        "Score": None,
                    }
                )
            continue

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
                    results.append(
                        {"PatientID": pid, "Difference": diff_label, "PGK": val_pgk, "MATA": val_mata, "Score": None}
                    )
            elif comp_mode == "Exact Text":
                if normalize_compact(val_mata) != normalize_compact(val_pgk):
                    results.append(
                        {"PatientID": pid, "Difference": diff_label, "PGK": val_pgk, "MATA": val_mata, "Score": None}
                    )
            else:
                if normalize_compact(val_mata) != normalize_compact(val_pgk):
                    results.append(
                        {
                            "PatientID": pid,
                            "Difference": diff_label,
                            "PGK": val_pgk,
                            "MATA": val_mata,
                            "Score": compute_similarity(val_mata, val_pgk),
                        }
                    )

    return pd.DataFrame(results, columns=["PatientID", "Difference", "PGK", "MATA", "Score"])


# ==============================================================================
# 6. INITIALIZE SESSION STATE & SHARED SIDEBAR SETTINGS
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_default_gender_model():
    """Load `gender_predictor_model.pkl` once across sessions using cross-platform Path resolution."""
    candidates = [
        APP_DIR / "gender_predictor_model.pkl",
        Path.cwd() / "gender_predictor_model.pkl",
        Path(tempfile.gettempdir()) / "gender_predictor_model.pkl",
    ]
    for path in candidates:
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                continue
    return None


def save_gender_model_cross_platform(pipeline) -> None:
    """Save trained model to app directory or OS temp directory (works on Windows, macOS, Linux, iOS)."""
    for target in [APP_DIR / "gender_predictor_model.pkl", Path(tempfile.gettempdir()) / "gender_predictor_model.pkl"]:
        try:
            joblib.dump(pipeline, target)
            return
        except Exception:
            continue


if "gender_model" not in st.session_state:
    cached_model = load_default_gender_model()
    st.session_state["gender_model"] = cached_model
    st.session_state["gender_model_source"] = (
        "Loaded from gender_predictor_model.pkl" if cached_model is not None else "Not loaded"
    )

default_url, default_key = get_default_supabase_credentials()

with st.sidebar:
    st.subheader("🗄️ Supabase Connection")
    supabase_url = st.text_input(
        "Supabase URL",
        value=default_url,
        placeholder="https://kocihpxevlowqbguhstf.supabase.co",
    )
    supabase_key = st.text_input(
        "Supabase Key",
        value=default_key,
        type="password",
        placeholder="Enter Supabase publishable / anon key",
    )

    if st.button("🔄 Refresh Supabase Cache", width="stretch"):
        fetch_supabase_table.clear()
        gc.collect()
        st.rerun()

    st.divider()
    st.subheader("🤖 Gender Prediction Model")
    st.caption(f"Status: **{st.session_state.get('gender_model_source', 'Not loaded')}**")
    uploaded_model_pkl = st.file_uploader(
        "Upload Pre-Trained Model (.pkl)",
        type=["pkl", "joblib"],
        key="sidebar_model_uploader",
    )
    if uploaded_model_pkl is not None:
        try:
            st.session_state["gender_model"] = joblib.load(BytesIO(uploaded_model_pkl.getvalue()))
            st.session_state["gender_model_source"] = f"Uploaded ({uploaded_model_pkl.name})"
            st.success("Loaded uploaded gender model!")
        except Exception as exc:
            st.error(f"Could not load model: {exc}")


# ==============================================================================
# 7. THREE SEPARATE PAGE FUNCTIONS
# ==============================================================================
def page_ppm_database_verification():
    """Page 1: PPM (Database Verification)"""
    st.title("📋 Page 1: PPM (Database Verification)")
    st.caption(
        "Verify PPM / MATA database records (`CheckData`, `CheckGender`, `CheckDifference`, and `checkUnique`), "
        "and run **Fuzzy Data Duplication Checking (`CheckDuplication`)** separately as an optional command."
    )

    src_col1, src_col2 = st.columns([3, 1])
    with src_col1:
        verification_source = st.radio(
            "Select Source Data for PPM Verification",
            options=["Existing Supabase Data (ygntbpro)", "Upload Excel File"],
            horizontal=True,
        )
    with src_col2:
        target_year = st.number_input("Target Validation Year", min_value=2020, max_value=2035, value=2026, step=1)

    if verification_source == "Existing Supabase Data (ygntbpro)":
        supa_ygn_table = st.text_input(
            "Supabase Merged Table Name (`YgnTBPro` + `Notified TB Register`)",
            value=SUPABASE_DEFAULT_TABLE,
        )
        uploaded_curr_excel = None
    else:
        supa_ygn_table = SUPABASE_DEFAULT_TABLE
        uploaded_curr_excel = st.file_uploader(
            "Upload Current PPM / MATA Report Excel Workbook (.xlsx, .xls, .xlsm)",
            type=["xlsx", "xls", "xlsm"],
            key="curr_report_uploader",
        )

    with st.expander("📂 Optional: Upload Previous Report Excel Workbook (for `CheckDifference`)", expanded=False):
        uploaded_prev_excel = st.file_uploader(
            "Upload Previous Report Excel (.xlsx, .xls, .xlsm)",
            type=["xlsx", "xls", "xlsm"],
            key="prev_report_uploader",
        )

    cmd_col1, cmd_col2, cmd_col3 = st.columns([2, 2, 1])
    with cmd_col1:
        run_ppm_btn = st.button(
            "🚀 1. Run PPM Verification (`CheckData`, `CheckGender`, `CheckDifference`, `checkUnique`)",
            type="primary",
            width="stretch",
        )
    with cmd_col2:
        run_dup_btn = st.button(
            "🔍 2. Optional Command: Run Data Duplication Check (`CheckDuplication`)",
            type="secondary",
            width="stretch",
        )
    with cmd_col3:
        dup_threshold = st.number_input(
            "Duplicate Score %",
            min_value=70,
            max_value=100,
            value=85,
            step=1,
            help="Minimum fuzzy similarity score (default 85%) for optional duplicate detection.",
        )

    def _load_ppm_combined_data():
        """Helper to load and prepare `df_combined` and `df_dict_current` from selected source."""
        df_dict_curr: dict[str, pd.DataFrame] = {}
        df_comb = pd.DataFrame()
        if verification_source == "Existing Supabase Data (ygntbpro)":
            if not supabase_url or not supabase_key:
                st.warning("Please enter your Supabase URL and Key in the sidebar.")
            else:
                try:
                    with st.spinner(f"Fetching merged table `{supa_ygn_table}` from Supabase..."):
                        ygn_df = fetch_supabase_table(supabase_url, supabase_key, table_name=supa_ygn_table.strip())
                        df_dict_curr["YgnTBPro"] = ygn_df
                        df_comb = prepare_combined_dataframe(ygn_df, None)
                    st.success(f"Loaded **{len(df_comb):,}** merged records from Supabase `{supa_ygn_table}`.")
                except Exception as exc:
                    st.error(f"Error loading `{supa_ygn_table}` from Supabase: {exc}")
        else:
            if uploaded_curr_excel is None:
                st.warning("Please upload a Current Report Excel file first.")
            else:
                try:
                    with st.spinner("Reading sheets from uploaded Current Report Excel..."):
                        expected_sheets = [
                            "YgnTBPro",
                            "Presumptive Register",
                            "Notified TB Register",
                            "Outcome",
                            "TPT Register",
                        ]
                        df_dict_curr = read_excel_workbook_sheets(
                            uploaded_curr_excel.getvalue(), candidate_sheets=expected_sheets
                        )
                        if "YgnTBPro" in df_dict_curr:
                            df_comb = prepare_combined_dataframe(
                                df_dict_curr["YgnTBPro"],
                                df_dict_curr.get("Notified TB Register"),
                            )
                        st.success(
                            f"Loaded sheets: **{', '.join(df_dict_curr.keys())}** "
                            f"({len(df_comb):,} combined rows)."
                        )
                except Exception as exc:
                    st.error(f"Failed to read Current Excel file: {exc}")
        return df_comb, df_dict_curr

    # Command 1: Run Standard PPM Verification (without heavy fuzzy duplication)
    if run_ppm_btn:
        df_combined, df_dict_current = _load_ppm_combined_data()
        if not df_combined.empty:
            st.session_state["ppm_df_combined"] = df_combined
            with st.spinner("Running PPM rule validation, gender check, unique profiling, and report diff..."):
                validator = TBDataValidator(target_year=int(target_year))
                dfValidation = validator.validate(df_combined)

                dfGenderPredictor = run_gender_prediction_check(
                    df_combined,
                    model=st.session_state.get("gender_model"),
                    id_col="PatientID",
                    name_col="Name_x",
                    sex_col="Sex_x",
                    min_confidence=0.5,
                )

                df_unique = compute_unique_values_table(df_combined)

                if uploaded_prev_excel is not None:
                    try:
                        df_dict_prev = read_excel_workbook_sheets(
                            uploaded_prev_excel.getvalue(),
                            candidate_sheets=[
                                "YgnTBPro",
                                "Presumptive Register",
                                "Notified TB Register",
                                "Outcome",
                                "TPT Register",
                            ],
                        )
                        df_diff = compare_workbook_dicts(df_dict_current, df_dict_prev)
                    except Exception as exc:
                        st.warning(f"Could not compute CheckDifference: {exc}")
                        df_diff = pd.DataFrame()
                else:
                    df_diff = pd.DataFrame()

                existing_dup = (
                    st.session_state["ppm_results"].get("CheckDuplication", pd.DataFrame())
                    if "ppm_results" in st.session_state
                    else pd.DataFrame()
                )
                st.session_state["ppm_results"] = {
                    "CheckData": dfValidation,
                    "CheckGender": dfGenderPredictor,
                    "CheckDuplication": existing_dup,
                    "CheckDifference": df_diff,
                    "checkUnique": df_unique,
                }
                gc.collect()

    # Command 2 (Separate Optional Command): Run Data Duplication Checking
    if run_dup_btn:
        if "ppm_df_combined" in st.session_state and not st.session_state["ppm_df_combined"].empty:
            df_combined = st.session_state["ppm_df_combined"]
        else:
            df_combined, _ = _load_ppm_combined_data()
            if not df_combined.empty:
                st.session_state["ppm_df_combined"] = df_combined

        if not df_combined.empty:
            with st.spinner(f"Running optional fuzzy data duplication check (threshold >= {int(dup_threshold)}%)..."):
                if "Reasonforexamination" in df_combined.columns:
                    dfpresumptive = df_combined[
                        pd.to_numeric(df_combined["Reasonforexamination"], errors="coerce").isin([1, 2])
                    ].copy()
                    dfDuplication = fuzzyMatchingData(dfpresumptive, score_threshold=int(dup_threshold))
                else:
                    dfDuplication = pd.DataFrame()

                if "ppm_results" not in st.session_state:
                    st.session_state["ppm_results"] = {
                        "CheckData": pd.DataFrame(),
                        "CheckGender": pd.DataFrame(),
                        "CheckDuplication": dfDuplication,
                        "CheckDifference": pd.DataFrame(),
                        "checkUnique": pd.DataFrame(),
                    }
                else:
                    st.session_state["ppm_results"]["CheckDuplication"] = dfDuplication
                gc.collect()
            st.success(f"Completed Data Duplication Check: found **{len(dfDuplication):,}** potential duplicate pairs.")

    if "ppm_results" in st.session_state:
        res = st.session_state["ppm_results"]
        dfValidation = res["CheckData"]
        dfGenderPredictor = res["CheckGender"]
        dfDuplication = res["CheckDuplication"]
        df_diff = res["CheckDifference"]
        df_unique = res["checkUnique"]

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("CheckData Issues", f"{len(dfValidation):,}")
        k2.metric("CheckGender Mismatches", f"{len(dfGenderPredictor):,}")
        k3.metric("CheckDuplication Pairs", f"{len(dfDuplication):,}")
        k4.metric("CheckDifference Rows", f"{len(df_diff):,}")
        k5.metric("Unique Columns Profiled", f"{len(df_unique):,}")

        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        excel_out_bytes = generate_multisheet_excel_bytes(res)

        st.download_button(
            label="📥 Generate & Download PPM Verification Excel (`CheckData`, `CheckGender`, `CheckDuplication`, `CheckDifference`, `checkUnique`)",
            data=excel_out_bytes,
            file_name=f"vsoutput_PPM_DatabaseVerification_{timestamp_str}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            width="stretch",
        )

        tab_val, tab_gen, tab_dup, tab_diff, tab_uniq = st.tabs(
            [
                f"📋 CheckData ({len(dfValidation):,})",
                f"🚻 CheckGender ({len(dfGenderPredictor):,})",
                f"🔍 CheckDuplication ({len(dfDuplication):,})",
                f"🔄 CheckDifference ({len(df_diff):,})",
                f"📊 checkUnique ({len(df_unique):,})",
            ]
        )

        with tab_val:
            st.dataframe(dfValidation, width="stretch", hide_index=True, height=480)

        with tab_gen:
            if st.session_state.get("gender_model") is None:
                st.info(
                    "No gender prediction model is currently loaded. Upload a `.pkl` model in the sidebar "
                    "or train one on **Page 3: Train Gender Prediction Model**."
                )
            else:
                st.dataframe(dfGenderPredictor, width="stretch", hide_index=True, height=480)

        with tab_dup:
            if dfDuplication.empty:
                st.info(
                    "Click **🔍 2. Optional Command: Run Data Duplication Check (`CheckDuplication`)** above "
                    "to run fuzzy duplicate matching when needed."
                )
            else:
                st.dataframe(dfDuplication, width="stretch", hide_index=True, height=480)

        with tab_diff:
            if df_diff.empty and uploaded_prev_excel is None:
                st.info("Upload a Previous Report Excel file above to compare Current vs. Previous sheets.")
            else:
                st.dataframe(df_diff, width="stretch", hide_index=True, height=480)

        with tab_uniq:
            st.dataframe(df_unique, width="stretch", hide_index=True, height=480)


def page_cbtbc_pgk_verification():
    """Page 2: CBTBC (PGK Data Verification)"""
    st.title("🔄 Page 2: CBTBC (PGK Data Verification)")
    st.caption(
        "Automatically selects MATA data from Supabase table **`ygntbpro`**, pairs columns against your uploaded "
        "CBTBC (PGK) Excel file (`Presumptive` & `CI` sheets), displays discrepancies in an interactive table, "
        "and generates an Excel verification report."
    )

    filter_vol_only = st.checkbox("Filter Supabase `ygntbpro` to `VOL == 1` only", value=True)

    col_left, col_right = st.columns(2)
    dfmata = pd.DataFrame()
    dfpgk = pd.DataFrame()

    with col_left:
        st.subheader("1️⃣ MATA Data: Supabase (`ygntbpro`)")
        if not supabase_url or not supabase_key:
            st.warning("Please provide `SUPABASE_URL` and `SUPABASE_KEY` in the sidebar or `st.secrets`.")
        else:
            try:
                with st.spinner("Fetching `ygntbpro` records from Supabase..."):
                    df_supa_raw = fetch_supabase_table(
                        supabase_url, supabase_key, table_name=SUPABASE_DEFAULT_TABLE
                    )
                    dfmata = clean_mata_supabase_df(df_supa_raw, filter_vol_only=filter_vol_only)
                st.success(
                    f"Loaded **{len(dfmata):,}** valid rows from Supabase `ygntbpro` "
                    f"(Total raw rows: {len(df_supa_raw):,})."
                )
                with st.expander("Preview Supabase `ygntbpro` Data", expanded=False):
                    st.dataframe(dfmata.head(50), width="stretch")
            except Exception as exc:
                st.error(f"Failed to load `ygntbpro` from Supabase: {exc}")

    with col_right:
        st.subheader("2️⃣ CBTBC (PGK) Data: Upload Excel File")
        uploaded_pgk_file = st.file_uploader(
            "Upload PGK Excel Workbook (must contain `Presumptive` and `CI` sheets)",
            type=["xlsx", "xls", "xlsm"],
            key="pgk_uploader",
        )
        if uploaded_pgk_file is not None:
            try:
                with st.spinner("Reading and standardizing PGK Excel sheets (`Presumptive` & `CI`)..."):
                    dfpgk = load_and_clean_pgk_excel(uploaded_pgk_file.getvalue())
                st.success(f"Loaded **{len(dfpgk):,}** valid rows from PGK Excel (`Presumptive` + `CI`).")
                with st.expander("Preview Cleaned PGK Excel Data", expanded=False):
                    st.dataframe(dfpgk.head(50), width="stretch")
            except Exception as exc:
                st.error(f"Error reading PGK Excel file: {exc}")

    st.subheader("3️⃣ Pair-Wise Column Mapping for CBTBC Data Checking")

    if dfmata.empty or dfpgk.empty:
        st.info(
            "Once both Supabase `ygntbpro` data and the CBTBC (PGK) Excel file are loaded above, "
            "configure your column pairs and view the verification results here."
        )
    else:
        mata_cols = list(dfmata.columns)
        pgk_cols = list(dfpgk.columns)

        key_col1, key_col2, key_col3 = st.columns([2, 2, 2])
        with key_col1:
            id_col_mata = st.selectbox(
                "🔑 Supabase (`ygntbpro`) Primary ID Column",
                options=mata_cols,
                index=mata_cols.index("PatientID") if "PatientID" in mata_cols else 0,
            )
        with key_col2:
            id_col_pgk = st.selectbox(
                "🔑 PGK Excel Primary ID Column",
                options=pgk_cols,
                index=pgk_cols.index("PatientID") if "PatientID" in pgk_cols else 0,
            )
        with key_col3:
            st.write("")
            st.write("")
            check_missing_ids = st.checkbox("Report missing PatientIDs in either source", value=True)

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
                    initial_rows.append(
                        {
                            "Enabled": True,
                            "Supabase_Column": m_c,
                            "PGK_Column": p_c,
                            "Difference_Label": label,
                            "Comparison_Type": c_type,
                        }
                    )
            st.session_state.column_pairs_df = pd.DataFrame(initial_rows)

        with st.expander("➕ Add Custom Column Pair", expanded=False):
            add_c1, add_c2, add_c3, add_c4, add_c5 = st.columns([2, 2, 2, 2, 1])
            with add_c1:
                new_supa_col = st.selectbox("Supabase (`ygntbpro`) Column", options=mata_cols, key="new_supa_col")
            with add_c2:
                new_pgk_col = st.selectbox(
                    "PGK Excel Column",
                    options=pgk_cols,
                    index=pgk_cols.index(new_supa_col) if new_supa_col in pgk_cols else 0,
                    key="new_pgk_col",
                )
            with add_c3:
                new_label = st.text_input("Difference Label", value=new_supa_col, key="new_diff_label")
            with add_c4:
                new_comp_type = st.selectbox(
                    "Comparison Type",
                    options=["Fuzzy Text (with Score)", "Exact Text", "Numeric / Age"],
                    key="new_comp_type",
                )
            with add_c5:
                st.write("")
                st.write("")
                if st.button("Add Pair", width="stretch"):
                    new_row = pd.DataFrame(
                        [
                            {
                                "Enabled": True,
                                "Supabase_Column": new_supa_col,
                                "PGK_Column": new_pgk_col,
                                "Difference_Label": new_label.strip() or f"{new_supa_col} vs {new_pgk_col}",
                                "Comparison_Type": new_comp_type,
                            }
                        ]
                    )
                    st.session_state.column_pairs_df = pd.concat(
                        [st.session_state.column_pairs_df, new_row], ignore_index=True
                    )
                    st.rerun()

        edited_pairs_df = st.data_editor(
            st.session_state.column_pairs_df,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "Enabled": st.column_config.CheckboxColumn("Check?", default=True),
                "Supabase_Column": st.column_config.SelectboxColumn(
                    "Supabase (ygntbpro) Column", options=mata_cols, required=True
                ),
                "PGK_Column": st.column_config.SelectboxColumn("PGK Excel Column", options=pgk_cols, required=True),
                "Difference_Label": st.column_config.TextColumn("Difference Label", required=True),
                "Comparison_Type": st.column_config.SelectboxColumn(
                    "Comparison Type",
                    options=["Fuzzy Text (with Score)", "Exact Text", "Numeric / Age"],
                    default="Fuzzy Text (with Score)",
                    required=True,
                ),
            },
            key="pairs_editor",
        )

        st.subheader("4️⃣ CBTBC Verification Output Table & Excel Export")
        pair_records = edited_pairs_df.dropna(subset=["Supabase_Column", "PGK_Column"]).to_dict(orient="records")

        with st.spinner("Comparing paired columns between Supabase `ygntbpro` and CBTBC (PGK) Excel..."):
            df_difference = run_pairwise_verification(
                df_mata=dfmata,
                df_pgk=dfpgk,
                id_col_mata=id_col_mata,
                id_col_pgk=id_col_pgk,
                column_pairs=pair_records,
                check_missing_ids=check_missing_ids,
            )

        missing_in_pgk_count = (
            int((df_difference["Difference"] == "PatientID missing in PGK").sum()) if not df_difference.empty else 0
        )
        missing_in_mata_count = (
            int((df_difference["Difference"] == "PatientID missing in MATA").sum()) if not df_difference.empty else 0
        )
        field_mismatch_count = int(len(df_difference) - missing_in_pgk_count - missing_in_mata_count)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Discrepancies", f"{len(df_difference):,}")
        m2.metric("Column Value Mismatches", f"{field_mismatch_count:,}")
        m3.metric("Missing in PGK", f"{missing_in_pgk_count:,}")
        m4.metric("Missing in Supabase (MATA)", f"{missing_in_mata_count:,}")

        filter_col1, filter_col2, export_col = st.columns([2, 2, 2])
        diff_types = sorted(df_difference["Difference"].dropna().unique().tolist()) if not df_difference.empty else []
        with filter_col1:
            selected_diff_types = st.multiselect("Filter by Difference Type", options=diff_types, default=diff_types)
        with filter_col2:
            search_query = st.text_input("Search by PatientID or Value", value="", placeholder="e.g., KMD26 or Aye...")

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
            excel_filename = f"vsoutput_CBTBC_PGK_Verification_{timestamp_create}.xlsx"
            export_scope = st.radio(
                "Excel Export Scope", options=["All Discrepancies", "Filtered Table Only"], horizontal=True
            )
            df_to_export = df_difference if export_scope == "All Discrepancies" else df_filtered
            excel_bytes = generate_multisheet_excel_bytes({"CheckData": df_to_export})

            st.download_button(
                label=f"📥 Generate & Download CBTBC Excel ({len(df_to_export):,} rows)",
                data=excel_bytes,
                file_name=excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                width="stretch",
            )

        st.dataframe(df_filtered, width="stretch", hide_index=True, height=500)


def page_train_gender_model():
    """Page 3: Train Gender Prediction Model"""
    st.title("🧠 Page 3: Train Gender Prediction Model (Supabase)")
    st.caption(
        "Manually train or re-train the gender prediction model (`TfidfVectorizer` + `NameFeatureExtractor` + `RandomForestClassifier`) "
        "using existing data from your Supabase table **`ygntbpro`**, evaluate hold-out accuracy, and download `gender_predictor_model.pkl`."
    )

    if not supabase_url or not supabase_key:
        st.warning(
            "Please enter your Supabase URL and Key in the sidebar (or configure `st.secrets`) to load training data."
        )
    else:
        t_col1, t_col2, t_col3 = st.columns([2, 1, 1])
        with t_col1:
            train_table_name = st.text_input("Supabase Table for Training", value=SUPABASE_DEFAULT_TABLE)
        with t_col2:
            test_size_ratio = st.slider("Test Split Ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        with t_col3:
            n_estimators = st.number_input(
                "RandomForest Trees (`n_estimators`)", min_value=50, max_value=500, value=200, step=50
            )

        try:
            with st.spinner(f"Fetching training dataset from Supabase `{train_table_name}`..."):
                df_train_raw = fetch_supabase_table(supabase_url, supabase_key, table_name=train_table_name.strip())
                df_train_raw = align_canonical_columns(df_train_raw)
        except Exception as exc:
            st.error(f"Failed to fetch Supabase table `{train_table_name}`: {exc}")
            df_train_raw = pd.DataFrame()

        if not df_train_raw.empty:
            cols_available = list(df_train_raw.columns)

            default_name_idx = (
                cols_available.index("Name_x")
                if "Name_x" in cols_available
                else (cols_available.index("Name") if "Name" in cols_available else 0)
            )
            default_sex_idx = (
                cols_available.index("Sex_x")
                if "Sex_x" in cols_available
                else (cols_available.index("Sex") if "Sex" in cols_available else 0)
            )

            c_map1, c_map2, c_map3 = st.columns(3)
            with c_map1:
                name_col_train = st.selectbox(
                    "Select Name Column",
                    options=cols_available,
                    index=default_name_idx,
                )
            with c_map2:
                sex_col_train = st.selectbox(
                    "Select Sex / Gender Label Column",
                    options=cols_available,
                    index=default_sex_idx,
                )
            with c_map3:
                dedup_names = st.checkbox("Deduplicate (Name, Sex) pairs before training", value=True)

            df_prep = df_train_raw[[name_col_train, sex_col_train]].copy()
            df_prep.columns = ["name", "gender"]
            df_prep["name"] = df_prep["name"].fillna("").astype(str).str.strip()
            df_prep["gender"] = (
                df_prep["gender"]
                .astype(str)
                .str.strip()
                .replace({"1": "M", "2": "F", "1.0": "M", "2.0": "F", "Male": "M", "Female": "F", "m": "M", "f": "F"})
            )
            df_prep = df_prep[
                df_prep["name"].ne("")
                & ~df_prep["name"].str.lower().isin(["nan", "none", "<na>"])
                & df_prep["gender"].isin(["M", "F"])
            ].copy()

            if dedup_names:
                df_prep = df_prep.drop_duplicates(subset=["name", "gender"])

            m_count = int((df_prep["gender"] == "M").sum())
            f_count = int((df_prep["gender"] == "F").sum())

            s1, s2, s3 = st.columns(3)
            s1.metric("Valid Training Samples", f"{len(df_prep):,}")
            s2.metric("Male (`M`) Samples", f"{m_count:,}")
            s3.metric("Female (`F`) Samples", f"{f_count:,}")

            if st.button("🚀 Train Gender Prediction Model Now", type="primary", width="stretch"):
                if len(df_prep) < 20 or m_count < 5 or f_count < 5:
                    st.error("Insufficient labeled `M` / `F` records to train a reliable model.")
                else:
                    with st.spinner("Training TF-IDF + NameFeatureExtractor + RandomForest pipeline..."):
                        X = df_prep["name"].values
                        y = df_prep["gender"].values

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=float(test_size_ratio), random_state=42, stratify=y
                        )

                        pipeline = build_gender_pipeline(n_estimators=int(n_estimators), random_state=42)
                        pipeline.fit(X_train, y_train)

                        y_pred = pipeline.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

                        # Refit on 100% of clean labeled data for production use
                        pipeline.fit(X, y)

                        st.session_state["gender_model"] = pipeline
                        st.session_state["gender_model_source"] = (
                            f"Trained on Supabase `{train_table_name}` ({len(df_prep):,} samples, Acc={acc:.2%})"
                        )
                        st.session_state["gender_train_metrics"] = {
                            "accuracy": acc,
                            "report_df": pd.DataFrame(report_dict)
                            .T.reset_index()
                            .rename(columns={"index": "Class / Metric"}),
                            "samples": len(df_prep),
                        }
                        save_gender_model_cross_platform(pipeline)
                        gc.collect()

                    st.success(
                        f"Model trained and activated! Hold-out Test Accuracy: **{acc:.2%}** on {len(X_test):,} test names."
                    )

            if "gender_train_metrics" in st.session_state:
                metrics = st.session_state["gender_train_metrics"]
                st.subheader("📊 Hold-Out Evaluation Report")
                st.metric("Hold-Out Test Accuracy", f"{metrics['accuracy']:.2%}")
                st.dataframe(metrics["report_df"], width="stretch", hide_index=True)

            if st.session_state.get("gender_model") is not None:
                st.subheader("💾 Download & Test Active Gender Model")
                model_buf = BytesIO()
                joblib.dump(st.session_state["gender_model"], model_buf)

                dl_col, test_col = st.columns([1, 2])
                with dl_col:
                    st.download_button(
                        label="📥 Download `gender_predictor_model.pkl`",
                        data=model_buf.getvalue(),
                        file_name="gender_predictor_model.pkl",
                        mime="application/octet-stream",
                        type="primary",
                        width="stretch",
                    )
                with test_col:
                    sample_name = st.text_input("Test a Name Live", value="Aye Aye Myint")
                    if sample_name.strip():
                        pred_g, conf_g = NameFeatureExtractor.predict_gender(
                            sample_name.strip(), model=st.session_state["gender_model"]
                        )
                        st.info(
                            f"Predicted Gender for **{sample_name}**: **`{pred_g}`** (Confidence: **{conf_g:.1%}**)"
                        )


# ==============================================================================
# 8. MULTI-PAGE ROUTER (NATIVE `st.navigation` WITH FALLBACK)
# ==============================================================================
if hasattr(st, "navigation") and hasattr(st, "Page"):
    pg = st.navigation(
        [
            st.Page(
                page_ppm_database_verification,
                title="1. PPM (Database Verification)",
                icon="📋",
                url_path="ppm-database-verification",
                default=True,
            ),
            st.Page(
                page_cbtbc_pgk_verification,
                title="2. CBTBC (PGK Data Verification)",
                icon="🔄",
                url_path="cbtbc-pgk-verification",
            ),
            st.Page(
                page_train_gender_model,
                title="3. Train Gender Prediction Model",
                icon="🧠",
                url_path="train-gender-model",
            ),
        ]
    )
    pg.run()
else:
    with st.sidebar:
        selected_page = st.radio(
            "📌 Navigation",
            options=[
                "1. PPM (Database Verification)",
                "2. CBTBC (PGK Data Verification)",
                "3. Train Gender Prediction Model",
            ],
        )
    if selected_page == "1. PPM (Database Verification)":
        page_ppm_database_verification()
    elif selected_page == "2. CBTBC (PGK Data Verification)":
        page_cbtbc_pgk_verification()
    else:
        page_train_gender_model()