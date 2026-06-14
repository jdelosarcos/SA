import re
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
TARGET = "degree_status"
CLASSES_2 = ["Completed", "Stopped"]
CLASSES_3 = ["Completed", "Stopped", "Shifted"]

RENAME_MAP = {
    "unnamed_0": "id",
    "name_for_female_if_married_already_use_your_maiden_name": "name",
    "age_when_you_are_in_first_year_college": "age_first_year",
    "name_address_of_school_attended_in_high_school": "high_school",
    "general_average_obtained_in_high_school": "hs_average_raw",
    "civil_status_when_entered_in_college_asscat": "civil_status",
    "while_studying_in_asscat_are_you_staying_in": "residence",
    "extra_curricular_activities_involvement_check_all_that_applies": "extracurricular",
    "mother_occupation_when_you_are_still_in_college": "mother_occupation",
    "father_occupation_when_you_are_still_in_college": "father_occupation",
    "family_monthly_gross_income_when_you_are_still_in_college": "family_income",
    "strand_enrolled_in_senior_high": "shs_strand",
    "scholarship_availed_in_college_just_write_none_if_no_scholarship_availed": "scholarship",
    "grade_obtained_in_intro_to_computer_fundamentals_operations_or_introduction_to_computing": "intro_computing_raw",
    "grade_obtained_in_fundamentals_of_programming": "programming_raw",
    "general_grade_average_obtained_in_1st_year_college": "gwa_1st_year_raw",
    "year_started_in_college_asscat": "year_started",
    "year_graduated_in_college_asscat_if_stopped_or_shifted_to_other_course_pls_indicate_and_also_the_year_ex_stopped_2023": "year_graduated_raw",
    "ethnicity_ex_manobo_kamayo_etc_or_surigaonon_agusanon_bol_anon_cebuano_etc": "ethnicity",
    "course": "course",
}

ENROLLMENT_FEATURES = [
    "age_first_year", "hs_average", "income_rank", "extracurricular_count",
    "civil_status", "residence", "family_income", "shs_strand", "has_scholarship",
    "has_extracurricular", "ethnicity", "course", "mother_occ_group", "father_occ_group",
    "age_first_year_missing", "hs_average_missing", "income_rank_missing",
]

ACADEMIC_FEATURES = ENROLLMENT_FEATURES + [
    "intro_computing", "programming", "gwa_1st_year",
    "intro_performance", "programming_performance", "performance_1st_year",
    "intro_computing_missing", "programming_missing", "gwa_1st_year_missing",
]


def clean_colname(col):
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")


def midpoint_from_range(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower().strip()
    nums = re.findall(r"\d+\.?\d*", text)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def normalize_text(x):
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none", "n/a", "na", "no answer", "null"]:
        return "Unknown"
    return re.sub(r"\s+", " ", s)


def extract_year(x):
    if pd.isna(x):
        return np.nan
    m = re.findall(r"(?:19|20)\d{2}", str(x))
    return float(m[0]) if m else np.nan


def derive_degree_status(x):
    if pd.isna(x):
        return "Other/Ongoing/Unknown"
    s = str(x).strip().lower()
    if any(k in s for k in ["shift", "shif"]):
        return "Shifted"
    if any(k in s for k in ["stop", "drop", "withdraw", "quit"]):
        return "Stopped"
    if any(k in s for k in ["enrolled", "ongoing", "currently", "continuing"]):
        return "Other/Ongoing/Unknown"
    if "graduate" in s:
        return "Completed"
    if not pd.isna(extract_year(s)):
        return "Completed"
    return "Other/Ongoing/Unknown"


def categorize_gwa_ph(value):
    if pd.isna(value):
        return "Unknown"
    if value <= 1.75:
        return "High performer"
    if value <= 2.50:
        return "Average performer"
    return "At-risk performer"


def income_rank(x):
    s = normalize_text(x).lower().replace(",", "")
    if "below 10000" in s or "10000 and below" in s or "10 000 and below" in s:
        return 1
    if "above 10000" in s and "below 20000" in s:
        return 2
    if "20000" in s and "30000" in s:
        return 3
    if "30000" in s and "40000" in s:
        return 4
    if "40000" in s and "50000" in s:
        return 5
    if "above 50000" in s or "50000 above" in s:
        return 6
    return np.nan


def has_scholarship(x):
    s = normalize_text(x).lower()
    if s in ["unknown", "none", "no", "n/a", "na"] or "no scholar" in s:
        return "No"
    return "Yes"


def extracurricular_count(x):
    s = normalize_text(x).lower()
    if s in ["unknown", "none", "no", "no activities", "no activity"] or "no activ" in s:
        return 0
    return max(1, len([p for p in re.split(r",|;|/", s) if p.strip()]))


def group_occupation(x):
    s = normalize_text(x).lower()
    if s == "unknown":
        return "Unknown"
    if any(k in s for k in ["farmer", "farm", "laborer", "labour"]):
        return "Farming/Labor"
    if any(k in s for k in ["teacher", "government", "employee", "office", "clerk", "nia", "deped", "barangay"]):
        return "Employed/Government"
    if any(k in s for k in ["business", "vendor", "store", "self", "entrepreneur"]):
        return "Business/Self-employed"
    if any(k in s for k in ["house", "wife", "home"]):
        return "Homemaker"
    if any(k in s for k in ["ofw", "abroad", "seaman"]):
        return "Overseas/Seafarer"
    return "Other"


def rare_group(series, min_count=4):
    s = series.astype(str).fillna("Unknown")
    counts = s.value_counts()
    rare = counts[counts < min_count].index
    return s.where(~s.isin(rare), "Other/Rare")


def load_excel(file):
    return pd.read_excel(file)


def clean_student_data(raw_df):
    clean_df = raw_df.copy()
    clean_df.columns = [clean_colname(c) for c in clean_df.columns]
    clean_df = clean_df.rename(columns=RENAME_MAP)

    required_defaults = {
        "year_graduated_raw": np.nan, "year_started": np.nan, "scholarship": "Unknown",
        "extracurricular": "Unknown", "mother_occupation": "Unknown", "father_occupation": "Unknown",
        "family_income": "Unknown", "course": "Unknown", "ethnicity": "Unknown", "residence": "Unknown",
        "civil_status": "Unknown", "shs_strand": "Unknown", "high_school": "Unknown",
    }
    for col, default in required_defaults.items():
        if col not in clean_df.columns:
            clean_df[col] = default

    for col in ["hs_average_raw", "intro_computing_raw", "programming_raw", "gwa_1st_year_raw"]:
        if col in clean_df.columns:
            clean_df[col.replace("_raw", "")] = clean_df[col].apply(midpoint_from_range)
        else:
            clean_df[col.replace("_raw", "")] = np.nan

    clean_df["age_first_year"] = pd.to_numeric(clean_df.get("age_first_year", np.nan), errors="coerce")
    clean_df["year_started"] = pd.to_numeric(clean_df.get("year_started", np.nan), errors="coerce")
    clean_df["year_graduated_numeric"] = clean_df["year_graduated_raw"].apply(extract_year)
    clean_df["degree_status"] = clean_df["year_graduated_raw"].apply(derive_degree_status)
    clean_df["time_to_completion"] = clean_df["year_graduated_numeric"] - clean_df["year_started"]
    clean_df["on_time_completion"] = np.where(
        (clean_df["degree_status"] == "Completed") & (clean_df["time_to_completion"] <= 4), "On-time",
        np.where(clean_df["degree_status"] == "Completed", "Delayed", "Not completed")
    )

    clean_df["has_scholarship"] = clean_df["scholarship"].apply(has_scholarship)
    clean_df["extracurricular_count"] = clean_df["extracurricular"].apply(extracurricular_count)
    clean_df["has_extracurricular"] = np.where(clean_df["extracurricular_count"] > 0, "Yes", "No")
    clean_df["income_rank"] = clean_df["family_income"].apply(income_rank)
    clean_df["mother_occ_group"] = clean_df["mother_occupation"].apply(group_occupation)
    clean_df["father_occ_group"] = clean_df["father_occupation"].apply(group_occupation)
    clean_df["performance_1st_year"] = clean_df["gwa_1st_year"].apply(categorize_gwa_ph)
    clean_df["programming_performance"] = clean_df["programming"].apply(categorize_gwa_ph)
    clean_df["intro_performance"] = clean_df["intro_computing"].apply(categorize_gwa_ph)

    base_cat = ["civil_status", "residence", "family_income", "shs_strand", "scholarship", "ethnicity", "course", "high_school"]
    for c in base_cat:
        clean_df[c] = clean_df[c].apply(normalize_text)

    for c in ["shs_strand", "ethnicity", "residence", "family_income", "mother_occ_group", "father_occ_group", "course"]:
        clean_df[c] = rare_group(clean_df[c], min_count=4)

    for c in ["age_first_year", "hs_average", "income_rank", "intro_computing", "programming", "gwa_1st_year"]:
        clean_df[f"{c}_missing"] = clean_df[c].isna().astype(int)

    return clean_df


def select_modeling_classes(clean_df):
    counts = clean_df[TARGET].value_counts()
    classes = [c for c in CLASSES_3 if counts.get(c, 0) >= 2]
    if "Shifted" in classes and counts.get("Shifted", 0) < 5:
        classes = [c for c in classes if c != "Shifted"]
    if len(classes) < 2:
        raise ValueError("At least two target classes with adequate samples are required for modeling.")
    return classes


def get_modeling_data(clean_df, feature_set="academic", classes=None):
    if classes is None:
        classes = select_modeling_classes(clean_df)
    df = clean_df[clean_df[TARGET].isin(classes)].copy()
    features = ENROLLMENT_FEATURES if feature_set == "enrollment" else ACADEMIC_FEATURES
    features = [f for f in features if f in df.columns]
    X = df[features].copy()
    y = df[TARGET].copy()
    return X, y, features, df, classes
