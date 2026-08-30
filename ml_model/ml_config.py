import os

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "children_all_scenarios.csv")

# --- TARGET CONFIGURATION ---
TARGET_COL = "final_outcome"
TARGET_MAP = {
    "RECOV": 0,
    "DROPO": 1,
    "NON_R": 2
}

# --- GROUPING CONFIGURATION (For Group K-Fold / Seed Splitting) ---
GROUP_COL = "replication_id"

# --- ENTRY-LEVEL FEATURES (Strictly Baseline/Early Information Only) ---
CATEGORICAL_FEATURES = [
    "initial_health_status",
    "initial_medical_complication",
    "scenario"
]

NUMERICAL_FEATURES = [
    "entry_day",
    "nrc_capacity",
    "family_readiness_parameter",
    "followup_attendance_parameter"
]

FEATURE_COLS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES