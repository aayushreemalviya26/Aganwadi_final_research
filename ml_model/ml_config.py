import os

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "children_all_scenarios.csv")

# --- TARGET CONFIGURATION ---
TARGET_COL = "final_outcome"
TARGET_MAP = {
    "RECOVERED": 0,
    "DROPOUT": 1,
    "NON_RESPONDER": 2
}

# --- GROUPING CONFIGURATION ---
GROUP_COL = "replication"

# --- ENTRY-LEVEL BASELINE FEATURES ---
CATEGORICAL_FEATURES = [
    "initial_health_state",
    "initial_medical_complication",
    "scenario"
]

NUMERICAL_FEATURES = [
    "entry_day",
    "nrc_capacity",
    "family_referral_compliance_parameter",
    "followup_attendance_parameter"
]

FEATURE_COLS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES