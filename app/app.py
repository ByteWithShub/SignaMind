"""
|| ASL Learning Intelligence Dashboard ||

A behavior-aware, data-driven ASL learning system that combines computer vision,
machine learning, and Business Intelligence principles to enhance user learning outcomes.

This application goes beyond static prediction by modeling user interaction patterns,
identifying failure trends, and delivering adaptive, confidence-aware feedback to guide
skill development in American Sign Language (ASL) fingerspelling.

Core Capabilities:
  - Real-time ASL sign recognition using MobileNetV2
  - BI-driven analytics for confidence, accuracy, and user performance trends
  - Curriculum-guided learning (easy to hard progression)
  - Confusion pattern detection and targeted practice modes
  - Word construction and sequential learning evaluation
  - Typewriter (Sign-to-Text) mode with intelligent suggestions

Learning Intelligence Features:
  - Anti-Frustration Logic: detects repeated failures and provides corrective guidance
  - Adaptive Feedback System: interprets predictions using confidence thresholds
  - XP & Gamification Engine: engagement through levels, streaks, and achievements
  - Daily Streak Tracking (DAU): persistent user engagement metrics
  - Behavioral Insights: tracks mistakes, confidence drops, and learning patterns

Objective:
  To bridge the communication gap between hearing and non-hearing communities by
  leveraging AI to create an intelligent, interpretable, and user-adaptive ASL
  learning experience.

Author : Shubhangi Singh
Model  : MobileNetV2 (model3_mobilenetv2.keras)
Classes: 36 (A-Z, 0-9)
"""

#Import statements for necessary libraries and modules, including optional spellchecker. 
import io
import json
import os
import random
from collections import Counter
from datetime import date
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from utils import predict_frame_with_debug, load_word_curriculum, get_random_word, save_word_attempt

try:
    from spellchecker import SpellChecker
    _spell = SpellChecker()
    SPELL_AVAILABLE = True
except ImportError:
    SPELL_AVAILABLE = False


#Page configuration for the Streamlit app, setting title, icon, and layout
st.set_page_config(
    page_title="SignaMind: Learns How You Learn Sign",
    page_icon="c:\\Users\\shubh\\Downloads\\sign.png",
    layout="wide",
)


#Root finding utility to ensure consistent file access regardless of execution context
def find_project_root(start_path: str) -> str:
    current = os.path.abspath(start_path)
    if os.path.isfile(current):
        current = os.path.dirname(current)
    for _ in range(10):
        if any(
            os.path.exists(os.path.join(current, name))
            for name in ["artifacts", "app", "src", "models"]
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.dirname(os.path.abspath(start_path))


def first_existing_path(paths: List[str]) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


BASE_DIR        = find_project_root(__file__)
ARTIFACTS_DIR   = os.path.join(BASE_DIR, "artifacts")
CURRICULUM_DIR  = os.path.join(ARTIFACTS_DIR, "curriculum")
EVAL_DIR        = os.path.join(ARTIFACTS_DIR, "evaluation")
APP_DIR         = os.path.join(BASE_DIR, "app")
LOGS_DIR        = os.path.join(APP_DIR, "logs")

CLASS_MAP_PATH = first_existing_path([
    os.path.join(ARTIFACTS_DIR, "class_mapping.json"),
    os.path.join(BASE_DIR,      "class_mapping.json"),
    os.path.join(ARTIFACTS_DIR, "metadata", "class_mapping.json"),
])

CURRICULUM_JSON_PATH = os.path.join(CURRICULUM_DIR, "curriculum.json")
EASY_PATH            = os.path.join(CURRICULUM_DIR, "easy_letters.csv")
HARD_PATH            = os.path.join(CURRICULUM_DIR, "hard_letters.csv")
CONFUSION_PAIRS_PATH = os.path.join(CURRICULUM_DIR, "confusion_pairs.csv")
WORD_CURRICULUM_PATH = os.path.join(CURRICULUM_DIR, "word_curriculum.csv")

MODEL_SUMMARY_PATH   = os.path.join(EVAL_DIR, "model_comparison_summary.csv")
PER_CLASS_PATH       = os.path.join(EVAL_DIR, "per_class_metrics_model3_mobilenetv2.csv")
TOP_CONFUSIONS_PATH  = os.path.join(EVAL_DIR, "top_confusions_model3_mobilenetv2.csv")

USER_PROGRESS_PATH   = os.path.join(LOGS_DIR, "user_progress.json")

REFERENCE_DIRS = [
    os.path.join(BASE_DIR, "app", "assets", "reference_signs"),
    os.path.join(BASE_DIR, "app", "assets", "letters"),
    os.path.join(BASE_DIR, "app", "assets", "numbers"),
]


#XP multipliers for different practice modes to encourage engagement with harder content.
MODE_MULTIPLIERS = {
    "Target Practice":         1.0,
    "Free Detection":          0.5,
    "Hard Gesture Practice":   1.5,   #harder signs worth more
    "Confusion Pair Practice":  1.3,
    "Word Practice":           1.2,
    "Typewriter Mode":         1.1,
}

XP_LEVELS = [
    (0,    "Beginner"),
    (100,  "Learner"),
    (300,  "Practitioner"),
    (600,  "Skilled"),
    (1000, "Advanced"),
    (1500, "Expert"),
    (2500, "Master"),
]

#Badge definitions  {id: (label, description, emoji)}
BADGE_DEFS = {
    "first_blood":       ("First Blood",       "Complete your first correct sign",              "🩸"),
    "accuracy_king":     ("Accuracy King",      "10 correct signs in a row",                    "👑"),
    "confidence_master": ("Confidence Master",  "20 attempts with avg confidence >= 90%",       "💎"),
    "hard_hitter":       ("Hard Hitter",        "10 correct Hard-Focus signs",                  "🔥"),
    "word_builder":      ("Word Builder",       "Complete your first full word",                "🔤"),
    "typewriter":        ("Typewriter",         "Save a word in Typewriter Mode",               "⌨️"),
    "comeback_kid":      ("Comeback Kid",       "Correct after 3+ consecutive failures",        "🔄"),
    "century":           ("Century",            "100 total attempts",                           "💯"),
}

#Consecutive-failure threshold for anti-frustration trigger
FRUSTRATION_THRESHOLD = 3


#Basic Helper Functions for file handling, data normalization, and label processing. These ensure robust data access and consistent formatting across the application.
def file_exists(path: str) -> bool:
    return os.path.exists(path)


def safe_read_json(path: str) -> dict:
    if not file_exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        for c in out.columns
    ]
    return out


def safe_read_csv(path: str) -> pd.DataFrame:
    if not file_exists(path):
        return pd.DataFrame()
    try:
        return normalize_columns(pd.read_csv(path))
    except Exception:
        return pd.DataFrame()


def read_single_column_csv(path: str) -> List[str]:
    if not file_exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if df.empty:
            return []
        return df.iloc[:, 0].astype(str).str.strip().tolist()
    except Exception:
        return []


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    return str(label).strip().lower()


def pretty_label(label: Optional[str]) -> str:
    if label is None:
        return "Free Detection"
    s = str(label).strip()
    if len(s) == 1 and s.isalpha():
        return s.upper()
    return s


#Spell-check suggestion function for Typewriter Mode, providing intelligent feedback on potential misspellings in user-constructed words.
#This enhances the learning experience by guiding users towards correct spelling and reinforcing letter recognition skills.
def spell_suggest(word: str) -> Optional[str]:
    """Return the top spell-check suggestion, or None if unavailable / already correct."""
    if not SPELL_AVAILABLE or not word:
        return None
    correction = _spell.correction(word.lower())
    if correction and correction.upper() != word.upper():
        return correction.upper()
    return None


#User progress management functions for loading, saving, and updating user performance data, including XP, streaks, badges, and anti-frustration tracking.
def load_user_progress() -> dict:
    defaults = {
        "total_xp":           0,
        "daily_streak":       0,
        "last_active_date":   "",
        "earned_badges":      [],
        "lifetime_correct":   0,
        "lifetime_attempts":  0,
    }
    saved = safe_read_json(USER_PROGRESS_PATH)
    return {**defaults, **saved}


def save_user_progress(progress: dict) -> None:
    safe_write_json(USER_PROGRESS_PATH, progress)


def update_daily_streak(progress: dict) -> dict:
    today_str     = str(date.today())
    last          = progress.get("last_active_date", "")

    if last == today_str:
        return progress  # already updated today

    yesterday_str = str(date.fromordinal(date.today().toordinal() - 1))

    if last == yesterday_str:
        progress["daily_streak"] = progress.get("daily_streak", 0) + 1
    else:
        progress["daily_streak"] = 1  #first session or streak broken

    progress["last_active_date"] = today_str
    return progress


def get_level(xp: int) -> Tuple[str, int, int]:
    """Return (level_name, xp_into_level, xp_needed_for_next)."""
    current_threshold, current_name = XP_LEVELS[0]
    next_xp = XP_LEVELS[1][0]

    for i, (threshold, name) in enumerate(XP_LEVELS):
        if xp >= threshold:
            current_threshold, current_name = threshold, name
            next_xp = XP_LEVELS[i + 1][0] if i + 1 < len(XP_LEVELS) else threshold + 1000

    xp_into  = xp - current_threshold
    xp_range = next_xp - current_threshold
    return current_name, xp_into, xp_range


def award_xp(confidence: float, mode: str, is_correct: bool) -> int:
    if not is_correct:
        return 0
    multiplier = MODE_MULTIPLIERS.get(mode, 1.0)
    return max(1, round(confidence * 100 * multiplier))


def check_and_award_badges(ss: dict, progress: dict, new_correct: bool) -> List[str]:
    earned     = set(progress.get("earned_badges", []))
    new_badges: List[str] = []

    def award(badge_id: str) -> None:
        if badge_id not in earned:
            earned.add(badge_id)
            new_badges.append(badge_id)

    if new_correct:
        award("first_blood")

    if ss.get("streak", 0) >= 10:
        award("accuracy_king")

    hist = ss.get("confidence_history", [])
    if ss.get("attempts", 0) >= 20 and len(hist) >= 20 and float(np.mean(hist[-20:])) >= 0.90:
        award("confidence_master")

    if ss.get("hard_correct", 0) >= 10:
        award("hard_hitter")

    if ss.get("words_completed", 0) >= 1:
        award("word_builder")

    if ss.get("typewriter_words_completed", 0) >= 1:
        award("typewriter")

    if new_correct and ss.get("consec_failures_before", 0) >= FRUSTRATION_THRESHOLD:
        award("comeback_kid")

    if ss.get("attempts", 0) >= 100:
        award("century")

    progress["earned_badges"] = list(earned)
    return new_badges


#Data loading functions for class mappings, curriculum data, and evaluation metrics, all cached for performance optimization in the Streamlit app.
@st.cache_data(show_spinner=False)
def load_class_mapping() -> Tuple[Dict[int, str], List[str]]:
    cm = safe_read_json(CLASS_MAP_PATH)
    if "id_to_label" not in cm:
        raise KeyError(f"class_mapping.json must have 'id_to_label'. Loaded: {CLASS_MAP_PATH}")
    idx_to_class = {int(k): str(v) for k, v in cm["id_to_label"].items()}
    class_names  = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    return idx_to_class, class_names


@st.cache_data(show_spinner=False)
def load_curriculum_data() -> dict:
    return {
        "curriculum":         safe_read_json(CURRICULUM_JSON_PATH),
        "easy_letters":       set(read_single_column_csv(EASY_PATH)),
        "hard_letters":       set(read_single_column_csv(HARD_PATH)),
        "confusion_pairs_df": safe_read_csv(CONFUSION_PAIRS_PATH),
    }


@st.cache_data(show_spinner=False)
def load_evaluation_data() -> dict:
    return {
        "model_summary":  safe_read_csv(MODEL_SUMMARY_PATH),
        "per_class":      safe_read_csv(PER_CLASS_PATH),
        "top_confusions": safe_read_csv(TOP_CONFUSIONS_PATH),
    }


#Image preprocessing function to convert uploaded files into the BGR format expected by OpenCV and the model.
def file_to_bgr(uploaded_file) -> np.ndarray:
    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def confidence_band(confidence: float, threshold: float) -> str:
    if confidence >= max(0.95, threshold):
        return "Excellent"
    if confidence >= max(0.85, threshold - 0.05):
        return "Good"
    if confidence >= max(0.70, threshold - 0.10):
        return "Moderate"
    return "Low"


def get_target_difficulty(target: Optional[str], easy_letters: set, hard_letters: set) -> str:
    norm = normalize_label(target)
    if not norm:
        return "Free Detection"
    if norm in {normalize_label(x) for x in hard_letters}:
        return "Hard Focus"
    if norm in {normalize_label(x) for x in easy_letters}:
        return "Easy Start"
    return "Standard"


def build_confusion_lookup(df: pd.DataFrame) -> Dict[str, List[str]]:
    if df.empty:
        return {}
    tc = first_existing_column(df, ["true", "true_label", "actual", "label", "source"])
    pc = first_existing_column(df, ["pred", "pred_label", "predicted", "target", "confused_with"])
    if not tc or not pc:
        return {}
    lookup: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        tl = normalize_label(row[tc])
        pl = normalize_label(row[pc])
        if tl and pl:
            lookup.setdefault(tl, [])
            if pl not in lookup[tl]:
                lookup[tl].append(pl)
    return lookup


def get_target_row(per_class_df: pd.DataFrame, target: str) -> pd.DataFrame:
    if per_class_df.empty:
        return pd.DataFrame()
    lc = first_existing_column(per_class_df, ["label", "class", "class_label", "character", "target"])
    if not lc:
        return pd.DataFrame()
    mask = per_class_df[lc].astype(str).map(normalize_label) == normalize_label(target)
    return per_class_df[mask]


def get_reference_image_path(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    s    = str(label).strip()
    for ref_dir in REFERENCE_DIRS:
        for ext in exts:
            for variant in [s, s.upper(), s.lower()]:
                p = os.path.join(ref_dir, f"{variant}{ext}")
                if os.path.exists(p):
                    return p
    return None


def build_reasoning(
    pred_label: str,
    confidence: float,
    threshold: float,
    target_label: Optional[str],
    target_difficulty: str,
    confusion_lookup: Dict[str, List[str]],
    mode_label: str,
    quality: Optional[dict] = None,
) -> List[str]:
    reasons: List[str] = []
    pn = normalize_label(pred_label)
    tn = normalize_label(target_label)

    if confidence < threshold:
        reasons.append(
            f"Model confidence is below your threshold ({confidence:.3f} < {threshold:.2f}) -- "
            "treat this result with caution."
        )
    else:
        reasons.append(
            f"Model confidence is above your threshold ({confidence:.3f} >= {threshold:.2f}) -- "
            "this result is reliable enough for practice feedback."
        )

    reasons.append(f"Result came from **{mode_label}** mode.")

    if quality:
        if quality.get("is_blurry"):
            reasons.append(
                f"Image looks blurry (blur score: {quality.get('blur_score', 0):.1f}) -- "
                "this can hurt prediction quality."
            )
        if quality.get("is_dark"):
            reasons.append(
                f"Image looks dark (brightness: {quality.get('brightness', 0):.1f}) -- "
                "better lighting will improve results."
            )
        if quality.get("is_low_contrast"):
            reasons.append(
                f"Low contrast (contrast: {quality.get('contrast', 0):.1f}) -- "
                "finger boundaries may be harder to detect."
            )

    if target_label:
        if pn == tn:
            reasons.append(f"Predicted sign matches target '{pretty_label(target_label)}'.")
        else:
            reasons.append(
                f"No match. Expected '{pretty_label(target_label)}', "
                f"model predicted '{pretty_label(pred_label)}'."
            )

    if target_label and target_difficulty == "Hard Focus":
        reasons.append(
            f"'{pretty_label(target_label)}' is in the hard group -- "
            "try slowing down your hand positioning and hold the sign steady."
        )
    elif target_label and target_difficulty == "Easy Start":
        reasons.append(
            f"'{pretty_label(target_label)}' is a beginner-friendly sign -- "
            "great for building early confidence."
        )

    if tn and tn in confusion_lookup:
        known       = confusion_lookup[tn][:3]
        pretty_known = [pretty_label(x) for x in known]
        if pn != tn and pn in confusion_lookup.get(tn, []):
            reasons.append(
                f"This is a known confusion pattern -- '{pretty_label(target_label)}' "
                f"is often mistaken for '{pretty_label(pred_label)}'."
            )
        else:
            reasons.append(
                f"Signs commonly confused with '{pretty_label(target_label)}': "
                f"{', '.join(pretty_known)}."
            )

    return reasons


#Session state initialization and management functions to track user performance, attempt history, anti-frustration metrics, and practice queue status across interactions within the Streamlit app.
def init_session_state() -> None:
    defaults = {
        #general tracking
        "attempts": 0,
        "correct_attempts": 0,
        "streak": 0,
        "best_streak": 0,
        "confidence_history": [],
        "prediction_history": [],
        "target_history": [],
        "mode_history": [],
        "result_rows": [],
        "last_result": None,
        "session_xp": 0,
        "new_badges_this_session": [],
        "hard_correct": 0,
        "words_completed": 0,
        "typewriter_words_completed": 0,
        
        #anti-frustration: {normalize_label(target): consecutive_fail_count}
        "consec_fail_map": {},
        "consec_failures_before": 0,
        
        #letter-queue practice
        "current_target": None,
        "practice_queue": [],
        "queue_index": 0,
        
        #word practice
        "word_target": "",
        "word_difficulty": "all",
        "word_category": "",
        "word_predictions": [],
        "word_confidences": [],
        "word_index": 0,
        "word_complete": False,
        "word_history": [],
        
        #typewriter mode
        "tw_buffer": [],
        "tw_pending": None,
        "tw_pending_confidence": 0.0,
        "tw_suggest": None,
        "tw_completed_words": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session_state() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_session_state()


def clear_current_attempt() -> None:
    st.session_state["last_result"] = None


def labels_match(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return normalize_label(a) == normalize_label(b)


#Anti frustration tracking functions to monitor consecutive failures on the same target sign and trigger adaptive feedback when users struggle with specific signs, helping to maintain motivation and guide improvement.
def update_frustration_tracker(target_label: Optional[str], is_correct: bool) -> None:
    if not target_label:
        st.session_state["consec_failures_before"] = 0
        return
    key = normalize_label(target_label)
    fm  = st.session_state.get("consec_fail_map", {})
    # snapshot BEFORE updating, so comeback_kid logic works
    st.session_state["consec_failures_before"] = fm.get(key, 0)
    fm[key] = 0 if is_correct else fm.get(key, 0) + 1
    st.session_state["consec_fail_map"] = fm


def get_frustration_count(target_label: Optional[str]) -> int:
    if not target_label:
        return 0
    return st.session_state.get("consec_fail_map", {}).get(normalize_label(target_label), 0)


def render_pro_tip(target_label: str, confusion_lookup: Dict[str, List[str]]) -> None:
    """Auto-shown panel after FRUSTRATION_THRESHOLD consecutive failures on the same sign."""
    tn         = normalize_label(target_label)
    fail_count = get_frustration_count(target_label)

    st.markdown(
        f"""
        <div style="border:2px solid #e67e22;border-radius:10px;
                    padding:14px 18px;background:#fef9f0;margin-top:10px;">
        <h4 style="color:#e67e22;margin:0 0 8px 0">
            🧠 Pro-Tip Activated -- You've missed
            <b>{pretty_label(target_label)}</b> {fail_count}x in a row!
        </h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tip_col1, tip_col2 = st.columns([1, 1])

    with tip_col1:
        known = confusion_lookup.get(tn, [])
        if known:
            st.markdown("**What the model is probably seeing instead:**")
            for c in known[:4]:
                st.write(f"  - It may be reading your sign as **{pretty_label(c)}**")
        else:
            st.markdown("**Focus on finger clarity** -- no specific confusion data recorded.")

        st.markdown("**Checklist before your next attempt:**")
        for tip in [
            "Hold your hand completely still before clicking Analyze",
            "Keep your hand centered and fully in the frame",
            "Use a plain wall background -- avoid patterned clothing",
            "Make sure ALL fingers are clearly separated and visible",
            "Move your hand 5-10 cm closer to the camera",
            "Check your lighting -- avoid harsh shadows on your hand",
        ]:
            st.write(f"  ✓ {tip}")

    with tip_col2:
        ref_path = get_reference_image_path(target_label)
        if ref_path:
            st.image(
                ref_path,
                caption=f"Reference: '{pretty_label(target_label)}' -- align your hand to match this exactly",
                use_container_width=True,
            )
            st.markdown(
                """
                <div style='background:#eaf4fb;border-radius:8px;padding:10px;
                            font-size:13px;color:#2c3e50;margin-top:6px;'>
                <b>Alignment Guide:</b> Compare your hand shape to the reference above.
                Focus on which fingers are extended vs. curled, the angle of your wrist,
                and the spacing between fingers. Even a small difference changes the prediction.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "No reference image found for this sign. "
                "Try searching 'ASL letter " + pretty_label(target_label) + "' online for a visual guide."
            )


#Record an attempt in the session state and update user progress, including XP, badges, and anti-frustration tracking. 
#This function centralizes all the logic for handling the outcome of a prediction attempt and its implications for the user's learning journey.
def record_attempt(
    target_label: Optional[str],
    pred_label: str,
    confidence: float,
    mode_name: str,
    progress: dict,
) -> dict:
    is_match = labels_match(target_label, pred_label)
    update_frustration_tracker(target_label, is_match)

    st.session_state["attempts"] += 1

    if is_match:
        st.session_state["correct_attempts"] += 1
        st.session_state["streak"]           += 1
        if st.session_state["streak"] > st.session_state["best_streak"]:
            st.session_state["best_streak"] = st.session_state["streak"]
        # count hard-mode corrects separately for the Hard Hitter badge
        if get_target_difficulty(target_label, set(), set()) == "Hard Focus":
            st.session_state["hard_correct"] = st.session_state.get("hard_correct", 0) + 1
    else:
        st.session_state["streak"] = 0

    stored_target = pretty_label(target_label) if target_label else "Free Detection"
    st.session_state["confidence_history"].append(round(confidence, 4))
    st.session_state["prediction_history"].append(pretty_label(pred_label))
    st.session_state["target_history"].append(stored_target)
    st.session_state["mode_history"].append(mode_name)
    st.session_state["result_rows"].append({
        "attempt_no": st.session_state["attempts"],
        "mode":       mode_name,
        "target":     stored_target,
        "prediction": pretty_label(pred_label),
        "confidence": round(confidence, 4),
        "match":      "Yes" if is_match else "No",
    })

    #XP
    xp_gained = award_xp(confidence, mode_name, is_match)
    st.session_state["session_xp"]  = st.session_state.get("session_xp", 0) + xp_gained
    progress["total_xp"]            = progress.get("total_xp", 0) + xp_gained
    progress["lifetime_attempts"]   = progress.get("lifetime_attempts", 0) + 1
    if is_match:
        progress["lifetime_correct"] = progress.get("lifetime_correct", 0) + 1

    #Badges
    new_b = check_and_award_badges(st.session_state, progress, is_match)
    if new_b:
        existing = st.session_state.get("new_badges_this_session", [])
        st.session_state["new_badges_this_session"] = existing + new_b

    return progress


def session_average_confidence() -> float:
    vals = st.session_state["confidence_history"]
    return float(np.mean(vals)) if vals else 0.0


def session_success_rate() -> float:
    a = st.session_state["attempts"]
    return st.session_state["correct_attempts"] / a if a else 0.0


def most_common_wrong_prediction(target: Optional[str]) -> str:
    if not target:
        return "N/A"
    tp    = pretty_label(target)
    rows  = st.session_state["result_rows"]
    wrong = [r["prediction"] for r in rows if r["target"] == tp and r["match"] == "No"]
    return Counter(wrong).most_common(1)[0][0] if wrong else "None"


#Practice queue management functions to initialize a randomized queue of target signs for focused practice sessions, track the current position in the queue, 
#and allow reshuffling to keep practice engaging and varied.
def init_practice_queue(targets: List[str]) -> None:
    shuffled = targets.copy()
    random.shuffle(shuffled)
    st.session_state["practice_queue"] = shuffled
    st.session_state["queue_index"]    = 0


def get_current_queue_target() -> Optional[str]:
    q   = st.session_state.get("practice_queue", [])
    idx = st.session_state.get("queue_index", 0)
    return q[idx] if q and idx < len(q) else None


def advance_queue() -> None:
    st.session_state["queue_index"] = st.session_state.get("queue_index", 0) + 1


def reshuffle_queue(targets: List[str]) -> None:
    init_practice_queue(targets)


#Word Practice management functions to handle the state of word construction exercises, including initializing a new word, updating progress with each letter attempt, 
#and rendering a visual progress bar to guide the user through the sequential learning process of building words from individual signs.
def start_new_word(word_df: pd.DataFrame, difficulty: str = "all") -> None:
    info = get_random_word(word_df, difficulty)
    base = {
        "word_difficulty": difficulty,
        "word_predictions": [],
        "word_confidences": [],
        "word_index": 0,
        "word_complete": False,
    }
    if not info:
        st.session_state.update({**base, "word_target": "", "word_category": ""})
        return
    st.session_state.update({
        **base,
        "word_target":   info["word"],
        "word_category": info["category"],
    })


def update_word_progress(pred_label: str, confidence: float) -> None:
    if not st.session_state.get("word_target") or st.session_state.get("word_complete"):
        return
    st.session_state["word_predictions"].append(pretty_label(pred_label))
    st.session_state["word_confidences"].append(float(confidence))
    st.session_state["word_index"] += 1

    wt = st.session_state["word_target"]
    if st.session_state["word_index"] >= len(wt):
        st.session_state["word_complete"]  = True
        st.session_state["words_completed"] = st.session_state.get("words_completed", 0) + 1
        final = "".join(st.session_state["word_predictions"])
        save_word_attempt(
            target_word=wt,
            predicted_word=final,
            difficulty=st.session_state["word_difficulty"],
            category=st.session_state["word_category"],
            confidences=st.session_state["word_confidences"],
        )
        st.session_state["word_history"].append({
            "target":          wt,
            "predicted":       final,
            "correct":         int(wt == final),
            "letter_accuracy": round(
                sum(1 for t, p in zip(wt, final) if t == p) / max(len(wt), 1), 3
            ),
        })


def render_word_progress_bar(target_word: str, predictions: List[str], word_index: int) -> None:
    if not target_word:
        return
    cells = []
    for i, letter in enumerate(target_word):
        if i < len(predictions):
            pred    = predictions[i]
            correct = letter.upper() == pred.upper()
            colour  = "#2ecc71" if correct else "#e74c3c"
            label   = f"{letter}<br><small style='color:#fff'>{pred}</small>"
        elif i == word_index:
            colour = "#3498db"
            label  = f"<b>{letter}</b><br><small>now</small>"
        else:
            colour = "#7f8c8d"
            label  = f"{letter}<br>&nbsp;"
        cells.append(
            f"<div style='display:inline-block;text-align:center;width:44px;"
            f"background:{colour};color:#fff;border-radius:6px;"
            f"margin:3px;padding:6px 0;font-weight:bold;font-size:18px'>{label}</div>"
        )
    st.markdown("".join(cells), unsafe_allow_html=True)


#Typewriter Mode management functions to handle the letter-by-letter construction of words, including buffering confirmed letters, managing pending predictions, 
#and providing real-time feedback on potential misspellings with spell-check suggestions.
def tw_confirm_letter() -> None:
    pending = st.session_state.get("tw_pending")
    if pending:
        st.session_state["tw_buffer"].append(pending)
        st.session_state["tw_pending"]            = None
        st.session_state["tw_pending_confidence"] = 0.0
        word = "".join(st.session_state["tw_buffer"])
        st.session_state["tw_suggest"] = spell_suggest(word)


def tw_backspace() -> None:
    buf = st.session_state.get("tw_buffer", [])
    if buf:
        buf.pop()
        st.session_state["tw_buffer"] = buf
    st.session_state["tw_pending"]            = None
    st.session_state["tw_pending_confidence"] = 0.0
    word = "".join(st.session_state["tw_buffer"])
    st.session_state["tw_suggest"] = spell_suggest(word) if word else None


def tw_clear_buffer() -> None:
    st.session_state["tw_buffer"]             = []
    st.session_state["tw_pending"]            = None
    st.session_state["tw_pending_confidence"] = 0.0
    st.session_state["tw_suggest"]            = None


def render_typewriter_buffer() -> None:
    buf     = st.session_state.get("tw_buffer", [])
    pending = st.session_state.get("tw_pending")
    word    = "".join(buf)

    cells = []
    for ch in buf:
        cells.append(
            f"<div style='display:inline-block;text-align:center;width:42px;"
            f"background:#2ecc71;color:#fff;border-radius:6px;"
            f"margin:2px;padding:6px 0;font-weight:bold;font-size:22px'>{ch}</div>"
        )
    if pending:
        conf = st.session_state.get("tw_pending_confidence", 0.0)
        cells.append(
            f"<div style='display:inline-block;text-align:center;width:42px;"
            f"background:#e67e22;color:#fff;border-radius:6px;"
            f"margin:2px;padding:6px 0;font-weight:bold;font-size:22px;"
            f"border:2px dashed #c0392b'>{pending}"
            f"<br><small style='font-size:10px'>{conf:.0%}</small></div>"
        )

    if cells:
        st.markdown("".join(cells), unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size:24px;letter-spacing:5px;margin:10px 0'>"
            f"<b>{word}{pending or ''}</b></p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<p style='color:#aaa;font-style:italic;font-size:16px'>"
            "Buffer is empty -- analyze a sign, then confirm it to start typing.</p>",
            unsafe_allow_html=True,
        )

    suggest = st.session_state.get("tw_suggest")
    if suggest:
        st.info(f"🔤 Did you mean: **{suggest}**?")


#Target selection logic to determine which signs should be included in the practice queue based on the selected mode, user progress, and curriculum data.
def build_target_options(
    mode: str,
    class_names: List[str],
    hard_letters: set,
    confusion_df: pd.DataFrame,
) -> Tuple[List[str], Optional[str], List[str]]:
    if mode in ("Free Detection", "Typewriter Mode"):
        return class_names, None, []

    if mode == "Hard Gesture Practice":
        hard = [c for c in class_names if normalize_label(c) in {normalize_label(h) for h in hard_letters}]
        return (hard if hard else class_names), None, []

    if mode == "Confusion Pair Practice":
        if confusion_df.empty:
            return class_names, None, []
        tc = first_existing_column(confusion_df, ["true", "true_label", "actual", "label", "source"])
        pc = first_existing_column(confusion_df, ["pred", "pred_label", "predicted", "target", "confused_with"])
        if not tc or not pc:
            return class_names, None, []

        pairs, seen = [], set()
        for _, row in confusion_df.iterrows():
            a, b = str(row[tc]).strip(), str(row[pc]).strip()
            key  = tuple(sorted([normalize_label(a), normalize_label(b)]))
            if key not in seen:
                seen.add(key)
                pairs.append((a, b))

        if not pairs:
            return class_names, None, []

        labels  = [f"{pretty_label(a)} <> {pretty_label(b)}" for a, b in pairs]
        chosen  = st.sidebar.selectbox("Choose confusion pair", labels, key="conf_pair")
        l, r    = chosen.split(" <> ")
        return [l, r], chosen, [l, r]

    return class_names, None, []


#Shared Inference function to run the model prediction on the input image, extract relevant information for feedback, 
#and store the results in the session state for use across the app.
def run_inference_and_store_result(
    bgr_image: np.ndarray,
    analyzed_target: Optional[str],
    practice_mode: str,
    confidence_threshold: float,
    idx_to_class: Dict[int, str],
    easy_letters: set,
    hard_letters: set,
    confusion_lookup: Dict[str, List[str]],
    target_options: List[str],
    progress: dict,
) -> dict:
    debug      = predict_frame_with_debug(bgr_image)
    pred_label  = str(debug["pred_label"]).strip()
    confidence  = float(debug["confidence"])
    probs       = debug["probs"]
    quality     = debug["quality"]
    meta        = debug["meta"]

    top_indices = np.argsort(probs)[::-1][:5]
    top5 = [
        {"label": pretty_label(idx_to_class[int(i)]), "confidence": round(float(probs[int(i)]), 4)}
        for i in top_indices
    ]

    progress = record_attempt(analyzed_target, pred_label, confidence, practice_mode, progress)

    if practice_mode == "Word Practice":
        update_word_progress(pred_label, confidence)

    difficulty = get_target_difficulty(analyzed_target, easy_letters, hard_letters)
    reasons    = build_reasoning(
        pred_label=pred_label,
        confidence=confidence,
        threshold=confidence_threshold,
        target_label=analyzed_target,
        target_difficulty=difficulty,
        confusion_lookup=confusion_lookup,
        mode_label=practice_mode,
        quality=quality,
    )

    is_match        = labels_match(analyzed_target, pred_label)
    meets_threshold = confidence >= confidence_threshold

    st.session_state["last_result"] = {
        "pred_label":       pretty_label(pred_label),
        "confidence":       confidence,
        "is_match":         is_match,
        "meets_threshold":  meets_threshold,
        "top5":             top5,
        "reasons":          reasons,
        "target":           pretty_label(analyzed_target) if analyzed_target else None,
        "raw_target":       analyzed_target,
        "band":             confidence_band(confidence, confidence_threshold),
        "mode":             practice_mode,
        "quality":          quality,
        "meta":             meta,
        "preprocessed_rgb": debug["preprocessed_rgb"],
    }

    if (
        analyzed_target is not None
        and is_match
        and confidence >= confidence_threshold
        and practice_mode in ["Target Practice", "Hard Gesture Practice"]
    ):
        advance_queue()
        if get_current_queue_target() is None and target_options:
            init_practice_queue(target_options)

    return progress


#Gamification rendering functions to display the user's current level, XP progress, 
#and earned badges in an engaging way within the Streamlit app, providing motivation and a sense of achievement as they practice.
def render_xp_bar(total_xp: int) -> None:
    level_name, xp_into, xp_range = get_level(total_xp)
    pct = min(100, round(xp_into / max(xp_range, 1) * 100))
    st.markdown(
        f"""
        <div style='margin:4px 0 10px 0'>
          <span style='font-weight:bold;font-size:15px'>{level_name}</span>
          <span style='color:#888;font-size:12px;margin-left:8px'>
            {xp_into} / {xp_range} XP to next level
          </span>
          <div style='background:#eee;border-radius:8px;height:12px;margin-top:5px'>
            <div style='background:linear-gradient(90deg,#3498db,#2ecc71);
                        border-radius:8px;height:12px;width:{pct}%'></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_badges(earned_ids: List[str]) -> None:
    if not earned_ids:
        st.caption("No badges yet -- keep practicing!")
        return
    cols = st.columns(min(len(earned_ids), 4))
    for i, bid in enumerate(earned_ids):
        defn = BADGE_DEFS.get(bid)
        if defn:
            label, desc, emoji = defn
            with cols[i % 4]:
                st.markdown(
                    f"""
                    <div style='text-align:center;border:1px solid #ddd;
                                border-radius:10px;padding:12px 8px;
                                background:#f9f9f9;margin-bottom:8px'>
                      <div style='font-size:30px'>{emoji}</div>
                      <div style='font-weight:bold;font-size:13px;margin-top:4px'>{label}</div>
                      <div style='color:#888;font-size:11px;margin-top:2px'>{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


#Main app initialization: load all necessary data, set up session state, 
#and prepare the UI based on the selected practice mode and user progress.
init_session_state()

user_progress = load_user_progress()
user_progress = update_daily_streak(user_progress)
save_user_progress(user_progress)

try:
    idx_to_class, class_names = load_class_mapping()
except Exception as e:
    st.error(f"Failed to load class mapping: {e}")
    st.stop()

curriculum_data    = load_curriculum_data()
evaluation_data    = load_evaluation_data()
word_df            = load_word_curriculum(WORD_CURRICULUM_PATH)

easy_letters        = curriculum_data["easy_letters"]
hard_letters        = curriculum_data["hard_letters"]
confusion_pairs_df  = curriculum_data["confusion_pairs_df"]
curriculum_json     = curriculum_data["curriculum"]

model_summary_df   = evaluation_data["model_summary"]
per_class_df       = evaluation_data["per_class"]
top_confusions_df  = evaluation_data["top_confusions"]

confusion_lookup   = build_confusion_lookup(confusion_pairs_df)


#Sidebar controls for practice mode selection, confidence threshold adjustment, display options, 
#and session management, providing users with the tools to customize their practice experience and track their progress effectively.
st.sidebar.title("Practice Controls")

#XP / level in sidebar
render_xp_bar(user_progress.get("total_xp", 0))
streak_days = user_progress.get("daily_streak", 0)
session_xp  = st.session_state.get("session_xp", 0)
st.sidebar.caption(
    f"Daily Streak: **{streak_days} day(s)**  |  Session XP: **+{session_xp}**"
)
st.sidebar.markdown("---")

practice_mode = st.sidebar.selectbox(
    "Practice Mode",
    [
        "Target Practice",
        "Free Detection",
        "Hard Gesture Practice",
        "Confusion Pair Practice",
        "Word Practice",
        "Typewriter Mode",
    ],
)

#Word practice controls
selected_word_difficulty = "all"
if practice_mode == "Word Practice":
    selected_word_difficulty = st.sidebar.selectbox(
        "Word Difficulty", ["all", "easy", "medium", "hard"], key="word_diff_sel"
    )
    wc1, wc2 = st.sidebar.columns(2)
    with wc1:
        if st.button("Start Word", use_container_width=True):
            start_new_word(word_df, selected_word_difficulty)
            clear_current_attempt()
            st.rerun()
    with wc2:
        if st.button("Skip Word", use_container_width=True):
            start_new_word(word_df, selected_word_difficulty)
            clear_current_attempt()
            st.rerun()

#Typewriter controls
if practice_mode == "Typewriter Mode":
    st.sidebar.markdown("**Typewriter Controls**")
    if st.sidebar.button("Confirm Letter", use_container_width=True):
        tw_confirm_letter()
        st.rerun()
    if st.sidebar.button("Backspace", use_container_width=True):
        tw_backspace()
        st.rerun()
    if st.sidebar.button("Clear Buffer", use_container_width=True):
        tw_clear_buffer()
        st.rerun()

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.50,
    max_value=0.99,
    value=0.85,
    step=0.01,
    help="Higher values require the model to be more certain before accepting a result.",
)

st.sidebar.markdown("**Display Options**")
show_top5                = st.sidebar.checkbox("Show top-5 predictions",    value=True)
show_reference_tables    = st.sidebar.checkbox("Show BI reference tables",  value=True)
show_debug_preprocessing = st.sidebar.checkbox("Show preprocessing debug",  value=False)

st.sidebar.markdown("---")
if st.sidebar.button("Reset Session", use_container_width=True):
    reset_session_state()
    st.rerun()

with st.sidebar.expander("How to Use"):
    st.markdown(
        """
        1. Choose a **Practice Mode**.
        2. Select **Take Picture** or **Upload Image**.
        3. Show your hand sign clearly, then click **Analyze Sign**.
        4. Review the **Latest Result** for XP, feedback, and pro-tips.
        5. Check **Analytics & Badges** to track your full progress.

        **Modes:**
        - **Target Practice** -- all 36 signs in shuffled order.
        - **Free Detection** -- detect any sign, no target.
        - **Hard Gesture Practice** -- drill the confusing signs.
        - **Confusion Pair Practice** -- two similar signs side-by-side.
        - **Word Practice** -- spell words letter-by-letter.
        - **Typewriter Mode** -- build a custom word with spell-check.
        """
    )

st.sidebar.caption("Model: MobileNetV2 | CST2213 BI | Shubhangi Singh")


#Based on the selected practice mode and the curriculum data, determine which signs should be included in the practice queue, 
#and set the initial target sign for the session. This logic ensures that users are practicing relevant signs based on their chosen focus area and progress, 
#and it supports the adaptive learning experience of the app
target_options, selected_pair_label, pair_targets = build_target_options(
    practice_mode, class_names, hard_letters, confusion_pairs_df,
)

selected_target: Optional[str] = None

if practice_mode in ("Free Detection", "Typewriter Mode"):
    selected_target = None

elif practice_mode == "Word Practice":
    wt  = st.session_state.get("word_target", "")
    idx = st.session_state.get("word_index", 0)
    selected_target = wt[idx] if wt and idx < len(wt) else None

elif practice_mode == "Confusion Pair Practice":
    if pair_targets:
        if st.session_state.get("current_target") not in pair_targets:
            st.session_state["current_target"] = pair_targets[0]
        selected_target = st.session_state["current_target"]
        if st.sidebar.button("Switch Pair Target", use_container_width=True):
            other = [x for x in pair_targets if x != st.session_state["current_target"]]
            if other:
                st.session_state["current_target"] = other[0]
                clear_current_attempt()
                st.rerun()

else:  #Target Practice / Hard Gesture Practice
    if target_options:
        ex_set = {normalize_label(x) for x in st.session_state.get("practice_queue", [])}
        nw_set = {normalize_label(x) for x in target_options}
        if not st.session_state["practice_queue"] or ex_set != nw_set:
            init_practice_queue(target_options)
        selected_target = get_current_queue_target()

        ca, cb = st.sidebar.columns(2)
        with ca:
            if st.button("Next Sign", use_container_width=True):
                advance_queue()
                if get_current_queue_target() is None:
                    init_practice_queue(target_options)
                clear_current_attempt()
                st.rerun()
        with cb:
            if st.button("Reshuffle", use_container_width=True):
                reshuffle_queue(target_options)
                clear_current_attempt()
                st.rerun()


#Header and key metrics display: show the current practice mode, target sign, difficulty level, session confidence average, success rate, and streak information at the top of the dashboard for quick reference and motivation.
st.title("SignaMind: Learns how you sign")
st.caption(
    "ASL fingerspelling recognition with BI analytics, XP gamification, "
    "anti-frustration logic, and sign-to-text.")

#Badge unlock notifications
new_badges = st.session_state.get("new_badges_this_session", [])
if new_badges:
    for bid in new_badges:
        defn = BADGE_DEFS.get(bid)
        if defn:
            label, desc, emoji = defn
            st.success(f"{emoji} Badge Unlocked: **{label}** -- {desc}")
    st.session_state["new_badges_this_session"] = []

#Mode banners
if practice_mode == "Word Practice":
    wt    = st.session_state.get("word_target", "")
    built = "".join(st.session_state.get("word_predictions", []))
    pos   = st.session_state.get("word_index", 0)
    total = len(wt) if wt else 0
    done  = st.session_state.get("word_complete", False)
    if wt:
        st.info(
            f"**Word Practice** | Target: `{wt}` | Built: `{built or '...'}` | "
            f"Letter {min(pos+1, total)}/{total}" + (" -- Complete!" if done else "")
        )
    else:
        st.info("Word Practice -- click **Start Word** in the sidebar to begin.")

if practice_mode == "Typewriter Mode":
    buf  = st.session_state.get("tw_buffer", [])
    pend = st.session_state.get("tw_pending")
    word = "".join(buf) + (pend or "")
    st.info(
        f"**Typewriter Mode** | Buffer: `{word or '(empty)'}` -- "
        "Analyze a sign, then use **Confirm Letter** to type it."
    )

#Key metrics row
target_display = pretty_label(selected_target)
difficulty     = get_target_difficulty(selected_target, easy_letters, hard_letters)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Practicing",     target_display)
    st.caption("Current target sign.")
with m2:
    st.metric("Difficulty",     difficulty)
    st.caption("Curriculum level.")
with m3:
    st.metric("Avg Confidence", f"{session_average_confidence():.3f}")
    st.caption("This session.")
with m4:
    st.metric("Success Rate",   f"{session_success_rate() * 100:.1f}%")
    st.caption("Correct / total.")
with m5:
    streak      = st.session_state.get("streak", 0)
    best_streak = st.session_state.get("best_streak", 0)
    st.metric("Streak",         f"{streak}  (best: {best_streak})")
    st.caption("Consecutive correct.")

st.markdown("---")


#Main tabbed interface for different practice modes and analytics views, allowing users to easily navigate between practicing their signs, reviewing analytics 
#and badges, comparing model performance, and analyzing failure cases in a structured and user-friendly way.
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Practice",
    "Typewriter",
    "Analytics & Badges",
    "Model Comparison",
    "Failure Analysis",])


#Tab 1: Camera & Upload Practice: allows users to take a picture or upload an image of their hand sign, which is then processed through the model's prediction pipeline.
#The interface provides tips for getting the best results and displays the current target sign and relevant information for focused practice.
with tab1:
    st.subheader("Camera & Upload Practice")
    st.caption(
        "Use your camera or an uploaded image. "
        "Both go through the same CLAHE -> foreground-crop -> TTA prediction pipeline."
    )

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        with st.expander("Best Input Tips", expanded=False):
            st.markdown(
                """
                - One clearly visible hand, centered in the frame
                - Plain background (wall is better than clothing)
                - All fingers clearly separated and fully visible
                - Hold the sign completely still before clicking Analyze
                - Even lighting -- avoid harsh shadows on your hand
                - Hand roughly 20-40 cm from the camera
                """
            )

        input_mode = st.radio(
            "Input Method",
            ["Take Picture", "Upload Image"],
            horizontal=True,
        )

        chosen_bgr = None

        if input_mode == "Take Picture":
            cam_file = st.camera_input("Take a picture of your hand sign")
            if cam_file is not None:
                chosen_bgr = file_to_bgr(cam_file)
                st.image(
                    cv2.cvtColor(chosen_bgr, cv2.COLOR_BGR2RGB),
                    caption="Captured image",
                    use_container_width=True,
                )
        else:
            up_file = st.file_uploader(
                "Upload a sign image",
                type=["png", "jpg", "jpeg", "webp"],
                key="upload_practice_file",
            )
            if up_file is not None:
                chosen_bgr = file_to_bgr(up_file)
                st.image(
                    cv2.cvtColor(chosen_bgr, cv2.COLOR_BGR2RGB),
                    caption="Uploaded image",
                    use_container_width=True,
                )

        analyze_clicked = st.button(
            "Analyze Sign", type="primary", use_container_width=True
        )

        if analyze_clicked:
            if chosen_bgr is None:
                st.warning("Please take a picture or upload an image first.")
            else:
                with st.spinner("Running prediction..."):
                    try:
                        user_progress = run_inference_and_store_result(
                            bgr_image=chosen_bgr,
                            analyzed_target=selected_target,
                            practice_mode=practice_mode,
                            confidence_threshold=confidence_threshold,
                            idx_to_class=idx_to_class,
                            easy_letters=easy_letters,
                            hard_letters=hard_letters,
                            confusion_lookup=confusion_lookup,
                            target_options=target_options,
                            progress=user_progress,
                        )
                        save_user_progress(user_progress)

                        # Set typewriter pending letter
                        if practice_mode == "Typewriter Mode":
                            lr = st.session_state.get("last_result", {})
                            if lr:
                                st.session_state["tw_pending"]            = lr["pred_label"]
                                st.session_state["tw_pending_confidence"] = lr["confidence"]

                        q = st.session_state["last_result"]["quality"]
                        if q["is_blurry"]:
                            st.warning(
                                f"Blurry image (score: {q['blur_score']:.1f}) -- "
                                "prediction may be unreliable."
                            )
                        if q["is_dark"]:
                            st.warning(
                                f"Dark image (brightness: {q['brightness']:.1f}) -- "
                                "try better lighting."
                            )
                        if q["is_low_contrast"]:
                            st.warning(
                                f"Low contrast (contrast: {q['contrast']:.1f}) -- "
                                "finger boundaries may be unclear."
                            )
                        st.rerun()

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    with right_col:
        st.subheader("Target Guide")

        # Word progress strip
        if practice_mode == "Word Practice" and st.session_state.get("word_target"):
            st.markdown("**Word Progress**")
            render_word_progress_bar(
                st.session_state["word_target"],
                st.session_state.get("word_predictions", []),
                st.session_state.get("word_index", 0),
            )
            st.markdown("")

        if selected_target:
            known      = confusion_lookup.get(normalize_label(selected_target), [])
            known_text = ", ".join(pretty_label(x) for x in known[:5]) if known else "None recorded"
            st.markdown(
                f"""
                - **Current target:** `{pretty_label(selected_target)}`
                - **Practice level:** `{difficulty}`
                - **Often confused with:** `{known_text}`
                - **Confidence threshold:** `{confidence_threshold:.2f}`
                """
            )
            ref_path = get_reference_image_path(selected_target)
            if ref_path:
                st.image(
                    ref_path,
                    caption=f"Reference: '{pretty_label(selected_target)}'",
                    width=220,
                )
            else:
                st.info(f"No reference image found for '{pretty_label(selected_target)}'.")
        else:
            mode_notes = {
                "Free Detection": "Predicts whichever sign the model sees -- no target.",
                "Typewriter Mode": "Analyze any sign, then use Confirm Letter to append it to the buffer.",
            }
            st.markdown(
                f"""
                - **Mode:** `{practice_mode}`
                - {mode_notes.get(practice_mode, 'No target selected.')}
                - **Confidence threshold:** `{confidence_threshold:.2f}`
                """
            )

    #ANTI-FRUSTRATION BLOCK ── shown automatically after FRUSTRATION_THRESHOLD fails
    if selected_target:
        fail_count = get_frustration_count(selected_target)
        if fail_count >= FRUSTRATION_THRESHOLD:
            render_pro_tip(selected_target, confusion_lookup)


#Shared result display block: after each prediction, show the predicted label, confidence score, whether it matched the target, and detailed reasoning for the result.
last_result = st.session_state.get("last_result")

if last_result:
    st.markdown("-")
    st.markdown("## Latest Result")

    xp_gained = award_xp(last_result["confidence"], practice_mode, last_result["is_match"])
    if xp_gained > 0:
        st.success(f"+{xp_gained} XP earned!")

    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        st.metric("Mode",       last_result["mode"])
    with r2:
        st.metric("Predicted",  last_result["pred_label"])
    with r3:
        st.metric("Confidence", f'{last_result["confidence"]:.3f}')
    with r4:
        status = (
            "Match" if last_result["is_match"]
            else ("N/A" if last_result["target"] is None else "Mismatch")
        )
        st.metric("Result", status)
    with r5:
        st.metric("Band", last_result["band"])

    # Outcome message
    if last_result["target"] is None:
        if last_result["meets_threshold"]:
            st.success("Reliable free-detection prediction.")
        else:
            st.warning("Prediction made, but confidence is below your threshold.")
    else:
        if last_result["is_match"] and last_result["meets_threshold"]:
            st.success(
                f"Great job! '{last_result['target']}' matched with reliable confidence."
            )
        elif last_result["is_match"] and not last_result["meets_threshold"]:
            st.warning(
                f"Sign matched '{last_result['target']}', but confidence was below threshold. "
                "Try a cleaner image."
            )
        else:
            st.error(
                f"No match for '{last_result['target']}'. "
                f"Model predicted '{last_result['pred_label']}' instead."
            )

    # Word practice progress
    if practice_mode == "Word Practice" and st.session_state.get("word_target"):
        wt       = st.session_state["word_target"]
        built    = "".join(st.session_state.get("word_predictions", []))
        complete = st.session_state.get("word_complete", False)

        st.markdown("### Word Practice Progress")
        render_word_progress_bar(
            wt,
            st.session_state.get("word_predictions", []),
            st.session_state.get("word_index", 0),
        )

        if complete:
            if built == wt:
                st.success(f"Word complete! Perfect match: **{built}**")
            else:
                st.warning(f"Word complete. Target: **{wt}** | Spelled: **{built}**")
                errors = [
                    f"Position {i+1}: expected **{t}**, got **{p}**"
                    for i, (t, p) in enumerate(zip(wt, built)) if t != p
                ]
                if errors:
                    st.markdown("**Letter errors:**")
                    for e in errors:
                        st.write(f"  - {e}")

    # Typewriter pending letter notification
    if practice_mode == "Typewriter Mode":
        pending = st.session_state.get("tw_pending")
        if pending:
            st.info(
                f"Predicted letter: **{pending}** -- click **Confirm Letter** (sidebar or "
                "Typewriter tab) to add it to your buffer, or **Backspace** to discard."
            )

    #Reasoning
    st.markdown("### Why this result?")
    for reason in last_result["reasons"]:
        st.write(f"- {reason}")

    #Top-5
    if show_top5:
        st.markdown("### Top-5 Predictions")
        st.dataframe(pd.DataFrame(last_result["top5"]), use_container_width=True, hide_index=True)

    #Confidence interpretation
    st.markdown("### Confidence Interpretation")
    c = last_result["confidence"]
    if c >= 0.95:
        st.success("Model is extremely certain about this prediction.")
    elif c >= 0.85:
        st.info("Model is fairly confident.")
    elif c >= 0.70:
        st.warning("Moderate confidence -- try a cleaner, better-lit image.")
    else:
        st.error(
            "Low confidence. Tips: plain background, hand centered, "
            "fingers separated, good lighting, sign held completely still."
        )

    #Preprocessing debug (opt-in)
    if show_debug_preprocessing and "preprocessed_rgb" in last_result:
        with st.expander("Preprocessing Debug"):
            q    = last_result.get("quality", {})
            meta = last_result.get("meta", {})
            st.write("Image Quality Metrics", q)
            st.write(
                "Preprocessing Metadata",
                {k: v for k, v in meta.items()
                 if k not in ["mask_preview", "gray_preview", "cropped_preview"]},
            )
            pre_img = (last_result["preprocessed_rgb"] * 255).astype(np.uint8)
            st.image(pre_img, caption="Preprocessed model input", width=240)
            dc = st.columns(3)
            with dc[0]:
                if (gp := meta.get("gray_preview")) is not None:
                    st.image(gp, caption="CLAHE grayscale",  use_container_width=True)
            with dc[1]:
                if (mp := meta.get("mask_preview")) is not None:
                    st.image(mp, caption="Foreground mask",  use_container_width=True)
            with dc[2]:
                if (cp := meta.get("cropped_preview")) is not None:
                    st.image(cp, caption="Cropped region",   use_container_width=True)

    #Action buttons
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("Try Again",          use_container_width=True):
            clear_current_attempt()
            st.rerun()
    with a2:
        if st.button("Retest Same Sign",   use_container_width=True):
            clear_current_attempt()
            st.rerun()
    with a3:
        if st.button("Shuffle New Target", use_container_width=True):
            if target_options and practice_mode in ["Target Practice", "Hard Gesture Practice"]:
                reshuffle_queue(target_options)
            clear_current_attempt()
            st.rerun()


#Tab 2: Typewriter / Sign-to-Text Mode: allows users to build custom words by analyzing signs one at a time and confirming them to append to a live buffer.
with tab2:
    st.subheader("Typewriter / Sign-to-Text Mode")
    st.caption(
        "Sign a letter in the Practice tab -> click Analyze Sign -> "
        "come back here and click Confirm Letter to append it. "
        "Build words and sentences, letter by letter."
    )

    st.markdown("### Live Buffer")
    render_typewriter_buffer()

    st.markdown("---")
    tw_c1, tw_c2, tw_c3 = st.columns(3)
    with tw_c1:
        if st.button("Confirm Letter", use_container_width=True, key="tw_confirm_tab"):
            tw_confirm_letter()
            buf = st.session_state.get("tw_buffer", [])
            if len(buf) >= 4:
                st.session_state["typewriter_words_completed"] = (
                    st.session_state.get("typewriter_words_completed", 0) + 1
                )
            st.rerun()
    with tw_c2:
        if st.button("Backspace",      use_container_width=True, key="tw_back_tab"):
            tw_backspace()
            st.rerun()
    with tw_c3:
        if st.button("Clear Buffer",   use_container_width=True, key="tw_clear_tab"):
            tw_clear_buffer()
            st.rerun()

    buf  = st.session_state.get("tw_buffer", [])
    word = "".join(buf)

    if word:
        st.markdown("---")
        st.markdown("### Spell-Check & Auto-Suggest")

        if SPELL_AVAILABLE:
            suggestion = spell_suggest(word)
            if suggestion:
                st.info(f"Did you mean: **{suggestion}**?")
                sc1, sc2 = st.columns(2)
                with sc1:
                    if st.button(f"Use '{suggestion}'", use_container_width=True):
                        st.session_state["tw_buffer"] = list(suggestion)
                        st.session_state["tw_suggest"] = None
                        st.rerun()
                with sc2:
                    if st.button("Keep my spelling", use_container_width=True):
                        st.session_state["tw_suggest"] = None
                        st.rerun()
            else:
                st.success(f"'{word}' looks correctly spelled!")
        else:
            st.info(
                "Install `pyspellchecker` for live auto-suggest: "
                "`pip install pyspellchecker`"
            )

        st.markdown("---")
        st.markdown("### Save Word")
        fin1, fin2 = st.columns(2)
        with fin1:
            if st.button("Save Word to History", use_container_width=True):
                st.session_state["typewriter_words_completed"] = (
                    st.session_state.get("typewriter_words_completed", 0) + 1
                )
                st.session_state.setdefault("tw_completed_words", []).append(word)
                tw_clear_buffer()
                st.success(f"Saved: **{word}**")
                st.rerun()
        with fin2:
            st.markdown(
                f"<p style='padding:6px;font-size:18px'><b>Current word:</b> {word}</p>",
                unsafe_allow_html=True,
            )

    completed = st.session_state.get("tw_completed_words", [])
    if completed:
        st.markdown("---")
        st.markdown("### Words Completed This Session")
        st.write("  |  ".join(f"`{w}`" for w in completed))

    st.markdown(
        """
        ---
        **How Typewriter Mode works:**
        1. Switch to the **Practice** tab, set mode to **Typewriter Mode** in the sidebar.
        2. Analyze a sign -- the predicted letter appears as a **pending letter** (orange, above).
        3. Click **Confirm Letter** here or in the sidebar to lock it in (turns green).
        4. Use **Backspace** to fix mistakes, **Clear** to start over.
        5. The spell-checker runs automatically when you have 2+ letters in the buffer.
        """
    )


#Tab 3: Session Analytics & Gamification: provides a comprehensive breakdown of the user's performance during the current session, 
# including metrics like total attempts, success rate, average confidence, XP earned, level progress, badges earned, and various charts to visualize confidence trends and accuracy by letter. This tab turns raw performance data into actionable insights and motivational feedback to encourage continued practice and improvement.
with tab3:
    st.subheader("Session Analytics & Gamification")
    st.caption(
        "Your performance breakdown, XP progress, earned badges, and BI insights."
    )

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("Total Attempts",  st.session_state["attempts"])
    with s2:
        st.metric("Correct",         st.session_state["correct_attempts"])
    with s3:
        st.metric("Success Rate",    f"{session_success_rate() * 100:.1f}%")
    with s4:
        st.metric("Avg Confidence",  f"{session_average_confidence():.3f}")
    with s5:
        st.metric("Session XP",      f"+{st.session_state.get('session_xp', 0)}")

    st.markdown("---")

    #XP / Level
    st.markdown("### Level Progress")
    render_xp_bar(user_progress.get("total_xp", 0))

    lv1, lv2 = st.columns(2)
    with lv1:
        st.metric("Total Lifetime XP",   user_progress.get("total_xp", 0))
        st.metric("Lifetime Attempts",   user_progress.get("lifetime_attempts", 0))
    with lv2:
        st.metric("Daily Streak",        f"{user_progress.get('daily_streak', 0)} day(s)")
        st.metric("Best Session Streak", st.session_state.get("best_streak", 0))

    st.markdown("---")

    #Badges
    st.markdown("### Earned Badges")
    render_badges(user_progress.get("earned_badges", []))

    earned_set = set(user_progress.get("earned_badges", []))
    unearned   = [(bid, d) for bid, d in BADGE_DEFS.items() if bid not in earned_set]
    if unearned:
        with st.expander(f"{len(unearned)} badge(s) still locked"):
            for bid, (label, desc, emoji) in unearned:
                st.write(f"{emoji} **{label}** -- {desc}")

    st.markdown("---")

    #Charts
    results_df = pd.DataFrame(st.session_state["result_rows"])

    if not results_df.empty:
        st.markdown("### Confidence Over Time")
        conf_chart = pd.DataFrame({
            "Attempt":    range(1, len(st.session_state["confidence_history"]) + 1),
            "Confidence": st.session_state["confidence_history"],
        }).set_index("Attempt")
        st.line_chart(conf_chart)

        targeted = results_df[results_df["target"] != "Free Detection"]
        if not targeted.empty:
            st.markdown("### Per-Letter Accuracy (This Session)")
            letter_acc = (
                targeted.groupby("target")
                .apply(lambda g: round((g["match"] == "Yes").sum() / len(g), 3))
                .reset_index()
                .rename(columns={"target": "Letter", 0: "Accuracy"})
                .sort_values("Accuracy")
            )
            st.bar_chart(letter_acc.set_index("Letter"))

        wrong_df = results_df[results_df["match"] == "No"]
        if not wrong_df.empty:
            st.markdown("### Top Prediction Errors")
            error_counts = (
                wrong_df.groupby(["target", "prediction"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(10)
            )
            st.dataframe(error_counts, use_container_width=True, hide_index=True)

        st.markdown("### Usage by Mode")
        mc = results_df["mode"].value_counts().reset_index()
        mc.columns = ["Mode", "Attempts"]
        st.dataframe(mc, use_container_width=True, hide_index=True)

        st.markdown("### Full Attempt History")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Session Results (CSV)",
            data=csv_bytes,
            file_name="asl_session_results.csv",
            mime="text/csv",
        )

        wh = st.session_state.get("word_history", [])
        if wh:
            st.markdown("### Word Practice History")
            st.dataframe(pd.DataFrame(wh), use_container_width=True, hide_index=True)
    else:
        st.info("No attempts yet -- start practicing in the Practice tab!")

    st.markdown(
        "> **BI Insight:** XP and badge tracking turn raw prediction logs into "
        "a motivational feedback loop -- the same principle behind DAU/retention analytics "
        "in products like Duolingo. The daily streak tracks consistent engagement over time."
    )


#Tab 4: Model Comparison & Curriculum Design: presents a detailed comparison of the three trained models (MobileNetV2, ResNet50, EfficientNetB0) across various evaluation metrics, including overall accuracy, per-class performance, and confusion patterns. 
#This analysis explains why MobileNetV2 was selected as the deployed model and how the curriculum design (easy vs hard letters) is informed by the model's strengths and weaknesses.
with tab4:
    st.subheader("Model Comparison")
    st.caption(
        "Compare the three trained models and understand why MobileNetV2 was "
        "selected as the deployed model.")

    if not model_summary_df.empty:
        display_df = model_summary_df.copy()
        for col in display_df.columns:
            if pd.api.types.is_numeric_dtype(display_df[col]):
                display_df[col] = display_df[col].round(6)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.markdown("### Interpretation")
        st.write(
            "MobileNetV2 was selected because it achieved the highest overall evaluation "
            "performance across all 36 ASL classes. Transfer learning allowed the model to "
            "leverage pre-trained ImageNet features, significantly improving its ability to "
            "distinguish visually similar hand shapes. This model selection decision is "
            "directly reflected in the curriculum design -- hard letters are those where "
            "even the best model struggles."
        )
    else:
        st.info("Model comparison summary not found. Check artifacts/evaluation/.")

    if selected_target:
        st.markdown("### Per-Class Metrics for Current Target")
        row_df = get_target_row(per_class_df, selected_target)
        if not row_df.empty:
            d = row_df.copy()
            for col in d.columns:
                if pd.api.types.is_numeric_dtype(d[col]):
                    d[col] = d[col].round(4)
            st.dataframe(d, use_container_width=True, hide_index=True)
        else:
            st.info(f"No per-class metrics for '{pretty_label(selected_target)}'.")

    st.markdown(
        "> **BI Insight:** Per-class metrics reveal which signs are hardest for the model, "
        "directly informing the easy/hard curriculum split and the anti-frustration trigger threshold."
    )


#Tab 5 
with tab5:
    st.subheader("Failure Analysis & Insights")
    st.caption(
        "Understand domain shift, confusion patterns, and curriculum recommendations "
        "to guide more effective ASL practice."
    )

    st.markdown("### Key Deployment Insight")
    st.write(
        "The model performs strongly on held-out benchmark images but may show reduced accuracy "
        "on real-world camera or phone captures. This domain shift occurs because benchmark "
        "images are cleaner and more controlled than live input. The CLAHE preprocessing and "
        "foreground-crop pipeline in this app mitigate many of these issues."
    )

    st.markdown("### Why Practical Images Can Be Harder")
    st.write(
        "Background clutter disrupts foreground segmentation. "
        "Varying lighting changes pixel intensity distributions the model was not trained on. "
        "Hand distance and angle shift the apparent sign shape between attempts. "
        "Motion blur or soft focus reduces finger boundary clarity. "
        "The anti-frustration pro-tip panel is designed specifically to address these issues "
        "by surfacing known confusion patterns and alignment guidance when a user is stuck."
    )

    st.markdown("### Confusion Insights for Current Target")
    if selected_target:
        confusions = confusion_lookup.get(normalize_label(selected_target), [])
        if confusions:
            st.dataframe(
                pd.DataFrame({"Signs confused with": [pretty_label(x) for x in confusions]}),
                use_container_width=True,
                hide_index=True,
            )
            st.write(
                f"When the model sees '{pretty_label(selected_target)}', it may mistake it for: "
                f"{', '.join(pretty_label(x) for x in confusions[:5])}. "
                "Use Confusion Pair Practice to drill these specifically."
            )
        else:
            st.write(f"No recorded confusion pairs for '{pretty_label(selected_target)}'.")
    else:
        st.info("Select a target sign to see its confusion data.")

    if show_reference_tables and not confusion_pairs_df.empty:
        with st.expander("Full Curriculum Confusion Pairs"):
            st.dataframe(confusion_pairs_df, use_container_width=True, hide_index=True)

    if show_reference_tables and not top_confusions_df.empty:
        with st.expander("Top Model Confusions from Evaluation"):
            st.dataframe(top_confusions_df, use_container_width=True, hide_index=True)

    st.markdown("### Recommended Learning Path")
    easy_from_json = curriculum_json.get("easy_start", [])
    hard_from_json = curriculum_json.get("hard_focus", [])

    col_e, col_h = st.columns(2)
    with col_e:
        st.markdown("#### Easy Start Signs")
        letters = easy_from_json or sorted(easy_letters)
        st.write(", ".join(pretty_label(x) for x in letters) if letters else "Not loaded.")
    with col_h:
        st.markdown("#### Hard Focus Signs")
        letters = hard_from_json or sorted(hard_letters)
        st.write(", ".join(pretty_label(x) for x in letters) if letters else "Not loaded.")

    if selected_target:
        st.markdown("#### Recommendation for Current Target")
        n = normalize_label(selected_target)
        if n in {normalize_label(x) for x in hard_letters}:
            st.write(
                f"Spend extra time on '{pretty_label(selected_target)}'. "
                "Aim for repeated high-confidence matches before moving on. "
                "The Pro-Tip panel activates automatically after 3 consecutive misses "
                "to give you targeted guidance."
            )
        elif n in {normalize_label(x) for x in easy_letters}:
            st.write(
                f"'{pretty_label(selected_target)}' is a beginner-friendly sign. "
                "Once consistent here, move to visually similar signs that are easier to confuse."
            )
        else:
            st.write(
                f"Keep practicing '{pretty_label(selected_target)}' until your confidence "
                "stays above the threshold across multiple tries."
            )

    st.markdown(
        "> **BI Insight:** Failure analysis transforms raw model errors into actionable learning "
        "recommendations. The anti-frustration system is a direct BI application: it detects "
        "a pattern (repeated failures), diagnoses it (confusion lookup), and surfaces a targeted "
        "intervention (pro-tip + reference image) -- closing the feedback loop automatically."
    )


#Footer with contact info, course info, and model details for reference and attribution.
st.markdown("---")
st.caption(
    "SignaMind: Learns how you learn sign | ASL Learning Intelligence Dashboard "
    "Shubhangi Singh (041162377) | Model: MobileNetV2 | 36 classes (A-Z, 0-9)")
