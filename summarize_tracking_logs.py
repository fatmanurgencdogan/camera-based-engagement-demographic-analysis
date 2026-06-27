from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = PROJECT_ROOT / "logs"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


AGE_CONFIDENCE_THRESHOLD = 0.60
GENDER_CONFIDENCE_THRESHOLD = 0.60


INTEREST_RANK = {
    "uninterested": 0,
    "low_interest": 1,
    "medium_interest": 2,
    "high_interest": 3,
}


def get_latest_tracking_log():
    files = list(LOG_DIR.glob("tracking_session_*.csv"))

    if not files:
        raise FileNotFoundError("No tracking_session_*.csv file found in logs/")

    return max(files, key=lambda path: path.stat().st_mtime)


def final_interest_from_seconds(seconds):
    """
    Visitor'ın final ilgi seviyesini maksimum kesintisiz kameraya bakma süresine göre belirler.
    """

    try:
        seconds = float(seconds)
    except Exception:
        return "unknown"

    if seconds <= 0:
        return "uninterested"
    if seconds < 3:
        return "low_interest"
    if seconds < 6:
        return "medium_interest"

    return "high_interest"


def highest_interest_reached(series):
    valid = series.dropna()

    if valid.empty:
        return "unknown"

    best_value = "uninterested"
    best_rank = -1

    for value in valid:
        rank = INTEREST_RANK.get(value, -1)

        if rank > best_rank:
            best_rank = rank
            best_value = value

    return best_value


def most_common_label(df, column):
    if column not in df.columns:
        return "unknown"

    valid = df[
        (df[column].notna()) &
        (df[column] != "unknown")
    ]

    if valid.empty:
        return "unknown"

    return valid[column].mode().iloc[0]


def confidence_weighted_label(df, label_col, confidence_col, min_confidence):
    """
    Her saniye gelen age/gender tahminleri bazen değişebilir.
    Bu yüzden visitor için tek sonuç seçerken confidence ağırlıklı oy kullanıyoruz.

    Önce düşük confidence değerlerini eliyoruz.
    Eğer hiç yeterli confidence yoksa, fallback olarak tüm tahminlere bakıyoruz.
    """

    if label_col not in df.columns or confidence_col not in df.columns:
        return "unknown"

    valid = df[
        (df[label_col].notna()) &
        (df[label_col] != "unknown") &
        (df[confidence_col].notna()) &
        (df[confidence_col] >= min_confidence)
    ]

    if valid.empty:
        valid = df[
            (df[label_col].notna()) &
            (df[label_col] != "unknown") &
            (df[confidence_col].notna())
        ]

    if valid.empty:
        return "unknown"

    scores = valid.groupby(label_col)[confidence_col].sum()

    return scores.idxmax()


def average_confidence_for_label(df, label_col, confidence_col, selected_label, min_confidence):
    if selected_label == "unknown":
        return 0.0

    valid = df[
        (df[label_col] == selected_label) &
        (df[confidence_col].notna()) &
        (df[confidence_col] >= min_confidence)
    ]

    if valid.empty:
        valid = df[
            (df[label_col] == selected_label) &
            (df[confidence_col].notna())
        ]

    if valid.empty:
        return 0.0

    return valid[confidence_col].mean()


def safe_mode(series):
    valid = series.dropna()

    if valid.empty:
        return "unknown"

    return valid.mode().iloc[0]


def summarize_tracking_log(log_path):
    df = pd.read_csv(log_path)

    if df.empty:
        print("Log file is empty.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    numeric_columns = [
        "face_detected",
        "orientation_confidence",
        "facing_seconds",
        "age_confidence",
        "gender_confidence",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    visitor_df = df[df["visitor_id"] != "none"].copy()

    if visitor_df.empty:
        print("No visitor records found.")
        return

    visitor_summaries = []

    for visitor_id, group in visitor_df.groupby("visitor_id"):
        group = group.sort_values("timestamp")

        start_time = group["timestamp"].iloc[0]
        end_time = group["timestamp"].iloc[-1]

        duration_seconds = (end_time - start_time).total_seconds() + 1
        record_count = len(group)

        front_facing_seconds = int((group["stabilized_orientation"] == "front_facing").sum())
        not_facing_seconds = int((group["stabilized_orientation"] == "not_facing").sum())

        if record_count > 0:
            front_facing_ratio_percent = front_facing_seconds / record_count * 100
        else:
            front_facing_ratio_percent = 0.0

        max_facing_seconds = float(group["facing_seconds"].max())

        final_interest_level = final_interest_from_seconds(max_facing_seconds)
        dominant_interest_level = most_common_label(group, "interest_level")
        highest_interest_level = highest_interest_reached(group["interest_level"])

        dominant_age_group = confidence_weighted_label(
            group,
            label_col="age_group",
            confidence_col="age_confidence",
            min_confidence=AGE_CONFIDENCE_THRESHOLD,
        )

        dominant_gender = confidence_weighted_label(
            group,
            label_col="gender",
            confidence_col="gender_confidence",
            min_confidence=GENDER_CONFIDENCE_THRESHOLD,
        )

        average_age_confidence = average_confidence_for_label(
            group,
            label_col="age_group",
            confidence_col="age_confidence",
            selected_label=dominant_age_group,
            min_confidence=AGE_CONFIDENCE_THRESHOLD,
        )

        average_gender_confidence = average_confidence_for_label(
            group,
            label_col="gender",
            confidence_col="gender_confidence",
            selected_label=dominant_gender,
            min_confidence=GENDER_CONFIDENCE_THRESHOLD,
        )

        visitor_summaries.append({
            "session_id": group["session_id"].iloc[0],
            "visitor_id": visitor_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": round(duration_seconds, 2),
            "front_facing_seconds": front_facing_seconds,
            "not_facing_seconds": not_facing_seconds,
            "front_facing_ratio_percent": round(front_facing_ratio_percent, 2),
            "max_continuous_facing_seconds": round(max_facing_seconds, 2),
            "dominant_interest_level": dominant_interest_level,
            "highest_interest_level_reached": highest_interest_level,
            "final_interest_level": final_interest_level,
            "dominant_age_group": dominant_age_group,
            "average_age_confidence": round(float(average_age_confidence), 4),
            "dominant_gender": dominant_gender,
            "average_gender_confidence": round(float(average_gender_confidence), 4),
            "record_count": record_count,
        })

    visitor_summary_df = pd.DataFrame(visitor_summaries)

    session_id = visitor_df["session_id"].iloc[0]

    visitor_report_path = REPORT_DIR / f"visitor_summary_{session_id}.csv"
    visitor_summary_df.to_csv(visitor_report_path, index=False)

    high_interest_visitor_count = int(
        (visitor_summary_df["final_interest_level"] == "high_interest").sum()
    )

    session_summary = {
        "session_id": session_id,
        "visitor_count": len(visitor_summary_df),
        "total_records": len(visitor_df),
        "avg_duration_seconds": round(visitor_summary_df["duration_seconds"].mean(), 2),
        "avg_front_facing_ratio_percent": round(visitor_summary_df["front_facing_ratio_percent"].mean(), 2),
        "avg_max_continuous_facing_seconds": round(visitor_summary_df["max_continuous_facing_seconds"].mean(), 2),
        "high_interest_visitor_count": high_interest_visitor_count,
        "most_common_age_group": safe_mode(visitor_summary_df["dominant_age_group"]),
        "most_common_gender": safe_mode(visitor_summary_df["dominant_gender"]),
        "most_common_final_interest": safe_mode(visitor_summary_df["final_interest_level"]),
    }

    session_summary_df = pd.DataFrame([session_summary])

    session_report_path = REPORT_DIR / f"session_summary_{session_id}.csv"
    session_summary_df.to_csv(session_report_path, index=False)

    demographic_interest = pd.crosstab(
        [visitor_summary_df["dominant_age_group"], visitor_summary_df["dominant_gender"]],
        visitor_summary_df["final_interest_level"],
    )

    demographic_interest_path = REPORT_DIR / f"demographic_interest_{session_id}.csv"
    demographic_interest.to_csv(demographic_interest_path)

    print("==============================")
    print("Tracking Log Summary")
    print("==============================")
    print(f"Log file: {log_path}")
    print()
    print("Visitor summary:")
    print(visitor_summary_df)
    print()
    print("Session summary:")
    print(session_summary_df)
    print()
    print("==============================")
    print("Reports saved")
    print("==============================")
    print(f"Visitor report: {visitor_report_path}")
    print(f"Session report: {session_report_path}")
    print(f"Demographic interest report: {demographic_interest_path}")


def main():
    log_path = get_latest_tracking_log()
    summarize_tracking_log(log_path)


if __name__ == "__main__":
    main()
