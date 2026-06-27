from pathlib import Path
import subprocess
import sys

import pandas as pd
import streamlit as st
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
LOG_DIR = PROJECT_ROOT / "logs"
REPORT_DIR = PROJECT_ROOT / "reports"

TRACKING_APP = SRC_DIR / "main_app_tracking_landmark_rule.py"
SUMMARY_SCRIPT = SRC_DIR / "summarize_tracking_logs.py"


st.set_page_config(
    page_title="Audience Analytics Dashboard",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #07111f 0%, #0f172a 45%, #111827 100%);
            color: #e5e7eb;
        }

        [data-testid="stSidebar"] {
            background-color: #050816;
            border-right: 1px solid rgba(148, 163, 184, 0.20);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.5rem 1.7rem;
            border-radius: 24px;
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.24), transparent 28%),
                radial-gradient(circle at top right, rgba(129, 140, 248, 0.20), transparent 30%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.92));
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.30);
            margin-bottom: 1.2rem;
        }

        .hero-title {
            font-size: 2.15rem;
            font-weight: 850;
            color: #f8fafc;
            margin-bottom: 0.35rem;
            letter-spacing: -0.03em;
        }

        .hero-subtitle {
            color: #cbd5e1;
            font-size: 1rem;
            max-width: 950px;
            line-height: 1.55;
        }

        .pipeline {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .pill {
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            color: #dbeafe;
            background: rgba(59, 130, 246, 0.14);
            border: 1px solid rgba(96, 165, 250, 0.34);
            font-size: 0.84rem;
            font-weight: 600;
        }

        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 12px 35px rgba(0,0,0,0.20);
        }

        div[data-testid="stMetricLabel"] {
            color: #94a3b8;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc;
            font-weight: 850;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 800;
            color: #f8fafc;
            margin-top: 0.5rem;
            margin-bottom: 0.75rem;
        }

        .stButton button {
            border-radius: 14px;
            font-weight: 750;
            border: 1px solid rgba(125, 211, 252, 0.38);
            background: rgba(14, 165, 233, 0.13);
            color: #e0f2fe;
        }

        .stButton button:hover {
            border: 1px solid rgba(125, 211, 252, 0.95);
            color: white;
        }

        .metric-card-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1.1rem;
            margin-top: 0.4rem;
            margin-bottom: 1.2rem;
        }

        .metric-card {
            position: relative;
            min-height: 145px;
            padding: 1.25rem 1.35rem;
            border-radius: 22px;
            background: rgba(15, 23, 42, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 12px 35px rgba(0,0,0,0.20);
            overflow: hidden;
            transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            border-color: rgba(125, 211, 252, 0.70);
            box-shadow: 0 18px 45px rgba(14, 165, 233, 0.18);
        }

        .metric-label {
            color: #dbeafe;
            font-size: 0.95rem;
            font-weight: 750;
            margin-bottom: 0.8rem;
        }

        .metric-value {
            color: #f8fafc;
            font-size: 2.35rem;
            line-height: 1;
            font-weight: 900;
            letter-spacing: -0.04em;
        }

        .metric-hover {
            position: absolute;
            left: 0.85rem;
            right: 0.85rem;
            bottom: 0.85rem;
            min-height: 0;
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: rgba(8, 13, 26, 0.82);
            border: 1px solid rgba(125, 211, 252, 0.35);
            backdrop-filter: blur(12px);
            color: #e0f2fe;
            font-size: 0.86rem;
            line-height: 1.45;
            opacity: 0;
            transform: translateY(18px) scale(0.98);
            transition: opacity 0.20s ease, transform 0.20s ease;
            pointer-events: none;
        }

        .metric-card:hover .metric-hover {
            opacity: 1;
            transform: translateY(0) scale(1);
        }

        @media (max-width: 1200px) {
            .metric-card-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 700px) {
            .metric-card-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def latest_file(folder: Path, pattern: str):
    files = list(folder.glob(pattern))

    if not files:
        return None

    return max(files, key=lambda path: path.stat().st_mtime)


def load_csv(path):
    if path is None or not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"CSV okunamadı: {path}")
        st.exception(exc)
        return None


def final_interest_from_seconds(seconds):
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


def prepare_visitor_df(visitor_df):
    if visitor_df is None or visitor_df.empty:
        return visitor_df

    visitor_df = visitor_df.copy()

    if "final_interest_level" not in visitor_df.columns:
        visitor_df["final_interest_level"] = visitor_df["max_continuous_facing_seconds"].apply(
            final_interest_from_seconds
        )

    return visitor_df


def get_value(row, key, default="-"):
    if key not in row:
        return default

    value = row[key]

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    return value


def run_tracking_session():
    if not TRACKING_APP.exists():
        st.error(f"Tracking app bulunamadı: {TRACKING_APP}")
        return

    st.info("Kamera penceresi ayrı açılacak. Oturumu bitirmek için kamera penceresinde q tuşuna bas.")

    result = subprocess.run(
        [sys.executable, str(TRACKING_APP)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        st.error("Tracking session çalışırken hata oluştu.")
        st.code(result.stderr)
        return

    st.success("Tracking session tamamlandı.")

    with st.expander("Tracking output"):
        st.code(result.stdout[-2500:])


def generate_latest_report():
    if not SUMMARY_SCRIPT.exists():
        st.error(f"Summary script bulunamadı: {SUMMARY_SCRIPT}")
        return

    result = subprocess.run(
        [sys.executable, str(SUMMARY_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        st.error("Rapor üretirken hata oluştu.")
        st.code(result.stderr)
        return

    st.success("Latest report generated.")

    with st.expander("Report output"):
        st.code(result.stdout[-2500:])


def show_header():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Camera-Based Audience Analytics</div>
            <div class="hero-subtitle">
                Real-time visitor tracking, landmark-based orientation analysis,
                engagement estimation, demographic classification and statistical reporting.
            </div>
            <div class="pipeline">
                <span class="pill">MediaPipe Face Landmarks</span>
                <span class="pill">Landmark Rule Orientation</span>
                <span class="pill">Visitor Tracking</span>
                <span class="pill">Age Group Classifier</span>
                <span class="pill">Gender Classifier</span>
                <span class="pill">CSV Reports</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_sidebar(latest_log, latest_visitor, latest_session):
    st.sidebar.title("Control Panel")
    st.sidebar.caption("Tracking, reporting and dashboard controls")

    st.sidebar.divider()

    if st.sidebar.button("▶ Start Tracking Session", use_container_width=True):
        run_tracking_session()

    if st.sidebar.button("📄 Generate Latest Report", use_container_width=True):
        generate_latest_report()

    if st.sidebar.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()

    st.sidebar.divider()

    st.sidebar.markdown("### Latest Files")

    if latest_log:
        st.sidebar.caption(f"Raw log: `{latest_log.name}`")
    else:
        st.sidebar.caption("Raw log: not found")

    if latest_visitor:
        st.sidebar.caption(f"Visitor summary: `{latest_visitor.name}`")
    else:
        st.sidebar.caption("Visitor summary: not found")

    if latest_session:
        st.sidebar.caption(f"Session summary: `{latest_session.name}`")
    else:
        st.sidebar.caption("Session summary: not found")

    st.sidebar.divider()

    st.sidebar.markdown("### Active Tracking App")
    st.sidebar.code(str(TRACKING_APP.relative_to(PROJECT_ROOT)))




def show_metrics(session_df, visitor_df):
    if session_df is None or session_df.empty:
        st.warning("Henüz session summary yok. Önce tracking çalıştırıp rapor üretmelisin.")
        return

    row = session_df.iloc[-1]

    visitor_count = int(get_value(row, "visitor_count", 0))
    avg_duration = float(get_value(row, "avg_duration_seconds", 0))
    avg_front = float(get_value(row, "avg_front_facing_ratio_percent", 0))
    avg_max_facing = float(get_value(row, "avg_max_continuous_facing_seconds", 0))

    if "high_interest_visitor_count" in session_df.columns:
        high_interest_count = int(get_value(row, "high_interest_visitor_count", 0))
    elif visitor_df is not None and not visitor_df.empty:
        high_interest_count = int((visitor_df["final_interest_level"] == "high_interest").sum())
    else:
        high_interest_count = 0

    common_age = get_value(row, "most_common_age_group", "-")
    common_gender = get_value(row, "most_common_gender", "-")

    if "most_common_final_interest" in session_df.columns:
        common_interest = get_value(row, "most_common_final_interest", "-")
    elif visitor_df is not None and not visitor_df.empty:
        common_interest = visitor_df["final_interest_level"].mode().iloc[0]
    else:
        common_interest = "-"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Visitors", visitor_count)
    col2.metric("Avg Duration", f"{avg_duration:.1f}s")
    col3.metric("Avg Front-Facing", f"{avg_front:.1f}%")
    col4.metric("High Interest", high_interest_count)

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg Max Facing", f"{avg_max_facing:.1f}s")
    col6.metric("Dominant Age", common_age)
    col7.metric("Dominant Gender", common_gender)
    col8.metric("Common Interest", common_interest)

def show_overview(visitor_df):
    if visitor_df is None or visitor_df.empty:
        st.info("Visitor summary bulunamadı.")
        return

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            visitor_df,
            x="visitor_id",
            y="duration_seconds",
            title="Visitor Duration",
            labels={
                "visitor_id": "Visitor",
                "duration_seconds": "Duration (s)",
            },
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            visitor_df,
            x="visitor_id",
            y="front_facing_ratio_percent",
            title="Front-Facing Ratio",
            labels={
                "visitor_id": "Visitor",
                "front_facing_ratio_percent": "Front-Facing %",
            },
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


def show_visitor_table(visitor_df):
    if visitor_df is None or visitor_df.empty:
        st.info("Visitor summary bulunamadı.")
        return

    columns = [
        "visitor_id",
        "duration_seconds",
        "front_facing_seconds",
        "not_facing_seconds",
        "front_facing_ratio_percent",
        "max_continuous_facing_seconds",
        "final_interest_level",
        "dominant_age_group",
        "average_age_confidence",
        "dominant_gender",
        "average_gender_confidence",
        "record_count",
    ]

    existing_columns = [col for col in columns if col in visitor_df.columns]

    display_df = visitor_df[existing_columns].rename(columns={
        "visitor_id": "Visitor",
        "duration_seconds": "Duration (s)",
        "front_facing_seconds": "Front-Facing (s)",
        "not_facing_seconds": "Not-Facing (s)",
        "front_facing_ratio_percent": "Front-Facing %",
        "max_continuous_facing_seconds": "Max Facing (s)",
        "final_interest_level": "Final Interest",
        "dominant_age_group": "Age Group",
        "average_age_confidence": "Age Confidence",
        "dominant_gender": "Gender",
        "average_gender_confidence": "Gender Confidence",
        "record_count": "Records",
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def show_demographics(visitor_df):
    if visitor_df is None or visitor_df.empty:
        st.info("Demographic summary bulunamadı.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        gender_df = visitor_df["dominant_gender"].value_counts().reset_index()
        gender_df.columns = ["Gender", "Count"]

        fig = px.pie(
            gender_df,
            names="Gender",
            values="Count",
            title="Gender Distribution",
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        age_df = visitor_df["dominant_age_group"].value_counts().reset_index()
        age_df.columns = ["Age Group", "Count"]

        fig = px.pie(
            age_df,
            names="Age Group",
            values="Count",
            title="Age Group Distribution",
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        interest_df = visitor_df["final_interest_level"].value_counts().reset_index()
        interest_df.columns = ["Interest", "Count"]

        fig = px.pie(
            interest_df,
            names="Interest",
            values="Count",
            title="Interest Distribution",
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Age / Gender vs Final Interest")

    pivot = pd.crosstab(
        [visitor_df["dominant_age_group"], visitor_df["dominant_gender"]],
        visitor_df["final_interest_level"],
    )

    st.dataframe(pivot, use_container_width=True)


def show_raw_log(latest_log):
    if latest_log is None:
        st.info("Raw log bulunamadı.")
        return

    log_df = load_csv(latest_log)

    if log_df is None or log_df.empty:
        st.info("Raw log boş.")
        return

    st.caption(f"Showing latest raw log: `{latest_log.name}`")
    st.dataframe(log_df.tail(60), use_container_width=True, hide_index=True)


def show_exports(latest_log, latest_visitor, latest_session):
    st.markdown("### Download CSV Files")

    col1, col2, col3 = st.columns(3)

    if latest_visitor and latest_visitor.exists():
        col1.download_button(
            "Download Visitor Summary",
            data=latest_visitor.read_bytes(),
            file_name=latest_visitor.name,
            mime="text/csv",
            use_container_width=True,
        )

    if latest_session and latest_session.exists():
        col2.download_button(
            "Download Session Summary",
            data=latest_session.read_bytes(),
            file_name=latest_session.name,
            mime="text/csv",
            use_container_width=True,
        )

    if latest_log and latest_log.exists():
        col3.download_button(
            "Download Raw Log",
            data=latest_log.read_bytes(),
            file_name=latest_log.name,
            mime="text/csv",
            use_container_width=True,
        )


def main():
    latest_log = latest_file(LOG_DIR, "tracking_session_*.csv")
    latest_visitor = latest_file(REPORT_DIR, "visitor_summary_*.csv")
    latest_session = latest_file(REPORT_DIR, "session_summary_*.csv")

    visitor_df = prepare_visitor_df(load_csv(latest_visitor))
    session_df = load_csv(latest_session)

    show_sidebar(latest_log, latest_visitor, latest_session)
    show_header()

    st.markdown('<div class="section-title">Latest Session Metrics</div>', unsafe_allow_html=True)
    show_metrics(session_df, visitor_df)

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Visitors",
        "Demographics",
        "Raw Log",
        "Exports",
    ])

    with tab1:
        show_overview(visitor_df)

    with tab2:
        show_visitor_table(visitor_df)

    with tab3:
        show_demographics(visitor_df)

    with tab4:
        show_raw_log(latest_log)

    with tab5:
        show_exports(latest_log, latest_visitor, latest_session)


if __name__ == "__main__":
    main()
