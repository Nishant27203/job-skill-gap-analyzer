from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from src.jobskill.core import (
    AnalysisInputs,
    analyze_jobs_dataframe,
    forecast_skill_demand_linear,
)


st.set_page_config(page_title="Job Skill Gap Analyzer", layout="wide")
_BASE_DIR = Path(__file__).resolve().parent

_DASHBOARD_CSS = """
<style>
  /* Layout tweaks */
  .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1200px; }
  /* Hide Streamlit chrome */
  header[data-testid="stHeader"] { background: transparent; }
  footer { visibility: hidden; }

  /* Hero */
  .hero {
    padding: 1.25rem 1.25rem;
    border-radius: 18px;
    background: radial-gradient(1200px 400px at 10% 10%, rgba(99, 102, 241, 0.35), transparent 60%),
                radial-gradient(1000px 420px at 90% 30%, rgba(34, 197, 94, 0.25), transparent 55%),
                linear-gradient(135deg, rgba(15, 23, 42, 1) 0%, rgba(2, 6, 23, 1) 100%);
    border: 1px solid rgba(148, 163, 184, 0.18);
  }
  .hero-title {
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: rgba(248, 250, 252, 1);
    margin: 0;
  }
  .hero-subtitle {
    margin-top: 0.35rem;
    font-size: 0.95rem;
    color: rgba(226, 232, 240, 0.86);
  }

  /* Cards */
  .card {
    padding: 1rem 1rem;
    border-radius: 16px;
    background: rgba(2, 6, 23, 0.35);
    border: 1px solid rgba(148, 163, 184, 0.18);
  }
  .kpi-label { color: rgba(148, 163, 184, 1); font-size: 0.85rem; margin-bottom: 0.2rem; }
  .kpi-value { color: rgba(15, 23, 42, 1); font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; }
  .kpi-delta { color: rgba(100, 116, 139, 1); font-size: 0.85rem; margin-top: 0.1rem; }

  /* Make metric cards look consistent */
  div[data-testid="stMetric"] {
    padding: 0.9rem 1rem;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    background: rgba(248, 250, 252, 1);
  }
  div[data-testid="stMetricLabel"] > div { color: rgba(100, 116, 139, 1); }
  div[data-testid="stMetricValue"] > div { color: rgba(15, 23, 42, 1); }
  div[data-testid="stMetricDelta"] > div { color: rgba(71, 85, 105, 1); }

  /* Dataframe rounding */
  div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; border: 1px solid rgba(148, 163, 184, 0.18); }
</style>
"""


def _df_download_button(df: pd.DataFrame, *, label: str, filename: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")

def _safe_read_csv(uploaded) -> Optional[pd.DataFrame]:
    """
    Try hard to read "any" CSV:
    - auto delimiter detection (python engine)
    - tries utf-8 then latin-1
    """
    data = uploaded.getvalue()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                tempfile.NamedTemporaryFile(delete=False).name,  # placeholder (won't be used)
            )
        except Exception:
            pass

    # Fallback: use BytesIO with pandas; try variations
    import io

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        for kwargs in (
            dict(sep=None, engine="python"),
            dict(sep=",", engine="python"),
            dict(sep=";", engine="python"),
            dict(sep="\t", engine="python"),
        ):
            try:
                bio = io.BytesIO(data)
                return pd.read_csv(bio, encoding=encoding, **kwargs)
            except Exception:
                continue
    return None


def _guess_best_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _auto_text_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    hit = _guess_best_column(
        cols,
        [
            "job description",
            "description",
            "job_description",
            "jobdescription",
            "summary",
            "details",
            "requirements",
            "minimum qualifications",
            "preferred skills",
        ],
    )
    if hit:
        return hit
    # heuristic: pick the text-like column with highest average length
    text_cols = [c for c in cols if df[c].dtype == object]
    if not text_cols:
        return None
    lengths = []
    for c in text_cols[:60]:
        s = df[c].fillna("").astype(str)
        lengths.append((c, float(s.str.len().clip(upper=5000).mean())))
    lengths.sort(key=lambda x: x[1], reverse=True)
    return lengths[0][0] if lengths else None


def _auto_title_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    return _guess_best_column(cols, ["title", "job title", "business title", "position", "role"])


def _auto_date_column(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    hit = _guess_best_column(cols, ["posting date", "post date", "posted date", "date", "created_at", "created at"])
    return hit


def _combine_text_columns(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    If there is no obvious description column, combine several object columns into one text field.
    """
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        return None
    take = obj_cols[:8]
    parts = []
    for c in take:
        parts.append(df[c].fillna("").astype(str))
    out = parts[0]
    for p in parts[1:]:
        out = out + " \n " + p
    return out


def main() -> None:
    st.markdown(_DASHBOARD_CSS, unsafe_allow_html=True)
    st.markdown(
        """
<div class="hero">
  <div class="hero-title">Job Skill Gap Analyzer</div>
  <div class="hero-subtitle">
    Upload job data, extract in-demand skills, compare with candidate skills, and export a clean report.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Upload first so we can offer column dropdowns.
    uploaded = st.sidebar.file_uploader("Job dataset (CSV)", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV to begin. You can use `data/jobs_sample.csv` as a starter dataset.")
        return

    df = _safe_read_csv(uploaded)
    if df is None:
        st.error("Could not read the uploaded file as a CSV. Please upload a valid .csv file.")
        return
    columns = list(df.columns)

    default_text = _auto_text_column(df) or columns[0]
    default_title = _auto_title_column(df)
    default_date = _auto_date_column(df)

    with st.sidebar:
        st.header("Inputs")
        taxonomy_path = st.text_input("Skill taxonomy path", value="skill_taxonomy/skills.json")

        st.divider()
        st.header("Columns / Filters")
        text_column = st.selectbox(
            "Text column (job description)",
            options=["(auto)"] + columns,
            index=0,
            help="(auto) picks the best column by name/length; use a specific column if needed.",
        )
        title_column = st.selectbox(
            "Title column (optional)",
            options=["(none)"] + columns,
            index=(["(none)"] + columns).index(default_title) if default_title in columns else 0,
        )
        date_column = st.selectbox(
            "Date column (optional, for forecasting)",
            options=["(none)"] + columns,
            index=(["(none)"] + columns).index(default_date) if default_date in columns else 0,
        )
        role_filter = st.text_input("Role filter (optional)", value="")

        st.divider()
        st.header("Performance")
        max_rows = st.slider(
            "Max rows to analyze",
            min_value=200,
            max_value=20000,
            value=5000,
            step=200,
            help="For very large datasets, analyzing a sample keeps the app fast.",
        )

        st.divider()
        st.header("Candidate skills")
        candidate_text = st.text_area("Candidate skills (comma/newline separated)", value="Python, SQL, Excel", height=120)
        candidate_file = st.file_uploader("Or upload candidate skills file (txt)", type=["txt"])

        st.divider()
        st.header("Thresholds")
        top_n = st.slider("Top-N skills", min_value=5, max_value=100, value=20, step=5)
        min_required_percent = st.slider("Min required percent", min_value=0, max_value=100, value=30, step=5)
        min_required_count = st.slider("Min required count", min_value=1, max_value=50, value=1, step=1)

        run = st.button("Run analysis", type="primary", use_container_width=True)

    if not run:
        st.warning("Set options in the sidebar, then click **Run analysis**.")
        return

    # Determine actual columns (auto mode)
    resolved_text_col = default_text if text_column == "(auto)" else text_column
    if resolved_text_col not in df.columns:
        # Fallback: combine text columns
        combined = _combine_text_columns(df)
        if combined is None:
            st.error("Could not find any text columns to analyze in this dataset.")
            return
        df = df.copy()
        df["_combined_text"] = combined
        resolved_text_col = "_combined_text"

    resolved_title_col = None if title_column == "(none)" else title_column.strip()
    resolved_date_col = None if date_column == "(none)" else date_column.strip()

    # If role_filter set but no usable title column, just ignore role filter (no crash)
    if role_filter.strip() and (resolved_title_col is None or resolved_title_col not in df.columns):
        st.warning("Role filter ignored because a valid Title column was not selected in this dataset.")
        role_filter = ""

    taxonomy_file = Path(taxonomy_path)
    if not taxonomy_file.is_absolute():
        taxonomy_file = (_BASE_DIR / taxonomy_file).resolve()
    if not taxonomy_file.exists() or not taxonomy_file.is_file():
        st.error(f"Skill taxonomy file not found: {taxonomy_file}")
        return

    candidate_tmp_path = None
    if candidate_file is not None:
        tmp = tempfile.NamedTemporaryFile(prefix="candidate_skills_", suffix=".txt", delete=False)
        tmp.write(candidate_file.getvalue())
        tmp.flush()
        tmp.close()
        candidate_tmp_path = Path(tmp.name)

    try:
        if df.empty:
            st.error("Your CSV has 0 rows. Please upload a dataset with at least 1 job posting.")
            return
        if df[resolved_text_col].dropna().astype(str).str.strip().eq("").all():
            st.error("The selected job description column appears to be empty for all rows.")
            return

        # Sample for performance
        if len(df) > int(max_rows):
            df = df.sample(n=int(max_rows), random_state=42).reset_index(drop=True)
            st.info(f"Analyzing a random sample of {max_rows:,} rows for performance.")

        result = analyze_jobs_dataframe(
            df,
            taxonomy_path=taxonomy_file,
            inputs=AnalysisInputs(
                text_column=resolved_text_col,
                title_column=resolved_title_col,
                role_filter=role_filter,
                date_column=resolved_date_col,
                candidate_text=candidate_text,
                candidate_file=candidate_tmp_path,
                top_n=int(top_n),
                min_required_percent=float(min_required_percent),
                min_required_count=int(min_required_count),
            ),
        )
    except Exception as e:
        st.error(
            "Something went wrong while analyzing this dataset. "
            "Fix the settings below and try again."
        )
        with st.expander("Diagnostics (for fixing dataset settings)"):
            st.code(f"{type(e).__name__}: {e}")
            st.caption(
                "Most common fixes: choose the correct Job Description column, "
                "pick a Title column if Role filter is set, and choose a valid Date column for forecasting."
            )
        return
    finally:
        if candidate_tmp_path is not None and candidate_tmp_path.exists():
            candidate_tmp_path.unlink(missing_ok=True)

    # Top KPIs
    rows_analyzed = int(result.gap_json["notes"]["rows_analyzed"])
    role_filter_used = result.gap_json["notes"]["role_filter"]
    unique_skills = int(result.counts_all["skill"].nunique()) if not result.counts_all.empty else 0
    required_cnt = len(result.required_skills)
    missing_cnt = len(result.missing_skills)

    k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 1])
    k1.metric("Rows analyzed", f"{rows_analyzed:,}")
    k2.metric("Unique skills found", f"{unique_skills:,}")
    k3.metric("Required skills", f"{required_cnt:,}", help="Based on the thresholds on the left")
    k4.metric("Candidate skills", f"{len(result.candidate_skills):,}")
    k5.metric("Missing skills", f"{missing_cnt:,}")

    if role_filter_used:
        st.caption(f"Filtered by title contains: **{role_filter_used}**")
    else:
        st.caption("No role filter applied.")

    st.divider()

    tab_overview, tab_top, tab_gap, tab_forecast, tab_downloads = st.tabs(
        ["Overview", "Top skills", "Gap & recommendations", "Forecast", "Downloads"]
    )

    with tab_overview:
        left, right = st.columns([1.25, 1])
        with left:
            st.subheader("Top skills (market demand)")
            st.dataframe(result.top_skills, use_container_width=True, height=420)

        with right:
            st.subheader("Skill gap summary")
            st.write(
                {
                    "rows_analyzed": rows_analyzed,
                    "role_filter": role_filter_used,
                    "required_skills_count": required_cnt,
                    "candidate_skills_count": len(result.candidate_skills),
                    "missing_skills_count": missing_cnt,
                }
            )
            if required_cnt == 0:
                st.info("No required skills under the current thresholds. Lower thresholds to see recommendations.")
            elif missing_cnt == 0:
                st.success("No missing skills for the chosen thresholds.")
            else:
                st.warning("Missing skills found — check the recommendations tab.")

    with tab_top:
        st.subheader("Top skills (market demand)")
        st.dataframe(result.top_skills, use_container_width=True, height=420)

        if not result.top_skills.empty:
            fig = px.bar(
                result.top_skills.sort_values("percent", ascending=True),
                x="percent",
                y="skill",
                orientation="h",
                title="Top skills in dataset",
                template="plotly_white",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    with tab_gap:
        st.subheader("Missing skills (ranked by demand)")
        if not result.missing_ranked_df.empty:
            st.dataframe(result.missing_ranked_df[["skill", "count", "percent"]], use_container_width=True, height=420)
        elif required_cnt == 0:
            st.info("No required skills under the current thresholds. Lower thresholds to see recommendations.")
        else:
            st.success("No missing skills for the chosen thresholds.")

        st.divider()
        st.subheader("Required skills vs candidate coverage")
        if result.comparison_df.empty:
            st.info("No required skills under the current thresholds. Lower thresholds to see comparisons.")
        else:
            st.dataframe(result.comparison_df, use_container_width=True, height=420)
            fig2 = px.bar(
                result.comparison_df.sort_values(["percent", "skill"], ascending=[True, True]).tail(30),
                x="percent",
                y="skill",
                orientation="h",
                color="candidate_has",
                title="Required skills vs candidate coverage (top 30 by demand)",
                labels={"candidate_has": "Candidate has"},
                color_discrete_map={True: "#16A34A", False: "#DC2626"},
                template="plotly_white",
            )
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)

    with tab_downloads:
        st.subheader("Downloads")
        c1, c2 = st.columns([1, 1])
        with c1:
            _df_download_button(result.top_skills, label="Download top_skills.csv", filename="top_skills.csv")
            if not result.comparison_df.empty:
                _df_download_button(result.comparison_df, label="Download skill_comparison.csv", filename="skill_comparison.csv")
            if not result.missing_ranked_df.empty:
                _df_download_button(
                    result.missing_ranked_df,
                    label="Download missing_skills_ranked.csv",
                    filename="missing_skills_ranked.csv",
                )
        with c2:
            st.download_button(
                label="Download skill_gap.json",
                data=json.dumps(result.gap_json, indent=2).encode("utf-8"),
                file_name="skill_gap.json",
                mime="application/json",
            )
            st.caption("Tip: attach these files in your GitHub README as sample outputs.")

    with tab_forecast:
        st.subheader("Future demand prediction (experimental)")
        st.caption(
            "This forecasts skill demand using a simple linear trend per skill over monthly history. "
            "It’s explainable, but not guaranteed—treat it as guidance."
        )

        if result.skill_time_series is None:
            reason = None
            try:
                reason = result.gap_json.get("notes", {}).get("forecast_unavailable_reason")
            except Exception:
                reason = None

            if reason:
                st.warning(f"Forecasting unavailable: {reason}")
            else:
                st.info("Select a valid **Date column** in the sidebar to enable forecasting.")
            return

        cA, cB, cC = st.columns([1, 1, 1])
        with cA:
            horizon_years = st.slider("Forecast horizon (years)", min_value=1, max_value=5, value=2, step=1)
        with cB:
            top_k = st.slider("Skills to forecast (top-K)", min_value=5, max_value=25, value=10, step=1)
        with cC:
            min_points = st.slider("Minimum history points", min_value=3, max_value=24, value=6, step=1)

        horizon_months = int(horizon_years) * 12
        try:
            df_forecast = forecast_skill_demand_linear(
                result.skill_time_series,
                horizon_months=horizon_months,
                top_k=int(top_k),
                min_points=int(min_points),
            )
        except Exception as e:
            st.exception(e)
            return

        if df_forecast.empty:
            st.warning("Not enough dated history to forecast. Try lowering ‘Minimum history points’ or choose another date column.")
            return

        st.dataframe(df_forecast, use_container_width=True, height=420)
        _df_download_button(df_forecast, label="Download forecast.csv", filename="forecast.csv")

        figF = px.line(
            df_forecast,
            x="period",
            y="percent",
            color="skill",
            line_dash="kind",
            title="Skill demand forecast (percent of job posts)",
            template="plotly_white",
        )
        st.plotly_chart(figF, use_container_width=True)


if __name__ == "__main__":
    main()

