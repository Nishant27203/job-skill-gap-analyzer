from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import spacy
from sklearn.linear_model import LinearRegression


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\+\#\.\s\-\/]")


def normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def load_taxonomy(path: Path) -> Dict[str, List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("skills taxonomy must be a JSON object")
    out: Dict[str, List[str]] = {}
    for canonical, synonyms in data.items():
        if isinstance(synonyms, str):
            synonyms = [synonyms]
        if not isinstance(synonyms, list) or not all(isinstance(s, str) for s in synonyms):
            raise ValueError(f"Invalid synonyms for '{canonical}'")
        out[str(canonical).strip().lower()] = [str(s).strip().lower() for s in synonyms]
    return out


def build_phrase_index(taxonomy: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Map normalized synonym phrase -> canonical skill.
    If duplicates exist, the first canonical encountered wins.
    """
    index: Dict[str, str] = {}
    for canonical, synonyms in taxonomy.items():
        for s in [canonical, *synonyms]:
            key = normalize(s)
            if key and key not in index:
                index[key] = canonical
    return index


def get_tokenizer() -> "spacy.language.Language":
    """
    Deployment-friendly tokenizer.
    We don't *need* a full spaCy model; blank English is enough for token boundaries.
    """
    return spacy.blank("en")


@dataclass(frozen=True)
class ExtractionResult:
    skills: Set[str]
    matched_phrases: Set[str]


def extract_skills(text: str, *, nlp, phrase_index: Dict[str, str], max_ngram: int = 4) -> ExtractionResult:
    norm = normalize(text)
    if not norm:
        return ExtractionResult(skills=set(), matched_phrases=set())

    doc = nlp(norm)
    tokens = [t.text for t in doc if not t.is_space]

    found: Set[str] = set()
    matched: Set[str] = set()
    for n in range(1, max_ngram + 1):
        for i in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            canonical = phrase_index.get(phrase)
            if canonical:
                found.add(canonical)
                matched.add(phrase)

    return ExtractionResult(skills=found, matched_phrases=matched)


def parse_candidate(candidate: str) -> Set[str]:
    raw = re.split(r"[,;\n]", candidate or "")
    return {s.strip().lower() for s in raw if s.strip()}


def load_candidate_file(path: Path) -> Set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return parse_candidate(text)


def canonicalize_candidate(candidate_skills: Set[str], *, phrase_index: Dict[str, str]) -> Set[str]:
    out: Set[str] = set()
    for s in candidate_skills:
        norm = normalize(s)
        if norm in phrase_index:
            out.add(phrase_index[norm])
        else:
            out.add(norm)
    return {s for s in out if s}


def filter_jobs(df: pd.DataFrame, *, title_column: Optional[str], role_filter: str) -> pd.DataFrame:
    if not role_filter.strip():
        return df
    if not title_column or title_column not in df.columns:
        # Be permissive: if the dataset doesn't have a title column, ignore role_filter.
        return df
    mask = df[title_column].fillna("").astype(str).str.contains(role_filter, case=False, regex=False)
    return df.loc[mask].copy()


def compute_skill_counts(skills_per_job: Sequence[Set[str]]) -> pd.DataFrame:
    counts: Dict[str, int] = {}
    total = len(skills_per_job) if skills_per_job else 0
    for skills in skills_per_job:
        for s in skills:
            counts[s] = counts.get(s, 0) + 1

    rows = []
    for skill, c in counts.items():
        pct = (c / total * 100.0) if total else 0.0
        rows.append((skill, c, pct))

    df = pd.DataFrame(rows, columns=["skill", "count", "percent"])
    df = df.sort_values(["count", "skill"], ascending=[False, True]).reset_index(drop=True)
    return df


@dataclass(frozen=True)
class AnalysisInputs:
    text_column: str
    title_column: Optional[str]
    role_filter: str
    candidate_text: str
    candidate_file: Optional[Path]
    top_n: int
    min_required_percent: float
    min_required_count: int
    date_column: Optional[str] = None


@dataclass(frozen=True)
class AnalysisOutputs:
    df_used: pd.DataFrame
    counts_all: pd.DataFrame
    top_skills: pd.DataFrame
    skills_per_job: List[Set[str]]
    required_skills: Set[str]
    candidate_skills: Set[str]
    missing_skills: List[str]
    required_df: pd.DataFrame
    comparison_df: pd.DataFrame
    missing_ranked_df: pd.DataFrame
    gap_json: Dict[str, object]
    skill_time_series: Optional[pd.DataFrame] = None


def analyze_jobs_dataframe(
    df: pd.DataFrame,
    *,
    taxonomy_path: Path,
    inputs: AnalysisInputs,
    nlp=None,
) -> AnalysisOutputs:
    if inputs.text_column not in df.columns:
        raise ValueError(f"Column '{inputs.text_column}' not found. Available: {list(df.columns)}")

    df_used = filter_jobs(df, title_column=inputs.title_column, role_filter=inputs.role_filter)
    if df_used.empty:
        raise ValueError("No rows matched your role_filter.")

    taxonomy = load_taxonomy(taxonomy_path)
    phrase_index = build_phrase_index(taxonomy)

    if nlp is None:
        nlp = get_tokenizer()

    skills_per_job: List[Set[str]] = []
    for text in df_used[inputs.text_column].fillna("").astype(str).tolist():
        res = extract_skills(text, nlp=nlp, phrase_index=phrase_index)
        skills_per_job.append(res.skills)

    counts_all = compute_skill_counts(skills_per_job)
    top = counts_all.head(int(inputs.top_n)).copy()

    required_df = counts_all.loc[
        (counts_all["percent"] >= float(inputs.min_required_percent))
        & (counts_all["count"] >= int(inputs.min_required_count))
    ].copy()
    required = set(required_df["skill"].tolist())

    candidate_raw: Set[str] = set()
    if inputs.candidate_file is not None:
        candidate_raw |= load_candidate_file(inputs.candidate_file)
    if inputs.candidate_text:
        candidate_raw |= parse_candidate(inputs.candidate_text)

    candidate_canon = canonicalize_candidate(candidate_raw, phrase_index=phrase_index)
    missing_set = required - candidate_canon
    missing = sorted(missing_set)

    df_cmp = required_df.copy()
    if not df_cmp.empty:
        df_cmp["candidate_has"] = df_cmp["skill"].isin(candidate_canon)
        df_cmp["status"] = df_cmp["candidate_has"].map(lambda x: "have" if x else "missing")
        df_cmp = df_cmp.sort_values(["candidate_has", "percent", "skill"], ascending=[True, False, True])
        df_missing_ranked = df_cmp.loc[~df_cmp["candidate_has"]].copy()
    else:
        df_missing_ranked = df_cmp.copy()

    gap = {
        "required_skills": sorted(required),
        "candidate_skills": sorted(candidate_canon),
        "missing_skills": missing,
        "notes": {
            "required_skills_definition": (
                "Skills extracted from the (optionally filtered) dataset meeting thresholds: "
                f"percent >= {inputs.min_required_percent} and count >= {inputs.min_required_count}."
            ),
            "rows_analyzed": int(len(df_used)),
            "role_filter": inputs.role_filter or None,
        },
    }

    ts: Optional[pd.DataFrame] = None
    if inputs.date_column:
        if inputs.date_column not in df_used.columns:
            ts = None
            gap["notes"]["date_column"] = inputs.date_column
            gap["notes"]["forecast_unavailable_reason"] = (
                f"Date column '{inputs.date_column}' was not found in the dataset."
            )
        else:
            try:
                ts = compute_skill_time_series(
                    df_used=df_used,
                    skills_per_job=skills_per_job,
                    date_column=inputs.date_column,
                    freq="M",
                )
            except ValueError as e:
                # Don't fail the whole analysis if forecasting can't run.
                ts = None
                gap["notes"]["date_column"] = inputs.date_column
                gap["notes"]["forecast_unavailable_reason"] = str(e)

    return AnalysisOutputs(
        df_used=df_used,
        counts_all=counts_all,
        top_skills=top,
        skills_per_job=skills_per_job,
        required_skills=required,
        candidate_skills=candidate_canon,
        missing_skills=missing,
        required_df=required_df,
        comparison_df=df_cmp,
        missing_ranked_df=df_missing_ranked,
        gap_json=gap,
        skill_time_series=ts,
    )


def compute_skill_time_series(
    *,
    df_used: pd.DataFrame,
    skills_per_job: Sequence[Set[str]],
    date_column: str,
    freq: str = "M",
) -> pd.DataFrame:
    """
    Build a period-based time series of skill demand.

    Returns columns: period, skill, count, percent, total_posts
    Where percent is (count / total_posts_in_period) * 100
    """
    if len(df_used) != len(skills_per_job):
        raise ValueError("df_used and skills_per_job length mismatch")

    raw = df_used[date_column]
    dates = pd.to_datetime(raw, errors="coerce")
    if dates.notna().sum() == 0:
        dates = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    tmp = pd.DataFrame({"_date": dates})
    tmp = tmp.dropna(subset=["_date"]).reset_index(drop=True)

    # Align skills to non-null dates
    skills_aligned: List[Set[str]] = []
    for idx, d in enumerate(dates.tolist()):
        if pd.isna(d):
            continue
        skills_aligned.append(set(skills_per_job[idx]))
    if not skills_aligned:
        raise ValueError("No valid dates found after parsing; cannot build time series.")

    tmp["_period"] = tmp["_date"].dt.to_period(freq).dt.to_timestamp()
    tmp["_skills"] = skills_aligned

    exploded = tmp.explode("_skills").dropna(subset=["_skills"])
    exploded = exploded.rename(columns={"_skills": "skill", "_period": "period"})

    total_posts = tmp.groupby("period").size().rename("total_posts").reset_index()
    counts = exploded.groupby(["period", "skill"]).size().rename("count").reset_index()
    out = counts.merge(total_posts, on="period", how="left")
    out["percent"] = out["count"] / out["total_posts"] * 100.0
    out = out.sort_values(["period", "count", "skill"], ascending=[True, False, True]).reset_index(drop=True)
    return out


def forecast_skill_demand_linear(
    ts: pd.DataFrame,
    *,
    horizon_months: int,
    top_k: int = 10,
    min_points: int = 6,
) -> pd.DataFrame:
    """
    Simple, explainable forecast:
    - For each skill, fit LinearRegression over time index -> percent
    - Predict next horizon_months

    Returns columns: period, skill, percent, kind in {"actual","forecast"}
    """
    if ts.empty:
        return pd.DataFrame(columns=["period", "skill", "percent", "kind"])

    ts = ts.copy()
    ts["period"] = pd.to_datetime(ts["period"])

    # Pick top skills by latest available period percent
    latest = ts["period"].max()
    latest_slice = ts.loc[ts["period"] == latest].sort_values("percent", ascending=False)
    skills = latest_slice["skill"].head(int(top_k)).tolist()
    if not skills:
        skills = ts.sort_values("percent", ascending=False)["skill"].drop_duplicates().head(int(top_k)).tolist()

    all_periods = sorted(ts["period"].dropna().unique().tolist())
    if not all_periods:
        return pd.DataFrame(columns=["period", "skill", "percent", "kind"])

    start = all_periods[0]
    def _t(p: pd.Timestamp) -> int:
        # months since start
        return (p.year - start.year) * 12 + (p.month - start.month)

    future_periods: List[pd.Timestamp] = []
    last = all_periods[-1]
    for i in range(1, int(horizon_months) + 1):
        future_periods.append((last + pd.offsets.MonthBegin(i)).normalize())

    rows: List[Tuple[pd.Timestamp, str, float, str]] = []
    for skill in skills:
        s = ts.loc[ts["skill"] == skill, ["period", "percent"]].dropna().sort_values("period")
        s = s.drop_duplicates(subset=["period"], keep="last")
        if len(s) < int(min_points):
            continue
        X = [[_t(p)] for p in s["period"].tolist()]
        y = s["percent"].astype(float).tolist()
        model = LinearRegression()
        model.fit(X, y)

        for p, v in zip(s["period"].tolist(), y):
            rows.append((p, skill, float(v), "actual"))

        for p in future_periods:
            pred = float(model.predict([[float(_t(p))]])[0])
            pred = max(0.0, min(100.0, pred))
            rows.append((p, skill, pred, "forecast"))

    out = pd.DataFrame(rows, columns=["period", "skill", "percent", "kind"])
    out = out.sort_values(["skill", "period", "kind"]).reset_index(drop=True)
    return out

