from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import typer
from rich.console import Console
from rich.table import Table

from .core import AnalysisInputs, analyze_jobs_dataframe


app = typer.Typer(add_completion=False)
console = Console()

def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _save_charts(df_top: pd.DataFrame, out_dir: Path) -> None:
    if df_top.empty:
        return

    # Matplotlib PNG
    plt.figure(figsize=(10, 6))
    plt.barh(df_top["skill"][::-1], df_top["percent"][::-1])
    plt.xlabel("Percent of job posts (%)")
    plt.title("Top skills in job dataset")
    plt.tight_layout()
    (out_dir / "top_skills.png").write_bytes(_fig_to_png_bytes())
    plt.close()

    # Plotly HTML
    fig = px.bar(df_top, x="percent", y="skill", orientation="h", title="Top skills in job dataset")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.write_html(str(out_dir / "top_skills.html"), include_plotlyjs="cdn")


def _fig_to_png_bytes() -> bytes:
    import io

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160)
    buf.seek(0)
    return buf.read()

def _save_skill_comparison_chart(df_cmp: pd.DataFrame, out_dir: Path) -> None:
    """
    Bar chart: required skill demand %, colored by whether candidate has it.
    """
    if df_cmp.empty:
        return

    df = df_cmp.sort_values(["percent", "skill"], ascending=[True, True]).tail(30)
    colors = ["#2E7D32" if bool(v) else "#C62828" for v in df["candidate_has"]]

    plt.figure(figsize=(10, 7))
    plt.barh(df["skill"], df["percent"], color=colors)
    plt.xlabel("Percent of job posts (%)")
    plt.title("Required skills vs candidate coverage (top 30 by demand)")
    plt.tight_layout()
    (out_dir / "skill_comparison.png").write_bytes(_fig_to_png_bytes())
    plt.close()

    fig = px.bar(
        df,
        x="percent",
        y="skill",
        orientation="h",
        color="candidate_has",
        title="Required skills vs candidate coverage (top 30 by demand)",
        labels={"candidate_has": "Candidate has"},
        color_discrete_map={True: "#2E7D32", False: "#C62828"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.write_html(str(out_dir / "skill_comparison.html"), include_plotlyjs="cdn")


@app.command()
def main(
    jobs_csv: Path = typer.Option(..., exists=True, readable=True, help="CSV with job descriptions"),
    text_column: str = typer.Option("description", help="Column containing job description text"),
    title_column: Optional[str] = typer.Option(
        None, help="Optional job title column (used with --role-filter), e.g. 'title'"
    ),
    role_filter: str = typer.Option(
        "", help="If set, only analyze rows whose title contains this text (case-insensitive)"
    ),
    taxonomy_json: Path = typer.Option(
        Path("skill_taxonomy/skills.json"), exists=True, readable=True, help="Skill taxonomy JSON path"
    ),
    candidate: str = typer.Option("", help='Candidate skills, e.g. "Python, SQL, Excel"'),
    candidate_file: Optional[Path] = typer.Option(
        None, exists=True, readable=True, help="Optional file containing candidate skills (txt, comma/newline separated)"
    ),
    top_n: int = typer.Option(20, min=1, max=200, help="Top-N skills to report"),
    min_required_percent: float = typer.Option(
        0.0,
        min=0.0,
        max=100.0,
        help="Define required skills as those appearing in >= this percent of filtered job posts",
    ),
    min_required_count: int = typer.Option(
        1, min=1, help="Define required skills as those appearing in >= this many filtered job posts"
    ),
    out_dir: Path = typer.Option(Path("outputs"), help="Output directory"),
):
    """
    Extract skills from job descriptions, compute market demand, and compare to candidate skills.
    """
    console.print(f"Loading dataset from [bold]{jobs_csv}[/bold]")
    df = pd.read_csv(jobs_csv)
    try:
        result = analyze_jobs_dataframe(
            df,
            taxonomy_path=taxonomy_json,
            inputs=AnalysisInputs(
                text_column=text_column,
                title_column=title_column,
                role_filter=role_filter,
                date_column=None,
                candidate_text=candidate,
                candidate_file=candidate_file,
                top_n=top_n,
                min_required_percent=min_required_percent,
                min_required_count=min_required_count,
            ),
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    _ensure_out_dir(out_dir)
    result.top_skills.to_csv(out_dir / "top_skills.csv", index=False)
    _save_charts(result.top_skills, out_dir)
    (out_dir / "skill_gap.json").write_text(json.dumps(result.gap_json, indent=2), encoding="utf-8")

    if not result.comparison_df.empty:
        result.comparison_df.to_csv(out_dir / "skill_comparison.csv", index=False)
        result.missing_ranked_df.to_csv(out_dir / "missing_skills_ranked.csv", index=False)
        _save_skill_comparison_chart(result.comparison_df, out_dir)

    # Pretty console output
    title = "Top Skills (Market Demand)"
    if role_filter.strip():
        title += f" — filtered by title contains '{role_filter}'"
    table = Table(title=title)
    table.add_column("Skill", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Percent", justify="right")
    for _, row in result.top_skills.iterrows():
        table.add_row(str(row["skill"]), str(int(row["count"])), f'{float(row["percent"]):.1f}%')
    console.print(table)

    if candidate or candidate_file is not None:
        console.print()
        console.print("[bold]Skill Gap[/bold]")
        console.print(f"Candidate skills (canonicalized): {', '.join(sorted(result.candidate_skills)) or '(none)'}")
        console.print(f"Missing skills: {', '.join(result.missing_skills) or '(none)'}")

    console.print()
    console.print(f"Wrote outputs to [bold]{out_dir}[/bold]")


if __name__ == "__main__":
    app()

