# Job Skill Gap Analyzer

An end-to-end **NLP + analytics** web app that extracts skills from job descriptions, measures what’s trending in the market, and compares it with a candidate’s skills to highlight the **skill gap**.

- **Live demo (Streamlit)**: `https://job-skill-gap-analyzer-bcnhhwxboav4mewn75jbzl.streamlit.app/`
- **Tech stack**: Python, Pandas, spaCy (tokenization), Plotly, Streamlit

## Features
- **Upload any job dataset CSV** (robust column auto-detection + safe fallbacks)
- **Skill extraction** using a curated taxonomy (`skill_taxonomy/skills.json`)
- **Market demand analytics**: top skills, counts, and % of postings
- **Candidate vs required skills**: missing skills + ranked recommendations
- **Forecasting (optional)**: predicts future demand when a valid date column exists
- **Export**: download CSV/JSON reports directly from the UI

## Project structure
- `app.py`: Streamlit dashboard (web app)
- `src/jobskill/core.py`: reusable analysis engine (used by web + CLI)
- `src/jobskill/analyze.py`: CLI runner (batch mode)
- `skill_taxonomy/skills.json`: skills + synonyms dictionary (editable)
- `data/jobs_sample.csv`: small sample dataset
- `runtime.txt`: pins Python version for Streamlit Cloud

## Quickstart (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## CLI usage (optional)

```bash
python -m src.jobskill.analyze \
  --jobs-csv data/jobs_sample.csv \
  --title-column title \
  --text-column description \
  --candidate "Python, SQL, Excel" \
  --top-n 15 \
  --out-dir outputs
```

## Outputs (CLI and/or UI downloads)
- `top_skills.csv`: market demand table
- `skill_gap.json`: required vs candidate vs missing skills
- `missing_skills_ranked.csv`: missing skills ranked by demand
- `skill_comparison.csv`: required skills with candidate coverage

## Deployment

### Streamlit Community Cloud (recommended)
- Create a new Streamlit app from this repo
- **Main file path**: `app.py`
- **Python**: 3.12 (already enforced via `runtime.txt`)

### Docker (optional)

```bash
docker build -t job-skill-gap-analyzer .
docker run -p 8501:8501 job-skill-gap-analyzer
```

## Customize the skill taxonomy
Edit `skill_taxonomy/skills.json` to add new skills and synonyms, for example:

```json
{
  "python": ["python", "py"],
  "power bi": ["power bi", "powerbi"]
}
```

## Notes & limitations
- Skill extraction is dictionary-based for **speed + explainability**. Accuracy improves as you expand the taxonomy.
- Forecasting is a simple trend model intended for **guidance**, not a guarantee of market outcomes.

## License
MIT. See `LICENSE`.

