# Job Skill Gap Analyzer

Analyze job descriptions to extract in-demand skills, then compare them to a candidate’s skills to identify gaps.

## Live demo
- Streamlit app: `https://job-skill-gap-analyzer-acfaj3syywwhdfc2faf7u.streamlit.app/`

## What this repo includes
- `data/jobs_sample.csv`: tiny sample dataset you can run immediately
- `skill_taxonomy/skills.json`: curated skill dictionary (editable)
- `src/jobskill/analyze.py`: CLI to run extraction + gap analysis + charts
- `app.py`: Streamlit web app (upload CSV + interactive analysis)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (web app)
```bash
source .venv/bin/activate
streamlit run app.py
```

## Run (sample)
```bash
python -m src.jobskill.analyze \
  --jobs-csv data/jobs_sample.csv \
  --title-column title \
  --text-column description \
  --candidate "Python, SQL, Excel" \
  --top-n 15 \
  --out-dir outputs
```

## Outputs
- `outputs/top_skills.csv`: market demand table
- `outputs/skill_gap.json`: required vs candidate vs missing skills
- `outputs/top_skills.png`: bar chart
- `outputs/top_skills.html`: interactive chart
- `outputs/skill_comparison.csv`: required skills with candidate coverage
- `outputs/missing_skills_ranked.csv`: missing skills ranked by demand
- `outputs/skill_comparison.png`: required vs missing chart
- `outputs/skill_comparison.html`: interactive required vs missing chart

## Useful options
- Filter by role/title:

```bash
python -m src.jobskill.analyze --jobs-csv data/jobs_sample.csv --text-column description --title-column title --role-filter "Data Scientist"
```

- Define “required skills” by demand threshold (instead of simple union):

```bash
python -m src.jobskill.analyze --jobs-csv data/jobs_sample.csv --text-column description --title-column title --min-required-percent 40 --min-required-count 1
```

- Load candidate skills from a file:

```bash
python -m src.jobskill.analyze --jobs-csv data/jobs_sample.csv --text-column description --title-column title --candidate-file data/candidate_skills.txt
```

## Notes
- Skill extraction uses a lightweight approach: text normalization + spaCy tokenization + matching against `skill_taxonomy/skills.json`.
- To improve accuracy, expand the taxonomy (add synonyms) and/or plug in a dedicated NER / skill extraction model later.

