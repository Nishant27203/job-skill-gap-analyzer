## Contributing

Thanks for your interest in improving **Job Skill Gap Analyzer**.

### Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Guidelines
- Keep changes focused and easy to review.
- Prefer adding skills/synonyms in `skill_taxonomy/skills.json` over hardcoding.
- Ensure the app still works with “unknown” datasets (different column names/formats).

### Pull requests
- Include a short summary of your change and how to test it.
- If you add a new feature, update `README.md` with usage notes.

