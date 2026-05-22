# Software Effort Estimator

This repo contains a Streamlit app (`app.py`) that predicts software effort using a weighted-kernel SVR and SHAP explainability.

## Quick local run

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (recommended)

1. Push this repository to GitHub:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-git-repo-url>
git push -u origin main
```

2. Go to https://share.streamlit.io, sign in with GitHub, and create a new app by selecting this repository and the `app.py` file. Streamlit Cloud will install dependencies from `requirements.txt` and deploy the app.

## Deploy to Heroku (alternative)

1. Install the Heroku CLI and login:

```bash
heroku login
```

2. Create a Heroku app and push:

```bash
heroku create your-app-name
git push heroku main
```

3. Ensure the `Procfile` is present (included) so Heroku runs Streamlit correctly.

## Notes

- Ensure `model.pkl` is committed (or use Git LFS if large).
- If deployment fails due to binary packages (e.g., `shap`), consider building a smaller runtime or using Docker.
