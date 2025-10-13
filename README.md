# Streamlit Congruence Explorer

This app recreates the controls from `draft.py` in a Streamlit dashboard so collaborators can browse the local CSV outputs without installing Python locally.

## Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Streamlit:
   ```bash
   streamlit run app.py
   ```
3. When deploying to Streamlit Community Cloud, upload this `streamlit_app` folder and the four CSV files located one level above (`exp_c_map_score_metrics_GD.csv`, etc.).

The app loads the four experiment CSV files on start and exposes the same dataset, experiment, realization, length, and error bar controls found in the Tkinter UI.
