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

## Hotspot variants

- Each region (Crown, Rim, Sulcus) can now point to multiple hotspot-specific CSV bundles. Use the **Dataset Selection** controls in the sidebar to pick both the region and hotspot variant you want to visualize.
- When focusing on a single region, toggle **Aggregate all hotspots** to merge every hotspot variant into one combined view—plots, summaries, and meta analysis will all use the aggregated data.
- When "Compare all regions" is enabled, you can specify the hotspot variant for each region individually and, inside the **Meta Analysis** tab, choose whether to average across all hotspots or restrict the aggregation to a subset per region.
