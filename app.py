import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union, cast


DATA_ROOT = Path(__file__).resolve().parent

RealizationType = Union[str, float, int]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_value = hex_color.lstrip("#")
    if len(hex_value) != 6:
        return hex_color
    r = int(hex_value[0:2], 16)
    g = int(hex_value[2:4], 16)
    b = int(hex_value[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def plot_score_results(
    df: pd.DataFrame,
    length: int,
    error_bars: bool = False,
    realization_id: Optional[RealizationType] = None,
    title_prefix: Optional[str] = None,
    fig: Optional[go.Figure] = None,
    row: int = 1,
    col: int = 1,
    legend_seen: Optional[Set[str]] = None,
    region_suffix: str = "",
    region_color_offset: int = 0,
) -> bool:
    if fig is None:
        raise ValueError("Plotly figure is required for plotting")
    plot_df = df.copy()
    if realization_id is not None:
        plot_df = plot_df[plot_df["realization_id"] == realization_id]

    if plot_df.empty:
        return False

    agg_df = (
        plot_df.groupby(["experiment_name", "sequence_length"], as_index=False)
        .agg(mean_score=("score", "mean"), sem_score=("score", "sem"))
    )
    agg_df = agg_df.sort_values(["experiment_name", "sequence_length"])
    agg_df = agg_df[agg_df["sequence_length"] <= length]

    if agg_df.empty:
        return False

    # Color palettes for region comparison
    color_palettes = {
        0: {  # Default palette
            "RANDOM": "#000000",
            "ROW": "#1f77b4",
            "COL": "#ff7f0e",
        },
        1: {  # Crown palette (cooler tones)
            "RANDOM": "#2c3e50",
            "ROW": "#3498db",
            "COL": "#1abc9c",
        },
        2: {  # Rim palette (warmer tones)
            "RANDOM": "#34495e",
            "ROW": "#e74c3c",
            "COL": "#f39c12",
        },
    }
    
    color_map = color_palettes.get(region_color_offset, color_palettes[0])
    linestyle_map = {
        "CO": "-",
        "ED": ":",
    }
    marker_map = {
        "mean": None,
        "extreme": "star",
    }

    def parse_label(label: str) -> Tuple[str, str, str]:
        upper = label.upper()
        if "RAND" in upper:
            base = "RANDOM"
        elif "ROW" in upper:
            base = "ROW"
        elif "COL" in upper:
            base = "COL"
        else:
            base = "OTHER"

        if "CORR" in upper:
            metric = "CO"
        elif "ED" in upper:
            metric = "ED"
        else:
            metric = "UNK"

        if "MEAN" in upper:
            agg = "mean"
        elif "MAX" in upper or "MIN" in upper:
            agg = "extreme"
        else:
            agg = "mean"

        return base, metric, agg

    for exp_name, subdf in agg_df.groupby("experiment_name"):
        base, metric, agg = parse_label(str(exp_name))
        color = color_map.get(base, "gray")
        linestyle = linestyle_map.get(metric, "dashdot")
        marker_symbol = marker_map.get(agg)
        linewidth = 1 if base == "RANDOM" else 1.8
        alpha = 1.0 if base == "RANDOM" else 0.85
        mode = "lines+markers" if marker_symbol else "lines"
        dash_style = {
            "-": "solid",
            ":": "dot",
        }.get(linestyle, "dashdot")
        showlegend = legend_seen is None or str(exp_name) not in legend_seen
        marker_size = 7 if marker_symbol else 0
        marker_line_width = 1.2 if marker_symbol else 0

        realization_context = "Mean across realizations" if realization_id is None else f"Realization {realization_id}"
        hover_details = realization_context
        
        # Add region suffix to legend name for comparison mode
        display_name = f"{exp_name}{region_suffix}" if region_suffix else str(exp_name)
        legend_key = f"{exp_name}{region_suffix}"
        
        showlegend = legend_seen is None or legend_key not in legend_seen
        
        fig.add_trace(
            go.Scatter(
                x=subdf["sequence_length"],
                y=subdf["mean_score"],
                name=display_name,
                legendgroup=legend_key,
                mode=mode,
                line=dict(color=color, width=linewidth, dash=dash_style),
                marker=dict(
                    symbol=marker_symbol or "circle",
                    size=marker_size,
                    color=color,
                    line=dict(color="#ffffff", width=marker_line_width),
                ),
                opacity=alpha,
                hovertemplate=(
                    "<b>%{text}</b><br>%{customdata}<br>Steps: %{x}<br>Mean score: %{y:.4f}<extra></extra>"
                ),
                text=[display_name] * len(subdf),
                customdata=[hover_details] * len(subdf),
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )
        if legend_seen is not None and showlegend:
            legend_seen.add(legend_key)

        if error_bars and "sem_score" in subdf.columns and not subdf["sem_score"].isna().all():
            lower = subdf["mean_score"] - 1.96 * subdf["sem_score"]
            upper = subdf["mean_score"] + 1.96 * subdf["sem_score"]
            fig.add_trace(
                go.Scatter(
                    x=list(subdf["sequence_length"]) + list(subdf["sequence_length"][::-1]),
                    y=list(upper) + list(lower[::-1]),
                    fill="toself",
                    fillcolor=_hex_to_rgba(color, 0.3 if base != "RANDOM" else 0.4),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    legendgroup=legend_key,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    fig.update_xaxes(title_text="Steps", row=row, col=col)
    fig.update_yaxes(title_text="Mean congruence score", row=row, col=col)

    return True


def _format_realization_label(rid: RealizationType) -> str:
    try:
        rid_float = float(rid)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        rid_display = rid
    else:
        rid_display = int(rid_float) if rid_float.is_integer() else rid_float
    return f"ID {rid_display}"


def _collect_metadata(df_list: Iterable[pd.DataFrame]) -> Tuple[List[str], List[RealizationType], int, int]:
    experiments: List[str] = []
    realizations: List[RealizationType] = []
    seq_min: Optional[int] = None
    seq_max: Optional[int] = None

    for df in df_list:
        experiments.extend(df["experiment_name"].dropna().unique().tolist())
        realizations.extend(df["realization_id"].dropna().unique().tolist())
        current_min = df["sequence_length"].min()
        current_max = df["sequence_length"].max()
        seq_min = current_min if seq_min is None else min(seq_min, current_min)
        seq_max = current_max if seq_max is None else max(seq_max, current_max)

    experiments = sorted(set(experiments))
    realizations = sorted(set(realizations), key=lambda value: str(value))

    seq_min = int(seq_min) if seq_min is not None else 1
    seq_max = int(seq_max) if seq_max is not None else 100

    return experiments, realizations, seq_min, seq_max


def _dataset_config() -> Dict[str, List[Tuple[str, str]]]:
    return {
        "crown": [
            ("GD", "exp_c_map_score_metrics_GD_sigmoid4log_crown.csv"),
            ("NRMSD", "exp_c_map_score_metrics_NRMSD_sigmoid4log_crown.csv"),
        ],
        "rim": [
            ("GD", "exp_c_map_score_metrics_GD_sigmoid4log_rim.csv"),
            ("NRMSD", "exp_c_map_score_metrics_NRMSD_sigmoid4log_rim.csv"),
        ],
    }


@st.cache_data(show_spinner=False)
def load_dataset(dataset_key: str) -> Dict[str, object]:
    collections = _dataset_config()
    if dataset_key not in collections:
        raise KeyError(f"Unknown dataset key: {dataset_key}")

    frames: Dict[str, pd.DataFrame] = {}
    for metric_label, filename in collections[dataset_key]:
        csv_path = DATA_ROOT / filename
        frames[metric_label] = pd.read_csv(csv_path)

    experiments, realizations, seq_min, seq_max = _collect_metadata(frames.values())
    bundle: Dict[str, object] = {metric: df for metric, df in frames.items()}
    bundle["all_experiments"] = experiments
    bundle["all_realizations"] = realizations
    bundle["seq_min"] = seq_min
    bundle["seq_max"] = seq_max
    return bundle


def render_page() -> None:
    st.set_page_config(page_title="Congruence Score Explorer", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f8f9fa;
            color: #2c3e50;
        }
        div[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #dee2e6;
        }
        div[data-testid="stSidebar"] * {
            color: #2c3e50 !important;
        }
        .stSlider > div[data-baseweb="slider"] > div:nth-child(2) {
            background-color: #e9ecef;
        }
        .stSlider > div[data-baseweb="slider"] div[role="slider"] {
            background-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .stButton button {
            border-radius: 6px;
            font-size: 0.85rem;
            padding: 0.35rem 0.75rem;
            transition: all 0.2s ease;
        }
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        div[data-testid="stExpander"] {
            border-radius: 8px;
            border: 1px solid #dee2e6;
            margin-bottom: 0.5rem;
        }
        div[data-testid="stExpander"] summary {
            font-weight: 600;
            padding: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Congruence Score Explorer")

    dataset_labels: List[Tuple[str, str]] = [
        ("Crown", "crown"),
        ("Rim", "rim"),
    ]

    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Dataset selection with expander for cleaner look
        with st.expander("📊 Dataset Selection", expanded=True):
            # Add comparison mode toggle
            compare_regions = st.checkbox(
                "🔄 Compare Crown vs Rim",
                value=False,
                help="Enable this to overlay Crown and Rim data for the same experiments"
            )
            
            if not compare_regions:
                dataset_display = st.selectbox(
                    "Region",
                    options=[label for label, _ in dataset_labels],
                    index=0,
                    help="Select the brain region to analyze"
                )
                dataset_map = {label: key for label, key in dataset_labels}
                dataset_key = dataset_map[dataset_display]
                data_bundle = load_dataset(dataset_key)
                datasets_to_plot = [(dataset_key, data_bundle)]
            else:
                st.info("📊 Comparing both Crown and Rim regions")
                # Load both datasets
                crown_bundle = load_dataset("crown")
                rim_bundle = load_dataset("rim")
                datasets_to_plot = [("crown", crown_bundle), ("rim", rim_bundle)]
                # Use first dataset for experiment list
                data_bundle = crown_bundle
                dataset_key = "crown_vs_rim"

            if "all_experiments" not in data_bundle:
                st.warning("No experiments found in the selected dataset.")
                return

            experiments = cast(List[str], data_bundle["all_experiments"])
            if not experiments:
                st.warning("No experiments found in the selected dataset.")
                return

        # Experiment filtering with quick actions
        with st.expander("🧪 Experiment Filters", expanded=True):
            # Initialize or sanitize session state for current experiment list
            if "experiment_selection" not in st.session_state:
                st.session_state["experiment_selection"] = experiments.copy()
            else:
                current_selection = [exp for exp in st.session_state["experiment_selection"] if exp in experiments]
                if not current_selection:
                    current_selection = experiments.copy()
                st.session_state["experiment_selection"] = current_selection

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✓ Select All", use_container_width=True, key="select_all_btn"):
                    st.session_state["experiment_selection"] = experiments.copy()
            with col2:
                if st.button("✗ Clear All", use_container_width=True, key="clear_all_btn"):
                    st.session_state["experiment_selection"] = []

            experiment_selection = st.multiselect(
                "Experiments",
                options=experiments,
                default=st.session_state["experiment_selection"],
                key="experiment_selection",
                help=f"Select from {len(experiments)} available experiments"
            )

            if not experiment_selection:
                st.info("👆 Select at least one experiment to display")
            else:
                st.success(f"✓ {len(experiment_selection)} experiments selected")

        # Data range controls
        with st.expander("📏 Data Range", expanded=True):
            seq_min = int(cast(int, data_bundle.get("seq_min", 1)))
            seq_max = int(cast(int, data_bundle.get("seq_max", 100)))
            
            col1, col2 = st.columns([3, 1])
            with col1:
                length = st.slider(
                    "Max steps",
                    min_value=seq_min,
                    max_value=seq_max,
                    value=seq_max,
                    step=1,
                    help="Adjust the maximum sequence length to display"
                )
            with col2:
                st.metric("Range", f"{seq_min}-{length}", delta=None)

        # Realization selection
        with st.expander("🔄 Realization", expanded=True):
            realization_labels = ["Average across realizations"]
            realization_lookup: Dict[str, Optional[RealizationType]] = {"Average across realizations": None}
            for rid in cast(List[RealizationType], data_bundle.get("all_realizations", [])):
                label = _format_realization_label(rid)
                realization_labels.append(label)
                realization_lookup[label] = rid
            realization_label = st.selectbox(
                "Select realization",
                options=realization_labels,
                help="View individual realizations or averaged results"
            )
            realization_value = realization_lookup[realization_label]

        # Visualization options
        with st.expander("⚙️ Display Options", expanded=True):
            error_bars = st.checkbox(
                "Show confidence intervals (95%)",
                value=False,
                help="Display 95% confidence intervals around mean scores"
            )
            
        # Summary statistics at bottom
        st.divider()
        st.caption("📈 **Quick Stats**")
        if experiment_selection:
            st.caption(f"• Experiments: {len(experiment_selection)}")
            st.caption(f"• Steps range: {seq_min} → {length}")
            all_realizations = data_bundle.get('all_realizations', [])
            if isinstance(all_realizations, list):
                st.caption(f"• Realizations: {len(all_realizations)}")

    metrics: Iterable[str] = ["GD", "NRMSD"]
    
    if not experiment_selection:
        st.warning("No data available for the current selection.")
        return
    
    # Collect payload based on comparison mode
    payload: List[Tuple[str, List[Tuple[str, pd.DataFrame, int]]]] = []
    
    if compare_regions:
        # Comparison mode: combine data from both regions
        for metric_label in metrics:
            region_data: List[Tuple[str, pd.DataFrame, int]] = []
            for idx, (region_key, region_bundle) in enumerate(datasets_to_plot, start=1):
                metric_df = region_bundle.get(metric_label)
                if isinstance(metric_df, pd.DataFrame):
                    filtered_df = metric_df[metric_df["experiment_name"].isin(experiment_selection)]
                    if not filtered_df.empty:
                        region_data.append((region_key, filtered_df, idx))
            if region_data:
                payload.append((metric_label, region_data))
    else:
        # Single region mode
        for metric_label in metrics:
            metric_df = data_bundle.get(metric_label)
            if isinstance(metric_df, pd.DataFrame):
                filtered_df = metric_df[metric_df["experiment_name"].isin(experiment_selection)]
                if not filtered_df.empty:
                    payload.append((metric_label, [(dataset_key, filtered_df, 0)]))

    if not payload:
        st.warning("No data available for the current selection.")
        return

    rows = len(payload)
    subplot_titles = [f"{metric_label} · {realization_label}" for metric_label, _ in payload]
    plotly_fig = make_subplots(
        rows=rows,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.18,
    )

    empty_metrics: List[str] = []
    legend_seen: Set[str] = set()
    
    for idx, (metric_label, region_data_list) in enumerate(payload, start=1):
        for region_key, filtered_df, color_offset in region_data_list:
            region_suffix = ""
            if compare_regions:
                region_suffix = f" ({region_key.capitalize()})"
            
            rendered = plot_score_results(
                filtered_df,
                length=length,
                error_bars=error_bars,
                realization_id=realization_value,
                title_prefix=metric_label,
                fig=plotly_fig,
                row=idx,
                col=1,
                legend_seen=legend_seen,
                region_suffix=region_suffix,
                region_color_offset=color_offset,
            )
            if not rendered:
                empty_metrics.append(f"{metric_label} ({region_key})")

    for idx in range(1, rows + 1):
        plotly_fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(0, 0, 0, 0.08)",
            zeroline=False,
            linecolor="rgba(0, 0, 0, 0.2)",
            mirror=True,
            title_font=dict(color="#2c3e50", size=12),
            tickfont=dict(color="#495057", size=11),
            row=idx,
            col=1,
        )
        plotly_fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(0, 0, 0, 0.06)",
            zeroline=False,
            linecolor="rgba(0, 0, 0, 0.2)",
            mirror=True,
            title_font=dict(color="#2c3e50", size=12),
            tickfont=dict(color="#495057", size=11),
            row=idx,
            col=1,
        )

    base_height = 420
    plot_height = max(440, rows * base_height)

    plotly_fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="rgba(0, 0, 0, 0.15)",
            borderwidth=1,
            itemclick="toggleothers",
            font=dict(color="#2c3e50", size=10),
        ),
        margin=dict(l=70, r=40, t=130, b=70),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f8f9fa",
        font=dict(family="Segoe UI", size=13, color="#2c3e50"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#dee2e6", font=dict(color="#2c3e50", size=12)),
        height=plot_height,
    )
    plotly_fig.update_annotations(font=dict(size=15, color="#2c3e50"), yshift=10)

    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"congruence_{dataset_key}_{realization_label.replace(' ', '_')}",
            "height": plot_height,
            "width": 1400,
            "scale": 2,
        },
    }
    
    # Add tabs for different views
    tab1, tab2 = st.tabs(["📊 Visualization", "📋 Data Summary"])
    
    with tab1:
        st.plotly_chart(plotly_fig, use_container_width=True, config=config)
        
        if empty_metrics:
            st.info("No data within the selected length for: " + ", ".join(empty_metrics))
    
    with tab2:
        st.subheader("Data Summary")
        
        for metric_label, region_data_list in payload:
            with st.expander(f"📈 {metric_label} Statistics", expanded=True):
                # Calculate summary statistics
                for region_key, filtered_df, _ in region_data_list:
                    if compare_regions:
                        st.markdown(f"**{region_key.capitalize()}**")
                    
                    summary_data = []
                    for exp_name in experiment_selection:
                        exp_df = filtered_df[filtered_df["experiment_name"] == exp_name]
                        if not exp_df.empty:
                            exp_df_filtered = exp_df[exp_df["sequence_length"] <= length]
                            if not exp_df_filtered.empty:
                                summary_data.append({
                                    "Experiment": exp_name,
                                    "Mean Score": f"{exp_df_filtered['score'].mean():.4f}",
                                    "Std Dev": f"{exp_df_filtered['score'].std():.4f}",
                                    "Min": f"{exp_df_filtered['score'].min():.4f}",
                                    "Max": f"{exp_df_filtered['score'].max():.4f}",
                                    "Data Points": len(exp_df_filtered)
                                })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(
                            summary_df,
                            use_container_width=True,
                            hide_index=True,
                        )
                        
                        # Download button for CSV
                        csv = summary_df.to_csv(index=False).encode('utf-8')
                        file_suffix = f"_{region_key}" if compare_regions else ""
                        st.download_button(
                            label=f"📥 Download {metric_label}{file_suffix} Summary (CSV)",
                            data=csv,
                            file_name=f"{metric_label}_summary{file_suffix}_{dataset_key}.csv",
                            mime="text/csv",
                        )
                    
                    if compare_regions and region_key == "crown":
                        st.divider()

    # st.caption(
    #     "This dashboard mirrors the controls of the local Tkinter app so collaborators can explore results without a Python environment."
    # )


if __name__ == "__main__":
    render_page()
