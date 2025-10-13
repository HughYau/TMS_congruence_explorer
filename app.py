import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union, cast


DATA_ROOT = Path(__file__).resolve().parent

RealizationType = Union[str, float, int]


def plot_score_results(
    df: pd.DataFrame,
    length: int,
    error_bars: bool = False,
    realization_id: Optional[RealizationType] = None,
    title_prefix: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Tuple[List, List]:
    if ax is None:
        raise ValueError("Axes instance is required for plotting")
    plot_df = df.copy()
    if realization_id is not None:
        plot_df = plot_df[plot_df["realization_id"] == realization_id]

    if plot_df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return [], []

    agg_df = (
        plot_df.groupby(["experiment_name", "sequence_length"], as_index=False)
        .agg(mean_score=("score", "mean"), std_score=("score", "std"))
    )
    agg_df = agg_df.sort_values(["experiment_name", "sequence_length"])
    agg_df = agg_df[agg_df["sequence_length"] <= length]

    if agg_df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data within length", ha="center", va="center")
        return [], []

    color_map = {
        "RANDOM": "#000000",
        "ROW": "#1f77b4",
        "COL": "#ff7f0e",
    }
    linestyle_map = {
        "CO": "-",
        "ED": ":",
    }
    marker_map = {
        "mean": "",
        "extreme": "*",
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

    legend_handles: List = []
    legend_labels: List[str] = []

    for exp_name, subdf in agg_df.groupby("experiment_name"):
        base, metric, agg = parse_label(str(exp_name))
        color = color_map.get(base, "gray")
        linestyle = linestyle_map.get(metric, "-.")
        marker = marker_map.get(agg, "")
        linewidth = 1 if base == "RANDOM" else 1.8
        alpha = 1.0 if base == "RANDOM" else 0.7
        zorder = 6 if base == "RANDOM" else 3

        (line,) = ax.plot(
            subdf["sequence_length"],
            subdf["mean_score"],
            label=exp_name,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
        legend_handles.append(line)
        legend_labels.append(str(exp_name))

        if error_bars and "std_score" in subdf.columns and not subdf["std_score"].isna().all():
            ax.fill_between(
                subdf["sequence_length"],
                subdf["mean_score"] - subdf["std_score"],
                subdf["mean_score"] + subdf["std_score"],
                color=color,
                alpha=0.15,
                linewidth=0,
            )

    suffix = " (mean across realizations)" if realization_id is None else f" (realization {realization_id})"
    prefix = f"{title_prefix} - " if title_prefix else ""
    ax.set_xlabel("Sequence length", fontsize=12)
    ax.set_ylabel("Mean congruence score", fontsize=12)
    ax.set_title(f"{prefix}Congruence evaluation{suffix}", fontsize=13, pad=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return legend_handles, legend_labels


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
        "mutual_information": [
            ("GD", "exp_c_map_score_metrics_GD.csv"),
            ("NRMSD", "exp_c_map_score_metrics_NRMSD.csv"),
        ],
        "sigmoid4log": [
            ("GD", "exp_c_map_score_metrics_GD_sigmoid4log.csv"),
            ("NRMSD", "exp_c_map_score_metrics_NRMSD_sigmoid4log.csv"),
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
    st.title("Congruence Score Explorer")

    dataset_labels: List[Tuple[str, str]] = [
        ("Mutual Information", "mutual_information"),
        ("Sigmoid4Log", "sigmoid4log"),
    ]

    with st.sidebar:
        st.header("Controls")
        dataset_display = st.selectbox(
            "Transform",
            options=[label for label, _ in dataset_labels],
            index=0,
        )
        dataset_map = {label: key for label, key in dataset_labels}
        dataset_key = dataset_map[dataset_display]
        data_bundle = load_dataset(dataset_key)

        if "all_experiments" not in data_bundle:
            st.warning("No experiments found in the selected dataset.")
            return

        experiments = cast(List[str], data_bundle["all_experiments"])
        if not experiments:
            st.warning("No experiments found in the selected dataset.")
            return

        experiment_selection = st.multiselect(
            "Experiments",
            options=experiments,
            default=experiments,
        )
        if not experiment_selection:
            st.info("Select at least one experiment to display.")

        seq_min = int(cast(int, data_bundle.get("seq_min", 1)))
        seq_max = int(cast(int, data_bundle.get("seq_max", 100)))
        length = st.slider(
            "Max length",
            min_value=seq_min,
            max_value=seq_max,
            value=seq_max,
            step=1,
        )

        realization_labels = ["Average across realizations"]
        realization_lookup: Dict[str, Optional[RealizationType]] = {"Average across realizations": None}
        for rid in cast(List[RealizationType], data_bundle.get("all_realizations", [])):
            label = _format_realization_label(rid)
            realization_labels.append(label)
            realization_lookup[label] = rid
        realization_label = st.selectbox("Realization", options=realization_labels)
        realization_value = realization_lookup[realization_label]

        error_bars = st.checkbox("Show error bars", value=False)

    metrics: Iterable[str] = ["GD", "NRMSD"]
    payload: List[Tuple[str, pd.DataFrame]] = []
    for metric_label in metrics:
        metric_df = data_bundle.get(metric_label)
        if isinstance(metric_df, pd.DataFrame):
            filtered_df = metric_df[metric_df["experiment_name"].isin(experiment_selection)]
            if not filtered_df.empty:
                payload.append((metric_label, filtered_df))

    if not payload:
        st.warning("No data available for the current selection.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes_raw = plt.subplots(1, len(payload), figsize=(12, 5))
    if isinstance(axes_raw, Axes):
        axes_list = [axes_raw]
    else:
        axes_list = list(axes_raw)

    legend_handles: List = []
    legend_labels: List[str] = []

    for ax, (metric_label, filtered_df) in zip(axes_list, payload):
        handles, labels = plot_score_results(
            filtered_df,
            length=length,
            error_bars=error_bars,
            realization_id=realization_value,
            title_prefix=metric_label,
            ax=ax,
        )
        if handles:
            legend_handles.extend(handles)
            legend_labels.extend(labels)

    for ax in axes_list[len(payload):]:
        ax.set_visible(False)

    if legend_handles:
        unique_handles: Dict[str, Artist] = {}
        for handle, label in zip(legend_handles, legend_labels):
            if label not in unique_handles:
                unique_handles[label] = cast(Artist, handle)

        fig.legend(
            list(unique_handles.values()),
            list(unique_handles.keys()),
            loc="upper center",
            ncol=min(len(legend_labels), 3),
            fontsize=9,
            frameon=True,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # st.caption(
    #     "This dashboard mirrors the controls of the local Tkinter app so collaborators can explore results without a Python environment."
    # )


if __name__ == "__main__":
    render_page()
