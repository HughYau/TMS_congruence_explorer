import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union, cast
import hashlib
import colorsys
import numpy as np
from dataclasses import dataclass


DATA_ROOT = Path(__file__).resolve().parent

RealizationType = Union[str, float, int]

DEFAULT_MESH_FILENAME = "geo.hdf5"
REGION_DISPLAY_NAMES: Dict[str, str] = {
    "crown": "Crown",
    "rim": "Rim",
    "sulcus": "Sulcus",
}
REGION_COLOR_MAP: Dict[str, str] = {
    "crown": "#3498db",
    "rim": "#e67e22",
    "sulcus": "#9b59b6",
}

HOTSPOT_FIG_BASE_WIDTH = 900
HOTSPOT_FIG_BASE_HEIGHT = 750
HOTSPOT_FIG_WIDTH = int(HOTSPOT_FIG_BASE_WIDTH * 1.5)


def _region_display_name(region_key: str) -> str:
    return REGION_DISPLAY_NAMES.get(region_key, region_key.capitalize())


def _normalize_realization_key(value: RealizationType) -> str:
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _to_zero_based_index(raw_index: RealizationType, size: int) -> Optional[int]:
    try:
        idx_value = int(raw_index)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    arr = np.asarray([idx_value], dtype=int)
    if arr.size == 0:
        return None
    if arr.min() == 1 and arr.max() <= size:
        arr = arr - 1
    idx = int(arr[0])
    if 0 <= idx < size:
        return idx
    return None


@dataclass
class MeshContext:
    vertices: np.ndarray
    triangles: np.ndarray
    tri_points: np.ndarray
    tri_centers: np.ndarray
    normals_unit: np.ndarray
    epsilon: float
    tissue_type: Optional[np.ndarray] = None


def _build_mesh_context(
    vertices: np.ndarray,
    triangles: np.ndarray,
    tissue_type: Optional[np.ndarray] = None,
) -> MeshContext:
    triangles_int = triangles.astype(int)
    if triangles_int.size > 0 and triangles_int.min() == 1 and triangles_int.max() <= vertices.shape[0]:
        triangles_int = triangles_int - 1
    tri_points = vertices[triangles_int]
    tri_centers = tri_points.mean(axis=1)
    v0 = tri_points[:, 0, :]
    v1 = tri_points[:, 1, :]
    v2 = tri_points[:, 2, :]
    normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals_unit = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 0)
    e1 = np.linalg.norm(v1 - v0, axis=1)
    e2 = np.linalg.norm(v2 - v1, axis=1)
    e3 = np.linalg.norm(v0 - v2, axis=1)
    edge_len_mean = np.mean((e1 + e2 + e3) / 3.0)
    epsilon = 0.01 * edge_len_mean if np.isfinite(edge_len_mean) and edge_len_mean > 0 else 1e-6
    return MeshContext(
        vertices=vertices,
        triangles=triangles_int,
        tri_points=tri_points,
        tri_centers=tri_centers,
        normals_unit=normals_unit,
        epsilon=float(epsilon),
        tissue_type=tissue_type,
    )


@st.cache_resource(show_spinner=False)
def load_mesh_context(mesh_path: str) -> Optional[MeshContext]:
    path = Path(mesh_path)
    if not path.is_absolute():
        path = DATA_ROOT / path
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    tissue_type: Optional[np.ndarray] = None
    if suffix == ".npz":
            with np.load(path) as data:
                if "vertices" not in data or "triangles" not in data:
                    return None
                vertices = data["vertices"]
                triangles = data["triangles"]
                try:
                    tissue_type = data["tissue_type"]
                except KeyError:
                    tissue_type = None
    elif suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        if isinstance(data, dict) and "vertices" in data and "triangles" in data:
            vertices = data["vertices"]
            triangles = data["triangles"]
            tissue_type = data.get("tissue_type") if isinstance(data, dict) else None
        else:
            return None
    elif suffix in {".h5", ".hdf5"}:
        try:
            import h5py  # type: ignore
        except ImportError:
            return None
        with h5py.File(path, "r") as handle:
            if (
                "/mesh/nodes/node_coord" not in handle
                or "/mesh/elm/triangle_number_list" not in handle
            ):
                return None
            vertices = handle["/mesh/nodes/node_coord"][...]
            triangles = handle["/mesh/elm/triangle_number_list"][...]
            tissue_type = (
                handle["/mesh/elm/tri_tissue_type"][...].flatten()
                if "/mesh/elm/tri_tissue_type" in handle
                else None
            )
    else:
        return None
    if not isinstance(vertices, np.ndarray) or not isinstance(triangles, np.ndarray):
        return None
    return _build_mesh_context(vertices, triangles, tissue_type=tissue_type)


def _triangle_vertices(mesh_ctx: MeshContext, tri_idx: int) -> Tuple[List[float], List[float], List[float], List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    verts = mesh_ctx.tri_points[tri_idx] + mesh_ctx.normals_unit[tri_idx] * mesh_ctx.epsilon
    xs = verts[:, 0].astype(float).tolist()
    ys = verts[:, 1].astype(float).tolist()
    zs = verts[:, 2].astype(float).tolist()
    xs_loop = xs + [xs[0]]
    ys_loop = ys + [ys[0]]
    zs_loop = zs + [zs[0]]
    return xs, ys, zs, xs_loop, ys_loop, zs_loop


def _triangle_patch_traces(
    mesh_ctx: MeshContext,
    tri_idx: int,
    color: str,
    name: str,
    legendgroup: Optional[str] = None,
    opacity: float = 0.8,
    showlegend: bool = True,
) -> Tuple[go.Mesh3d, go.Scatter3d]:
    xs, ys, zs, xs_loop, ys_loop, zs_loop = _triangle_vertices(mesh_ctx, tri_idx)
    fill_trace = go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=[0],
        j=[1],
        k=[2],
        color=color,
        opacity=opacity,
        flatshading=True,
        name=name,
        showlegend=showlegend,
        legendgroup=legendgroup,
        hovertemplate=f"{name}<extra></extra>",
    )
    outline_trace = go.Scatter3d(
        x=xs_loop,
        y=ys_loop,
        z=zs_loop,
        mode="lines",
        line=dict(color=color, width=5),
        name=f"{name} outline",
        showlegend=False,
        legendgroup=legendgroup,
        hoverinfo="skip",
    )
    return fill_trace, outline_trace


def _experiment_color_map(experiments: Iterable[str]) -> Dict[str, str]:
    try:
        import plotly.express as px  # type: ignore

        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.T10
    except Exception:
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    unique_experiments = sorted(set(experiments))
    return {exp: palette[idx % len(palette)] for idx, exp in enumerate(unique_experiments)}


def _build_hotspot_lookup(
    df: pd.DataFrame,
    experiments: Iterable[str],
    hotspot_idx: RealizationType,
    triangle_count: int,
) -> Dict[Tuple[int, str, Optional[str]], int]:
    subset = df[df["experiment_name"].isin(experiments)].copy()
    if subset.empty:
        return {}
    subset = subset[subset["hotspot_idx"].notna() & subset["current_hotspot_idx"].notna()]
    if subset.empty:
        return {}
    hotspot_numeric = pd.to_numeric(subset["hotspot_idx"], errors="coerce")
    subset = subset[hotspot_numeric == pd.to_numeric(pd.Series([hotspot_idx]), errors="coerce").iloc[0]]
    if subset.empty:
        return {}
    subset = subset.reset_index(drop=True)
    subset["__order__"] = np.arange(len(subset))
    group_cols = ["sequence_length", "experiment_name"]
    if "realization_id" in subset.columns:
        group_cols.append("realization_id")
    agg = subset.sort_values("__order__").groupby(group_cols, as_index=False).last()
    lookup: Dict[Tuple[int, str, Optional[str]], int] = {}
    for _, row in agg.iterrows():
        seq_val = int(row["sequence_length"])
        exp_val = str(row["experiment_name"])
        real_val = row.get("realization_id")
        real_key = _normalize_realization_key(real_val) if pd.notna(real_val) else None
        tri_idx = _to_zero_based_index(row["current_hotspot_idx"], triangle_count)
        if tri_idx is None:
            continue
        lookup[(seq_val, exp_val, real_key)] = tri_idx
    return lookup


def _base_mesh_trace(mesh_ctx: MeshContext) -> go.Mesh3d:
    return go.Mesh3d(
        x=mesh_ctx.vertices[:, 0],
        y=mesh_ctx.vertices[:, 1],
        z=mesh_ctx.vertices[:, 2],
        i=mesh_ctx.triangles[:, 0],
        j=mesh_ctx.triangles[:, 1],
        k=mesh_ctx.triangles[:, 2],
        color="rgba(210,210,210,1.0)",
        flatshading=False,
        lighting=dict(ambient=0.25, diffuse=0.9, specular=0.05, roughness=0.9, fresnel=0.2),
        lightposition=dict(x=100, y=200, z=100),
        name="Cortical mesh",
        hoverinfo="skip",
        showscale=False,
        opacity=1.0,
    )


def _format_hotspot_layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(itemsizing="constant", groupclick="togglegroup"),
        width=HOTSPOT_FIG_WIDTH,
        height=HOTSPOT_FIG_BASE_HEIGHT,
    )


def _create_average_hotspot_figure(
    mesh_ctx: MeshContext,
    regions: List[Tuple[str, str, int]],
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(_base_mesh_trace(mesh_ctx))
    for region_key, display_name, tri_idx in regions:
        if tri_idx is None:
            continue
        color = REGION_COLOR_MAP.get(region_key, _hash_color(display_name))
        legendgroup = f"region-{region_key}"
        fill_trace, outline_trace = _triangle_patch_traces(
            mesh_ctx,
            tri_idx,
            color=color,
            name=f"{display_name} hotspot",
            legendgroup=legendgroup,
            opacity=0.72,
            showlegend=True,
        )
        fig.add_trace(fill_trace)
        fig.add_trace(outline_trace)
        center = mesh_ctx.tri_centers[tri_idx] + mesh_ctx.normals_unit[tri_idx] * mesh_ctx.epsilon
        fig.add_trace(
            go.Scatter3d(
                x=[float(center[0])],
                y=[float(center[1])],
                z=[float(center[2])],
                mode="markers",
                name=f"{display_name} center",
                legendgroup=legendgroup,
                marker=dict(size=6, symbol="x", color=color, line=dict(width=2, color="#111111")),
                hovertemplate=f"{display_name} hotspot<extra></extra>",
                showlegend=False,
            )
        )
    _format_hotspot_layout(fig, "Target hotspots")
    return fig


def _create_realization_hotspot_figure(
    mesh_ctx: MeshContext,
    experiments: List[str],
    seq_value: int,
    realization_key: str,
    realization_title: str,
    region_payload: List[Tuple[str, str, int, Dict[Tuple[int, str, Optional[str]], int]]],
) -> Tuple[go.Figure, Dict[str, List[str]]]:
    fig = go.Figure()
    fig.add_trace(_base_mesh_trace(mesh_ctx))
    color_map = _experiment_color_map(experiments)
    experiments_seen: Set[str] = set()
    missing: Dict[str, List[str]] = {}
    for region_key, display_name, tri_idx, lookup in region_payload:
        color_region = REGION_COLOR_MAP.get(region_key, _hash_color(display_name))
        legendgroup_region = f"region-{region_key}"
        center = mesh_ctx.tri_centers[tri_idx] + mesh_ctx.normals_unit[tri_idx] * mesh_ctx.epsilon
        fig.add_trace(
            go.Scatter3d(
                x=[float(center[0])],
                y=[float(center[1])],
                z=[float(center[2])],
                mode="markers",
                name=f"{display_name} target",
                marker=dict(size=6, symbol="x", color=color_region, line=dict(width=2, color="#111111")),
                legendgroup=legendgroup_region,
                showlegend=True,
                hovertemplate=f"{display_name} target<extra></extra>",
            )
        )
        for experiment in experiments:
            legendgroup_exp = f"exp-{experiment}"
            key = (seq_value, experiment, realization_key)
            tri_lookup = lookup.get(key)
            if tri_lookup is None:
                missing.setdefault(display_name, []).append(experiment)
                continue
            fill_trace, outline_trace = _triangle_patch_traces(
                mesh_ctx,
                tri_lookup,
                color=color_map[experiment],
                name=f"{experiment}",
                legendgroup=legendgroup_exp,
                opacity=0.85,
                showlegend=experiment not in experiments_seen,
            )
            fill_trace.hovertemplate = (
                f"{experiment}<br>{display_name} · seq {seq_value}<extra></extra>"
            )
            fig.add_trace(fill_trace)
            fig.add_trace(outline_trace)
            experiments_seen.add(experiment)
    _format_hotspot_layout(fig, f"Realization {realization_title} · seq {seq_value}")
    return fig, missing


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_value = hex_color.lstrip("#")
    if len(hex_value) != 6:
        return hex_color
    r = int(hex_value[0:2], 16)
    g = int(hex_value[2:4], 16)
    b = int(hex_value[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _stable_hash(value: str) -> int:
    """Return a stable positive integer hash for a string across sessions.

    Uses MD5 to avoid Python's randomized hash salt.
    """
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    # Use first 8 hex digits to form a 32-bit int
    return int(digest[:8], 16)


def _hash_color(label: str, region_color_offset: int = 0) -> str:
    """Generate a deterministic, aesthetically pleasing HEX color from a label.

    We map the label to a hue on the HSV color wheel and apply a small
    region-specific offset so Crown/Rim colors remain distinguishable.
    """
    base = _stable_hash(label)
    # Spread hues around the wheel; add an offset per region
    hue = (base % 360 + region_color_offset * 30) % 360  # degrees
    sat = 0.65  # keep reasonably saturated for visibility
    val = 0.75  # medium brightness to work on light background
    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, sat, val)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def _region_color(label: str) -> str:
    key = label.lower()
    return REGION_COLOR_MAP.get(key, _hash_color(label))


def _auc_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if y.size == 0 or x.size == 0:
        return float("nan")
    return float(np.trapz(y, x))


def _steps_to_threshold_value(y: np.ndarray, x: np.ndarray, threshold: float) -> float:
    if y.size == 0 or x.size == 0:
        return float("nan")
    mask = y <= threshold
    if not np.any(mask):
        return float("nan")
    first_idx = int(np.argmax(mask))
    return float(x[first_idx])


def _hit_and_hold_first_step(
    y: np.ndarray,
    x: np.ndarray,
    threshold: float,
    window: int,
) -> float:
    if window <= 0 or y.size == 0 or x.size == 0 or y.size != x.size:
        return float("nan")
    if y.size < window:
        return float("nan")
    mask = y <= threshold
    if not np.any(mask):
        return float("nan")
    kernel = np.ones(window, dtype=int)
    conv = np.convolve(mask.astype(int), kernel, mode="valid")
    idx_candidates = np.where(conv == window)[0]
    if idx_candidates.size == 0:
        return float("nan")
    idx = int(idx_candidates[0])
    idx = min(idx, x.size - 1)
    return float(x[idx])


def _summarize_gd_metrics(
    df: pd.DataFrame,
    region_label: str,
    max_step: int,
    auc_step_limit: int,
    steps_threshold: float,
    hnh_threshold: float,
    hnh_window: int,
) -> pd.DataFrame:
    required_cols = {"experiment_name", "sequence_length", "score"}
    if df.empty or not required_cols.issubset(df.columns):
        return pd.DataFrame(columns=["region", "experiment", "realization", "auc", "steps_to_mm", "hnh_step"])

    working = df.copy()
    working = working.dropna(subset=["score", "sequence_length"])
    if working.empty:
        return pd.DataFrame(columns=["region", "experiment", "realization", "auc", "steps_to_mm", "hnh_step"])

    working = working[working["sequence_length"] <= max_step]
    if working.empty:
        return pd.DataFrame(columns=["region", "experiment", "realization", "auc", "steps_to_mm", "hnh_step"])

    realization_col = "__realization__"
    if "realization_id" in working.columns:
        working[realization_col] = working["realization_id"].apply(
            lambda value: _normalize_realization_key(value) if pd.notna(value) else "Aggregate"
        )
    else:
        working[realization_col] = "Aggregate"

    rows = []
    limit = max(1, min(int(auc_step_limit), int(max_step)))

    for (experiment, realization), sub in working.groupby(["experiment_name", realization_col]):
        sub = sub.sort_values("sequence_length")
        x = sub["sequence_length"].to_numpy(dtype=float)
        y = sub["score"].to_numpy(dtype=float)
        if x.size == 0:
            continue

        mask_limit = x <= limit
        auc_value = _auc_trapezoid(y[mask_limit], x[mask_limit]) if np.any(mask_limit) else float("nan")
        steps_value = _steps_to_threshold_value(y, x, steps_threshold)
        hnh_value = _hit_and_hold_first_step(y, x, hnh_threshold, hnh_window)

        rows.append(
            {
                "region": region_label,
                "experiment": str(experiment),
                "realization": str(realization),
                "auc": auc_value,
                "steps_to_mm": steps_value,
                "hnh_step": hnh_value,
            }
        )

    return pd.DataFrame(rows)


def _aggregate_metric_summary(metrics_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if metrics_df.empty or value_col not in metrics_df:
        return pd.DataFrame(columns=["region", "experiment", "mean", "std", "count", "ci95"])

    df_clean = metrics_df.dropna(subset=[value_col])
    if df_clean.empty:
        return pd.DataFrame(columns=["region", "experiment", "mean", "std", "count", "ci95"])

    summary = (
        df_clean.groupby(["region", "experiment"], as_index=False)
        .agg(
            mean=(value_col, "mean"),
            std=(value_col, "std"),
            count=(value_col, "count"),
        )
    )
    summary["count"] = summary["count"].fillna(0).astype(int)
    summary["std"] = summary["std"].fillna(0.0)
    summary["sem"] = summary.apply(
        lambda row: row["std"] / np.sqrt(row["count"]) if row["count"] > 0 else 0.0,
        axis=1,
    )
    summary["ci95"] = summary["sem"] * 1.96
    summary["ci95"] = summary["ci95"].fillna(0.0)
    return summary.drop(columns=["sem"])


def _build_meta_bar_figure(summary_df: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        barmode="group",
        template="plotly_white",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=70, r=30, t=70, b=70),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.12)",
            borderwidth=1,
            font=dict(color="#1f2933", size=11),
        ),
        font=dict(family="Segoe UI", size=13, color="#1f2933"),
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color="#1f2933")),
    )
    fig.update_xaxes(
        title=dict(text="Experiment", font=dict(color="#1f2933", size=12)),
        tickfont=dict(color="#1f2933", size=11),
        linecolor="rgba(0,0,0,0.2)",
        gridcolor="rgba(0,0,0,0.08)",
    )
    fig.update_yaxes(
        title=dict(text=y_label, font=dict(color="#1f2933", size=12)),
        tickfont=dict(color="#1f2933", size=11),
        rangemode="tozero",
        gridcolor="rgba(0,0,0,0.08)",
        linecolor="rgba(0,0,0,0.2)",
    )

    if summary_df.empty:
        return fig

    summary_df = summary_df.sort_values(["experiment", "region"])

    for idx, (region_label_value, region_df) in enumerate(summary_df.groupby("region", sort=False)):
        region_label = str(region_label_value)
        color = _region_color(region_label)
        ci_data = region_df["ci95"].to_numpy(dtype=float)
        count_data = region_df["count"].to_numpy(dtype=float)
        std_data = region_df["std"].to_numpy(dtype=float)
        if region_df.shape[0] > 0:
            custom = np.column_stack([
                region_df["region"].astype(str).to_numpy(dtype=object),
                ci_data,
                count_data,
                std_data,
            ])
        else:
            custom = None

        fig.add_trace(
            go.Bar(
                name=region_label,
                x=region_df["experiment"],
                y=region_df["mean"],
                offsetgroup=str(idx),
                legendgroup=region_label,
                marker=dict(color=color),
                error_y=dict(type="data", array=ci_data, visible=True),
                customdata=custom,
                hovertemplate=(
                    "<b>%{x}</b><br>Region: %{customdata[0]}<br>Mean: %{y:.3f}"
                    "<br>95% CI: %{customdata[1]:.3f}<br>Samples: %{customdata[2]:.0f}<extra></extra>"
                ),
            )
        )

    return fig


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
        3: {  # Sulcus palette (purple tones)
            "RANDOM": "#3d3d5c",
            "ROW": "#9b59b6",
            "COL": "#8e44ad",
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
        # Pick color: use predefined map for known bases; otherwise generate a stable color
        if base in color_map:
            color = color_map[base]
        else:
            color = _hash_color(str(exp_name), region_color_offset=region_color_offset)
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
        "sulcus": [
            ("GD", "exp_c_map_score_metrics_GD_sigmoid4log_sulcus.csv"),
            ("NRMSD", "exp_c_map_score_metrics_NRMSD_sigmoid4log_sulcus.csv"),
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
    st.set_page_config(page_title="TMS Mapping Congruence Explorer", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --control-surface: ##0e1117;
            --control-border: #2c3e55;
            --control-foreground: #f1f5f9;
        }
        .stApp {
            background-color: var(--control-surface);
            color: var(--control-foreground);
        }
        div[data-testid="stSidebar"] {
            background-color: var(--control-surface);
            border-right: 1px solid var(--control-border);
        }
        div[data-testid="stSidebar"] * {
            color: var(--control-foreground) !important;
        }
        .main .block-container {
            background-color: var(--control-surface);
            color: var(--control-foreground);
        }
        .stSlider > div[data-baseweb="slider"] > div:nth-child(2) {
            background-color: #e9ecef;
        }
        .stSlider > div[data-baseweb="slider"] div[role="slider"] {
            background-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25);
        }
        h1, h2, h3 {
            color: var(--control-foreground);
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
    st.title("TMS Mapping Congruence Explorer")

    selected_hotspots: Dict[str, int] = {}
    region_hotspot_frames: Dict[str, pd.DataFrame] = {}
    mesh_file_path: str = st.session_state.get("mesh_file_path", DEFAULT_MESH_FILENAME)

    dataset_labels: List[Tuple[str, str]] = [
        ("Crown", "crown"),
        ("Rim", "rim"),
        ("Sulcus", "sulcus"),
    ]
    # Initialize to satisfy static analyzers; will be set based on selection
    data_bundle: Dict[str, object] = {}

    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Dataset selection with expander for cleaner look
        with st.expander("📊 Dataset Selection", expanded=True):
            # Comparison mode: show all three regions together
            compare_regions = st.checkbox(
                "🔄 Compare all regions (Crown/Rim/Sulcus)",
                value=False,
                help="Enable this to overlay Crown, Rim and Sulcus data for the same experiments"
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

                experiments = cast(List[str], data_bundle["all_experiments"]) if "all_experiments" in data_bundle else []
                all_realizations_list = cast(List[RealizationType], data_bundle.get("all_realizations", []))
                seq_min = int(cast(int, data_bundle.get("seq_min", 1)))
                seq_max = int(cast(int, data_bundle.get("seq_max", 100)))
            else:
                st.info("📊 Comparing Crown, Rim and Sulcus regions")
                # Load all three datasets
                crown_bundle = load_dataset("crown")
                rim_bundle = load_dataset("rim")
                sulcus_bundle = load_dataset("sulcus")
                datasets_to_plot = [("crown", crown_bundle), ("rim", rim_bundle), ("sulcus", sulcus_bundle)]
                dataset_key = "all_regions"

                # Union of experiments across regions
                exp_sets = []
                real_sets = []
                seq_mins: List[int] = []
                seq_maxs: List[int] = []
                for _, bundle in datasets_to_plot:
                    exp_sets.append(set(cast(List[str], bundle.get("all_experiments", []))))
                    real_sets.append(set(cast(List[RealizationType], bundle.get("all_realizations", []))))
                    if "seq_min" in bundle:
                        seq_mins.append(int(cast(int, bundle["seq_min"])))
                    if "seq_max" in bundle:
                        seq_maxs.append(int(cast(int, bundle["seq_max"])))
                experiments = sorted(set().union(*exp_sets)) if exp_sets else []
                all_realizations_list = sorted(set().union(*real_sets), key=lambda v: str(v)) if real_sets else []
                seq_min = min(seq_mins) if seq_mins else 1
                seq_max = max(seq_maxs) if seq_maxs else 100

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
                key="experiment_selection",
                help=f"Select from {len(experiments)} available experiments"
            )

            if not experiment_selection:
                st.info("👆 Select at least one experiment to display")
            else:
                st.success(f"✓ {len(experiment_selection)} experiments selected")

        # Data range controls
        with st.expander("📏 Data Range", expanded=True):
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
            for rid in cast(List[RealizationType], all_realizations_list):
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

        with st.expander("🔥 Hotspot Selection", expanded=False):
            st.caption("Configure the cortical mesh file and focus hotspots for 3D viewing.")
            mesh_candidate = st.text_input(
                "Mesh data file",
                value=mesh_file_path,
                help="Path to an .npz or .npy file (relative to data folder) containing 'vertices' and 'triangles' arrays",
            ).strip()
            mesh_file_path = mesh_candidate or DEFAULT_MESH_FILENAME
            st.session_state["mesh_file_path"] = mesh_file_path

            region_hotspot_frames = {}
            region_hotspot_options: Dict[str, List[int]] = {}
            for region_key, region_bundle in datasets_to_plot:
                metric_df = region_bundle.get("GD")
                if not isinstance(metric_df, pd.DataFrame):
                    continue
                filtered_df = metric_df[metric_df["experiment_name"].isin(experiment_selection)]
                filtered_df = filtered_df[filtered_df["sequence_length"] <= length]
                if filtered_df.empty:
                    continue
                region_hotspot_frames[region_key] = filtered_df
                hotspot_vals = pd.to_numeric(filtered_df["hotspot_idx"], errors="coerce").dropna().astype(int)
                unique_vals = sorted(set(int(val) for val in hotspot_vals.tolist()))
                if unique_vals:
                    region_hotspot_options[region_key] = unique_vals

            if region_hotspot_options:
                for region_key, options in region_hotspot_options.items():
                    label = _region_display_name(region_key)
                    select_key = f"hotspot_selection_{region_key}"
                    default_value = options[0]
                    current_value = st.session_state.get(select_key, default_value)
                    if current_value not in options:
                        current_value = default_value
                    selected_value = st.selectbox(
                        f"{label} hotspot idx",
                        options,
                        index=options.index(current_value),
                        key=select_key,
                    )
                    selected_hotspots[region_key] = int(selected_value)
            else:
                st.caption("No hotspot metadata available for the current selection.")
            
        # Summary statistics at bottom
        st.divider()
        st.caption("📈 **Quick Stats**")
        if experiment_selection:
            st.caption(f"• Experiments: {len(experiment_selection)}")
            st.caption(f"• Steps range: {seq_min} → {length}")
            if isinstance(all_realizations_list, list):
                st.caption(f"• Realizations: {len(all_realizations_list)}")

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

    meta_region_frames: List[Tuple[str, pd.DataFrame, int]] = []
    for metric_label, region_data_list in payload:
        if metric_label == "GD":
            meta_region_frames = [(region_key, region_df, color_offset) for region_key, region_df, color_offset in region_data_list]
            break

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

    base_height = 630
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
    region_hotspot_payload: List[Tuple[str, str, int, pd.DataFrame]] = []
    for region_key, frame in region_hotspot_frames.items():
        hotspot_choice = selected_hotspots.get(region_key)
        if hotspot_choice is None:
            continue
        region_hotspot_payload.append((region_key, _region_display_name(region_key), hotspot_choice, frame))

    tab_plot, tab_hotspot, tab_summary, tab_meta = st.tabs([
        "📊 Visualization",
        "🧠 Hotspot Map",
        "📋 Data Summary",
        "🧮 Meta Analysis",
    ])

    with tab_plot:
        st.plotly_chart(plotly_fig, use_container_width=True, config=config)
        
        if empty_metrics:
            st.info("No data within the selected length for: " + ", ".join(empty_metrics))
    
    with tab_hotspot:
        mesh_ctx = load_mesh_context(mesh_file_path) if region_hotspot_payload else None
        if not region_hotspot_payload:
            st.info("Select at least one hotspot in the sidebar to visualize the surface.")
        elif mesh_ctx is None:
            st.info("Mesh file not found, missing required arrays, or h5py is not installed. Update the path or install h5py under 🔥 Hotspot Selection.")
        else:
            mesh_size = mesh_ctx.triangles.shape[0]
            if realization_value is None:
                targets = []
                for region_key, display_name, hotspot_idx, _ in region_hotspot_payload:
                    tri_idx = _to_zero_based_index(hotspot_idx, mesh_size)
                    if tri_idx is not None:
                        targets.append((region_key, display_name, tri_idx))
                if not targets:
                    st.warning("Hotspot indices are outside the mesh range.")
                else:
                    avg_fig = _create_average_hotspot_figure(mesh_ctx, targets)
                    hotspot_config = {
                        "displayModeBar": True,
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["hovercompare", "lasso3d", "select3d"],
                    }
                    st.plotly_chart(avg_fig, use_container_width=False, config=hotspot_config)
                    st.caption("Average view shows the selected target hotspots across regions.")
            else:
                norm_real = _normalize_realization_key(realization_value)
                region_lookup_payload: List[Tuple[str, str, int, Dict[Tuple[int, str, Optional[str]], int]]] = []
                seq_candidates: Set[int] = set()
                for region_key, display_name, hotspot_idx, frame in region_hotspot_payload:
                    tri_idx = _to_zero_based_index(hotspot_idx, mesh_size)
                    if tri_idx is None:
                        continue
                    lookup = _build_hotspot_lookup(frame, experiment_selection, hotspot_idx, mesh_size)
                    region_lookup_payload.append((region_key, display_name, tri_idx, lookup))
                    for (seq_val, _, real_key), _ in lookup.items():
                        if real_key == norm_real:
                            seq_candidates.add(seq_val)
                if not region_lookup_payload:
                    st.warning("Unable to build hotspot traces for the selected configuration.")
                else:
                    seq_options = sorted(seq_candidates)
                    if not seq_options:
                        st.info("No hotspot traces found for the chosen realization.")
                    else:
                        default_seq = seq_options[-1]
                        seq_value = st.select_slider(
                            "Sequence step",
                            options=seq_options,
                            value=default_seq,
                            help="Select a sequence length to inspect algorithm hotspot predictions.",
                        )
                        realization_key = norm_real
                        real_fig, missing_map = _create_realization_hotspot_figure(
                            mesh_ctx,
                            experiment_selection,
                            seq_value,
                            realization_key,
                            realization_label,
                            region_lookup_payload,
                        )
                        hotspot_config = {
                            "displayModeBar": True,
                            "displaylogo": False,
                            "modeBarButtonsToRemove": ["hovercompare", "lasso3d", "select3d"],
                        }
                        st.plotly_chart(real_fig, use_container_width=False, config=hotspot_config)
                        if missing_map:
                            missing_lines = [
                                f"{region}: {', '.join(sorted(set(exps)))}"
                                for region, exps in missing_map.items()
                                if exps
                            ]
                            if missing_lines:
                                st.caption("No hotspot record for → " + " | ".join(missing_lines))

    with tab_summary:
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

    with tab_meta:
        st.subheader("Meta Analysis")

        if not meta_region_frames:
            st.info("Meta analysis is available when GD results are loaded.")
        else:
            score_min_list: List[float] = []
            score_max_list: List[float] = []
            for _, gd_frame, _ in meta_region_frames:
                if not gd_frame.empty:
                    series = gd_frame["score"].dropna()
                    if not series.empty:
                        score_min_list.append(float(series.min()))
                        score_max_list.append(float(series.max()))

            score_min = min(score_min_list) if score_min_list else 0.0
            score_max = max(score_max_list) if score_max_list else 25.0
            if score_min == score_max:
                score_max = score_min + 1.0

            threshold_min = float(np.floor(max(0.0, score_min)))
            threshold_max = float(np.ceil(score_max))
            if threshold_max <= threshold_min:
                threshold_max = threshold_min + 1.0

            default_steps_threshold = float(min(max(10.0, threshold_min), threshold_max))
            default_hnh_threshold = float(min(max(5.0, threshold_min), threshold_max))

            max_step_limit = max(1, int(length))
            default_auc_limit = min(max_step_limit, 50)
            window_max = max(1, min(20, max_step_limit))
            default_hnh_window = min(window_max, 8)

            meta_config = {
                "displaylogo": False,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            }

            def _meta_summary(value_col: str, auc_limit: int, steps_thr: float, hnh_thr: float, hnh_win: int) -> pd.DataFrame:
                frames: List[pd.DataFrame] = []
                for region_key, gd_frame, _ in meta_region_frames:
                    region_label = _region_display_name(region_key)
                    metrics_df = _summarize_gd_metrics(
                        gd_frame,
                        region_label=region_label,
                        max_step=max_step_limit,
                        auc_step_limit=auc_limit,
                        steps_threshold=steps_thr,
                        hnh_threshold=hnh_thr,
                        hnh_window=hnh_win,
                    )
                    if not metrics_df.empty:
                        frames.append(metrics_df)
                if not frames:
                    return pd.DataFrame()
                combined = pd.concat(frames, ignore_index=True)
                return _aggregate_metric_summary(combined, value_col)

            auc_state_default = int(st.session_state.get("meta_auc_limit", default_auc_limit))
            steps_state_default = float(st.session_state.get("meta_steps_threshold", default_steps_threshold))
            hnh_threshold_state = float(st.session_state.get("meta_hnh_threshold", default_hnh_threshold))
            hnh_window_state = int(st.session_state.get("meta_hnh_window", default_hnh_window))

            st.markdown("**AUC**")
            auc_step_limit = st.slider(
                "AUC integration limit (steps)",
                min_value=1,
                max_value=max_step_limit,
                value=min(max_step_limit, max(1, auc_state_default)),
                step=1,
                key="meta_auc_limit",
                help="Upper bound on steps when computing the area under the GD curve.",
            )

            steps_for_auc = float(st.session_state.get("meta_steps_threshold", default_steps_threshold))
            hnh_thr_for_auc = float(st.session_state.get("meta_hnh_threshold", default_hnh_threshold))
            hnh_window_for_auc = int(st.session_state.get("meta_hnh_window", default_hnh_window))

            auc_summary = _meta_summary("auc", auc_step_limit, steps_for_auc, hnh_thr_for_auc, hnh_window_for_auc)
            if auc_summary.empty:
                st.info("AUC data is not available with the selected parameters.")
            else:
                auc_fig = _build_meta_bar_figure(
                    auc_summary,
                    title=f"AUC (0-{auc_step_limit} steps)",
                    y_label="Area under curve",
                )
                st.plotly_chart(auc_fig, use_container_width=True, config=meta_config)

            st.markdown("**Steps to Threshold**")
            steps_threshold = st.slider(
                "Steps-to-threshold (mm)",
                min_value=threshold_min,
                max_value=threshold_max,
                value=float(np.clip(steps_state_default, threshold_min, threshold_max)),
                step=0.5,
                key="meta_steps_threshold",
                help="Threshold for GD distance when computing steps-to-mm.",
            )

            current_auc_limit = int(st.session_state.get("meta_auc_limit", auc_step_limit))
            current_hnh_threshold = float(st.session_state.get("meta_hnh_threshold", default_hnh_threshold))
            current_hnh_window = int(st.session_state.get("meta_hnh_window", default_hnh_window))

            steps_summary = _meta_summary("steps_to_mm", current_auc_limit, steps_threshold, current_hnh_threshold, current_hnh_window)
            if steps_summary.empty:
                st.info("Steps-to-threshold data is not available with the selected parameters.")
            else:
                steps_fig = _build_meta_bar_figure(
                    steps_summary,
                    title=f"Steps to <= {steps_threshold:.2f} mm",
                    y_label="Steps",
                )
                st.plotly_chart(steps_fig, use_container_width=True, config=meta_config)

            st.markdown("**Hit-and-Hold**")
            col_thr, col_win = st.columns([2, 1])
            with col_thr:
                hnh_threshold = st.slider(
                    "HnH threshold (mm)",
                    min_value=threshold_min,
                    max_value=threshold_max,
                    value=float(np.clip(hnh_threshold_state, threshold_min, threshold_max)),
                    step=0.5,
                    key="meta_hnh_threshold",
                    help="Threshold for GD distance when evaluating hit-and-hold.",
                )
            with col_win:
                hnh_window = st.slider(
                    "HnH window size (steps)",
                    min_value=1,
                    max_value=window_max,
                    value=int(np.clip(hnh_window_state, 1, window_max)),
                    step=1,
                    key="meta_hnh_window",
                    help="Number of consecutive steps that must satisfy the threshold.",
                )

            hnh_summary = _meta_summary("hnh_step", current_auc_limit, steps_threshold, hnh_threshold, hnh_window)
            if hnh_summary.empty:
                st.info("Hit-and-hold data is not available with the selected parameters.")
            else:
                hnh_fig = _build_meta_bar_figure(
                    hnh_summary,
                    title=f"Hit-and-Hold (threshold {hnh_threshold:.2f} mm, window {hnh_window})",
                    y_label="Steps",
                )
                st.plotly_chart(hnh_fig, use_container_width=True, config=meta_config)

    # st.caption(
    #     "This dashboard mirrors the controls of the local Tkinter app so collaborators can explore results without a Python environment."
    # )


if __name__ == "__main__":
    render_page()
