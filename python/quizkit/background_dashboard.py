import sys
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.ndimage import binary_dilation, find_objects, label

from quizkit.hologram_experiment import HologramExperiment
from quizkit.run_phase_retrieval import PerformanceMetrics

#
#  streamlit run python/quizkit/background_dashboard.py
# 

st.set_page_config(page_title="Artifact Analysis", layout="wide")

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="./results")
parser.add_argument("--run_hash", type=str, default=None)
parser.add_argument("--solver_hash", type=str, default=None)

args, _ = parser.parse_known_args()


def get_most_recent_hashes(base_dir: str):
    base_path = Path(base_dir)
    fallback_run, fallback_hash = "xyz", "abc1234"

    if not base_path.exists():
        return fallback_run, fallback_hash

    run_dirs = [
        d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]
    if not run_dirs:
        return fallback_run, fallback_hash

    latest_run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
    latest_run_hash = latest_run_dir.name.replace("run_", "")

    phase_dir = latest_run_dir / "phase_retrieval"
    if not phase_dir.exists() or not phase_dir.is_dir():
        return latest_run_hash, fallback_hash

    solver_dirs = [d for d in phase_dir.iterdir() if d.is_dir()]
    if not solver_dirs:
        return latest_run_hash, fallback_hash

    latest_solver = max(solver_dirs, key=lambda d: d.stat().st_mtime)

    return latest_run_hash, latest_solver.name


auto_run_hash, auto_solver_hash = get_most_recent_hashes(args.base_dir)
default_run_hash = args.run_hash if args.run_hash else auto_run_hash
default_solver_hash = args.solver_hash if args.solver_hash else auto_solver_hash

st.sidebar.header("Run Selection")
run_hash = st.sidebar.text_input("Run Hash", default_run_hash)
solver_hash = st.sidebar.text_input("Solver Hash", default_solver_hash)


def extract_background_artifacts(
    forward_intensity,
    trap_labels,
    exclusion_mask=None,
    exclusion_pad=15,
    percentile_q=99.9,
    max_artifacts=100,
):
    if exclusion_pad > 0:
        struct = np.ones((exclusion_pad * 2 + 1, exclusion_pad * 2 + 1), dtype=bool)
        trap_mask = binary_dilation(trap_labels > 0, structure=struct)
    else:
        trap_mask = trap_labels > 0

    if exclusion_mask is not None:
        trap_mask = np.logical_or(trap_mask, exclusion_mask)

    bg_mask = ~trap_mask
    residual_int = forward_intensity * bg_mask

    bg_pixels = residual_int[bg_mask]
    if len(bg_pixels) == 0:
        return [], residual_int

    artifact_threshold = np.percentile(bg_pixels, percentile_q)
    binary_artifacts = residual_int > artifact_threshold

    labeled_artifacts, _ = label(binary_artifacts)
    slices = find_objects(labeled_artifacts)

    artifacts = []
    for i, s in enumerate(slices):
        cy = (s[0].start + s[0].stop) // 2
        cx = (s[1].start + s[1].stop) // 2
        comp_mask = labeled_artifacts[s] == (i + 1)
        comp_power = np.sum(residual_int[s][comp_mask])
        artifacts.append({"id": i, "cy": cy, "cx": cx, "power": comp_power})

    artifacts.sort(key=lambda x: x["power"], reverse=True)
    return artifacts[:max_artifacts], residual_int


@st.cache_resource
def load_data(base_dir: str, r_hash: str, s_hash: str):
    run_dir = Path(base_dir) / f"run_{r_hash}"
    solver_dir = run_dir / "phase_retrieval" / s_hash

    exp = HologramExperiment.from_run_h5(run_dir)

    with open(solver_dir / "solver_config.json", "r") as f:
        solver_config = json.load(f)
    with open(solver_dir / "performance_metrics.json", "r") as f:
        metrics = json.load(f)

    sol_h5 = solver_dir / f"phase_solution_{s_hash}.h5"
    with h5py.File(sol_h5, "r") as f:
        ff_int = f["target/inferred_farfield_intensity"][:]
        artifact_stacks = f["background/off_target_intensity_stacks"][:]

    try:
        with h5py.File(run_dir / "experiment.h5", "r") as f:
            recip_coords = f["dual_coords/dual_coords"][:]
    except Exception:
        recip_coords = None

    artifacts, residual_int = extract_background_artifacts(
        forward_intensity=ff_int,
        trap_labels=np.array(exp.trap_labels),
        exclusion_pad=15,
        percentile_q=99.9,
    )

    return (
        exp,
        solver_config,
        metrics,
        ff_int,
        residual_int,
        artifacts,
        artifact_stacks,
        recip_coords,
    )


try:
    (
        exp,
        solver_config,
        metrics,
        ff_int,
        residual_int,
        artifacts,
        artifact_stacks,
        recip_coords,
    ) = load_data(args.base_dir, run_hash, solver_hash)
except Exception as e:
    st.error(f"Failed to load data:\n\n{e}")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("Run Configuration")
st.sidebar.markdown(f"**Trap Type:** `{exp.run_config.trap_config.trap_type}`")
st.sidebar.markdown(f"**Method:** `{solver_config.get('method', 'Unknown')}`")
st.sidebar.markdown(
    f"**Backend:** `{solver_config.get('solver_backend', 'slm_suite')}`"
)
st.sidebar.markdown(f"**Smooth Phase:** `{solver_config.get('smooth_phase', False)}`")

total_bg_power = np.sum(residual_int)
df_list = []
perimeter_mask = np.array(exp.trap_array_perimeter_mask)

for i, art in enumerate(artifacts):
    cx, cy = art["cx"], art["cy"]
    is_fuzz = bool(perimeter_mask[cy, cx])
    rel_power = (art["power"] / total_bg_power) * 100
    df_list.append(
        {"id": i + 1, "cx": cx, "cy": cy, "rel_power": rel_power, "is_fuzz": is_fuzz}
    )

df = pd.DataFrame(df_list)

col_left, col_right = st.columns([3, 1], gap="large")

with col_left:
    st.subheader("Performance Metrics")
    method_name = solver_config.get("method", "Solver")
    metric_names = [k for k in metrics.keys() if k != "trap_powers"]
    ui_rows = []

    for m in metric_names:
        val = metrics.get(m, np.nan)
        if isinstance(val, float) and not np.isnan(val):
            val_str = (
                f"{val:.2e}"
                if (val != 0 and (abs(val) < 1e-3 or abs(val) > 1e4))
                else f"{val:.4f}"
            )
        else:
            val_str = str(val)

        desc = PerformanceMetrics._DESCRIPTIONS.get(m, "")
        ui_rows.append({"Metric Key": m, method_name: val_str, "Description": desc})

    st.dataframe(pd.DataFrame(ui_rows), use_container_width=True, hide_index=True)

    st.subheader("Forward intensity")

    col_toggles1, col_toggles2 = st.columns(2)
    with col_toggles1:
        use_log_scale = st.checkbox("Log Scale Intensity", value=True)
    with col_toggles2:
        exclude_fuzz = st.checkbox("Exclude interference artefacts", value=False)

    active_df = df[~df["is_fuzz"]].copy() if exclude_fuzz else df.copy()
    map_container = st.container()

if active_df.empty:
    st.warning("No artifacts match the current filters.")
    st.stop()

with col_right:
    st.subheader("Top Artifacts")
    display_df = active_df.head(5)[
        ["id", "cx", "cy", "rel_power", "is_fuzz"]
    ].set_index("id")
    st.dataframe(
        display_df.style.format({"rel_power": "{:.2f}%"}), use_container_width=True
    )

if use_log_scale:
    ff_int_plot = np.log(ff_int + 1e-12)
else:
    ff_int_plot = ff_int

fig_map = px.imshow(ff_int_plot, color_continuous_scale="viridis")
fig_map.update_traces(hoverinfo="skip", hovertemplate=None, selector=dict(type="image"))

slm_h, slm_w = exp.run_config.slm_shape
fig_map.add_scatter(
    x=[slm_w / 2.0],
    y=[slm_h / 2.0],
    mode="markers",
    marker=dict(
        color="gold", size=32, opacity=0.8, symbol="pentagon-open", line=dict(width=2.5)
    ),
    hoverinfo="skip",
    showlegend=True,
    name="0th Order",
)

trap_coords = np.array(exp.trap_coords)
fig_map.add_scatter(
    x=trap_coords[:, 1],
    y=trap_coords[:, 0],
    mode="markers",
    marker=dict(color="cyan", size=21, symbol="circle-open", line=dict(width=1.5)),
    hoverinfo="skip",
    showlegend=True,
    name="Traps",
)

if recip_coords is not None and len(recip_coords) > 0:
    fig_map.add_scatter(
        x=recip_coords[:, 1],
        y=recip_coords[:, 0],
        mode="markers",
        marker=dict(
            color="magenta", size=21, symbol="circle-open", line=dict(width=1.5)
        ),
        hoverinfo="skip",
        showlegend=True,
        name="Dual Traps",
    )

py, px_coords = np.where(perimeter_mask)
if len(py) > 0 and len(px_coords) > 0:
    fig_map.add_shape(
        type="rect",
        x0=px_coords.min(),
        y0=py.min(),
        x1=px_coords.max(),
        y1=py.max(),
        line=dict(color="white", width=1.5, dash="dash"),
        fillcolor="rgba(0,0,0,0)",
        name="Perimeter Bound",
    )

fig_map.add_scatter(
    x=active_df["cx"],
    y=active_df["cy"],
    mode="markers",
    marker=dict(color="rgba(0,0,0,0)", size=16),
    customdata=np.stack(
        (active_df["id"], active_df["cx"], active_df["cy"], active_df["rel_power"]),
        axis=-1,
    ),
    hovertemplate="<b>ID: %{customdata[0]}</b><br>X: %{customdata[1]}<br>Y: %{customdata[2]}<br>Rel Power: %{customdata[3]:.1f}%<extra></extra>",
    showlegend=False,
    name="Artifacts",
)

fig_map.update_layout(
    height=800,
    margin=dict(l=0, r=0, t=30, b=40),
    legend=dict(
        yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0)"
    ),
    coloraxis_showscale=False,
)

with map_container:
    event = st.plotly_chart(
        fig_map, on_select="rerun", selection_mode="points", use_container_width=True
    )

with col_right:
    st.divider()

    selected_id_from_map = int(active_df["id"].iloc[0])
    if event and event.selection.points:
        clicked_trace_idx = event.selection.points[0]["curve_number"]
        if fig_map.data[clicked_trace_idx].name == "Artifacts":
            clicked_idx = event.selection.points[0]["point_index"]
            selected_id_from_map = int(active_df.iloc[clicked_idx]["id"])

    def format_dropdown(item_id):
        row = active_df[active_df["id"] == item_id].iloc[0]
        fuzz_label = " (Inside Perimeter)" if row["is_fuzz"] else " (Ghost Order)"
        return f"Rank #{row['id']} - Rel Power: {row['rel_power']:.2f}%{fuzz_label}"

    selected_id = st.selectbox(
        "Selected Artifact Viewer",
        options=active_df["id"],
        index=active_df["id"].tolist().index(selected_id_from_map),
        format_func=format_dropdown,
    )

    if artifact_stacks is not None and len(artifact_stacks) > 0:
        stack_idx = selected_id - 1
        if stack_idx < len(artifact_stacks):
            current_stack = artifact_stacks[stack_idx]

            if use_log_scale:
                stack_plot = np.log(current_stack + 1e-12)
            else:
                stack_plot = current_stack

            fig_stack = px.imshow(stack_plot, color_continuous_scale="viridis")
            cy_s, cx_s = current_stack.shape[0] // 2, current_stack.shape[1] // 2

            fig_stack.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                coloraxis_showscale=False,
                xaxis_visible=False,
                yaxis_visible=False,
            )
            st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("Artifact stack imagery missing from phase solution HDF5.")
