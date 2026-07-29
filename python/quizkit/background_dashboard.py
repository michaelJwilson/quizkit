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

from quizkit.run_phase_retrieval import PerformanceMetrics

st.set_page_config(page_title="Artifact Analysis", layout="wide")

# ==========================================
# COMMAND LINE ARGUMENTS & AUTO-DISCOVERY
# ==========================================
parser = argparse.ArgumentParser(description="Artifact Analysis Dashboard")
parser.add_argument("--base_dir", type=str, default="./results", help="Base directory to search for runs")
parser.add_argument("--run_hash", type=str, default=None, help="Explicit run hash (overrides auto-discovery)")
parser.add_argument("--solver_hash", type=str, default=None, help="Explicit solver hash (overrides auto-discovery)")

args, _ = parser.parse_known_args()

def get_most_recent_hashes(base_dir: str):
    """Scans the base directory for the most recently modified run and solver hashes."""
    base_path = Path(base_dir)
    fallback_run, fallback_hash = "xyz", "abc1234"
    
    if not base_path.exists():
        return fallback_run, fallback_hash
        
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("run_")]
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

# ==========================================
# DATA LOADING & SIDEBAR
# ==========================================
st.sidebar.header("Run Selection")
run_hash = st.sidebar.text_input("Run Hash", default_run_hash)
solver_hash = st.sidebar.text_input("Solver Hash", default_solver_hash)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def encode_target_traps(target_intensity, threshold_frac=0.0):
    """Extracts trap coordinates inline so we don't rely on the external class."""
    threshold = threshold_frac * target_intensity.max()
    labeled_mask, num_traps = label(target_intensity > threshold)
    slices = find_objects(labeled_mask)
    coords = []
    for s in slices:
        center_y = (s[0].start + s[0].stop) // 2
        center_x = (s[1].start + s[1].stop) // 2
        coords.append((center_y, center_x))
    return labeled_mask, num_traps, np.array(coords)

def extract_background_artifacts(forward_intensity, trap_labels, exclusion_mask=None, exclusion_pad=15, percentile_q=99.9, max_artifacts=100):
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

def generate_latex_table(metrics_dict: dict, method_name: str) -> str:
    metric_names = [k for k in metrics_dict.keys() if k != "trap_powers"]
    rows = []
    for m_name in metric_names:
        escaped_m_name = m_name.replace("_", "\\_")
        val = metrics_dict.get(m_name, np.nan)
        if isinstance(val, float) and not np.isnan(val):
            val_str = f"{val:.2e}" if (val != 0 and (abs(val) < 1e-3 or abs(val) > 1e4)) else f"{val:.4f}"
        else:
            val_str = str(val)
        desc = PerformanceMetrics._DESCRIPTIONS.get(m_name, "")
        rows.append(f"\\texttt{{{escaped_m_name}}} & {val_str} & {desc} \\\\")

    return f"""\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{lccp{{10cm}}}}
\\toprule
\\textbf{{Metric Key}} & \\textbf{{{method_name.capitalize()}}} & \\textbf{{Description}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Computed performance metrics for the {method_name}-optimized SLM phase.}}
\\label{{tab:hologram_metrics}}
\\end{{table}}"""

@st.cache_data
def load_data(base_dir: str, r_hash: str, s_hash: str):
    run_dir = Path(base_dir) / f"run_{r_hash}"
    solver_dir = run_dir / "phase_retrieval" / s_hash
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not solver_dir.exists():
        raise FileNotFoundError(f"Solver directory not found: {solver_dir}")
    
    # 1. Load Configs
    with open(run_dir / "run_config.json", "r") as f:
        run_config = json.load(f)
    with open(solver_dir / "solver_config.json", "r") as f:
        solver_config = json.load(f)
    with open(solver_dir / "performance_metrics.json", "r") as f:
        metrics = json.load(f)
        
    exp_h5 = run_dir / "experiment.h5"
    sol_h5 = solver_dir / f"phase_solution_{s_hash}.h5"
    
    # Data contracts
    required_exp_keys = {"target": "target/target"}
    optional_exp_keys = {
        "perimeter_mask": "trap_array_perimeter_mask/trap_array_perimeter_mask",
        "reciprocal_coords": "reciprocal_coords/reciprocal_coords"
    }
    
    required_sol_keys = {"ff_int": "target/inferred_farfield_intensity"}
    optional_sol_keys = {"artifact_stacks": "background/off_target_intensity_stacks"}
    
    missing_required = []
    loaded_data = {}
    warnings = []

    # 2. Safe loading of experiment.h5
    if not exp_h5.exists():
        missing_required.append(f"File missing entirely: {exp_h5}")
    else:
        with h5py.File(exp_h5, "r") as f:
            for k, path in required_exp_keys.items():
                if path in f:
                    loaded_data[k] = f[path][:]
                else:
                    missing_required.append(f"{exp_h5.name} -> '{path}'")
            for k, path in optional_exp_keys.items():
                if path in f:
                    loaded_data[k] = f[path][:]
                else:
                    loaded_data[k] = None
                    warnings.append(f"Optional dataset '{path}' not found in {exp_h5.name}")
                    
    # 3. Safe loading of phase_solution.h5
    if not sol_h5.exists():
        missing_required.append(f"File missing entirely: {sol_h5}")
    else:
        with h5py.File(sol_h5, "r") as f:
            for k, path in required_sol_keys.items():
                if path in f:
                    loaded_data[k] = f[path][:]
                else:
                    missing_required.append(f"{sol_h5.name} -> '{path}'")
            for k, path in optional_sol_keys.items():
                if path in f:
                    loaded_data[k] = f[path][:]
                else:
                    loaded_data[k] = None
                    warnings.append(f"Optional dataset '{path}' not found in {sol_h5.name}")

    # If any required data is missing, we throw a specific Exception format to catch later
    if missing_required:
        raise ValueError("MISSING_REQUIRED|" + "\n".join(missing_required))

    # 4. Compute derived coordinates dynamically
    target_int = np.abs(loaded_data["target"])**2
    trap_labels, num_traps, trap_coords = encode_target_traps(target_int)
    
    artifacts, residual_int = extract_background_artifacts(
        forward_intensity=loaded_data["ff_int"],
        trap_labels=trap_labels,
        exclusion_pad=15,
        percentile_q=99.9
    )
    
    return run_config, solver_config, metrics, loaded_data, trap_coords, artifacts, residual_int, warnings

# ==========================================
# EXECUTE LOAD & HANDLE ERRORS
# ==========================================
try:
    (
        run_config, 
        solver_config, 
        metrics, 
        loaded_data, 
        trap_coords, 
        artifacts, 
        residual_int,
        warnings
    ) = load_data(args.base_dir, run_hash, solver_hash)
except ValueError as e:
    err_str = str(e)
    if err_str.startswith("MISSING_REQUIRED|"):
        st.error("### Fatal Error: Required Datasets Missing")
        st.write("The following datasets are required for the dashboard but were not found in the HDF5 files:")
        st.code(err_str.split("|")[1], language="text")
        st.info("Check your simulation pipeline to ensure these arrays are being written to the HDF5 files.")
        st.stop()
    else:
        st.error(f"Failed to load data:\n\n{e}")
        st.stop()
except Exception as e:
    st.error(f"Failed to load data:\n\n{e}")
    st.stop()

if warnings:
    for w in warnings:
        st.toast(w, icon="⚠️")

# ==========================================
# SIDEBAR METADATA
# ==========================================
st.sidebar.divider()
st.sidebar.subheader("Run Configuration")
st.sidebar.markdown(f"**Trap Type:** `{run_config.get('trap_config', {}).get('trap_type', 'Unknown')}`")
st.sidebar.markdown(f"**Method:** `{solver_config.get('method', 'Unknown')}`")
st.sidebar.markdown(f"**Backend:** `{solver_config.get('solver_backend', 'slm_suite')}`")
st.sidebar.markdown(f"**Smooth Phase:** `{solver_config.get('smooth_phase', False)}`")

# ==========================================
# BUILD ARTIFACT DATAFRAME
# ==========================================
total_bg_power = np.sum(residual_int)
df_list = []
perimeter_mask = loaded_data.get("perimeter_mask")

for i, art in enumerate(artifacts):
    cx, cy = art["cx"], art["cy"]
    is_fuzz = False
    if perimeter_mask is not None:
        is_fuzz = bool(perimeter_mask[cy, cx])
    
    rel_power = (art["power"] / total_bg_power) * 100
    df_list.append({"id": i + 1, "cx": cx, "cy": cy, "rel_power": rel_power, "is_fuzz": is_fuzz})

df = pd.DataFrame(df_list)

# ==========================================
# LAYOUT & RENDERING
# ==========================================
col_left, col_right = st.columns([3, 1], gap="large")

with col_left:
    st.subheader("Forward intensity")
    map_container = st.container()
    
    if perimeter_mask is not None:
        show_fuzz = st.toggle("Include Array Fuzz (Inside Perimeter)", value=True)
        active_df = df.copy() if show_fuzz else df[~df["is_fuzz"]].copy()
    else:
        st.caption("Perimeter mask missing; showing all artifacts.")
        active_df = df.copy()

if active_df.empty:
    st.warning("No artifacts match the current filters.")
    st.stop()

with col_right:
    st.subheader("Top Artifacts")
    display_df = active_df.head(5)[["id", "cx", "cy", "rel_power", "is_fuzz"]].set_index("id")
    st.dataframe(display_df.style.format({"rel_power": "{:.2f}%"}), use_container_width=True)

# >>>>>>>
fig_map = px.imshow(np.log(loaded_data["ff_int"] + 1e-12), color_continuous_scale="viridis")
fig_map.update_traces(hoverinfo="skip", hovertemplate=None, selector=dict(type="image"))

# 1. 0th Order DC Peak (Gold, 4x bigger -> 56)
slm_h, slm_w = run_config.get("slm_shape", (1200, 1920))
fig_map.add_scatter(
    x=[slm_w / 2.0], y=[slm_h / 2.0],
    mode="markers",
    marker=dict(color="gold", size=32, opacity=0.8, symbol="circle-open", line=dict(width=2.5)),
    hoverinfo="skip", showlegend=True, name="0th Order"
)

# 2. Target Traps (Cyan, 50% bigger -> 21)
fig_map.add_scatter(
    x=trap_coords[:, 1], y=trap_coords[:, 0],
    mode="markers",
    marker=dict(color="cyan", size=21, symbol="circle-open", line=dict(width=1.5)),
    hoverinfo="skip", showlegend=True, name="Traps"
)

# 3. Reciprocal Traps (Magenta, 50% bigger -> 21)
recip_coords = loaded_data.get("reciprocal_coords")
if recip_coords is not None and recip_coords.size > 0:
    fig_map.add_scatter(
        x=recip_coords[:, 1], y=recip_coords[:, 0],
        mode="markers",
        marker=dict(color="magenta", size=21, symbol="circle-open", line=dict(width=1.5)),
        hoverinfo="skip", showlegend=True, name="Reciprocal Traps"
    )

# 4. Perimeter Box & Default Zoom Calculation
if perimeter_mask is not None:
    py, px_coords = np.where(perimeter_mask)
    if len(py) > 0 and len(px_coords) > 0:
        p_xmin, p_xmax = px_coords.min(), px_coords.max()
        p_ymin, p_ymax = py.min(), py.max()
        
        # Draw the boundary rectangle
        fig_map.add_shape(
            type="rect",
            x0=p_xmin, y0=p_ymin, x1=p_xmax, y1=p_ymax,
            line=dict(color="white", width=1.5, dash="dash"),
            fillcolor="rgba(0,0,0,0)", name="Perimeter Bound"
        )
        
        # Calculate centroids and dimensions
        p_w = p_xmax - p_xmin
        p_h = p_ymax - p_ymin
        p_cx = p_xmin + p_w / 2.0
        p_cy = p_ymin + p_h / 2.0
        
        # Determine the maximum dimension to force a perfectly square bounding box
        max_dim = max(p_w, p_h)

# 5. Invisible Artifact Click Targets
fig_map.add_scatter(
    x=active_df["cx"], y=active_df["cy"], mode="markers",
    marker=dict(color="rgba(0,0,0,0)", size=16),
    customdata=np.stack((active_df["id"], active_df["cx"], active_df["cy"], active_df["rel_power"]), axis=-1),
    hovertemplate="<b>ID: %{customdata[0]}</b><br>X: %{customdata[1]}<br>Y: %{customdata[2]}<br>Rel Power: %{customdata[3]:.1f}%<extra></extra>",
    showlegend=False, name="Artifacts"
)

fig_map.update_layout(
    height=800, margin=dict(l=0, r=0, t=30, b=40),
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0.5)"),
    coloraxis_showscale=False,
)

# <<<<<
with map_container:
    event = st.plotly_chart(fig_map, on_select="rerun", selection_mode="points", use_container_width=True)

# ------------------------------------------
# METRICS RENDERING (2nd col_left block)
# ------------------------------------------
with col_left:
    st.subheader("Performance Metrics")
    
    method_name = solver_config.get('method', 'Solver')
    
    # 1. Render a clean table for the Streamlit UI
    metric_names = [k for k in metrics.keys() if k != "trap_powers"]
    ui_rows = []
    
    for m in metric_names:
        val = metrics.get(m, np.nan)
        if isinstance(val, float) and not np.isnan(val):
            val_str = f"{val:.2e}" if (val != 0 and (abs(val) < 1e-3 or abs(val) > 1e4)) else f"{val:.4f}"
        else:
            val_str = str(val)
            
        desc = PerformanceMetrics._DESCRIPTIONS.get(m, "")
        ui_rows.append({
            "Metric Key": m, 
            method_name: val_str, 
            "Description": desc
        })
        
    st.dataframe(pd.DataFrame(ui_rows), use_container_width=True, hide_index=True)

    # 2. Keep the raw LaTeX accessible for your manuscripts
    with st.expander("View LaTeX Source"):
        latex_code = generate_latex_table(metrics, method_name=method_name)
        st.code(latex_code, language="latex")

# ------------------------------------------
# VIEWER RENDERING (2nd col_right block)
# ------------------------------------------
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

    artifact_stacks = loaded_data.get("artifact_stacks")
    if artifact_stacks is not None and len(artifact_stacks) > 0:
        stack_idx = selected_id - 1
        if stack_idx < len(artifact_stacks):
            current_stack = artifact_stacks[stack_idx]
            fig_stack = px.imshow(current_stack, color_continuous_scale="viridis")
            cy_s, cx_s = current_stack.shape[0] // 2, current_stack.shape[1] // 2
            
            fig_stack.add_scatter(
                x=[cx_s], y=[cy_s], mode="markers",
                marker=dict(color="white", size=15, symbol="cross", opacity=0.6),
                hoverinfo="skip", showlegend=False
            )
            fig_stack.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False, xaxis_visible=False, yaxis_visible=False)
            st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("Artifact stack imagery missing from phase solution HDF5.")