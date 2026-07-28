import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# Setup page layout
st.set_page_config(page_title="Artifact Analysis", layout="wide")

@st.cache_data
def load_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Load the data exported from your JAX script
try:
    data = load_data("./results/data/artifact_data.pkl")
    residual_int = data['residual_int']
    artifacts = data['artifacts']
    artifact_stacks = data['artifact_stacks']
    slm_shape = data['slm_shape']
    array_shape = data['array_shape']
    array_pitch = data['array_pitch']
    array_center = data['array_center']
except FileNotFoundError:
    st.error("Data file not found. Please run the JAX export script first.")
    st.stop()

# ==========================================
# RECONSTRUCT TARGET GRID
# ==========================================
# Base center (0th order DC term)
center_y, center_x = slm_shape[0] / 2.0, slm_shape[1] / 2.0

if array_center is not None:
    center_x += array_center[0]
    center_y += array_center[1]

Ny, Nx = array_shape
dy, dx = array_pitch

x_offsets = (np.arange(Nx) - (Nx - 1) / 2.0) * dx
y_offsets = (np.arange(Ny) - (Ny - 1) / 2.0) * dy
xx_offsets, yy_offsets = np.meshgrid(x_offsets, y_offsets)

trap_x_coords = center_x + xx_offsets.flatten()
trap_y_coords = center_y + yy_offsets.flatten()

pad = 25
x_min, x_max = trap_x_coords.min() - pad, trap_x_coords.max() + pad
y_min, y_max = trap_y_coords.min() - pad, trap_y_coords.max() + pad


# ==========================================
# BUILD DATAFRAME
# ==========================================
total_bg_power = np.sum(residual_int)
df_list = []
for i, art in enumerate(artifacts):
    cx, cy = art['cx'], art['cy']
    
    is_fuzz = bool((x_min <= cx <= x_max) and (y_min <= cy <= y_max))
    rel_power = (art['power'] / total_bg_power) * 100
    
    df_list.append({
        'id': i + 1,
        'cx': cx,
        'cy': cy,
        'rel_power': rel_power,
        'is_fuzz': is_fuzz
    })
df = pd.DataFrame(df_list)


# ==========================================
# LAYOUT & RENDERING
# ==========================================
# Increased left column ratio for a larger map
col_left, col_right = st.columns([3, 1], gap="large")

with col_left:
    st.subheader("Intensity Plane")
    # Use a placeholder so we can render the toggle below the map, 
    # but still use its value to filter the dataframe first.
    map_container = st.container()
    show_fuzz = st.toggle("Include SLM Fuzz", value=True)
    
# Filter dataframe based on the toggle
if not show_fuzz:
    active_df = df[~df['is_fuzz']].copy()
else:
    active_df = df.copy()
    
if active_df.empty:
    st.warning("No artifacts match the current filters.")
    st.stop()

with col_right:
    st.subheader("Top Artifacts")
    # Table only displays Top 5, but the dropdown below will have all
    display_df = active_df.head(5)[['id', 'cx', 'cy', 'rel_power', 'is_fuzz']].set_index('id')
    st.dataframe(
        display_df.style.format({'rel_power': '{:.2f}%'}), 
        use_container_width=True
    )

# --- LEFT COLUMN: Construct Map ---
fig_map = px.imshow(
    np.log(residual_int + 1e-12), 
    color_continuous_scale='viridis'
)

# Disable the native image hover to stop "color" and raw coords from showing up
fig_map.update_traces(hoverinfo='skip', hovertemplate=None, selector=dict(type='image'))

# 1. Overlay the Target Extent Bounding Box
fig_map.add_shape(
    type="rect",
    x0=x_min, y0=y_min, x1=x_max, y1=y_max,
    line=dict(color="cyan", width=1.5, dash="dash"),
    fillcolor="rgba(0,0,0,0)",
    name="Trap Extent"
)

# 2. 0th Order Bounding Circle (Replaces Cyan Cross)
circle_radius = 40
fig_map.add_shape(
    type="circle",
    x0=center_x - circle_radius, y0=center_y - circle_radius, 
    x1=center_x + circle_radius, y1=center_y + circle_radius,
    line=dict(color="cyan", width=1.5, dash="dot"),
    fillcolor="rgba(0,0,0,0)",
    name="0th Order"
)

# 3. Overlay the Reconstructed Trap Circles
fig_map.add_scatter(
    x=trap_x_coords, y=trap_y_coords,
    mode='markers',
    marker=dict(
        color='rgba(255, 255, 255, 0.7)',
        size=14,
        symbol='circle-open',
        line=dict(width=1.5)
    ),
    hoverinfo='skip',
    showlegend=False,
    name="Ideal Traps"
)

# 4. Invisible Artifact Markers (for hover and click detection only)
fig_map.add_scatter(
    x=active_df['cx'], 
    y=active_df['cy'], 
    mode='markers', 
    marker=dict(color='rgba(0,0,0,0)', size=16), # Fully transparent
    customdata=np.stack((active_df['id'], active_df['cx'], active_df['cy'], active_df['rel_power']), axis=-1),
    hovertemplate="<b>ID: %{customdata[0]}</b><br>X: %{customdata[1]}<br>Y: %{customdata[2]}<br>Rel Power: %{customdata[3]:.1f}%<extra></extra>",
    showlegend=False,
    name="Artifacts"
)

fig_map.update_layout(
    height=900, # Increased height to make the map ~1.5x larger
    margin=dict(l=0, r=0, t=10, b=40), 
    coloraxis_showscale=False,
    xaxis=dict(showticklabels=True, title="Pixels (W)"),
    yaxis=dict(showticklabels=True, title="Pixels (H)")
)

with map_container:
    event = st.plotly_chart(fig_map, on_select="rerun", selection_mode="points", use_container_width=True)


# --- RIGHT COLUMN: Viewer & Dropdown ---
with col_right:
    st.divider()
    
    # Logic to determine selected artifact
    default_id = int(active_df['id'].iloc[0])
    selected_id_from_map = default_id
    
    # Sync click event to dropdown index
    if event and event.selection.points:
        clicked_trace_idx = event.selection.points[0]["curve_number"]
        clicked_trace_name = fig_map.data[clicked_trace_idx].name
        
        if clicked_trace_name == "Artifacts":
            clicked_idx = event.selection.points[0]["point_index"]
            selected_id_from_map = int(active_df.iloc[clicked_idx]['id'])

    def format_dropdown(item_id):
        row = active_df[active_df['id'] == item_id].iloc[0]
        fuzz_label = " (SLM Fuzz)" if row['is_fuzz'] else ""
        return f"Rank #{row['id']} - Rel Power: {row['rel_power']:.2f}%{fuzz_label}"
    
    # Dropdown acts as the title/selector for the viewer section
    selected_id = st.selectbox(
        "Selected Artifact", 
        options=active_df['id'], 
        index=active_df['id'].tolist().index(selected_id_from_map),
        format_func=format_dropdown
    )
    
    # Render Crop Image
    stack_idx = selected_id - 1 
    current_stack = artifact_stacks[stack_idx]
    
    fig_stack = px.imshow(current_stack, color_continuous_scale='viridis')
    
    center_y_stack, center_x_stack = current_stack.shape[0] // 2, current_stack.shape[1] // 2
    fig_stack.add_scatter(
        x=[center_x_stack], y=[center_y_stack], 
        mode='markers', 
        marker=dict(color='white', size=15, symbol='cross', opacity=0.6),
        hoverinfo='skip',
        showlegend=False
    )
    
    fig_stack.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), 
        coloraxis_showscale=False,
        xaxis=dict(showticklabels=False, title=None),
        yaxis=dict(showticklabels=False, title=None)
    )
    
    st.plotly_chart(fig_stack, use_container_width=True)