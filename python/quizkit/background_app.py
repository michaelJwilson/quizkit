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
except FileNotFoundError:
    st.error("Data file not found. Please run the JAX export script first.")
    st.stop()

# Build DataFrame
total_bg_power = np.sum(residual_int)
df_list = []
for i, art in enumerate(artifacts):
    rel_power = (art['power'] / total_bg_power) * 100
    df_list.append({
        'id': i + 1,
        'cx': art['cx'],
        'cy': art['cy'],
        'rel_power': rel_power,
        'is_fuzz': art.get('is_fuzz', False)
    })
df = pd.DataFrame(df_list)

# Layout: 2/3 Left (Map), 1/3 Right (Controls & Viewer)
col_left, col_right = st.columns([2, 1], gap="large")

with col_right:
    st.subheader("Controls & Top Artifacts")
    
    # Toggle for SLM Fuzz
    show_fuzz = st.toggle("Show 'SLM Fuzz' (artifacts within trap extent)", value=True)
    
    # Filter dataframe based on toggle
    if not show_fuzz:
        active_df = df[~df['is_fuzz']].copy()
    else:
        active_df = df.copy()
        
    # Restrict to Top 5 hits
    active_df = active_df.head(5)
    
    if active_df.empty:
        st.warning("No artifacts match the current filters.")
        st.stop()
        
    # Display the top hits table at the top right
    display_df = active_df[['id', 'cx', 'cy', 'rel_power', 'is_fuzz']].set_index('id')
    st.dataframe(
        display_df.style.format({'rel_power': '{:.2f}%'}), 
        use_container_width=True
    )

with col_left:
    # --- LEFT COLUMN: Full Width Map ---
    # Render the log intensity map with Viridis
    fig_map = px.imshow(
        np.log(residual_int + 1e-12), 
        color_continuous_scale='viridis'
    )
    
    # Overlay the artifact coordinates
    fig_map.add_scatter(
        x=active_df['cx'], 
        y=active_df['cy'], 
        mode='markers', 
        marker=dict(color='red', size=12, symbol='x'),
        customdata=np.stack((active_df['id'], active_df['cx'], active_df['cy'], active_df['rel_power']), axis=-1),
        hovertemplate="<b>ID: %{customdata[0]}</b><br>X: %{customdata[1]}<br>Y: %{customdata[2]}<br>Rel Power: %{customdata[3]:.1f}%<extra></extra>"
    )
    
    # Strip whitespace, left-align, remove axes labels for a clean image
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), 
        coloraxis_showscale=False,
        xaxis=dict(showticklabels=False, title=None),
        yaxis=dict(showticklabels=False, title=None)
    )

    # Capture click events from the map
    event = st.plotly_chart(fig_map, on_select="rerun", selection_mode="points", use_container_width=True)

with col_right:
    # --- BOTTOM RIGHT: Viewer & Dropdown ---
    st.divider()
    st.subheader("Artifact Viewer")
    
    # Logic to determine selected artifact
    # Default to the first available ID in the filtered list
    default_id = int(active_df['id'].iloc[0])
    selected_id_from_map = default_id
    
    # If the user clicked a point on the map, sync it
    if event and event.selection.points:
        clicked_idx = event.selection.points[0]["point_index"]
        selected_id_from_map = int(active_df.iloc[clicked_idx]['id'])

    # Format dropdown labels nicely
    def format_dropdown(item_id):
        row = active_df[active_df['id'] == item_id].iloc[0]
        fuzz_label = " (SLM Fuzz)" if row['is_fuzz'] else ""
        return f"Rank #{row['id']} - {row['rel_power']:.1f}%{fuzz_label}"
    
    # Render the stack crop for the selected artifact
    stack_idx = selected_id_from_map - 1 # IDs are 1-indexed, lists are 0-indexed
    current_stack = artifact_stacks[stack_idx]
    
    fig_stack = px.imshow(current_stack, color_continuous_scale='viridis')
    
    # Add a crosshair to the center of the stack
    center_y, center_x = current_stack.shape[0] // 2, current_stack.shape[1] // 2
    fig_stack.add_scatter(
        x=[center_x], y=[center_y], 
        mode='markers', 
        marker=dict(color='white', size=15, symbol='cross', opacity=0.6),
        hoverinfo='skip'
    )
    
    fig_stack.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), 
        coloraxis_showscale=False,
        xaxis=dict(showticklabels=False, title=None),
        yaxis=dict(showticklabels=False, title=None)
    )
    
    # Display the plot
    st.plotly_chart(fig_stack, use_container_width=True)
    
    # Dropdown positioned below the viewer
    selected_id = st.selectbox(
        "Select Artifact:", 
        options=active_df['id'], 
        index=active_df['id'].tolist().index(selected_id_from_map),
        format_func=format_dropdown
    )