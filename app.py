import streamlit as st
import plotly.graph_objects as go
import numpy as np
from engine.optimizer import Pathfinder
from engine.functions import bowl_f, bowl_grad, saddle_f, saddle_grad

st.set_page_config(page_title="Surface Pathfinder", layout="wide")

st.title(" Surface Pathfinder")
st.markdown("Visualize how **Gradient Descent** navigates mathematical landscapes.")

# --- Sidebar Configuration ---
st.sidebar.header("Navigation Settings")
surface_choice = st.sidebar.selectbox("Pick a Terrain", ["Bowl", "Saddle"])
lr = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
iters = st.sidebar.slider("Max Iterations", 10, 500, 100)

# --- Logic Selection ---
if surface_choice == "Bowl":
    f, grad = bowl_f, bowl_grad
    x_range = np.linspace(-5, 5, 50)
else:
    f, grad = saddle_f, saddle_grad
    x_range = np.linspace(-2, 2, 50)

# --- Pathfinding Execution ---
optimizer = Pathfinder(learning_rate=lr, max_iters=iters)
# Start the pathfinder at a specific coordinate
history = optimizer.minimize(start_pos=[4, 3] if surface_choice == "Bowl" else [1.5, 0.1], grad_func=grad)

# --- 3D Visualization (Plotly) ---
X, Y = np.meshgrid(x_range, x_range)
Z = f(X, Y)

fig = go.Figure(data=[
    # 1. The Surface
    go.Surface(x=X, y=Y, z=Z, opacity=0.7, colorscale='Viridis'),
    # 2. The Path (The Red Dots)
    go.Scatter3d(x=history[:,0], y=history[:,1], z=f(history[:,0], history[:,1]),
                 mode='markers+lines', marker=dict(color='red', size=4), name='Descent Path')
])

fig.update_layout(title=f"Pathfinding on {surface_choice}", autosize=True, height=700)
st.plotly_chart(fig, use_container_width=True)
