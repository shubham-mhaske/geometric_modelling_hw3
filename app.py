"""
app.py — Streamlit UI for the Tensor‑Product Surface Lab
Run with: streamlit run app.py
"""
import json
from io import StringIO
import numpy as np
import streamlit as st
import plotly.graph_objects as go


from geometry import (
    preset_surfaces, random_grid, uniform_knot_vector, nonuniform_knot_vector,
    eval_surface, differential, param_domain, export_spec, import_spec
)

def main():
    # Page config - MUST be the first Streamlit command, and only called ONCE
    st.set_page_config(
        page_title="Tensor‑Product Surface Lab",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load presets
    presets = preset_surfaces()

    # Initialize session state ONCE with a default preset
    if 'surface_spec' not in st.session_state:
        st.session_state.surface_spec = presets["Saddle (Bezier)"].copy()

    # Callback to update state when preset is changed
    def update_spec_from_preset():
        # Loads the selected preset into session state
        preset_key = st.session_state.get("preset_selector", "Saddle (Bezier)")
        st.session_state.surface_spec = presets[preset_key].copy()

    # Title in a container with custom styling
    with st.container():
        st.title("Tensor‑Product Surface Lab")
        st.caption("Interactive exploration of Bézier, B-spline, and NURBS surfaces")

    # Organize main content into tabs
    tab1, tab2 = st.tabs(["Surface Viewer", "Analysis"])

    # Simplified sidebar header
    st.sidebar.header("Surface Controls")

    # ---------- Sidebar Controls ----------
    with st.sidebar:
        st.header("Surface Setup")
        
        # Use on_change to update the session state when a new preset is selected
        preset_name = st.selectbox(
            "Preset surface", 
            list(presets.keys()),
            key="preset_selector", # Add a key for the callback
            on_change=update_spec_from_preset, # Add the callback
            help="Choose a starting surface specification to edit"
        )
        
        # --- SESSION STATE FIX ---
        # Load the CURRENT spec from session_state, not the preset list
        spec = st.session_state.surface_spec
        
        # Get the index for the selectbox from the *current* spec's kind
        kind_options = ["Bezier", "B-spline", "NURBS"]
        kind_index = kind_options.index(spec["kind"]) if spec["kind"] in kind_options else 0
        
        kind = st.selectbox(
            "Surface type", 
            kind_options, 
            index=kind_index, 
            help="Switch between analytic/basis representations"
        )
        p = st.slider("Degree p (u-direction)", 1, 5, int(spec["p"]), help="Polynomial degree in u (lower = smoother control)")
        q = st.slider("Degree q (v-direction)", 1, 5, int(spec["q"]), help="Polynomial degree in v")

        m_default = max(p+1, spec["grid"].m)
        n_default = max(q+1, spec["grid"].n)
        m = st.number_input("Control points in u (m)", min_value=p+1, max_value=20, value=m_default, step=1, help="Number of control points along u (topology)")
        n = st.number_input("Control points in v (n)", min_value=q+1, max_value=20, value=n_default, step=1, help="Number of control points along v (topology)")

        with st.expander("Control Net", expanded=False):
            # Fixed defaults for randomization
            amp = 0.5  # Default amplitude for random perturbations
            rational = kind == "NURBS"
            w_mean = 1.0  # Default base weight for NURBS

            if st.button("🎲 Randomize Control Points"):
                g = random_grid(int(m), int(n), amp=amp, seed=None, rational=rational)  # No seed for true randomness
                if rational:  # For NURBS, apply default weight
                    for cp in g.points:
                        cp.w *= w_mean
                spec["grid"] = g
                # --- SESSION STATE FIX ---
                # Save the new grid back to session state immediately
                st.session_state.surface_spec = spec.copy()
                st.rerun() # Rerun to show the change

        # Knot vectors
        with st.expander("Knot Vectors (B‑spline / NURBS)", expanded=False):
            nonuni = st.checkbox("Use non‑uniform knots", value=True, help="Choose non-uniform knot spacing to see localized features")
            if kind == "Bezier":
                U = uniform_knot_vector(int(m), int(p)) # Use uniform for consistency
                V = uniform_knot_vector(int(n), int(q))
                st.info("Bézier surfaces use fixed (clamped) knot vectors.")
            else:
                if nonuni:
                    U = nonuniform_knot_vector(int(m), int(p), seed=None)  # Random knots each time
                    V = nonuniform_knot_vector(int(n), int(q), seed=None)
                else:
                    U = uniform_knot_vector(int(m), int(p))
                    V = uniform_knot_vector(int(n), int(q))
                
                # Visualize knot vectors
                st.write("**Knot Vector Visualization**")
                
                # U knots
                st.write(f"U knots (degree {p}, {len(U)} knots):")
                u_fig = go.Figure()
                u_fig.add_trace(go.Scatter(
                    x=U, y=[0]*len(U), mode='markers+text',
                    text=[f"{x:.2f}" for x in U],
                    textposition="top center",
                    marker=dict(size=10, symbol='line-ns'),
                    name="U knots"
                ))
                u_fig.update_layout(
                    height=100, margin=dict(l=40,r=40,t=30,b=20),
                    showlegend=False,
                    xaxis=dict(range=[-0.1,1.1], showgrid=True),
                    yaxis=dict(visible=False, range=[-1,1])
                )
                st.plotly_chart(u_fig, use_container_width=True)
                
                # V knots
                st.write(f"V knots (degree {q}, {len(V)} knots):")
                v_fig = go.Figure()
                v_fig.add_trace(go.Scatter(
                    x=V, y=[0]*len(V), mode='markers+text',
                    text=[f"{x:.2f}" for x in V],
                    textposition="top center",
                    marker=dict(size=10, symbol='line-ns'),
                    name="V knots"
                ))
                v_fig.update_layout(
                    height=100, margin=dict(l=40,r=40,t=30,b=20),
                    showlegend=False,
                    xaxis=dict(range=[-0.1,1.1], showgrid=True),
                    yaxis=dict(visible=False, range=[-1,1])
                )
                st.plotly_chart(v_fig, use_container_width=True)
                
                # Multiplicity info
                st.write("**Knot multiplicities:**")
                def get_multiplicities(knots):
                    from collections import Counter
                    return Counter([f"{x:.3f}" for x in knots])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("U knots:", dict(get_multiplicities(U)))
                with col2:
                    st.write("V knots:", dict(get_multiplicities(V)))

        # Import / Export
        with st.expander("Import / Export", expanded=False):
            # Clear upload state if requested
            if st.button("Clear Imported Surface"):
                st.session_state.surface_spec = presets["Saddle (Bezier)"].copy() # Reset to default
                st.rerun()
            
            uploaded = st.file_uploader(
                "Import surface JSON",
                type=["json"],
                help="Upload a previously exported surface spec (JSON)",
                key="surface_upload"
            )
            
            if uploaded is not None:
                try:
                    data = json.load(uploaded)
                    imported_spec = import_spec(data)
                    # Validate imported spec
                    if all(key in imported_spec for key in ["kind", "p", "q", "grid"]):
                        spec = imported_spec
                        # --- SESSION STATE FIX ---
                        # Save the imported spec to session state and rerun
                        st.session_state.surface_spec = spec
                        st.success("✅ Surface imported successfully!")
                        st.rerun() # Rerun to load the new spec into all widgets
                    else:
                        st.error("Invalid surface specification format")
                except json.JSONDecodeError:
                    st.error("Invalid JSON file")
                except Exception as e:
                    st.error(f"Import failed: {str(e)}")

        # Sync spec object with sidebar choices
        spec["kind"] = kind
        spec["p"] = int(p); spec["q"] = int(q)
        spec["U"] = U; spec["V"] = V
        if spec["grid"].m != int(m) or spec["grid"].n != int(n):
            # Resize by regenerating random grid to match new topology
            st.warning("Grid size changed. Regenerating control net.")
            spec["grid"] = random_grid(int(m), int(n), amp=float(amp), seed=None, rational=(kind=="NURBS"))

        # Cache the current spec back into session state
        st.session_state.surface_spec = spec.copy()
        
        # Export with unique filename based on content
        spec_json = json.dumps(export_spec(spec), indent=2)
        file_name = f"surface_{hash(spec_json) % 10000:04d}.json"
        st.download_button(
            "💾 Save Surface",
            spec_json,
            file_name=file_name,
            mime="application/json",
            help="Download current surface configuration as JSON"
        )

    # ---------- Sampling parameters ----------
    st.subheader("Sampling & Analysis")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        res_u = st.slider("Surface resolution (u)", 10, 100, 40, 1, key="res_u_main")
    with c2:
        res_v = st.slider("Surface resolution (v)", 10, 100, 40, 1, key="res_v_main")
    with c3:
        hstep = st.number_input("Finite difference step (relative)", value=1e-3, format="%.1e")
    with c4:
        color_mode = st.selectbox("Surface coloring", 
                                 ["Height", "Gaussian Curvature", "Mean Curvature"],
                                 help="Color the surface by Z coordinate or curvature")

    u_dom = param_domain(spec["kind"], spec["p"], spec["U"] or [0,0,1,1])
    v_dom = param_domain(spec["kind"], spec["q"], spec["V"] or [0,0,1,1])

    uc = st.slider("Pick u", float(u_dom[0]), float(u_dom[1]), float(sum(u_dom)/2), step=(u_dom[1]-u_dom[0])/100, key="uc_main")
    vc = st.slider("Pick v", float(v_dom[0]), float(v_dom[1]), float(sum(v_dom)/2), step=(v_dom[1]-v_dom[0])/100, key="vc_main")

    # ---------- Evaluate surface grid and curvature ----------
    u_vals = np.linspace(u_dom[0], u_dom[1], int(res_u))
    v_vals = np.linspace(v_dom[0], v_dom[1], int(res_v))
    X = np.zeros((len(u_vals), len(v_vals)))
    Y = np.zeros((len(u_vals), len(v_vals)))
    Z = np.zeros((len(u_vals), len(v_vals)))
    K = np.zeros((len(u_vals), len(v_vals)))  # Gaussian curvature
    H = np.zeros((len(u_vals), len(v_vals)))  # Mean curvature

    for i, uu in enumerate(u_vals):
        for j, vv in enumerate(v_vals):
            # Surface point
            S = eval_surface(spec, float(uu), float(vv))
            X[i,j], Y[i,j], Z[i,j] = S[0], S[1], S[2]
            
            # Curvature
            diff = differential(spec, float(uu), float(vv), h=float(hstep))
            K[i,j] = diff["K"]
            H[i,j] = diff["H"]

    # Prepare coloring based on selected mode
    if color_mode == "Height":
        color_data = Z
        colorbar_title = "Z coordinate"
    elif color_mode == "Gaussian Curvature":
        color_data = K
        colorbar_title = "Gaussian Curvature (K)"
    else:  # Mean Curvature
        color_data = H
        colorbar_title = "Mean Curvature (H)"

    # Clean up NaN/Inf for better visualization
    color_data = np.nan_to_num(color_data, nan=0.0, posinf=1e3, neginf=-1e3)



    # ---------- Plotly figure ----------
    fig = go.Figure()

    # Surface mesh with coloring by height/curvature
    try:
        vmin = float(np.nanmin(color_data))
        vmax = float(np.nanmax(color_data))
        # For curvature, use symmetric range around 0
        if color_mode != "Height":
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
    except Exception:
        vmin, vmax = None, None

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=color_data,
        colorscale='RdBu' if color_mode != "Height" else 'Viridis',
        cmin=vmin, cmax=vmax,
        opacity=0.95,
        showscale=True,
        name="Surface",
        colorbar=dict(
            title=colorbar_title,
            titleside="right"
        ),
        lighting=dict(ambient=0.6, diffuse=0.7, roughness=0.9)
    ))

    # Control net (points + poly-lines along u and v)
    m = spec["grid"].m; n = spec["grid"].n
    PX = np.zeros((m,n)); PY = np.zeros((m,n)); PZ = np.zeros((m,n))
    W = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            cp = spec["grid"].points[i*n + j]
            PX[i,j], PY[i,j], PZ[i,j] = cp.p.x, cp.p.y, cp.p.z
            W[i,j] = cp.w

    # points (control points) with hover info, weight display, and click handling
    hover_pts = []
    marker_colors = []
    for i in range(m):
        for j in range(n):
            hover_pts.append(f"i={i}, j={j}, w={W[i,j]:.3f}")
            marker_colors.append('crimson')

    # Add control points with click events
    points = go.Scatter3d(
        x=PX.flatten(), y=PY.flatten(), z=PZ.flatten(),
        mode="markers", name="Control Points",
        marker=dict(size=8, color=marker_colors, symbol='circle'),
        hovertext=hover_pts, hoverinfo='text',
        customdata=[(i,j) for i in range(m) for j in range(n)]  # Store indices for click
    )
    fig.add_trace(points)



    # poly-lines along u (blue) and v (orange) with single legend entries
    for j in range(n):
        fig.add_trace(go.Scatter3d(x=PX[:,j], y=PY[:,j], z=PZ[:,j],
                                   mode="lines", line=dict(width=3, color='royalblue', dash='solid'), name="Ctrl poly (u)",
                                   showlegend=(j==0)))
    # poly-lines along v
    for i in range(m):
        fig.add_trace(go.Scatter3d(x=PX[i,:], y=PY[i,:], z=PZ[i,:],
                                   mode="lines", line=dict(width=3, color='darkorange', dash='dash'), name="Ctrl poly (v)",
                                   showlegend=(i==0)))

    # ---------- Differential geometry at (uc, vc) ----------
    diff = differential(spec, float(uc), float(vc), h=float(hstep))
    S = diff["S"]; nvec = diff["normal"]; Su = diff["Su"]; Sv = diff["Sv"]
    d1 = diff["d1"]; d2 = diff["d2"]

    def add_vector(anchor, vec, name):
        A = np.array(anchor); B = A + vec
        fig.add_trace(go.Scatter3d(x=[A[0], B[0]], y=[A[1], B[1]], z=[A[2], B[2]],
                                   mode="lines+markers",
                                   marker=dict(size=3), line=dict(width=6),
                                   name=name))

    # Normal & principal directions
    scale = max(1e-6, np.ptp(Z)) * 0.2
    add_vector(S, nvec*scale, "Normal")
    add_vector(S, d1*scale, "Principal dir 1")
    add_vector(S, d2*scale, "Principal dir 2")

    fig.update_layout(scene=dict(aspectmode="data"),
                      margin=dict(l=0,r=0,t=30,b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=0.02))

    # Configure plot for click events
    fig.update_layout(
        clickmode='event+select',
        dragmode='select',
        hovermode='closest',
        height=600,  # Fixed height for better layout
        margin=dict(l=0, r=0, t=0, b=0),  # Maximize plot area
    )

    with tab1:
        # Left column for 3D view
        col1, col2 = st.columns([7, 3])
        
        with col1:
            # Display plot and capture click events
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                }
            )
        
        with col2:
            st.info("Control point editing is disabled.")



    # ---------- Analysis Tab Content ----------
    with tab2:
        st.subheader("Surface Analysis")
        
        # Parameter selection
        c1, c2 = st.columns(2)
        with c1:
            res_u_analysis = st.slider("Surface resolution (u)", 10, 100, 40, 1, key="res_u_analysis")
            uc_analysis = st.slider("Pick u", float(u_dom[0]), float(u_dom[1]), 
                                  float(sum(u_dom)/2), step=(u_dom[1]-u_dom[0])/100, key="uc_analysis")
        with c2:
            res_v_analysis = st.slider("Surface resolution (v)", 10, 100, 40, 1, key="res_v_analysis")
            vc_analysis = st.slider("Pick v", float(v_dom[0]), float(v_dom[1]), 
                                  float(sum(v_dom)/2), step=(v_dom[1]-v_dom[0])/100, key="vc_analysis")
        
        # Use the sliders from the Analysis tab to recalculate diff
        diff_analysis = differential(spec, float(uc_analysis), float(vc_analysis), h=float(hstep))
        
        # Curvature analysis at current point
        E,F,G,e,f,g = diff_analysis["E"], diff_analysis["F"], diff_analysis["G"], diff_analysis["e"], diff_analysis["f"], diff_analysis["g"]
        K,H,k1,k2 = diff_analysis["K"], diff_analysis["H"], diff_analysis["k1"], diff_analysis["k2"]
        
        st.write("**Analysis at Selected Point**")
        cols = st.columns(3)
        
        with cols[0]:
            st.metric("Gaussian Curvature (K)", f"{K:.3f}")
        with cols[1]:
            st.metric("Mean Curvature (H)", f"{H:.3f}")
        with cols[2]:
            st.metric("Principal Curvatures", f"κ₁={k1:.3f}, κ₂={k2:.3f}")
        
        # Expandable detailed forms
        with st.expander("View Fundamental Forms"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("**First Fundamental Form**")
                st.write({
                    "E": f"{E:.3f}",
                    "F": f"{F:.3f}",
                    "G": f"{G:.3f}"
                })
            with c2:
                st.write("**Second Fundamental Form**")
                st.write({
                    "e": f"{e:.3f}",
                    "f": f"{f:.3f}",
                    "g": f"{g:.3f}"
                })

    st.info("Tip: Switch between Bézier / B‑spline / NURBS, toggle non‑uniform knots, randomize control nets, "
            "and move the (u,v) pickers to inspect normals, principal directions, and curvatures.")

if __name__ == "__main__":
    main()