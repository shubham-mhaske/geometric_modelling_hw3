import json
from io import StringIO
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from collections import Counter

# Import core geometry functions and data structures
from geometry import (
    preset_surfaces, random_grid, uniform_knot_vector, nonuniform_knot_vector,
    eval_surface, differential, param_domain, export_spec, import_spec,
    Vec3, ControlPoint
)


# --- Helper Functions for Streamlit Callbacks and UI ---

def get_presets_for_kind(presets: dict, kind: str) -> dict:
    """
    Filters the dictionary of preset surfaces to include only those of a specific kind.

    Args:
        presets: A dictionary where keys are preset names and values are surface specifications.
        kind: The desired surface kind (e.g., "Bezier", "B-spline", "NURBS").

    Returns:
        A new dictionary containing only the presets matching the specified kind.
    """
    return {k: v for k, v in presets.items() if v["kind"] == kind}


def update_kind_and_reload_preset(presets: dict):
    """
    Callback function triggered when the main 'kind' selector changes.

    This function updates the session state to load the first available preset
    of the newly selected surface kind.

    Args:
        presets: The full dictionary of all available preset surfaces.
    """
    new_kind = st.session_state.kind_selector  # Get the new kind from the selector
    # Find the key of the first preset that matches the new kind
    first_matching_preset = next(
        (key for key, spec in presets.items() if spec["kind"] == new_kind),
        "Saddle (Bezier)"  # Fallback to a default if no matching preset is found
    )
    # Update the surface specification in session state with the chosen preset
    st.session_state.surface_spec = presets[first_matching_preset].copy()


def update_spec_from_preset(presets: dict):
    """
    Callback function triggered when the 'preset' selector changes.

    This function updates the session state with the specification of the
    newly selected preset surface.

    Args:
        presets: The full dictionary of all available preset surfaces.
    """
    preset_key = st.session_state.preset_selector
    if preset_key in presets:
        st.session_state.surface_spec = presets[preset_key].copy()


def add_vector(fig: go.Figure, anchor: np.ndarray, vec: np.ndarray, name: str,
               color: str = 'black', dash: str = 'solid', width: int = 6):
    """
    Adds a 3D vector trace to a Plotly figure.

    Args:
        fig: The Plotly figure object to add the trace to.
        anchor: A NumPy array (x, y, z) representing the starting point of the vector.
        vec: A NumPy array (dx, dy, dz) representing the vector direction and magnitude.
        name: The name of the vector trace for the legend.
        color: The color of the vector line.
        dash: The dash style of the vector line (e.g., 'solid', 'dash', 'dot').
        width: The width of the vector line.
    """
    A = np.array(anchor)
    B = A + vec
    fig.add_trace(go.Scatter3d(
        x=[A[0], B[0]], y=[A[1], B[1]], z=[A[2], B[2]],
        mode="lines+markers",
        marker=dict(size=3, color=color),
        line=dict(width=width, color=color, dash=dash),
        name=name
    ))


# --- Main Streamlit Application Logic ---

def main():
    """
    Main function to run the Streamlit Tensor-Product Surface Lab application.

    This function sets up the UI, handles user interactions, performs surface
    calculations, and visualizes the results.
    """
    # Page configuration - MUST be the first Streamlit command and called ONCE
    st.set_page_config(
        page_title="Tensor-Product Surface Lab",
        layout="wide",  # Use a wide layout for better visualization space
        initial_sidebar_state="collapsed"  # Sidebar starts collapsed
    )

    # Load predefined surface presets from the geometry module
    presets = preset_surfaces()

    # Initialize session state ONCE with a default preset if not already set
    if 'surface_spec' not in st.session_state:
        st.session_state.surface_spec = presets["Saddle (Bezier)"].copy()

    # --- Application Title ---
    st.title("Tensor-Product Surface Lab")

    # --- Top: Surface Kind Selector ---
    spec = st.session_state.surface_spec
    current_kind = spec["kind"]
    kind_options = ["Bezier", "B-spline", "NURBS"]
    # Determine the current index for the selectbox
    kind_index = kind_options.index(current_kind)

    st.selectbox(
        "Select Surface Type",
        kind_options,
        index=kind_index,
        key="kind_selector",
        on_change=update_kind_and_reload_preset,  # Callback to load a default preset
        args=(presets,),
        help="This will reload the configuration with a default preset for the selected type."
    )

    # --- Main Layout: Visualization on Left, Configuration on Right ---
    # Create two columns with a 2:1 width ratio
    col1, col2 = st.columns([2, 1])

    # --- Right Column: Configuration Panel ---
    with col2:
        st.header("Configuration")

        # --- Presets Section ---
        with st.container(border=True):
            st.subheader("Presets")
            # Filter presets based on the currently selected surface kind
            kind_presets = get_presets_for_kind(presets, current_kind)
            preset_keys = list(kind_presets.keys())

            # Try to find the index of the current preset in the filtered list
            try:
                current_preset_name = st.session_state.get("preset_selector", "Saddle (Bezier)")
                if current_preset_name not in preset_keys:
                    current_preset_name = preset_keys[0]  # Default to first if mismatched
                preset_index = preset_keys.index(current_preset_name)
            except Exception:
                preset_index = 0  # Fallback if preset not found

            st.selectbox(
                "Preset surface",
                preset_keys,
                index=preset_index,
                key="preset_selector",
                on_change=update_spec_from_preset,  # Callback to load the selected preset
                args=(presets,),
                help="Choose a starting surface specification to edit"
            )

        # --- Parameters Section (Degrees and Control Point Counts) ---
        with st.container(border=True):
            st.subheader("Parameters")
            c1_params, c2_params = st.columns(2)
            with c1_params:
                p = st.slider("Degree p (u-direction)", 1, 5, int(spec["p"]), help="Polynomial degree in u")
            with c2_params:
                q = st.slider("Degree q (v-direction)", 1, 5, int(spec["q"]), help="Polynomial degree in v")

            c1_counts, c2_counts = st.columns(2)
            with c1_counts:
                # Ensure minimum control points based on degree (m >= p+1)
                m_default = max(p + 1, spec["grid"].m)
                m = st.number_input("Control points in u (m)", min_value=p + 1, max_value=20, value=m_default, step=1,
                                    help="Number of control points along u")
            with c2_counts:
                # Ensure minimum control points based on degree (n >= q+1)
                n_default = max(q + 1, spec["grid"].n)
                n = st.number_input("Control points in v (n)", min_value=q + 1, max_value=20, value=n_default, step=1,
                                    help="Number of control points along v")

        # --- Control Net Editor ---
        with st.expander("Control Net", expanded=False):
            st.subheader("Control Point Editor")
            grid = spec["grid"]
            rational = current_kind == "NURBS"

            # 1. Convert ControlGrid object to a list of dictionaries for st.data_editor
            grid_data = []
            for i in range(grid.m):
                for j in range(grid.n):
                    cp = grid.at(i, j)
                    grid_data.append({
                        "i": i, "j": j,
                        "x": cp.p.x, "y": cp.p.y, "z": cp.p.z,
                        "w": cp.w
                    })

            # Define column configuration for the data editor
            column_config = {
                "i": st.column_config.NumberColumn(disabled=True),  # Index columns are read-only
                "j": st.column_config.NumberColumn(disabled=True),
                "x": st.column_config.NumberColumn(format="%.3f"),
                "y": st.column_config.NumberColumn(format="%.3f"),
                "z": st.column_config.NumberColumn(format="%.3f"),
                "w": st.column_config.NumberColumn(
                    format="%.3f", disabled=(not rational)  # Weight is editable only for NURBS
                )
            }

            # Display the data editor
            edited_data = st.data_editor(
                grid_data,
                column_config=column_config,
                num_rows="fixed",  # Prevent adding/deleting rows
                use_container_width=True,
                key="control_grid_editor"
            )

            # 3. If data was changed by the user, update the session state
            if edited_data != grid_data:
                new_points = []
                for row in edited_data:
                    p_vec = Vec3(row["x"], row["y"], row["z"])
                    w = row["w"] if rational else 1.0  # Ensure weight is 1.0 for non-rational surfaces
                    new_points.append(ControlPoint(p_vec, w))

                spec["grid"].points = new_points
                st.session_state.surface_spec = spec
                st.rerun()  # Rerun the app to reflect changes

        # Initialize U and V from the spec *before* conditional blocks
        # This guarantees they always have a value, preventing UnboundLocalError
        U = spec.get("U", [])
        V = spec.get("V", [])

        # --- Knot Vectors Section (Conditional for B-spline/NURBS) ---
        # This expander is always rendered to maintain stable UI layout
        with st.expander("Knot Vectors", expanded=False):
            st.subheader("Knot Vector Editor")
            if current_kind == "Bezier":
                st.info("Knot vectors are fixed for Bézier surfaces and determined by the degree. They are implicitly [0,0,...,0,1,1,...,1].")
                # Still assign U and V so they are defined for calculations
                U = uniform_knot_vector(int(m), int(p))
                V = uniform_knot_vector(int(n), int(q))

            if current_kind in ["B-spline", "NURBS"]:
                # Use sub-columns for U and V knot editors
                knot_col1, knot_col2 = st.columns(2)

                # --- U Knots Editor ---
                with knot_col1:
                    expected_len_u = m + p + 1
                    st.write(f"**U Knots (m+p+1 = {expected_len_u})**")

                    u_knots = spec.get("U", [])

                    # Regenerate default U knot vector if counts don't match
                    if len(u_knots) != expected_len_u:
                        u_knots = uniform_knot_vector(m, p)  # Regenerate uniform default
                        spec["U"] = u_knots  # Save to state
                        st.session_state.surface_spec = spec
                        # No rerun here, just populate the editor with the new default
                        # A rerun will happen if the user edits the data

                    u_knot_data_in = [{"index": i, "knot": u_knots[i]} for i in range(len(u_knots))]

                    u_knot_data_out = st.data_editor(
                        u_knot_data_in,
                        column_config={
                            "index": st.column_config.NumberColumn(disabled=True),
                            "knot": st.column_config.NumberColumn(format="%.3f")
                        },
                        num_rows="fixed",  # Prevent adding/deleting rows
                        key="u_knot_editor",
                        use_container_width=True
                    )

                    new_U = [row["knot"] for row in u_knot_data_out]
                    if new_U != u_knots:
                        # Validate non-decreasing property
                        if not all(new_U[i] <= new_U[i + 1] for i in range(len(new_U) - 1)):
                            st.error("Knot vector must be non-decreasing.")
                        else:
                            spec["U"] = new_U
                            st.session_state.surface_spec = spec
                            st.rerun()  # Rerun to update surface with new knots
                    else:
                        U = u_knots  # Use the current knots if no changes or invalid

                # --- V Knots Editor ---
                with knot_col2:
                    expected_len_v = n + q + 1
                    st.write(f"**V Knots (n+q+1 = {expected_len_v})**")

                    v_knots = spec.get("V", [])

                    # Regenerate default V knot vector if counts don't match
                    if len(v_knots) != expected_len_v:
                        v_knots = uniform_knot_vector(n, q)  # Regenerate uniform default
                        spec["V"] = v_knots  # Save to state
                        st.session_state.surface_spec = spec
                        # No rerun here, just populate the editor with the new default

                    v_knot_data_in = [{"index": i, "knot": v_knots[i]} for i in range(len(v_knots))]

                    v_knot_data_out = st.data_editor(
                        v_knot_data_in,
                        column_config={
                            "index": st.column_config.NumberColumn(disabled=True),
                            "knot": st.column_config.NumberColumn(format="%.3f")
                        },
                        num_rows="fixed",  # Prevent adding/deleting rows
                        key="v_knot_editor",
                        use_container_width=True
                    )

                    new_V = [row["knot"] for row in v_knot_data_out]
                    if new_V != v_knots:
                        # Validate non-decreasing property
                        if not all(new_V[i] <= new_V[i + 1] for i in range(len(new_V) - 1)):
                            st.error("Knot vector must be non-decreasing.")
                        else:
                            spec["V"] = new_V
                            st.session_state.surface_spec = spec
                            st.rerun()  # Rerun to update surface with new knots
                    else:
                        V = v_knots  # Use the current knots if no changes or invalid

        # --- Sampling & Visualization Settings ---
        with st.expander("Sampling & Visualization"):
            st.subheader("Sampling Parameters")
            res_u = st.slider("Surface resolution (u)", 10, 100, 40, 1,
                              help="Number of samples along the u-parameter for surface rendering.")
            res_v = st.slider("Surface resolution (v)", 10, 100, 40, 1,
                              help="Number of samples along the v-parameter for surface rendering.")
            hstep = st.number_input("Finite difference step (h)", value=1e-3, format="%.1e",
                                    help="Step size for numerical differentiation to calculate curvatures.")
            color_mode = st.selectbox("Surface coloring", ["Height", "Gaussian Curvature", "Mean Curvature"],
                                      help="Choose how the surface is colored.")

            st.markdown("---")
            st.subheader("Point for Analysis")
            # Determine the valid parameter domain for the sliders
            u_dom = param_domain(current_kind, p, spec["U"] or [0, 0, 1, 1])
            v_dom = param_domain(current_kind, q, spec["V"] or [0, 0, 1, 1])

            # Sliders to pick a specific (u,v) point for detailed analysis
            uc = st.slider("Pick u", float(u_dom[0]), float(u_dom[1]), float(sum(u_dom) / 2),
                           step=(u_dom[1] - u_dom[0]) / 100,
                           help="Select a u-parameter value for point-specific analysis.")
            vc = st.slider("Pick v", float(v_dom[0]), float(v_dom[1]), float(sum(v_dom) / 2),
                           step=(v_dom[1] - v_dom[0]) / 100,
                           help="Select a v-parameter value for point-specific analysis.")

        # --- Import / Export / Reset Section ---
        with st.expander("Import / Export / Reset"):
            st.subheader("Surface Management")
            uploaded = st.file_uploader("Import surface JSON", type=["json"], key="surface_upload",
                                        help="Upload a JSON file containing a surface specification.")
            if uploaded is not None:
                try:
                    data = json.load(uploaded)
                    imported_spec = import_spec(data)
                    # Basic validation of imported spec
                    if all(key in imported_spec for key in ["kind", "p", "q", "grid"]):
                        st.session_state.surface_spec = imported_spec
                        st.success("✅ Surface imported successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid surface specification format in the uploaded file.")
                except Exception as e:
                    st.error(f"Import failed: {str(e)}")

            spec_json = json.dumps(export_spec(spec), indent=2)
            file_name = f"surface_{hash(spec_json) % 10000:04d}.json"
            st.download_button("💾 Save Surface", spec_json, file_name=file_name, mime="application/json",
                               help="Download the current surface specification as a JSON file.")

            st.write("---")  # Separator

            c1_actions, c2_actions = st.columns(2)
            with c1_actions:
                if st.button("🔄 Reset to Default", use_container_width=True,
                             help="Reset the surface to the default 'Saddle (Bezier)' preset."):
                    st.session_state.surface_spec = presets["Saddle (Bezier)"].copy()
                    st.rerun()
            with c2_actions:
                if st.button("🎲 Randomize Params", use_container_width=True,
                             help="Generate a new random surface with random degrees, control point counts, and grid."):
                    rng = np.random.default_rng()

                    # 1. Get the CURRENT kind, don't change it.
                    current_kind_for_random = st.session_state.surface_spec["kind"]

                    # 2. Randomize parameters (and cast to int)
                    new_p = int(rng.integers(1, 6))  # Degree p from 1 to 5
                    new_q = int(rng.integers(1, 6))  # Degree q from 1 to 5
                    new_m = int(rng.integers(new_p + 1, 11))  # Control points m from p+1 to 10
                    new_n = int(rng.integers(new_q + 1, 11))  # Control points n from q+1 to 10

                    # 3. Generate knots based on the current kind
                    if current_kind_for_random == "Bezier":
                        new_U = uniform_knot_vector(new_m, new_p)
                        new_V = uniform_knot_vector(new_n, new_q)
                    else:
                        # Random non-uniform for B-spline and NURBS
                        new_U = nonuniform_knot_vector(new_m, new_p, seed=int(rng.integers(1, 1000)))
                        new_V = nonuniform_knot_vector(new_n, new_q, seed=int(rng.integers(1, 1000)))

                    # 4. Generate grid
                    new_grid = random_grid(new_m, new_n, amp=0.5,
                                           rational=(current_kind_for_random == "NURBS"),
                                           seed=int(rng.integers(1, 1000)))

                    # 5. Build and save the new spec, keeping the kind
                    st.session_state.surface_spec = {
                        "kind": current_kind_for_random,
                        "p": new_p, "q": new_q,
                        "U": new_U, "V": new_V,
                        "grid": new_grid
                    }
                    st.rerun()

        # --- Final Synchronization and State Update ---
        # Ensure the spec in session state reflects the current UI selections
        spec["kind"] = current_kind
        spec["p"] = int(p)
        spec["q"] = int(q)
        spec["U"] = U
        spec["V"] = V

        # If grid dimensions (m, n) changed, regenerate the control net
        if spec["grid"].m != int(m) or spec["grid"].n != int(n):
            st.warning("Grid size changed. Regenerating control net with random points.")
            spec["grid"] = random_grid(int(m), int(n), amp=0.5, seed=None, rational=(current_kind == "NURBS"))

        st.session_state.surface_spec = spec.copy()


    # --- Calculations Section (Performed after all UI inputs are processed) ---
    # Calculate parameter values for surface evaluation
    u_vals = np.linspace(u_dom[0], u_dom[1], int(res_u))
    v_vals = np.linspace(v_dom[0], v_dom[1], int(res_v))

    # Initialize arrays to store surface points and curvature data
    X, Y, Z = np.zeros((len(u_vals), len(v_vals))), np.zeros((len(u_vals), len(v_vals))), np.zeros((len(u_vals), len(v_vals)))
    K, H = np.zeros((len(u_vals), len(v_vals))), np.zeros((len(u_vals), len(v_vals)))

    # Evaluate surface and calculate curvatures at each grid point
    for i, uu in enumerate(u_vals):
        for j, vv in enumerate(v_vals):
            S_grid = eval_surface(spec, float(uu), float(vv))
            X[i, j], Y[i, j], Z[i, j] = S_grid[0], S_grid[1], S_grid[2]
            diff_grid = differential(spec, float(uu), float(vv), h=float(hstep))
            K[i, j], H[i, j] = diff_grid["K"], diff_grid["H"]

    # Calculate differential properties at the user-selected point (uc, vc)
    diff_at_point = differential(spec, float(uc), float(vc), h=float(hstep))

    # Prepare color data for surface visualization based on user selection
    if color_mode == "Height":
        color_data, colorbar_title = Z, "Z coordinate"
    elif color_mode == "Gaussian Curvature":
        color_data, colorbar_title = K, "Gaussian Curvature (K)"
    else:  # Mean Curvature
        color_data, colorbar_title = H, "Mean Curvature (H)"
    # Handle potential NaN/Inf values in curvature data for Plotly
    color_data = np.nan_to_num(color_data, nan=0.0, posinf=1e3, neginf=-1e3)


    # --- Left Column: 3D Surface Visualization ---
    with col1:
        st.header("Visualization")

        fig = go.Figure()
        # Determine color scale range for Plotly surface
        try:
            vmin, vmax = float(np.nanmin(color_data)), float(np.nanmax(color_data))
            if color_mode != "Height":
                # For curvatures, center the color scale around zero
                vmax = max(abs(vmin), abs(vmax))
                vmin = -vmax
        except Exception:
            vmin, vmax = None, None  # Fallback if min/max calculation fails

        # Add the main surface trace
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z, surfacecolor=color_data,
            colorscale='RdBu' if color_mode != "Height" else 'Viridis',  # Diverging for curvature, sequential for height
            cmin=vmin, cmax=vmax, opacity=0.95, showscale=True, name="Surface",
            colorbar=dict(title=colorbar_title, titleside="right"),
            lighting=dict(ambient=0.6, diffuse=0.7, roughness=0.9)
        ))

        # Extract control point positions and weights for visualization
        m_grid, n_grid = spec["grid"].m, spec["grid"].n
        PX, PY, PZ, W = np.zeros((m_grid, n_grid)), np.zeros((m_grid, n_grid)), np.zeros((m_grid, n_grid)), np.zeros((m_grid, n_grid))
        for i in range(m_grid):
            for j in range(n_grid):
                cp = spec["grid"].points[i * n_grid + j]
                PX[i, j], PY[i, j], PZ[i, j], W[i, j] = cp.p.x, cp.p.y, cp.p.z, cp.w

        # Add control points as markers
        hover_pts = [f"i={i}, j={j}, w={W[i, j]:.3f}" for i in range(m_grid) for j in range(n_grid)]
        fig.add_trace(go.Scatter3d(
            x=PX.flatten(), y=PY.flatten(), z=PZ.flatten(),
            mode="markers", name="Control Points",
            marker=dict(size=8, color='crimson', symbol='circle'),
            hovertext=hover_pts, hoverinfo='text'
        ))

        # Add control polygons (lines connecting control points)
        for j in range(n_grid):
            fig.add_trace(go.Scatter3d(x=PX[:, j], y=PY[:, j], z=PZ[:, j], mode="lines",
                                       line=dict(width=3, color='royalblue'), name="Ctrl poly (u)",
                                       showlegend=(j == 0)))  # Show legend only once
        for i in range(m_grid):
            fig.add_trace(go.Scatter3d(x=PX[i, :], y=PY[i, :], z=PZ[i, :], mode="lines",
                                       line=dict(width=3, color='darkorange', dash='dash'), name="Ctrl poly (v)",
                                       showlegend=(i == 0)))  # Show legend only once

        # Add normal and principal direction vectors at the selected point (uc, vc)
        S_point, nvec, d1, d2 = diff_at_point["S"], diff_at_point["normal"], diff_at_point["d1"], diff_at_point["d2"]

        # Scale vectors for better visibility
        scale = max(1e-6, np.ptp(Z)) * 0.25
        add_vector(fig, S_point, nvec * scale, "Normal", color='blue', width=8)
        add_vector(fig, S_point, d1 * scale, "Principal Dir 1 (k₁)", color='green')
        add_vector(fig, S_point, d2 * scale, "Principal Dir 2 (k₂)", color='purple', dash='dot')

        # Configure Plotly layout
        fig.update_layout(
            scene=dict(aspectmode="data"),  # Maintain aspect ratio for 3D data
            margin=dict(l=0, r=0, t=0, b=0),  # Reduce margins
            legend=dict(orientation="h", yanchor="bottom", y=0.02),  # Horizontal legend at bottom
            hovermode='closest',  # Show hover info for closest point
            height=650  # Fixed height for the plot
        )

        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

    # --- Bottom: Analysis Container ---
    st.markdown("---")  # Add a horizontal separator
    with st.container(border=True):
        st.header("Analysis")
        st.write(f"Displaying analysis for the selected point: **(u, v) = ({uc:.3f}, {vc:.3f})**")

        # Extract differential properties at the selected point
        E, F, G = diff_at_point["E"], diff_at_point["F"], diff_at_point["G"]
        e, f, g = diff_at_point["e"], diff_at_point["f"], diff_at_point["g"]
        K_point, H_point = diff_at_point["K"], diff_at_point["H"]
        k1, k2 = diff_at_point["k1"], diff_at_point["k2"]

        st.write("**Curvature at Selected Point**")
        cols_curvature = st.columns(3)
        cols_curvature[0].metric("Gaussian Curvature (K)", f"{K_point:.3f}")
        cols_curvature[1].metric("Mean Curvature (H)", f"{H_point:.3f}")
        cols_curvature[2].metric("Principal Curvatures", f"κ₁={k1:.3f}, κ₂={k2:.3f}")

        with st.expander("View Fundamental Forms"):
            st.write("**First Fundamental Form**")
            cols_ff = st.columns(3)
            cols_ff[0].metric("E", f"{E:.3f}")
            cols_ff[1].metric("F", f"{F:.3f}")
            cols_ff[2].metric("G", f"{G:.3f}")

            st.write("**Second Fundamental Form**")
            cols_sf = st.columns(3)
            cols_sf[0].metric("e", f"{e:.3f}")
            cols_sf[1].metric("f", f"{f:.3f}")
            cols_sf[2].metric("g", f"{g:.3f}")


if __name__ == "__main__":
    main()