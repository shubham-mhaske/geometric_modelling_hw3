from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Literal, Dict, Any
import math
import numpy as np

# --- Type Definitions ---

SurfaceKind = Literal["Bezier", "B-spline", "NURBS"]
"""Type alias for the kinds of surfaces supported."""


# --- Core Dataclasses ---

@dataclass
class Vec3:
    """Represents a 3D vector or point."""
    x: float
    y: float
    z: float

    def as_np(self) -> np.ndarray:
        """Return the vector as a NumPy array."""
        return np.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_np(v: np.ndarray) -> "Vec3":
        """Create a Vec3 from a NumPy array."""
        return Vec3(float(v[0]), float(v[1]), float(v[2]))


@dataclass
class ControlPoint:
    """Represents a control point, with a position and a weight."""
    p: Vec3
    w: float = 1.0  # Weight; w=1.0 for non-rational (Bézier, B-spline) points.


@dataclass
class ControlGrid:
    """
    Represents a grid of control points for a tensor-product surface.

    Attributes:
        m: The number of control points in the u-direction.
        n: The number of control points in the v-direction.
        points: A flat list of m*n control points, stored in row-major order.
    """
    m: int
    n: int
    points: List[ControlPoint]

    def at(self, i: int, j: int) -> ControlPoint:
        """
        Access the control point at grid index (i, j).

        Args:
            i: The row index (in u-direction, from 0 to m-1).
            j: The column index (in v-direction, from 0 to n-1).

        Returns:
            The ControlPoint at the specified grid location.
        """
        return self.points[i * self.n + j]

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the control grid to NumPy arrays for efficient computation.

        Returns:
            A tuple (P, W) where:
            - P is an (m, n, 3) array of control point positions.
            - W is an (m, n) array of control point weights.
        """
        P = np.zeros((self.m, self.n, 3), dtype=float)
        W = np.ones((self.m, self.n), dtype=float)
        for i in range(self.m):
            for j in range(self.n):
                cp = self.at(i, j)
                P[i, j, :] = cp.p.as_np()
                W[i, j] = cp.w
        return P, W


# --- Basis Functions ---

def bernstein_all(d: int, t: float) -> np.ndarray:
    """
    Calculates all Bernstein basis polynomials of degree d at parameter t.

    This uses a stable, iterative method (de Casteljau's algorithm) to
    compute the basis functions.

    Args:
        d: The degree of the Bernstein basis.
        t: The parameter value, typically in [0, 1].

    Returns:
        A NumPy array of size (d+1) containing the basis function values.
    """
    B = np.zeros(d + 1, dtype=float)
    B[0] = 1.0
    u1 = 1.0 - t
    for j in range(1, d + 1):
        saved = 0.0
        for k in range(j):
            temp = B[k]
            B[k] = saved + u1 * temp
            saved = t * temp
        B[j] = saved
    return B


def find_span(n: int, p: int, u: float, U: List[float]) -> int:
    """
    Determines the knot span index for a given parameter u.

    This function finds the index `i` such that `U[i] <= u < U[i+1]`.

    Args:
        n: The number of control points minus 1.
        p: The degree of the basis function.
        u: The parameter value.
        U: The knot vector.

    Returns:
        The index of the knot span containing u.
    """
    if u >= U[n + 1]:
        return n  # Special case for u at the end of the domain
    if u <= U[p]:
        return p  # Special case for u at the start of the domain

    # Binary search for the span
    low = p
    high = n + 1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def basis_funs(i: int, u: float, p: int, U: List[float]) -> np.ndarray:
    """
    Computes the non-zero B-spline basis functions for a given knot span.

    This function uses the Cox-de Boor recursion formula.

    Args:
        i: The knot span index (from find_span).
        u: The parameter value.
        p: The degree of the basis function.
        U: The knot vector.

    Returns:
        A NumPy array of size (p+1) containing the values of the non-zero
        basis functions N_{i-p,p}(u), ..., N_{i,p}(u).
    """
    N = np.zeros(p + 1, dtype=float)
    left = np.zeros(p + 1, dtype=float)
    right = np.zeros(p + 1, dtype=float)
    N[0] = 1.0

    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(j):
            # Numerator of the first term in the recursion
            temp = N[r] / (right[r + 1] + left[j - r])
            # Contribution from the first term
            N[r] = saved + right[r + 1] * temp
            # Contribution from the second term
            saved = left[j - r] * temp
        N[j] = saved
    return N


# --- Knot Vector Generation ---

def uniform_knot_vector(n_ctrl: int, degree: int) -> List[float]:
    """
    Creates a non-periodic (clamped) uniform knot vector.

    A clamped knot vector has `degree+1` identical knots at the beginning
    and end, which makes the B-spline curve/surface pass through its
    endpoints.

    Args:
        n_ctrl: The number of control points.
        degree: The polynomial degree.

    Returns:
        A list of floats representing the knot vector.
    """
    # Total number of knots is m + 1 = (n_ctrl + degree) + 1
    num_knots = n_ctrl + degree + 1

    # Start with (degree + 1) zeros for clamping
    kv = [0.0] * (degree + 1)

    # Generate the uniform interior knots
    num_interior_knots = num_knots - 2 * (degree + 1)
    if num_interior_knots > 0:
        # The number of intervals in the interior part of the domain
        interior_intervals = num_interior_knots + 1
        kv += [i / interior_intervals for i in range(1, num_interior_knots + 1)]

    # End with (degree + 1) ones for clamping
    kv += [1.0] * (degree + 1)

    return kv


def nonuniform_knot_vector(n_ctrl: int, degree: int, seed: int = 0) -> List[float]:
    """
    Creates a non-periodic (clamped) non-uniform random knot vector.

    The interior knots are chosen randomly but sorted to ensure they are
    non-decreasing.

    Args:
        n_ctrl: The number of control points.
        degree: The polynomial degree.
        seed: A seed for the random number generator for reproducibility.

    Returns:
        A list of floats representing the knot vector.
    """
    rng = np.random.default_rng(seed)
    num_knots = n_ctrl + degree + 1

    # Clamped ends, random interior
    kv = [0.0] * (degree + 1)
    num_interior_knots = num_knots - 2 * (degree + 1)
    if num_interior_knots > 0:
        interior_vals = np.sort(rng.random(num_interior_knots)).tolist()
        kv += interior_vals
    kv += [1.0] * (degree + 1)
    return kv


# --- Surface Evaluation ---

def param_domain(kind: SurfaceKind, degree: int, knots: List[float]) -> Tuple[float, float]:
    """
    Determines the valid parameter domain [u_min, u_max] for a surface.

    For Bézier surfaces, the domain is always [0, 1]. For B-spline and NURBS
    surfaces, it is determined by the knot vector.

    Args:
        kind: The type of surface.
        degree: The polynomial degree in the direction of interest.
        knots: The knot vector in the direction of interest.

    Returns:
        A tuple (min_val, max_val) representing the parameter domain.
    """
    if kind == "Bezier":
        return 0.0, 1.0
    else:
        # For a clamped B-spline, the domain is [U_p, U_{m-p}]
        # where m = n_ctrl + degree.
        # The last knot index is n_ctrl + degree.
        return knots[degree], knots[len(knots) - degree - 1]


def eval_surface(spec: Dict[str, Any], u: float, v: float) -> np.ndarray:
    """
    Evaluates the surface point S(u,v) for a given surface specification.

    This function dispatches to the correct evaluation method based on the
    surface `kind` provided in the specification.

    Args:
        spec: A dictionary containing the surface definition:
              - "kind": The SurfaceKind.
              - "p", "q": The integer degrees in u and v.
              - "grid": The ControlGrid.
              - "U", "V": The knot vectors (for B-spline/NURBS).
        u: The parameter value in the u-direction.
        v: The parameter value in the v-direction.

    Returns:
        A NumPy array of shape (3,) representing the 3D point on the surface.
    """
    kind: SurfaceKind = spec["kind"]
    p = int(spec["p"])
    q = int(spec["q"])
    grid: ControlGrid = spec["grid"]
    P, W = grid.to_numpy()

    # --- Bézier Surface Evaluation (special case of B-spline) ---
    if kind == "Bezier":
        # For a Bézier surface, the number of control points is degree + 1
        # and the basis functions are the Bernstein polynomials.
        Bu = bernstein_all(p, u)  # Bernstein basis for u, size (p+1)
        Bv = bernstein_all(q, v)  # Bernstein basis for v, size (q+1)

        S_h = np.zeros(3, dtype=float)  # Homogeneous coordinate for the point
        w_sum = 0.0
        for i in range(p + 1):
            for j in range(q + 1):
                w_ij = W[i, j]
                basis_val = Bu[i] * Bv[j]
                S_h += (basis_val * w_ij) * P[i, j, :]
                w_sum += basis_val * w_ij

        return S_h / w_sum if w_sum != 0 else S_h

    # --- B-spline / NURBS Surface Evaluation ---
    U: List[float] = spec.get("U") or []
    V: List[float] = spec.get("V") or []
    m, n = grid.m, grid.n

    # Find the knot spans that contain u and v
    u_span = find_span(m - 1, p, u, U)
    v_span = find_span(n - 1, q, v, V)

    # Compute the non-zero basis functions for each parameter
    Nu = basis_funs(u_span, u, p, U)  # size (p+1)
    Nv = basis_funs(v_span, v, q, V)  # size (q+1)

    S_h = np.zeros(3, dtype=float)  # Homogeneous coordinate for the point
    w_sum = 0.0
    # Sum the contributions of the (p+1)x(q+1) active control points
    for a in range(p + 1):
        i = u_span - p + a
        for b in range(q + 1):
            j = v_span - q + b
            w_ij = W[i, j]
            basis_val = Nu[a] * Nv[b]
            S_h += (basis_val * w_ij) * P[i, j, :]
            w_sum += basis_val * w_ij

    return S_h / w_sum if w_sum != 0 else S_h


# --- Differential Geometry ---

def differential(spec: Dict[str, Any], u: float, v: float, h: float = 1e-4) -> Dict[str, Any]:
    """
    Calculates differential properties of a surface at (u,v) via finite differences.

    Args:
        spec: The surface specification dictionary.
        u: The u-parameter for evaluation.
        v: The v-parameter for evaluation.
        h: The step size for finite difference calculations.

    Returns:
        A dictionary containing:
        - S: The surface point.
        - Su, Sv: First partial derivatives.
        - Suu, Svv, Suv: Second partial derivatives.
        - normal: The unit normal vector.
        - E, F, G: Coefficients of the First Fundamental Form.
        - e, f, g: Coefficients of the Second Fundamental Form.
        - K, H: Gaussian and Mean curvatures.
        - k1, k2: Principal curvatures.
        - d1, d2: Principal direction vectors.
    """
    kind: SurfaceKind = spec["kind"]
    p = int(spec["p"])
    q = int(spec["q"])
    U = spec.get("U") or []
    V = spec.get("V") or []

    # Use parameter-domain-aware step sizes for better numerical stability
    u_min, u_max = param_domain(kind, p, U if U else [0, 0, 1, 1])
    v_min, v_max = param_domain(kind, q, V if V else [0, 0, 1, 1])
    h_u = h * (u_max - u_min)
    h_v = h * (v_max - v_min)

    def clamp(val, min_val, max_val):
        return max(min_val, min(max_val, val))

    # Clamped parameter values for finite differences
    u_L, u_R = clamp(u - h_u, u_min, u_max), clamp(u + h_u, u_min, u_max)
    v_L, v_R = clamp(v - h_v, v_min, v_max), clamp(v + h_v, v_min, v_max)

    # Central difference for first and second partial derivatives
    S = eval_surface(spec, u, v)
    S_uR = eval_surface(spec, u_R, v)
    S_uL = eval_surface(spec, u_L, v)
    S_vR = eval_surface(spec, u, v_R)
    S_vL = eval_surface(spec, u, v_L)

    Su = (S_uR - S_uL) / max(1e-12, u_R - u_L)
    Sv = (S_vR - S_vL) / max(1e-12, v_R - v_L)
    Suu = (S_uR + S_uL - 2 * S) / max(1e-12, (0.5 * (u_R - u_L)) ** 2)
    Svv = (S_vR + S_vL - 2 * S) / max(1e-12, (0.5 * (v_R - v_L)) ** 2)

    S_uRvR = eval_surface(spec, u_R, v_R)
    S_uRvL = eval_surface(spec, u_R, v_L)
    S_uLvR = eval_surface(spec, u_L, v_R)
    S_uLvL = eval_surface(spec, u_L, v_L)
    Suv = (S_uRvR - S_uRvL - S_uLvR + S_uLvL) / max(1e-12, (u_R - u_L) * (v_R - v_L))

    # --- First Fundamental Form (Metric Tensor) ---
    # Measures distances and angles on the surface.
    E = float(np.dot(Su, Su))
    F = float(np.dot(Su, Sv))
    G = float(np.dot(Sv, Sv))

    # --- Normal Vector ---
    normal_vec = np.cross(Su, Sv)
    n_norm = np.linalg.norm(normal_vec)
    n = normal_vec / n_norm if n_norm > 1e-12 else np.array([0.0, 0.0, 1.0])

    # --- Second Fundamental Form ---
    # Measures how the surface curves away from the tangent plane.
    e = float(np.dot(n, Suu))
    f = float(np.dot(n, Suv))
    g = float(np.dot(n, Svv))

    # --- Curvatures (from Fundamental Forms) ---
    denom = (E * G - F * F)
    if abs(denom) < 1e-20:
        K = H = float("nan")
    else:
        # Gaussian curvature (K): Intrinsic, product of principal curvatures.
        K = (e * g - f * f) / denom
        # Mean curvature (H): Extrinsic, average of principal curvatures.
        H = (E * g - 2 * F * f + G * e) / (2 * denom)

    # --- Principal Curvatures and Directions (from Shape Operator) ---
    k1, k2 = float("nan"), float("nan")
    d1, d2 = np.array([np.nan] * 3), np.array([np.nan] * 3)
    try:
        # The shape operator (Weingarten map) relates the tangent plane to itself.
        # Its eigenvalues are the principal curvatures (k1, k2) and its
        # eigenvectors are the principal directions.
        I = np.array([[E, F], [F, G]], dtype=float)
        II = np.array([[e, f], [f, g]], dtype=float)
        I_inv = np.linalg.inv(I)
        S_shape = I_inv @ II

        vals, vecs = np.linalg.eig(S_shape)
        order = np.argsort(vals)[::-1]  # Sort so k1 >= k2
        vals, vecs = np.real(vals[order]), np.real(vecs[:, order])
        k1, k2 = float(vals[0]), float(vals[1])

        # Convert eigenvectors from (u,v) basis to 3D vectors in the tangent plane
        d1 = vecs[0, 0] * Su + vecs[1, 0] * Sv
        d2 = vecs[0, 1] * Su + vecs[1, 1] * Sv

        # Normalize and ensure orthogonality to the normal for visualization
        def norm(v):
            nv = np.linalg.norm(v)
            return v / nv if nv > 0 else v
        d1 = norm(d1 - np.dot(d1, n) * n)
        d2 = norm(d2 - np.dot(d2, n) * n)

    except np.linalg.LinAlgError:
        # This can happen if the first fundamental form is singular (e.g., at a cusp).
        pass

    return {
        "S": S, "Su": Su, "Sv": Sv, "Suu": Suu, "Svv": Svv, "Suv": Suv,
        "normal": n, "E": E, "F": F, "G": G, "e": e, "f": f, "g": g,
        "K": K, "H": H, "k1": k1, "k2": k2, "d1": d1, "d2": d2
    }


# --- Presets & Generators ---

def grid_from_zfunc(m: int, n: int, zfunc, scale: float = 1.0, weight: Any = 1.0) -> ControlGrid:
    """
    Creates a control grid from a height function z=f(u,v).

    The (x,y) coordinates are generated on a uniform grid from [-scale, scale].

    Args:
        m: Grid dimension in u-direction.
        n: Grid dimension in v-direction.
        zfunc: A function `f(u,v) -> z` where u,v are in [0,1].
        scale: A scale factor for the x and y coordinates.
        weight: A constant weight or a function `f(u,v) -> w` for NURBS.

    Returns:
        A new ControlGrid.
    """
    pts = []
    for i in range(m):
        u = i / (m - 1 if m > 1 else 1)
        for j in range(n):
            v = j / (n - 1 if n > 1 else 1)
            x = (u - 0.5) * scale * 2
            y = (v - 0.5) * scale * 2
            z = zfunc(u, v)
            w = weight(u, v) if callable(weight) else weight
            pts.append(ControlPoint(Vec3(x, y, z), w=w))
    return ControlGrid(m, n, pts)


def random_grid(m: int, n: int, amp: float = 1.0, seed: int = 0, rational: bool = False) -> ControlGrid:
    """
    Creates a grid with random z-heights and optionally random weights.

    Args:
        m: Grid dimension in u-direction.
        n: Grid dimension in v-direction.
        amp: Amplitude multiplier for the random z-values.
        seed: A seed for the random number generator.
        rational: If True, generate random weights for a NURBS surface.

    Returns:
        A new ControlGrid.
    """
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(m):
        u = i / (m - 1 if m > 1 else 1)
        for j in range(n):
            v = j / (n - 1 if n > 1 else 1)
            x = (u - 0.5) * 2
            y = (v - 0.5) * 2
            z = float(rng.normal(scale=amp * 0.5))
            w = float(np.clip(rng.normal(loc=1.0, scale=0.3), 0.2, 3.0)) if rational else 1.0
            pts.append(ControlPoint(Vec3(x, y, z), w=w))
    return ControlGrid(m, n, pts)


def preset_surfaces() -> Dict[str, Dict[str, Any]]:
    """
    Returns a dictionary of pre-defined surface specifications.

    This provides a set of interesting starting points for exploration.
    """
    return {
        "Saddle (Bezier)": {
            "kind": "Bezier", "p": 3, "q": 3,
            "grid": grid_from_zfunc(4, 4, lambda u, v: (u - 0.5) * (v - 0.5) * 4, scale=1.5),
            "U": uniform_knot_vector(4, 3), "V": uniform_knot_vector(4, 3)
        },
        "Wave (B-spline)": {
            "kind": "B-spline", "p": 3, "q": 3,
            "grid": grid_from_zfunc(6, 6, lambda u, v: 0.4 * math.sin(2 * math.pi * u) * math.cos(2 * math.pi * v), scale=1.5),
            "U": uniform_knot_vector(6, 3), "V": uniform_knot_vector(6, 3)
        },
        "Hyperbolic Paraboloid (Bezier)": {
            "kind": "Bezier", "p": 3, "q": 3,
            "grid": grid_from_zfunc(4, 4, lambda u, v: (u * u - v * v), scale=1.5),
            "U": uniform_knot_vector(4, 3), "V": uniform_knot_vector(4, 3)
        },
        "Torus Section (NURBS)": {
            "kind": "NURBS", "p": 2, "q": 2,
            "grid": grid_from_zfunc(7, 7,
                lambda u, v: math.cos(2 * math.pi * u) * (2 + math.cos(2 * math.pi * v)),
                scale=1.0,
                weight=lambda u, v: 1 + 0.5 * math.cos(2 * math.pi * v)),
            "U": nonuniform_knot_vector(7, 2, seed=1),
            "V": nonuniform_knot_vector(7, 2, seed=2)
        },
        "Ripple Bowl (B-spline)": {
            "kind": "B-spline", "p": 3, "q": 3,
            "grid": grid_from_zfunc(8, 8,
                lambda u, v: math.exp(-5 * ((u - 0.5) ** 2 + (v - 0.5) ** 2)) *
                           math.sin(10 * math.pi * math.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2)),
                scale=2.0),
            "U": nonuniform_knot_vector(8, 3, seed=3),
            "V": nonuniform_knot_vector(8, 3, seed=4)
        },
        "Monkey Saddle (Bezier)": {
            "kind": "Bezier", "p": 3, "q": 3,
            "grid": grid_from_zfunc(4, 4,
                lambda u, v: 4 * ((u - 0.5) ** 3 - 3 * (u - 0.5) * (v - 0.5) ** 2),
                scale=1.5),
            "U": uniform_knot_vector(4, 3),
            "V": uniform_knot_vector(4, 3)
        },
        "Cap (NURBS Dome)": {
            "kind": "NURBS", "p": 3, "q": 3,
            "grid": grid_from_zfunc(6, 6,
                lambda u, v: 1 - 2 * ((u - 0.5) ** 2 + (v - 0.5) ** 2),
                scale=1.0,
                weight=lambda u, v: 1 / (1 + 2 * ((u - 0.5) ** 2 + (v - 0.5) ** 2))),
            "U": nonuniform_knot_vector(6, 3, seed=7),
            "V": nonuniform_knot_vector(6, 3, seed=9)
        }
    }


# --- Import / Export ---

def export_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serializes a surface specification dictionary into a JSON-compatible format.

    The ControlGrid object is converted into a nested dictionary.

    Args:
        spec: The surface specification dictionary to export.

    Returns:
        A new dictionary that is safe to serialize to JSON.
    """
    grid: ControlGrid = spec["grid"]
    return {
        "kind": spec["kind"],
        "p": int(spec["p"]), "q": int(spec["q"]),
        "U": spec.get("U"), "V": spec.get("V"),
        "grid": {
            "m": grid.m, "n": grid.n,
            "points": [{
                "p": {"x": cp.p.x, "y": cp.p.y, "z": cp.p.z},
                "w": cp.w
            } for cp in grid.points]
        }
    }


def import_spec(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserializes a dictionary (from JSON) into a surface specification.

    The nested dictionary for the control grid is converted back into a
    ControlGrid object.

    Args:
        data: The raw dictionary, typically loaded from JSON.

    Returns:
        A surface specification dictionary with the correct object types.
    """
    g = data["grid"]
    pts = []
    for item in g["points"]:
        p_data = item["p"]
        pts.append(ControlPoint(Vec3(p_data["x"], p_data["y"], p_data["z"]), w=float(item.get("w", 1.0))))
    grid = ControlGrid(int(g["m"]), int(g["n"]), pts)

    return {
        "kind": data["kind"],
        "p": int(data["p"]), "q": int(data["q"]),
        "U": data.get("U"), "V": data.get("V"),
        "grid": grid
    }