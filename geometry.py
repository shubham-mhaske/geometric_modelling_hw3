
"""
geometry.py — Core geometry for tensor‑product Bézier / B‑spline / NURBS surfaces.

This module is UI‑agnostic and safe to unit test. It provides:
- Control grid dataclasses
- Basis functions (Bernstein, Cox–de Boor)
- Surface evaluation for Bezier, B‑spline, NURBS
- Numerical derivatives and curvature (normal, principal directions, K & H)
- Generators for random/preset surfaces
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Literal, Dict, Any
import math
import numpy as np

SurfaceKind = Literal["Bezier", "B-spline", "NURBS"]

@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def as_np(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_np(v: np.ndarray) -> "Vec3":
        return Vec3(float(v[0]), float(v[1]), float(v[2]))

@dataclass
class ControlPoint:
    p: Vec3
    w: float = 1.0  # weight; 1 for non‑rational

@dataclass
class ControlGrid:
    m: int  # u count
    n: int  # v count
    points: List[ControlPoint]  # row‑major length m*n

    def at(self, i: int, j: int) -> ControlPoint:
        return self.points[i*self.n + j]

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (P, W) where P shape (m,n,3) and W shape (m,n)."""
        P = np.zeros((self.m, self.n, 3), dtype=float)
        W = np.ones((self.m, self.n), dtype=float)
        for i in range(self.m):
            for j in range(self.n):
                cp = self.at(i, j)
                P[i, j, :] = cp.p.as_np()
                W[i, j] = cp.w
        return P, W

def bernstein_all(d: int, t: float) -> np.ndarray:
    """Stable iterative Bernstein basis of degree d at t∈[0,1]."""
    B = np.zeros(d+1, dtype=float)
    B[0] = 1.0
    u1 = 1.0 - t
    for j in range(1, d+1):
        saved = 0.0
        for k in range(0, j):
            temp = B[k]
            B[k] = saved + u1 * temp
            saved = t * temp
        B[j] = saved
    return B

def find_span(n: int, p: int, u: float, U: List[float]) -> int:
    """Find knot span index as in The NURBS Book (Piegl & Tiller)."""
    if u >= U[n+1]:
        return n
    if u <= U[p]:
        return p
    low = p
    high = n + 1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid+1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid

def basis_funs(i: int, u: float, p: int, U: List[float]) -> np.ndarray:
    N = np.zeros(p+1, dtype=float)
    left = np.zeros(p+1, dtype=float)
    right = np.zeros(p+1, dtype=float)
    N[0] = 1.0
    for j in range(1, p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        saved = 0.0
        for r in range(0, j):
            denom = right[r+1] + left[j-r]
            val = 0.0 if denom == 0 else N[r] / denom
            N[r] = saved + right[r+1] * val
            saved = left[j-r] * val
        N[j] = saved
    return N

def uniform_knot_vector(n_ctrl: int, degree: int, nonperiodic: bool=True) -> List[float]:
    m = n_ctrl + degree + 1
    if nonperiodic:
        # Open uniform (clamped)
        kv = [0.0]* (degree+1)
        interior = m - 2*(degree+1)
        if interior > 0:
            kv += [i/interior for i in range(1, interior)]
        kv += [1.0]* (degree+1)
    else:
        kv = [i/(m-1) for i in range(m)]
    return kv

def nonuniform_knot_vector(n_ctrl: int, degree: int, seed: int=0) -> List[float]:
    rng = np.random.default_rng(seed)
    m = n_ctrl + degree + 1
    # clamped ends, random interior nondecreasing
    kv = [0.0]*(degree+1)
    interior = m - 2*(degree+1)
    if interior > 0:
        interior_vals = np.sort(rng.random(interior)).tolist()
        kv += interior_vals
    kv += [1.0]*(degree+1)
    return kv

def param_domain(kind: SurfaceKind, degree: int, knots: List[float]) -> Tuple[float,float]:
    if kind == "Bezier":
        return 0.0, 1.0
    else:
        return knots[degree], knots[len(knots)-degree-1]

def eval_surface(spec: Dict[str, Any], u: float, v: float) -> np.ndarray:
    """
    Evaluate S(u,v) for spec with keys:
        kind: SurfaceKind
        p, q: degrees
        grid: ControlGrid
        U, V: knot vectors (for B-spline/NURBS)
    Returns np.array shape (3,).
    """
    kind: SurfaceKind = spec["kind"]
    p = int(spec["p"]); q = int(spec["q"])
    grid: ControlGrid = spec["grid"]
    U: List[float] = spec.get("U") or []
    V: List[float] = spec.get("V") or []
    m, n = grid.m, grid.n
    P, W = grid.to_numpy()

    if kind == "Bezier":
        Bu = bernstein_all(p, u)  # len p+1
        Bv = bernstein_all(q, v)  # len q+1
        S = np.zeros(3, dtype=float)
        wsum = 0.0
        for i in range(p+1):
            for j in range(q+1):
                wij = W[i, j]
                b = Bu[i]*Bv[j]
                S += b*wij*P[i, j, :]
                wsum += b*wij
        if wsum != 0:
            S = S/wsum
        return S

    # B‑spline / NURBS
    up = p; vq = q
    u_span = find_span(m-1, up, u, U)
    v_span = find_span(n-1, vq, v, V)
    Nu = basis_funs(u_span, u, up, U)  # len p+1
    Nv = basis_funs(v_span, v, vq, V)  # len q+1

    Sw = np.zeros(3, dtype=float)
    wsum = 0.0
    for a in range(up+1):
        i = u_span - up + a
        for b in range(vq+1):
            j = v_span - vq + b
            wij = W[i, j]
            B = Nu[a]*Nv[b]
            Sw += B*wij*P[i, j, :]
            wsum += B*wij
    if wsum != 0:
        Sw = Sw/wsum
    return Sw

def differential(spec: Dict[str, Any], u: float, v: float, h: float=1e-3):
    """Return dict with Su, Sv, Suu, Svv, Suv, normal, E,F,G,e,f,g,K,H,k1,k2,d1,d2"""
    kind: SurfaceKind = spec["kind"]
    p = int(spec["p"]); q = int(spec["q"])
    U = spec.get("U") or []
    V = spec.get("V") or []

    u0, u1 = param_domain(kind, p, U if U else [0,0,1,1])
    v0, v1 = param_domain(kind, q, V if V else [0,0,1,1])
    # Scale‑aware steps
    hu = h*(u1 - u0)
    hv = h*(v1 - v0)

    def clamp(x,a,b): return max(a, min(b, x))
    uL, uR = clamp(u - hu, u0, u1), clamp(u + hu, u0, u1)
    vL, vR = clamp(v - hv, v0, v1), clamp(v + hv, v0, v1)

    S   = eval_surface(spec, u, v)
    Su  = (eval_surface(spec, uR, v) - eval_surface(spec, uL, v)) / max(1e-12, (uR - uL))
    Sv  = (eval_surface(spec, u, vR) - eval_surface(spec, u, vL)) / max(1e-12, (vR - vL))
    Suu = (eval_surface(spec, uR, v) + eval_surface(spec, uL, v) - 2*S) / max(1e-12, (0.5*(uR-uL))**2)
    Svv = (eval_surface(spec, u, vR) + eval_surface(spec, u, vL) - 2*S) / max(1e-12, (0.5*(vR-vL))**2)
    Suv = (eval_surface(spec, uR, vR) - eval_surface(spec, uR, vL) - eval_surface(spec, uL, vR) + eval_surface(spec, uL, vL)) / max(1e-12, (uR-uL)*(vR-vL))

    # First fundamental form
    E = float(np.dot(Su, Su))
    F = float(np.dot(Su, Sv))
    G = float(np.dot(Sv, Sv))

    # Normal
    n = np.cross(Su, Sv)
    n_norm = np.linalg.norm(n)
    if n_norm > 0: n = n / n_norm

    # Second fundamental form coefficients: project second partials onto normal
    e = float(np.dot(n, Suu))
    f = float(np.dot(n, Suv))
    g = float(np.dot(n, Svv))

    # Curvatures via forms
    denom = (E*G - F*F)
    K = (e*g - f*f) / denom if abs(denom) > 1e-20 else float("nan")
    H = (E*g - 2*F*f + G*e) / (2*denom) if abs(denom) > 1e-20 else float("nan")

    # Principal curvatures / directions via shape operator S = I^{-1} II
    I = np.array([[E, F],[F, G]], dtype=float)
    II = np.array([[e, f],[f, g]], dtype=float)
    try:
        Iinv = np.linalg.inv(I)
        Sshape = Iinv @ II
        vals, vecs = np.linalg.eig(Sshape)  # eigenvalues = principal curvatures
        # Sort so k1 >= k2
        order = np.argsort(vals)[::-1]
        vals = np.real(vals[order])
        vecs = np.real(vecs[:, order])
        k1, k2 = float(vals[0]), float(vals[1])
        # Convert eigenvectors (a,b) in (u,v) basis to 3D directions a*Su + b*Sv
        a1, b1 = vecs[0,0], vecs[1,0]
        a2, b2 = vecs[0,1], vecs[1,1]
        d1 = a1*Su + b1*Sv
        d2 = a2*Su + b2*Sv
        # Normalize and orthogonalize to n for visualization
        def norm(v):
            nv = np.linalg.norm(v)
            return v/nv if nv>0 else v
        d1 = norm(d1 - np.dot(d1, n)*n)
        d2 = norm(d2 - np.dot(d2, n)*n)
    except np.linalg.LinAlgError:
        k1=k2=float("nan")
        d1=d2=np.array([float("nan")]*3)

    return {
        "S": S, "Su": Su, "Sv": Sv,
        "Suu": Suu, "Svv": Svv, "Suv": Suv,
        "normal": n, "E":E, "F":F, "G":G, "e":e, "f":f, "g":g,
        "K": K, "H": H, "k1": k1, "k2": k2, "d1": d1, "d2": d2
    }

# ---------- Presets & Generators ----------

def grid_from_zfunc(m: int, n: int, zfunc, scale=1.0, weight=1.0) -> ControlGrid:
    """Create a control grid from a height function z=f(u,v) and optional weight function.
    Args:
        m, n: Grid dimensions
        zfunc: Function (u,v) -> z where u,v in [0,1]
        scale: Scale factor for x,y coordinates
        weight: Either a constant or function (u,v) -> w for NURBS weights
    """
    pts = []
    for i in range(m):
        u = i/(m-1 if m>1 else 1)
        for j in range(n):
            v = j/(n-1 if n>1 else 1)
            x = (u - 0.5)*scale*2
            y = (v - 0.5)*scale*2
            z = zfunc(u, v)
            w = weight(u, v) if callable(weight) else weight
            pts.append(ControlPoint(Vec3(x,y,z), w=w))
    return ControlGrid(m, n, pts)

def random_grid(m: int, n: int, amp: float=1.0, seed: int=0, rational=False) -> ControlGrid:
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(m):
        for j in range(n):
            x = (i/(m-1)-0.5)*2
            y = (j/(n-1)-0.5)*2
            z = float(rng.normal(scale=amp*0.5))
            w = float(np.clip(rng.normal(loc=1.0, scale=0.3), 0.2, 3.0)) if rational else 1.0
            pts.append(ControlPoint(Vec3(x,y,z), w=w))
    return ControlGrid(m, n, pts)

def preset_surfaces() -> dict:
    return {
        "Saddle (Bezier)": {
            "kind":"Bezier", "p":3, "q":3,
            "grid": grid_from_zfunc(4,4, lambda u,v: (u-0.5)*(v-0.5)*2, scale=2.0),
            "U":[0,0,0,0,1,1,1,1], "V":[0,0,0,0,1,1,1,1]
        },
        "Wave (B-spline)": {
            "kind":"B-spline", "p":3, "q":3,
            "grid": grid_from_zfunc(6,6, lambda u,v: 0.6*math.sin(2*math.pi*u)*math.cos(2*math.pi*v)),
            "U": uniform_knot_vector(6,3), "V": uniform_knot_vector(6,3)
        },
        "Hyperbolic Paraboloid (Bezier)": {
            "kind":"Bezier", "p":3, "q":3,
            "grid": grid_from_zfunc(4,4, lambda u,v: (u*u - v*v), scale=1.5),
            "U":[0,0,0,0,1,1,1,1], "V":[0,0,0,0,1,1,1,1]
        },
        "Torus Section (NURBS)": {
            "kind":"NURBS", "p":3, "q":3,
            "grid": grid_from_zfunc(6,6, 
                lambda u,v: math.sqrt((1.5 + math.cos(2*math.pi*u))**2 + math.sin(2*math.pi*u)**2) * 
                           math.cos(math.pi*v), 
                scale=1.0,
                weight=lambda u,v: 1 + 0.5*math.cos(2*math.pi*u)*math.cos(math.pi*v)),
            "U": nonuniform_knot_vector(6,3,seed=1), 
            "V": nonuniform_knot_vector(6,3,seed=2)
        },
        "Ripple Bowl (B-spline)": {
            "kind":"B-spline", "p":3, "q":3,
            "grid": grid_from_zfunc(8,8, 
                lambda u,v: math.exp(-4*((u-0.5)**2 + (v-0.5)**2)) * 
                           math.sin(8*math.pi*math.sqrt((u-0.5)**2 + (v-0.5)**2)), 
                scale=2.0),
            "U": nonuniform_knot_vector(8,3,seed=3), 
            "V": nonuniform_knot_vector(8,3,seed=4)
        },
        "Monkey Saddle (Bezier)": {
            "kind":"Bezier", "p":4, "q":4,
            "grid": grid_from_zfunc(5,5, 
                lambda u,v: (u-0.5)**3 - 3*(u-0.5)*(v-0.5)**2, 
                scale=2.0),
            "U":[0,0,0,0,0,1,1,1,1,1], 
            "V":[0,0,0,0,0,1,1,1,1,1]
        },
        "Cap (NURBS Dome)": {
            "kind":"NURBS", "p":3, "q":3,
            "grid": grid_from_zfunc(6,6,
                lambda u,v: 1 - 2*((u-0.5)**2 + (v-0.5)**2),
                scale=1.0,
                weight=lambda u,v: 1/(1 + 2*((u-0.5)**2 + (v-0.5)**2))),
            "U": nonuniform_knot_vector(6,3,seed=7), 
            "V": nonuniform_knot_vector(6,3,seed=9)
        }
    }

def export_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
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
    g = data["grid"]
    pts = []
    for item in g["points"]:
        P = item["p"]
        pts.append(ControlPoint(Vec3(P["x"], P["y"], P["z"]), w=float(item.get("w",1.0))))
    grid = ControlGrid(int(g["m"]), int(g["n"]), pts)
    return {
        "kind": data["kind"],
        "p": int(data["p"]), "q": int(data["q"]),
        "U": data.get("U"), "V": data.get("V"),
        "grid": grid
    }
