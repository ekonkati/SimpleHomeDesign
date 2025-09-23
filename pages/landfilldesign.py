# Landfill Design, Stability, Visualization & Submission App (Streamlit)
# -------------------------------------------------------------------
# Major update: precise BBL/ABL split (Base→GL, GL→TOB, TOB→TOL),
# corrected frustum math, and a full 3D stacked solid (Plotly Mesh3d).
# Keeps Bishop stability, BOQ, and robust Excel/KML/PDF exports.
#
# To run:
#   pip install streamlit numpy pandas matplotlib reportlab simplekml shapely plotly XlsxWriter openpyxl
#   streamlit run landfill_streamlit_app.py

import io
import math
import json
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Optional imports
try:
    from shapely.geometry import Polygon, Point
    from shapely.affinity import translate
except Exception:
    Polygon = None
    Point = None
try:
    import simplekml
except Exception:
    simplekml = None
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# ---------------------------
# Utility & Data Structures
# ---------------------------

SI = {
    "length": "m",
    "area": "m²",
    "volume": "m³",
    "thickness": "m",
    "density": "t/m³",
    "rate_area": "₹/m²",
    "rate_vol": "₹/m³",
    "rate_item": "₹/item",
}

WASTE_PRESETS = {
    "MSW": {
        "gamma_unsat": 9.5,  # kN/m³ (≈ density 0.97 t/m³)
        "gamma_sat": 12.5,   # kN/m³
        "phi": 25.0,         # degrees
        "c": 5.0,            # kPa
        "liner": {
            "clay_thk": 0.9,
            "clay_k": 1e-7,
            "hdpe_thk": 1.5e-3,
            "gcl": True,
            "drain_thk": 0.3,
        },
    },
    "Hazardous": {
        "gamma_unsat": 11.0,
        "gamma_sat": 14.0,
        "phi": 28.0,
        "c": 8.0,
        "liner": {
            "clay_thk": 1.0,
            "clay_k": 1e-9,
            "hdpe_thk": 2.0e-3,
            "gcl": True,
            "drain_thk": 0.4,
        },
    },
}

DEFAULT_RATES = {
    "Clay (compacted)": 500.0,        # ₹/m³
    "HDPE liner install": 350.0,      # ₹/m²
    "GCL": 420.0,                    # ₹/m²
    "Drainage gravel": 900.0,        # ₹/m³
    "Geotextile": 120.0,            # ₹/m²
    "Earthworks (cut/fill)": 180.0,  # ₹/m³
    "Gas well": 95000.0,            # ₹/item
    "Monitoring well": 125000.0,     # ₹/item
    "Topsoil": 300.0,               # ₹/m³
}

@dataclass
class SiteInputs:
    project_name: str
    agency_template: str  # "CPCB" | "EPA"
    latitude: float
    longitude: float
    avg_ground_rl: float
    water_table_depth: float
    waste_type: str
    inflow_tpd: float
    waste_density_tpm3: float
    compaction_factor: float
    lifespan_years_target: Optional[float]

@dataclass
class GeometryInputs:
    inside_slope_h: float   # fill slope above TOB (H)
    inside_slope_v: float   # fill slope above TOB (V)
    outside_slope_h: float  # not used in BBL capacity, kept for BOQ/visuals
    outside_slope_v: float
    berm_width: float       # bench width (legacy)
    berm_height: float      # bench height (legacy)
    lift_thickness: float
    final_height_above_gl: float  # H_final = Hb + H_above
    depth_below_gl: float        # D (excavation depth)

@dataclass
class StabilityInputs:
    gamma_unsat: float
    gamma_sat: float
    phi: float
    cohesion: float
    soil_phi: float
    soil_c: float
    soil_gamma: float
    groundwater_rl: Optional[float]
    ks: float  # seismic coefficient (horizontal)
    target_fos_static: float
    target_fos_seismic: float

# ---------------------------
# Helpers: geometry & volumes
# ---------------------------

def polygon_area(coords: List[Tuple[float, float]]) -> float:
    x = np.array([p[0] for p in coords])
    y = np.array([p[1] for p in coords])
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def rectangle_polygon(width: float, length: float) -> List[Tuple[float, float]]:
    return [(0, 0), (width, 0), (width, length), (0, length)]


def frustum_volume(h: float, A1: float, A2: float) -> float:
    """Volume of a truncated pyramid/prism with parallel faces areas A1 (bottom) and A2 (top)."""
    if h <= 0 or A1 <= 0 or A2 <= 0:
        return 0.0
    return h * (A1 + A2 + math.sqrt(A1 * A2)) / 3.0


def compute_bbl_abl(
    W_GL: float,
    L_GL: float,
    Hb: float,            # bund height (GL→TOB)
    bc: float,            # bund crest width
    m_bund_in: float,     # bund inner slope H:V
    D: float,             # excavation depth below GL (Base→GL)
    m_excav: float,       # excavation slope H:V (outward)
    H_final: float,       # total height above GL (Hb + H_above)
    m_fill: float,        # fill slope H:V above TOB (inward)
    top_area_ratio_min: float = 0.30,  # min A_TOL / A_TOB
) -> dict:
    """Compute plan dimensions, areas, and volumes for each vertical segment and totals."""
    # Sanitize
    Hb = max(Hb, 0.0)
    D = max(D, 0.0)
    H_above = max(H_final - Hb, 0.0)

    # Base (at depth D) expands by excavation slope
    W_Base = max(W_GL + 2.0 * m_excav * D, 0.0)
    L_Base = max(L_GL + 2.0 * m_excav * D, 0.0)

    # At TOB, inner opening shrinks by inner slope + crest both sides
    W_TOB = max(W_GL - 2.0 * (m_bund_in * Hb + bc), 0.0)
    L_TOB = max(L_GL - 2.0 * (m_bund_in * Hb + bc), 0.0)

    # Above TOB, fill shrinks inward
    W_TOL = max(W_TOB - 2.0 * m_fill * H_above, 0.0)
    L_TOL = max(L_TOB - 2.0 * m_fill * H_above, 0.0)

    # Areas
    A_Base = max(W_Base * L_Base, 0.0)
    A_GL   = max(W_GL   * L_GL,   0.0)
    A_TOB  = max(W_TOB  * L_TOB,  0.0)
    A_TOL  = max(W_TOL  * L_TOL,  0.0)

    # Enforce minimum top area ratio (to avoid extremely sharp apex)
    if A_TOB > 0.0:
        A_min = top_area_ratio_min * A_TOB
        if A_TOL < A_min:
            ratio = math.sqrt(A_min / max(A_TOL, 1e-9))
            W_TOL *= ratio
            L_TOL *= ratio
            A_TOL  = W_TOL * L_TOL

    # Volumes: frusta
    V_Base_to_GL = frustum_volume(D, A_Base, A_GL)
    V_GL_to_TOB  = frustum_volume(Hb, A_GL, A_TOB)
    V_TOB_to_TOL = frustum_volume(H_above, A_TOB, A_TOL)

    V_BBL   = V_Base_to_GL + V_GL_to_TOB
    V_ABL   = V_TOB_to_TOL
    V_total = V_BBL + V_ABL

    return {
        "W_Base": W_Base, "L_Base": L_Base,
        "W_GL": W_GL,     "L_GL": L_GL,
        "W_TOB": W_TOB,   "L_TOB": L_TOB,
        "W_TOL": W_TOL,   "L_TOL": L_TOL,
        "A_Base": A_Base, "A_GL": A_GL, "A_TOB": A_TOB, "A_TOL": A_TOL,
        "Hb": Hb, "D": D, "H_above": H_above, "H_final": H_final,
        "V_Base_to_GL": V_Base_to_GL,
        "V_GL_to_TOB": V_GL_to_TOB,
        "V_TOB_to_TOL": V_TOB_to_TOL,
        "V_BBL": V_BBL, "V_ABL": V_ABL, "V_total": V_total,
        "m_bund_in": m_bund_in, "bc": bc, "m_excav": m_excav, "m_fill": m_fill,
    }

# ---------------------------
# Legacy cross-section for stability (kept)
# ---------------------------

def generate_section(geom: GeometryInputs, footprint_area: float) -> dict:
    """Simplified cross-section for plotting & stability. Independent of BBL/ABL math."""
    W0 = max(1.0, math.sqrt(footprint_area))
    H_final = geom.final_height_above_gl
    H_below = geom.depth_below_gl
    m_in = geom.inside_slope_h / max(geom.inside_slope_v, 1e-6)
    m_out = geom.outside_slope_h / max(geom.outside_slope_v, 1e-6)

    z = [ -H_below, 0.0, H_final ]
    x_in = [ 0.0, 0.0, m_in * H_final ]
    x_out = [ W0/2 + m_out*H_below, W0/2, W0/2 + m_out*H_final ]

    A_inner = 0.5 * H_final * (m_in * H_final)
    A_below = 0.5 * H_below * (m_out * H_below)
    A_outer = 0.5 * H_final * (m_out * H_final)

    plan_length_equivalent = footprint_area / max(W0, 1e-6)

    base_area = footprint_area
    side_area = (math.hypot(m_in, 1.0) * H_final) * (2 * plan_length_equivalent)

    return {
        "A_inner": A_inner,
        "A_outer": A_outer,
        "A_below": A_below,
        "base_area": base_area,
        "side_area": side_area,
        "x_in": x_in,
        "z_in": z,
        "x_out": x_out,
        "z_out": z,
        "plan_length_equiv": plan_length_equivalent,
    }

# ---------------------------
# Stability (Bishop simplified)
# ---------------------------

def bishop_simplified(section, stab: StabilityInputs, n_slices: int = 72,
                      center_x: float = 0.0, center_z: float = -10.0,
                      radius: float = 50.0) -> Tuple[float, pd.DataFrame]:
    phi = math.radians(stab.phi)
    c = stab.cohesion  # kPa -> kN/m²
    gamma = stab.gamma_unsat  # kN/m³
    ks = stab.ks

    x_top = np.linspace(0, section["x_in"][-1], n_slices + 1)
    z_top = (section["z_in"][0] + (section["z_in"][-1] - section["z_in"][0]) * (x_top / max(section["x_in"][-1], 1e-6)))

    def circle_z(x):
        return center_z + math.sqrt(max(radius**2 - (x - center_x)**2, 0.0))

    widths = np.diff(x_top)
    x_mid = (x_top[:-1] + x_top[1:]) / 2
    z_surface = (z_top[:-1] + z_top[1:]) / 2
    z_base = np.array([circle_z(x) for x in x_mid])
    h = np.maximum(z_surface - z_base, 1e-3)

    alpha = np.arctan2((x_mid - center_x), (z_surface - center_z))
    W = gamma * h * widths * 1.0

    u = 0.0
    if stab.groundwater_rl is not None:
        gw = stab.groundwater_rl
        head = np.maximum(gw - z_base, 0.0)
        u = 9.81 * head
    N = W * np.cos(alpha)
    N_eff = N - u * widths

    FoS = 1.3
    for _ in range(50):
        num = np.sum(c * widths + (N_eff / FoS) * np.tan(phi))
        den = np.sum(W * np.sin(alpha) + ks * W)
        new_FoS = num / max(den, 1e-6)
        if abs(new_FoS - FoS) < 1e-4:
            FoS = new_FoS
            break
        FoS = new_FoS

    df = pd.DataFrame({
        "x_mid": x_mid,
        "z_surf": z_surface,
        "z_base": z_base,
        "width": widths,
        "W": W,
        "alpha_rad": alpha,
        "N_eff": N_eff,
    })
    return FoS, df


def grid_search_bishop(section, stab: StabilityInputs, n_slices=72) -> Tuple[float, dict, pd.DataFrame]:
    x_max = section["x_in"][-1]
    best = {"FoS": 9e9}
    best_df = None
    for cx in np.linspace(-x_max, x_max*2, 12):
        for cz in np.linspace(-section["z_in"][0]-section["z_in"][-1], -1.0, 10):
            for r in np.linspace(x_max*0.8, x_max*3.0, 12):
                FoS, df = bishop_simplified(section, stab, n_slices, cx, cz, r)
                if FoS < best["FoS"]:
                    best = {"FoS": FoS, "cx": cx, "cz": cz, "r": r}
                    best_df = df
    return best["FoS"], best, best_df

# ---------------------------
# BOQ & Costing
# ---------------------------

def compute_boq(section: dict, liner: dict, rates: dict, A_base_for_liner: float,
                 V_earthworks_approx: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    items = []
    # Liner materials
    clay_vol = liner["clay_thk"] * A_base_for_liner
    # Simple side area approximation using section slope length
    hdpe_area = A_base_for_liner + section["side_area"]
    gcl_area = hdpe_area if liner.get("gcl", False) else 0.0
    drain_vol = liner["drain_thk"] * A_base_for_liner
    # Earthworks (approx)
    cut_fill = max(V_earthworks_approx, 0.0)
    # Topsoil (assume 0.3 m over base area)
    topsoil_vol = 0.3 * A_base_for_liner

    items.append(["Clay (compacted)", clay_vol, "m³", rates.get("Clay (compacted)", 0.0)])
    items.append(["HDPE liner install", hdpe_area, "m²", rates.get("HDPE liner install", 0.0)])
    items.append(["GCL", gcl_area, "m²", rates.get("GCL", 0.0)])
    items.append(["Drainage gravel", drain_vol, "m³", rates.get("Drainage gravel", 0.0)])
    items.append(["Geotextile", hdpe_area, "m²", rates.get("Geotextile", 0.0)])
    items.append(["Earthworks (cut/fill)", cut_fill, "m³", rates.get("Earthworks (cut/fill)", 0.0)])
    items.append(["Topsoil", topsoil_vol, "m³", rates.get("Topsoil", 0.0)])

    df = pd.DataFrame(items, columns=["Item", "Quantity", "Unit", "Rate (₹)"])
    df["Amount (₹)"] = df["Quantity"] * df["Rate (₹)"]

    summary = pd.DataFrame({
        "Metric": ["Total capital cost", "Waste capacity (m³)", "Cost per m³ (₹/m³)"],
        "Value": [df["Amount (₹)"].sum(), st.session_state.get("V_total", np.nan),
                  df["Amount (₹)"].sum() / max(st.session_state.get("V_total", 1e-6), 1e-6)],
    })

    return df, summary

# ---------------------------
# Exports
# ---------------------------

def export_excel(inputs: dict, section: dict, bblabl: dict, boq: pd.DataFrame, summary: pd.DataFrame) -> bytes:
    # Choose an available engine (prefer XlsxWriter)
    engine = "xlsxwriter"
    try:
        import xlsxwriter  # noqa: F401
    except Exception:
        engine = "openpyxl"

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine=engine) as writer:
        pd.DataFrame({k: [v] for k, v in inputs.items()}).to_excel(writer, sheet_name="Inputs", index=False)
        pd.DataFrame([section]).to_excel(writer, sheet_name="Section", index=False)
        pd.DataFrame([bblabl]).to_excel(writer, sheet_name="BBL_ABL", index=False)
        boq.to_excel(writer, sheet_name="BOQ", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------
# Visualization helpers
# ---------------------------

def plot_cross_section(section: dict, title: str = "Cross-Section") -> bytes:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(section["x_in"], section["z_in"], label="Inside slope")
    ax.plot(section["x_out"], section["z_out"], label="Outside slope")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def make_frustum_mesh(Wb, Lb, zb, Wt, Lt, zt, name):
    if go is None:
        return None
    xb, yb = Wb/2.0, Lb/2.0
    xt, yt = Wt/2.0, Lt/2.0
    verts = np.array([
        [-xb, -yb, zb], [ xb, -yb, zb], [ xb,  yb, zb], [ -xb,  yb, zb],
        [-xt, -yt, zt], [ xt, -yt, zt], [ xt,  yt, zt], [ -xt,  yt, zt],
    ])
    faces = np.array([
        [0,1,2], [0,2,3],
        [4,6,5], [4,7,6],
        [0,4,5], [0,5,1],
        [1,5,6], [1,6,2],
        [2,6,7], [2,7,3],
        [3,7,4], [3,4,0],
    ])
    i, j, k = faces[:,0], faces[:,1], faces[:,2]
    return go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], i=i, j=j, k=k,
                     opacity=0.75, flatshading=True, name=name)


def plotly_3d_full_stack(bblabl: dict, D: float):
    if go is None:
        return None
    z_base = -D
    z_gl   = 0.0
    z_tob  = bblabl["Hb"]
    z_tol  = bblabl["Hb"] + bblabl["H_above"]
    traces = []
    traces.append(make_frustum_mesh(bblabl["W_Base"], bblabl["L_Base"], z_base,
                                    bblabl["W_GL"],   bblabl["L_GL"],   z_gl,  "BBL: Base→GL"))
    traces.append(make_frustum_mesh(bblabl["W_GL"],   bblabl["L_GL"],   z_gl,
                                    bblabl["W_TOB"],  bblabl["L_TOB"],  z_tob, "BBL: GL→TOB"))
    traces.append(make_frustum_mesh(bblabl["W_TOB"],  bblabl["L_TOB"],  z_tob,
                                    bblabl["W_TOL"],  bblabl["L_TOL"],  z_tol, "ABL: TOB→TOL"))
    fig = go.Figure(data=[t for t in traces if t is not None])
    fig.update_layout(title="3D Landfill (BBL + ABL)", height=600,
                      scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                                 aspectmode="data"))
    return fig

# ---------------------------
# Streamlit App
# ---------------------------

st.set_page_config(page_title="Landfill Design App", layout="wide")

st.title("Landfill Design, Stability, Visualization & Submission App")

with st.expander("About this app", expanded=False):
    st.markdown(
        """
        This app models landfill capacity using strict BBL/ABL rectangular frusta:
        Base→GL (excavation), GL→TOB (bund interior), and TOB→TOL (fill). It also
        provides Bishop stability, BOQ, and exportable reports with a 3D solid view.
        """
    )

# Wizard tabs
site_tab, geom_tab, stab_tab, boq_tab, report_tab = st.tabs([
    "1) Site & Inputs", "2) Geometry", "3) Stability", "4) BOQ & Costing", "5) Reports/Export"
])

# ---------------------------
# 1) Site & Inputs
# ---------------------------
with site_tab:
    col1, col2, col3 = st.columns(3)
    with col1:
        project_name = st.text_input("Project name", value="Sample Landfill Cell")
        agency_template = st.selectbox("Template (regs)", ["CPCB", "EPA"], index=0)
        waste_type = st.radio("Waste type", ["MSW", "Hazardous"], index=0)
    with col2:
        latitude = st.number_input("Latitude", value=17.3850, format="%.6f")
        longitude = st.number_input("Longitude", value=78.4867, format="%.6f")
        avg_ground_rl = st.number_input("Avg. ground RL (m)", value=100.0)
    with col3:
        water_table_depth = st.number_input("Water table depth below GL (m)", value=5.0)
        inflow_tpd = st.number_input("Waste inflow (TPD)", value=1000.0)
        waste_density_tpm3 = st.number_input("Waste density (t/m³)", value=0.95)
        compaction_factor = st.number_input("Compaction factor", value=0.85)
        lifespan_years_target = st.number_input("Target life (yrs) (0=auto)", value=0.0)

    st.session_state.site = SiteInputs(
        project_name, agency_template, latitude, longitude, avg_ground_rl,
        water_table_depth, waste_type, inflow_tpd, waste_density_tpm3,
        compaction_factor, None if lifespan_years_target <= 0 else lifespan_years_target,
    )

    st.info("Water table clearance warning shown if < 2 m.")
    if water_table_depth < 2.0:
        st.warning("Water table depth < 2 m below GL. Consider raising base or improving liner/drainage.")

    # Footprint definition (rectangular for BBL/ABL exactness)
    st.subheader("Footprint Polygon (rectangular model for BBL/ABL)")
    colA, colB = st.columns(2)
    with colA:
        W_GL = st.number_input("Inner opening width at GL (m)", value=120.0, min_value=1.0)
        L_GL = st.number_input("Inner opening length at GL (m)", value=180.0, min_value=1.0)
        coords = rectangle_polygon(W_GL, L_GL)
        footprint_area = polygon_area(coords)
        st.write(f"Area at GL ≈ **{footprint_area:,.0f} m²**")
    with colB:
        up = st.file_uploader("Import polygon (KML/GeoJSON) [optional]", type=["kml", "geojson", "json"]) 
        if up is not None:
            try:
                if up.name.lower().endswith("kml"):
                    text = up.read().decode("utf-8", errors="ignore")
                    import re
                    m = re.search(r"<coordinates>(.*?)</coordinates>", text, re.S)
                    if m:
                        raw = m.group(1).strip().split()
                        pts = [(float(v.split(",")[0]), float(v.split(",")[1])) for v in raw]
                        xs, ys = zip(*pts)
                        bw, bl = max(xs)-min(xs), max(ys)-min(ys)
                        A = polygon_area(pts)
                        if bw*bl > 0 and A > 0:
                            ratio = bw/bl if bl > 0 else 1.0
                            W_GL = math.sqrt(A*ratio)
                            L_GL = A / max(W_GL,1e-6)
                            coords = rectangle_polygon(W_GL, L_GL)
                            footprint_area = polygon_area(coords)
                            st.success("Polygon imported → fit to equivalent rectangle for BBL/ABL.")
                else:
                    gj = json.load(up)
                    if gj.get("type") == "FeatureCollection":
                        geom0 = gj["features"][0]["geometry"]
                    else:
                        geom0 = gj
                    if geom0["type"] == "Polygon":
                        pts = geom0["coordinates"][0]
                        xs, ys = zip(*pts)
                        bw, bl = max(xs)-min(xs), max(ys)-min(ys)
                        A = polygon_area(pts)
                        if bw*bl > 0 and A > 0:
                            ratio = bw/bl if bl > 0 else 1.0
                            W_GL = math.sqrt(A*ratio)
                            L_GL = A / max(W_GL,1e-6)
                            coords = rectangle_polygon(W_GL, L_GL)
                            footprint_area = polygon_area(coords)
                            st.success("GeoJSON imported → equivalent rectangle applied.")
            except Exception as e:
                st.error(f"Failed to parse polygon: {e}")

    st.session_state.footprint = {"coords": coords, "area": footprint_area, "W_GL": W_GL, "L_GL": L_GL}

# ---------------------------
# 2) Geometry (BBL/ABL inputs + legacy slopes)
# ---------------------------
with geom_tab:
    st.markdown("### BBL/ABL Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        Hb = st.number_input("Bund height Hb (GL→TOB) (m)", value=5.0, min_value=0.0)
        bc = st.number_input("Bund crest width bc (m)", value=4.0, min_value=0.0)
        bund_in_H = st.number_input("Bund inner slope H (per 1V)", value=3.0, min_value=0.0)
        bund_in_V = st.number_input("Bund inner slope V", value=1.0, min_value=0.1)
    with c2:
        D = st.number_input("Excavation depth D (Base→GL) (m)", value=3.0, min_value=0.0)
        excav_H = st.number_input("Excavation slope H (per 1V)", value=1.0, min_value=0.0)
        excav_V = st.number_input("Excavation slope V", value=1.0, min_value=0.1)
        top_ratio_min = st.slider("Min top area ratio (A_TOL/A_TOB)", 0.1, 0.8, 0.3, 0.05)
    with c3:
        # Legacy slopes for stability & visuals
        inside_slope_h = st.number_input("Fill slope above TOB: H", value=3.0)
        inside_slope_v = st.number_input("Fill slope above TOB: V", value=1.0)
        outside_slope_h = st.number_input("Outer slope H (legacy)", value=2.5)
        outside_slope_v = st.number_input("Outer slope V (legacy)", value=1.0)
        lift_thickness = st.number_input("Lift thickness (m)", value=2.5)

    final_height_above_gl = st.number_input("Total height above GL H_final (m) [= Hb + H_above]", value=30.0)

    geom = GeometryInputs(
        inside_slope_h, inside_slope_v, outside_slope_h, outside_slope_v,
        berm_width=4.0, berm_height=5.0, lift_thickness=lift_thickness,
        final_height_above_gl=final_height_above_gl, depth_below_gl=D,
    )

    # Compute BBL/ABL using exact frusta
    m_bund_in = bund_in_H / max(bund_in_V, 1e-6)
    m_fill    = inside_slope_h / max(inside_slope_v, 1e-6)
    m_excav   = excav_H / max(excav_V, 1e-6)

    bblabl = compute_bbl_abl(
        st.session_state.footprint["W_GL"], st.session_state.footprint["L_GL"],
        Hb, bc, m_bund_in, D, m_excav, final_height_above_gl, m_fill,
        top_area_ratio_min=top_ratio_min,
    )
    st.session_state.bblabl = bblabl
    st.session_state.V_total = bblabl["V_total"]

    # Legacy section (for stability/plots) but align base area & V_total for BOQ
    section = generate_section(geom, st.session_state.footprint["area"])
    section["base_area"] = bblabl["A_Base"]

    img = plot_cross_section(section)
    st.image(img, caption="Cross-section (schematic for stability)")

    # Metrics & table
    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("BBL volume (m³)", f"{bblabl['V_BBL']:,.0f}")
    colm2.metric("ABL volume (m³)", f"{bblabl['V_ABL']:,.0f}")
    colm3.metric("Total capacity (m³)", f"{bblabl['V_total']:,.0f}")

    dims_df = pd.DataFrame({
        "Level": ["Base (D)", "GL", "TOB", "TOL"],
        "W (m)": [bblabl["W_Base"], bblabl["W_GL"], bblabl["W_TOB"], bblabl["W_TOL"]],
        "L (m)": [bblabl["L_Base"], bblabl["L_GL"], bblabl["L_TOB"], bblabl["L_TOL"]],
        "Area (m²)": [bblabl["A_Base"], bblabl["A_GL"], bblabl["A_TOB"], bblabl["A_TOL"]],
        "Δh (m)": [bblabl["D"], bblabl["Hb"], bblabl["H_above"], 0.0],
    })
    st.dataframe(dims_df, use_container_width=True)

    # Life calculation using V_total
    capacity_tonnes = bblabl["V_total"] * st.session_state.site.waste_density_tpm3 * st.session_state.site.compaction_factor
    life_days = capacity_tonnes / max(st.session_state.site.inflow_tpd, 1e-6)
    life_years = life_days / 365.0
    st.metric("Estimated life (years)", f"{life_years:,.1f}")

    # 3D model
    st.subheader("3D Landfill model (BBL + ABL)")
    if go is None:
        st.caption("Install Plotly for 3D view: pip install plotly")
    else:
        fig3d = plotly_3d_full_stack(bblabl, D)
        if fig3d:
            st.plotly_chart(fig3d, use_container_width=True)

# ---------------------------
# 3) Stability
# ---------------------------
with stab_tab:
    preset = WASTE_PRESETS[st.session_state.site.waste_type]
    col1, col2, col3 = st.columns(3)
    with col1:
        gamma_unsat = st.number_input("Waste γ (unsat) kN/m³", value=float(preset["gamma_unsat"]))
        gamma_sat = st.number_input("Waste γ (sat) kN/m³", value=float(preset["gamma_sat"]))
        phi = st.number_input("Waste φ (deg)", value=float(preset["phi"]))
        cohesion = st.number_input("Waste cohesion c (kPa)", value=float(preset["c"]))
    with col2:
        soil_phi = st.number_input("Berm soil φ (deg)", value=28.0)
        soil_c = st.number_input("Berm soil c (kPa)", value=5.0)
        soil_gamma = st.number_input("Berm soil γ (kN/m³)", value=18.0)
        ks = st.number_input("Seismic coeff. k_h", value=0.0, step=0.02)
    with col3:
        groundwater_rl = st.number_input("Groundwater RL (m, abs)", value=st.session_state.site.avg_ground_rl - st.session_state.site.water_table_depth)
        target_fos_static = st.number_input("Target FoS static", value=1.5)
        target_fos_seismic = st.number_input("Target FoS seismic", value=1.2)
        n_slices = st.slider("Slices (Bishop)", min_value=24, max_value=120, value=72, step=8)

    stab = StabilityInputs(
        gamma_unsat, gamma_sat, phi, cohesion, soil_phi, soil_c, soil_gamma,
        groundwater_rl, ks, target_fos_static, target_fos_seismic,
    )

    st.write("**Search for critical slip (Bishop simplified)**")
    section_for_stab = generate_section(geom, st.session_state.footprint["area"])  # independent from BBL/ABL
    FoS, best_params, df_slices = grid_search_bishop(section_for_stab, stab, n_slices)

    colA, colB = st.columns([1,1])
    with colA:
        st.metric("Critical FoS", f"{FoS:0.3f}")
        if FoS < target_fos_static:
            st.error(f"FoS below target {target_fos_static}")
        else:
            st.success("FoS meets target")
        st.dataframe(df_slices.describe())
    with colB:
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(section_for_stab["x_in"], section_for_stab["z_in"], label="Inside slope")
        th = np.linspace(0, 2*np.pi, 400)
        cx, cz, r = best_params.get("cx",0), best_params.get("cz",-10), best_params.get("r",50)
        ax.plot(cx + r*np.cos(th), cz + r*np.sin(th), linestyle="--", label="Critical slip")
        ax.axhline(0, linewidth=0.8)
        ax.legend(); ax.grid(True, alpha=0.3)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight"); plt.close(fig)
        st.image(buf.getvalue(), caption="Bishop slip circle")

    st.download_button("Download Slice Table (CSV)", data=df_slices.to_csv(index=False).encode("utf-8"), file_name="slice_table.csv", mime="text/csv")

# ---------------------------
# 4) BOQ & Costing
# ---------------------------
with boq_tab:
    st.subheader("Unit Rates (editable)")
    rates = {}
    cols = st.columns(3)
    for i, (k, v) in enumerate(DEFAULT_RATES.items()):
        with cols[i % 3]:
            rates[k] = st.number_input(f"{k} (₹)", value=float(v), min_value=0.0)

    liner = WASTE_PRESETS[st.session_state.site.waste_type]["liner"].copy()
    st.markdown("**Liner preset (editable)**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        liner["clay_thk"] = st.number_input("Clay thickness (m)", value=float(liner["clay_thk"]))
        liner["drain_thk"] = st.number_input("Drainage layer (m)", value=float(liner["drain_thk"]))
    with c2:
        liner["hdpe_thk"] = st.number_input("HDPE thickness (m)", value=float(liner["hdpe_thk"]))
        liner["clay_k"] = st.number_input("Clay k (m/s)", value=float(liner["clay_k"]))
    with c3:
        liner["gcl"] = st.checkbox("GCL included", value=bool(liner.get("gcl", True)))
    with c4:
        st.write("")

    # Earthworks approximation (minimum): excavation inside the GL opening
    V_earthworks_approx = st.session_state.bblabl["V_Base_to_GL"]

    # Build section dict (for side area approx) and compute BOQ
    section_for_boq = generate_section(geom, st.session_state.footprint["area"])  # keeps side_area
    df_boq, df_summary = compute_boq(section_for_boq, liner, rates, st.session_state.bblabl["A_Base"], V_earthworks_approx)

    # Persist for export
    st.session_state.boq = df_boq
    st.session_state.summary = df_summary
    st.session_state.section_for_boq = section_for_boq

    st.dataframe(df_boq, use_container_width=True)
    st.dataframe(df_summary)

# ---------------------------
# 5) Reports & Export
# ---------------------------
with report_tab:
    st.subheader("Export")

    input_dump = {
        **asdict(st.session_state.site),
        **asdict(geom),
        "footprint_area_GL": st.session_state.footprint["area"],
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }

    _df_boq = st.session_state.get("boq")
    _df_summary = st.session_state.get("summary")
    section_for_boq = st.session_state.get("section_for_boq")
    if _df_boq is None or _df_summary is None or section_for_boq is None:
        section_for_boq = generate_section(geom, st.session_state.footprint["area"])
        liner_tmp = WASTE_PRESETS[st.session_state.site.waste_type]["liner"].copy()
        _df_boq, _df_summary = compute_boq(section_for_boq, liner_tmp, DEFAULT_RATES, st.session_state.bblabl["A_Base"], st.session_state.bblabl["V_Base_to_GL"])
        st.caption("BOQ not visited—using defaults for export.")

    excel_bytes = export_excel(input_dump, section_for_boq, st.session_state.bblabl, _df_boq, _df_summary)
    st.download_button(
        "Download Excel (Inputs+BBL/ABL+BOQ)",
        data=excel_bytes,
        file_name="landfill_design.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # KML
    kml_bytes = None
    if simplekml is not None:
        kml = simplekml.Kml()
        ls = kml.newlinestring(name="Landfill GL Inner Opening", coords=[(x, y) for x, y in st.session_state.footprint["coords"]])
        ls.extrude = 0
        ls.altitudemode = simplekml.AltitudeMode.clamptoground
        kml_bytes = kml.kml().encode("utf-8")
    if kml_bytes:
        st.download_button("Download KML (GL Inner Opening)", data=kml_bytes, file_name="footprint_gl.kml", mime="application/vnd.google-earth.kml+xml")
    else:
        st.caption("Install simplekml for KML export: pip install simplekml")

    # PDF (minimal report via ReportLab)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=A4)
        w, h = A4
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, h-2*cm, "Landfill Design Report (MVP)")
        c.setFont("Helvetica", 10)
        y = h - 3*cm
        lines = [
            f"Project: {st.session_state.site.project_name}",
            f"Template: {st.session_state.site.agency_template} | Waste: {st.session_state.site.waste_type}",
            f"A_GL: {st.session_state.bblabl['A_GL']:.0f} m² | A_TOB: {st.session_state.bblabl['A_TOB']:.0f} m² | A_TOL: {st.session_state.bblabl['A_TOL']:.0f} m²",
            f"Volumes (m³): BBL={st.session_state.bblabl['V_BBL']:,.0f}, ABL={st.session_state.bblabl['V_ABL']:,.0f}, Total={st.session_state.bblabl['V_total']:,.0f}",
        ]
        for line in lines:
            c.drawString(2*cm, y, line)
            y -= 0.6*cm
        sec_png = plot_cross_section(generate_section(geom, st.session_state.footprint["area"]))
        img = io.BytesIO(sec_png)
        from reportlab.lib.utils import ImageReader
        c.drawImage(ImageReader(img), 2*cm, y-8*cm, width=14*cm, height=8*cm, preserveAspectRatio=True)
        c.showPage(); c.save()
        pdf_bytes = pdf_buf.getvalue()
        st.download_button("Download PDF Report (MVP)", data=pdf_bytes, file_name="landfill_report.pdf", mime="application/pdf")
    except Exception as e:
        st.caption(f"PDF export unavailable: {e}")

    st.divider()
    st.markdown("**Planned Enhancements:** polygonal BBL/ABL via straight-skeleton offsets, exact liner side-area from faces, bund shell earthworks, non-circular stability (Janbu).")

# End of app
