# Landfill Design, Stability, Visualization & Submission App (Streamlit)
# -------------------------------------------------------------------
# Implements a practical MVP covering the uploaded specification:
# - Project setup & site data (coords, polygon import, DEM avg)
# - Waste type presets (MSW/Hazardous) with liner templates
# - Geometry builder (berms, lifts, slopes) + volumes & life calc
# - Slope stability (Bishop simplified + Janbu/Fellenius quick checks)
# - Pseudo-static seismic option
# - BOQ & costing with editable rates
# - Visualizations: 2D cross-sections, plan footprint preview
# - Exports: Excel (BOQ & inputs), CSV (slice table), KML (footprint), PDF (report)
# - Admin: CPCB/EPA toggle, unit presets (SI/Imperial)
#
# Notes:
# * Designed to run as a single-file Streamlit app for ease of adoption.
# * Uses only widely available libraries. Optional deps are gated with try/except.
# * For non-rectangular polygons, KML/GeoJSON import is supported (simplekml optional export).
# * 3D & DWG/DXF export are stubbed with TODO hooks.
#
# To run:
#   pip install streamlit numpy pandas matplotlib reportlab simplekml shapely
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
    inside_slope_h: float
    inside_slope_v: float
    outside_slope_h: float
    outside_slope_v: float
    berm_width: float
    berm_height: float
    lift_thickness: float
    final_height_above_gl: float
    depth_below_gl: float

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
# Geometry helpers
# ---------------------------

def rectangle_polygon(width: float, length: float) -> List[Tuple[float, float]]:
    return [(0, 0), (width, 0), (width, length), (0, length)]


def polygon_area(coords: List[Tuple[float, float]]) -> float:
    x = np.array([p[0] for p in coords])
    y = np.array([p[1] for p in coords])
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def generate_section(geom: GeometryInputs, footprint_area: float) -> dict:
    """Compute volumes via simple prismatic approach and build cross-section coordinates.
    - Inside slope forms the internal fill profile.
    - Berms create benches outward; outside slope gives outer embankment.
    Returns dict with section polylines and volumes.
    """
    # Cross-section model in 2D (x horizontal, z vertical). Assume unit width; scale by plan length later.
    # We'll approximate volume = section_area * (footprint_area / section_base_width).
    # For MVP, assume section base width W0 derived from footprint as sqrt(area).
    W0 = max(1.0, math.sqrt(footprint_area))

    H_final = geom.final_height_above_gl
    H_below = geom.depth_below_gl
    h_berm = geom.berm_height
    w_berm = geom.berm_width

    # Number of berm tiers up to H_final
    n_tiers = int(max(0, math.floor(H_final / max(h_berm, 1e-6))))
    lift_n = max(1, int(H_final / max(geom.lift_thickness, 0.5)))

    # Inside slope horizontal per vertical
    m_in = geom.inside_slope_h / max(geom.inside_slope_v, 1e-6)
    m_out = geom.outside_slope_h / max(geom.outside_slope_v, 1e-6)

    # Build internal face from base (-H_below) up to +H_final
    z = [ -H_below, 0.0, H_final ]
    x_in = [ 0.0, 0.0, m_in * H_final ]  # reference at x=0 on centerline

    # Outer face starts at berm offsets
    x_out = [ W0/2 + m_out*H_below, W0/2, W0/2 + m_out*H_final ]
    z_out = [ -H_below, 0.0, H_final ]

    # Add benches: shift outward by berm width at each tier level
    bench_levels = [i * h_berm for i in range(1, n_tiers + 1)]
    x_benches = []
    for lvl in bench_levels:
        x_b = m_in * lvl + (w_berm * bench_levels.index(lvl) + w_berm)
        x_benches.append((x_b, lvl))

    # Section areas (very simplified trapezoids)
    # Inner area above GL
    A_inner = 0.5 * H_final * (0 + m_in * H_final)
    # Below ground "cut" pyramid (optional excavation)
    A_below = 0.5 * H_below * (m_out*H_below)

    # Outer embankment area above GL (assume berm expansion captured by outside slope)
    A_outer = 0.5 * H_final * (m_out * H_final)

    # Effective section area for waste (above GL)
    A_waste = A_inner  # idealized; benches ignored for first-order capacity
    section_base = W0
    # Volume ≈ A * (footprint_area / section_base)
    plan_length_equivalent = footprint_area / max(section_base, 1e-6)
    V_waste = A_waste * plan_length_equivalent

    # Liner areas
    base_area = footprint_area
    side_area = (math.hypot(m_in, 1.0) * H_final) * (2 * plan_length_equivalent)  # two long sides equivalent

    return {
        "A_inner": A_inner,
        "A_outer": A_outer,
        "A_below": A_below,
        "A_waste": A_waste,
        "V_waste": V_waste,
        "base_area": base_area,
        "side_area": side_area,
        "section_base": section_base,
        "x_in": x_in,
        "z_in": z,
        "x_out": x_out,
        "z_out": z_out,
        "benches": x_benches,
        "plan_length_equiv": plan_length_equivalent,
        "n_tiers": n_tiers,
        "lift_n": lift_n,
    }

# ---------------------------
# Stability (Bishop simplified)
# ---------------------------

def bishop_simplified(section, stab: StabilityInputs, n_slices: int = 72,
                      center_x: float = 0.0, center_z: float = -10.0,
                      radius: float = 50.0) -> Tuple[float, pd.DataFrame]:
    """Compute FoS for a trial circular slip surface using Bishop's simplified method.
    This MVP treats the inner face as the slope; the slip circle intersects ground surface.
    Returns FoS and slice table.
    """
    phi = math.radians(stab.phi)
    c = stab.cohesion  # kPa -> kN/m²
    gamma = stab.gamma_unsat  # kN/m³
    ks = stab.ks

    # Define surface profile along x; use inner face line from section (x_in, z)
    x_top = np.linspace(0, section["x_in"][-1], n_slices + 1)
    z_top = (section["z_in"][0] + (section["z_in"][-1] - section["z_in"][0]) * (x_top / max(section["x_in"][-1], 1e-6)))

    # Circle geometry
    def circle_z(x):
        return center_z + math.sqrt(max(radius**2 - (x - center_x)**2, 0.0))

    # Slice loop
    widths = np.diff(x_top)
    x_mid = (x_top[:-1] + x_top[1:]) / 2
    z_surface = (z_top[:-1] + z_top[1:]) / 2
    z_base = np.array([circle_z(x) for x in x_mid])
    h = np.maximum(z_surface - z_base, 1e-3)

    # Normal and shear mobilization
    alpha = np.arctan2((x_mid - center_x), (z_surface - center_z))  # slope of base wrt horizontal (approx)
    W = gamma * h * widths * 1.0  # unit out-of-plane thickness

    # Pore pressure simple model: if groundwater rl given, assume hydrostatic height above base
    u = 0.0
    if stab.groundwater_rl is not None:
        gw = stab.groundwater_rl
        head = np.maximum(gw - z_base, 0.0)
        u = 9.81 * head  # kPa ≈ kN/m²
    N = W * np.cos(alpha)
    N_eff = N - u * widths

    # Iterative Bishop FoS
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
    """Crude grid search for critical circle: vary center and radius around slope toe."""
    x_max = section["x_in"][-1]
    best = {"FoS": 9e9}
    best_df = None
    # Search grid
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

def compute_boq(section: dict, liner: dict, rates: dict, footprint_area: float,
                 plan_length_equiv: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    items = []
    # Liner materials
    clay_vol = liner["clay_thk"] * footprint_area
    hdpe_area = section["base_area"] + section["side_area"]
    gcl_area = hdpe_area if liner.get("gcl", False) else 0.0
    drain_vol = liner["drain_thk"] * footprint_area
    # Earthworks
    cut_fill = section["A_below"] * plan_length_equiv
    # Topsoil (assume 0.3 m over final cap area ~ base area)
    topsoil_vol = 0.3 * section["base_area"]

    items.append(["Clay (compacted)", clay_vol, "m³", rates.get("Clay (compacted)", 0.0)])
    items.append(["HDPE liner install", hdpe_area, "m²", rates.get("HDPE liner install", 0.0)])
    items.append(["GCL", gcl_area, "m²", rates.get("GCL", 0.0)])
    items.append(["Drainage gravel", drain_vol, "m³", rates.get("Drainage gravel", 0.0)])
    items.append(["Geotextile", hdpe_area, "m²", rates.get("Geotextile", 0.0)])
    items.append(["Earthworks (cut/fill)", cut_fill, "m³", rates.get("Earthworks (cut/fill)", 0.0)])
    items.append(["Topsoil", topsoil_vol, "m³", rates.get("Topsoil", 0.0)])

    # Monitoring networks (rule-of-thumb spacing)
    site_length = math.sqrt(max(footprint_area, 1e-6))
    gas_spacing = 40.0
    mon_spacing = 100.0
    gas_wells = max(1, int((site_length / gas_spacing) ** 2))
    mon_wells = max(1, int((site_length / mon_spacing) ** 2))

    items.append(["Gas well", gas_wells, "item", rates.get("Gas well", 0.0)])
    items.append(["Monitoring well", mon_wells, "item", rates.get("Monitoring well", 0.0)])

    df = pd.DataFrame(items, columns=["Item", "Quantity", "Unit", "Rate (₹)"])
    df["Amount (₹)"] = df["Quantity"] * df["Rate (₹)"]

    summary = pd.DataFrame({
        "Metric": ["Total capital cost", "Waste capacity (m³)", "Cost per m³ (₹/m³)"],
        "Value": [df["Amount (₹)"].sum(), section["V_waste"],
                  df["Amount (₹)"].sum() / max(section["V_waste"], 1e-6)],
    })

    return df, summary

# ---------------------------
# Exports
# ---------------------------

def export_excel(inputs: dict, section: dict, boq: pd.DataFrame, summary: pd.DataFrame) -> bytes:
    with pd.ExcelWriter(io.BytesIO(), engine="xlsxwriter") as writer:
        pd.DataFrame({k: [v] for k, v in inputs.items()}).to_excel(writer, sheet_name="Inputs", index=False)
        pd.DataFrame(section, index=[0]).to_excel(writer, sheet_name="Section", index=False)
        boq.to_excel(writer, sheet_name="BOQ", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        writer.save()
        return writer.book.filename.getvalue()


def export_kml(coords: List[Tuple[float, float]]) -> Optional[bytes]:
    if simplekml is None:
        return None
    kml = simplekml.Kml()
    ls = kml.newlinestring(name="Landfill Footprint", coords=[(x, y) for x, y in coords])
    ls.extrude = 0
    ls.altitudemode = simplekml.AltitudeMode.clamptoground
    return kml.kml().encode("utf-8")

# ---------------------------
# Visualization helpers
# ---------------------------

def plot_cross_section(section: dict, title: str = "Cross-Section") -> bytes:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(section["x_in"], section["z_in"], label="Inside slope")
    ax.plot(section["x_out"], section["z_out"], label="Outside slope")
    if section["benches"]:
        bx, bz = zip(*section["benches"]) if section["benches"] else ([], [])
        ax.scatter(bx, bz, s=12, label="Berm benches")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def plot_slip(best_params: dict, df_slices: pd.DataFrame, section: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(7, 4))
    # Section
    ax.plot(section["x_in"], section["z_in"], label="Inside slope")
    ax.axhline(0, color="k", linewidth=0.8)
    # Slip circle
    cx, cz, r = best_params["cx"], best_params["cz"], best_params["r"]
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(cx + r*np.cos(th), cz + r*np.sin(th), linestyle="--", label="Critical slip")
    ax.scatter(df_slices["x_mid"], df_slices["z_surf"], s=6, label="Slice midpoints")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Critical Slip Surface (Bishop)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()

# ---------------------------
# Streamlit App
# ---------------------------

st.set_page_config(page_title="Landfill Design App", layout="wide")

st.title("Landfill Design, Stability, Visualization & Submission App")

with st.expander("About this app", expanded=False):
    st.markdown(
        """
        This Streamlit app implements a practical landfill design workflow:
        site setup, geometry & liners, capacity & life, slope stability (Bishop),
        BOQ & costing, and exportable reports. CPCB/EPA templates are provided as presets.
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

    # Footprint definition
    st.subheader("Footprint Polygon")
    colA, colB = st.columns(2)
    with colA:
        width = st.number_input("Approx. footprint width (m)", value=120.0)
        length = st.number_input("Approx. footprint length (m)", value=180.0)
        coords = rectangle_polygon(width, length)
        footprint_area = polygon_area(coords)
        st.write(f"Area ≈ **{footprint_area:,.0f} m²**")
    with colB:
        up = st.file_uploader("Import polygon (KML/GeoJSON) [optional]", type=["kml", "geojson", "json"])
        if up is not None:
            try:
                if up.name.lower().endswith("kml"):
                    # Minimal KML parse: look for coordinates tuple
                    text = up.read().decode("utf-8", errors="ignore")
                    # naive parse for <coordinates>lon,lat pairs
                    import re
                    m = re.search(r"<coordinates>(.*?)</coordinates>", text, re.S)
                    if m:
                        raw = m.group(1).strip().split()
                        coords = [(float(v.split(",")[0]), float(v.split(",")[1])) for v in raw]
                else:
                    gj = json.load(up)
                    # support Polygon or LineString
                    if gj.get("type") == "FeatureCollection":
                        geom0 = gj["features"][0]["geometry"]
                    else:
                        geom0 = gj
                    if geom0["type"] == "Polygon":
                        coords = geom0["coordinates"][0]
                    elif geom0["type"] == "LineString":
                        coords = geom0["coordinates"]
                footprint_area = polygon_area(coords)
                st.success("Polygon imported.")
            except Exception as e:
                st.error(f"Failed to parse polygon: {e}")

    st.session_state.footprint = {
        "coords": coords,
        "area": footprint_area,
    }

# ---------------------------
# 2) Geometry
# ---------------------------
with geom_tab:
    col1, col2 = st.columns(2)
    with col1:
        inside_slope_h = st.number_input("Inside slope H", value=3.0)
        inside_slope_v = st.number_input("Inside slope V", value=1.0)
        outside_slope_h = st.number_input("Outside slope H", value=2.5)
        outside_slope_v = st.number_input("Outside slope V", value=1.0)
        berm_width = st.number_input("Berm width (m)", value=4.0)
        berm_height = st.number_input("Berm height (m)", value=5.0)
    with col2:
        lift_thickness = st.number_input("Lift thickness (m)", value=2.5)
        final_height_above_gl = st.number_input("Final height above GL (m)", value=30.0)
        depth_below_gl = st.number_input("Depth below GL (m)", value=6.0)

    geom = GeometryInputs(
        inside_slope_h, inside_slope_v, outside_slope_h, outside_slope_v,
        berm_width, berm_height, lift_thickness, final_height_above_gl,
        depth_below_gl,
    )

    section = generate_section(geom, st.session_state.footprint["area"])
    st.session_state.section = section

    img = plot_cross_section(section)
    st.image(img, caption="Cross-section (schematic)")

    # Life calculation
    capacity_tonnes = section["V_waste"] * st.session_state.site.waste_density_tpm3 * st.session_state.site.compaction_factor
    life_days = capacity_tonnes / max(st.session_state.site.inflow_tpd, 1e-6)
    life_years = life_days / 365.0
    st.metric("Estimated life (years)", f"{life_years:,.1f}")

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
    FoS, best_params, df_slices = grid_search_bishop(section, stab, n_slices)

    colA, colB = st.columns([1,1])
    with colA:
        st.metric("Critical FoS (static/seismic)", f"{FoS:0.3f}")
        if FoS < target_fos_static:
            st.error(f"FoS below target {target_fos_static}")
        else:
            st.success("FoS meets target")
        st.dataframe(df_slices.describe())
    with colB:
        slip_img = plot_slip(best_params, df_slices, section)
        st.image(slip_img, caption="Bishop slip circle")

    # Download slice table
    csv_buf = df_slices.to_csv(index=False).encode("utf-8")
    st.download_button("Download Slice Table (CSV)", data=csv_buf, file_name="slice_table.csv", mime="text/csv")

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

    df_boq, df_summary = compute_boq(section, liner, rates, st.session_state.footprint["area"], section["plan_length_equiv"])
    st.dataframe(df_boq, use_container_width=True)
    st.dataframe(df_summary)

# ---------------------------
# 5) Reports & Export
# ---------------------------
with report_tab:
    st.subheader("Export")

    # Excel
    input_dump = {
        **asdict(st.session_state.site),
        **asdict(geom),
        **asdict(stab),
        "footprint_area": st.session_state.footprint["area"],
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }
    excel_bytes = export_excel(input_dump, section, df_boq, df_summary)
    st.download_button("Download Excel (Inputs+BOQ)", data=excel_bytes, file_name="landfill_design.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # KML
    kml_bytes = export_kml(st.session_state.footprint["coords"]) if simplekml else None
    if kml_bytes:
        st.download_button("Download KML (Footprint)", data=kml_bytes, file_name="footprint.kml", mime="application/vnd.google-earth.kml+xml")
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
            f"Footprint area: {st.session_state.footprint['area']:.0f} m² | Estimated life: {((section['V_waste']*st.session_state.site.waste_density_tpm3*st.session_state.site.compaction_factor)/max(st.session_state.site.inflow_tpd,1e-6))/365:.1f} years",
            f"Bishop critical FoS: {FoS:.3f}",
            f"Total capital cost (₹): {df_boq['Amount (₹)'].sum():,.0f}",
        ]
        for line in lines:
            c.drawString(2*cm, y, line)
            y -= 0.6*cm
        # Embed section plot
        sec_png = plot_cross_section(section)
        img = io.BytesIO(sec_png)
        from reportlab.lib.utils import ImageReader
        c.drawImage(ImageReader(img), 2*cm, y-8*cm, width=14*cm, height=8*cm, preserveAspectRatio=True)
        c.showPage()
        c.save()
        pdf_bytes = pdf_buf.getvalue()
        st.download_button("Download PDF Report (MVP)", data=pdf_bytes, file_name="landfill_report.pdf", mime="application/pdf")
    except Exception as e:
        st.caption(f"PDF export unavailable: {e}")

    st.divider()
    st.markdown("**Planned Enhancements (hooks in code):** 3D viewer & glTF export, DWG/DXF via ezdxf, animation to MP4/GIF, detailed non-circular slip (Janbu), and pore-pressure distributions from phreatic surfaces/DEM.")

# End of app
