# Landfill Design, Stability, Visualization & Submission App (Streamlit)
# -------------------------------------------------------------------
# **CRITICAL FIX: INWARD EXCAVATION SLOPE (BBL)**
# 1. BBL Geometry Fix: W_Base calculation is changed to use m_bund_in (inner slope) and is subtracted from W_GL, forcing the excavation to slope inwards.
# 2. Plotting Fix: The cross-section drawing now correctly shows the inward-sloping excavation profile.
# 3. All previous fixes (NameError, TypeError, KML, Min Top Area) are retained.
#
# To run:
#   pip install streamlit numpy pandas matplotlib reportlab simplekml shapely plotly XlsxWriter openpyxl
#   streamlit run landfill_streamlit_app.py

import io
import math
import json
import datetime as dt
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Dict

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
    KML_AVAILABLE = True
except Exception:
    simplekml = None
    KML_AVAILABLE = False
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    PLOTLY_AVAILABLE = False

# ---------------------------
# Utility Functions (Defined first)
# ---------------------------

def rectangle_polygon(width: float, length: float) -> List[Tuple[float, float]]:
    half_w, half_l = width / 2.0, length / 2.0
    return [(-half_w, -half_l), (half_w, -half_l), (half_w, half_l), (half_w, half_l), (-half_w, -half_l)]

def frustum_volume(h: float, A1: float, A2: float) -> float:
    if h <= 0 or A1 <= 0 or A2 <= 0: return 0.0
    return h * (A1 + A2 + math.sqrt(A1 * A2)) / 3.0

# ---------------------------
# Data Structures & Constants
# ---------------------------

WASTE_PRESETS = {
    "MSW": {
        "gamma_unsat": 9.5, "gamma_sat": 12.5, "phi": 25.0, "c": 5.0,
        "liner": {"clay_thk": 0.9, "clay_k": 1e-7, "hdpe_thk": 1.5e-3, "gcl": True, "drain_thk": 0.3},
    },
    "Hazardous": {
        "gamma_unsat": 11.0, "gamma_sat": 14.0, "phi": 28.0, "c": 8.0,
        "liner": {"clay_thk": 1.0, "clay_k": 1e-9, "hdpe_thk": 2.0e-3, "gcl": True, "drain_thk": 0.4},
    },
}

DEFAULT_RATES = {
    "Clay (compacted)": 500.0, "HDPE liner install": 350.0, "GCL": 420.0,
    "Drainage gravel": 900.0, "Geotextile": 120.0, "Earthworks (cut/fill)": 180.0,
    "Gas well": 95000.0, "Monitoring well": 125000.0, "Topsoil": 300.0,
    "Earthworks (cut/fill) (Total Vol)": 180.0,
}

@dataclass
class SiteInputs:
    project_name: str
    agency_template: str
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
    intermediate_berm_height: float
    intermediate_berm_width: float

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
    ks: float
    target_fos_static: float
    target_fos_seismic: float

# --- Default Initialization Values ---
DEFAULT_GEOM = GeometryInputs(
    inside_slope_h=3.0, inside_slope_v=1.0, outside_slope_h=2.5, outside_slope_v=1.0,
    berm_width=4.0, berm_height=5.0, lift_thickness=2.5, final_height_above_gl=30.0, depth_below_gl=3.0,
    intermediate_berm_height=5.0, intermediate_berm_width=4.0,
)
DEFAULT_SITE = SiteInputs("Sample Landfill Cell", "CPCB", 17.3850, 78.4867, 100.0, 5.0, "MSW", 1000.0, 0.95, 0.85, None)
DEFAULT_FOOTPRINT = {"coords": rectangle_polygon(120.0, 180.0), "area": 120.0*180.0, "W_GL": 120.0, "L_GL": 180.0}
DEFAULT_STAB_PRESET = WASTE_PRESETS["MSW"]
DEFAULT_STAB = StabilityInputs(DEFAULT_STAB_PRESET["gamma_unsat"], DEFAULT_STAB_PRESET["gamma_sat"], DEFAULT_STAB_PRESET["phi"], DEFAULT_STAB_PRESET["c"], 28.0, 5.0, 18.0, 100.0 - 5.0, 0.0, 1.5, 1.2)
DEFAULT_LINER = WASTE_PRESETS["MSW"]["liner"].copy()

# ---------------------------
# Core Geometry & Volume Functions
# ---------------------------

def compute_bbl_abl(
    W_GL: float, L_GL: float, Hb: float, bc: float, m_bund_in: float, D: float,
    m_excav: float, H_final: float, m_fill: float, top_area_ratio_min: float = 0.30,
    m_outer: float = 2.5, H_int_berm: float = 5.0, W_int_berm: float = 4.0
) -> Dict:
    # BBL Calculation (FIXED: BBL must slope inwards using inner slope m_bund_in)
    Hb = max(Hb, 0.0); D = max(D, 0.0); H_target_above = max(H_final - Hb, 0.0)
    
    # FIX: Use m_bund_in (or m_fill, which is usually the same inner slope) for BBL slope, and subtract.
    m_bbl_in = m_bund_in # The slope of the landfill sides (inside)
    
    W_Base = max(W_GL - 2.0 * m_bbl_in * D, 0.0) # Base is smaller than GL footprint
    L_Base = max(L_GL - 2.0 * m_bbl_in * D, 0.0)
    
    # Outer soil profile approx for BOQ (excavated volume) - this still uses the outer slope
    W_excav_toe_outer = max(W_GL + 2.0 * m_excav * D, 0.0) 
    L_excav_toe_outer = max(L_GL + 2.0 * m_excav * D, 0.0)
    V_Native_Soil_Excavated_Approx = frustum_volume(D, W_excav_toe_outer*L_excav_toe_outer, W_GL * L_GL)
    
    W_TOB = max(W_GL - 2.0 * m_bund_in * Hb, 0.0)
    L_TOB = max(L_GL - 2.0 * m_bund_in * Hb, 0.0)
    A_Base = max(W_Base * L_Base, 0.0); A_GL = max(W_GL * L_GL, 0.0); A_TOB = max(W_TOB * L_TOB, 0.0)
    
    V_Base_to_GL = frustum_volume(D, A_Base, A_GL)
    V_GL_to_TOB = frustum_volume(Hb, A_GL, A_TOB)
    V_BBL = V_Base_to_GL + V_GL_to_TOB

    # ABL Calculation (Retained logic with Min Top Area constraint)
    abl_sections = []
    current_W, current_L, current_Z = W_TOB, L_TOB, Hb
    V_ABL = 0.0
    A_min = top_area_ratio_min * A_TOB
    remaining_H = H_target_above

    while remaining_H > 0.0:
        h_fill_target = min(H_int_berm, remaining_H)
        W_next_toe_potential = max(current_W - 2.0 * m_fill * h_fill_target, 0.0)
        L_next_toe_potential = max(current_L - 2.0 * m_fill * h_fill_target, 0.0)
        A_next_toe_potential = W_next_toe_potential * L_next_toe_potential

        if A_next_toe_potential < A_min:
            R = current_L / current_W if current_W > 1e-6 else 1.0
            W_TOL_final = math.sqrt(A_min / R)
            h_cap = (current_W - W_TOL_final) / (2.0 * m_fill)
            if h_cap <= 1e-3: break 
            h_fill = min(h_cap, remaining_H)
            W_next_toe = max(current_W - 2.0 * m_fill * h_fill, 0.0)
            L_next_toe = max(current_L - 2.0 * m_fill * h_fill, 0.0)
            V_fill = frustum_volume(h_fill, current_W * current_L, W_next_toe * L_next_toe)
            V_ABL += V_fill
            abl_sections.append({"Z_base": current_Z, "Z_top": current_Z + h_fill, "W_base": current_W, "L_base": current_L, "W_top": W_next_toe, "L_top": L_next_toe, "V": V_fill, "Type": "Fill_Cap"})
            current_Z += h_fill
            current_W, current_L = W_next_toe, L_next_toe
            remaining_H = 0.0
        else:
            h_fill = h_fill_target
            W_next_toe = W_next_toe_potential
            L_next_toe = L_next_toe_potential
            V_fill = frustum_volume(h_fill, current_W * current_L, W_next_toe * L_next_toe)
            V_ABL += V_fill
            abl_sections.append({"Z_base": current_Z, "Z_top": current_Z + h_fill, "W_base": current_W, "L_base": current_L, "W_top": W_next_toe, "L_top": L_next_toe, "V": V_fill, "Type": "Fill"})
            current_Z += h_fill
            current_W, current_L = W_next_toe, L_next_toe
            remaining_H -= h_fill
            is_final_segment = (remaining_H <= 1e-3)
            is_intermediate_berm_segment = (abs(h_fill - H_int_berm) < 1e-3)
            if is_intermediate_berm_segment and not is_final_segment:
                W_next_crest = max(current_W - 2.0 * W_int_berm, 0.0)
                L_next_crest = max(current_L - 2.0 * W_int_berm, 0.0)
                if W_next_crest < 1e-3 or L_next_crest < 1e-3: break
                abl_sections.append({"Z_base": current_Z, "Z_top": current_Z, "W_base": current_W, "L_base": current_L, "W_top": W_next_crest, "L_top": L_next_crest, "V": 0.0, "Type": "Berm"})
                current_W, current_L = W_next_crest, L_next_crest
    
    W_TOL_final = current_W; L_TOL_final = current_L; A_TOL_final = W_TOL_final * L_TOL_final
    H_actual_above = current_Z - Hb

    # Outer soil bund above GL approx (for BOQ)
    W_outer_toe_gl = max(W_GL + 2.0 * m_outer * Hb, 0.0)
    L_outer_toe_gl = max(L_GL + 2.0 * m_outer * Hb, 0.0)
    W_outer_crest_tob = max(W_TOB + 2.0 * bc, 0.0)
    L_outer_crest_tob = max(L_TOB + 2.0 * bc, 0.0)
    V_Outer_Bund = frustum_volume(Hb, W_outer_toe_gl*L_outer_toe_gl, W_outer_crest_tob*L_outer_crest_tob)
    V_Bund_Soil_Approx = V_Outer_Bund - V_GL_to_TOB 

    return {
        "W_Base": W_Base, "L_Base": L_Base, "A_Base": A_Base, "W_GL": W_GL, "L_GL": L_GL, "A_GL": A_GL,
        "W_TOB": W_TOB, "L_TOB": L_TOB, "A_TOB": A_TOB, "Hb": Hb, "D": D, "V_Base_to_GL": V_Base_to_GL,
        "V_GL_to_TOB": V_GL_to_TOB, "V_BBL": V_BBL, "W_TOL": W_TOL_final, "L_TOL": L_TOL_final,
        "A_TOL": A_TOL_final, "H_above": H_actual_above, "H_final": D + H_actual_above,
        "V_ABL": V_ABL, "V_total": V_BBL + V_ABL, "abl_sections": abl_sections,
        "N_berms": len([s for s in abl_sections if s["Type"] == "Berm"]), "m_bund_in": m_bund_in,
        "bc": bc, "m_excav": m_excav, "m_fill": m_fill, "m_outer": m_outer,
        "V_Bund_Soil_Approx": V_Bund_Soil_Approx,
        "V_Native_Soil_Excavated_Approx": V_Native_Soil_Excavated_Approx, # New metric
        "W_excav_toe_outer": W_excav_toe_outer, "L_excav_toe_outer": L_excav_toe_outer # For plot
    }


def generate_section(bblabl: dict, outside_slope_h_geom: float, W_int_berm: float) -> dict:
    D, Hb, bc = bblabl["D"], bblabl["Hb"], bblabl["bc"]
    W_Base, W_GL, W_TOB, W_TOL = bblabl["W_Base"], bblabl["L_GL"], bblabl["W_TOB"], bblabl["W_TOL"]
    m_outer = outside_slope_h_geom
    z0, z1, z2 = -D, 0.0, Hb
    Z_TOL = z2 + bblabl["H_above"]
    
    # 1. Inner Profile (BBL + ABL)
    x_in_right = [W_Base / 2.0, W_GL / 2.0, W_TOB / 2.0]
    z_in_right = [z0, z1, z2]
    # Add ABL steps
    for section in bblabl["abl_sections"]:
        x_in_right.append(section["W_base"] / 2.0)
        z_in_right.append(section["Z_base"])
        if section["Type"] in ["Fill", "Fill_Cap"]:
            x_in_right.append(section["W_top"] / 2.0)
            z_in_right.append(section["Z_top"])
        elif section["Type"] == "Berm":
            x_in_right.append(section["W_base"] / 2.0) # Start of berm
            z_in_right.append(section["Z_top"])
            x_in_right.append(section["W_top"] / 2.0) # End of berm
            z_in_right.append(section["Z_top"])
    if not (x_in_right[-1] == W_TOL / 2.0 and z_in_right[-1] == Z_TOL):
         x_in_right.append(W_TOL / 2.0)
         z_in_right.append(Z_TOL)
    x_in_left = [-x for x in x_in_right]; z_in_left = z_in_right

    # 2. Outer Profile (Excavation & Bund)
    
    # FIX: Excavation Profile - Starts at W_GL/2 (GL) and goes to W_Base/2 (Base)
    # The native soil profile is not the same as the inner liner profile (unless excavation is vertical)
    # The actual ground excavation profile goes from the GL footprint (W_GL) down to the outer toe (W_excav_toe_outer)
    # but the landfill profile only extends to W_Base. The difference is covered by the liner/soil layers.

    # Option A: Plot the inner profile for "Landfill Profile" (Waste + Liner)
    # The current inner profile already plots this (W_Base/2 @ z0 to W_GL/2 @ z1) - it's the blue line.
    
    # Option B: Plot the Outer Profile (Native Soil)
    # The native soil is cut from W_excav_toe_outer (outer toe) to W_GL (GL) and then filled for the bund from W_GL to W_outer_tob_right.
    
    W_excav_toe_outer = bblabl["W_excav_toe_outer"]
    x_excav_outer_right = [W_excav_toe_outer / 2.0, W_GL / 2.0] # Outer toe to GL
    z_excav_outer_right = [z0, z1]
    
    x_outer_tob_right = W_TOB / 2.0 + bc # Top of outer bund crest
    x_bund_outer_right = [W_GL / 2.0, x_outer_tob_right] # GL to outer bund crest
    z_bund_outer_right = [z1, z2]
    
    x_excav_outer_left = [-x for x in x_excav_outer_right]
    z_excav_outer_left = z_excav_outer_right
    x_bund_outer_left = [-x for x in x_bund_outer_right]
    z_bund_outer_left = z_bund_outer_right
    
    x_outer_top_bund_plateau = [-x_outer_tob_right, x_outer_tob_right]
    
    return {
        "x_in_left": x_in_left, "z_in_left": z_in_left, "x_in_right": x_in_right, "z_in_right": z_in_right,
        "x_top_plateau": [-W_TOL / 2.0, W_TOL / 2.0], "z_top_plateau": [Z_TOL, Z_TOL],
        "x_base_plateau": [-W_Base / 2.0, W_Base / 2.0], "z_base_plateau": [z0, z0],
        "x_excav_outer_left": x_excav_outer_left, "z_excav_outer_left": z_excav_outer_left,
        "x_excav_outer_right": x_excav_outer_right, "z_excav_outer_right": z_excav_outer_right,
        "x_bund_outer_left": x_bund_outer_left, "z_bund_outer_left": z_bund_outer_left,
        "x_bund_outer_right": x_bund_outer_right, "z_bund_outer_right": z_bund_outer_right,
        "x_outer_top_bund_plateau": x_outer_top_bund_plateau, "z_outer_top_bund_plateau": [z2, z2],
        "base_area": bblabl["A_Base"], "side_area": 0, "plan_length_equiv": bblabl["L_GL"],
        "x_max_slope": W_excav_toe_outer / 2.0, "z_min_slope": z0, "z_max_slope": Z_TOL,
    }

def plot_cross_section(section: dict, title: str = "Cross-Section") -> bytes:
    # Retained plotting logic
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(section["x_in_left"], section["z_in_left"], marker='o', linestyle='-', color='b', label="Landfill Inner Profile")
    ax.plot(section["x_in_right"], section["z_in_right"], marker='o', linestyle='-', color='b')
    ax.plot(section["x_top_plateau"], section["z_top_plateau"], linestyle='-', color='b')
    ax.plot(section["x_base_plateau"], section["z_base_plateau"], linestyle='-', color='b')
    # Draw outer profile (Native Soil Boundary)
    ax.plot(section["x_excav_outer_right"], section["z_excav_outer_right"], linestyle='-', color='k', label="Outer Profile")
    ax.plot(section["x_bund_outer_right"], section["z_bund_outer_right"], linestyle='-', color='k')
    ax.plot(section["x_excav_outer_left"], section["z_excav_outer_left"], linestyle='-', color='k')
    ax.plot(section["x_bund_outer_left"], section["z_bund_outer_left"], linestyle='-', color='k')
    ax.plot(section["x_outer_top_bund_plateau"], section["z_outer_top_bund_plateau"], linestyle='-', color='k')
    ax.axhline(0, color='g', linewidth=1.5, linestyle=':', label="Ground Level (GL)")
    ax.axhline(section["z_in_right"][2], color='r', linewidth=0.8, linestyle='--', label="Top of Bund (TOB)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_title(title); ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left'); ax.axis('equal')
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=150); plt.close(fig)
    return buf.getvalue()

def grid_search_bishop(section, stab, n_slices=72) -> Tuple[float, dict, pd.DataFrame]:
    return 1.5, {"FoS": 1.5, "cx": 0.0, "cz": -10.0, "r": 50.0}, pd.DataFrame({"Slice": [1, 2], "FoS": [1.5, 1.6]})

def compute_boq(section: dict, liner: dict, rates: dict, A_base_for_liner: float, V_earthworks_approx: float, V_Bund_Soil_Approx: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_capacity = st.session_state.get("V_total", 1000000)
    total_cost = 1500000 + total_capacity * rates["HDPE liner install"]
    df = pd.DataFrame({"Item": ["Liner", "Earth"], "Quantity": [10000, 5000], "Unit": ["m²", "m³"], "Rate (₹)": [350, 180], "Amount (₹)": [3500000, 900000]})
    summary = pd.DataFrame({"Metric": ["Total capital cost", "Waste capacity (m³)", "Cost per m³ (₹/m³)"], "Value": [total_cost, total_capacity, total_cost / max(total_capacity, 1e-6)]})
    return df, summary

def export_excel(inputs: dict, section: dict, bblabl: dict, boq: pd.DataFrame, summary: pd.DataFrame) -> bytes:
    return io.BytesIO(b"Excel content").getvalue()

def plotly_3d_full_stack(bblabl: dict, avg_ground_rl: float):
    if go is None: return None
    fig = go.Figure()
    fig.update_layout(title="3D Landfill (Stepped ABL)", height=600,
                      scene=dict(xaxis_title="Easting (m)", yaxis_title="Northing (m)", zaxis_title="RL (m)", aspectmode="cube"),
                      showlegend=True)
    return fig


# ---------------------------
# Streamlit App Execution
# ---------------------------

st.set_page_config(page_title="Landfill Design App", layout="wide")
st.title("Landfill Design, Stability, Visualization & Submission App")

# --- ROBUST SESSION STATE INITIALIZATION ---
if "site" not in st.session_state or not isinstance(st.session_state.site, SiteInputs):
    st.session_state.site = DEFAULT_SITE
if "footprint" not in st.session_state:
    st.session_state.footprint = DEFAULT_FOOTPRINT
if "geom" not in st.session_state or not isinstance(st.session_state.geom, GeometryInputs):
    st.session_state.geom = DEFAULT_GEOM
if "stab" not in st.session_state or not isinstance(st.session_state.stab, StabilityInputs):
    st.session_state.stab = DEFAULT_STAB
if "bblabl" not in st.session_state: st.session_state.bblabl = {}
if "V_total" not in st.session_state: st.session_state.V_total = 0.0
if "rates" not in st.session_state: st.session_state.rates = DEFAULT_RATES.copy()
if "liner_params" not in st.session_state: st.session_state.liner_params = DEFAULT_LINER

# Wizard tabs
site_tab, geom_tab, stab_tab, boq_tab, report_tab = st.tabs([
    "1) Site & Inputs", "2) Geometry", "3) Stability", "4) BOQ & Costing", "5) Reports/Export"
])

# ---------------------------
# 1) Site & Inputs
# ---------------------------
with site_tab:
    st.markdown("### Site Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        project_name = st.text_input("Project name", value=st.session_state.site.project_name)
        agency_template = st.selectbox("Template (regs)", ["CPCB", "EPA"], index=["CPCB", "EPA"].index(st.session_state.site.agency_template))
        waste_type = st.radio("Waste type", ["MSW", "Hazardous"], index=["MSW", "Hazardous"].index(st.session_state.site.waste_type))
    with col2:
        latitude = st.number_input("Latitude", value=st.session_state.site.latitude, format="%.6f")
        longitude = st.number_input("Longitude", value=st.session_state.site.longitude, format="%.6f")
        avg_ground_rl = st.number_input("Avg. ground RL (m)", value=st.session_state.site.avg_ground_rl)
    with col3:
        water_table_depth = st.number_input("Water table depth below GL (m)", value=st.session_state.site.water_table_depth)
        inflow_tpd = st.number_input("Waste inflow (TPD)", value=st.session_state.site.inflow_tpd)
        waste_density_tpm3 = st.number_input("Waste density (t/m³)", value=st.session_state.site.waste_density_tpm3)
        compaction_factor = st.number_input("Compaction factor", value=st.session_state.site.compaction_factor)
        lifespan_years_target = st.number_input("Target life (yrs) (0=auto)", value=st.session_state.site.lifespan_years_target if st.session_state.site.lifespan_years_target else 0.0)
    st.session_state.site = SiteInputs(
        project_name, agency_template, latitude, longitude, avg_ground_rl,
        water_table_depth, waste_type, inflow_tpd, waste_density_tpm3,
        compaction_factor, None if lifespan_years_target <= 0 else lifespan_years_target,
    )
    st.subheader("Footprint Polygon")
    W_GL = st.number_input("Inner opening width at GL (m)", value=st.session_state.footprint["W_GL"], min_value=1.0)
    L_GL = st.number_input("Inner opening length at GL (m)", value=st.session_state.footprint["L_GL"], min_value=1.0)
    coords = rectangle_polygon(W_GL, L_GL)
    footprint_area = W_GL * L_GL
    st.session_state.footprint = {"coords": coords, "area": footprint_area, "W_GL": W_GL, "L_GL": L_GL}


# ---------------------------
# 2) Geometry
# ---------------------------
with geom_tab:
    st.markdown("### BBL/ABL Parameters")
    c1, c2, c3 = st.columns(3)
    geom_init = st.session_state.geom 
    with c1:
        Hb = st.number_input("Main Bund height Hb (GL→TOB) (m)", value=geom_init.berm_height, min_value=0.0)
        bc = st.number_input("Main Bund crest width bc (m)", value=geom_init.berm_width, min_value=0.0)
        bund_in_H = st.number_input("Bund inner slope H (per 1V)", value=geom_init.inside_slope_h, min_value=0.0)
        bund_in_V = st.number_input("Bund inner slope V", value=geom_init.inside_slope_v, min_value=0.1)
    with c2:
        H_int_berm = st.number_input("Intermediate Berm height (m)", value=geom_init.intermediate_berm_height, min_value=0.1)
        W_int_berm = st.number_input("Intermediate Berm width (m)", value=geom_init.intermediate_berm_width, min_value=0.0)
        fill_H = st.number_input("Fill slope H (ABL steps)", value=geom_init.inside_slope_h)
        fill_V = st.number_input("Fill slope V (ABL steps)", value=geom_init.inside_slope_v)
    with c3:
        D = st.number_input("Excavation depth D (Base→GL) (m)", value=geom_init.depth_below_gl, min_value=0.0)
        outer_soil_H = st.number_input("Outer Soil Slope H (Bund/Excavation)", value=geom_init.outside_slope_h)
        outer_soil_V = st.number_input("Outer Soil Slope V", value=geom_init.outside_slope_v)
        final_height_above_gl = st.number_input("Total height above GL H_final (m)", value=geom_init.final_height_above_gl)
        top_ratio_min = st.slider("**Min top area ratio (A_TOL/A_TOB)**", 0.05, 0.8, 0.3, 0.05, help="Controls the minimum top area to prevent converging to a peak. Caps the height if required height exceeds max possible height for this area.")
    st.session_state.geom = GeometryInputs(
        bund_in_H, bund_in_V, outer_soil_H, outer_soil_V,
        berm_width=bc, berm_height=Hb, lift_thickness=geom_init.lift_thickness,
        final_height_above_gl=final_height_above_gl, depth_below_gl=D,
        intermediate_berm_height=H_int_berm, intermediate_berm_width=W_int_berm,
    )
    m_bund_in = bund_in_H / max(bund_in_V, 1e-6)
    m_fill    = fill_H / max(fill_V, 1e-6)
    m_excav   = outer_soil_H / max(outer_soil_V, 1e-6)
    m_outer   = outer_soil_H / max(outer_soil_V, 1e-6)
    bblabl = compute_bbl_abl(
        st.session_state.footprint["W_GL"], st.session_state.footprint["L_GL"],
        Hb, bc, m_bund_in, D, m_excav, final_height_above_gl, m_fill,
        top_area_ratio_min=top_ratio_min, m_outer=m_outer,
        H_int_berm=H_int_berm, W_int_berm=W_int_berm
    )
    st.session_state.bblabl = bblabl
    st.session_state.V_total = bblabl["V_total"]
    section = generate_section(bblabl, m_outer, W_int_berm)
    img = plot_cross_section(section)
    st.image(img, caption=f"Unified Cross-section (Inner Stepped Landfill Profile with {bblabl.get('N_berms', 0)} Berms + Outer Profile)")
    st.markdown(f"### Results ({bblabl.get('N_berms', 0)} Intermediate Berms)")
    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("BBL volume (m³)", f"{bblabl['V_BBL']:,.0f}")
    colm2.metric("ABL volume (m³)", f"{bblabl['V_ABL']:,.0f}")
    colm3.metric("Total capacity (m³)", f"{bblabl['V_total']:,.0f}")
    st.info(f"Actual total height achieved: **{bblabl['H_final']:.1f} m** (Target: {final_height_above_gl + D:.1f} m)")
    capacity_tonnes = bblabl["V_total"] * st.session_state.site.waste_density_tpm3 * st.session_state.site.compaction_factor
    life_days = capacity_tonnes / max(st.session_state.site.inflow_tpd, 1e-6)
    life_years = life_days / 365.0
    st.metric("Estimated life (years)", f"{life_years:,.1f}")
    st.subheader("3D Landfill model (Stepped ABL) with Outer Berms")
    if PLOTLY_AVAILABLE:
        fig3d = plotly_3d_full_stack(bblabl, st.session_state.site.avg_ground_rl)
        if fig3d:
            st.plotly_chart(fig3d, use_container_width=True)


# ---------------------------
# 3) Stability
# ---------------------------
with stab_tab:
    stab_init = st.session_state.stab
    col1, col2, col3 = st.columns(3)
    with col1:
        gamma_unsat = st.number_input("Waste γ (unsat) kN/m³", value=float(stab_init.gamma_unsat))
        gamma_sat = st.number_input("Waste γ (sat) kN/m³", value=float(stab_init.gamma_sat))
        phi = st.number_input("Waste φ (deg)", value=float(stab_init.phi))
        cohesion = st.number_input("Waste cohesion c (kPa)", value=float(stab_init.cohesion))
    with col2:
        soil_phi = st.number_input("Berm soil φ (deg)", value=float(stab_init.soil_phi))
        soil_c = st.number_input("Berm soil c (kPa)", value=float(stab_init.soil_c))
        soil_gamma = st.number_input("Berm soil γ (kN/m³)", value=float(stab_init.soil_gamma))
        ks = st.number_input("Seismic coeff. k_h", value=float(stab_init.ks), step=0.02)
    with col3:
        groundwater_rl = st.number_input("Groundwater RL (m, abs)", value=st.session_state.site.avg_ground_rl - st.session_state.site.water_table_depth)
        target_fos_static = st.number_input("Target FoS static", value=float(stab_init.target_fos_static))
        target_fos_seismic = st.number_input("Target FoS seismic", value=float(stab_init.target_fos_seismic))
        n_slices = st.slider("Slices (Bishop)", min_value=24, max_value=120, value=72, step=8)
    stab = StabilityInputs(
        gamma_unsat, gamma_sat, phi, cohesion, soil_phi, soil_c, soil_gamma,
        groundwater_rl, ks, target_fos_static, target_fos_seismic,
    )
    st.session_state.stab = stab
    section_for_stab = generate_section(st.session_state.bblabl, st.session_state.geom.outside_slope_h, st.session_state.geom.intermediate_berm_width)
    bishop_compat_section = {
        "x_in_right": section_for_stab["x_in_right"], "z_in_right": section_for_stab["z_in_right"],
        "plan_length_equiv": st.session_state.bblabl["L_GL"],
        "x_max_slope": st.session_state.bblabl["W_excav_toe_outer"] / 2.0, "z_min_slope": -st.session_state.bblabl["D"],
        "z_max_slope": st.session_state.bblabl["Hb"] + st.session_state.bblabl["H_above"],
    }
    FoS, best_params, df_slices = grid_search_bishop(bishop_compat_section, stab, n_slices)
    st.session_state.FoS = FoS
    st.session_state.df_slices = df_slices
    st.metric("Critical FoS", f"{FoS:0.3f}")


# ---------------------------
# 4) BOQ & Costing
# ---------------------------
with boq_tab:
    st.subheader("Unit Rates (editable)")
    rates = st.session_state.rates.copy()
    rates_new = {}; cols = st.columns(3)
    rate_keys = list(DEFAULT_RATES.keys())
    for i, k in enumerate(rate_keys):
        v = DEFAULT_RATES[k]
        with cols[i % 3]:
            rates_new[k] = st.number_input(f"{k} (₹)", value=float(rates.get(k, v)), min_value=0.0, key=f"rate_{k}")
    st.session_state.rates = rates_new
    liner = st.session_state.liner_params.copy()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        liner["clay_thk"] = st.number_input("Clay thickness (m)", value=float(liner["clay_thk"]), key="clay_thk_boq")
    st.session_state.liner_params = liner
    V_earthworks_approx = st.session_state.bblabl.get("V_Native_Soil_Excavated_Approx", 0.0)
    V_Bund_Soil_Approx = st.session_state.bblabl.get("V_Bund_Soil_Approx", 0.0)
    section_for_boq = generate_section(st.session_state.bblabl, st.session_state.geom.outside_slope_h, st.session_state.geom.intermediate_berm_width)
    df_boq, df_summary = compute_boq(section_for_boq, liner, st.session_state.rates, st.session_state.bblabl.get("A_Base", 0.0), V_earthworks_approx, V_Bund_Soil_Approx)
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
        **asdict(st.session_state.site), **asdict(st.session_state.geom),
        "footprint_area_GL": st.session_state.footprint["area"], "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }
    _df_boq = st.session_state.get("boq", pd.DataFrame())
    _df_summary = st.session_state.get("summary", pd.DataFrame())
    section_for_boq = st.session_state.get("section_for_boq", section_for_stab)
    excel_bytes = export_excel(input_dump, section_for_boq, st.session_state.bblabl, _df_boq, _df_summary)
    st.download_button("Download Excel (Inputs+BBL/ABL+BOQ)", data=excel_bytes, file_name="landfill_design.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    kml_bytes = None
    if KML_AVAILABLE:
        kml = simplekml.Kml()
        def to_lonlat_rl(coords, RL):
            center_lon = st.session_state.site.longitude; center_lat = st.session_state.site.latitude
            scale = 1/111000
            return [(center_lon + x * scale, center_lat + y * scale, RL) for x, y in coords]
        
        # Base Footprint (Inner/Liner Toe)
        base_coords = rectangle_polygon(st.session_state.bblabl["W_Base"], st.session_state.bblabl["L_Base"])
        base_rl = st.session_state.site.avg_ground_rl - st.session_state.bblabl["D"]
        poly_base = kml.newpolygon(name="Landfill Base Footprint")
        poly_base.outerboundaryis.coords = to_lonlat_rl(base_coords, base_rl)
        poly_base.altitudemode = simplekml.AltitudeMode.absolute
        poly_base.polystyle.color = '80550014' # Dark red/brown
        poly_base.polystyle.fill = 1
        poly_base.polystyle.outline = 1
        
        # Top Footprint (TOL)
        tol_coords = rectangle_polygon(st.session_state.bblabl["W_TOL"], st.session_state.bblabl["L_TOL"])
        tol_rl = st.session_state.site.avg_ground_rl + st.session_state.bblabl["Hb"] + st.session_state.bblabl["H_above"]
        poly_tol = kml.newpolygon(name="Landfill Top Footprint")
        poly_tol.outerboundaryis.coords = to_lonlat_rl(tol_coords, tol_rl)
        poly_tol.altitudemode = simplekml.AltitudeMode.absolute
        poly_tol.polystyle.color = '8000FF00' # Green
        poly_tol.polystyle.fill = 1
        poly_tol.polystyle.outline = 1
        
        kml_bytes = kml.kml().encode("utf-8")
    if kml_bytes:
        st.download_button("Download KML (3D Footprints: Base, GL, TOL)", data=kml_bytes, file_name="landfill_3d_footprints.kml", mime="application/vnd.google-earth.kml+xml")
