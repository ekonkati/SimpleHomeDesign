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
# Utility Functions
# ---------------------------

def rectangle_polygon(width: float, length: float) -> List[Tuple[float, float]]:
    half_w, half_l = width / 2.0, length / 2.0
    return [(-half_w, -half_l), (half_w, -half_l), (half_w, half_l), (-half_w, half_l), (-half_w, -half_l)]

def frustum_volume(h: float, A1: float, A2: float) -> float:
    # A1, A2 are areas. W1*L1, W2*L2
    if h <= 0 or A1 < 0 or A2 < 0: return 0.0
    return h * (A1 + A2 + math.sqrt(A1 * A2)) / 3.0

# ---------------------------
# 3D Plotting Helper Functions
# ---------------------------

def get_footprint_corners(W: float, L: float, Z: float, RL_ref: float) -> List[Tuple[float, float, float]]:
    """Returns the 4 corner (X, Y, Z_abs) coordinates for a rectangular footprint."""
    half_W, half_L = W / 2.0, length / 2.0
    RL = Z + RL_ref
    # Order: [(-W/2, -L/2), (W/2, -L/2), (W/2, L/2), (-W/2, L/2)]
    return [
        (-half_W, -half_L, RL),
        (half_W, -half_L, RL),
        (half_W, half_L, RL),
        (-half_W, half_L, RL)
    ]

def frustum_mesh(W1, L1, Z1, W2, L2, Z2, RL_ref, start_v_index):
    """Generates mesh vertices and faces for a rectangular frustum section."""
    # Vertices (Z1 is 'bottom', Z2 is 'top')
    corners1 = get_footprint_corners(W1, L1, Z1, RL_ref)
    corners2 = get_footprint_corners(W2, L2, Z2, RL_ref)
    verts = corners1 + corners2 # 8 vertices in total

    # Faces (Triangles) connecting the two levels
    faces = []
    for i in range(4):
        i1, i2 = i, (i + 1) % 4  # Bottom corners
        i3, i4 = i + 4, (i + 1) % 4 + 4 # Top corners

        # Face 1: (Z1_i, Z1_{i+1}, Z2_{i+1})
        faces.append([i1, i2, i4])
        # Face 2: (Z1_i, Z2_{i+1}, Z2_i)
        faces.append([i1, i4, i3])

    # Offset the indices by start_v_index
    faces_np = np.array(faces) + start_v_index
    
    X = [v[0] for v in verts]
    Y = [v[1] for v in verts]
    Z_abs = [v[2] for v in verts]
    
    return X, Y, Z_abs, faces_np.tolist(), 8 # Return the vertex count

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
    """
    Computes BBL and ABL volumes and dimensions.
    """
    Hb = max(Hb, 0.0); D = max(D, 0.0); H_target_above = max(H_final - Hb, 0.0)
    
    # 1. Landfill Inner Profile Dimensions (Based on m_bund_in)
    W_TOB = max(W_GL - 2.0 * m_bund_in * Hb, 0.0)
    L_TOB = max(L_GL - 2.0 * m_bund_in * Hb, 0.0)
    
    W_Base = max(W_TOB - 2.0 * m_bund_in * D, 0.0) 
    L_Base = max(L_TOB - 2.0 * m_bund_in * D, 0.0)
    
    A_Base = max(W_Base * L_Base, 0.0) 
    A_GL = max(W_GL * L_GL, 0.0)      
    A_TOB = max(W_TOB * L_TOB, 0.0)   
    
    # Volumes (Waste Capacity):
    V_Base_to_GL = frustum_volume(D, A_Base, A_GL)
    V_GL_to_TOB = frustum_volume(Hb, A_GL, A_TOB) 
    V_BBL = V_Base_to_GL + V_GL_to_TOB

    # 2. ABL Calculation
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
            L_TOL_final = W_TOL_final * R
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

    # 3. Outer Bund Geometry & Volume (Soil)
    W_outer_toe_gl = max(W_GL + 2.0 * m_outer * Hb, 0.0) 
    L_outer_toe_gl = max(L_GL + 2.0 * m_outer * Hb, 0.0)
    W_outer_crest_tob = max(W_TOB + 2.0 * bc, 0.0) 
    L_outer_crest_tob = max(L_TOB + 2.0 * bc, 0.0)
    
    V_Outer_Bund = frustum_volume(Hb, W_outer_toe_gl*L_outer_toe_gl, W_outer_crest_tob*L_outer_crest_tob)
    V_Bund_Soil_Approx = V_Outer_Bund 
    
    # 4. Excavation Profile Dimensions (Based on m_excav)
    W_Excav_Base = max(W_GL - 2.0 * m_excav * D, 0.0) 
    L_Excav_Base = max(L_GL - 2.0 * m_excav * D, 0.0)
    A_Excav_Base = max(W_Excav_Base * L_Excav_Base, 0.0)
    
    V_Native_Soil_Excavated_Approx = frustum_volume(D, A_Excav_Base, A_GL)

    return {
        # Waste/Liner Dimensions
        "W_Base": W_Base, "L_Base": L_Base, "A_Base": A_Base, 
        "W_GL": W_GL, "L_GL": L_GL, "A_GL": A_GL,
        "W_TOB": W_TOB, "L_TOB": L_TOB, "A_TOB": A_TOB,
        "W_TOL": W_TOL_final, "L_TOL": L_TOL_final, "A_TOL": A_TOL_final,
        
        # Excavation Dimensions 
        "W_Excav_Base": W_Excav_Base, "L_Excav_Base": L_Excav_Base, "A_Excav_Base": A_Excav_Base,
        
        # Overall Geometry
        "Hb": Hb, "D": D, 
        "V_BBL": V_BBL, "V_ABL": V_ABL, "V_total": V_BBL + V_ABL,
        "H_above": H_actual_above, "H_final": D + H_actual_above,
        "abl_sections": abl_sections, "N_berms": len([s for s in abl_sections if s["Type"] == "Berm"]), 
        
        # Slopes/Rates/Volumes
        "m_bund_in": m_bund_in, "m_excav": m_excav, "m_fill": m_fill, "m_outer": m_outer,
        "bc": bc, 
        "V_Bund_Soil_Approx": V_Bund_Soil_Approx,
        "V_Native_Soil_Excavated_Approx": V_Native_Soil_Excavated_Approx,
        "W_outer_toe_gl": W_outer_toe_gl, "L_outer_toe_gl": L_outer_toe_gl,
        "W_outer_crest_tob": W_outer_crest_tob, "L_outer_crest_tob": L_outer_crest_tob,
    }


def generate_section(bblabl: dict, outside_slope_h_geom: float, W_int_berm: float, axis: str = "W") -> dict:
    D, Hb, bc = bblabl["D"], bblabl["Hb"], bblabl["bc"]
    
    # ------------------ DIMENSION MAPPING ------------------
    # Select dimensions based on axis for the 2D plot (W-axis cut uses W dimensions, L-axis cut uses L dimensions)
    if axis == "W": 
        W_Base, L_ref = bblabl["W_Base"], bblabl["L_GL"]
        W_GL = bblabl["W_GL"]
        W_TOB = bblabl["W_TOB"]
        W_TOL = bblabl["W_TOL"]
        W_Excav_Base = bblabl["W_Excav_Base"]
        W_outer_toe_gl = bblabl["W_outer_toe_gl"]
        W_outer_crest_tob = bblabl["W_outer_crest_tob"]
        def get_abl_w(section): return section["W_base"], section["W_top"]

    else: # axis == "L"
        W_Base, L_ref = bblabl["L_Base"], bblabl["W_GL"]
        W_GL = bblabl["L_GL"]
        W_TOB = bblabl["L_TOB"]
        W_TOL = bblabl["L_TOL"]
        W_Excav_Base = bblabl["L_Excav_Base"]
        W_outer_toe_gl = bblabl["L_outer_toe_gl"]
        W_outer_crest_tob = bblabl["L_outer_crest_tob"]
        def get_abl_w(section): return section["L_base"], section["L_top"]
    # -------------------------------------------------------------

    m_outer = outside_slope_h_geom
    z0, z1, z2 = -D, 0.0, Hb
    Z_TOL = z2 + bblabl["H_above"]
    
    # 1. Landfill Inner Profile (BBL + ABL) - Blue Line
    x_in_right = [W_Base / 2.0] 
    z_in_right = [z0]
    
    # Slope up from Base to GL
    x_in_right.append(W_GL / 2.0)
    z_in_right.append(z1)
    
    # Slope up from GL to TOB
    x_in_right.append(W_TOB / 2.0)
    z_in_right.append(z2)
    
    # Add ABL steps 
    for section in bblabl["abl_sections"]:
        W_base_abl, W_top_abl = get_abl_w(section)

        if section["Z_base"] > z_in_right[-1] or W_base_abl != x_in_right[-1]*2.0:
            x_in_right.append(W_base_abl / 2.0)
            z_in_right.append(section["Z_base"])
            
        if section["Type"] in ["Fill", "Fill_Cap"]:
            x_in_right.append(W_top_abl / 2.0)
            z_in_right.append(section["Z_top"])
         
    # Final TOL Point
    if not (abs(x_in_right[-1] - W_TOL / 2.0) < 1e-3 and abs(z_in_right[-1] - Z_TOL) < 1e-3):
          x_in_right.append(W_TOL / 2.0)
          z_in_right.append(Z_TOL)
          
    x_in_left = [-x for x in x_in_right];
    z_in_left = z_in_right

    # 2. Outer Profile (Native Soil Bund) - Black Line (ONLY ABOVE GL)
    x_outer_right = []
    z_outer_right = []
    
    # Start at GL (W_GL/2.0, z1)
    x_outer_right.append(W_GL / 2.0)
    z_outer_right.append(z1)

    if Hb > 0:
        # Go from GL inner width to Outer Toe
        x_outer_right.append(W_outer_toe_gl / 2.0)
        z_outer_right.append(z1)

        # Go up to Outer Crest 
        x_outer_right.append(W_outer_crest_tob / 2.0)
        z_outer_right.append(z2)
    
    x_outer_left = [-x for x in x_outer_right]
    z_outer_left = z_outer_right

    # Outer Bund Top Crest (Horizontal black line at TOB)
    x_outer_top_bund_plateau = [-W_outer_crest_tob / 2.0, W_outer_crest_tob / 2.0]
    z_outer_top_bund_plateau = [z2, z2]
    
    # 3. Excavation Profile (Pit Boundary) - Red Line (ONLY BELOW GL)
    x_excav_right = [W_GL / 2.0] 
    z_excav_right = [z1]
    
    # Slope down to the Excavation Base
    x_excav_right.append(W_Excav_Base / 2.0)
    z_excav_right.append(z0)
    
    # Horizontal Excavation Base Plateau
    x_excav_base_plateau = [-W_Excav_Base / 2.0, W_Excav_Base / 2.0]
    z_excav_base_plateau = [z0, z0]

    x_excav_left = [-x for x in x_excav_right]; z_excav_left = z_excav_right
    
    return {
        "x_in_left": x_in_left, "z_in_left": z_in_left, "x_in_right": x_in_right, "z_in_right": z_in_right,
        "x_top_plateau": [-W_TOL / 2.0, W_TOL / 2.0], "z_top_plateau": [Z_TOL, Z_TOL],
        "x_base_plateau": [-W_Base / 2.0, W_Base / 2.0], "z_base_plateau": [z0, z0], # Liner base
        
        "x_outer_left": x_outer_left, "z_outer_left": z_outer_left,
        "x_outer_right": x_outer_right, "z_outer_right": z_outer_right,
        "x_outer_top_bund_plateau": x_outer_top_bund_plateau, "z_outer_top_bund_plateau": z_outer_top_bund_plateau,
        
        # New for Excavation Profile (Red)
        "x_excav_right": x_excav_right, "z_excav_right": z_excav_right,
        "x_excav_left": x_excav_left, "z_excav_left": z_excav_left,
        "x_excav_base_plateau": x_excav_base_plateau, "z_excav_base_plateau": z_excav_base_plateau,
        
        "base_area": bblabl["A_Base"], "plan_length_equiv": L_ref, 
        "x_max_slope": W_outer_crest_tob / 2.0, "z_min_slope": z0, "z_max_slope": Z_TOL,
        "axis": axis # Return the axis
    }

def plot_cross_section(section: dict, title: str = "Cross-Section") -> bytes:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 1. Inner Profile (Landfill Waste - Blue line)
    ax.plot(section["x_in_left"], section["z_in_left"], marker='o', linestyle='-', color='b', label="Landfill Inner Profile")
    ax.plot(section["x_in_right"], section["z_in_right"], marker='o', linestyle='-', color='b')
    ax.plot(section["x_top_plateau"], section["z_top_plateau"], linestyle='-', color='b')
    
    # Landfill Base (Liner Base) - This is the blue line at z0
    ax.plot(section["x_base_plateau"], section["z_base_plateau"], linestyle='-', color='b')
    
    # 2. Outer Profile (Native Soil Bund) - Black line (Above GL only)
    ax.plot(section["x_outer_right"], section["z_outer_right"], linestyle='-', color='k', label="Outer Profile")
    ax.plot(section["x_outer_left"], section["z_outer_left"], linestyle='-', color='k')
    ax.plot(section["x_outer_top_bund_plateau"], section["z_outer_top_bund_plateau"], linestyle='-', color='k') 
    
    # 3. Excavation Profile (Pit Boundary) - RED Line (Below GL only)
    ax.plot(section["x_excav_right"], section["z_excav_right"], linestyle='-', color='r', linewidth=3, label="Excavation Profile")
    ax.plot(section["x_excav_left"], section["z_excav_left"], linestyle='-', color='r', linewidth=3)
    ax.plot(section["x_excav_base_plateau"], section["z_excav_base_plateau"], linestyle='-', color='r', linewidth=3)
    
    # 4. Reference Lines
    ax.axhline(0, color='g', linewidth=1.5, linestyle=':', label="Ground Level (GL)")
    if section["z_in_right"] and len(section["z_in_right"]) > 2 and section["z_in_right"][2] > -1e-3: 
        ax.axhline(section["z_in_right"][2], color='r', linewidth=0.8, linestyle='--', label="Top of Bund (TOB)")
    
    # 5. Final Plot Setup
    ax.set_xlabel(f"{section['axis']}-axis distance (m)"); ax.set_ylabel("z (m)"); ax.set_title(title); ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left'); ax.axis('equal')
    buf = io.BytesIO();
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150); plt.close(fig)
    return buf.getvalue()

# ---------------------------
# Other Functions
# ---------------------------

def grid_search_bishop(section, stab, n_slices=72) -> Tuple[float, dict, pd.DataFrame]:
    # Placeholder for actual stability analysis
    return 1.5, {"FoS": 1.5, "cx": 0.0, "cz": -10.0, "r": 50.0}, pd.DataFrame({"Slice": [1, 2], "FoS": [1.5, 1.6]})

def compute_boq(section: dict, liner: dict, rates: dict, A_base_for_liner: float, V_earthworks_approx: float, V_Bund_Soil_Approx: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_capacity = st.session_state.get("V_total", 1000000)
    # Crude estimation for liner area and costs
    m_bund_in = st.session_state.bblabl["m_bund_in"]
    D = st.session_state.bblabl["D"]
    W_Base = st.session_state.bblabl["W_Base"]
    L_Base = st.session_state.bblabl["L_Base"]
    
    # Liner area is based on the Landfill Inner Profile dimensions (W_Base/L_Base to W_GL/L_GL)
    # Simple estimate: Base Area + 2 * Side Area (W) + 2 * Side Area (L)
    side_len_w = D * math.sqrt(1 + m_bund_in**2) # Sloped length for the W face
    # Note: Trapezoidal sides are complex, this uses GL width for simplicity, overestimating slightly
    side_area_w = st.session_state.bblabl["W_GL"] * side_len_w 
    side_area_l = st.session_state.bblabl["L_GL"] * side_len_w # Assuming same slope for L faces for simplicity
    
    liner_area_approx = A_base_for_liner + 2 * side_area_w + 2 * side_area_l 

    cost_liner = liner_area_approx * rates.get("HDPE liner install", 0)
    cost_earth = (V_earthworks_approx + V_Bund_Soil_Approx) * rates.get("Earthworks (cut/fill) (Total Vol)", 0)
    total_cost = 500000 + cost_liner + cost_earth 

    df = pd.DataFrame({
        "Item": ["Waste Capacity", "Liner System (m²)", "Earthworks (m³)", "Bund Soil (m³)", "Total Cost"], 
        "Quantity": [total_capacity, liner_area_approx, V_earthworks_approx, V_Bund_Soil_Approx, total_cost], 
        "Unit": ["m³", "m²", "m³", "m³", "₹"], 
        "Rate (₹)": [0, rates.get("HDPE liner install", 0), rates.get("Earthworks (cut/fill) (Total Vol)", 0), rates.get("Earthworks (cut/fill) (Total Vol)", 0), 1.0], 
        "Amount (₹)": [0, cost_liner, V_earthworks_approx * rates.get("Earthworks (cut/fill) (Total Vol)", 0), V_Bund_Soil_Approx * rates.get("Earthworks (cut/fill) (Total Vol)", 0), total_cost]
    })
    
    summary = pd.DataFrame({
        "Metric": ["Total capital cost (₹)", "Waste capacity (m³)", "Cost per m³ (₹/m³)"], 
        "Value": [f"{total_cost:,.0f}", f"{total_capacity:,.0f}", f"{total_cost / max(total_capacity, 1e-6):,.2f}"]
    })
    return df, summary

def export_excel(inputs: dict, section: dict, bblabl: dict, boq: pd.DataFrame, summary: pd.DataFrame) -> bytes:
    # Placeholder for actual Excel export
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        input_df = pd.DataFrame(inputs.items(), columns=['Parameter', 'Value'])
        input_df.to_excel(writer, sheet_name='Inputs', index=False)
        
        bblabl_df = pd.DataFrame(bblabl.items(), columns=['Metric', 'Value'])
        bblabl_df.to_excel(writer, sheet_name='Geometry_Results', index=False)
        
        boq.to_excel(writer, sheet_name='BOQ', index=False)
        summary.to_excel(writer, sheet_name='Cost_Summary', index=False)
    
    return output.getvalue()

def plotly_3d_full_stack(bblabl: dict, avg_ground_rl: float):
    """
    Generates a full 3D mesh model of the landfill (Waste, Excavation, Bund).
    """
    if go is None: return None
    
    RL_ref = avg_ground_rl 
    traces = []
    v_count = 0 
    
    # --- 1. Waste Mass / Liner (BBL and ABL) (Blue) ---
    W_Base, L_Base = bblabl["W_Base"], bblabl["L_Base"]
    W_GL, L_GL = bblabl["W_GL"], bblabl["L_GL"]
    D, Hb = bblabl["D"], bblabl["Hb"]
    
    # 1.1 Base Excavation to GL (BBL)
    X_bbl, Y_bbl, Z_bbl, faces_bbl, v_num_bbl = frustum_mesh(W_Base, L_Base, -D, W_GL, L_GL, 0.0, RL_ref, v_count)
    
    # 1.2 GL to TOB section
    W_TOB, L_TOB = bblabl["W_TOB"], bblabl["L_TOB"]
    X_gl_tob, Y_gl_tob, Z_gl_tob, faces_gl_tob, v_num_gl_tob = frustum_mesh(W_GL, L_GL, 0.0, W_TOB, L_TOB, Hb, RL_ref, v_count + v_num_bbl)

    # 1.3 ABL stepped sections (from TOB to TOL)
    X_abl, Y_abl, Z_abl, faces_abl = [], [], [], []
    current_W, current_L, current_Z = W_TOB, L_TOB, Hb
    abl_v_count = v_count + v_num_bbl + v_num_gl_tob
    
    for section in bblabl["abl_sections"]:
        W_top, L_top, Z_top = section["W_top"], section["L_top"], section["Z_top"]
        
        X_seg, Y_seg, Z_seg, faces_seg, v_num_seg = frustum_mesh(current_W, current_L, current_Z, W_top, L_top, Z_top, RL_ref, abl_v_count)
        
        X_abl.extend(X_seg); Y_abl.extend(Y_seg); Z_abl.extend(Z_seg); faces_abl.extend(faces_seg);
        abl_v_count += v_num_seg
        
        current_W, current_L, current_Z = W_top, L_top, Z_top

    # Combine all waste vertices/faces
    all_X_waste = X_bbl + X_gl_tob + X_abl
    all_Y_waste = Y_bbl + Y_gl_tob + Y_abl
    all_Z_waste = Z_bbl + Z_gl_tob + Z_abl
    all_faces_waste = faces_bbl + faces_gl_tob + faces_abl
    
    I_waste = [f[0] for f in all_faces_waste]
    J_waste = [f[1] for f in all_faces_waste]
    K_waste = [f[2] for f in all_faces_waste]
    
    # Waste Mesh Trace (Blue)
    traces.append(
        go.Mesh3d(
            x=all_X_waste, y=all_Y_waste, z=all_Z_waste,
            i=I_waste, j=J_waste, k=K_waste,
            opacity=0.8, color='rgba(0, 0, 255, 0.6)', 
            name='Waste (Landfill Profile)', hoverinfo='name'
        )
    )
    v_count += len(all_X_waste)
    
    # --- 2. Outer Excavation Pit (Red/Yellow boundary representation) ---
    W_Excav_Base, L_Excav_Base = bblabl["W_Excav_Base"], bblabl["L_Excav_Base"]
    
    X_excav, Y_excav, Z_excav, faces_excav, v_num_excav = frustum_mesh(W_GL, L_GL, 0.0, W_Excav_Base, L_Excav_Base, -D, RL_ref, v_count)
    
    I_excav = [f[0] for f in faces_excav]
    J_excav = [f[1] for f in faces_excav]
    K_excav = [f[2] for f in faces_excav]

    # Excavation Mesh Trace (Orange-Yellow)
    traces.append(
        go.Mesh3d(
            x=X_excav, y=Y_excav, z=Z_excav,
            i=I_excav, j=J_excav, k=K_excav,
            opacity=0.8, color='rgba(255, 165, 0, 0.8)', 
            name='Excavation Pit Wall', hoverinfo='name'
        )
    )
    v_count += v_num_excav

    # --- 3. Outer Bund (Black) ---
    W_outer_toe_gl, L_outer_toe_gl = bblabl["W_outer_toe_gl"], bblabl["L_outer_toe_gl"]
    W_outer_crest_tob, L_outer_crest_tob = bblabl["W_outer_crest_tob"], bblabl["L_outer_crest_tob"]
    
    X_bund, Y_bund, Z_bund, faces_bund, v_num_bund = frustum_mesh(W_outer_toe_gl, L_outer_toe_gl, 0.0, W_outer_crest_tob, L_outer_crest_tob, Hb, RL_ref, v_count)
    
    I_bund = [f[0] for f in faces_bund]
    J_bund = [f[1] for f in faces_bund]
    K_bund = [f[2] for f in faces_bund]
    
    # Bund Mesh Trace (Black/Grey)
    traces.append(
        go.Mesh3d(
            x=X_bund, y=Y_bund, z=Z_bund,
            i=I_bund, j=J_bund, k=K_bund,
            opacity=0.9, color='rgba(50, 50, 50, 0.8)', 
            name='Outer Soil Bund', hoverinfo='name'
        )
    )
    v_count += v_num_bund
    
    # --- 4. GL Plane (FIXED: Removed erroneous surfacecolor argument) ---
    gl_corners = get_footprint_corners(W_outer_toe_gl * 1.1, L_outer_toe_gl * 1.1, 0.0, RL_ref)
    
    X_coords = np.unique([c[0] for c in gl_corners[:4]])
    Y_coords = np.unique([c[1] for c in gl_corners[:4]])
    RL_val = gl_corners[0][2]
    Z_matrix = np.full((len(Y_coords), len(X_coords)), RL_val)
    
    traces.append(
        go.Surface(
            x=X_coords,
            y=Y_coords, 
            z=Z_matrix,
            colorscale=[[0, 'rgba(0, 128, 0, 0.2)'], [1, 'rgba(0, 128, 0, 0.2)']],
            showscale=False, 
            opacity=0.3, 
            name='Ground Level',
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="3D Landfill Model (BBL/ABL, Excavation, Bund)", 
        height=700,
        scene=dict(
            xaxis_title="Width (m)", yaxis_title="Length (m)", zaxis_title="RL (m)", 
            aspectmode="data", 
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.25, y=1.25, z=1.25)
            )
        ),
        showlegend=True
    )
    return fig


# ---------------------------
# Streamlit App Execution (Rest of the code remains the same)
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
        bund_in_H = st.number_input("Bund inner slope H (per 1V)", value=geom_init.inside_slope_h, min_value=0.0, help="Slope for Landfill Inner Profile (Waste volume)")
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
    
    # Cross-Section Selection
    st.markdown("### Cross-Section & 3D Visualization")
    axis_col, title_col = st.columns([1, 2])
    
    cross_section_axis = axis_col.radio("View Cross-Section Along:", ["Width (W)", "Length (L)"], key="cross_section_axis")
    plot_axis = "W" if cross_section_axis == "Width (W)" else "L"
    
    bblabl = compute_bbl_abl(
        st.session_state.footprint["W_GL"], st.session_state.footprint["L_GL"],
        Hb, bc, m_bund_in, D, m_excav, final_height_above_gl, m_fill,
        top_area_ratio_min=top_ratio_min, m_outer=m_outer,
        H_int_berm=H_int_berm, W_int_berm=W_int_berm
    )
    st.session_state.bblabl = bblabl
    st.session_state.V_total = bblabl["V_total"]
    
    section = generate_section(bblabl, m_outer, W_int_berm, axis=plot_axis)
    
    title_caption = f"{cross_section_axis} Axis Cross-section: Waste Profile (Blue), Excavation (Red, slope 1:{m_excav:.1f}), Bund (Black)"
    img = plot_cross_section(section, title=title_caption)
    st.image(img, caption=f"Unified Cross-section (Inner Stepped Landfill Profile with {bblabl.get('N_berms', 0)} Berms + Outer Profile) along the {cross_section_axis} axis.")
    
    st.markdown("---")
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
        "x_max_slope": bblabl["W_outer_crest_tob"] / 2.0, "z_min_slope": -st.session_state.bblabl["D"],
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
        "footprint_area_GL": st.session_state.footprint["area"], 
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
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
        st.download_button("Download KML (3D Footprints: Base, GL, TOL)", 
data=kml_bytes, file_name="landfill_3d_footprints.kml", mime="application/vnd.google-earth.kml+xml")
