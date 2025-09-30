import io
import math
from dataclasses import dataclass
from typing import Dict
from datetime import datetime
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Optional libs
try:
    import ezdxf
except Exception:
    ezdxf = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.lib.units import mm
except Exception:
    pdfcanvas = None

# ------------------------- Helpers & Constants ------------------------- #
GAMMA_W = 9.81  # kN/m3
STEEL_DENSITY = 7850  # kg/m3

# Bar areas (mm^2)
BAR_AREAS = {6: 28.27, 8: 50.27, 10: 78.54, 12: 113.10, 16: 201.06, 20: 314.16}

def as_per_m(dia_mm: int, spacing_mm: int) -> float:
    """Calculates steel area (mm2/m) based on bar diameter and spacing."""
    if dia_mm not in BAR_AREAS:
        return 0.0
    return (BAR_AREAS[dia_mm] / spacing_mm) * 1000

# NEW FUNCTION: Create a serializable state dictionary for saving
def create_state_dict(geo, soil, loads, mat, bearing, project_name, wall_type, design_state):
    """Creates a dictionary of all current inputs for saving."""
    geo_dict = geo.__dict__.copy()
    if 'B' in geo_dict:
        del geo_dict['B'] 
    
    # Store widget values from session state to ensure latest are saved
    return {
        'project_name': st.session_state['project_name'],
        'wall_type': st.session_state['wall_type'],
        'design_state': st.session_state['design_state'],
        'geo': geo_dict,
        'soil': soil.__dict__,
        'loads': loads.__dict__,
        'mat': mat.__dict__,
        'bearing': bearing.__dict__,
        'version': 1.2, # Updated version
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# ------------------------- Dataclasses ------------------------- #

@dataclass
class Soil:
    gamma: float
    phi: float
    c_base: float
    gamma_sub: float
    gamma_fill: float

@dataclass
class Geometry:
    H: float
    Df: float
    t_stem: float
    t_base: float
    toe: float
    heel: float
    B: float = 0.0
    shear_key_depth: float = 0.0

@dataclass
class Loads:
    surcharge_q: float
    gwl_h_from_base: float
    seismic_kh: float
    seismic_kv: float
    use_seismic: bool

@dataclass
class Materials:
    fck: int
    fy: int
    gamma_c: float
    cover: int

@dataclass
class Bearing:
    SBC_allow: float
    mu_base: float
    include_passive: bool
    passive_reduction: float

# ------------------------- Core Calculation Functions ------------------------- #

def calc_K(phi: float, state: str, loads: Loads) -> Dict[str, float]:
    """Calculates Rankine and other coefficients."""
    phi_rad = math.radians(phi)
    
    if state == 'Active':
        Ka = (1 - math.sin(phi_rad)) / (1 + math.sin(phi_rad))
        Kp = (1 + math.sin(phi_rad)) / (1 - math.sin(phi_rad))
        delta = 0.0
        return {'Ka': Ka, 'Kp': Kp, 'delta': delta}
    elif state == 'At-Rest':
        Ko = 1 - math.sin(phi_rad)
        Kp = (1 + math.sin(phi_rad)) / (1 - math.sin(phi_rad))
        delta = 0.0
        return {'Ka': Ko, 'Kp': Kp, 'delta': delta}
    elif state == 'Seismic (Mononobe-Okabe)':
        Ka = (1 - math.sin(phi_rad)) / (1 + math.sin(phi_rad)) 
        Kp = (1 + math.sin(phi_rad)) / (1 - math.sin(phi_rad))
        delta = math.radians(phi / 2) 
   
        if loads.seismic_kh > 0:
            psi = math.radians(math.atan(loads.seismic_kh / (1-loads.seismic_kv)))
            num = (math.cos(phi_rad-psi)**2)
            den = (math.cos(psi)**2) * math.cos(psi+delta) * (1 + math.sqrt(math.sin(phi_rad+delta) * math.sin(phi_rad-psi) / (math.cos(psi+delta))))**2
            KaE = num/den
        else:
            KaE = Ka
            
        return {'Ka': KaE, 'Kp': Kp, 'delta': delta, 'KaE': KaE}
    else:
        return {'Ka': 0.0, 'Kp': 0.0, 'delta': 0.0}

def pressures(H: float, soil: Soil, loads: Loads, design_state: str) -> Dict[str, float]:
    """Calculates earth and water pressures (UNFACTORED)."""
    
    K = calc_K(soil.phi, design_state, loads) 
    Ka = K['Ka']
    delta = K['delta']
    
    # --- Earth Pressure (Triangular) ---
    p0_earth = Ka * soil.gamma_fill * H 
    P_earth = 0.5 * p0_earth * H
    y_earth = H / 3
    
    # --- Surcharge Pressure (Uniform) ---
    p_surcharge = Ka * loads.surcharge_q
    P_surcharge = p_surcharge * H
    y_surcharge = H / 2
    
    # --- Water Pressure ---
    gwl = loads.gwl_h_from_base
    if gwl > 0:
        p0_water = GAMMA_W * gwl
        P_water = 0.5 * p0_water * gwl
        y_water = gwl / 3
    else:
        p0_water = 0.0
        P_water = 0.0
        y_water = 0.0
        
    # --- Seismic Pressure (Inverted Triangular - Mononobe-Okabe Approx) ---
    P_dynamic = 0.0
    y_seismic = 0.0
    p0_seismic = 0.0
    
    if loads.use_seismic and design_state == 'Seismic (Mononobe-Okabe)':
        KaE = K['KaE']
        
        P_static = 0.5 * Ka * soil.gamma_fill * H**2
        P_total_seismic = 0.5 * KaE * soil.gamma_fill * H**2
        
        P_earth = P_total_seismic 
        P_dynamic = P_total_seismic - P_static 
        y_earth = H / 3 
        y_seismic = 0.6 * H 
        
    return {
        'Ka': Ka, 'delta': math.degrees(delta), 'KaE': K.get('KaE', Ka),
        'p0_earth': p0_earth, 'P_earth': P_earth, 'y_earth': y_earth,
        'p_surcharge': p_surcharge, 'P_surcharge': P_surcharge, 'y_surcharge': y_surcharge,
        'p0_water': p0_water, 'P_water': P_water, 'y_water': y_water,
        'P_seismic': P_dynamic, 'y_seismic': y_seismic, 'p0_seismic': p0_seismic,
    }

# NEW: Function to calculate vertical weights (UNFACTORED)
def calculate_vertical_weights(geo: Geometry, soil: Soil, loads: Loads, pres: Dict[str, float]) -> Dict[str, float]:
    """Calculates all vertical forces (weights and uplift, UNFACTORED)."""
    gamma_c = 25.0 
    
    # W1: Stem
    W1 = geo.t_stem * (geo.H + geo.t_base) * gamma_c
    x1 = geo.toe + geo.t_stem / 2
    
    # W2: Base
    W2 = geo.B * geo.t_base * gamma_c
    x2 = geo.B / 2
    
    # W3: Heel soil 
    W3 = geo.heel * geo.H * soil.gamma_fill
    x3 = geo.toe + geo.t_stem + geo.heel / 2
    
    # W4: Surcharge over heel (LL)
    W4 = loads.surcharge_q * geo.heel
    x4 = geo.toe + geo.t_stem + geo.heel / 2
    
    # W5: Water uplift (DL)
    gwl = loads.gwl_h_from_base
    if gwl > 0:
        p_uplift_base = GAMMA_W * gwl
        W5_uplift = 0.5 * p_uplift_base * geo.B 
        x5_uplift = geo.toe + geo.t_stem + geo.heel * (2/3) 
    else:
        W5_uplift = 0.0
        x5_uplift = 0.0
        
    # W_V_earth: Vertical component of Earth Pressure (DL)
    P_V_earth = pres['P_earth'] * math.sin(math.radians(pres['delta']))
    x_V_earth = geo.toe + geo.t_stem
    
    # Grouping weights by type for load combinations
    W_DL = W1 + W2 + W3 + P_V_earth
    M_DL = (W1 * x1) + (W2 * x2) + (W3 * x3) + (P_V_earth * x_V_earth)
    
    W_LL = W4
    M_LL = W4 * x4
    
    W_UPLIFT = W5_uplift
    M_UPLIFT = W5_uplift * x5_uplift
    
    return {
        'W1': W1, 'x1': x1, 'W2': W2, 'x2': x2, 'W3': W3, 'x3': x3, 'W4': W4, 'x4': x4, 
        'W5_uplift': W5_uplift, 'x5_uplift': x5_uplift, 'P_V_earth': P_V_earth, 
        'W_DL': W_DL, 'M_DL': M_DL, 'W_LL': W_LL, 'M_LL': M_LL, 'W_UPLIFT': W_UPLIFT, 'M_UPLIFT': M_UPLIFT
    }

def stability(geo: Geometry, soil: Soil, loads: Loads, bearing: Bearing, pres: Dict[str, float]) -> Dict[str, float]:
    """Performs global stability checks (SLS) and finds worst-case ULS bearing pressure."""
    
    # --- IS 456 ULS Load Combinations (Simplified for Retaining Walls) ---
    # Case 1: Max Overturning/Max Bearing (Factor on stabilizing DL = 1.5)
    # Case 2: Min Resisting/Max Uplift (Factor on stabilizing DL = 0.9)
    # Lateral Loads (Earth, Water, Surcharge) always factored by 1.5.

    gamma_H = 1.5 # Factor for lateral (overturning) loads
    gamma_V_max = 1.5 # Factor for maximum vertical (resisting) loads
    gamma_V_min = 0.9 # Factor for minimum vertical (resisting) loads
    
    W_unfactored = calculate_vertical_weights(geo, soil, loads, pres)

    # ------------------ Lateral Forces (Common to both cases) ------------------
    P_H_earth = pres['P_earth'] * math.cos(math.radians(pres['delta']))
    P_H_surcharge = pres['P_surcharge'] 
    
    P_H_total = P_H_earth + P_H_surcharge + pres['P_water'] + pres['P_seismic']
    
    Mo_unfactored = (P_H_earth * pres['y_earth']) + \
                    (P_H_surcharge * pres['y_surcharge']) + \
                    (pres['P_water'] * pres['y_water']) + \
                    (pres['P_seismic'] * pres['y_seismic'])
    
    # ULS Overturning Moment (Mo)
    Mo_uls = Mo_unfactored * gamma_H
    
    # ULS Horizontal Force (H)
    H_uls = P_H_total * gamma_H

    # ------------------ Case 1: Max Resisting Vertical Load (ULS 1.5) ------------------
    V1_uls = (W_unfactored['W_DL'] * gamma_V_max) + (W_unfactored['W_LL'] * gamma_H) - (W_unfactored['W_UPLIFT'] * gamma_V_max)
    Mr1_uls = (W_unfactored['M_DL'] * gamma_V_max) + (W_unfactored['M_LL'] * gamma_H) - (W_unfactored['M_UPLIFT'] * gamma_V_max)
    
    # Case 2: Min Resisting Vertical Load (ULS 0.9)
    V2_uls = (W_unfactored['W_DL'] * gamma_V_min) + (W_unfactored['W_LL'] * gamma_H) - (W_unfactored['W_UPLIFT'] * gamma_V_min)
    Mr2_uls = (W_unfactored['M_DL'] * gamma_V_min) + (W_unfactored['M_LL'] * gamma_H) - (W_unfactored['M_UPLIFT'] * gamma_V_min)

    # ------------------ Stability Checks (SLS - Unfactored Loads) ------------------
    V_sls = W_unfactored['W_DL'] + W_unfactored['W_LL'] - W_unfactored['W_UPLIFT']
    Mr_sls = W_unfactored['M_DL'] + W_unfactored['M_LL'] - W_unfactored['M_UPLIFT']
    Mo_sls = Mo_unfactored
    
    # 1. Overturning (FOS_OT) - SLS Check
    FOS_OT = Mr_sls / Mo_sls if Mo_sls != 0 else 99.0
        
    # 2. Sliding (FOS_SL) - SLS Check
    Kp = calc_K(soil.phi, 'Active', loads)['Kp'] 
    Dp = geo.Df + geo.shear_key_depth
    Pp = 0.5 * Kp * soil.gamma * Dp**2 
    
    Pp_allow = Pp * bearing.passive_reduction if bearing.include_passive else 0.0
        
    F_cohesion = soil.c_base * geo.B
    F_friction = V_sls * bearing.mu_base # Use SLS Vertical Load
    
    Fr_sls = F_friction + Pp_allow + F_cohesion
    FOS_SL = Fr_sls / P_H_total if P_H_total != 0 else 99.0
        
    # 3. Bearing Pressure (q_max/q_min) - ULS Check for Max Pressure
    
    # Check both ULS cases for worst bearing pressure (q_max)
    
    # Case 1: Max V (V1_uls, Mr1_uls) -> Max q_max
    x_res1 = Mr1_uls / V1_uls
    e1 = x_res1 - geo.B / 2
    
    if abs(e1) <= geo.B / 6:
        q_max1 = (V1_uls / geo.B) * (1 + 6 * abs(e1) / geo.B)
        q_min1 = (V1_uls / geo.B) * (1 - 6 * abs(e1) / geo.B)
    else:
        a = geo.B / 2 - e1
        if a < 0 or V1_uls < 0: 
             q_max1 = 9999.0 
             q_min1 = 9999.0
        else:
            q_max1 = 2 * V1_uls / (3 * abs(a))
            q_min1 = 0.0
            
    # Case 2: Min V (V2_uls, Mr2_uls) -> Possibly lower q_max, higher uplift/tension
    x_res2 = Mr2_uls / V2_uls
    e2 = x_res2 - geo.B / 2

    if abs(e2) <= geo.B / 6:
        q_max2 = (V2_uls / geo.B) * (1 + 6 * abs(e2) / geo.B)
        q_min2 = (V2_uls / geo.B) * (1 - 6 * abs(e2) / geo.B)
    else:
        a = geo.B / 2 - e2
        if a < 0 or V2_uls < 0: 
             q_max2 = 9999.0 
             q_min2 = 9999.0
        else:
            q_max2 = 2 * V2_uls / (3 * abs(a))
            q_min2 = 0.0
            
    # Use the max q_max from the two ULS cases for design check
    q_max = max(q_max1, q_max2)
    q_min = min(q_min1, q_min2)
    
    # Resultant position and eccentricity for the worst-case q_max
    x_res_final = x_res1 if q_max1 >= q_max2 else x_res2
    e_final = e1 if q_max1 >= q_max2 else e2
    V_final = V1_uls if q_max1 >= q_max2 else V2_uls
            
    return {
        'H_sls': P_H_total, 'V_sls': V_sls, 'M_o_sls': Mo_sls, 'M_r_sls': Mr_sls, 
        'FOS_OT': FOS_OT, 'FOS_SL': FOS_SL, 
        'Pp_allow': Pp_allow, 'F_friction': F_friction, 'F_cohesion': F_cohesion,
        'e': e_final, 'x_res': x_res_final, 'q_avg': V_final / geo.B, 'q_max': q_max, 'q_min': q_min,
        'SBC_allow': bearing.SBC_allow,
        'V_uls1': V1_uls, 'Mr_uls1': Mr1_uls, 'q_max1': q_max1, 'q_min1': q_min1, # <-- FIX: Added q_min1
        'V_uls2': V2_uls, 'Mr_uls2': Mr2_uls, 'q_max2': q_max2, 'q_min2': q_min2, # <-- FIX: Added q_min2
        'Mo_uls': Mo_uls, 'H_uls': H_uls
    }

def member_design(geo: Geometry, soil: Soil, loads: Loads, pres: Dict[str, float], mat: Materials, stab: Dict[str, float]) -> Dict[str, float]:
    """Calculates reinforcement area (ULS)."""
    
    # Load Factors for ULS Design of Components (Moment must be max, load must be min)
    gamma_H = 1.5 # Lateral
    gamma_V_max = 1.5 # Downward for Max Toe Moment
    gamma_V_min = 0.9 # Downward for Max Heel Moment
    
    fck = mat.fck
    fy = mat.fy
    W_unfactored = calculate_vertical_weights(geo, soil, loads, pres)

    # Assume 12mm bar for effective depth calculation 
    d_stem = geo.t_stem * 1000 - mat.cover - 12 
    d_base = geo.t_base * 1000 - mat.cover - 12 
    
    # --- STEM DESIGN ---
    # Factored forces at base of stem (lateral loads only, always max)
    P_earth_u = pres['P_earth'] * gamma_H
    P_surcharge_u = pres['P_surcharge'] * gamma_H
    P_water_u = pres['P_water'] * gamma_H
    P_seismic_u = pres['P_seismic'] * gamma_H
    
    Mu_stem = (P_earth_u * pres['y_earth']) + \
              (P_surcharge_u * pres['y_surcharge']) + \
              (P_water_u * pres['y_water']) + \
              (P_seismic_u * pres['y_seismic'])
     
    R_stem = (4.6 * Mu_stem * 1e6) / (fck * 1000 * d_stem**2)
    if R_stem >= 1.0: R_stem = 0.999 
    k_stem = 1 - math.sqrt(1 - R_stem)
    As_stem_req = (0.5 * fck / fy) * k_stem * 1000 * d_stem
    
    # --- BASE SLAB DESIGN ---
    
    # Use max q_max from stability check for toe uplift/moment
    q_max_u = stab['q_max']
    q_min_u = stab['q_min']
    
    # 1. Toe Cantilever (Design for Max q_max - ULS 1.5 or 0.9 from stability check)
    toe_len = geo.toe
    # Interpolate pressure at stem-toe junction using the worst-case pressure distribution
    q_sf_toe = q_min_u + (q_max_u - q_min_u) * (geo.toe / geo.B) 
    q_avg_toe = (q_sf_toe + q_max_u) / 2
    M_q_toe = q_avg_toe * toe_len**2 / 2 

    # Unfactored self-weight of toe
    W_toe = W_unfactored['W2'] * (geo.toe / geo.B) 
    M_W_toe = W_toe * toe_len / 2
    
    # Toe moment is maximized by ULS Case 1 (Max Downward on Concrete, Max Uplift q_max)
    # The moment from the base pressure M_q_toe already includes all load factors
    Mu_toe = M_q_toe - (M_W_toe * gamma_V_min) # Use 0.9 factor on self-weight to be conservative
 
    R_toe = (4.6 * Mu_toe * 1e6) / (fck * 1000 * d_base**2)
    if R_toe >= 1.0: R_toe = 0.999 
    k_toe = 1 - math.sqrt(1 - R_toe)
    As_toe_req = (0.5 * fck / fy) * k_toe * 1000 * d_base
    
    # 2. Heel Cantilever (Design for Max Downward Load - ULS 1.5)
    
    # Downward Loads (Factored by 1.5)
    W_soil_u = W_unfactored['W3'] * gamma_V_max
    W_surcharge_u = W_unfactored['W4'] * gamma_H
    W_base_u = W_unfactored['W2'] * (geo.heel / geo.B) * gamma_V_max # Base weight under heel
    
    M_W_soil = (W_unfactored['W3'] * (geo.heel / 2)) * gamma_V_max
    M_W_surcharge = (W_unfactored['W4'] * (geo.heel / 2)) * gamma_H
    M_W_base = ((W_unfactored['W2'] * (geo.heel / geo.B)) * (geo.heel / 2)) * gamma_V_max
    
    M_total_resisting = M_W_soil + M_W_surcharge + M_W_base
    
    # Uplift Load (Minimum Uplift Pressure - ULS 0.9)
    # FIX: q_max1 and q_min1 are now accessible
    q_max_heel = stab['q_max1'] # Max Downward Check - Max V (ULS 1.5)
    q_min_heel = stab['q_min1']
    
    q_at_heel_end = q_min_heel
    q_at_stem_face = q_min_heel + (q_max_heel - q_min_heel) * ((geo.toe + geo.t_stem) / geo.B)

    # Force 1: Rectangular component
    P_rect = q_at_heel_end * geo.heel
    M_rect = P_rect * geo.heel / 2

    # Force 2: Triangular component
    delta_q = q_at_stem_face - q_at_heel_end
    P_tri = 0.5 * delta_q * geo.heel
    M_tri = P_tri * geo.heel / 3
    
    M_q_heel_uplift = M_rect + M_tri
    
    Mu_heel = M_total_resisting - M_q_heel_uplift
    
    R_heel = (4.6 * Mu_heel * 1e6) / (fck * 1000 * d_base**2)
    if R_heel >= 1.0: R_heel = 0.999 
    k_heel = 1 - math.sqrt(1 - R_heel)
    As_heel_req = (0.5 * fck / fy) * k_heel * 1000 * d_base
    
    # Minimum Steel Area (As_min) - IS 456
    As_min_stem = 0.12 / 100 * geo.t_stem * 1000 * 1000 
    As_min_base = 0.12 / 100 * geo.t_base * 1000 * 1000
    
    return {
        'Mu_stem': Mu_stem, 'd_stem': d_stem/1000, 'As_stem_req': max(As_stem_req, As_min_stem),
        'Mu_toe': Mu_toe, 'd_toe': d_base/1000, 'As_toe_req': max(As_toe_req, As_min_base),
        'Mu_heel': Mu_heel, 'd_heel': d_base/1000, 'As_heel_req': max(As_heel_req, As_min_base),
    }

# ------------------------- Drawings (Matplotlib) ------------------------- #

# (plot_wall_section, plot_pressure_diagram, plot_load_resultants, make_boq functions remain unchanged)

def plot_wall_section(geo: Geometry, mat: Materials, stab: Dict[str, float], bearing: Bearing, show_rebar=True):
    """
    Plots the wall section geometry, reinforcement schematic, and bearing pressure.
    """
    fig, ax = plt.subplots(figsize=(10, 6)) 
    B = geo.B
    H_wall = geo.H + geo.t_base
    
    # ------------------ Concrete Outline ------------------
    ax.add_patch(plt.Rectangle((0, 0), B, geo.t_base, fill=False, edgecolor='k', linewidth=2)) # Base
    stem_x = geo.toe
    stem_y = geo.t_base
    ax.add_patch(plt.Rectangle((stem_x, stem_y), geo.t_stem, geo.H, fill=False, edgecolor='k', linewidth=2)) # Stem
    if geo.shear_key_depth > 0:
        ax.add_patch(plt.Rectangle((stem_x, -geo.shear_key_depth), geo.t_stem, geo.shear_key_depth, fill=False, edgecolor='k', linewidth=2)) # Shear Key

    # 
    # Ground Level
    ax.plot([-0.3, B + 0.3], [geo.t_base + geo.Df, geo.t_base + geo.Df], 'k--')
    ax.text(B + 0.3, geo.t_base + geo.Df, 'GL', va='bottom', ha='left')
    ax.plot([geo.toe + geo.t_stem, B], [H_wall, H_wall], 'k:') # Backfill Line

    # ------------------ Reinforcement Schematic ------------------
    if show_rebar:
        c = mat.cover / 1000
        # Stem vertical steel (inner/tension face)
        ax.plot([geo.toe + geo.t_stem - c, geo.toe + geo.t_stem - c], [stem_y + c, H_wall - c], 'r--')
        # Heel top steel (tension face)
        ax.plot([geo.toe + geo.t_stem + c, B - c], [geo.t_base + c, geo.t_base + c], 'b--')
        # Toe bottom steel (tension face)
        ax.plot([c, geo.toe + geo.t_stem - c], [c, c], 'g--')
        
    # ------------------ Bearing Pressure Diagram ------------------
    q_max = stab['q_max']
    q_min = stab['q_min']
    SBC_allow = bearing.SBC_allow
    base_y = 0 
    
    pressure_scale = 1.0 / (SBC_allow * 2) 
    
    ax.plot([0, B], [base_y - SBC_allow * pressure_scale, base_y - SBC_allow * pressure_scale], 
            'k:', linewidth=1) 
    ax.text(B + 0.1, base_y - SBC_allow * pressure_scale, f'SBC={SBC_allow:.0f} kPa', va='center', ha='left', color='k')
    
    if q_max < 9999.0:
        
        y_max = base_y - q_max * pressure_scale
        y_min = base_y - q_min * pressure_scale

        e = stab['e']
        if abs(e) <= B / 6:
      
            X_press = [0, B, B, 0]
            Y_press = [y_max, y_min, base_y, base_y]
            ax.fill(X_press, Y_press, 'red', alpha=0.3, label='Actual Pressure')
            ax.plot([0, B], [y_max, y_min], 'r-', linewidth=2)
            
            ax.text(0, y_max, f'{q_max:.1f}', va='top', ha='center', color='r')
         
            ax.text(B, y_min, f'{q_min:.1f}', va='top', ha='center', color='r')

        else:
            e = stab['e']
            
            if e >= 0: 
                L_contact = 3 * (geo.B / 2 - e)
                X_press = [0, L_contact, 0]
                Y_press = [y_max, base_y, base_y]
                ax.fill(X_press, Y_press, 'red', alpha=0.3)
                ax.plot([0, L_contact], [y_max, base_y], 'r-', linewidth=2)
                ax.text(0, y_max, f'{q_max:.1f} kPa', va='top', ha='center', color='r')

            else: 
   
                L_contact = 3 * (geo.B / 2 + e) 
                start_x = B - L_contact
                X_press = [start_x, B, B]
                Y_press = [base_y, y_max, base_y]
                ax.fill(X_press, Y_press, 'red', alpha=0.3)
                ax.plot([start_x, B], [base_y, y_max], 'r-', linewidth=2)
                ax.text(B, y_max, f'{q_max:.1f} kPa', va='top', ha='center', color='r')
    
    # ------------------ Plot Settings ------------------
    ax.set_aspect('equal', 'box') 
    ax.set_xlim(-0.3, B + 0.3)
    ax.set_ylim(min(-0.1, base_y - SBC_allow * pressure_scale * 1.5 - 0.2), H_wall + 0.3) 
    
    ax.set_title('Wall Section, Bar Schematic, and Base Pressure (kPa)') 
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Height (m)')
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

def plot_pressure_diagram(H: float, p_earth: float, p_surcharge: float, p_water: float, gwl_h_from_base: float):
    """Plots the lateral pressure diagram with correct inversion (Depth increases down)."""
    fig, ax = plt.subplots(figsize=(6, 8))
    
    Y = np.array([0, H]) 
    
    # Earth pressure
    ax.plot([0, p_earth], Y, 'k-')
    ax.fill_betweenx(Y, [0, p_earth], 0, color='yellow', alpha=0.3, label='Earth')
    ax.text(p_earth, H, f'{p_earth:.2f} kPa', va='top', ha='right', bbox=dict(facecolor='white', alpha=0.7))
    
    # Surcharge pressure
    ax.plot([p_surcharge, p_surcharge], Y, 'b--')
    ax.fill_betweenx(Y, [0, p_surcharge], color='blue', alpha=0.1, label='Surcharge')
    ax.text(p_surcharge, H / 2, f'{p_surcharge:.2f} kPa', va='center', ha='right', bbox=dict(facecolor='white', alpha=0.7))
    
    # Water pressure
    if p_water > 0:
        H_w_top = H - gwl_h_from_base 
        Y_w = np.array([H_w_top, H])
        ax.plot([0, p_water], Y_w, 'c-')
     
        ax.fill_betweenx(Y_w, [0, p_water], color='cyan', alpha=0.3, label='Water')
        ax.text(p_water, H, f'{p_water:.2f} kPa', va='bottom', ha='right', bbox=dict(facecolor='white', alpha=0.7))
        
    ax.set_title('Lateral Pressure Distribution (Unfactored)')
    ax.set_xlabel('Pressure (kPa)')
    ax.set_ylabel('Depth from Top of Stem (m)')
    ax.invert_yaxis() 
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')
    st.pyplot(fig)

def plot_load_resultants(P_earth, P_surcharge, P_water):
    """Plots a simple bar chart of resultant forces."""
    forces = {
        'P_earth': P_earth,
        'P_surcharge': P_surcharge,
        'P_water': P_water
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(forces.keys())
    values = list(forces.values())
    ax.barh(names, values, color=['yellow', 'blue', 'cyan'])
    ax.set_title('Resultant Lateral Forces (kN/m, Unfactored)')
    ax.set_xlabel('Force (kN/m)')
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    for i, v in enumerate(values):
        ax.text(v + 0.5, i, f'{v:.2f}', va='center')
    st.pyplot(fig)

def make_boq(geo: Geometry, mat: Materials, provided_steel: Dict[str, float]) -> pd.DataFrame:
    """Calculates Bill of Quantities (per meter length)."""
    V_stem = geo.t_stem * geo.H
    V_base = geo.B * geo.t_base
    V_key = geo.t_stem * geo.shear_key_depth
    V_conc = V_stem + V_base + V_key
    steel_kg_m3 = 80 
    M_steel = V_conc * steel_kg_m3
    df = pd.DataFrame({
        'Item': ['Concrete Volume', 'Steel Mass (Approx.)', 'Formwork Area'],
        'Unit': ['m³/m', 'kg/m³', 'm²/m'],
       
        'Value': [V_conc, steel_kg_m3, (geo.H * 2) + (geo.t_base * 2)]
    })
    df.loc[len(df)] = ['Total Steel Mass', 'kg/m', M_steel]
    return df

# ------------------------- Streamlit App ------------------------- #

st.set_page_config(layout="wide", page_title="Retaining Wall Design Wizard")

st.title('Retaining Wall Design Wizard')
st.header('Design Inputs and Calculations (Single Page) 📝')

# --- Initial State/Default Values ---
# Function to load state from JSON (USED FOR FILE UPLOAD)
def load_state_from_dict(data):
    st.session_state['project_name'] = data.get('project_name', 'RW-01')
    st.session_state['wall_type'] = data.get('wall_type', 'Cantilever T-Wall')
    st.session_state['design_state'] = data.get('design_state', 'Active')
    
    geo_data = data.get('geo', {})
    st.session_state['geo'] = Geometry(
        H=geo_data.get('H', 4.0), Df=geo_data.get('Df', 1.0), 
        t_stem=geo_data.get('t_stem', 0.4), t_base=geo_data.get('t_base', 0.6), 
        toe=geo_data.get('toe', 1.0), heel=geo_data.get('heel', 2.0), 
        shear_key_depth=geo_data.get('shear_key_depth', 0.0)
    )
    soil_data = data.get('soil', {})
    st.session_state['soil'] = Soil(
        gamma=soil_data.get('gamma', 18.0), phi=soil_data.get('phi', 30.0), 
        c_base=soil_data.get('c_base', 0.0), gamma_sub=soil_data.get('gamma_sub', 9.81), 
        gamma_fill=soil_data.get('gamma_fill', 18.0)
    )
    loads_data = data.get('loads', {})
    st.session_state['loads'] = Loads(
        surcharge_q=loads_data.get('surcharge_q', 10.0), 
        gwl_h_from_base=loads_data.get('gwl_h_from_base', 0.0), 
        seismic_kh=loads_data.get('seismic_kh', 0.0), 
        seismic_kv=loads_data.get('seismic_kv', 0.0), 
        use_seismic=loads_data.get('use_seismic', False)
    )
    mat_data = data.get('mat', {})
    st.session_state['mat'] = Materials(
        fck=mat_data.get('fck', 25), fy=mat_data.get('fy', 415), 
        gamma_c=mat_data.get('gamma_c', 25.0), cover=mat_data.get('cover', 50)
    )
    bearing_data = data.get('bearing', {})
    st.session_state['bearing'] = Bearing(
        SBC_allow=bearing_data.get('SBC_allow', 150.0), 
        mu_base=bearing_data.get('mu_base', 0.5), 
        include_passive=bearing_data.get('include_passive', True), 
        passive_reduction=bearing_data.get('passive_reduction', 0.66)
    )
    st.success(f"Design '{st.session_state['project_name']}' loaded successfully.")
    # Rerunning the script after updating all state variables
    st.rerun()

# Check for initial state or load state
if 'geo' not in st.session_state:
    st.session_state['geo'] = Geometry(H=4.0, Df=1.0, t_stem=0.4, t_base=0.6, toe=1.0, heel=2.0, shear_key_depth=0.0)
if 'soil' not in st.session_state:
    st.session_state['soil'] = Soil(gamma=18.0, phi=30.0, c_base=0.0, gamma_sub=9.81, gamma_fill=18.0)
if 'loads' not in st.session_state:
    st.session_state['loads'] = Loads(surcharge_q=10.0, gwl_h_from_base=0.0, seismic_kh=0.0, seismic_kv=0.0, use_seismic=False)
if 'mat' not in st.session_state:
    st.session_state['mat'] = Materials(fck=25, fy=415, gamma_c=25.0, cover=50)
if 'bearing' not in st.session_state:
    st.session_state['bearing'] = Bearing(SBC_allow=150.0, mu_base=0.5, include_passive=True, passive_reduction=0.66)
if 'project_name' not in st.session_state:
    st.session_state['project_name'] = 'RW-01'
if 'wall_type' not in st.session_state:
    st.session_state['wall_type'] = 'Cantilever T-Wall'
if 'design_state' not in st.session_state:
    st.session_state['design_state'] = 'Active'

# --- Load Design File Input (NEW) ---
with st.sidebar:
    st.header("Load Existing Design")
    uploaded_file = st.file_uploader("Upload .json Design File", type="json")
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            data = json.loads(bytes_data.decode("utf-8"))
            load_state_from_dict(data) # This function calls st.rerun() on success
        except Exception as e:
            st.error(f"Error loading file: {e}")
    st.markdown("---")
    
    # <--- START: ADDED PRINT BUTTON --->
    st.header("Print Utility")
    # Inject HTML button that calls the browser's window.print() function
    st.markdown('''
        <button onclick="window.print()" style="
            width: 100%; 
            height: 50px; 
            background-color: #4CAF50; 
            color: white; 
            border-radius: 5px; 
            border: none; 
            font-size: 16px; 
            cursor: pointer;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        ">
        Print Design Screen 🖨️
        </button>
    ''', unsafe_allow_html=True)
    st.markdown("---")
    # <--- END: ADDED PRINT BUTTON --->


# --- Inputs (Now on Main Page) ---
with st.expander('1. Project & General Parameters', expanded=False):
    col1, col2, col3 = st.columns(3)
    project_name = col1.text_input('Project Name', st.session_state['project_name'], key='proj_name_input')
    wall_type = col2.selectbox('Wall Type', ['Cantilever T-Wall', 'Cantilever L-Wall', 'Cantilever Inverted-T Wall'], index=['Cantilever T-Wall', 'Cantilever L-Wall', 'Cantilever Inverted-T Wall'].index(st.session_state['wall_type']) if st.session_state['wall_type'] in ['Cantilever T-Wall', 'Cantilever L-Wall', 'Cantilever Inverted-T Wall'] else 0, key='wall_type_select')
    design_state = col3.selectbox('Earth Pressure State', ['Active', 'At-Rest', 'Seismic (Mononobe-Okabe)'], index=['Active', 'At-Rest', 'Seismic (Mononobe-Okabe)'].index(st.session_state['design_state']) if st.session_state['design_state'] in ['Active', 'At-Rest', 'Seismic (Mononobe-Okabe)'] else 0, key='design_state_select')
    
    # Update session state with values from widgets
    st.session_state['project_name'] = project_name
    st.session_state['wall_type'] = wall_type
    st.session_state['design_state'] = design_state

st.divider()

# --- Geometry and Dimensions ---
with st.expander('2. Wall Geometry and Dimensions', expanded=False):
    geo = st.session_state['geo']
    c1, c2, c3, c4 = st.columns(4)
    geo.H = c1.number_input('Height of Stem (H, m)', 1.0, 15.0, geo.H, 0.1, key='H_input')
    geo.t_stem = c2.number_input('Stem Thickness (t_stem, m)', 0.2, 2.0, geo.t_stem, 0.05, key='t_stem_input')
    geo.t_base = c3.number_input('Base Slab Thickness (t_base, m)', 0.3, 2.0, geo.t_base, 0.05, key='t_base_input')
    geo.shear_key_depth = c4.number_input('Shear Key Depth (m)', 0.0, 1.0, geo.shear_key_depth, 0.1, key='sk_depth_input')
    
    c5, c6, c7 = st.columns(3)
    geo.toe = c5.number_input('Toe Length (toe, m)', 0.1, 5.0, geo.toe, 0.1, key='toe_input')
    geo.heel = c6.number_input('Heel Length (heel, m)', 0.1, 5.0, geo.heel, 0.1, key='heel_input')
    
    geo.B = geo.toe + geo.t_stem + geo.heel 
    c7.markdown(f'**Base Width (B): {geo.B:.2f} m**')
    
# --- Soil, Loads, Materials ---
with st.expander('3. Soil, Loads, and Materials', expanded=False):
    col_soil, col_loads, col_mat = st.columns(3)

    with col_soil:
        st.subheader('Soil Properties')
        soil = st.session_state['soil']
        soil.gamma_fill = st.number_input('Unit Weight of Backfill (γ_fill, kN/m³)', 16.0, 22.0, soil.gamma_fill, 1.0, key='g_fill')
        soil.phi = st.slider('Angle of Internal Friction (φ, deg)', 25.0, 40.0, soil.phi, 0.5, key='phi')
        soil.gamma = st.number_input('Unit Weight of Foundation Soil (γ, kN/m³)', 16.0, 22.0, soil.gamma, 1.0, key='g_foun')
        soil.c_base = st.number_input('Base-Soil Cohesion (c_base, kPa)', 0.0, 50.0, soil.c_base, 1.0, key='c_base')

    with col_loads:
        st.subheader('Loads and Embedment')
        loads = st.session_state['loads']
        geo.Df = st.number_input('Depth of Embedment (Df, m)', 0.5, 5.0, geo.Df, 0.1, key='df')
        loads.surcharge_q = st.number_input('Surcharge (q, kPa)', 0.0, 50.0, loads.surcharge_q, 5.0, key='q_sur')
        gwl_h = st.number_input('GWL Height from Base (m)', 0.0, geo.H + geo.t_base, loads.gwl_h_from_base, 0.1, key='gwl')
        loads.gwl_h_from_base = gwl_h
        
        loads.use_seismic = st.checkbox('Include Seismic Load (MO Approx.)', loads.use_seismic, key='seis_chk')
        if loads.use_seismic:
            loads.seismic_kh = st.number_input('Horizontal Acc. (kh)', 0.01, 0.3, loads.seismic_kh, 0.01, key='kh')
            loads.seismic_kv = st.number_input('Vertical Acc. (kv)', 0.0, 0.1, loads.seismic_kv, 0.01, key='kv')
        else:
            loads.seismic_kh = 0.0
            loads.seismic_kv = 0.0

    with col_mat:
        st.subheader('Materials & Bearing')
        mat = st.session_state['mat']
        mat.fck = st.selectbox('Concrete Grade (fck)', [20, 25, 30, 35], index=[20, 25, 30, 35].index(mat.fck) if mat.fck in [20, 25, 30, 35] else 1, key='fck')
        mat.fy = st.selectbox('Steel Grade (fy)', [415, 500], index=[415, 500].index(mat.fy) if mat.fy in [415, 500] else 0, key='fy')
        mat.cover = st.number_input('Clear Cover (mm)', 25, 100, mat.cover, 5, key='cover')

        bearing = st.session_state['bearing']
        bearing.SBC_allow = st.number_input('Allowable Bearing Capacity (SBC, kPa)', 50.0, 500.0, bearing.SBC_allow, 10.0, key='sbc')
        bearing.mu_base = st.slider('Base-Soil Friction Coeff (μ)', 0.3, 0.7, bearing.mu_base, 0.05, key='mu')
        bearing.include_passive = st.checkbox('Include Passive Resistance (Pp)', bearing.include_passive, key='pp_chk')
        if bearing.include_passive:
            bearing.passive_reduction = st.slider('Pp Reduction Factor (η)', 0.5, 1.0, bearing.passive_reduction, 0.05, key='pp_red')


# ------------------------- Core Calculations ------------------------- #
# These functions run on every Streamlit rerun, capturing the latest inputs.
pres = pressures(geo.H, soil, loads, design_state)
stab = stability(geo, soil, loads, bearing, pres)
# Note: pres and stab are now calculated using IS 456 ULS load combinations
desg = member_design(geo, soil, loads, pres, mat, stab)

# ------------------------- Sequential Output Display (Main Page) ------------------------- #

st.divider()
st.subheader('4. Results and Diagrams')

# --- SECTION 4.1: Drawings ---
with st.expander('4.1 Geometry and Pressure Diagrams', expanded=True):
    col_draw, col_press = st.columns([2, 1])
    with col_draw:
        st.subheader('Wall Section, Reinforcement, and Base Pressure (ULS Max)')
        plot_wall_section(geo, mat, stab, bearing) 
    with col_press:
        st.subheader('Lateral Pressure Distribution')
        plot_pressure_diagram(geo.H, pres['p0_earth'], pres['p_surcharge'], pres['p0_water'], loads.gwl_h_from_base)
        st.subheader('Resultant Forces (Unfactored)')
        plot_load_resultants(pres['P_earth'], pres['P_surcharge'], pres['P_water'])


# --- SECTION 4.2: Stability Checks ---
st.divider()
with st.expander('4.2 Stability Checks (SLS) - Narrative & Results', expanded=True):
    
    st.markdown("### Detailed Stability Calculation Narrative (SLS Check)")
    st.info("**Note:** Overturning and Sliding checks are performed on **unfactored** (SLS) loads. Bearing is checked for **factored** (ULS) loads, using the worst case for max pressure.")
    
    # Pre-calculate and format ALL values for simplest F-strings
    
    # Values for Overturning (SLS)
    P_H_val = f"{stab['H_sls']:.2f}"
    V_val = f"{stab['V_sls']:.2f}"
    Mo_val = f"{stab['M_o_sls']:.2f}"
    Mr_val = f"{stab['M_r_sls']:.2f}"
    FOS_OT_val = f"**{stab['FOS_OT']:.2f}**"
    
    # NEW: Inclination and Resultant Height
    delta_val = f"{pres['delta']:.2f}"
    y_earth_val = f"{pres['y_earth']:.2f}"

    # Values for Sliding (SLS)
    mu_base_val = f"{bearing.mu_base:.2f}"
    F_fric_val = f"{stab['F_friction']:.2f}"
    F_cohesion_val = f"{stab['F_cohesion']:.2f}"
    Pp_allow_val = f"{stab['Pp_allow']:.2f}"
    F_r_val = f"{stab['F_friction'] + stab['Pp_allow'] + stab['F_cohesion']:.2f}"
    FOS_SL_val = f"**{stab['FOS_SL']:.2f}**"
    
    # Values for Bearing (ULS Worst Case)
    B_val = f"{geo.B:.2f}"
    x_res_val = f"{stab['x_res']:.2f}"
    B_over_2_val = f"{geo.B / 2:.2f}"
    e_val = f"{stab['e']:.2f}"
    B_over_6_val = f"{geo.B / 6:.2f}"
    q_max_val = f"**{stab['q_max']:.2f}**"
    SBC_allow_val_f = f"{bearing.SBC_allow:.0f}"


    # 1. Overturning Check
    st.markdown("#### 1. Overturning Check (SLS)")
    st.markdown(f"**Lateral Force Inclination:** $\\delta = {delta_val}^\circ$ | **Earth Force Resultant Height:** $y_{{earth}} = {y_earth_val} \\text{{ m}}$")
    st.markdown(f"**Total Lateral Force ($P_H$):** ${P_H_val} \\text{{ kN/m}}$")
    st.markdown(f"**Total Vertical Force ($V$):** ${V_val} \\text{{ kN/m}}$")
    st.markdown(f"**FOS Overturning:** $\\text{{FOS}}_{{OT}} = M_r / M_o = {Mr_val} / {Mo_val} = {FOS_OT_val}$ (Required $\\ge 2.0$)")

    # 2. Sliding Check
    st.markdown("#### 2. Sliding Check (SLS)")
    st.markdown(f"**Total Resisting Force ($F_r$):** $F_r = F_{{friction}} + P_p + F_{{cohesion}} = {F_r_val} \\text{{ kN/m}}$")
    st.markdown(f"**FOS Sliding:** $\\text{{FOS}}_{{SL}} = F_r / P_H = {F_r_val} / {P_H_val} = {FOS_SL_val}$ (Required $\\ge 1.5$)")

    # 3. Bearing Check
    st.markdown("#### 3. Bearing Capacity Check (ULS Max Pressure)")
    st.markdown(f"**Governing ULS Case:** Max $q_{{max}}$ determined by comparing **ULS 1.5** ($q_{{max1}}={stab['q_max1']:.1f} \\text{{ kPa}}$) and **ULS 0.9** ($q_{{max2}}={stab['q_max2']:.1f} \\text{{ kPa}}$)")
    st.markdown(f"**Eccentricity ($e$):** $e = {e_val} \\text{{ m}}$ (Limit $B/6 = {B_over_6_val} \\text{{ m}}$)")
    st.markdown(f"**Max Bearing Pressure ($q_{{max}}$):** $q_{{max}} = {q_max_val} \\text{{ kPa}}$ (Allowed $\\le {SBC_allow_val_f} \\text{{ kPa}}$)")
    
    st.divider()
    st.subheader('Stability Results Summary')
    
    # Use raw values for the DataFrame
    df_stab = pd.DataFrame({
        'Quantity': ['FOS Overturning (SLS)', 'FOS Sliding (SLS)', 'Max Bearing Pressure (ULS, kPa)', 'Allowed SBC (kPa)', 'Eccentricity (m)', 'B/6 Limit (m)'],
        'Value': [stab['FOS_OT'], stab['FOS_SL'], stab['q_max'], bearing.SBC_allow, stab['e'], geo.B/6]
    })
    st.dataframe(df_stab, use_container_width=True, hide_index=True)


# --- SECTION 4.3: Member Design & BOQ ---
st.divider()
with st.expander('4.3 Member Design (ULS) and Bill of Quantities', expanded=True):
    col_des, col_boq = st.columns([2, 1])

    with col_des:
        st.subheader('Detailed Member Design Narrative (ULS)')
        
        # Rebar input fields 
        c1, c2, c3 = st.columns(3)
        with c1:
            stem_dia = st.selectbox('Stem main dia (mm)', [10,12,16], index=0, key='stem_dia')
            stem_sp = st.number_input('Stem spacing (mm)', 100, 300, 200, 25, key='stem_sp')
        with c2:
            heel_dia = st.selectbox('Heel main dia (mm)', [10,12,16], index=0, key='heel_dia')
            heel_sp = st.number_input('Heel spacing (mm)', 100, 300, 150, 25, key='heel_sp')
        with c3:
            toe_dia = st.selectbox('Toe main dia (mm)', [10,12,16], index=0, key='toe_dia')
            toe_sp = st.number_input('Toe spacing (mm)', 100, 300, 150, 25, key='toe_sp')

        # Calculate Provided Area
        stem_As_prov = as_per_m(stem_dia, stem_sp)
        heel_As_prov = as_per_m(heel_dia, heel_sp)
        toe_As_prov = as_per_m(toe_dia, toe_sp)
        
        # Format values for final output
        Mu_stem_val = f"{desg['Mu_stem']:.2f}"
        As_stem_req_val = f"**{desg['As_stem_req']:.0f}**"
        stem_As_prov_val = f"{stem_As_prov:.0f}"

        Mu_heel_val = f"{desg['Mu_heel']:.2f}"
        As_heel_req_val = f"**{desg['As_heel_req']:.0f}**"
        heel_As_prov_val = f"{heel_As_prov:.0f}"

        Mu_toe_val = f"{desg['Mu_toe']:.2f}"
        As_toe_req_val = f"**{desg['As_toe_req']:.0f}**"
        toe_As_prov_val = f"{toe_As_prov:.0f}"

        # Stem Design Narrative
        st.markdown(f"**Stem Design (Vertical):** $M_{{u, stem}} = {Mu_stem_val} \\text{{ kNm/m}}$. $A_{{st, req}} = {As_stem_req_val} \\text{{ mm}}^2/\\text{{m}}$. Provided: $\\phi{stem_dia} @ {stem_sp} \\text{{ mm}} \\rightarrow A_{{st, prov}} = {stem_As_prov_val} \\text{{ mm}}^2/\\text{{m}}$.")
        
        # Heel Design Narrative
        st.markdown(f"**Heel Design (Top):** $M_{{u, heel}} = {Mu_heel_val} \\text{{ kNm/m}}$. $A_{{st, req}} = {As_heel_req_val} \\text{{ mm}}^2/\\text{{m}}$. Provided: $\\phi{heel_dia} @ {heel_sp} \\text{{ mm}} \\rightarrow A_{{st, prov}} = {heel_As_prov_val} \\text{{ mm}}^2/\\text{{m}}$.")
        
        # Toe Design Narrative
        st.markdown(f"**Toe Design (Bottom):** $M_{{u, toe}} = {Mu_toe_val} \\text{{ kNm/m}}$. $A_{{st, req}} = {As_toe_req_val} \\text{{ mm}}^2/\\text{{m}}$. Provided: $\\phi{toe_dia} @ {toe_sp} \\text{{ mm}} \\rightarrow A_{{st, prov}} = {toe_As_prov_val} \\text{{ mm}}^2/\\text{{m}}$.")
        
        # Design Table
        df_as = pd.DataFrame([
            ['Stem (Inner)', desg['As_stem_req'], stem_As_prov, 'OK' if stem_As_prov >= desg['As_stem_req'] else 'INC'],
            ['Heel (Top)', desg['As_heel_req'], heel_As_prov, 'OK' if heel_As_prov >= desg['As_heel_req'] else 'INC'],
            ['Toe (Bottom)', desg['As_toe_req'], toe_As_prov, 'OK' if toe_As_prov >= desg['As_toe_req'] else 'INC'],
        ], columns=['Member', 'As req (mm²/m)', 'As prov (mm²/m)', 'Status'])
        st.dataframe(df_as, use_container_width=True, hide_index=True)

        # Store provided steel area for BOQ
        provided_steel = { 
            'As_stem': stem_As_prov, 'As_heel': heel_As_prov, 'As_toe': toe_As_prov, 
        }

    with col_boq:
        st.subheader('Bill of Quantities (per meter run)')
        boq_df = make_boq(geo, mat, provided_steel)
        st.dataframe(boq_df, use_container_width=True, hide_index=True)


# --- SECTION 5: Reports and Downloads ---
st.divider()
with st.expander('5. Report Generation and Downloads', expanded=False):
    st.subheader('Final Report & CAD Export')

    # Prepare the design state for download
    current_state = create_state_dict(geo, soil, loads, mat, bearing, project_name, wall_type, design_state)
    json_state = json.dumps(current_state, indent=4)
    
    c_save, c_dxf, c_pdf, c_excel = st.columns(4)

    # Download Design File
    c_save.download_button(
        label="Download Design File (.json) 💾",
        data=json_state,
        file_name=f'{project_name}_RW_Design.json',
        mime="application/json"
    )

    c_dxf.info('DXF is supported with ezdxf.')
    c_pdf.info('PDF report is supported with reportlab.')
    
    boq_df = make_boq(geo, mat, provided_steel)
    f = io.BytesIO()
    boq_df.to_excel(f, sheet_name='BOQ', index=False)
    c_excel.download_button('Download BOQ (Excel) 📊', data=f, file_name=f'{project_name}_BOQ.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
