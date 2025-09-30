# Retaining Wall Design Wizard (Streamlit)
# -----------------------------------------------------------
# Covers L / T / Inverted-L walls. Handles Active/At-Rest/Passive
# pressures, water, surcharge, optional seismic (MO approx), stability,
# member design, pressure & load drawings, reinforcement schematic,
# BOQ, **DXF export (ezdxf)**, **PDF report (reportlab)**,
# **Shear-key modeling and Passive resistance**.
# -----------------------------------------------------------

import io
import math
from dataclasses import dataclass
from typing import Dict
from datetime import datetime

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

def calc_K(phi: float, state: str) -> Dict[str, float]:
    """Calculates Rankine and other coefficients."""
    phi_rad = math.radians(phi)
    
    if state == 'Active':
        Ka = (1 - math.sin(phi_rad)) / (1 + math.sin(phi_rad))
        Kp = (1 + math.sin(phi_rad)) / (1 - math.sin(phi_rad))
        delta = 0.0 # Rankine/Active
        return {'Ka': Ka, 'Kp': Kp, 'delta': delta}
    elif state == 'At-Rest':
        Ko = 1 - math.sin(phi_rad)
        Kp = (1 + math.sin(phi_rad)) / (1 - math.sin(phi_rad))
        delta = 0.0
        return {'Ka': Ko, 'Kp': Kp, 'delta': delta}
    elif state == 'Seismic (Mononobe-Okabe)':
        # Simplified/Approximate for MO. Requires Ka and Delta to be manually input for full accuracy.
        Ka = (1 - math.sin(phi_rad)) / (1 + math.sin(phi_rad)) 
        Kp = (1 + math.sin(phi_rad)) / (1 - math.sin(phi_rad))
        delta = math.radians(phi / 2) # Typical assumption for wall-soil friction (delta)
        
        # Calculate angle of inertia (I) and wall inclination (beta)
        i = math.degrees(math.atan(loads.seismic_kh / (1 - loads.seismic_kv))) # Eqn 7.15 IS 1893
        beta = 90 # Vertical wall face assumed (90 deg from horizontal)

        # Calculate KaE (Seismic Active Earth Pressure Coefficient)
        # Simplified formula from IS 1893 (Part 5):2016 for vertical wall, horizontal backfill
        if loads.seismic_kh > 0:
            psi = math.radians(math.atan(loads.seismic_kh / (1-loads.seismic_kv)))
            num = (math.cos(phi_rad-psi)**2)
            den = (math.cos(psi)**2) * math.cos(psi+delta) * (1 + math.sqrt(math.sin(phi_rad+delta) * math.sin(phi_rad-psi) / (math.cos(psi+delta))))**2
            KaE = num/den
        else:
            KaE = Ka
            
        return {'Ka': KaE, 'Kp': Kp, 'delta': delta}
    else:
        return {'Ka': 0.0, 'Kp': 0.0, 'delta': 0.0}

def pressures(H: float, soil: Soil, loads: Loads, design_state: str) -> Dict[str, float]:
    """Calculates earth and water pressures."""
    
    K = calc_K(soil.phi, design_state)
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
    if loads.use_seismic:
        # P_seismic = 0.375 * kh * gamma * H^2 (Simplified formula)
        # Using IS 1893 (Part 5):2016 (Approximate)
        if design_state == 'Seismic (Mononobe-Okabe)':
            KaE = K['Ka']
            Kh = loads.seismic_kh
            
            # P_A and P_AE are combined in KaE calculation (P_AE = KaE*0.5*gamma*H^2)
            # P_dynamic (Delta P_AE) is the difference P_AE - P_A
            P_static = 0.5 * Ka * soil.gamma_fill * H**2
            P_total_seismic = 0.5 * KaE * soil.gamma_fill * H**2
            P_dynamic = P_total_seismic - P_static
            
            y_seismic = 0.6 * H # Typical point of application for dynamic part
            
            # Update P_earth (P_static) with P_total_seismic
            # Note: The design forces P_earth must now be P_total_seismic for stability check.
            P_earth = P_total_seismic
            y_earth = H / 3 # Center of pressure for combined (earth + dynamic) is typically still close to H/3 for design checks
            
            # For drawing, let's keep the dynamic pressure separate (as P_seismic)
            p0_seismic = 0.0 # This pressure is non-linear, so P_dynamic is used instead of p0
            
            # Surcharge pressure is usually NOT increased by seismic factor (or a smaller factor is used)
            # We assume it remains constant (P_surcharge)
        else:
            P_dynamic = 0.0
            y_seismic = 0.0
            p0_seismic = 0.0
    else:
        P_dynamic = 0.0
        y_seismic = 0.0
        p0_seismic = 0.0

    return {
        'Ka': Ka, 'delta': math.degrees(delta), 'KaE': K.get('KaE', Ka),
        'p0_earth': p0_earth, 'P_earth': P_earth, 'y_earth': y_earth,
        'p_surcharge': p_surcharge, 'P_surcharge': P_surcharge, 'y_surcharge': y_surcharge,
        'p0_water': p0_water, 'P_water': P_water, 'y_water': y_water,
        'P_seismic': P_dynamic, 'y_seismic': y_seismic, 'p0_seismic': p0_seismic,
    }

def stability(geo: Geometry, soil: Soil, loads: Loads, bearing: Bearing, pres: Dict[str, float]) -> Dict[str, float]:
    """Performs global stability checks (SLS)."""
    
    # ------------------ Vertical Loads (Self-weight) ------------------
    gamma_c = 25.0 # Typical concrete density (kN/m3)
    
    # W1: Stem
    W1 = geo.t_stem * (geo.H + geo.t_base) * gamma_c
    x1 = geo.toe + geo.t_stem / 2
    
    # W2: Base
    W2 = geo.B * geo.t_base * gamma_c
    x2 = geo.B / 2
    
    # W3: Heel soil (backfill above heel)
    W3 = geo.heel * geo.H * soil.gamma_fill
    x3 = geo.toe + geo.t_stem + geo.heel / 2
    
    # W4: Surcharge over heel
    W4 = loads.surcharge_q * geo.heel
    x4 = geo.toe + geo.t_stem + geo.heel / 2
    
    # W5: Water uplift (Simplified as triangular load beneath base)
    # Assumes no drainage, GWL at the base.
    gwl = loads.gwl_h_from_base
    if gwl > 0:
        p_uplift_base = GAMMA_W * gwl
        W5_uplift = 0.5 * p_uplift_base * geo.B # Triangular uplift (max at back, zero at front)
        x5_uplift = geo.toe + geo.t_stem + geo.heel / 3
    else:
        W5_uplift = 0.0
        x5_uplift = 0.0
        
    # Total Vertical Force (V)
    # V = W_conc + W_soil + W_surcharge - W_uplift
    V_total = W1 + W2 + W3 + W4 - W5_uplift
    
    # Overturning Moments (Mo) - From lateral forces
    P_H = pres['P_earth'] + pres['P_surcharge'] + pres['P_water'] + pres['P_seismic']
    Mo = (pres['P_earth'] * pres['y_earth']) + \
         (pres['P_surcharge'] * pres['y_surcharge']) + \
         (pres['P_water'] * pres['y_water']) + \
         (pres['P_seismic'] * pres['y_seismic'])
         
    # Resisting Moments (Mr) - From vertical loads
    Mr = (W1 * x1) + (W2 * x2) + (W3 * x3) + (W4 * x4) - (W5_uplift * x5_uplift)
    
    # ------------------ Stability Checks ------------------
    
    # 1. Overturning (FOS_OT)
    if Mo == 0:
        FOS_OT = 99.0
    else:
        FOS_OT = Mr / Mo
        
    # 2. Sliding (FOS_SL)
    # Resisting force (Fr) = V*mu + Pp + c_base*B
    
    # Passive Resistance (Pp)
    Kp = calc_K(soil.phi, 'Active')['Kp'] # Kp is always calculated using phi
    Dp = geo.Df # Depth of passive wedge (toe embedment)
    
    # Check for Shear Key (Passive on key is optional and complex, we include Passive on toe only)
    if geo.shear_key_depth > 0:
        # Increase the passive resistance depth to Df + shear_key_depth (approx)
        Dp = geo.Df + geo.shear_key_depth
        
    # Passive force (Pp)
    # Pp = 0.5 * Kp * gamma * Dp^2 + 2*c*sqrt(Kp)*Dp + Ka * gamma_fill * Dp * Df (approx)
    Pp = 0.5 * Kp * soil.gamma * Dp**2 # Simplified, ignoring toe surcharge and cohesion
    
    # Allowable Passive Force (Pp_allow)
    if bearing.include_passive:
        Pp_allow = Pp * bearing.passive_reduction
    else:
        Pp_allow = 0.0
        
    # Base Cohesion
    F_cohesion = soil.c_base * geo.B
    
    # Friction Resistance
    F_friction = V_total * bearing.mu_base
    
    Fr = F_friction + Pp_allow + F_cohesion
    
    if P_H == 0:
        FOS_SL = 99.0
    else:
        FOS_SL = Fr / P_H
        
    # 3. Bearing Pressure (q_max/q_min)
    
    # Eccentricity (e)
    x_res = Mr / V_total
    e = x_res - geo.B / 2
    
    # Pressure distribution (q)
    if abs(e) <= geo.B / 6:
        # Full contact (Trapezoidal)
        q_avg = V_total / geo.B
        q_max = q_avg * (1 + 6 * abs(e) / geo.B)
        q_min = q_avg * (1 - 6 * abs(e) / geo.B)
    else:
        # Partial contact (Triangular)
        a = geo.B / 2 - e
        if a < 0: # Check if resultant is outside base
             q_max = 9999.0 # Fail
             q_min = 9999.0
        else:
            q_max = 2 * V_total / (3 * a)
            q_min = 0.0
            
    # Correct q_max/q_min for sign convention
    # If e is positive (towards toe), max pressure is at the toe (q_max)
    # If e is negative (towards heel), max pressure is at the heel (q_max)
    if e < 0: # Eccentricity is towards the heel (right side)
        # Recalculate based on a different reference point, or just flip q_max/q_min if full contact
        q_max, q_min = q_max, q_min # q_max is at toe, q_min at heel (correct for e towards toe)
    else:
        q_max, q_min = q_max, q_min
        
    
    return {
        'H': P_H, 'V': V_total, 'M_o': Mo, 'M_r': Mr, 
        'FOS_OT': FOS_OT, 'FOS_SL': FOS_SL, 
        'Pp_allow': Pp_allow, 'F_friction': F_friction, 
        'e': e, 'q_avg': V_total / geo.B, 'q_max': q_max, 'q_min': q_min
    }

def member_design(geo: Geometry, soil: Soil, loads: Loads, pres: Dict[str, float], mat: Materials) -> Dict[str, float]:
    """Calculates reinforcement area (ULS)."""
    
    # ULS Factors (from IS 456 / IS 875)
    gamma_f_DL = 1.5
    gamma_f_LL = 1.5
    gamma_m = 1.5
    
    # M_limit (for check against failure) - THIS LINE IS REMOVED/COMMENTED OUT
    # fck = mat.fck
    # fy = mat.fy
    # M_limit = 0.138 * fck * 1000 * (mat.H_wall * 1000)**2 # <-- REMOVE THIS LINE
    
    # --- STEM DESIGN (Factored Earth + Surcharge + Water Moments) ---
    
    # Stem depth (d) = t_stem - cover - bar_dia/2 (simplified to -cover)
    d_stem = geo.t_stem * 1000 - mat.cover - 12 # Assume 12mm bar for initial calc (mm)
    
    
    # Moments at Base of Stem (Critical Section)
    P_earth_u = pres['P_earth'] * gamma_f_DL
    P_surcharge_u = pres['P_surcharge'] * gamma_f_LL
    P_water_u = pres['P_water'] * gamma_f_DL
    P_seismic_u = pres['P_seismic'] * gamma_f_DL # Assuming seismic is DL for simplicity
    
    Mu_stem = (P_earth_u * pres['y_earth']) + \
              (P_surcharge_u * pres['y_surcharge']) + \
              (P_water_u * pres['y_water']) + \
              (P_seismic_u * pres['y_seismic'])
              
    # Required Steel Area (As_req)
    # As = (0.5*fck/fy) * (1 - sqrt(1 - 4.6*Mu / (fck*b*d^2))) * b*d
    
    # Calculate term inside the square root
    R = (4.6 * Mu_stem * 1e6) / (fck * 1000 * d_stem**2)
    
    if R >= 1:
        # Section is over-stressed (requires compression steel or a larger section)
        # For simplicity, we limit R to 0.999 to get max As for singly reinforced section.
        # This will flag the need for a larger section.
        st.warning(f"Stem section is highly stressed (R={R:.2f}). Increase thickness or fck.")
        R = 0.999
        
    k = 1 - math.sqrt(1 - R)
    As_stem_req = (0.5 * fck / fy) * k * 1000 * d_stem
    
    # --- BASE SLAB DESIGN (Cantilever Moments) ---
    
    # Factored bearing pressures
    q_max_u = pres['q_max'] * gamma_f_DL
    q_min_u = pres['q_min'] * gamma_f_DL
    
    d_base = geo.t_base * 1000 - mat.cover - 12 # mm
    
    # 1. Toe Cantilever (Critical Section at Stem Face)
    toe_len = geo.toe
    # We need the pressure at the stem face (q_sf)
    q_sf = q_min_u + (q_max_u - q_min_u) * (toe_len / geo.B)
    
    # Moment from pressure (M_q)
    M_q_toe = (q_sf * toe_len**2 / 2) + ((q_max_u - q_sf) * toe_len**2 / 6)
    
    # Moment from toe self-weight (W_toe) (Resisting)
    W_toe = geo.t_base * gamma_c * toe_len # kN/m
    M_W_toe = W_toe * toe_len / 2
    
    # Net Factored Moment
    Mu_toe = (M_q_toe - M_W_toe) * gamma_f_DL # Factor already applied to q and W
    
    k_toe = 1 - math.sqrt(1 - (4.6 * Mu_toe * 1e6) / (fck * 1000 * d_base**2))
    As_toe_req = (0.5 * fck / fy) * k_toe * 1000 * d_base
    
    # 2. Heel Cantilever (Critical Section at Stem Face)
    heel_len = geo.heel
    # We need the pressure at the stem face (q_sf)
    q_sf = q_min_u + (q_max_u - q_min_u) * (geo.toe / geo.B) # Already calculated above
    
    # Pressure at the heel end (q_heel)
    q_heel = q_min_u
    
    # Moment from pressure (M_q)
    # The pressure under the heel is trapezoidal
    M_q_heel = (q_heel * heel_len**2 / 2) + ((q_sf - q_heel) * heel_len**2 / 6)
    
    # Moment from soil and surcharge (W_soil, W_surcharge) (Resisting/Stabilizing)
    W_soil = geo.H * soil.gamma_fill * heel_len
    W_surcharge = loads.surcharge_q * heel_len
    
    M_W_soil = W_soil * heel_len / 2
    M_W_surcharge = W_surcharge * heel_len / 2
    
    # Moment from base self-weight (W_base) (Resisting/Stabilizing)
    W_base = geo.t_base * gamma_c * heel_len
    M_W_base = W_base * heel_len / 2
    
    # Net Factored Moment (Mu_heel)
    # Heel moment = (W_soil + W_surcharge + W_base - M_q_heel) * gamma_f
    Mu_heel = ((M_W_soil + M_W_surcharge + M_W_base) * gamma_f_DL) - (M_q_heel * gamma_f_DL)
    
    k_heel = 1 - math.sqrt(1 - (4.6 * Mu_heel * 1e6) / (fck * 1000 * d_base**2))
    As_heel_req = (0.5 * fck / fy) * k_heel * 1000 * d_base
    
    # Minimum Steel Area (As_min) - IS 456
    As_min_stem = 0.12 / 100 * geo.t_stem * 1000 * 1000 # for vertical, 0.12% of gross area
    As_min_base = 0.12 / 100 * geo.t_base * 1000 * 1000
    
    return {
        'Mu_stem': Mu_stem, 'd_stem': d_stem/1000, 'As_stem_req': max(As_stem_req, As_min_stem),
        'Mu_toe': Mu_toe, 'd_toe': d_base/1000, 'As_toe_req': max(As_toe_req, As_min_base),
        'Mu_heel': Mu_heel, 'd_heel': d_base/1000, 'As_heel_req': max(As_heel_req, As_min_base),
    }

# ------------------------- Drawings (Matplotlib) ------------------------- #

def plot_wall_section(geo: Geometry, mat: Materials, show_rebar=True):
    """
    Plots the wall section geometry and reinforcement schematic.
    
    CORRECTED: Added figsize and explicitly set aspect ratio for non-distortion.
    """
    
    # ------------------ Drawing Setup ------------------
    fig, ax = plt.subplots(figsize=(10, 6)) # Set fixed figure size
    
    # Wall Dimensions
    B = geo.B
    H_wall = geo.H + geo.t_base
    
    # ------------------ Concrete Outline ------------------
    # Base
    ax.add_patch(plt.Rectangle((0, 0), B, geo.t_base, fill=False, edgecolor='k', linewidth=2))
    
    # Stem (simplified as rectangular)
    stem_x = geo.toe
    stem_y = geo.t_base
    ax.add_patch(plt.Rectangle((stem_x, stem_y), geo.t_stem, geo.H, fill=False, edgecolor='k', linewidth=2))
    
    # Shear Key (if present)
    if geo.shear_key_depth > 0:
        key_x = stem_x + geo.t_stem / 2 - 0.5 * geo.t_stem # Centered under stem
        key_y = -geo.shear_key_depth
        ax.add_patch(plt.Rectangle((key_x, key_y), geo.t_stem, geo.shear_key_depth, fill=False, edgecolor='k', linewidth=2))

    # Ground Level
    ax.plot([-0.3, B + 0.3], [geo.t_base + geo.Df, geo.t_base + geo.Df], 'k--')
    ax.text(B + 0.3, geo.t_base + geo.Df, 'GL', va='bottom', ha='left')
    
    # Backfill Line
    ax.plot([geo.toe + geo.t_stem, B], [H_wall, H_wall], 'k:')

    # ------------------ Reinforcement Schematic ------------------
    if show_rebar:
        # Cover (m)
        c = mat.cover / 1000
        
        # Stem (Vertical bars - Tension face is back/heel side)
        ax.plot([geo.toe + geo.t_stem - c, geo.toe + geo.t_stem - c], [stem_y + c, H_wall - c], 'r--')
        
        # Heel (Top bars - Tension face is top)
        ax.plot([geo.toe + geo.t_stem + c, B - c], [geo.t_base - c, geo.t_base - c], 'b--')
        
        # Toe (Bottom bars - Tension face is bottom)
        ax.plot([c, geo.toe + geo.t_stem - c], [c, c], 'g--')
        
    # ------------------ Annotations and Labels ------------------
    # Dimensions (H, Df, B)
    ax.annotate('', xy=(B + 0.1, 0), xytext=(B + 0.1, H_wall), arrowprops={'arrowstyle': '<->'})
    ax.text(B + 0.15, H_wall / 2, f'H={geo.H:.2f}m', ha='left', va='center', rotation=90)
    
    ax.annotate('', xy=(0, -0.3), xytext=(B, -0.3), arrowprops={'arrowstyle': '<->'})
    ax.text(B / 2, -0.4, f'B={geo.B:.2f}m', ha='center', va='top')
    
    # Ground water level
    gwl = loads.gwl_h_from_base
    if gwl > 0:
        ax.plot([geo.toe + geo.t_stem, B], [geo.t_base + gwl, geo.t_base + gwl], 'c-.')
        ax.text(B + 0.3, geo.t_base + gwl, 'GWL', va='center', ha='left', color='c')
        
    # ------------------ Plot Settings ------------------
    ax.set_aspect('equal', 'box') # Enforce equal scale
    ax.set_xlim(-0.3, B + 0.3)
    ax.set_ylim(-max(0.3, geo.shear_key_depth + 0.1), H_wall + 0.3)
    
    ax.set_title('Wall Section & Bar Schematic (Drawn to Scale)') # Updated Title
    ax.set_xlabel('m')
    ax.set_ylabel('m')
    ax.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig)

def plot_pressure_diagram(H: float, p_earth: float, p_surcharge: float, p_water: float):
    """Plots the lateral pressure diagram."""
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Coordinates: Y-axis is depth (from top of wall)
    Y = np.array([0, H])
    
    # Earth pressure
    ax.plot([0, p_earth], Y, 'k-')
    ax.fill_between([0, p_earth], Y, 0, color='yellow', alpha=0.3, label='Earth')
    ax.text(p_earth, H, f'{p_earth:.2f} kPa', va='top', ha='left')
    
    # Surcharge pressure
    ax.plot([p_surcharge, p_surcharge], Y, 'b--')
    ax.fill_between([0, p_surcharge], Y, color='blue', alpha=0.1, label='Surcharge')
    ax.text(p_surcharge + 0.1, H / 2, f'{p_surcharge:.2f} kPa', va='center', ha='left')
    
    # Water pressure
    if p_water > 0:
        Y_w = np.array([H - loads.gwl_h_from_base, H])
        ax.plot([0, p_water], Y_w, 'c-')
        ax.fill_between([0, p_water], Y_w, H, color='cyan', alpha=0.3, label='Water')
        ax.text(p_water, H, f'{p_water:.2f} kPa', va='bottom', ha='left')
        
    ax.set_title('Lateral Pressure Diagram')
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
    ax.set_title('Resultant Lateral Forces (kN/m)')
    ax.set_xlabel('Force (kN/m)')
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    
    for i, v in enumerate(values):
        ax.text(v + 0.5, i, f'{v:.2f}', va='center')
        
    st.pyplot(fig)

# ------------------------- BOQ Functions ------------------------- #

def make_boq(geo: Geometry, mat: Materials, provided_steel: Dict[str, float]) -> pd.DataFrame:
    """Calculates Bill of Quantities (per meter length)."""
    
    # Concrete Volume (m3/m)
    V_stem = geo.t_stem * geo.H
    V_base = geo.B * geo.t_base
    V_key = geo.t_stem * geo.shear_key_depth
    V_conc = V_stem + V_base + V_key
    
    # Steel Mass (kg/m)
    # Main steel (Vertical stem, Top heel, Bottom toe)
    As_main = provided_steel['As_stem'] + provided_steel['As_heel'] + provided_steel['As_toe']
    
    # Distribution steel (Approx 0.12% gross area)
    As_dist_stem = 0.0012 * geo.t_stem * 1000 * geo.H * 1000 # mm2/m^2 (simplified as a continuous sheet)
    As_dist_base = 0.0012 * geo.t_base * 1000 * geo.B * 1000 # mm2/m^2
    
    As_total_mm2_m3 = As_main * 1000 / (V_conc * 1000) # Simplified total As (mm2/m^3)
    
    # Total Steel Mass (kg/m) = Vol_conc * density_steel * As_total / 1000
    # A simpler approach: total mass from As (mm2/m) * length (m) * density (kg/m3)
    
    # Approximate total steel mass (very rough estimate)
    # Mass/m = (As_total mm2/m * 1m * 7850 kg/m3) / 1000 mm2/m2
    # Let's use rebar in kg/m3 (standard industry metric)
    steel_kg_m3 = 80 # Assume 80 kg/m3 for an average wall
    
    # Better: use As_total (mm2/m) * L (m) * (density / 1000^2)
    # Total steel area per meter run (main + distribution)
    # A_total_mm2_m = As_main + As_dist_stem + As_dist_base # Simplified sum
    
    # Industry standard is kg/m3 of concrete
    M_steel = V_conc * steel_kg_m3
    
    df = pd.DataFrame({
        'Item': ['Concrete Volume', 'Steel Mass (Approx.)', 'Formwork Area'],
        'Unit': ['m³/m', 'kg/m³', 'm²/m'],
        'Value': [V_conc, steel_kg_m3, (geo.H * 2) + (geo.t_base * 2)]
    })
    
    # Add a row for total mass per meter run
    df.loc[len(df)] = ['Total Steel Mass', 'kg/m', M_steel]
    
    return df

# ------------------------- PDF Report Functions (Placeholder for brevity) ------------------------- #

# Note: The actual implementation of make_pdf_bytes requires the reportlab library
# and is often quite long. It is omitted here for brevity and focus on the Streamlit
# layout and drawing corrections.

def make_pdf_bytes(project_name, geo, soil, loads, bearing, pres, stab, desg):
    """Placeholder for PDF generation. Requires reportlab."""
    if pdfcanvas is None:
        raise RuntimeError('Reportlab library not found.')
        
    buffer = io.BytesIO()
    c = pdfcanvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    c.drawString(30 * mm, height - 20 * mm, f"Retaining Wall Design Report - {project_name}")
    c.drawString(30 * mm, height - 30 * mm, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Example content
    c.drawString(30 * mm, height - 50 * mm, f"Wall Height (H): {geo.H:.2f} m")
    c.drawString(30 * mm, height - 60 * mm, f"Base Width (B): {geo.B:.2f} m")
    c.drawString(30 * mm, height - 70 * mm, f"Overturning FOS: {stab['FOS_OT']:.2f}")
    c.drawString(30 * mm, height - 80 * mm, f"Sliding FOS: {stab['FOS_SL']:.2f}")
    c.drawString(30 * mm, height - 90 * mm, f"Max Bearing Pressure: {stab['q_max']:.2f} kPa")
    
    c.showPage()
    c.save()
    return buffer.getvalue()

def make_dxf_bytes(geo: Geometry, mat: Materials, project_name: str):
    """Placeholder for DXF generation. Requires ezdxf."""
    if ezdxf is None:
        raise RuntimeError('ezdxf library not found.')
        
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Add a rectangle for the base
    msp.add_lwpolyline([(0, 0), (geo.B, 0), (geo.B, geo.t_base), (0, geo.t_base), (0, 0)], close=True)
    
    # Add a rectangle for the stem
    stem_x = geo.toe
    stem_y = geo.t_base
    msp.add_lwpolyline([(stem_x, stem_y), (stem_x + geo.t_stem, stem_y), 
                        (stem_x + geo.t_stem, stem_y + geo.H), (stem_x, stem_y + geo.H), 
                        (stem_x, stem_y)], close=True)
                        
    # Save DXF to bytes buffer
    buffer = io.BytesIO()
    doc.saveas(buffer)
    return buffer.getvalue()

# ------------------------- Streamlit App ------------------------- #

st.set_page_config(layout="wide", page_title="Retaining Wall Design Wizard")

# --- Initial State/Default Values ---
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
    
# --- Sidebar (Inputs) ---
st.sidebar.title('Design Inputs 🏗️')

project_name = st.sidebar.text_input('Project Name', 'RW-01')
wall_type = st.sidebar.selectbox('Wall Type', ['Cantilever T-Wall', 'Cantilever L-Wall', 'Cantilever Inverted-T Wall'])
design_state = st.sidebar.selectbox('Earth Pressure State', ['Active', 'At-Rest', 'Seismic (Mononobe-Okabe)'])

st.sidebar.subheader('1. Geometry')
geo = st.session_state['geo']
geo.H = st.sidebar.number_input('Height of Stem (H, m)', 1.0, 15.0, geo.H, 0.1)
geo.Df = st.sidebar.number_input('Depth of Embedment (Df, m)', 0.5, 5.0, geo.Df, 0.1)
geo.t_stem = st.sidebar.number_input('Stem Thickness (t_stem, m)', 0.2, 2.0, geo.t_stem, 0.05)
geo.t_base = st.sidebar.number_input('Base Slab Thickness (t_base, m)', 0.3, 2.0, geo.t_base, 0.05)
geo.toe = st.sidebar.number_input('Toe Length (toe, m)', 0.1, 5.0, geo.toe, 0.1)
geo.heel = st.sidebar.number_input('Heel Length (heel, m)', 0.1, 5.0, geo.heel, 0.1)
geo.B = geo.toe + geo.t_stem + geo.heel # Calculated Base Width
st.sidebar.markdown(f'**Base Width (B): {geo.B:.2f} m**')
geo.shear_key_depth = st.sidebar.number_input('Shear Key Depth (m)', 0.0, 1.0, geo.shear_key_depth, 0.1)


st.sidebar.subheader('2. Soil Properties')
soil = st.session_state['soil']
soil.gamma_fill = st.sidebar.number_input('Unit Weight of Backfill (γ_fill, kN/m³)', 16.0, 22.0, soil.gamma_fill, 1.0)
soil.phi = st.sidebar.slider('Angle of Internal Friction (φ, deg)', 25.0, 40.0, soil.phi, 0.5)
soil.gamma = st.sidebar.number_input('Unit Weight of Foundation Soil (γ, kN/m³)', 16.0, 22.0, soil.gamma, 1.0)
soil.c_base = st.sidebar.number_input('Base-Soil Cohesion (c_base, kPa)', 0.0, 50.0, soil.c_base, 1.0)


st.sidebar.subheader('3. Loads')
loads = st.session_state['loads']
loads.surcharge_q = st.sidebar.number_input('Surcharge (q, kPa)', 0.0, 50.0, loads.surcharge_q, 5.0)
gwl_h = st.sidebar.number_input('GWL Height from Base (m)', 0.0, geo.H + geo.t_base, loads.gwl_h_from_base, 0.1)
loads.gwl_h_from_base = gwl_h

st.sidebar.subheader('4. Seismic')
loads.use_seismic = st.sidebar.checkbox('Include Seismic Load (MO Approx.)', loads.use_seismic)
if loads.use_seismic:
    loads.seismic_kh = st.sidebar.number_input('Horizontal Acc. (kh)', 0.01, 0.3, loads.seismic_kh, 0.01)
    loads.seismic_kv = st.sidebar.number_input('Vertical Acc. (kv)', 0.0, 0.1, loads.seismic_kv, 0.01)
else:
    loads.seismic_kh = 0.0
    loads.seismic_kv = 0.0

st.sidebar.subheader('5. Materials & Bearing')
mat = st.session_state['mat']
mat.fck = st.sidebar.selectbox('Concrete Grade (fck)', [20, 25, 30, 35], index=1)
mat.fy = st.sidebar.selectbox('Steel Grade (fy)', [415, 500], index=0)
mat.cover = st.sidebar.number_input('Clear Cover (mm)', 25, 100, mat.cover, 5)

bearing = st.session_state['bearing']
bearing.SBC_allow = st.sidebar.number_input('Allowable Bearing Capacity (SBC, kPa)', 50.0, 500.0, bearing.SBC_allow, 10.0)
bearing.mu_base = st.sidebar.slider('Base-Soil Friction Coeff (μ)', 0.3, 0.7, bearing.mu_base, 0.05)
bearing.include_passive = st.sidebar.checkbox('Include Passive Resistance (Pp)', bearing.include_passive)
if bearing.include_passive:
    bearing.passive_reduction = st.sidebar.slider('Pp Reduction Factor (η)', 0.5, 1.0, bearing.passive_reduction, 0.05)


# ------------------------- Core Calculations (Run Once) ------------------------- #
# These variables must be calculated before any output section.
pres = pressures(geo.H, soil, loads, design_state)
stab = stability(geo, soil, loads, bearing, pres)
pres.update({'q_avg': stab['q_avg'], 'q_max': stab['q_max'], 'q_min': stab['q_min']})
desg = member_design(geo, soil, loads, pres, mat)

# ------------------------- Sequential Output Display (Main Page) ------------------------- #
st.title('Retaining Wall Design Wizard')
st.header('Design Summary and Calculations')
st.caption('Use your browser\'s **Print** function (**Ctrl+P** or **Cmd+P**) to capture all calculations on this single page.')
st.divider()

# --- SECTION 1: Inputs Summary ---
with st.expander('1. Design Inputs Summary', expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Project:** {project_name}")
        st.markdown(f"**Wall Type:** {wall_type}")
        st.markdown(f"**Earth State:** {design_state}")
        st.markdown(f"**H:** {geo.H:.2f} m, **Df:** {geo.Df:.2f} m")
        st.markdown(f"**B:** {geo.B:.2f} m (toe {geo.toe:.2f} m, heel {geo.heel:.2f} m)")
    with c2:
        st.markdown(f"**Stem/Base:** {geo.t_stem:.2f} / {geo.t_base:.2f} m")
        st.markdown(f"**γ_soil:** {soil.gamma:.2f} kN/m³, **φ:** {soil.phi:.1f}°")
        st.markdown(f"**q:** {loads.surcharge_q:.1f} kPa, **GWL:** {gwl_h:.2f} m above base")
        st.markdown(f"**Seismic:** {loads.use_seismic} (kh={loads.seismic_kh:.02f}, kv={loads.seismic_kv:.02f})")
    with c3:
        st.markdown(f"**fck:** M{mat.fck}, **fy:** Fe{mat.fy}")
        st.markdown(f"**Cover:** {mat.cover:.0f} mm")
        st.markdown(f"**SBC_allow:** {bearing.SBC_allow:.0f} kPa, **μ:** {bearing.mu_base:.2f}")
        st.markdown(f"**Passive:** {bearing.include_passive} (η={bearing.passive_reduction:.2f})")
        st.markdown(f"**Shear Key:** {'Yes' if geo.shear_key_depth > 0 else 'No'}")

# --- SECTION 2: Drawings ---
st.divider()
with st.expander('2. Geometry and Pressure Diagrams', expanded=True):
    col_draw, col_press = st.columns([2, 1])
    with col_draw:
        st.subheader('Wall Section (Drawn to Scale)')
        plot_wall_section(geo, mat) 
    with col_press:
        st.subheader('Lateral Pressure Distribution')
        plot_pressure_diagram(geo.H, pres['p0_earth'], pres['p_surcharge'], pres['p0_water'])
        st.subheader('Resultant Forces')
        plot_load_resultants(pres['P_earth'], pres['P_surcharge'], pres['P_water'])


# --- SECTION 3: Stability Checks ---
st.divider()
with st.expander('3. Stability Checks (SLS)', expanded=True):
    st.subheader('Summary of Forces and Moments')
    df_stab = pd.DataFrame({
        'Quantity': ['Total Lateral Force, H (kN/m)', 'Total Vertical Force, V (kN/m)', 'Overturning Moment, M_o (kNm/m)', 'Resisting Moment, M_r (kNm/m)', 'FOS Overturning', 'FOS Sliding', 'Eccentricity, e (m)', 'Max Bearing Pressure, q_max (kPa)', 'Min Bearing Pressure, q_min (kPa)'],
        'Value': [stab['H'], stab['V'], stab['M_o'], stab['M_r'], stab['FOS_OT'], stab['FOS_SL'], stab['e'], stab['q_max'], stab['q_min']]
    })
    st.dataframe(df_stab, use_container_width=True, hide_index=True)

    st.subheader('Code Compliance Check')
    ok1 = stab['FOS_OT'] >= 2.0
    ok2 = stab['FOS_SL'] >= 1.5
    ok3 = (abs(stab['e']) <= geo.B/6.0) and (stab['q_max'] <= bearing.SBC_allow) and (stab['q_min'] >= 0)
    st.markdown(f"**Overturning FOS (Required ≥ 2.0):** {'✅ PASS' if ok1 else '❌ FAIL'}")
    st.markdown(f"**Sliding FOS (Required ≥ 1.5):** {'✅ PASS' if ok2 else '❌ FAIL'}")
    st.markdown(f"**Bearing & Eccentricity (q_max ≤ {bearing.SBC_allow:.0f} kPa, e ≤ B/6):** {'✅ PASS' if ok3 else '❌ FAIL'}")


# --- SECTION 4: Member Design & BOQ ---
st.divider()
with st.expander('4. Member Design (ULS) and Bill of Quantities', expanded=True):
    col_des, col_boq = st.columns([2, 1])

    with col_des:
        st.subheader('Flexural Design Summary (Required vs. Provided)')
        st.markdown(f"**Design Moments (kNm/m):** $M_u (stem)={desg['Mu_stem']:.2f}$, $M_u (heel)={desg['Mu_heel']:.2f}$, $M_u (toe)={desg['Mu_toe']:.2f}$")

        # Rebar input fields (must be kept here for design calculation)
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
        
        # Display Design Table
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

    # DXF Export
    c_dxf, c_pdf, c_excel = st.columns(3)
    if ezdxf is not None:
        try:
            dxf_bytes = make_dxf_bytes(geo, mat, project_name)
            c_dxf.download_button('Download DXF (CAD) 📐', data=dxf_bytes, file_name=f'{project_name}_wall_section.dxf', mime='application/octet-stream')
        except RuntimeError:
            c_dxf.warning('ezdxf issue. Check installation.')
    else:
        c_dxf.info('DXF requires "ezdxf". Run: pip install ezdxf')

    # PDF Report
    if pdfcanvas is not None:
        try:
            pdf_bytes = make_pdf_bytes(project_name, geo, soil, loads, bearing, pres, stab, desg)
            c_pdf.download_button('Download PDF Report 📄', data=pdf_bytes, file_name=f'{project_name}_Report.pdf', mime='application/pdf')
        except RuntimeError:
            c_pdf.warning('reportlab issue. Check implementation.')
    else:
        c_pdf.info('PDF requires "reportlab". Run: pip install reportlab')

    # BOQ Excel
    boq_df = make_boq(geo, mat, provided_steel)
    f = io.BytesIO()
    boq_df.to_excel(f, sheet_name='BOQ', index=False)
    c_excel.download_button('Download BOQ (Excel) 📊', data=f, file_name=f'{project_name}_BOQ.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
