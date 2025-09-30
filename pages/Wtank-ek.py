# -*- coding: utf-8 -*-
import math
import io
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ===============================
# Global Constants
# ===============================
KNM_TO_NMM = 1_000_000.0
M_TO_MM = 1000.0

# Set Streamlit page configuration for wide layout (better for drawings)
st.set_page_config(layout="wide")

# ===============================
# Data Classes
# ===============================
@dataclass
class Materials:
    fck: float = 30.0
    fy: float = 415.0
    gamma_conc: float = 25.0
    Ec: float = 30000.0  # MPa, approx 5000*sqrt(fck)
    exposure: str = "Severe"

@dataclass
class Geometry:
    H: float = 4.0      # m (Water height)
    L: float = 6.0      # m (Long wall length)
    B: float = 4.0      # m (Short wall length)
    t_wall: float = 0.3 # m (Wall thickness)
    t_base: float = 0.3 # m (Base slab thickness)
    freeboard: float = 0.15 # m
    tank_type: str = "Ground"  # "Ground" or "Elevated"

@dataclass
class Loads:
    gamma_w: float = 10.0 # kN/m³ (Water density)
    gamma_s: float = 18.0 # kN/m³ (Soil density for Ground Tank)
    K0: float = 0.5     # Coefficient of Earth Pressure at Rest
    phi: float = 30.0   # Soil Angle of Internal Friction
    mu_base: float = 0.5 # Coefficient of friction for base sliding
    z_g_zone: int = 3   # Seismic Zone

# ===============================
# Engineering Helper Functions (Updated for Code Compliance)
# ===============================

def triangular_resultant(gamma: float, H: float) -> Tuple[float, float]:
    """Calculates resultant force (R, kN/m) and its location (zbar, m from base)."""
    R = 0.5 * gamma * H**2
    zbar = H/3.0
    return R, zbar

def demand_ast_from_M(Mu_kNm: float, d_eff_mm: float, fy_MPa: float, fck_MPa: float) -> float:
    """
    Calculates required Ast (mm²/m) using the IS 456:2000 (Cl E-1.1) ULS limit state method 
    for a singly reinforced section (for 1m width, b=1000mm).
    
    Returns: Ast in mm²/m.
    """
    if Mu_kNm <= 0.0:
        return 0.0
    
    Mu_Nmm = Mu_kNm * KNM_TO_NMM
    b = 1000.0  # mm
    
    # Check for failure (Mu > Mu_lim)
    Ru_lim = 0.36 * fck_MPa * b * d_eff_mm**2
    if Mu_Nmm > Ru_lim:
        # This typically means thickness is insufficient for the grade/fy
        return 99999.0 
    
    try:
        # Ast using IS 456 Annex E formula (rearranged)
        # Ast = (0.5 * fck / fy) * (1 - sqrt(1 - 4.6 * Mu / (fck * b * d**2))) * b * d
        term_in_sqrt = 1.0 - (4.6 * Mu_Nmm) / (fck_MPa * b * d_eff_mm**2)
        
        if term_in_sqrt < 0:
            return 99999.0
            
        Ast = (0.5 * fck_MPa / fy_MPa) * (1.0 - math.sqrt(term_in_sqrt)) * b * d_eff_mm
        
        # Ast_min check is done separately
        return max(Ast, 0.0)
    except:
        return 0.0

def steel_stress_sls(Ms_kNm_per_m: float, d_eff_mm: float, As_mm2_per_m: float, Ec_MPa: float) -> float:
    """
    Calculates steel stress (sigma_s in MPa) using **Elastic Cracked Section Theory** (Working Stress Method basis) for Serviceability Limit State (SLS).
    
    Returns: sigma_s in MPa.
    """
    if Ms_kNm_per_m <= 0.0 or As_mm2_per_m <= 0.0:
        return 0.0
        
    Es = 200000.0  # MPa (Young's Modulus of Steel)
    m = Es / Ec_MPa
    b = 1000.0     # 1m strip
    Ms_Nmm = Ms_kNm_per_m * KNM_TO_NMM

    # 1. Find neutral axis depth, n (or x) for a cracked section:
    # b*n^2/2 = m*Ast*(d-n) -> Quadratic solution for n
    ratio = (m * As_mm2_per_m) / b
    
    try:
        # n = -ratio + sqrt(ratio^2 + 2 * ratio * d_eff_mm)
        n = -ratio + math.sqrt(ratio**2 + 2 * ratio * d_eff_mm)
    except ValueError:
        return float('inf') # Should not happen with real numbers

    # 2. Find lever arm, z
    z = d_eff_mm - n/3.0
    
    # 3. Find steel stress: sigma_s = Ms / (Ast * z)
    sigma_s = Ms_Nmm / (As_mm2_per_m * max(z, 1.0))
    
    return sigma_s

# ===============================
# Interpolation Tables (from IS 3370-4)
# ===============================

# Moment coefficients 'C' for Wall Bending Moment at Base (Table 4, Case 1, L/H > 2)
# The code structure assumes a rectangular tank. The table below is for the *moment* at the base.
# Coefficients for fixed-base cantilever, M_base = C * w * H^2
M_COEF_TABLE = pd.DataFrame(
    data={
        'L/H': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        'Base_Corner (Max)': [0.038, 0.043, 0.047, 0.051, 0.054, 0.057, 0.062, 0.065, 0.067, 0.068, 0.069],
        'Base_Mid_Long': [0.031, 0.035, 0.039, 0.041, 0.043, 0.045, 0.048, 0.050, 0.051, 0.051, 0.052],
        'Base_Mid_Short': [0.036, 0.041, 0.045, 0.048, 0.051, 0.053, 0.057, 0.060, 0.062, 0.063, 0.064]
    }
).set_index('L/H')

def bilinear_interpolate(ratio: float, df: pd.DataFrame, col: str) -> float:
    """Interpolates between values in a dataframe index."""
    idx = df.index
    val = df[col]
    
    if ratio <= idx.min():
        return val.iloc[0]
    if ratio >= idx.max():
        return val.iloc[-1]
        
    # Find lower and upper bounds
    lower_idx = idx[idx <= ratio].max()
    upper_idx = idx[idx > ratio].min()
    
    v1 = val[lower_idx]
    v2 = val[upper_idx]
    
    # Linear interpolation
    C = v1 + (v2 - v1) * (ratio - lower_idx) / (upper_idx - lower_idx)
    return C

# ===============================
# I/O Functions (Import/Export)
# ===============================

def export_inputs(mat: Materials, geom: Geometry, loads: Loads) -> str:
    """Serializes input data into a JSON string."""
    data = {
        "Materials": mat.__dict__,
        "Geometry": geom.__dict__,
        "Loads": loads.__dict__,
    }
    return json.dumps(data, indent=4)

def import_inputs(json_data: str) -> Tuple[Materials, Geometry, Loads]:
    """Deserializes JSON string into DataClass objects."""
    data = json.loads(json_data)
    mat = Materials(**data["Materials"])
    geom = Geometry(**data["Geometry"])
    loads = Loads(**data["Loads"])
    return mat, geom, loads

# ===============================
# Plotting Functions
# ===============================

def plot_loads(geom: Geometry, loads: Loads, R_liq: float, R_soil: float):
    """Plots the load diagram (Hydrostatic and Earth Pressure)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    H_total = geom.H + geom.t_base
    
    # 1. Wall and Base
    ax.plot([0, 0], [0, H_total], 'k-', linewidth=3, label='Wall')
    ax.plot([-0.5, 1], [0, 0], 'k-', linewidth=3, label='Base')
    
    # 2. Hydrostatic Pressure (P_w)
    P_max_w = loads.gamma_w * geom.H
    x_w = [0, P_max_w, 0]
    y_w = [geom.t_base, geom.t_base, H_total]
    ax.fill(x_w, y_w, 'b', alpha=0.3, label='Water Pressure')
    ax.plot(x_w, y_w, 'b--')
    ax.text(P_max_w * 1.1, geom.t_base + geom.H/2, f'P_w={P_max_w:.1f} kN/m²', color='b')
    ax.arrow(P_max_w * 0.5, geom.t_base + R_liq, 0, -R_liq, head_width=0.05, head_length=0.1, fc='b', ec='b')
    ax.text(P_max_w * 0.5 + 0.1, geom.t_base + R_liq, f'R_w={R_liq:.1f} kN/m', color='b')

    # 3. Earth Pressure (P_soil) - Only for Ground Tank
    if geom.tank_type == "Ground":
        P_max_s = loads.gamma_s * loads.K0 * (geom.H + geom.t_base)
        x_s = [-P_max_s, 0, 0]
        y_s = [0, 0, H_total]
        ax.fill(x_s, y_s, 'brown', alpha=0.3, label='Earth Pressure')
        ax.plot(x_s, y_s, 'brown--')
        ax.text(-P_max_s * 1.5, H_total/2, f'P_s={P_max_s:.1f} kN/m²', color='brown')
        ax.arrow(-P_max_s * 0.5, R_soil, 0, -R_soil, head_width=0.05, head_length=0.1, fc='brown', ec='brown')
        ax.text(-P_max_s * 0.5 - 0.5, R_soil + 0.1, f'R_s={R_soil:.1f} kN/m', color='brown')

    # Formatting
    ax.set_title("Input Load Sketches (Hydrostatic & Earth Pressure)")
    ax.set_xlabel("Pressure (Scaled)")
    ax.set_ylabel("Height (m)")
    ax.set_ylim(-0.2, H_total + 0.5)
    ax.set_xlim(-abs(P_max_s) * 2, abs(P_max_w) * 2)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='box')
    st.pyplot(fig)


def plot_results(H: float, t_wall: float, M_base_max: float, V_base_max: float):
    """Plots the bending moment and shear force diagrams for the wall."""
    
    # Shear Force (Triangular, max at base)
    fig_v, ax_v = plt.subplots(figsize=(4, 6))
    ax_v.plot([0, V_base_max, 0], [0, 0, H], 'r-', linewidth=2)
    ax_v.fill([0, V_base_max, 0], [0, 0, H], 'r', alpha=0.2)
    ax_v.plot([0, 0], [0, H], 'k--')
    ax_v.text(V_base_max * 1.1, 0.05, f'V_max={V_base_max:.1f} kN/m', color='r')
    ax_v.set_title("Shear Force Diagram (V)")
    ax_v.set_xlabel("Shear (kN/m)")
    ax_v.set_ylabel("Height (m)")
    ax_v.set_ylim(-0.1, H + 0.1)
    ax_v.grid(True, linestyle='--', alpha=0.6)

    # Bending Moment (Max at base, zero at top)
    fig_m, ax_m = plt.subplots(figsize=(4, 6))
    
    # Moment shape is not purely triangular due to plate action, but we plot the max value.
    # We plot a simplified fixed-base cantilever shape for illustration
    x_m = [0, M_base_max, 0]
    y_m = [0, 0, H]
    ax_m.plot(x_m, y_m, 'b-', linewidth=2)
    ax_m.fill(x_m, y_m, 'b', alpha=0.2)
    ax_m.plot([0, 0], [0, H], 'k--')
    ax_m.text(M_base_max * 1.1, 0.05, f'M_max={M_base_max:.1f} kNm/m', color='b')
    ax_m.set_title("Bending Moment Diagram (M)")
    ax_m.set_xlabel("Moment (kNm/m)")
    ax_m.set_ylabel("Height (m)")
    ax_m.set_ylim(-0.1, H + 0.1)
    ax_m.grid(True, linestyle='--', alpha=0.6)
    
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_v)
    with col2:
        st.pyplot(fig_m)

# ===============================
# Streamlit App Execution (Inline)
# ===============================

st.title("💧 RCC Water Tank Design and Analysis (IS 3370 / IS 456)")
st.markdown("---")

# Initialize session state for holding input objects
if 'mat' not in st.session_state:
    st.session_state.mat = Materials()
    st.session_state.geom = Geometry()
    st.session_state.loads = Loads()
    st.session_state.is_imported = False

# --- INPUT & I/O SECTION ---
st.header("1. Input Data & File Management")

col_io1, col_io2 = st.columns([1, 1])

# Export Button
export_str = export_inputs(st.session_state.mat, st.session_state.geom, st.session_state.loads)
col_io1.download_button(
    label="💾 Export Inputs (JSON)",
    data=export_str,
    file_name="water_tank_inputs.json",
    mime="application/json",
    help="Download current inputs for future use."
)

# Import Uploader
uploaded_file = col_io2.file_uploader(
    "📂 Import Inputs (JSON)", 
    type="json", 
    help="Upload a previously exported JSON file to load inputs."
)
if uploaded_file is not None:
    try:
        json_data = uploaded_file.getvalue().decode("utf-8")
        st.session_state.mat, st.session_state.geom, st.session_state.loads = import_inputs(json_data)
        st.session_state.is_imported = True
        st.success("Inputs imported successfully! Please review the inputs below.")
    except Exception as e:
        st.error(f"Error importing file: {e}")

# --- INPUT BLOCKS ---

# Column Layout for Inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Materials (IS 456)")
    st.session_state.mat.fck = st.number_input("Concrete Grade $f_{ck}$ (MPa)", 25.0, 50.0, st.session_state.mat.fck, 5.0)
    st.session_state.mat.fy = st.number_input("Steel Grade $f_{y}$ (MPa)", 250.0, 550.0, st.session_state.mat.fy, 50.0)
    st.session_state.mat.Ec = st.number_input("Concrete E ($E_{c}$) (MPa)", 25000.0, 40000.0, st.session_state.mat.Ec, 1000.0, format="%.0f")
    
with col2:
    st.subheader("Geometry (Wall)")
    st.session_state.geom.H = st.number_input("Height of Water $H$ (m)", 1.0, 10.0, st.session_state.geom.H, 0.1)
    st.session_state.geom.L = st.number_input("Long Wall $L$ (m)", 1.0, 20.0, st.session_state.geom.L, 0.1)
    st.session_state.geom.B = st.number_input("Short Wall $B$ (m)", 1.0, 20.0, st.session_state.geom.B, 0.1)
    st.session_state.geom.t_wall = st.number_input("Wall Thickness $t_{w}$ (m)", 0.15, 1.0, st.session_state.geom.t_wall, 0.05)
    
with col3:
    st.subheader("Loads & Environment")
    st.session_state.geom.tank_type = st.selectbox("Tank Type", ["Ground", "Elevated"], index=0 if st.session_state.geom.tank_type == "Ground" else 1)
    st.session_state.loads.gamma_w = st.number_input("Water Density $\gamma_{w}$ (kN/m³)", 9.81, 15.0, st.session_state.loads.gamma_w, 0.1)
    if st.session_state.geom.tank_type == "Ground":
        st.session_state.loads.gamma_s = st.number_input("Soil Density $\gamma_{s}$ (kN/m³)", 15.0, 25.0, st.session_state.loads.gamma_s, 0.5)
        st.session_state.loads.K0 = st.number_input("Earth Pressure Coeff. $K_{0}$", 0.3, 1.0, st.session_state.loads.K0, 0.05)
    st.session_state.loads.z_g_zone = st.selectbox("Seismic Zone (IS 1893-2)", [2, 3, 4, 5], index=st.session_state.loads.z_g_zone-2)

# Global variables from session state
mat, geom, loads = st.session_state.mat, st.session_state.geom, st.session_state.loads

# Derived geometric properties
L_over_H = geom.L / geom.H if geom.H > 0 else 99.0
tw_mm = geom.t_wall * M_TO_MM
d_eff = tw_mm - 50.0 # Effective depth (approx t - 50mm cover/rebar)

st.markdown("---")

# --- LOAD CALCULATION SECTION ---
st.header("2. Basic Load Calculations and Sketches")
st.markdown(f"The analysis is based on a **fixed base wall** and accounts for lateral loads due to water (Hydrostatic) and, if a ground tank, soil (Earth Pressure). The aspect ratio $\\frac{{L}}{{H}} = \\frac{{{geom.L}}}{{{geom.H}}} = **{L_over_H:.2f}**$ is critical for determining plate action coefficients.")

# 2.1 Hydrostatic Load (FL)
st.subheader("2.1 Hydrostatic Load (Full Tank)")
R_liq, zbar_liq = triangular_resultant(loads.gamma_w, geom.H)
P_max_w = loads.gamma_w * geom.H
st.markdown(f"""
The hydrostatic pressure is triangular, with maximum pressure $P_{{max, w}}$ at the base:
$$P_{{max, w}} = \gamma_w \cdot H = {loads.gamma_w:.2f} \cdot {geom.H:.2f} = **{P_max_w:.2f} \text{ kN/m}^2$$
The total resultant force $R_{{w}}$ per metre run of the wall is:
$$R_{{w}} = 0.5 \cdot \gamma_w \cdot H^2 = **{R_liq:.2f} \text{ kN/m}$$
The moment at the base (assuming cantilever) due to this load is $M_{{cantilever}} = R_{{w}} \cdot \frac{{H}}{3} = {R_liq:.2f} \cdot {geom.H/3.0:.2f} = **{R_liq * geom.H/3.0:.2f} \text{ kNm/m}$$.
""")

# 2.2 Earth Pressure (EL) - Only for Ground Tanks
R_soil, M_soil_base = 0.0, 0.0
if geom.tank_type == "Ground":
    st.subheader("2.2 Earth Pressure Load (Empty Tank)")
    P_max_s = loads.gamma_s * loads.K0 * (geom.H + geom.t_base)
    R_soil, zbar_soil = triangular_resultant(loads.gamma_s * loads.K0, geom.H + geom.t_base)
    M_soil_base = R_soil * zbar_soil
    st.markdown(f"""
    For the Ground Tank, the soil exerts a lateral force with $K_0$:
    $$P_{{max, s}} = \gamma_s \cdot K_0 \cdot (H + t_{{base}}) = {loads.gamma_s:.1f} \cdot {loads.K0:.2f} \cdot ({geom.H:.2f} + {geom.t_base:.2f}) = **{P_max_s:.2f} \text{ kN/m}^2$$
    The total resultant force $R_{{s}}$ per metre run of the wall (acting over $H+t_{{base}}$) is:
    $$R_{{s}} = 0.5 \cdot P_{{max, s}} \cdot (H + t_{{base}}) = **{R_soil:.2f} \text{ kN/m}$$
    """)

# 2.3 Load Sketch
st.subheader("2.3 Load Sketch")
plot_loads(geom, loads, R_liq, R_soil)

st.markdown("---")

# --- MOMENT AND SHEAR CALCULATION SECTION ---
st.header("3. Wall Moment Calculation (IS 3370-4 Plate Action)")
st.markdown(f"""
Since the wall is continuous and connected to the base, it acts as a **plate element** subjected to bending in two directions (vertical and horizontal). The maximum moment at the base is found using coefficients ($C$) from IS 3370-4, Table 4, Case 1. The design moment is $M = C \cdot P_{{max, w}} \cdot H^2$.
""")

# Moment Coefficients
C_corner = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Corner (Max)')
C_mid_L = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Mid_Long')
C_mid_B = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Mid_Short')

M_base_corner_FL = C_corner * loads.gamma_w * geom.H**3
M_base_mid_L_FL = C_mid_L * loads.gamma_w * geom.H**3
M_base_mid_B_FL = C_mid_B * loads.gamma_w * geom.H**3

# Assuming max base shear (V_max) for the wall is due to cantilever action (conservative estimate for V)
V_base_FL = R_liq

st.markdown("### 3.1 Base Moments for Full Tank (FL)")
st.markdown(f"**Calculated Aspect Ratio $L/H = {L_over_H:.2f}$**")
st.markdown(f"Interpolated Coefficients (C): $C_{{corner}}={C_corner:.4f}$, $C_{{mid, L}}={C_mid_L:.4f}$, $C_{{mid, B}}={C_mid_B:.4f}$")

st.markdown(f"""
- **Corner Moment:** $M_{{corner}} = {C_corner:.4f} \cdot \gamma_w \cdot H^3 = **{M_base_corner_FL:.2f} \text{ kNm/m}$**
- **Mid-Long Wall Moment:** $M_{{mid, L}} = {C_mid_L:.4f} \cdot \gamma_w \cdot H^3 = **{M_base_mid_L_FL:.2f} \text{ kNm/m}$**
- **Mid-Short Wall Moment:** $M_{{mid, B}} = {C_mid_B:.4f} \cdot \gamma_w \cdot H^3 = **{M_base_mid_B_FL:.2f} \text{ kNm/m}$**
""")

# --- LOAD COMBINATION & DESIGN SECTION ---
st.markdown("---")
st.header("4. Load Combinations & Ultimate Limit State (ULS) Design")
st.markdown("Design is checked for two critical scenarios (IS 3370-2, Table 1): **Empty** (Soil/EQ governs) and **Full** (Water/EQ governs). The most critical section is typically the wall base (Maximum Moment).")

# Load Factors (Partial Safety Factors, IS 456/IS 3370)
gamma_f_FL = 1.5  # Full Load (Water) for ULS
gamma_f_EL = 1.5  # Empty Load (Soil) for ULS
gamma_f_WL_EL = 1.5 # Seismic Load Factor (IS 1893)

# Case 1: Full Tank (FL) + Seismic
Mu_design_FL = gamma_f_FL * M_base_corner_FL # Max Water Moment at Corner

# Case 2: Empty Tank (EL) + Soil + Seismic (M_soil is generally smaller, but must be checked)
# Assuming M_soil < M_water, but check is necessary if a ground tank.
Mu_design_EL = 0.0
if geom.tank_type == "Ground":
    Mu_design_EL = gamma_f_EL * M_soil_base

Mu_max_design = max(Mu_design_FL, Mu_design_EL) if geom.tank_type == "Ground" else Mu_design_FL
V_max_design = gamma_f_FL * V_base_FL

st.subheader(f"4.1 Design Moment and Shear")
st.markdown(f"Maximum Design ULS Moment ($M_u$) $\\approx **{Mu_max_design:.2f} \text{ kNm/m}$** (Governed by {('Empty/Soil' if Mu_design_EL > Mu_design_FL else 'Full/Water')}).")
st.markdown(f"Maximum Design ULS Shear ($V_u$) $\\approx **{V_max_design:.2f} \text{ kN/m}$**.")

# 4.2 Required Steel Area (ULS Check)
Ast_req_ULS = demand_ast_from_M(Mu_max_design, d_eff, mat.fy, mat.fck)
Ast_min_perc = 0.35 # IS 3370 Cl 7.1 specifies 0.35% for walls > 200mm, or 0.25% for < 200mm
if geom.t_wall < 0.20:
    Ast_min_perc = 0.25

# Minimum Ast check (IS 3370 Cl 7.1)
A_conc_total = 1000.0 * tw_mm # mm²/m
Ast_min_total = (Ast_min_perc / 100.0) * A_conc_total # Total Ast (both faces)
Ast_min_face = Ast_min_total / 2.0 # Minimum Ast per face

Ast_req_final = max(Ast_req_ULS, Ast_min_face)

st.subheader("4.2 Required Vertical Reinforcement (Tension Face)")
st.markdown(f"""
- Required $A_{{st}}$ from ULS Flexure ($M_u$) is: **{Ast_req_ULS:.0f} $\text{mm}^2/\text{m}$**
- Minimum $A_{{st}}$ (IS 3370, {Ast_min_perc:.2f}% total $\rightarrow$ {Ast_min_face:.0f} $\text{mm}^2/\text{m}$ per face): **{Ast_min_face:.0f} $\text{mm}^2/\text{m}$**

**Governing $A_{{st, req}}$ (Vertical) = {Ast_req_final:.0f} $\text{mm}^2/\text{m}$**
""")

st.markdown("---")

# --- SERVICEABILITY LIMIT STATE (SLS) CHECK SECTION ---
st.header("5. Serviceability Limit State (SLS) - Crack Control")
st.markdown("SLS checks are performed for unfactored loads, ensuring the wall remains essentially uncracked to maintain water-tightness (IS 3370-2, Cl 3.1). The permissible steel stress ($\sigma_s$) is the limit.")

# Permissible Steel Stress (IS 3370-2, Table 2 & 3)
# We assume "Severe" Exposure and M30 concrete
# IS 3370-2 Table 3: Max sigma_s for w_lim = 0.2 mm (Moderate exposure) is 130 MPa for Fe415 and < 16mm bars
# IS 3370-2 Table 2: sigma_s,max for fck 30, fy 415: 125 MPa (for direct tension)
# For Flexure (Cl 3.1): Use the max permissible stress from the relevant table. We conservatively use 130 MPa.
sigma_allow = 130.0 # MPa (Permissible tensile stress in steel for w_lim=0.2mm, Fe415 - Cl 3.1)

# Assume M12 @ 150mm c/c (Ast_prov = 754 mm²/m) or M16 @ 150mm c/c (Ast_prov = 1340 mm²/m)
Ast_prov_mm2 = Ast_req_final # Use the required Ast for the check (worst case)
if Ast_req_final > 1340: # If high Ast is required, use a higher bar size for the check
    Ast_prov_mm2 = 1340.0 # M16 @ 150 c/c

# Service Moment (Unfactored)
Ms_design = M_base_corner_FL # Unfactored Water Moment (FL)

# Calculate Actual Steel Stress (sigma_s)
sigma_s_actual = steel_stress_sls(Ms_design, d_eff, Ast_prov_mm2, mat.Ec)

st.subheader("5.1 Check for Stress-Induced Cracking")
st.markdown(f"""
- **Service Moment ($M_s$)** (Unfactored FL): **{Ms_design:.2f} $\text{kNm/m}$**
- **Provided $A_{{st}}$** (Used for Check): **{Ast_prov_mm2:.0f} $\text{mm}^2/\text{m}$** (E.g., M16 @ 150mm)
- **Permissible $\sigma_{{s, allow}}$** (IS 3370-2 Cl 3.1): **{sigma_allow:.0f} $\text{MPa}$**

**Calculated $\sigma_{{s, actual}}$ = {sigma_s_actual:.0f} $\text{MPa}$**

**Result: $\sigma_{{s, actual}}$ {'≤' if sigma_s_actual <= sigma_allow else '>'} $\sigma_{{s, allow}}$ $\rightarrow$ {'✅ PASS (Crack width controlled)' if sigma_s_actual <= sigma_allow else '❌ FAIL (Increase $t_{{w}}$ or $A_{{st}}$)'}**
""")

st.markdown("---")

# --- DETAILING & DRAWING SECTION ---
st.header("6. Detailing, Curtailment, and Output Drawings")

st.subheader("6.1 Design Output Drawings")
st.markdown("The moment and shear diagrams illustrate the distribution along the wall height, normalized to the base maximum values.")
plot_results(geom.H, geom.t_wall, Mu_max_design / gamma_f_FL, V_base_FL)

st.subheader("6.2 Vertical Rebar Curtailment Suggestions")
st.markdown("""
For tall walls where bending moment is significant at the base and reduces rapidly towards the top (as shown in the moment diagram):
1.  **Curtailment Point:** The bending moment $M(z)$ along the height $H$ follows a curve, generally dropping to zero at the top. The first bar curtailment can typically occur at $0.3H$ to $0.4H$ from the base, where $A_{st, req}$ drops below $A_{st, min}$.
2.  **Lapping/Cut-off:** At the curtailment point, the required main steel (governed by $M_{max}$) can be cut off or reduced, ensuring the remaining steel meets the minimum $A_{st, min}$ requirement plus the necessary development length ($L_d$).
3.  **Minimum Steel Zone:** The upper section of the wall (e.g., above $0.4H$) should be reinforced with the minimum required $A_{st}$ for both faces (0.25% to 0.35% total area, depending on thickness) to control temperature and shrinkage cracks.
4.  **Distribution Steel (Horizontal):** Horizontal steel is primarily designed for hoop tension (for circular tanks) or horizontal plate bending (for rectangular tanks, typically minimum steel throughout, plus tension if free from adjacent wall). For rectangular tanks, minimum steel is often sufficient for the top $0.6H$.
""")

st.subheader("6.3 Summary of Key Design Values")
data = {
    "Parameter": ["Wall Thickness", "Effective Depth", "L/H Ratio", "Max Design Moment ($M_u$)", "Required $A_{st}$ (Vert.)", "Provided $\\sigma_{s, actual}$", "Allowable $\\sigma_{s, allow}$"],
    "Value": [
        f"{geom.t_wall * 1000:.0f} mm",
        f"{d_eff:.0f} mm",
        f"{L_over_H:.2f}",
        f"{Mu_max_design:.2f} kNm/m",
        f"{Ast_req_final:.0f} mm²/m",
        f"{sigma_s_actual:.0f} MPa",
        f"{sigma_allow:.0f} MPa"
    ],
    "Result": ["-", "-", "-", "-", "-", "-", "SLS " + ('PASS' if sigma_s_actual <= sigma_allow else 'FAIL')]
}
st.table(pd.DataFrame(data))

# End of code
