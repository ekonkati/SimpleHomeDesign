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
# Data Classes (Default values added for robust initialization)
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
# Engineering Helper Functions
# ===============================

def triangular_resultant(gamma: float, H: float) -> Tuple[float, float]:
    """Calculates resultant force (R, kN/m) and its location (zbar, m from base)."""
    R = 0.5 * gamma * H**2
    zbar = H/3.0
    return R, zbar

def demand_ast_from_M(Mu_kNm: float, d_eff_mm: float, fy_MPa: float, fck_MPa: float) -> float:
    """Calculates required Ast (mm²/m) using the IS 456:2000 (Cl E-1.1) ULS limit state method."""
    if Mu_kNm <= 0.0:
        return 0.0
    
    Mu_Nmm = Mu_kNm * KNM_TO_NMM
    b = 1000.0  # mm
    
    try:
        term_in_sqrt = 1.0 - (4.6 * Mu_Nmm) / (fck_MPa * b * d_eff_mm**2)
        
        if term_in_sqrt < 0:
            return 99999.0 
            
        Ast = (0.5 * fck_MPa / fy_MPa) * (1.0 - math.sqrt(term_in_sqrt)) * b * d_eff_mm
        
        return max(Ast, 0.0)
    except:
        return 0.0

def steel_stress_sls(Ms_kNm_per_m: float, d_eff_mm: float, As_mm2_per_m: float, Ec_MPa: float) -> float:
    """Calculates steel stress (sigma_s in MPa) using Elastic Cracked Section Theory (SLS)."""
    if Ms_kNm_per_m <= 0.0 or As_mm2_per_m <= 0.0:
        return 0.0
        
    Es = 200000.0  # MPa
    m = Es / Ec_MPa
    b = 1000.0     # 1m strip
    Ms_Nmm = Ms_kNm_per_m * KNM_TO_NMM

    ratio = (m * As_mm2_per_m) / b
    
    try:
        n = -ratio + math.sqrt(ratio**2 + 2 * ratio * d_eff_mm)
    except ValueError:
        return float('inf') 

    z = d_eff_mm - n/3.0
    sigma_s = Ms_Nmm / (As_mm2_per_m * max(z, 1.0))
    
    return sigma_s

# ===============================
# Interpolation Tables (from IS 3370-4)
# ===============================

# Moment coefficients 'C' for Wall Bending Moment at Base (Table 4, Case 1, L/H > 2)
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
        
    lower_idx = idx[idx <= ratio].max()
    upper_idx = idx[idx > ratio].min()
    
    v1 = val[lower_idx]
    v2 = val[upper_idx]
    
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
    mat = Materials(**data.get("Materials", {}))
    geom = Geometry(**data.get("Geometry", {}))
    loads = Loads(**data.get("Loads", {}))
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
    # CORRECTED: Use keyword arguments for clarity and to avoid the Matplotlib format error.
    ax.plot(x_w, y_w, color='b', linestyle='--') 
    ax.text(P_max_w * 1.1, geom.t_base + geom.H/2, f'$P_w$={P_max_w:.1f} kN/m²', color='b')
    ax.arrow(P_max_w * 0.5, geom.t_base + H_total/3, 0, -0.2, head_width=0.05, head_length=0.1, fc='b', ec='b')
    ax.text(P_max_w * 0.5 + 0.1, geom.t_base + H_total/3 + 0.1, f'$R_w$={R_liq:.1f} kN/m', color='b')

    # 3. Earth Pressure (P_soil) - Only for Ground Tank
    P_max_s = 0
    if geom.tank_type == "Ground":
        P_max_s = loads.gamma_s * loads.K0 * (geom.H + geom.t_base)
        x_s = [-P_max_s, 0, 0]
        y_s = [0, 0, H_total]
        ax.fill(x_s, y_s, 'brown', alpha=0.3, label='Earth Pressure')
        # CORRECTED: Using explicit color and linestyle keywords fixes the ValueError
        ax.plot(x_s, y_s, color='brown', linestyle='--')
        ax.text(-P_max_s * 1.5, H_total/2, f'$P_s$={P_max_s:.1f} kN/m²', color='brown')
        ax.arrow(-P_max_s * 0.5, H_total/3, 0, -0.2, head_width=0.05, head_length=0.1, fc='brown', ec='brown')
        ax.text(-P_max_s * 0.5 - 0.5, H_total/3 + 0.1, f'$R_s$={R_soil:.1f} kN/m', color='brown')

    # Formatting
    ax.set_title("Input Load Sketches (Hydrostatic & Earth Pressure)")
    ax.set_xlabel("Pressure (Scaled)")
    ax.set_ylabel("Height (m)")
    ax.set_ylim(-0.2, H_total + 0.5)
    ax.set_xlim(-max(abs(P_max_s)*2, 1), max(abs(P_max_w)*2, 1))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    ax.set_aspect('equal', adjustable='box')
    st.pyplot(fig)


def plot_results(H: float, M_base_max: float, V_base_max: float):
    """Plots the bending moment and shear force diagrams for the wall."""
    
    # Shear Force 
    fig_v, ax_v = plt.subplots(figsize=(4, 6))
    ax_v.plot([0, V_base_max, 0], [0, 0, H], 'r-', linewidth=2)
    ax_v.fill([0, V_base_max, 0], [0, 0, H], 'r', alpha=0.2)
    ax_v.plot([0, 0], [0, H], 'k--')
    ax_v.text(V_base_max * 1.1, 0.05, f'$V_{{max}}$={V_base_max:.1f} kN/m', color='r')
    ax_v.set_title("Shear Force Diagram ($V$)")
    ax_v.set_xlabel("Shear (kN/m)")
    ax_v.set_ylabel("Height (m)")
    ax_v.set_ylim(-0.1, H + 0.1)
    ax_v.grid(True, linestyle='--', alpha=0.6)

    # Bending Moment
    fig_m, ax_m = plt.subplots(figsize=(4, 6))
    x_m = [0, M_base_max, 0]
    y_m = [0, 0, H]
    ax_m.plot(x_m, y_m, 'b-', linewidth=2)
    ax_m.fill(x_m, y_m, 'b', alpha=0.2)
    ax_m.plot([0, 0], [0, H], 'k--')
    ax_m.text(M_base_max * 1.1, 0.05, f'$M_{{max}}$={M_base_max:.1f} kNm/m', color='b')
    ax_m.set_title("Bending Moment Diagram ($M$)")
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

# --- INITIALIZATION ---
if 'mat' not in st.session_state:
    st.session_state.mat = Materials()
    st.session_state.geom = Geometry()
    st.session_state.loads = Loads()
    st.session_state.is_imported = False

mat, geom, loads = st.session_state.mat, st.session_state.geom, st.session_state.loads


# --- INPUT & I/O SECTION ---
st.header("1. Input Data & File Management")
st.markdown("Use the controls below to **Import** or **Export** the current design parameters (Materials, Geometry, Loads).")

col_io1, col_io2 = st.columns([1, 1])

# Export Button 
export_str = export_inputs(mat, geom, loads) 
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
        mat, geom, loads = st.session_state.mat, st.session_state.geom, st.session_state.loads
        st.success("Inputs imported successfully! Please review the inputs below.")
    except Exception as e:
        st.error(f"Error importing file: {e}")

# Column Layout for Inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Materials (IS 456)")
    mat.fck = st.number_input("Concrete Grade $f_{ck}$ (MPa)", 25.0, 50.0, mat.fck, 5.0, key="mat_fck")
    mat.fy = st.number_input("Steel Grade $f_{y}$ (MPa)", 250.0, 550.0, mat.fy, 50.0, key="mat_fy")
    mat.Ec = st.number_input("Concrete E ($E_{c}$) (MPa)", 25000.0, 40000.0, mat.Ec, 1000.0, format="%.0f", key="mat_ec")
    
with col2:
    st.subheader("Geometry (Wall)")
    geom.H = st.number_input("Height of Water $H$ (m)", 1.0, 10.0, geom.H, 0.1, key="geom_h")
    geom.L = st.number_input("Long Wall $L$ (m)", 1.0, 20.0, geom.L, 0.1, key="geom_l")
    geom.B = st.number_input("Short Wall $B$ (m)", 1.0, 20.0, geom.B, 0.1, key="geom_b")
    geom.t_wall = st.number_input("Wall Thickness $t_{w}$ (m)", 0.15, 1.0, geom.t_wall, 0.05, key="geom_tw")
    
with col3:
    st.subheader("Loads & Environment")
    geom.tank_type = st.selectbox("Tank Type", ["Ground", "Elevated"], index=0 if geom.tank_type == "Ground" else 1, key="geom_type")
    loads.gamma_w = st.number_input("Water Density $\gamma_{w}$ (kN/m³)", 9.81, 15.0, loads.gamma_w, 0.1, key="load_gw")
    if geom.tank_type == "Ground":
        loads.gamma_s = st.number_input("Soil Density $\gamma_{s}$ (kN/m³)", 15.0, 25.0, loads.gamma_s, 0.5, key="load_gs")
        loads.K0 = st.number_input("Earth Pressure Coeff. $K_{0}$", 0.3, 1.0, loads.K0, 0.05, key="load_k0")
    loads.z_g_zone = st.selectbox("Seismic Zone (IS 1893-2)", [2, 3, 4, 5], index=loads.z_g_zone-2 if loads.z_g_zone in [2,3,4,5] else 1, key="load_zone")

# Derived geometric properties
L_over_H = geom.L / geom.H if geom.H > 0 else 99.0
tw_mm = geom.t_wall * M_TO_MM
d_eff = tw_mm - 50.0 

st.markdown("---")

# --- LOAD CALCULATION SECTION ---
st.header("2. Basic Load Calculations and Sketches")
st.markdown(f"The analysis is based on a **fixed base wall** acting as a plate element. The aspect ratio $\\frac{{L}}{{H}} = \\frac{{{geom.L}}}{{{geom.H}}} = **{L_over_H:.2f}**$ is used to determine plate bending coefficients (IS 3370-4).")

# 2.1 Hydrostatic Load (FL)
st.subheader("2.1 Hydrostatic Load (Full Tank)")
R_liq, zbar_liq = triangular_resultant(loads.gamma_w, geom.H)
P_max_w = loads.gamma_w * geom.H
st.markdown(f"""
The hydrostatic pressure is triangular, with maximum pressure $P_{{max, w}}$ at the base (height $H$):
$$P_{{max, w}} = \gamma_w \cdot H = {loads.gamma_w:.2f} \cdot {geom.H:.2f} = **{P_max_w:.2f} \text{{ kN/m}}^2$$
The total resultant force $R_{{w}}$ per metre run of the wall is:
$$R_{{w}} = 0.5 \cdot P_{{max, w}} \cdot H = **{R_liq:.2f} \text{{ kN/m}}$$
""")

# 2.2 Earth Pressure (EL) - Only for Ground Tanks
R_soil, M_soil_base = 0.0, 0.0
if geom.tank_type == "Ground":
    st.subheader("2.2 Earth Pressure Load (Empty Tank)")
    P_max_s = loads.gamma_s * loads.K0 * (geom.H + geom.t_base)
    R_soil, zbar_soil = triangular_resultant(loads.gamma_s * loads.K0, geom.H + geom.t_base)
    M_soil_base = R_soil * zbar_soil
    st.markdown(f"""
    For the Ground Tank, the soil exerts a lateral force defined by the earth pressure at rest coefficient $K_0$:
    $$P_{{max, s}} = \gamma_s \cdot K_0 \cdot (H + t_{{base}}) = {loads.gamma_s:.1f} \cdot {loads.K0:.2f} \cdot ({geom.H:.2f} + {geom.t_base:.2f}) = **{P_max_s:.2f} \text{{ kN/m}}^2$$
    The total resultant force $R_{{s}}$ per metre run (acting over $H+t_{{base}}$) is:
    $$R_{{s}} = 0.5 \cdot P_{{max, s}} \cdot (H + t_{{base}}) = **{R_soil:.2f} \text{{ kN/m}}$$
    """)

# 2.3 Load Sketch
st.subheader("2.3 Load Sketch")
st.markdown("Schematic representation of the lateral pressures acting on the wall.")
plot_loads(geom, loads, R_liq, R_soil)

st.markdown("---")

# --- MOMENT AND SHEAR CALCULATION SECTION ---
st.header("3. Wall Moment Calculation (IS 3370-4 Plate Action)")
st.markdown(f"""
Since the wall is rigidly connected to the base, it acts as a plate. The maximum vertical bending moment at the base, $M_{{base}}$, is found using coefficients ($C$) from **IS 3370-4 (Table 4, Case 1)**.
$$M_{{base}} = C \cdot \gamma_w \cdot H^3$$
""")

# Moment Coefficients
C_corner = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Corner (Max)')
C_mid_L = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Mid_Long')
C_mid_B = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Mid_Short')

M_base_corner_FL = C_corner * loads.gamma_w * geom.H**3
M_base_mid_L_FL = C_mid_L * loads.gamma_w * geom.H**3
M_base_mid_B_FL = C_mid_B * loads.gamma_w * geom.H**3

# Max base shear (V_max) is conservatively assumed from the resultant force R_liq
V_base_FL = R_liq

st.markdown("### 3.1 Unfactored Base Moments for Full Tank (FL)")
st.markdown(f"**Calculated Aspect Ratio $L/H = {L_over_H:.2f}$**")
st.markdown(f"Interpolated Coefficients (C): $C_{{corner}}={C_corner:.4f}$, $C_{{mid, L}}={C_mid_L:.4f}$, $C_{{mid, B}}={C_mid_B:.4f}$")

st.markdown(f"""
- **Maximum Corner Moment:** $M_{{corner}} = **{M_base_corner_FL:.2f} \text{{ kNm/m}}$**
- **Mid-Long Wall Moment:** $M_{{mid, L}} = **{M_base_mid_L_FL:.2f} \text{{ kNm/m}}$**
- **Mid-Short Wall Moment:** $M_{{mid, B}} = **{M_base_mid_B_FL:.2f} \text{{ kNm/m}}$**
""")

st.markdown("---")

# --- LOAD COMBINATION & DESIGN SECTION ---
st.header("4. Ultimate Limit State (ULS) Design and $A_{st}$ Calculation")
st.markdown("Design checks are performed for **strength (ULS)** and **serviceability (SLS)**. The maximum vertical moment at the wall base ($M_{{max}}$) governs the reinforcement.")

# Load Factors (Partial Safety Factors, IS 456/IS 3370)
gamma_f_FL = 1.5  # Full Load (Water) for ULS
gamma_f_EL = 1.5  # Empty Load (Soil) for ULS

# Case 1: Full Tank (FL)
Mu_design_FL = gamma_f_FL * M_base_corner_FL 

# Case 2: Empty Tank (EL) + Soil
Mu_design_EL = 0.0
if geom.tank_type == "Ground":
    Mu_design_EL = gamma_f_EL * M_soil_base

# Final ULS Design values
Mu_max_design = max(Mu_design_FL, Mu_design_EL) if geom.tank_type == "Ground" else Mu_design_FL
V_max_design = gamma_f_FL * V_base_FL

st.subheader(f"4.1 ULS Design Moment and Shear")
st.markdown(f"The maximum ULS Design Moment ($M_u$) is $\\approx **{Mu_max_design:.2f} \text{{ kNm/m}}$** (Governed by {('Empty/Soil' if Mu_design_EL > Mu_design_FL else 'Full/Water')} Condition).")

# 4.2 Required Steel Area (ULS Check)
Ast_req_ULS = demand_ast_from_M(Mu_max_design, d_eff, mat.fy, mat.fck)
Ast_min_perc = 0.35 
if geom.t_wall < 0.20:
    Ast_min_perc = 0.25

# Minimum Ast check (IS 3370 Cl 7.1)
A_conc_total = 1000.0 * tw_mm # mm²/m
Ast_min_total = (Ast_min_perc / 100.0) * A_conc_total 
Ast_min_face = Ast_min_total / 2.0 

Ast_req_final = max(Ast_req_ULS, Ast_min_face)

st.subheader("4.2 Vertical Reinforcement Requirement")
st.markdown(f"""
- Required $A_{{st}}$ from ULS Flexure ($M_u$): **{Ast_req_ULS:.0f} $\text{{mm}}^2/\text{{m}}$**
- Minimum $A_{{st}}$ (IS 3370, {Ast_min_perc:.2f}% total): **{Ast_min_face:.0f} $\text{{mm}}^2/\text{{m}}$** (per face)

**Governing $A_{{st, req}}$ (Vertical) = {Ast_req_final:.0f} $\text{{mm}}^2/\text{{m}}$**
""")

st.markdown("---")

# --- SERVICEABILITY LIMIT STATE (SLS) CHECK SECTION ---
st.header("5. Serviceability Limit State (SLS) - Crack Control")
st.markdown("Per **IS 3370-2 (Cl 3.1)**, the steel stress ($\sigma_s$) under unfactored service loads must be limited to prevent cracking, ensuring water-tightness. We use **Elastic Cracked Section Analysis** to find the actual stress.")

# Permissible Steel Stress (IS 3370-2, Table 3 - For w_lim=0.2mm, Fe415)
sigma_allow = 130.0 

Ast_prov_mm2 = Ast_req_final
if Ast_req_final > 1340: # M16 @ 150 c/c = 1340 mm2/m
    sigma_allow = 100.0 

# Service Moment (Unfactored)
Ms_design = M_base_corner_FL 

# Calculate Actual Steel Stress (sigma_s)
sigma_s_actual = steel_stress_sls(Ms_design, d_eff, Ast_prov_mm2, mat.Ec)

st.subheader("5.1 Stress Check for Water-Tightness")
st.markdown(f"""
- **Service Moment ($M_s$)** (Unfactored FL): **{Ms_design:.2f} $\text{{kNm/m}}$**
- **Provided $A_{{st}}$** (Used for Check): **{Ast_prov_mm2:.0f} $\text{{mm}}^2/\text{{m}}$**
- **Permissible $\sigma_{{s, allow}}$** (IS 3370-2): **{sigma_allow:.0f} $\text{{MPa}}$** **Calculated $\sigma_{{s, actual}}$ = {sigma_s_actual:.0f} $\text{{MPa}}$**

**Result: $\sigma_{{s, actual}}$ {'≤' if sigma_s_actual <= sigma_allow else '>'} $\sigma_{{s, allow}}$ $\rightarrow$ **{'**✅ PASS** (Crack width controlled)' if sigma_s_actual <= sigma_allow else '**❌ FAIL** (Increase $t_{{w}}$ or $A_{{st}}$)'}**
""")

st.markdown("---")

# --- DETAILING & DRAWING SECTION ---
st.header("6. Output Drawings and Detailing Suggestions")

st.subheader("6.1 Design Output Diagrams")
st.markdown("The following diagrams show the distribution of the design moments and shear forces along the wall height ($H$).")
plot_results(geom.H, Mu_max_design / gamma_f_FL, V_base_FL)

st.subheader("6.2 Rebar Reduction and Curtailment Suggestions")
st.markdown(f"""
For a wall height $H = {geom.H:.1f} \text{{ m}}$, the vertical bending moment is maximum at the base and zero at the top.

1.  **Curtailment Point:** The main tension reinforcement (designed for $M_u$) can be reduced or cut off where the moment $M(z)$ requires an $A_{{st}} < A_{{st, prov, upper}}$.
    * *Suggestion:* Consider curtailing **half of the main bars** at approximately **$0.4 \times {geom.H} \approx {0.4*geom.H:.2f} \text{{ m}}$** from the base, ensuring adequate development length ($L_d$).

2.  **Upper Zone Steel:** The top portion (e.g., above $0.6H$) should maintain the minimum steel **$A_{{st, min, face}} = {Ast_min_face:.0f} \text{{ mm}}^2/\text{{m}}$** (vertically and horizontally on both faces) to control temperature and shrinkage cracks.

3.  **Horizontal Steel:** Horizontal reinforcement should typically be held at **$A_{{st, min}}$ or greater** across the entire wall height and faces to resist plate action moments and shrinkage.
""")

st.markdown("---")
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
