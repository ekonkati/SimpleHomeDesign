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
# ==================================
KNM_TO_NMM = 1_000_000.0
M_TO_MM = 1000.0

# Set Streamlit page configuration for wide layout
st.set_page_config(layout="wide")

# Enable LaTeX rendering for Matplotlib texts (using cm font for math)
plt.rcParams['text.usetex'] = False 
plt.rcParams['mathtext.fontset'] = 'cm'

# ===============================
# Data Classes (Default values added)
# ===============================
@dataclass
class Materials:
    fck: float = 30.0
    fy: float = 415.0
    gamma_conc: float = 25.0
    Ec: float = 30000.0
    exposure: str = "Severe"

@dataclass
class Geometry:
    H: float = 4.0      # m (Water height)
    L: float = 6.0      # m (Long wall length)
    B: float = 4.0      # m (Short wall length)
    t_wall: float = 0.3 # m (Wall thickness)
    t_base: float = 0.3 # m (Base slab thickness)
    freeboard: float = 0.15 # m
    tank_type: str = "Ground"

@dataclass
class Loads:
    gamma_w: float = 10.0 # kN/m³
    gamma_s: float = 18.0 # kN/m³
    K0: float = 0.5     
    phi: float = 30.0   
    mu_base: float = 0.5 
    z_g_zone: int = 3   

# ===============================
# Engineering Helper Functions
# ===============================

def triangular_resultant(gamma: float, H: float) -> Tuple[float, float]:
    """Calculates resultant force (R, kN/m) and its location (zbar, m from base)."""
    R = 0.5 * gamma * H**2
    zbar = H/3.0
    return R, zbar

def demand_ast_from_M(Mu_kNm: float, d_eff_mm: float, fy_MPa: float, fck_MPa: float) -> float:
    """Calculates required Ast (mm²/m) using the IS 456:2000 (Cl E-1.1) ULS method."""
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
# Plotting Functions (FIXED Mathtext Errors)
# ===============================

def plot_loads(geom: Geometry, loads: Loads, R_liq: float, R_soil: float):
    """Plots the load diagram (Hydrostatic and Earth Pressure) with enhanced scaling and formatting."""
    
    # ----------------------------------------------------
    # 1. Setup and Dimensions
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6)) 
    
    H_wall = geom.H
    t_base = geom.t_base
    H_total = H_wall + t_base
    
    # Pressures
    P_max_w = loads.gamma_w * H_wall
    P_max_s = 0
    
    # Format numbers outside of the Mathtext string
    P_max_w_str = f'{P_max_w:.1f}'
    R_liq_str = f'{R_liq:.1f}'
    R_soil_str = f'{R_soil:.1f}'


    # ----------------------------------------------------
    # 2. Draw the Wall and Base Geometry (Profile View)
    # ----------------------------------------------------
    
    # Draw Wall Profile
    ax.plot([0, 0], [0, H_total], color='k', linewidth=3, label='Wall')
    
    # Draw Base Slab Profile (extending slightly left for soil load illustration)
    ax.plot([-0.5, 1], [0, 0], color='k', linewidth=3, label='Base Slab')
    ax.plot([-0.5, 0], [t_base, t_base], color='k', linestyle=':', linewidth=1) # Top of base line
    
    # Dimensioning the Wall Height H
    ax.arrow(-0.3, H_total, 0, -H_wall, head_width=0.05, head_length=0.1, fc='gray', ec='gray', length_includes_head=True)
    ax.text(-0.35, t_base + H_wall / 2, r'$H$', color='gray', ha='right')
    
    # Dimensioning the Base Slab Thickness t_base
    ax.arrow(-0.3, 0.05, 0, t_base - 0.1, head_width=0.05, head_length=0.1, fc='gray', ec='gray', length_includes_head=True)
    ax.text(-0.35, t_base / 2, r'$t_{base}$', color='gray', ha='right')

    # ----------------------------------------------------
    # 3. Hydrostatic Pressure (P_w) - Acting on inner face (right)
    # ----------------------------------------------------
    x_w = [0, P_max_w, 0]
    y_w = [t_base, t_base, H_total]
    ax.fill_betweenx([t_base, H_total], [0, P_max_w], 0, color='b', alpha=0.3, label='Water Pressure')
    ax.plot(x_w, y_w, color='b', linestyle='--') 
    
    # Pressure magnitude label (Pmax) - FIXED: Simplified Mathtext syntax
    ax.text(P_max_w * 1.05, t_base + 0.1, 
            r'$P_{\mathrm{w, max}} = ' + P_max_w_str + r'\ \mathrm{kN/m^2}$', 
            color='b', fontsize=10)
    
    # Resultant force label (R_w) - FIXED: Simplified Mathtext syntax
    ax.arrow(P_max_w * 0.5, t_base + H_wall/3, -0.05, 0, head_width=0.1, head_length=0.1, fc='b', ec='b')
    ax.text(P_max_w * 0.5, t_base + H_wall/3 + 0.2, 
            r'$R_w = ' + R_liq_str + r'\ \mathrm{kN/m}$', 
            color='b', ha='center', fontsize=10)

    # ----------------------------------------------------
    # 4. Earth Pressure (P_soil) - Acting on outer face (left)
    # ----------------------------------------------------
    if geom.tank_type == "Ground":
        P_max_s = loads.gamma_s * loads.K0 * H_total
        P_max_s_str = f'{P_max_s:.1f}'
        
        x_s = [-P_max_s, 0, 0]
        y_s = [0, 0, H_total]
        
        ax.fill_betweenx([0, H_total], [-P_max_s, 0], 0, color='brown', alpha=0.3, label='Earth Pressure')
        ax.plot(x_s, y_s, color='brown', linestyle='--')
        
        # Pressure magnitude label (Pmax) - FIXED: Simplified Mathtext syntax
        ax.text(-P_max_s * 1.05, 0.1, 
                r'$P_{\mathrm{s, max}} = ' + P_max_s_str + r'\ \mathrm{kN/m^2}$', 
                color='brown', ha='right', fontsize=10)
        
        # Resultant force label (R_s) - FIXED: Simplified Mathtext syntax
        ax.arrow(-P_max_s * 0.5, H_total/3, 0.05, 0, head_width=0.1, head_length=0.1, fc='brown', ec='brown')
        ax.text(-P_max_s * 0.5, H_total/3 + 0.2, 
                r'$R_s = ' + R_soil_str + r'\ \mathrm{kN/m}$', 
                color='brown', ha='center', fontsize=10)

    # ----------------------------------------------------
    # 5. Formatting and Display
    # ----------------------------------------------------
    ax.set_title(
        f"Load Diagram (Wall Profile: L={geom.L:.1f}m, B={geom.B:.1f}m)",
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel("Pressure (Scaled $\leftarrow$ Soil | Water $\rightarrow$)", fontsize=10)
    ax.set_ylabel("Height from Base (m)", fontsize=10)
    
    # Set limits based on pressures and dimensions
    x_limit = max(abs(P_max_s), abs(P_max_w)) * 1.5 
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-0.2, H_total + 0.5)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
    
    # Hide the y-axis ticks and spines to make the wall itself the central reference
    ax.yaxis.tick_left()
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Set y-tick marks at H and t_base for clarity
    ax.set_yticks(sorted(list(set([0, t_base, H_total]))))

    st.pyplot(fig)


def plot_results(H: float, M_base_L: float, M_base_B: float, V_base_max: float):
    """Plots the Bending Moment and Shear Force Diagrams."""
    
    # Format numbers outside of the Mathtext string
    V_max_str = f'{V_base_max:.1f}'
    M_L_str = f'{M_base_L:.1f}'
    M_B_str = f'{M_base_B:.1f}'

    # Shear Force (Common for both walls)
    fig_v, ax_v = plt.subplots(figsize=(3.5, 6))
    ax_v.plot([0, V_base_max, 0], [0, 0, H], 'r-', linewidth=2)
    ax_v.fill([0, V_base_max, 0], [0, 0, H], 'r', alpha=0.2)
    ax_v.plot([0, 0], [0, H], 'k--')
    # FIXED: Simplified Mathtext syntax
    ax_v.text(V_base_max * 1.1, 0.05, 
              r'$V_{\max} = ' + V_max_str + r'\ \mathrm{kN/m}$', 
              color='r', fontsize=10)
    ax_v.set_title("Shear Force ($V$)", fontsize=12)
    ax_v.set_xlabel("Shear (kN/m)", fontsize=10)
    ax_v.set_ylabel("Height (m)", fontsize=10)
    ax_v.set_ylim(-0.1, H + 0.1)
    ax_v.grid(True, linestyle='--', alpha=0.6)

    # Bending Moment - Long Wall (L)
    fig_m_L, ax_m_L = plt.subplots(figsize=(3.5, 6))
    x_m_L = [0, M_base_L, 0]
    y_m_L = [0, 0, H]
    ax_m_L.plot(x_m_L, y_m_L, 'b-', linewidth=2)
    ax_m_L.fill(x_m_L, y_m_L, 'b', alpha=0.2)
    ax_m_L.plot([0, 0], [0, H], 'k--')
    # FIXED: Simplified Mathtext syntax
    ax_m_L.text(M_base_L * 1.1, 0.05, 
                r'$M_{L} = ' + M_L_str + r'\ \mathrm{kNm/m}$', 
                color='b', fontsize=10)
    ax_m_L.set_title("Moment - Long Wall ($M_L$)", fontsize=12)
    ax_m_L.set_xlabel("Moment (kNm/m)", fontsize=10)
    ax_m_L.set_ylabel("Height (m)", fontsize=10)
    ax_m_L.set_ylim(-0.1, H + 0.1)
    ax_m_L.grid(True, linestyle='--', alpha=0.6)
    
    # Bending Moment - Short Wall (B)
    fig_m_B, ax_m_B = plt.subplots(figsize=(3.5, 6))
    x_m_B = [0, M_base_B, 0]
    y_m_B = [0, 0, H]
    ax_m_B.plot(x_m_B, y_m_B, 'g-', linewidth=2)
    ax_m_B.fill(x_m_B, y_m_B, 'g', alpha=0.2)
    ax_m_B.plot([0, 0], [0, H], 'k--')
    # FIXED: Simplified Mathtext syntax
    ax_m_B.text(M_base_B * 1.1, 0.05, 
                r'$M_{B} = ' + M_B_str + r'\ \mathrm{kNm/m}$', 
                color='g', fontsize=10)
    ax_m_B.set_title("Moment - Short Wall ($M_B$)", fontsize=12)
    ax_m_B.set_xlabel("Moment (kNm/m)", fontsize=10)
    ax_m_B.set_ylabel("Height (m)", fontsize=10)
    ax_m_B.set_ylim(-0.1, H + 0.1)
    ax_m_B.grid(True, linestyle='--', alpha=0.6)


    col_v, col_m_l, col_m_b = st.columns(3)
    with col_v:
        st.pyplot(fig_v)
    with col_m_l:
        st.pyplot(fig_m_L)
    with col_m_b:
        st.pyplot(fig_m_B)

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
Ast_min_perc = 0.35 if geom.t_wall >= 0.20 else 0.25 # IS 3370 Cl 7.1
A_conc_total = 1000.0 * tw_mm 
Ast_min_face = (Ast_min_perc / 100.0) * A_conc_total / 2.0

st.markdown("---")

# --- LOAD CALCULATION SECTION ---
st.header("2. Basic Load Calculations and Sketches")
st.markdown(f"The aspect ratio $\\frac{{L}}{{H}} = \\frac{{{geom.L}}}{{{geom.H}}} = **{L_over_H:.2f}**$ is used to determine plate bending coefficients (IS 3370-4).")

# Hydrostatic Load (FL)
R_liq, zbar_liq = triangular_resultant(loads.gamma_w, geom.H)
P_max_w = loads.gamma_w * geom.H

# Earth Pressure (EL)
R_soil, M_soil_base = 0.0, 0.0
if geom.tank_type == "Ground":
    P_max_s = loads.gamma_s * loads.K0 * (geom.H + geom.t_base)
    R_soil, zbar_soil = triangular_resultant(loads.gamma_s * loads.K0, geom.H + geom.t_base)
    M_soil_base = R_soil * zbar_soil

st.subheader("2.1 Load Parameters")
st.markdown(f"**Hydrostatic Pressure:** $P_{{max, w}} = **{P_max_w:.2f} \text{{ kN/m}}^2**$, Resultant $R_{{w}} = **{R_liq:.2f} \text{{ kN/m}}$**")
if geom.tank_type == "Ground":
    st.markdown(f"**Earth Pressure:** $P_{{max, s}} = **{P_max_s:.2f} \text{{ kN/m}}^2**$, Resultant $R_{{s}} = **{R_soil:.2f} \text{{ kN/m}}$**")

st.subheader("2.2 Load Sketch")
plot_loads(geom, loads, R_liq, R_soil)

st.markdown("---")

# --- MOMENT AND SHEAR CALCULATION SECTION ---
st.header("3. Wall Moment Calculation (IS 3370-4 Plate Action)")

# Moment Coefficients
C_corner = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Corner (Max)')
C_mid_L = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Mid_Long')
C_mid_B = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Mid_Short')

M_base_corner_FL = C_corner * loads.gamma_w * geom.H**3
M_base_mid_L_FL = C_mid_L * loads.gamma_w * geom.H**3
M_base_mid_B_FL = C_mid_B * loads.gamma_w * geom.H**3

# Max base shear (V_max) is conservatively assumed from the resultant force R_liq
V_base_FL = R_liq

st.markdown("### 3.1 Unfactored Base Moments (FL) and Coefficients")
st.markdown(f"Coefficients (C): $C_{{corner}}={C_corner:.4f}$, $C_{{mid, L}}={C_mid_L:.4f}$, $C_{{mid, B}}={C_mid_B:.4f}$")

col_moment_l, col_moment_b = st.columns(2)
with col_moment_l:
    st.markdown(f"**Long Wall ($L$) Mid-span Moment:** $M_{{L, mid}} = **{M_base_mid_L_FL:.2f} \text{{ kNm/m}}$**")
with col_moment_b:
    st.markdown(f"**Short Wall ($B$) Mid-span Moment:** $M_{{B, mid}} = **{M_base_mid_B_FL:.2f} \text{{ kNm/m}}$**")

st.markdown("---")

# --- DESIGN & RESULTS SECTION (SEPARATED) ---
st.header("4. Vertical Reinforcement Design")

gamma_f = 1.5 
V_max_design = gamma_f * V_base_FL
Ms_corner_FL = M_base_corner_FL # Corner moment governs SLS for both walls (for hoop tension zone)

# --- LONG WALL DESIGN ---
col_L, col_B = st.columns(2)

with col_L:
    st.subheader("4.1 Long Wall Vertical Design")
    
    # ULS Design Moment (Corner governs)
    Mu_design_FL = gamma_f * M_base_corner_FL 
    Mu_max_design = max(Mu_design_FL, gamma_f * M_soil_base) if geom.tank_type == "Ground" else Mu_design_FL
    
    Ast_req_ULS = demand_ast_from_M(Mu_max_design, d_eff, mat.fy, mat.fck)
    Ast_req_final = max(Ast_req_ULS, Ast_min_face)
    
    # SLS Check (using max Ast required)
    sigma_allow = 130.0 if Ast_req_final <= 1340 else 100.0 # Simplified check for <= 16mm bars
    sigma_s_actual = steel_stress_sls(Ms_corner_FL, d_eff, Ast_req_final, mat.Ec)

    st.markdown(f"**Governing ULS Moment ($M_u$):** **{Mu_max_design:.2f} kNm/m** (at corner/base)")
    st.markdown(f"""
    - Required $A_{{st}}$ from ULS Flexure: **{Ast_req_ULS:.0f} $\text{{mm}}^2/\text{{m}}$**
    - Minimum $A_{{st}}$ (IS 3370): **{Ast_min_face:.0f} $\text{{mm}}^2/\text{{m}}$**
    - **Governing $A_{{st, req}}$ (Vertical):** **{Ast_req_final:.0f} $\text{{mm}}^2/\text{{m}}$** (per face)
    """)

    st.markdown("#### SLS Check")
    st.markdown(f"**Calculated $\sigma_{{s, actual}}$** = **{sigma_s_actual:.0f} $\text{{MPa}}$**")
    st.markdown(f"**Allowable $\sigma_{{s, allow}}$** = **{sigma_allow:.0f} $\text{{MPa}}$**")
    st.markdown(f"**Result:** **{'✅ PASS' if sigma_s_actual <= sigma_allow else '❌ FAIL'}** (Crack width controlled)")

with col_B:
    st.subheader("4.2 Short Wall Vertical Design")
    
    # Use M_base_mid_B_FL for the short wall's central portion design (as it often governs local bending)
    Mu_design_FL_B = gamma_f * M_base_mid_B_FL
    # Conservatively use the overall max Mu (corner) for ULS check if corner moment is higher, or local moment
    Mu_max_design_B = max(Mu_design_FL_B, Mu_max_design) 

    Ast_req_ULS_B = demand_ast_from_M(Mu_max_design_B, d_eff, mat.fy, mat.fck)
    Ast_req_final_B = max(Ast_req_ULS_B, Ast_min_face)
    
    # SLS Check
    sigma_allow_B = 130.0 if Ast_req_final_B <= 1340 else 100.0 
    sigma_s_actual_B = steel_stress_sls(M_base_mid_B_FL, d_eff, Ast_req_final_B, mat.Ec)

    st.markdown(f"**Governing ULS Moment ($M_u$):** **{Mu_max_design_B:.2f} kNm/m** (at corner/base)")
    st.markdown(f"""
    - Required $A_{{st}}$ from ULS Flexure: **{Ast_req_ULS_B:.0f} $\text{{mm}}^2/\text{{m}}$**
    - Minimum $A_{{st}}$ (IS 3370): **{Ast_min_face:.0f} $\text{{mm}}^2/\text{{m}}$**
    - **Governing $A_{{st, req}}$ (Vertical):** **{Ast_req_final_B:.0f} $\text{{mm}}^2/\text{{m}}$** (per face)
    """)
    
    st.markdown("#### SLS Check")
    st.markdown(f"**Calculated $\sigma_{{s, actual}}$** = **{sigma_s_actual_B:.0f} $\text{{MPa}}$**")
    st.markdown(f"**Allowable $\sigma_{{s, allow}}$** = **{sigma_allow_B:.0f} $\text{{MPa}}$**")
    st.markdown(f"**Result:** **{'✅ PASS' if sigma_s_actual_B <= sigma_allow_B else '❌ FAIL'}** (Crack width controlled)")


st.markdown("---")

# --- DETAILING & DRAWING SECTION ---
st.header("5. Output Diagrams and Detailing Suggestions")

st.subheader("5.1 Design Output Diagrams")
st.markdown("These diagrams show the Shear Force (common) and Bending Moment distributions for the Long and Short Walls, based on unfactored moments ($M_{FL}$).")
# Pass unfactored moments for plotting shape representation
plot_results(geom.H, M_base_mid_L_FL, M_base_mid_B_FL, V_base_FL)

st.subheader("5.2 Rebar Reduction and Curtailment Suggestions")
st.markdown(f"""
The vertical reinforcement must meet the **Maximum Required $A_{{st}}$** (usually at the base/corner) and be reduced towards the top where moment approaches zero, always respecting $A_{{st, min}}$.

1.  **Curtailment Point:** For the main vertical bars, consider reducing **half of the bars** at approximately **$0.4 \times {geom.H} \approx {0.4*geom.H:.2f} \text{{ m}}$** from the base, ensuring $L_d$ beyond the theoretical cut-off point.

2.  **Upper Zone Steel:** The top zone ($0.6H$ to $H$) requires the minimum steel **$A_{{st, min, face}} = {Ast_min_face:.0f} \text{{ mm}}^2/\text{{m}}$** for both vertical and horizontal reinforcement.

3.  **Horizontal Steel:** Horizontal reinforcement must resist hoop tension and control shrinkage/temperature cracks. It should be maintained at $A_{{st, min}}$ or greater across the entire wall.
""")
