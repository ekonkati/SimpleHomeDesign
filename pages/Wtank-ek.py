# -*- coding: utf-8 -*-
import math
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ===============================
# Global Constants
# ==================================
KNM_TO_NMM = 1_000_000.0
M_TO_MM = 1000.0
ES = 200000.0  # MPa
BAR_AREAS = {8: 50.3, 10: 78.5, 12: 113.1, 16: 201.1, 20: 314.2} # mm^2

# Set Streamlit page configuration for wide layout
st.set_page_config(layout="wide")

# ===============================
# CUSTOM CSS INJECTION FOR VISIBILITY
# ===============================
st.markdown("""
<style>
/* Target the input area of the selectbox to give it a visible background */
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #f0f2f6; /* Light gray background for visibility */
}
</style>
""", unsafe_allow_html=True)

# ===============================
# IS 456 Shear Constants 
# ===============================
# Simplified from Table 19 (for M25 and M30 interpolation)
TAU_C_TABLE = pd.DataFrame(
    data={
        'Pt': [0.15, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00],
        'M25': [0.29, 0.36, 0.49, 0.57, 0.64, 0.70, 0.74, 0.78, 0.81, 0.82, 0.82],
        'M30': [0.29, 0.37, 0.50, 0.59, 0.67, 0.73, 0.78, 0.82, 0.85, 0.88, 0.90],
    }
).set_index('Pt')

# IS 3370-4 Interpolation Tables 
M_COEF_TABLE = pd.DataFrame(
    data={
        'L/H': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        'Base_Corner (Max)': [0.038, 0.043, 0.047, 0.051, 0.054, 0.057, 0.062, 0.065, 0.067, 0.068, 0.069],
        'Base_Mid_Long': [0.031, 0.035, 0.039, 0.041, 0.043, 0.045, 0.048, 0.050, 0.051, 0.051, 0.052],
        'Base_Mid_Short': [0.036, 0.041, 0.045, 0.048, 0.051, 0.053, 0.057, 0.060, 0.062, 0.063, 0.064]
    }
).set_index('L/H')

T_COEF_TABLE = pd.DataFrame(
    data={
        'L/H': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        'Tension_Max': [0.0075, 0.0125, 0.0185, 0.0255, 0.033, 0.041, 0.059, 0.076, 0.091, 0.106, 0.120]
    }
).set_index('L/H')

M_H_COEF_TABLE = pd.DataFrame(
    data={
        'L/H': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        'Corner_Max': [0.0003, 0.0005, 0.0008, 0.0011, 0.0014, 0.0018, 0.0025, 0.0033, 0.0040, 0.0046, 0.0052],
        'Mid_Span_Max': [0.0002, 0.0003, 0.0005, 0.0007, 0.0009, 0.0011, 0.0015, 0.0019, 0.0023, 0.0026, 0.0028]
    }
).set_index('L/H')

# IS 456 Annex D Coefficients (Simplified for two-way slab: Four Edges Continuous)
TWO_WAY_COEF = {
    1.0: (0.033, 0.033), 
    1.5: (0.050, 0.024), 
    2.0: (0.058, 0.019),
}

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
    SBC: float = 150.0 # kN/m^2

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

def bilinear_interpolate(ratio: float, df: pd.DataFrame, col: str) -> float:
    """Interpolates between values in a dataframe index."""
    idx = df.index
    val = df[col]
    
    if ratio <= idx.min(): return val.iloc[0]
    if ratio >= idx.max(): return val.iloc[-1]
        
    lower_idx = idx[idx <= ratio].max()
    upper_idx = idx[idx > ratio].min()
    
    v1 = val[lower_idx]
    v2 = val[upper_idx]
    
    C = v1 + (v2 - v1) * (ratio - lower_idx) / (upper_idx - lower_idx)
    return C

def demand_ast_from_M(Mu_kNm: float, d_eff_mm: float, fy_MPa: float, fck_MPa: float) -> float:
    """Calculates required Ast (mm²/m) using the IS 456:2000 (Cl E-1.1) ULS method."""
    if Mu_kNm <= 0.0: return 0.0
    
    Mu_Nmm = Mu_kNm * KNM_TO_NMM
    b = 1000.0  # mm
    
    try:
        # Check for section failure (Mu_lim for Fe 415 is 0.138 fck b d^2)
        Mu_lim = 0.138 * fck_MPa * b * d_eff_mm**2 
        if Mu_Nmm > Mu_lim: 
            return 99999.0 

        term_in_sqrt = 1.0 - (4.6 * Mu_Nmm) / (fck_MPa * b * d_eff_mm**2)
        if term_in_sqrt < 0: return 99999.0 
        Ast = (0.5 * fck_MPa / fy_MPa) * (1.0 - math.sqrt(term_in_sqrt)) * b * d_eff_mm
        return max(Ast, 0.0)
    except:
        return 0.0

def calc_Ast_prov(dia: int, spacing: int) -> float:
    """Calculates provided steel area (mm²/m) for a given bar size and spacing."""
    if dia not in BAR_AREAS or spacing <= 0: return 0.0
    return BAR_AREAS[dia] * 1000 / spacing

def get_k_factor(d_eff_mm):
    """IS 456:2000 Cl 40.2.1.1 (k factor for reduced effective depth)"""
    d = d_eff_mm
    if d > 300: return 1.0
    k = 1.6 - 0.002 * d 
    return min(k, 1.3) 

def get_tau_c(fck, pt):
    """IS 456:2000 Table 19 (Design shear strength of concrete)"""
    pt_clamped = max(0.15, min(3.0, pt))
    
    if fck <= 25.0: col = 'M25'
    elif fck >= 30.0: col = 'M30'
    else: col = 'M30'
    
    # Interpolate for Pt
    val = bilinear_interpolate(pt_clamped, TAU_C_TABLE, col)
    
    # Scale for fck > 30 
    if fck > 30:
        val = val * math.pow(fck/30, 0.33) 
    
    return val

def plot_results(H: float, M_base_L: float, M_base_B: float, V_base_max: float):
    """Generates simple plots for Wall forces."""
    V_max_str = f'{V_base_max:.1f}'
    M_L_str = f'{M_base_L:.1f}'
    M_B_str = f'{M_base_B:.1f}'

    def create_plotly_plot(title: str, x_values: List[float], y_values: List[float], label_text: str, color: str) -> go.Figure:
        fig = go.Figure()
        rgb_str = ','.join(str(int(color.lstrip("#")[i:i+2], 16)) for i in (0, 2, 4))
        fill_color_rgba = f'rgba({rgb_str}, 0.2)'
        
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', fill='toself', fillcolor=fill_color_rgba, line=dict(color=color, width=2), showlegend=False))
        fig.add_trace(go.Scatter(x=[0, 0], y=[0, H], mode='lines', line=dict(color='black', dash='dash'), showlegend=False))
        
        # Add force value annotation at base
        fig.add_annotation(x=x_values[1] * 1.1, y=0.05, text=label_text, showarrow=False, font=dict(color=color, size=10), xanchor='left')
        
        fig.update_layout(
            title=f"<b>{title}</b>",
            xaxis=dict(title=title.split('(')[0].replace('Vertical Moment -', 'Moment') + ' (kN/m or kNm/m)', zeroline=True, gridcolor='lightgray', range=[-0.1 * max(x_values) if max(x_values) > 0 else -1, max(x_values) * 1.3 if max(x_values) > 0 else 1]),
            yaxis=dict(title="Height (m)", range=[-0.1, H + 0.1], zeroline=True, gridcolor='lightgray'),
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig

    fig_v = create_plotly_plot(title="Shear Force ($V$)", x_values=[0, V_base_max, 0], y_values=[0, 0, H], label_text=f"V<sub>max</sub> = {V_max_str} kN/m", color='#FF0000')
    fig_m_L = create_plotly_plot(title="Vertical Moment - Long Wall ($M_L$)", x_values=[0, M_base_L, 0], y_values=[0, 0, H], label_text=f"M<sub>L</sub> = {M_L_str} kNm/m", color='#0000FF')
    fig_m_B = create_plotly_plot(title="Vertical Moment - Short Wall ($M_B$)", x_values=[0, M_base_B, 0], y_values=[0, 0, H], label_text=f"M<sub>B</sub> = {M_B_str} kNm/m", color='#008000')

    col_v, col_m_l, col_m_b = st.columns(3)
    with col_v:
        st.plotly_chart(fig_v, use_container_width=True)
    with col_m_l:
        st.plotly_chart(fig_m_L, use_container_width=True)
    with col_m_b:
        st.plotly_chart(fig_m_B, use_container_width=True)


# ===============================
# Streamlit App Execution (Inline)
# ===============================

st.title("💧 RCC Water Tank Design and Analysis (IS 3370 / IS 456)")
st.markdown("---")

# --- INITIALIZATION & INPUTS ---
if 'mat' not in st.session_state:
    st.session_state.mat = Materials()
    st.session_state.geom = Geometry()
    st.session_state.loads = Loads()

mat, geom, loads = st.session_state.mat, st.session_state.geom, st.session_state.loads


# Column Layout for Inputs
st.header("1. Input Data")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Materials (IS 456)")
    mat.fck = st.number_input("Concrete Grade $f_{ck}$ (MPa)", 25.0, 50.0, mat.fck, 5.0, key="mat_fck")
    mat.fy = st.number_input("Steel Grade $f_{y}$ (MPa)", 250.0, 550.0, mat.fy, 50.0, key="mat_fy")
    mat.Ec = st.number_input("Concrete E ($E_{c}$) (MPa)", 25000.0, 40000.0, mat.Ec, 1000.0, format="%.0f", key="mat_ec")
    mat.SBC = st.number_input("Soil Bearing Capacity (SBC) (kN/m²)", 50.0, 400.0, mat.SBC, 10.0, key="mat_sbc")
    
with col2:
    st.subheader("Geometry (Wall)")
    geom.H = st.number_input("Height of Water $H$ (m)", 1.0, 10.0, geom.H, 0.1, key="geom_h")
    geom.L = st.number_input("Long Wall $L$ (m)", 1.0, 20.0, geom.L, 0.1, key="geom_l")
    geom.B = st.number_input("Short Wall $B$ (m)", 1.0, 20.0, geom.B, 0.1, key="geom_b")
    geom.t_wall = st.number_input("Wall Thickness $t_{w}$ (m)", 0.15, 1.0, geom.t_wall, 0.05, key="geom_tw")
    geom.t_base = st.number_input("Base Thickness $t_{b}$ (m)", 0.15, 1.0, geom.t_base, 0.05, key="geom_tb")
    
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
d_eff = tw_mm - 50.0 # Effective depth for wall steel (approx. average)
Ast_min_perc = 0.35 if geom.t_wall >= 0.20 else 0.25 # IS 3370 Cl 7.1
A_conc_total = 1000.0 * tw_mm 
Ast_min_face = (Ast_min_perc / 100.0) * A_conc_total / 2.0
sigma_allow = 130.0 # IS 3370 Cl 8.3 (Fe 415, crack width control)
gamma_f = 1.5 


st.markdown("---")

# ----------------------------------------------------
# --- LOAD CALCULATION SECTION 
# ----------------------------------------------------
st.header("2. Load Calculations and Coefficients")
st.markdown(f"The aspect ratio $L/H = {geom.L}/{geom.H} = **{L_over_H:.2f}**$ is used to determine plate bending and tension coefficients (IS 3370-4).")


R_liq = 0.5 * loads.gamma_w * geom.H**2
P_max_w = loads.gamma_w * geom.H

if geom.tank_type == "Ground":
    R_soil_total = 0.5 * loads.gamma_s * loads.K0 * (geom.H + geom.t_base)**2
    M_soil_base = R_soil_total * ((geom.H + geom.t_base) / 3.0) 
else:
    R_soil_total = 0.0
    M_soil_base = 0.0

# Interpolate Coefficients
C_corner = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Corner (Max)')
C_mid_L = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Mid_Long')
C_mid_B = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Mid_Short')
C_T_max = bilinear_interpolate(L_over_H, T_COEF_TABLE, 'Tension_Max')
C_Mx_corner = bilinear_interpolate(L_over_H, M_H_COEF_TABLE, 'Corner_Max')

# Unfactored Forces (FL)
M_base_corner_FL = C_corner * loads.gamma_w * geom.H**3
M_base_mid_L_FL = C_mid_L * loads.gamma_w * geom.H**3
M_base_mid_B_FL = C_mid_B * loads.gamma_w * geom.H**3
T_H_max_FL = C_T_max * loads.gamma_w * geom.H**2
M_H_corner_FL = C_Mx_corner * loads.gamma_w * geom.H**3

st.subheader("2.1 Detailed Narrative on Coefficient Evaluation")
st.markdown("""
The coefficients are evaluated based on **IS 3370 (Part 4)**, which treats the wall as a **rectangular plate fixed at the base and continuous/fixed at the vertical corners**, subjected to triangular hydrostatic pressure ($P_w = \\gamma_w z$).
* **Vertical Moment Coefficients ($C_{My}$):** These determine the vertical moment ($M_{My}$) at the fixed base per unit width of the wall. The wall is considered fixed along its **vertical span $H$ (at the base)** and partially restrained horizontally by the adjacent walls/corners. The moment is calculated as:
    $$\\mathbf{M_{My}} = C_{My} \\cdot \\gamma_w \\cdot H^3$$
    $C_{My, corner}$ is the maximum moment near the vertical corner, and $C_{My, mid}$ is the moment at the horizontal mid-span of the wall.
* **Horizontal Tension Coefficient ($C_{T, max}$):** This coefficient determines the maximum **Hoop Tension** ($T_H$) per unit height, which acts horizontally due to the tank's circumferential restraint.
    $$\\mathbf{T_H} = C_{T, max} \\cdot \\gamma_w \\cdot H^2$$
* **Horizontal Moment Coefficient ($C_{Mx}$):** This determines the **Horizontal Bending Moment** ($M_{Mx}$) near the vertical corners, which arises due to the horizontal plate bending (incompatibility of hoop deflection).
    $$\\mathbf{M_{Mx}} = C_{Mx} \\cdot \\gamma_w \\cdot H^3$$
""")

st.subheader("2.2 Unfactored Design Forces (Load Factor$=1.0$)")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.markdown(f"Max Vertical Moment ($M_{{My, corner}}$): **{M_base_corner_FL:.2f} kNm/m**")
    st.markdown(f"Max Base Shear ($V_{{max}}$): **{R_liq:.2f} kN/m**")
with col_f2:
    st.markdown(f"Max Hoop Tension ($T_{{H, max}}$): **{T_H_max_FL:.2f} kN/m**")
    st.markdown(f"Max Horizontal Moment ($M_{{H, corner}}$): **{M_H_corner_FL:.2f} kNm/m**")

st.markdown("---")

# ----------------------------------------------------
# --- WALL DESIGN SECTION 
# ----------------------------------------------------
st.header("3. Wall Reinforcement Design")

# --- VERTICAL REINFORCEMENT DESIGN ---
st.subheader("3.1 Vertical Reinforcement ($M_{My}$ and Shear Check)")
st.markdown(f"Minimum steel area per face is $A_{{st, min}} = **{Ast_min_face:.0f} \\text{{ mm}}^2/\\text{{m}}$** ($\\rho_{{min}} = {Ast_min_perc/2.0:.2f}\\%$).")

# Moment Requirements
Mu_inner_tension = gamma_f * M_base_corner_FL 
Mu_outer_tension = gamma_f * M_soil_base if geom.tank_type == "Ground" else 0.0 
Ast_req_V_inner = max(demand_ast_from_M(Mu_inner_tension, d_eff, mat.fy, mat.fck), Ast_min_face)
Ast_req_V_outer = max(demand_ast_from_M(Mu_outer_tension, d_eff, mat.fy, mat.fck), Ast_min_face)

col_req_v, col_in, col_out = st.columns(3)

with col_req_v:
    st.markdown("#### Required $A_{st}$")
    st.markdown(f"**Inner Face (Water):** **{Ast_req_V_inner:.0f} $\\text{{mm}}^2/\\text{{m}}$**")
    st.markdown(f"**Outer Face (Air/Earth):** **{Ast_req_V_outer:.0f} $\\text{{mm}}^2/\\text{{m}}$**")

# --- User Selection for Vertical Steel (Inner/Outer) ---
with col_in:
    st.markdown("##### Inner Face Selection")
    dia_v_in = st.selectbox("Bar $\\phi$ (Vertical Inner)", options=list(BAR_AREAS.keys()), index=list(BAR_AREAS.keys()).index(12), key="dia_v_in")
    s_v_in = st.selectbox("Spacing $s$ (mm c/c) Inner", options=[100, 125, 150, 175, 200, 250, 300], index=1, key="s_v_in")
    Ast_prov_v_in = calc_Ast_prov(dia_v_in, s_v_in)
    pass_in = '✅ PASS' if Ast_prov_v_in >= Ast_req_V_inner else '❌ FAIL'
    st.markdown(f"**$A_{{st, prov}}$: {Ast_prov_v_in:.0f} $\\text{{mm}}^2/\\text{{m}}$** $\\rightarrow$ **{pass_in}**")
with col_out:
    st.markdown("##### Outer Face Selection")
    dia_v_out = st.selectbox("Bar $\\phi$ (Vertical Outer)", options=list(BAR_AREAS.keys()), index=list(BAR_AREAS.keys()).index(10), key="dia_v_out")
    s_v_out = st.selectbox("Spacing $s$ (mm c/c) Outer", options=[100, 125, 150, 175, 200, 250, 300], index=2, key="s_v_out")
    Ast_prov_v_out = calc_Ast_prov(dia_v_out, s_v_out)
    pass_out = '✅ PASS' if Ast_prov_v_out >= Ast_req_V_outer else '❌ FAIL'
    st.markdown(f"**$A_{{st, prov}}$: {Ast_prov_v_out:.0f} $\\text{{mm}}^2/\\text{{m}}$** $\\rightarrow$ **{pass_out}**")

st.markdown("#### Shear Check (Wall Base) $\\rightarrow$ IS 456:2000 Cl 40")
V_u = gamma_f * R_liq
tau_v = V_u * 1000 / (1000 * d_eff) 
# P_t is the percentage of tension steel (Ast_prov_v_in governs for max moment)
pt_inner = (Ast_prov_v_in / 1000) * 100 / d_eff  
tau_c = get_tau_c(mat.fck, pt_inner)
k = get_k_factor(d_eff)
tau_c_max_base = 0.6 * math.sqrt(mat.fck) # IS 456 Table 20 

st.markdown(f"""
- Ultimate Shear Force $V_u$: **{V_u:.2f} kN/m**
- Nominal Shear Stress $\\tau_v = V_u / (b \\cdot d)$: **{tau_v:.2f} $\\text{{MPa}}$**
- Percentage **Tension Steel** $P_{{t}}$ (Inner Face $A_{{st}}$): **{pt_inner:.2f} %**
- Design Shear Strength $\\tau_c$ (from Table 19): **{tau_c:.2f} $\\text{{MPa}}$**
- Depth Factor $k$ (IS 456 Cl 40.2.1.1): **{k:.2f}**
- Maximum Shear Strength $k \\cdot \\tau_{{c, max}}$ (IS 456 Table 20): **{k * tau_c_max_base:.2f} $\\text{{MPa}}$**
""")
shear_result = "✅ PASS (Shear reinforcement is not required as $\\tau_v \\le \\tau_c$)" if tau_v <= tau_c else "❌ FAIL (Shear reinforcement required)"
if tau_v > k * tau_c_max_base: shear_result = "❌ FAIL (Section redesign required - $\\tau_v > k \\tau_{c, max}$)"

st.markdown(f"**Conclusion:** $\\tau_v$ ({tau_v:.2f}) $\\le$ $\\tau_c$ ({tau_c:.2f}) $\\rightarrow$ **{shear_result}**")

st.markdown("---")

# --- HORIZONTAL REINFORCEMENT DESIGN ---
st.subheader("3.2 Horizontal Reinforcement (Hoop Tension $T_H$ and Moment $M_{H}$)")

Ast_req_tension_total = T_H_max_FL * 1000 / sigma_allow 
Ast_req_H_min_face = max(Ast_req_tension_total / 2.0, Ast_min_face)

# Horizontal Moment requires moment steel at the corner (tension on inner face)
Mu_H_design = gamma_f * M_H_corner_FL
Ast_req_moment_face = demand_ast_from_M(Mu_H_design, d_eff, mat.fy, mat.fck)
Ast_req_H_inner = max(Ast_req_H_min_face, Ast_req_moment_face)
Ast_req_H_outer = Ast_req_H_min_face # Governed by T_H/2 or min steel

st.markdown(f"**Max Governing $A_{{st, req}}$ (Inner Face):** **{Ast_req_H_inner:.0f} $\\text{{mm}}^2/\\text{{m}}$**.")
st.markdown(f"**$A_{{st, req}}$ (Outer Face):** **{Ast_req_H_outer:.0f} $\\text{{mm}}^2/\\text{{m}}$**.")

# --- User Selection for Horizontal Steel (Inner/Outer) ---
col_req_h, col_h_in, col_h_out = st.columns(3)
with col_h_in:
    st.markdown("##### Inner Face Selection")
    dia_h_in = st.selectbox("Bar $\\phi$ (Horiz. Inner)", options=list(BAR_AREAS.keys()), index=list(BAR_AREAS.keys()).index(12), key="dia_h_in")
    s_h_in = st.selectbox("Spacing $s$ (mm c/c) Horiz. Inner", options=[100, 125, 150, 175, 200, 250, 300], index=1, key="s_h_in")
    Ast_prov_h_in = calc_Ast_prov(dia_h_in, s_h_in)
    pass_h_in = '✅ PASS' if Ast_prov_h_in >= Ast_req_H_inner else '❌ FAIL'
    st.markdown(f"**$A_{{st, prov}}$: {Ast_prov_h_in:.0f} $\\text{{mm}}^2/\\text{{m}}$** $\\rightarrow$ **{pass_h_in}**")
with col_h_out:
    st.markdown("##### Outer Face Selection")
    dia_h_out = st.selectbox("Bar $\\phi$ (Horiz. Outer)", options=list(BAR_AREAS.keys()), index=list(BAR_AREAS.keys()).index(12), key="dia_h_out")
    s_h_out = st.selectbox("Spacing $s$ (mm c/c) Horiz. Outer", options=[100, 125, 150, 175, 200, 250, 300], index=2, key="s_h_out")
    Ast_prov_h_out = calc_Ast_prov(dia_h_out, s_h_out)
    pass_h_out = '✅ PASS' if Ast_prov_h_out >= Ast_req_H_outer else '❌ FAIL'
    st.markdown(f"**$A_{{st, prov}}$: {Ast_prov_h_out:.0f} $\\text{{mm}}^2/\\text{{m}}$** $\\rightarrow$ **{pass_h_out}**")


st.markdown("---")

# ----------------------------------------------------
# --- BASE SLAB (RAFT) DESIGN SECTION 
# ----------------------------------------------------
st.header("4. Base Slab (Raft) Design")
tb_mm = geom.t_base * M_TO_MM
d_eff_b = tb_mm - 75.0 
Ast_min_base = (0.12 / 100.0) * 1000 * tb_mm # IS 456 Cl 26.5.2.1

st.subheader("4.1 Base Pressure Calculation and SBC Check")

W_water = geom.L * geom.B * loads.gamma_w * geom.H
W_base = geom.L * geom.B * mat.gamma_conc * geom.t_base
W_walls = 2 * (geom.L + geom.B) * geom.H * mat.gamma_conc * geom.t_wall
W_total = W_water + W_base + W_walls
Q_avg = W_total / (geom.L * geom.B)

st.markdown(f"""
- Total Weight $W_{{total}}$: **{W_total:.1f} kN**
- **Average Base Pressure $Q_{{avg}}$ (Tank Full):** **{Q_avg:.1f} $\\text{{kN/m}}^2$**
- **Allowable SBC:** **{mat.SBC:.1f} $\\text{{kN/m}}^2$**
- **SBC Check:** **{'✅ PASS' if Q_avg <= mat.SBC else '❌ FAIL'}**
""")

# Base pressure diagram visualization (conceptual)
pressure_diag_data = {
    'Pressure (kN/m²)': [0, Q_avg, Q_avg, 0],
    'X Position (m)': [0, 0, geom.L, geom.L]
}
fig_base_pressure = go.Figure()
fig_base_pressure.add_trace(go.Scatter(x=pressure_diag_data['X Position (m)'], y=pressure_diag_data['Pressure (kN/m²)'], fill='toself', fillcolor='rgba(255,165,0,0.3)', line=dict(color='orange'), name='Soil Reaction'))
fig_base_pressure.update_layout(
    title='<b>Conceptual Base Pressure Diagram (Uniform)</b>',
    xaxis_title='Tank Length (L) [m]',
    yaxis_title='Pressure (kN/m²)',
    yaxis_range=[0, max(Q_avg * 1.2, mat.SBC * 1.2)],
    height=300
)
fig_base_pressure.add_trace(go.Scatter(x=[0, geom.L], y=[mat.SBC, mat.SBC], mode='lines', line=dict(color='red', dash='dash'), name='Allowable SBC'))
st.plotly_chart(fig_base_pressure, use_container_width=True)

st.subheader("4.2 Base Slab Moment and Reinforcement (Two-Way Slab Action)")

# Max net pressure for design 
q_base_max = loads.gamma_w * geom.H + mat.gamma_conc * geom.t_base
q_design_unfactored = max(q_base_max - Q_avg, Ast_min_base * 0) 
q_design_uls = gamma_f * q_base_max 

# Two-way slab moments (IS 456 Annex D simplified: Four Edges Continuous)
# Mu_x is moment along B (short span) -> controls steel parallel to L
# Mu_y is moment along L (long span) -> controls steel parallel to B
L_ratio = geom.L / geom.B
L_ratio_clamped = max(1.0, min(2.0, L_ratio))
L_B_key = min(TWO_WAY_COEF.keys(), key=lambda x: abs(x - L_ratio_clamped))
alpha_x, alpha_y = TWO_WAY_COEF[L_B_key]

Mu_base_x = alpha_x * q_design_uls * geom.B**2 
Mu_base_y = alpha_y * q_design_uls * geom.B**2 

# Steel running parallel to L (long side), resisting Mu_x (short span moment)
Ast_req_parallel_L = max(demand_ast_from_M(Mu_base_x, d_eff_b, mat.fy, mat.fck), Ast_min_base)
# Steel running parallel to B (short side), resisting Mu_y (long span moment)
Ast_req_parallel_B = max(demand_ast_from_M(Mu_base_y, d_eff_b, mat.fy, mat.fck), Ast_min_base)

st.markdown(f"""
- Design ULS Pressure $q_{{u}}$: **{q_design_uls:.1f} $\\text{{kN/m}}^2$** (Using $1.5 \\times$ Total Downward Pressure)
- $L/B$ Ratio: **{L_ratio:.2f}** $\\rightarrow$ Coeff. $\\alpha_{{x}}={alpha_x:.3f}, \\alpha_{{y}}={alpha_y:.3f}$
- **Moment $M_{{u, x}}$ (Moment along Short Span $B$):** **{Mu_base_x:.2f} kNm/m** (Calculated with $\\alpha_x$, resisted by steel $\\parallel$ to **Long Span $L$**).
- **Moment $M_{{u, y}}$ (Moment along Long Span $L$):** **{Mu_base_y:.2f} kNm/m** (Calculated with $\\alpha_y$, resisted by steel $\\parallel$ to **Short Span $B$**).
- $A_{{st, min}}$ (Base): **{Ast_min_base:.0f} $\\text{{mm}}^2/\\text{{m}}$** (0.12% of total area)
- **$A_{{st, req}} \\parallel L$:** **{Ast_req_parallel_L:.0f} $\\text{{mm}}^2/\\text{{m}}$**
- **$A_{{st, req}} \\parallel B$:** **{Ast_req_parallel_B:.0f} $\\text{{mm}}^2/\\text{{m}}$**
""")

# --- User Selection for Base Slab Steel (Bottom/Top) ---
st.markdown("##### Base Slab Reinforcement Selection (Bottom Mesh Governs)")
col_bx, col_by = st.columns(2)
with col_bx:
    st.markdown("**Bottom Mesh - Parallel to L (Long Span)**")
    dia_b_x = st.selectbox("Bar $\\phi$ ($\\parallel L$)", options=list(BAR_AREAS.keys()), index=list(BAR_AREAS.keys()).index(12), key="dia_b_x")
    s_b_x = st.selectbox("Spacing $s$ (mm c/c) ($\\parallel L$)", options=[100, 125, 150, 175, 200, 250, 300], index=1, key="s_b_x")
    Ast_prov_b_x = calc_Ast_prov(dia_b_x, s_b_x)
    pass_bx = '✅ PASS' if Ast_prov_b_x >= Ast_req_parallel_L else '❌ FAIL'
    st.markdown(f"**$A_{{st, prov}}$: {Ast_prov_b_x:.0f} $\\text{{mm}}^2/\\text{{m}}$** $\\rightarrow$ **{pass_bx}**")
with col_by:
    st.markdown("**Bottom Mesh - Parallel to B (Short Span)**")
    dia_b_y = st.selectbox("Bar $\\phi$ ($\\parallel B$)", options=list(BAR_AREAS.keys()), index=list(BAR_AREAS.keys()).index(10), key="dia_b_y")
    s_b_y = st.selectbox("Spacing $s$ (mm c/c) ($\\parallel B$)", options=[100, 125, 150, 175, 200, 250, 300], index=2, key="s_b_y")
    Ast_prov_b_y = calc_Ast_prov(dia_b_y, s_b_y)
    pass_by = '✅ PASS' if Ast_prov_b_y >= Ast_req_parallel_B else '❌ FAIL'
    st.markdown(f"**$A_{{st, prov}}$: {Ast_prov_b_y:.0f} $\\text{{mm}}^2/\\text{{m}}$** $\\rightarrow$ **{pass_by}**")

st.markdown("---")

st.header("5. Visual Results and Summary")
st.subheader("Vertical Wall Forces (Moment & Shear)")
plot_results(geom.H, M_base_mid_L_FL, M_base_mid_B_FL, R_liq)
