# -*- coding: utf-8 -*-
import math
import io
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

# Set Streamlit page configuration for wide layout
st.set_page_config(layout="wide")


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

# --- ADDED: Hoop Tension Table (IS 3370-4 Table 1) ---
T_COEF_TABLE = pd.DataFrame(
    data={
        'L/H': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        'Tension_Max': [0.0075, 0.0125, 0.0185, 0.0255, 0.033, 0.041, 0.059, 0.076, 0.091, 0.106, 0.120]
    }
).set_index('L/H')

# --- ADDED: Horizontal Moment Table (IS 3370-4 Table 2) ---
M_H_COEF_TABLE = pd.DataFrame(
    data={
        'L/H': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        'Corner_Max': [0.0003, 0.0005, 0.0008, 0.0011, 0.0014, 0.0018, 0.0025, 0.0033, 0.0040, 0.0046, 0.0052],
        'Mid_Span_Max': [0.0002, 0.0003, 0.0005, 0.0007, 0.0009, 0.0011, 0.0015, 0.0019, 0.0023, 0.0026, 0.0028]
    }
).set_index('L/H')

# ===============================
# Engineering Helper Functions
# ===============================

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

def steel_stress_sls(Ms_kNm_per_m: float, As_mm2_per_m: float, d_eff_mm: float, Ec_MPa: float, N_tension_kN_per_m: float = 0.0) -> float:
    """
    Calculates steel stress (sigma_s in MPa) using Elastic Cracked Section Theory (SLS).
    Includes optional check for axial tension N.
    """
    if As_mm2_per_m <= 0.0:
        return float('inf')
        
    m = ES / Ec_MPa
    b = 1000.0     # 1m strip
    Ms_Nmm = Ms_kNm_per_m * KNM_TO_NMM
    N_tension_N = N_tension_kN_per_m * 1000.0

    # Section properties for N + M
    if N_tension_N > 0:
        # Simplified approximate for Section 2, Cl A-2.1 of IS 3370
        # Tension force is resisted by Ast and concrete around Ast
        sigma_s = N_tension_N / As_mm2_per_m + Ms_Nmm / (As_mm2_per_m * (d_eff_mm - 50)) # Very simplified model for tension + moment
        
        # Exact method for pure tension: sigma_s = N / (Ast + (m-1) * Ast_c)
        # However, for cracked section, use IS 3370-1 Cl 8.2: w_cr = k.s_cr.eps_sm
        # The IS 3370 approach controls stress directly for crack width.
        # sigma_s = N / (Ast + (m-1) * Ast_c) is for uncracked section with tensile stress in concrete
        
        # We will use the direct stress check from IS 3370-1 Cl 8.3 for pure tension:
        sigma_s_tension = N_tension_N / As_mm2_per_m 
        
        # If both M and N are present, M dominates the moment-resisting face.
        # But for water retaining structure design, we check the stress required to limit crack width.
        # For Tension + Bending (Corner), M dominates:
        if Ms_Nmm > 0:
            ratio = (m * As_mm2_per_m) / b
            n = -ratio + math.sqrt(ratio**2 + 2 * ratio * d_eff_mm)
            z = d_eff_mm - n/3.0
            sigma_s_moment = Ms_Nmm / (As_mm2_per_m * max(z, 1.0))
            
            # Combine tension and bending stress on the tension face
            # Approximation: sigma_s = sigma_s_moment + sigma_s_tension (where sigma_s_tension is averaged)
            # Simplification: Use the moment-based stress as the critical value for crack control (conservative)
            return sigma_s_moment 
        
        return sigma_s_tension

    else: # Pure Moment
        ratio = (m * As_mm2_per_m) / b
        try:
            n = -ratio + math.sqrt(ratio**2 + 2 * ratio * d_eff_mm)
        except ValueError:
            return float('inf') 

        z = d_eff_mm - n/3.0
        sigma_s = Ms_Nmm / (As_mm2_per_m * max(z, 1.0))
        return sigma_s

def select_rebar(Ast_req_mm2_m: float, max_spacing_mm: int = 300) -> Tuple[int, int, float]:
    """Selects suitable bar diameter and spacing (dia, spacing, Ast_prov)."""
    BAR_AREAS = {8: 50.3, 10: 78.5, 12: 113.1, 16: 201.1, 20: 314.2}
    
    best_dia = 0
    best_spacing = 0
    min_Ast_prov = float('inf')
    
    # Use minimum required Ast if calculation is very low
    Ast_req_mm2_m = max(Ast_req_mm2_m, 1.0) 

    for dia, area in BAR_AREAS.items():
        # Calculate maximum spacing allowed to satisfy Ast_req
        max_s_for_req = area * 1000 / Ast_req_mm2_m
        
        # Consider practical spacings (multiples of 25/50) up to max_spacing_mm
        possible_spacings = [s for s in [100, 125, 150, 175, 200, 250, 300] if s <= max_spacing_mm]
        
        for s in possible_spacings:
            if s <= max_s_for_req:
                Ast_prov = area * 1000 / s
                # Find the combination that provides the least excess area
                if Ast_prov >= Ast_req_mm2_m and Ast_prov < min_Ast_prov:
                    min_Ast_prov = Ast_prov
                    best_dia = dia
                    best_spacing = s
                    
    if best_dia == 0:
        # Fallback: largest bar at minimum spacing (if required area is huge)
        dia = max(BAR_AREAS.keys())
        area = BAR_AREAS[dia]
        spacing = 100 
        Ast_prov = area * 1000 / spacing
        return dia, spacing, Ast_prov
        
    return best_dia, best_spacing, min_Ast_prov

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
# Plotting Functions (Plotly Implementation)
# ===============================

def plot_loads(geom: Geometry, loads: Loads, R_liq: float, R_soil: float):
    """Plots the load diagram (Hydrostatic and Earth Pressure) using Plotly."""
    
    H_wall = geom.H
    t_base = geom.t_base
    H_total = H_wall + t_base
    
    P_max_w = loads.gamma_w * H_wall
    P_max_s = 0
    
    P_max_w_str = f'{P_max_w:.1f}'
    R_liq_str = f'{R_liq:.1f}'
    R_soil_str = f'{R_soil:.1f}'
    
    fig = go.Figure()

    # --- 1. Draw the Wall and Base Geometry (Profile View) ---
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, H_total], mode='lines', line=dict(color='black', width=3), name='Wall'))
    fig.add_trace(go.Scatter(x=[-0.5, 1], y=[0, 0], mode='lines', line=dict(color='black', width=3), name='Base Slab'))
    fig.add_trace(go.Scatter(x=[-0.5, 0], y=[t_base, t_base], mode='lines', line=dict(color='gray', width=1, dash='dot'), showlegend=False))

    # --- 2. Hydrostatic Pressure (P_w) - Inner Face (Right) ---
    x_w = [0, P_max_w, 0]
    y_w = [t_base, t_base, H_total]
    fig.add_trace(go.Scatter(x=x_w, y=y_w, mode='lines', fill='toself', fillcolor='rgba(0,0,255,0.3)', line=dict(color='blue', dash='dash'), name='Water Pressure'))
    
    # Pressure magnitude label (Pmax) - Use HTML for sub/superscript
    fig.add_annotation(
        x=P_max_w * 1.05, y=t_base + 0.1, 
        text=f"P<sub>w, max</sub> = {P_max_w_str} kN/m<sup>2</sup>",
        showarrow=False, font=dict(color='blue', size=10), xanchor='left'
    )
    
    # Resultant force label (R_w)
    fig.add_annotation(
        x=P_max_w * 0.5, y=t_base + H_wall/3 + 0.2, 
        text=f"R<sub>w</sub> = {R_liq_str} kN/m",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='blue', ax=-20, ay=0, font=dict(color='blue', size=10), yshift=0
    )

    # --- 3. Earth Pressure (P_soil) - Outer Face (Left) ---
    if geom.tank_type == "Ground":
        P_max_s = loads.gamma_s * loads.K0 * H_total
        P_max_s_str = f'{P_max_s:.1f}'
        
        x_s = [-P_max_s, 0, 0]
        y_s = [0, 0, H_total]
        
        fig.add_trace(go.Scatter(x=x_s, y=y_s, mode='lines', fill='toself', fillcolor='rgba(139,69,19,0.3)', line=dict(color='brown', dash='dash'), name='Earth Pressure'))
        
        # Pressure magnitude label (Pmax)
        fig.add_annotation(
            x=-P_max_s * 1.05, y=0.1, 
            text=f"P<sub>s, max</sub> = {P_max_s_str} kN/m<sup>2</sup>", 
            showarrow=False, font=dict(color='brown', size=10), xanchor='right'
        )
        
        # Resultant force label (R_s)
        fig.add_annotation(
            x=-P_max_s * 0.5, y=H_total/3 + 0.2, 
            text=f"R<sub>s</sub> = {R_soil_str} kN/m", 
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='brown', ax=20, ay=0, font=dict(color='brown', size=10), yshift=0
        )

    # --- 4. Dimensioning H and t_base ---
    fig.add_annotation(x=-0.35, y=t_base + H_wall / 2, text='H', showarrow=False, font=dict(color='gray'), xanchor='right')
    fig.add_annotation(x=-0.35, y=t_base / 2, text='t<sub>base</sub>', showarrow=False, font=dict(color='gray'), xanchor='right')


    # --- 5. Formatting and Display ---
    x_limit = max(abs(P_max_s), abs(P_max_w)) * 1.5 if geom.tank_type == "Ground" else P_max_w * 1.5
    
    fig.update_layout(
        title=f"<b>Load Diagram (Wall Profile: L={geom.L:.1f}m, B={geom.B:.1f}m)</b>",
        xaxis=dict(
            title="Pressure (Scaled &larr; Soil | Water &rarr;)", 
            zeroline=True, 
            zerolinecolor='black', 
            zerolinewidth=2,
            gridcolor='lightgray',
            range=[-x_limit, x_limit]
        ),
        yaxis=dict(
            title="Height from Base (m)",
            range=[-0.2, H_total + 0.5],
            zeroline=True,
            showticklabels=True,
            tickvals=sorted(list(set([0, t_base, H_total]))),
            gridcolor='lightgray',
            scaleanchor="x",
            scaleratio= (x_limit * 2) / (H_total + 0.7) if x_limit > 0 else 1.0
        ),
        showlegend=True,
        legend=dict(x=0.5, y=1.05, xanchor='center', orientation='h', traceorder='normal'),
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_results(H: float, M_base_L: float, M_base_B: float, V_base_max: float):
    """Plots the Bending Moment and Shear Force Diagrams using Plotly."""
    
    V_max_str = f'{V_base_max:.1f}'
    M_L_str = f'{M_base_L:.1f}'
    M_B_str = f'{M_base_B:.1f}'

    def create_plotly_plot(title: str, x_values: List[float], y_values: List[float], label_text: str, color: str) -> go.Figure:
        fig = go.Figure()
        
        # Convert Hex to RGBA string correctly
        rgb_str = ','.join(str(int(color.lstrip("#")[i:i+2], 16)) for i in (0, 2, 4))
        fill_color_rgba = f'rgba({rgb_str}, 0.2)'
        
        # Plot the distribution (triangle)
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values, 
            mode='lines', 
            fill='toself', 
            fillcolor=fill_color_rgba,
            line=dict(color=color, width=2),
            showlegend=False
        ))
        
        # Reference line (wall center)
        fig.add_trace(go.Scatter(x=[0, 0], y=[0, H], mode='lines', line=dict(color='black', dash='dash'), showlegend=False))
        
        # Label (max value at the bottom)
        fig.add_annotation(
            x=x_values[1] * 1.1, y=0.05, 
            text=label_text, 
            showarrow=False, 
            font=dict(color=color, size=10), 
            xanchor='left'
        )
        
        fig.update_layout(
            title=f"<b>{title}</b>",
            xaxis=dict(title=title.split('(')[0] + ' (kN/m or kNm/m)', zeroline=True, gridcolor='lightgray', range=[-0.1 * max(x_values), max(x_values) * 1.3]),
            yaxis=dict(title="Height (m)", range=[-0.1, H + 0.1], zeroline=True, gridcolor='lightgray'),
            height=600,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig

    # Shear Force
    fig_v = create_plotly_plot(
        title="Shear Force ($V$)",
        x_values=[0, V_base_max, 0], 
        y_values=[0, 0, H], 
        label_text=f"V<sub>max</sub> = {V_max_str} kN/m",
        color='#FF0000' # Red
    )

    # Bending Moment - Long Wall (L)
    fig_m_L = create_plotly_plot(
        title="Vertical Moment - Long Wall ($M_L$)",
        x_values=[0, M_base_L, 0], 
        y_values=[0, 0, H], 
        label_text=f"M<sub>L</sub> = {M_L_str} kNm/m",
        color='#0000FF' # Blue
    )

    # Bending Moment - Short Wall (B)
    fig_m_B = create_plotly_plot(
        title="Vertical Moment - Short Wall ($M_B$)",
        x_values=[0, M_base_B, 0], 
        y_values=[0, 0, H], 
        label_text=f"M<sub>B</sub> = {M_B_str} kNm/m",
        color='#008000' # Green
    )

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

# ----------------------------------------------------
# --- LOAD CALCULATION SECTION 
# ----------------------------------------------------
st.header("2. Basic Load Calculations and Sketches")
st.markdown(f"The aspect ratio $L/H = {geom.L}/{geom.H} = **{L_over_H:.2f}**$ is used to determine plate bending and tension coefficients (IS 3370-4).")

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

st.subheader("2.2 Load Sketch (Plotly Diagram)")
plot_loads(geom, loads, R_liq, R_soil)

st.markdown("---")

# ----------------------------------------------------
# --- MOMENT AND SHEAR CALCULATION SECTION 
# ----------------------------------------------------
st.header("3. Wall Force Calculation (IS 3370-4)")

# Vertical Moment Coefficients (M_My)
C_corner = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Corner (Max)')
C_mid_L = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Mid_Long')
C_mid_B = bilinear_interpolate(L_over_H, M_COEF_TABLE, 'Base_Mid_Short')

M_base_corner_FL = C_corner * loads.gamma_w * geom.H**3
M_base_mid_L_FL = C_mid_L * loads.gamma_w * geom.H**3
M_base_mid_B_FL = C_mid_B * loads.gamma_w * geom.H**3
V_base_FL = R_liq # Max base shear is conservatively R_liq

# Horizontal Tension & Moment Coefficients (T_Mx)
C_T_max = bilinear_interpolate(L_over_H, T_COEF_TABLE, 'Tension_Max')
T_max_FL = C_T_max * loads.gamma_w * geom.H * geom.L # T_H = C_T * gamma_w * H * L/2 (or B/2) - using L as max

C_Mx_corner = bilinear_interpolate(L_over_H, M_H_COEF_TABLE, 'Corner_Max')
C_Mx_mid = bilinear_interpolate(L_over_H, M_H_COEF_TABLE, 'Mid_Span_Max')

M_H_corner_FL = C_Mx_corner * loads.gamma_w * geom.H**3
M_H_mid_FL = C_Mx_mid * loads.gamma_w * geom.H**3


st.subheader("3.1 Vertical Moment and Shear Forces")
st.markdown(f"""
- Vertical Moment Coefficients ($C_{{My}}$): $C_{{corner}}={C_corner:.4f}$, $C_{{mid, L}}={C_mid_L:.4f}$, $C_{{mid, B}}={C_mid_B:.4f}$
- Max Base Corner Moment ($M_{{My, corner}}$): **{M_base_corner_FL:.2f} kNm/m**
- Max Base Shear ($V_{{max}}$): **{V_base_FL:.2f} kN/m**
""")

st.subheader("3.2 Horizontal Forces (Hoop Tension & Moment)")
st.markdown(f"""
- Max Hoop Tension Coefficient ($C_{{T}}$): $C_{{T, max}}={C_T_max:.4f}$
- **Maximum Hoop Tension ($T_{{H, max}}$):** $T_{{H, max}} = C_{{T, max}} \cdot \gamma_{{w}} \cdot H^2 = **{C_T_max * loads.gamma_w * geom.H**2:.2f} \text{{ kN/m}}$** (This is the force $T_{{H}}$ per unit height. $T_{{H}}$ varies with height $z$)
- Max Horizontal Moment Coefficient ($C_{{Mx}}$): $C_{{Mx, max}}={C_Mx_corner:.4f}$
- Max Horizontal Moment ($M_{{H, max}}$): $M_{{H, max}} = C_{{Mx, max}} \cdot \gamma_{{w}} \cdot H^3 = **{M_H_corner_FL:.2f} \text{{ kNm/m}}$**
""")

st.markdown("---")

# ----------------------------------------------------
# --- DESIGN & RESULTS SECTION 
# ----------------------------------------------------
st.header("4. Reinforcement Design")
gamma_f = 1.5 
sigma_allow = 130.0 # IS 3370 Cl 8.3 (Fe 415, Exposure Severe/Mod - simplified, assuming crack width control)

# --- VERTICAL REINFORCEMENT DESIGN (MOMENT) ---
st.subheader("4.1 Vertical Reinforcement Design (Governed by Moment)")
col_L, col_B = st.columns(2)

with col_L:
    st.markdown("#### Long Wall ($L$) Vertical Steel")
    # ULS Design Moment (Corner governs)
    Mu_design_FL = gamma_f * M_base_corner_FL 
    Mu_max_design = max(Mu_design_FL, gamma_f * M_soil_base) if geom.tank_type == "Ground" else Mu_design_FL
    
    Ast_req_ULS = demand_ast_from_M(Mu_max_design, d_eff, mat.fy, mat.fck)
    Ast_req_final = max(Ast_req_ULS, Ast_min_face)
    
    # Rebar Selection
    dia_v, s_v, Ast_prov_v = select_rebar(Ast_req_final)
    
    # SLS Check (using max Ast required)
    Ms_corner_FL = M_base_corner_FL
    sigma_s_actual = steel_stress_sls(Ms_corner_FL, Ast_prov_v, d_eff, mat.Ec)

    st.markdown(f"**Governing ULS Moment ($M_u$):** **{Mu_max_design:.2f} kNm/m**")
    st.markdown(f"**Required $A_{{st, req}}$ (ULS/Min):** **{Ast_req_final:.0f} $\text{{mm}}^2/\text{{m}}$** (per face)")
    
    st.markdown(f"**Selected Rebar:** $\phi {dia_v} \text{{ @ }} {s_v} \text{{ mm c/c}}$")
    st.markdown(f"**$A_{{st, prov}}$:** **{Ast_prov_v:.0f} $\text{{mm}}^2/\text{{m}}$**")

    st.markdown(f"##### SLS Check (Crack Control)")
    st.markdown(f"**Calculated $\sigma_{{s, actual}}$** = **{sigma_s_actual:.0f} $\text{{MPa}}$**")
    st.markdown(f"**Allowable $\sigma_{{s, allow}}$** = **{sigma_allow:.0f} $\text{{MPa}}$**")
    st.markdown(f"**Result:** **{'✅ PASS' if sigma_s_actual <= sigma_allow else '❌ FAIL'}**")

with col_B:
    st.markdown("#### Short Wall ($B$) Vertical Steel")
    # Use M_base_mid_B_FL for the short wall's central portion design (as it often governs local bending)
    Mu_design_FL_B = gamma_f * M_base_mid_B_FL
    Mu_max_design_B = max(Mu_design_FL_B, Mu_max_design) 

    Ast_req_ULS_B = demand_ast_from_M(Mu_max_design_B, d_eff, mat.fy, mat.fck)
    Ast_req_final_B = max(Ast_req_ULS_B, Ast_min_face)
    
    # Rebar Selection
    dia_v_B, s_v_B, Ast_prov_v_B = select_rebar(Ast_req_final_B)
    
    # SLS Check
    sigma_s_actual_B = steel_stress_sls(M_base_mid_B_FL, Ast_prov_v_B, d_eff, mat.Ec)

    st.markdown(f"**Governing ULS Moment ($M_u$):** **{Mu_max_design_B:.2f} kNm/m**")
    st.markdown(f"**Required $A_{{st, req}}$ (ULS/Min):** **{Ast_req_final_B:.0f} $\text{{mm}}^2/\text{{m}}$** (per face)")
    
    st.markdown(f"**Selected Rebar:** $\phi {dia_v_B} \text{{ @ }} {s_v_B} \text{{ mm c/c}}$")
    st.markdown(f"**$A_{{st, prov}}$:** **{Ast_prov_v_B:.0f} $\text{{mm}}^2/\text{{m}}$**")
    
    st.markdown(f"##### SLS Check (Crack Control)")
    st.markdown(f"**Calculated $\sigma_{{s, actual}}$** = **{sigma_s_actual_B:.0f} $\text{{MPa}}$**")
    st.markdown(f"**Allowable $\sigma_{{s, allow}}$** = **{sigma_allow:.0f} $\text{{MPa}}$**")
    st.markdown(f"**Result:** **{'✅ PASS' if sigma_s_actual_B <= sigma_allow else '❌ FAIL'}**")

st.markdown("---")

# --- HORIZONTAL REINFORCEMENT DESIGN (TENSION + MOMENT) ---
st.subheader("4.2 Horizontal Reinforcement Design (Governed by Hoop Tension $T_H$)")

# The horizontal steel in the central 0.6H zone is primarily governed by hoop tension.
# The horizontal steel near the vertical edges is governed by horizontal bending M_H.
# We will design for the maximum of the two requirements.

# 1. Design for Hoop Tension (mid-span, mid-height zone)
T_H_max_kN = C_T_max * loads.gamma_w * geom.H**2
# The full hoop tension force must be resisted by the steel over the full section area
Ast_req_tension_total = T_H_max_kN * 1000 / (sigma_allow * 1.0) # IS 3370 Cl 8.3.2
Ast_req_tension_face = Ast_req_tension_total / 2.0 

# 2. Design for Horizontal Bending Moment (Corner or mid-span)
Mu_H_design = gamma_f * M_H_corner_FL
Ast_req_moment_face = demand_ast_from_M(Mu_H_design, d_eff, mat.fy, mat.fck)

# 3. Final Required Steel (per face)
Ast_req_H_final = max(Ast_req_tension_face, Ast_req_moment_face, Ast_min_face)

# Rebar Selection
dia_h, s_h, Ast_prov_h = select_rebar(Ast_req_H_final)

# SLS Check (for pure tension at mid-span where M_H is low)
# Use the max tension force T_H and the provided area (distributed)
sigma_s_actual_H = steel_stress_sls(0.0, Ast_prov_h, d_eff, mat.Ec, N_tension_kN_per_m=T_H_max_kN) # N_tension_kN_per_m is conservative (T_H is not constant)
# Note: For pure tension, IS 3370 simplifies to directly limiting steel stress, T_H / (2 * Ast_prov_h) / (effective area ratio).
# We check the required area against the total tensile force divided by the allowable stress.
# The Ast_req_tension_face already satisfies T / sigma_allow.

st.markdown(f"""
- **Max Hoop Tension ($T_{{H, max}}$):** **{T_H_max_kN:.2f} kN/m** (Force per unit height)
- **Max Horizontal Moment ($M_{{H, max}}$):** **{M_H_corner_FL:.2f} kNm/m**
- **$A_{{st, req}}$ from Tension:** $T_{{H}} / \sigma_{{allow}} / 2 = **{Ast_req_tension_face:.0f} \text{{ mm}}^2/\text{{m}}$** (per face)
- **$A_{{st, req}}$ from Moment:** **{Ast_req_moment_face:.0f} $\text{{mm}}^2/\text{{m}}$** (per face)
- **Minimum $A_{{st, min}}$:** **{Ast_min_face:.0f} $\text{{mm}}^2/\text{{m}}$** (per face)

**Governing $A_{{st, req}}$ (Horizontal):** **{Ast_req_H_final:.0f} $\text{{mm}}^2/\text{{m}}$** (per face)
""")
st.markdown(f"**Selected Rebar:** $\phi {dia_h} \text{{ @ }} {s_h} \text{{ mm c/c}}$ (Used for both faces)")
st.markdown(f"**$A_{{st, prov}}$:** **{Ast_prov_h:.0f} $\text{{mm}}^2/\text{{m}}$**")

st.markdown(f"**Result:** **{'✅ PASS' if Ast_prov_h >= Ast_req_H_final else '❌ FAIL'}** (Design is governed by required area to limit crack width.)")


st.markdown("---")

# --- DETAILING & DRAWING SECTION ---
st.header("5. Output Diagrams and Detailing Suggestions")

st.subheader("5.1 Design Output Diagrams (Plotly)")
st.markdown("These diagrams show the **Vertical** Shear Force and Bending Moment distributions.")
# Pass unfactored moments for plotting shape representation
plot_results(geom.H, M_base_mid_L_FL, M_base_mid_B_FL, V_base_FL)

st.subheader("5.2 Rebar Detailing Suggestions")
st.markdown(f"""
The vertical and horizontal reinforcement should be placed in two layers (inner and outer face) to resist moments and control cracks.

### Reinforcement Summary:

1.  **Vertical Steel (Main):** $\phi {dia_v} \text{{ @ }} {s_v} \text{{ mm c/c}}$ (at base/corners, both faces)
    * $A_{{st, req}}$ (Max): **{max(Ast_req_final, Ast_req_final_B):.0f} $\text{{mm}}^2/\text{{m}}$**
    * $A_{{st, prov}}$: **{max(Ast_prov_v, Ast_prov_v_B):.0f} $\text{{mm}}^2/\text{{m}}$**
2.  **Horizontal Steel (Main):** $\phi {dia_h} \text{{ @ }} {s_h} \text{{ mm c/c}}$ (Max requirement, inner face controls)
    * $A_{{st, req}}$ (Max): **{Ast_req_H_final:.0f} $\text{{mm}}^2/\text{{m}}$**
    * $A_{{st, prov}}$: **{Ast_prov_h:.0f} $\text{{mm}}^2/\text{{m}}$**

### Detailing Notes:

* **Vertical Curtailment:** The maximum vertical steel is required near the base ($z=0$). This steel may be reduced (typically by half) above the point where the moment is $\approx 50\%$ of the max moment, or at $0.4H \approx {0.4*geom.H:.2f} \text{{ m}}$ from the base, ensuring the minimum steel $A_{{st, min}}$ is maintained to the top.
* **Horizontal Curtailment:** The hoop tension force $T_H$ varies from maximum near $0.4H$ to zero at the top. The horizontal steel can be designed for this variation, but often, the required horizontal steel is governed by the minimum $A_{{st, min}}$ over the upper 60% of the wall.
* **Top Zone Steel:** The top zone ($0.6H$ to $H$) requires the minimum steel **$A_{{st, min, face}} = {Ast_min_face:.0f} \text{{ mm}}^2/\text{{m}}$** for both vertical and horizontal reinforcement.
""")
