import math
import json
import base64
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# 1. CONSTANTS, UTILITIES, and SECTION DATACLASS
# -----------------------------
# Define all constants and simple utility functions outside the main class to keep them global.
ES = 200000.0  # MPa (N/mm2) - Young's Modulus of Steel
EPS_CU = 0.0035  # Ultimate concrete compressive strain

def bar_area(dia_mm: float) -> float:
    return math.pi * (dia_mm ** 2) / 4.0

def kN(value: float) -> str:
    return f"{value / 1e3:.1f}"

def kNm(value: float) -> str:
    return f"{value / 1e6:.1f}"

def effective_length_factor(restraint: str) -> float:
    if restraint == "Fixed-Fixed": return 0.65
    if restraint == "Fixed-Pinned": return 0.8
    if restraint == "Pinned-Pinned": return 1.0
    if restraint == "Fixed-Free (cantilever)": return 2.0
    return 1.0

def Ec_from_fck(fck: float) -> float:
    return 5000.0 * math.sqrt(max(fck, 1e-6))

@dataclass
class Section:
    b: float  
    D: float  
    cover: float  
    bars: List[Tuple[float, float, float]]
    tie_dia: float = 8.0 

    @property
    def Ag(self) -> float: return self.b * self.D
    @property
    def Ic_x(self) -> float: return self.b * self.D**3 / 12.0
    @property
    def Ic_y(self) -> float: return self.D * self.b**3 / 12.0
    @property
    def rx(self) -> float: return math.sqrt(self.Ic_x / self.Ag)
    @property
    def ry(self) -> float: return math.sqrt(self.Ic_y / self.Ag)
    @property
    def As_long(self) -> float: return sum(bar_area(dia) for _, _, dia in self.bars)

# -----------------------------
# 2. CORE ENGINEERING LOGIC (CLASS)
# -----------------------------

class ColumnDesigner:
    """Encapsulates all complex logic to minimize AST parser confusion."""

    def __init__(self, state):
        self.state = state
        self.section = self._load_section_and_bars()

    def _linspace_points(self, a: float, c: float, n: int) -> List[float]:
        if n <= 0: return []
        if n == 1: return [a + (c - a) / 2.0]
        return [a + i * (c - a) / (n - 1) for i in range(n)]

    def _generate_bar_layout(self) -> List[Tuple[float, float, float]]:
        b, D, cover = self.state["b"], self.state["D"], self.state["cover"]
        n_top, n_bot, n_left, n_right = self.state["n_top"], self.state["n_bot"], self.state["n_left"], self.state["n_right"]
        dia_top, dia_bot, dia_side = self.state["dia_top"], self.state["dia_bot"], self.state["dia_side"]
        
        bars_list = []
        corner_dia = max(dia_top, dia_bot, dia_side)
        dx_corner = cover + corner_dia / 2.0

        # Top and Bottom Rows
        if n_top > 0:
            x_coords = self._linspace_points(dx_corner, b - dx_corner, n_top)
            for x in x_coords: bars_list.append((x, D - dx_corner, dia_top))

        if n_bot > 0:
            x_coords = self._linspace_points(dx_corner, b - dx_corner, n_bot)
            for x in x_coords: bars_list.append((x, dx_corner, dia_bot))
                
        # Left and Right Columns (Intermediate bars)
        y_start = dx_corner
        y_end = D - dx_corner

        y_coords_l = self._linspace_points(y_start, y_end, n_left)
        for y in y_coords_l: bars_list.append((dx_corner, y, dia_side))

        y_coords_r = self._linspace_points(y_start, y_end, n_right)
        for y in y_coords_r: bars_list.append((b - dx_corner, y, dia_side))

        # Handle unique locations and max diameter
        unique_locations = {}
        for x, y, dia in bars_list:
            loc = (round(x, 4), round(y, 4))
            if loc not in unique_locations or dia > unique_locations[loc][2]:
                unique_locations[loc] = (x, y, dia)

        return sorted([v for v in unique_locations.values()], key=lambda x: (x[1], x[0]))

    def _load_section_and_bars(self):
        bars = self._generate_bar_layout()
        self.state["As_long"] = sum(bar_area(dia) for _, _, dia in bars)
        self.state["bars"] = bars
        return Section(b=self.state["b"], D=self.state["D"], cover=self.state["cover"], bars=bars, tie_dia=self.state["tie_dia"])

    def _forces_and_moment(self, c: float, axis: str):
        fck, fy = self.state["fck"], self.state["fy"]
        dimension = self.section.D if axis == 'x' else self.section.b
        
        Cc = 0.36 * fck * dimension * min(c, dimension)
        arm_Cc = 0.5 * dimension - 0.42 * min(c, dimension)
        Mc = Cc * arm_Cc
        Fs, Ms = 0.0, 0.0
        
        for (x_abs, y_abs, dia) in self.section.bars:
            As = bar_area(dia)
            if axis == 'x': y = self.section.D - y_abs
            else: y = x_abs 
            
            strain = EPS_CU * (1.0 - (y / max(c, 1e-6)))
            stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
            force = stress * As
            z = 0.5 * dimension - y 
            
            Fs += force
            Ms += force * z

        N_res = Cc + Fs
        M_res = Mc + Ms
        return N_res, M_res

    def _uniaxial_capacity_Mu_for_Pu(self, Pu: float, axis: str) -> float:
        dimension = self.section.D if axis == 'x' else self.section.b
        
        target = Pu
        c_min = 0.05 * dimension
        c_max = 1.50 * dimension

        cL, cR = c_min, c_max
        NL, ML = self._forces_and_moment(cL, axis)
        NR, MR = self._forces_and_moment(cR, axis)

        if target <= NL: return ML
        if target >= NR: return MR

        for _ in range(60):
            cm = 0.5 * (cL + cR)
            Nm, Mm = self._forces_and_moment(cm, axis)
            if abs(Nm - target) < 1.0: return float(Mm)
            
            if (NL - target) * (Nm - target) <= 0: cR, NR, MR = cm, Nm, Mm
            else: cL, NL, ML = cm, Nm, Mm
        
        return float(0.5 * (ML + MR))

    def calculate_slenderness_and_magnification(self):
        le_x = self.state["kx"] * self.state["storey_clear"]
        le_y = self.state["ky"] * self.state["storey_clear"]
        lam_x = le_x / max(self.section.rx, 1e-6)
        lam_y = le_y / max(self.section.ry, 1e-6)
        short_x = lam_x <= 12.0
        short_y = lam_y <= 12.0
        self.state.update(dict(le_x=le_x, le_y=le_y, lam_x=lam_x, lam_y=lam_y, short_x=short_x, short_y=short_y))

        delta_x = moment_magnifier(self.state["Pu"], le_x, self.state["fck"], self.section.Ic_x, Cm=0.85, sway=self.state["sway"]) if not short_x else 1.0
        delta_y = moment_magnifier(self.state["Pu"], le_y, self.state["fck"], self.section.Ic_y, Cm=0.85, sway=self.state["sway"]) if not short_y else 1.0

        Mux_eff = self.state["Mux"] * delta_x
        Muy_eff = self.state["Muy"] * delta_y
        self.state.update(dict(Mux_eff=Mux_eff, Muy_eff=Muy_eff, delta_x=delta_x, delta_y=delta_y))
        
    def calculate_biaxial_utilization(self):
        Pu, Mux_eff, Muy_eff, alpha = self.state["Pu"], self.state["Mux_eff"], self.state["Muy_eff"], self.state["alpha"]
        fck, fy = self.state["fck"], self.state["fy"]
        
        Mux_lim = self._uniaxial_capacity_Mu_for_Pu(Pu, axis='x')
        Muy_lim = self._uniaxial_capacity_Mu_for_Pu(Pu, axis='y')
        
        Mux_lim = max(Mux_lim, 1e-3) if abs(Mux_eff) > 1e-3 else Mux_lim
        Muy_lim = max(Muy_lim, 1e-3) if abs(Muy_eff) > 1e-3 else Muy_lim
            
        Rx = (abs(Mux_eff) / Mux_lim) ** alpha
        Ry = (abs(Muy_eff) / Muy_lim) ** alpha
        util = Rx + Ry
        self.state.update(dict(Mux_lim=Mux_lim, Muy_lim=Muy_lim, util=util))
        
    def calculate_shear_design(self):
        max_long_dia = max([bar[2] for bar in self.state["bars"]], default=16.0)
        d_eff = self.state["D"] - (self.state["cover"] + 0.5 * max_long_dia) 
        
        Ag, fck, Pu, fy = self.section.Ag, self.state["fck"], self.state["Pu"], self.state["fy"]
        
        # Vc calculation
        phiN_calc = 1.0 + (Pu / max(1.0, (0.25 * fck * Ag)))
        phiN = float(np.clip(phiN_calc, 0.5, 1.5))
        tau_c = 0.62 * math.sqrt(fck) / 1.0 
        Vc = tau_c * self.state["b"] * d_eff * phiN
        Vus = max(0.0, self.state["Vu"] - Vc)
        self.state.update(dict(Vc=Vc, Vus=Vus, phiN=phiN, d_eff=d_eff))
        
        # Tie spacing required for shear
        Asv = self.state["legs"] * bar_area(self.state["tie_dia"])
        s_required = (0.87 * fy * Asv * d_eff) / max(Vus, 1e-6) if Vus > 0 else 300.0
        self.state["s_required"] = s_required

        # Tie spacing max limits (detailing)
        s_lim1 = 16.0 * max_long_dia
        s_lim2 = min(self.state["b"], self.state["D"]) 
        s_lim3 = 300.0
        s_cap = min(s_lim1, s_lim2, s_lim3)
        self.state["s_governing_tie"] = min(s_cap, s_required)
        self.state["s_cap_detailing"] = s_cap

    def _plot_pm_curve(self, axis: str):
        dimension = self.section.D if axis == 'x' else self.section.b
        
        # Generate points for the curve
        cmin = 0.01 * dimension
        cmax = 1.50 * dimension
        cs = np.linspace(cmin, cmax, 120)
        P_list, M_list = [], []
        for c in cs:
            N, M = self._forces_and_moment(c, axis)
            P_list.append(N)
            M_list.append(abs(M))

        df = pd.DataFrame({'P (kN)': np.array(P_list)/1e3, f'M_{axis} (kNm)': np.array(M_list)/1e6})
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['P (kN)'], y=df[f'M_{axis} (kNm)'], mode='lines', name='Capacity Envelope (M_lim)', line=dict(color='#0a66c2', width=3)))
        
        # Add demand point
        Pu_kNm = self.state["Pu"] / 1e3
        Mu_eff_kNm = abs(self.state[f"M{axis}x_eff"]) / 1e6
        fig.add_trace(go.Scatter(x=[Pu_kNm], y=[Mu_eff_kNm], mode='markers', name='Demand (Pu, Mu_eff)', marker=dict(symbol='x', size=12, color='red', line=dict(width=2, color='red'))))
        
        fig.update_layout(title=f'P–M Capacity Envelope (Axis: {axis.upper()})', xaxis_title='Axial Load P (kN)', yaxis_title=f'Moment M_{axis} (kNm)', hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='white')
        return fig

# ----------------------------
# 3. PLOTLY AND DATA UTILITY FUNCTIONS (Defined outside class)
# ----------------------------

def plotly_cross_section(section: Section) -> go.Figure:
    b, D = section.b, section.D
    cover = section.cover
    fig = go.Figure()

    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=D, line=dict(color="black", width=2), fillcolor="rgba(240, 240, 240, 0.8)"))
    fig.add_shape(type="rect", x0=cover, y0=cover, x1=b - cover, y1=D - cover, line=dict(color="gray", width=1, dash="dot"), fillcolor="rgba(0,0,0,0)"))

    bar_x, bar_y, bar_size, bar_text = [], [], [], []
    for x_geom, y_geom, dia in section.bars:
        bar_x.append(x_geom)
        bar_y.append(y_geom) 
        bar_size.append(dia * 2) 
        bar_text.append(f"Ø{dia:.0f} mm ({bar_area(dia):.0f} mm²)")

    fig.add_trace(go.Scatter(
        x=bar_x, y=bar_y, mode='markers', name='Rebars',
        marker=dict(size=bar_size, sizemode='diameter', color='#0a66c2', line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate='Size: %{text}<extra></extra>', text=bar_text
    ))

    max_dim = max(b, D)
    fig.update_layout(
        title=f"Cross Section (b={b:.0f}, D={D:.0f} mm)",
        xaxis=dict(range=[-0.1 * max_dim, b + 0.1 * max_dim], showgrid=False, zeroline=False, title="Width (mm)"),
        yaxis=dict(range=[-0.1 * max_dim, D + 0.1 * max_dim], showgrid=False, zeroline=False, title="Depth (mm)"),
        width=400, height=400 * D / b if b != 0 else 400,
        showlegend=False, hovermode="closest", plot_bgcolor='white',
        yaxis_scaleanchor="x", yaxis_scaleratio=1
    )
    return fig

def plotly_elevation(state: dict, section: Section) -> go.Figure:
    b = section.b
    lo = state["storey_clear"]
    le = state["kx"] * lo
    
    fig = go.Figure()
    fig.add_shape(type="rect", x0=-b/2, y0=0, x1=b/2, y1=lo, line=dict(color="black", width=2), fillcolor="rgba(240, 240, 240, 0.8)"))

    x_rebars = [-b/2 + section.cover, b/2 - section.cover]
    for x_r in x_rebars:
        fig.add_trace(go.Scatter(x=[x_r, x_r], y=[0, lo], mode='lines', name='Rebar', line=dict(color='#0a66c2', width=2), showlegend=False))

    s_prov = state.get("tie_spacing", 150.0)
    n_ties = int(lo / s_prov)
    if n_ties > 0:
        tie_y = np.linspace(s_prov / 2, lo - s_prov / 2, n_ties)
        for y_t in tie_y:
            fig.add_trace(go.Scatter(x=[-b/2, b/2], y=[y_t, y_t], mode='lines', name='Tie', line=dict(color='#888', width=1), showlegend=False))
    
    fig.add_shape(type="line", x0=b/2 + 20, y0=lo/2 - le/2, x1=b/2 + 20, y1=lo/2 + le/2, line=dict(color="red", width=3, dash="dash"))
    fig.add_annotation(x=b/2 + 20, y=lo/2 + le/2 + 50, text=f"$l_e = {le:.0f} mm$", showarrow=False, font=dict(color="red", size=14))
    
    if n_ties > 0:
        fig.add_annotation(
            x=-b/2 - 20, y=s_prov / 2 + 50,
            text=f"Tie Spacing $s = {s_prov:.0f} mm$", showarrow=True, arrowhead=2, ax=-80, ay=50,
            bgcolor="white", font=dict(size=12)
        )

    max_x = b/2 + 100
    fig.update_layout(
        title=f"Column Elevation ($l_0 = {lo:.0f} mm$)",
        xaxis=dict(range=[-max_x, max_x], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.1 * lo, lo + 0.1 * lo], showgrid=False, zeroline=False, title="Height (mm)"),
        width=450, height=600, showlegend=False, plot_bgcolor='white',
    )
    return fig

def to_json_serializable(state: dict) -> dict:
    safe_state = {}
    for key, value in state.items():
        if isinstance(value, float): safe_state[key] = round(value, 6)
        elif isinstance(value, np.float64): safe_state[key] = round(float(value), 6)
        elif isinstance(value, (list, tuple)) and all(isinstance(v, tuple) for v in value):
            safe_state[key] = [list(item) for item in value]
        else: safe_state[key] = value
    return safe_state

def get_json_download_link(data_dict: dict, filename: str) -> str:
    json_str = json.dumps(data_dict, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">💾 Download State as JSON</a>'
    return href

def initialize_state():
    if "state" not in st.session_state:
        st.session_state.state = {
            "b": 450.0, "D": 600.0, "cover": 40.0, "fck": 30.0, "fy": 500.0,
            "Pu": 1200e3, "Mux": 120e6, "Muy": 80e6, "Vu": 150e3,
            "storey_clear": 3200.0, "kx": 1.0, "ky": 1.0, "restraint": "Pinned-Pinned", "sway": False,
            "n_top": 3, "n_bot": 3, "n_left": 2, "n_right": 2, 
            "dia_top": 16.0, "dia_bot": 16.0, "dia_side": 12.0, "tie_dia": 8.0, "tie_spacing": 150.0,
            "alpha": 1.0, "bars": [], "legs": 2
        }

# ----------------------------
# MAIN APP EXECUTION
# ----------------------------

def main():
    st.set_page_config(page_title="RCC Column (Biaxial) Designer", layout="wide")
    st.title("🧱 RCC Column Design — Biaxial Moments ± Axial ± Shear (IS 456/13920)")
    st.markdown("---")

    initialize_state()
    state = st.session_state.state

    # Instantiate the designer class
    designer = ColumnDesigner(state)
    section = designer.section # Get the current section definition
    
    # -----------------------------
    # 0. Data Management
    # -----------------------------
    st.header("🗄️ Data Management (Import / Export)")
    c_up, c_down = st.columns(2)

    with c_up:
        uploaded_file = st.file_uploader("Upload State (JSON)", type="json")
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                if 'bars' in data and isinstance(data['bars'], list): data['bars'] = [tuple(item) for item in data['bars']]
                st.session_state.state.update(data)
                st.rerun() 
            except Exception as e:
                st.error(f"Error loading JSON: {e}")

    with c_down:
        if state.get("bars"): 
            st.markdown("**Export Current Design State**")
            safe_state = to_json_serializable(state)
            st.markdown(get_json_download_link(safe_state, "column_design_state.json"), unsafe_allow_html=True)
        else:
            st.info("Define inputs first to enable data export.")
    st.markdown("---")

    # -----------------------------
    # 1. Geometry, Materials, and Loads
    # -----------------------------
    st.header("1️⃣ Column Geometry, Materials, and Loads")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        state["b"] = st.number_input("Width $b$ (mm)", 200.0, 2000.0, state["b"], 25.0, key='in_b')
        state["fck"] = st.number_input("$f_{ck}$ (MPa, M-Grade)", 20.0, 80.0, state["fck"], 1.0, key='in_fck')
        state["sway"] = st.checkbox("Sway Frame? (Yes/No)", value=state["sway"], key='in_sway')
    with c2:
        state["D"] = st.number_input("Depth $D$ (mm)", 200.0, 3000.0, state["D"], 25.0, key='in_D')
        state["fy"] = st.number_input("$f_{y}$ (MPa, Fe-Grade)", 415.0, 600.0, state["fy"], 5.0, key='in_fy')
        restraint_options = ["Fixed-Fixed", "Fixed-Pinned", "Pinned-Pinned", "Fixed-Free (cantilever)"]
        state["restraint"] = st.selectbox("End Restraint", restraint_options, index=restraint_options.index(state["restraint"]), key='in_restraint')
    with c3:
        state["cover"] = st.number_input("Clear Cover (mm)", 20.0, 75.0, state["cover"], 5.0, key='in_cover')
        state["storey_clear"] = st.number_input("Clear Storey Height $l_0$ (mm)", 2000.0, 6000.0, state["storey_clear"], 50.0, key='in_l0')
        k_factor = effective_length_factor(state["restraint"])
        state["kx"], state["ky"] = k_factor, k_factor
        st.markdown(f"Effective Length Factor **$k_x = k_y$**: **{k_factor:.2f}**")
    with c4:
        state["Pu"] = st.number_input("Axial Load $P_u$ (kN, +comp)", -3000.0, 6000.0, state["Pu"] / 1e3, 10.0, key='in_Pu') * 1e3
        state["Mux"] = st.number_input("Moment $M_{ux}$ (kNm)", -2000.0, 2000.0, state["Mux"] / 1e6, 5.0, key='in_Mux') * 1e6
        state["Muy"] = st.number_input("Moment $M_{uy}$ (kNm)", -2000.0, 2000.0, state["Muy"] / 1e6, 5.0, key='in_Muy') * 1e6
        state["Vu"] = st.number_input("Shear $V_u$ (kN)", 0.0, 5000.0, state["Vu"] / 1e3, 5.0, key='in_Vu') * 1e3

    # --- Re-calculate dependent properties ---
    designer = ColumnDesigner(state)
    designer.calculate_slenderness_and_magnification()
    section = designer.section

    st.markdown("---")

    # -----------------------------
    # 2 & 3. Slenderness & Moment Magnification
    # -----------------------------
    st.header("2️⃣/3️⃣ Slenderness and Second-Order Moment Magnification")

    c_slender, c_magnified = st.columns(2)
    with c_slender:
        st.markdown("### Slenderness Check (IS 456: Cl. 25.1)")
        st.metric(f"Slenderness $\lambda_x = l_{{e,x}}/r_x$", f"{state['lam_x']:.1f}")
        st.metric(f"Slenderness $\lambda_y = l_{{e,y}}/r_y$", f"{state['lam_y']:.1f}")
        st.info(f"Classification → About x: {'**Short**' if state['short_x'] else '**Slender**'}, About y: {'**Short**' if state['short_y'] else '**Slender**'}")
        
    with c_magnified:
        st.markdown("### Magnified Moments $M_{u}'$")
        st.metric("Magnifier $\\delta_x$", f"{state['delta_x']:.2f}")
        st.metric("Magnifier $\\delta_y$", f"{state['delta_y']:.2f}")
        st.metric("Magnified $M_{ux}'$ (kNm)", f"{kNm(state['Mux_eff'])}")
        st.metric("Magnified $M_{uy}'$ (kNm)", f"{kNm(state['Muy_eff'])}")

    st.markdown("---")
    
    # -----------------------------
    # 6. Detailing and Final Check (Rebar Adjustment)
    # -----------------------------
    st.header("6️⃣ Longitudinal Rebar Adjustment & Detailing Checks")
    
    cL1, cL2, cL3, cL4, cL5 = st.columns(5)
    bar_options = [12.0, 16.0, 20.0, 25.0, 28.0, 32.0]
    
    with cL1:
        state["n_top"] = st.number_input("Top row bars ($n_{top}$)", 0, 10, state["n_top"], 1, key='in_ntop')
        state["dia_top"] = st.selectbox("Top bar $\\phi$ (mm)", bar_options, index=bar_options.index(state["dia_top"]), key='in_dtop')
    with cL2:
        state["n_bot"] = st.number_input("Bottom row bars ($n_{bot}$)", 0, 10, state["n_bot"], 1, key='in_nbot')
        state["dia_bot"] = st.selectbox("Bottom bar $\\phi$ (mm)", bar_options, index=bar_options.index(state["dia_bot"]), key='in_dbot')
    with cL3:
        state["n_left"] = st.number_input("Left column bars ($n_{left}$)", 0, 10, state["n_left"], 1, key='in_nleft')
        state["dia_side"] = st.selectbox("Side bar $\\phi$ (mm)", bar_options[:-1], index=bar_options[:-1].index(state["dia_side"]), key='in_dside')
    with cL4:
        state["n_right"] = st.number_input("Right column bars ($n_{right}$)", 0, 10, state["n_right"], 1, key='in_nright')
        leg_options_map = {"2-legged": 2, "4-legged": 4, "6-legged": 6}
        current_legs_str = f"{state.get('legs', 2)}-legged"
        selected_legs_str = st.selectbox("Tie Legs ($A_{sv}$)", list(leg_options_map.keys()), index=list(leg_options_map.keys()).index(current_legs_str), key='in_legs')
        state["legs"] = leg_options_map[selected_legs_str]
    with cL5:
        tie_options = [6.0, 8.0, 10.0, 12.0]
        state["tie_dia"] = st.selectbox("Tie $\\phi$ (mm)", tie_options, index=tie_options.index(state["tie_dia"]), key='in_tdia_shear')
        state["tie_spacing"] = st.number_input("Tie Spacing $s$ (mm)", 50.0, 300.0, state["tie_spacing"], 5.0, key='in_ts')

    # Update Section and Run Calculations again
    designer = ColumnDesigner(state)
    designer.calculate_slenderness_and_magnification()
    designer.calculate_biaxial_utilization()
    designer.calculate_shear_design()
    section = designer.section

    c_plot_cs, c_plot_elev = st.columns(2)
    with c_plot_cs:
        st.plotly_chart(plotly_cross_section(section), use_container_width=True)
    with c_plot_elev:
        st.plotly_chart(plotly_elevation(state, section), use_container_width=True)
    
    # Detailing Checks
    As_long = section.As_long
    rho_long = 100.0 * As_long / section.Ag
    As_min = 0.008 * section.Ag 
    As_max = 0.06 * section.Ag

    c_l1, c_l2, c_l3 = st.columns(3)
    with c_l1: st.metric("$A_{st}$ Provided (mm²)", f"{As_long:.0f} ($\\rho_l$ = {rho_long:.2f}%)")
    with c_l2: st.metric("$A_{st}$ Minimum (0.8%)", f"{As_min:.0f} mm²")
    with c_l3: st.metric("$A_{st}$ Maximum (6.0%)", f"{As_max:.0f} mm²")
    
    if As_long < As_min: st.error("⚠️ **FAIL**: Provided steel is less than the 0.8% minimum.")
    elif As_long > As_max: st.error("⚠️ **FAIL**: Provided steel exceeds the 6.0% maximum.")
    else: st.success("✅ Minimum/Maximum steel limits are met.")

    st.markdown("---")

    # -----------------------------
    # 4. Interaction (Biaxial)
    # -----------------------------
    st.header("4️⃣ Biaxial Interaction and Capacity Check (Strength)")
    
    st.markdown("### Bresler's Interaction Formula (IS 456)")
    state["alpha"] = st.slider("Interaction Exponent $\\alpha$", 0.8, 2.0, state["alpha"], 0.1, key='in_alpha_check')
    st.latex(r"\left(\frac{|M_{ux}'|}{M_{ux,lim}}\right)^{\alpha} + \left(\frac{|M_{uy}'|}{M_{uy,lim}}\right)^{\alpha} \le 1.0")

    util, Mux_lim, Muy_lim = state["util"], state["Mux_lim"], state["Muy_lim"]

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Uniaxial $M_{ux,lim}$ (kNm)", f"{kNm(Mux_lim)}")
    with c2: st.metric("Uniaxial $M_{uy,lim}$ (kNm)", f"{kNm(Muy_lim)}")
    with c3: st.metric("Utilization (Σ ≤ 1)", f"**{util:.2f}**")
        
    if util <= 1.0: st.success("✅ Biaxial interaction PASS.")
    else: st.error("❌ Biaxial interaction FAIL — **Increase steel area** in Section 6.")

    # P-M Interaction Curves (Plotly)
    st.markdown("#### Interactive P–M Interaction Curves (Capacity vs. Demand)")
    colx, coly = st.columns(2)
    with colx:
        st.plotly_chart(designer._plot_pm_curve('x'), use_container_width=True)
    with coly:
        st.plotly_chart(designer._plot_pm_curve('y'), use_container_width=True)
    
    st.markdown("---")


    # -----------------------------
    # 5. Shear Design
    # -----------------------------
    st.header("5️⃣ Shear Design (IS 456 Cl. 40)")
    
    Vc, Vus, s_governing_tie, s_cap = state["Vc"], state["Vus"], state["s_governing_tie"], state["s_cap_detailing"]
    s_prov = state["tie_spacing"]

    c_met1, c_met2 = st.columns(2)
    with c_met1: st.metric("Concrete Capacity $V_c$ (kN)", f"{kN(Vc)}")
    with c_met2: st.metric("Excess Shear $V_{us}$ (kN)", f"{kN(Vus)}")
        
    if Vus > 0:
        st.warning(f"$V_{{us}}$ of {kN(Vus)} kN requires spacing $\\le$ **{state['s_required']:.0f} mm**.")
    else:
        st.info("Vu $\\le V_c \\rightarrow$ Provide minimum transverse reinforcement.")
    
    st.markdown("### Transverse Reinforcement Spacing Check")
    st.write(f"Tie Spacing Max (IS 456 Detailing) $\\le$ **{s_cap:.0f} mm**.")
    
    if s_prov <= s_governing_tie:
        st.success(f"Tie Spacing PASS: {s_prov:.0f} mm $\\le$ {s_governing_tie:.0f} mm.")
    else:
        st.error(f"Tie Spacing FAIL: Must provide spacing $\\le$ **{s_governing_tie:.0f} mm**.")

    st.markdown("---")

    # -----------------------------
    # 7. Output Summary (Printable Table)
    # -----------------------------
    st.header("7️⃣ Printable Output Summary")
    
    As_governing = max(As_min, As_long * (state.get("util", 1.0) ** (1/state.get("alpha", 1.0))))
    
    out = {
        "b (mm)": state["b"], "D (mm)": state["D"], "fck (MPa)": state["fck"], "fy (MPa)": state["fy"],
        "Pu (kN)": state["Pu"] / 1e3, "Mux′ (kNm)": kNm(state["Mux_eff"]), "Muy′ (kNm)": kNm(state["Muy_eff"]), "Vu (kN)": state["Vu"] / 1e3,
        "--- STRENGTH RESULTS ---": "---",
        "Slenderness λx": f"{state['lam_x']:.1f}", "Slenderness λy": f"{state['lam_y']:.1f}",
        "Moment Magnifier δx": f"{state['delta_x']:.2f}", 
        "Uniaxial Capacity Mux,lim (kNm)": kNm(state["Mux_lim"]),
        "Biaxial Utilization (≤1)": f"{state['util']:.2f}",
        "Concrete Shear Vc (kN)": state["Vc"] / 1e3, "Vus (kN)": state["Vus"] / 1e3,
        "--- DETAILING RESULTS ---": "---",
        "Ast Provided (mm²)": f"{As_long:.0f}",
        "Ast Governing Required (mm²)": f"{As_governing:.0f}",
        "Tie Spacing Provided (mm)": f"{state['tie_spacing']:.0f}",
        "Tie Spacing Governing Req (mm)": f"{s_governing_tie:.0f}",
    }

    df_out = pd.DataFrame({"Parameter": list(out.keys()), "Value": list(out.values())})
    st.dataframe(df_out, use_container_width=True)
    
    st.caption("Notes: All calculations follow IS 456 (2000) and IS 13920 (2016) principles.")

if __name__ == "__main__":
    main()
