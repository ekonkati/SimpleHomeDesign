# app.py — RCC Column (Biaxial) Design Canvas
# Single-page, narration-rich, Plotly visuals, printable
# Assumptions: IS 456:2000 + IS 13920:2016 checks (switchable)
# Units: N, mm, MPa, kN·m

import math
import json
import base64
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ------------------------- 0) PAGE & STYLES -------------------------
st.set_page_config(page_title="RCC Column (Biaxial) Designer", layout="wide")

st.markdown("""
<style>
@media print {
  header, .stToolbar, .stAppDeployButton, .stActionButton, .stDownloadButton, footer, .stSidebar { display: none !important; }
  .block-container { padding: 0.6cm 1.0cm !important; max-width: 100% !important; }
  .print-break { page-break-before: always; }
}
.highlight {
  background: #fff8e1; /* lighter yellow */
  border: 2px solid #ffcc80;
  border-radius: 12px;
  padding: 10px 15px;
  margin: 10px 0 20px 0;
}
.highlight .stSelectbox, .highlight .stNumberInput, .highlight .stSlider, .highlight .stTextInput {
  background: #fffceb !important;
}
.js-plotly-plot, .plotly, .user-select-none { margin: 0 !important; }
.badge {padding:4px 8px;border-radius:8px;background:#eef7ee;color:#0a7a0a;font-weight:700}
.badge.bad {background:#fdecee;color:#b00020}
</style>
""", unsafe_allow_html=True)

# ------------------------- 1) CONSTANTS & UTILS -------------------------
ES = 200000.0        # MPa
EPS_CU = 0.0035      # ultimate concrete compressive strain

def bar_area(dia_mm: float) -> float:
    return math.pi * (dia_mm**2) / 4.0

def kN(value: float) -> str:
    return f"{value / 1e3:.1f}"

def kNm(value: float) -> str:
    return f"{value / 1e6:.1f}"

def effective_length_factor(restraint: str) -> float:
    return {
        "Fixed-Fixed": 0.65,
        "Fixed-Pinned": 0.80,
        "Pinned-Pinned": 1.00,
        "Fixed-Free (cantilever)": 2.00
    }.get(restraint, 1.00)

def moment_magnifier(Pu: float, le_mm: float, fck: float, Ic: float, Cm: float = 0.85, sway: bool = False) -> float:
    Ec = 5000.0 * math.sqrt(max(fck, 1e-6))
    # cracked EI ~ 0.4 Ec Ic (IS 456 §39.7 hint)
    Pcr = (math.pi**2) * 0.4 * Ec * Ic / (le_mm**2 + 1e-9)
    if Pcr <= Pu:
        return 10.0  # avoid blow-up
    delta = 1.0 / max(1e-6, (1.0 - Pu / Pcr))
    if not sway:
        delta = max(1.0, Cm * delta)
        return float(np.clip(delta, 1.0, 2.5))
    return float(np.clip(delta, 1.0, 5.0))

def to_json_serializable(state: dict) -> dict:
    safe_state = {}
    for key, value in state.items():
        if isinstance(value, float): safe_state[key] = round(value, 6)
        elif isinstance(value, np.floating): safe_state[key] = round(float(value), 6)
        elif isinstance(value, (list, tuple)) and all(isinstance(v, tuple) for v in value):
            safe_state[key] = [list(item) for item in value]
        else: safe_state[key] = value
    return safe_state

def get_json_download_link(data_dict: dict, filename: str) -> str:
    json_str = json.dumps(data_dict, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    return f'<a href="data:file/json;base64,{b64}" download="{filename}">💾 Download Design State</a>'

# Unique chart keys to avoid StreamlitDuplicateElementId
def chart(fig, *, key: str):
    st.plotly_chart(fig, use_container_width=True, key=key)

def kkey(*parts) -> str:
    return "chart_" + "_".join(str(p) for p in parts)

# ------------------------- 2) REBAR & CAPACITY LOGIC -------------------------
def _linspace_points(a: float, c: float, n: int) -> List[float]:
    if n <= 0: return []
    if n == 1: return [a + (c - a) / 2.0]
    return [a + i * (c - a) / (n - 1) for i in range(n)]

def _generate_bar_layout(b: float, D: float, cover: float, state: dict) -> List[Tuple[float, float, float]]:
    """
    Perimeter bars with *no corner double counting*:
    - Top & bottom rows include corners.
    - Side rows EXCLUDE corners (so plot count matches inputs exactly).
    """
    n_top, n_bot, n_left, n_right = state["n_top"], state["n_bot"], state["n_left"], state["n_right"]
    dia_top, dia_bot, dia_side = state["dia_top"], state["dia_bot"], state["dia_side"]

    bars_list = []
    max_dia = max(dia_top, dia_bot, dia_side)
    dx_corner = cover + state["tie_dia"] + max_dia / 2.0
    dx_corner = min(dx_corner, min(b, D)/2 - 5.0)

    x0, x1 = dx_corner, b - dx_corner
    y0, y1 = dx_corner, D - dx_corner

    # Top (includes corners)
    for x in _linspace_points(x0, x1, n_top):
        bars_list.append((x, y1, dia_top))
    # Bottom (includes corners)
    for x in _linspace_points(x0, x1, n_bot):
        bars_list.append((x, y0, dia_bot))
    # Left side EXCLUDING corners
    if n_left > 0:
        ys = _linspace_points(y0, y1, n_left + 2)[1:-1]  # remove ends
        for y in ys:
            bars_list.append((x0, y, dia_side))
    # Right side EXCLUDING corners
    if n_right > 0:
        ys = _linspace_points(y0, y1, n_right + 2)[1:-1]  # remove ends
        for y in ys:
            bars_list.append((x1, y, dia_side))

    return bars_list  # no dedup needed now

def _uniaxial_capacity_Mu_for_Pu(b: float, D: float, bars: List[Tuple[float, float, float]],
                                 fck: float, fy: float, Pu: float, axis: str) -> float:
    """
    Strain-compatibility (teaching-grade) uniaxial Mu_lim for a given Pu.
    axis='x' means bending about x (compression depth = D, width=b).
    axis='y' means bending about y (compression depth = b, width=D).
    """
    depth = D if axis == 'x' else b    # direction of compression block depth (xu)
    width = b if axis == 'x' else D    # block width

    def forces_and_moment(c: float):
        xu = min(c, depth)
        Cc = 0.36 * fck * width * xu              # N
        arm_Cc = 0.5 * depth - 0.42 * xu          # mm
        Mc = Cc * arm_Cc

        Fs, Ms = 0.0, 0.0
        for (x_abs, y_abs, dia) in bars:
            As = bar_area(dia)
            y = (D - y_abs) if axis == 'x' else x_abs  # distance from compression edge
            strain = EPS_CU * (1.0 - (y / max(c, 1e-9)))
            stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
            force = stress * As
            z = 0.5 * depth - y
            Fs += force
            Ms += force * z
        return Cc + Fs, Mc + Ms  # N, N·mm

    target = Pu
    cL, cR = 0.01 * depth, 1.50 * depth
    NL, ML = forces_and_moment(cL)
    NR, MR = forces_and_moment(cR)

    if target <= NL: return float(ML)
    if target >= NR: return float(MR)

    for _ in range(60):
        cm = 0.5 * (cL + cR)
        Nm, Mm = forces_and_moment(cm)
        if abs(Nm - target) < 1.0:
            return float(Mm)
        if (NL - target) * (Nm - target) <= 0:
            cR, NR, MR = cm, Nm, Mm
        else:
            cL, NL, ML = cm, Nm, Mm
    return float(0.5 * (ML + MR))

def biaxial_utilization(b: float, D: float, bars: List[Tuple[float, float, float]],
                        fck: float, fy: float, Pu: float,
                        Mux_eff: float, Muy_eff: float, alpha: float):
    Mux_lim = _uniaxial_capacity_Mu_for_Pu(b, D, bars, fck, fy, Pu, axis='x')
    Muy_lim = _uniaxial_capacity_Mu_for_Pu(b, D, bars, fck, fy, Pu, axis='y')
    Mux_lim = max(Mux_lim, 1e-3) if abs(Mux_eff) > 1e-3 else Mux_lim
    Muy_lim = max(Muy_lim, 1e-3) if abs(Muy_eff) > 1e-3 else Muy_lim
    Rx = (abs(Mux_eff) / Mux_lim) ** alpha
    Ry = (abs(Muy_eff) / Muy_lim) ** alpha
    util = Rx + Ry
    return util, Mux_lim, Muy_lim

# ------------------------- 3) PLOTLY FIGURES -------------------------
def plotly_cross_section(b: float, D: float, cover: float, bars: List[Tuple[float, float, float]],
                         tie_dia: float, tie_spacing: float) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=D,
                  line=dict(color="black", width=2),
                  fillcolor="rgba(240,240,240,0.8)")
    eff_cover = cover + tie_dia
    fig.add_shape(type="rect", x0=eff_cover, y0=eff_cover, x1=b-eff_cover, y1=D-eff_cover,
                  line=dict(color="gray", width=1, dash="dot"),
                  fillcolor="rgba(0,0,0,0)")

    xs, ys, sizes, txt = [], [], [], []
    for xg, yg, dia in bars:
        xs.append(xg); ys.append(yg)
        sizes.append(max(6, dia*2))
        txt.append(f"Ø{dia:.0f} mm ({bar_area(dia):.0f} mm²)")
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers', name='Rebars',
        marker=dict(size=sizes, sizemode='diameter', line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate='Size: %{text}<extra></extra>', text=txt))

    fig.add_annotation(x=b/2, y=D+10, text=f"Ties: Ø{int(tie_dia)} @ {int(tie_spacing)} mm",
                       showarrow=False, yanchor="bottom")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title="Depth D (mm)")
    fig.update_xaxes(title="Width b (mm)")
    fig.update_layout(title=f"Cross Section (b={b:.0f} × D={D:.0f} mm)",
                      height=420, showlegend=False, hovermode="closest",
                      plot_bgcolor='white', margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_pm_curve_plotly(b: float, D: float, bars: List[Tuple[float, float, float]],
                         fck: float, fy: float, Pu: float, Mu_eff: float, axis: str):
    depth = D if axis == 'x' else b
    cs = np.linspace(0.01 * depth, 1.50 * depth, 120)
    P_list, M_list = [], []
    for c in cs:
        width = b if axis == 'x' else D
        Cc = 0.36 * fck * width * min(c, depth)
        arm_Cc = 0.5 * depth - 0.42 * min(c, depth)
        Mc = Cc * arm_Cc
        Fs, Ms = 0.0, 0.0
        for (x_abs, y_abs, dia) in bars:
            As = bar_area(dia)
            y = (D - y_abs) if axis == 'x' else x_abs
            strain = EPS_CU * (1.0 - (y / max(c, 1e-9)))
            stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
            force = stress * As
            z = 0.5 * depth - y
            Fs += force; Ms += force * z
        P_list.append(Cc + Fs)
        M_list.append(abs(Mc + Ms))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.array(P_list)/1e3, y=np.array(M_list)/1e6,
        mode='lines', name='Capacity Envelope', line=dict(width=3)))
    fig.add_trace(go.Scatter(
        x=[Pu/1e3], y=[abs(Mu_eff)/1e6], mode='markers',
        name='Design Demand (Pu, Mu_eff)', marker=dict(symbol='x', size=12, color='red', line=dict(width=2))))
    fig.update_layout(title=f'P–M Capacity Envelope (Axis: {axis.upper()})',
                      xaxis_title='Axial Load P (kN)', yaxis_title=f'Moment M_{axis} (kN·m)',
                      hovermode="x unified", plot_bgcolor='white',
                      margin=dict(l=20, r=20, t=40, b=20), height=420)
    return fig

# ------------------------- 4) CORE CALCULATIONS -------------------------
def initialize_state():
    if "state" not in st.session_state:
        st.session_state.state = {
            "b": 450.0, "D": 600.0, "cover": 40.0,
            "fck": 30.0, "fy": 500.0,
            "Pu": 1200e3, "Mux": 120e6, "Muy": 80e6, "Vu": 150e3,
            "storey_clear": 3200.0, "restraint": "Pinned-Pinned", "sway": False,
            "n_top": 3, "n_bot": 3, "n_left": 2, "n_right": 2,
            "dia_top": 16.0, "dia_bot": 16.0, "dia_side": 12.0,
            "tie_dia": 8.0, "tie_spacing": 150.0, "legs": 4,
            "alpha": 1.0, "bars": [], "util": 0.0, "Vc": 0.0, "Vus": 0.0, "Ag": 0.0
        }

def recalculate_properties(state):
    b, D, cover, fck, fy = state["b"], state["D"], state["cover"], state["fck"], state["fy"]
    Pu, Mux, Muy, Vu = state["Pu"], state["Mux"], state["Muy"], state["Vu"]
    lo = state["storey_clear"]

    # (1) Bars & section props
    bars = _generate_bar_layout(b, D, cover, state)
    As_long = sum(bar_area(d) for _, _, d in bars)
    Ag = b * D
    Ic_x = b * D**3 / 12.0
    Ic_y = D * b**3 / 12.0
    rx = math.sqrt(Ic_x / max(Ag, 1e-9))
    ry = math.sqrt(Ic_y / max(Ag, 1e-9))
    state.update({"As_long": As_long, "bars": bars, "Ag": Ag, "Ic_x": Ic_x, "Ic_y": Ic_y})

    # (2) Slenderness & moment magnification
    k_factor = effective_length_factor(state["restraint"])
    le_x, le_y = k_factor * lo, k_factor * lo
    lam_x, lam_y = le_x / max(rx, 1e-9), le_y / max(ry, 1e-9)
    short_x = lam_x <= 12.0
    short_y = lam_y <= 12.0

    emin_x = max(lo/500.0 + D/30.0, 20.0)
    emin_y = max(lo/500.0 + b/30.0, 20.0)
    Mux_base = max(abs(Mux), abs(Pu * emin_x))
    Muy_base = max(abs(Muy), abs(Pu * emin_y))

    delta_x = 1.0 if short_x else moment_magnifier(Pu, le_x, fck, Ic_x, sway=state["sway"])
    delta_y = 1.0 if short_y else moment_magnifier(Pu, le_y, fck, Ic_y, sway=state["sway"])

    Mux_eff = Mux_base * delta_x
    Muy_eff = Muy_base * delta_y

    state.update(dict(le_x=le_x, le_y=le_y, lam_x=lam_x, lam_y=lam_y,
                      short_x=short_x, short_y=short_y, delta_x=delta_x, delta_y=delta_y,
                      Mux_base=Mux_base, Muy_base=Muy_base, Mux_eff=Mux_eff, Muy_eff=Muy_eff,
                      emin_x=emin_x, emin_y=emin_y))

    # (3) Biaxial utilization (strength)
    util, Mux_lim, Muy_lim = biaxial_utilization(b, D, bars, fck, fy, Pu, Mux_eff, Muy_eff, state["alpha"])
    state.update(dict(Mux_lim=Mux_lim, Muy_lim=Muy_lim, util=util))

    # (4) Shear & ties (advisory)
    max_long_dia = max([bar[2] for bar in bars], default=16.0)
    d_eff = D - (cover + state["tie_dia"] + 0.5 * max_long_dia)
    phiN = float(np.clip(1.0 + (Pu / max(1.0, (0.25 * fck * Ag))), 0.5, 1.5))
    tau_c_base = 0.62 * math.sqrt(fck)   # conservative (N/mm²)
    Vc = tau_c_base * b * d_eff * phiN   # N
    Vus = max(0.0, Vu - Vc)              # N
    state.update(dict(Vc=Vc, Vus=Vus, phiN=phiN, d_eff=d_eff))

    # (5) Tie spacing requirements
    Asv = state["legs"] * bar_area(state["tie_dia"])
    s_required_shear = (0.87 * fy * Asv * d_eff) / max(Vus, 1e-6) if Vus > 0 else 300.0
    s_cap_detailing = min(16.0 * max_long_dia, min(b, D), 300.0)
    s_13920_confine = min(min(b, D) / 4.0, 100.0)
    s_governing_tie = min(s_cap_detailing, s_required_shear)
    state.update({"s_required_shear": s_required_shear,
                  "s_cap_detailing": s_cap_detailing,
                  "s_13920_confine": s_13920_confine,
                  "s_governing_tie": s_governing_tie})

    # (6) Longitudinal steel requirement (strength vs min %)
    As_min = 0.008 * Ag  # IS 456 minimum for columns (0.8% of gross)
    # Simple proportionality assumption: capacity ∝ Ast (teaching-grade)
    As_req_strength = As_long * (util ** (1.0 / max(state["alpha"], 1e-6)))
    As_req = max(As_min, As_req_strength)
    reason = "min % (0.8%·Ag)" if As_min >= As_req_strength else "strength (biaxial utilization)"
    state.update(dict(As_min=As_min, As_req_strength=As_req_strength, As_req=As_req, As_reason=reason))

    return b, D, cover, bars, Ag

# ------------------------- 5) APP -------------------------
def main():
    st.title("🧱 RCC Column Designer — Single Canvas Output")
    st.caption("Integrated biaxial $P$–$M$ and shear design with narration (IS 456/13920). Printable output below.")
    st.markdown("---")

    initialize_state()
    state = st.session_state.state

    # --- Data I/O ---
    c1, c2 = st.columns(2)
    with c1:
        uploaded_file = st.file_uploader("Upload Saved Design State (JSON)", type="json", key="uploader")
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                if 'bars' in data and isinstance(data['bars'], list):
                    data['bars'] = [tuple(item) for item in data['bars']]
                st.session_state.state.update(data)
                st.rerun()
            except Exception as e:
                st.error(f"Error loading JSON: {e}")
    with c2:
        if state.get("bars"):
            st.markdown(get_json_download_link(to_json_serializable(state), "column_design_state.json"),
                        unsafe_allow_html=True)

    st.markdown("---")

    # --- 1) Geometry / Materials / Loads ---
    st.header("1️⃣ Geometry, Materials, and Factored Loads")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Section Geometry")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["b"] = st.number_input("Width $b$ (mm)", 200.0, 2000.0, state["b"], 25.0, key='in_b')
        state["D"] = st.number_input("Depth $D$ (mm)", 200.0, 3000.0, state["D"], 25.0, key='in_D')
        state["cover"] = st.number_input("Clear Cover (mm)", 20.0, 75.0, state["cover"], 5.0, key='in_cover')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("Material Properties")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["fck"] = st.number_input("$f_{ck}$ (MPa, M-Grade)", 20.0, 80.0, state["fck"], 1.0, key='in_fck')
        state["fy"] = st.number_input("$f_{y}$ (MPa, Fe-Grade)", 415.0, 600.0, state["fy"], 5.0, key='in_fy')
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.subheader("Factored Loads")
        st.markdown("Inputs for $P_u$ (kN), $M_u$ (kNm), $V_u$ (kN)")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["Pu"] = st.number_input("Axial Load $P_u$ (kN)", -3000.0, 6000.0, state["Pu"]/1e3, 10.0, key='in_Pu') * 1e3
        state["Mux"] = st.number_input("Moment $M_{ux}$ (kNm)", -2000.0, 2000.0, state["Mux"]/1e6, 5.0, key='in_Mux') * 1e6
        state["Muy"] = st.number_input("Moment $M_{uy}$ (kNm)", -2000.0, 2000.0, state["Muy"]/1e6, 5.0, key='in_Muy') * 1e6
        state["Vu"] = st.number_input("Shear $V_u$ (kN)", 0.0, 5000.0, state["Vu"]/1e3, 5.0, key='in_Vu') * 1e3
        st.markdown('</div>', unsafe_allow_html=True)

    # first recalc after geometry/material/load
    b, D, cover, bars, Ag = recalculate_properties(state)
    st.markdown("---")
    st.markdown('<div class="print-break"></div>', unsafe_allow_html=True)

    # --- 2) Reinforcement (live recalc before plotting) + Quick Check ---
    st.header("2️⃣ Reinforcement and Slenderness Setup")
    col_r1, col_r2, col_r3 = st.columns([1,1,1.2])
    bar_options = [12.0, 16.0, 20.0, 25.0, 28.0, 32.0]

    with col_r1:
        st.subheader("Longitudinal Bar Counts")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["n_top"] = st.number_input("Top row bars ($n_{top}$)", 0, 10, state["n_top"], 1, key='in_ntop')
        state["n_bot"] = st.number_input("Bottom row bars ($n_{bot}$)", 0, 10, state["n_bot"], 1, key='in_nbot')
        state["n_left"] = st.number_input("Left column bars ($n_{left}$)", 0, 10, state["n_left"], 1, key='in_nleft')
        state["n_right"] = st.number_input("Right column bars ($n_{right}$)", 0, 10, state["n_right"], 1, key='in_nright')
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r2:
        st.subheader("Bar Diameters & Layout")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["dia_top"]  = st.selectbox("Top bar $\\phi$ (mm)", bar_options, index=bar_options.index(state["dia_top"]), key='in_dtop')
        state["dia_bot"]  = st.selectbox("Bottom bar $\\phi$ (mm)", bar_options, index=bar_options.index(state["dia_bot"]), key='in_dbot')
        state["dia_side"] = st.selectbox("Side bar $\\phi$ (mm)", bar_options[:-1], index=bar_options[:-1].index(state["dia_side"]), key='in_dside')
        st.markdown('</div>', unsafe_allow_html=True)

    # 🔧 Recalc *after* bar counts/diameters, then draw
    b, D, cover, bars, Ag = recalculate_properties(state)

    with col_r2:
        st.markdown(f"**$A_{{st}}$ Provided:** **{state['As_long']:.0f} mm$^2$** "
                    f"({state['As_long'] * 100 / Ag:.2f}% $\\rho$)")

    with col_r3:
        st.subheader("Section Sketch & Quick Check")
        fig_sec_inputs = plotly_cross_section(b, D, cover, bars, state['tie_dia'], state['tie_spacing'])
        chart(fig_sec_inputs, key=kkey("sec","inputs"))

        # Quick biaxial check + steel requirement right here
        util = state['util']
        passfail = "PASS" if util <= 1.0 else "FAIL"
        badge_class = "badge" if util <= 1.0 else "badge bad"
        st.markdown(f'<span class="{badge_class}">Quick biaxial Σ = {util:.2f} → {passfail}</span>', unsafe_allow_html=True)

        As_long = state['As_long']
        As_req  = state['As_req']
        reason  = state['As_reason']
        oksteel = As_long >= As_req
        steel_badge = "badge" if oksteel else "badge bad"
        st.markdown(
            f'<span class="{steel_badge}">Ast required: {As_req:.0f} mm² '
            f'({reason}); provided: {As_long:.0f} mm² → {"OK" if oksteel else "Increase steel"}</span>',
            unsafe_allow_html=True
        )
        st.caption(
            f"Limits used: As_min = {state['As_min']:.0f} mm² (0.8% of Ag = {state['Ag']:.0f} mm²). "
            f"Strength-based Ast = {state['As_req_strength']:.0f} mm²."
        )

    st.markdown("---")

    # --- 3) Slenderness & magnified moments ---
    st.header("3️⃣ Slenderness Effects and Magnified Moments")
    col_sl_in, col_sl_out = st.columns(2)

    with col_sl_in:
        st.subheader("Slenderness Inputs (IS 456 Cl. 25.1)")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["storey_clear"] = st.number_input("Clear Storey Height $l_0$ (mm)", 2000.0, 6000.0, state["storey_clear"], 50.0, key='in_l0')
        restraint_options = ["Fixed-Fixed", "Fixed-Pinned", "Pinned-Pinned", "Fixed-Free (cantilever)"]
        state["restraint"] = st.selectbox("End Restraint (for $k$ factor)", restraint_options,
                                          index=restraint_options.index(state["restraint"]), key='in_restraint')
        state["sway"] = st.checkbox("Check for Sway Frame (Moment Magnification)", value=state["sway"], key='in_sway')
        k_factor = effective_length_factor(state["restraint"])
        st.markdown(f"Calculated **$k$ Factor**: **{k_factor:.2f}**")
        st.markdown('</div>', unsafe_allow_html=True)

    # recalc again after slenderness changes
    b, D, cover, bars, Ag = recalculate_properties(state)

    with col_sl_out:
        st.subheader("Slenderness and Magnification Check")
        st.markdown(f"**Slenderness (IS 456 Cl. 25.1.2):** Column is **{'Short' if state['short_x'] and state['short_y'] else 'Slender'}**.")
        st.metric(f"Eff. Length $l_{{e,x}}$ (mm) / $\\lambda_x$", f"{state['le_x']:.0f} / {state['lam_x']:.1f}")
        st.metric(f"Eff. $M_{{ux,eff}}$ (kNm) ($\\delta_x={state['delta_x']:.2f}$)", f"{kNm(state['Mux_eff'])}")
        st.metric(f"Eff. $M_{{uy,eff}}$ (kNm) ($\\delta_y={state['delta_y']:.2f}$)", f"{kNm(state['Muy_eff'])}")

    st.markdown("#### Calculation Narration (Slenderness)")
    st.write(f"The column is short if $\\lambda = l_e/r \\le 12$. Here $\\lambda_x={state['lam_x']:.1f}$, $\\lambda_y={state['lam_y']:.1f}$. "
             "When slender, magnify base moments by $\\delta=\\dfrac{C_m}{1-P_u/P_{cr}}$ (IS 456 Cl. 39.7).")
    st.latex(r" M_{u,eff} = M_{u,base} \times \delta,\quad P_{cr} = \frac{\pi^2 (0.4E_c) I_c}{l_e^2} ")
    st.markdown("---")
    st.markdown('<div class="print-break"></div>', unsafe_allow_html=True)

    # --- 4) Biaxial interaction ---
    st.header("4️⃣ Biaxial Interaction Strength Check")
    st.write("Uniaxial $M_{u,lim}$ for the given $P_u$ via strain-compatibility, then power-law interaction (IS 456 Annex G).")
    col_int_in, col_int_out = st.columns([1, 2])

    with col_int_in:
        st.subheader("Interaction Parameter")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["alpha"] = st.slider("Interaction Exponent $\\alpha$ (IS 456 Annex G)", 0.8, 2.0, state["alpha"], 0.1, key='in_alpha_check')
        st.markdown('</div>', unsafe_allow_html=True)
        st.subheader("Biaxial Check Result")
        st.latex(r"\left(\frac{M_{ux,eff}}{M_{ux,lim}}\right)^{\alpha} + \left(\frac{M_{uy,eff}}{M_{uy,lim}}\right)^{\alpha} \le 1.0")
        st.metric("Uniaxial $M_{ux,lim}$ (kNm)", f"{kNm(state['Mux_lim'])}")
        st.metric("Uniaxial $M_{uy,lim}$ (kNm)", f"{kNm(state['Muy_lim'])}")
        util_status = "✅ Biaxial PASS" if state['util'] <= 1.0 else "❌ Biaxial FAIL"
        st.markdown(f"**Utilization (Σ ≤ 1):** **{state['util']:.2f}** ({util_status})")

        # Ast check inside biaxial section
        st.markdown("**Ast Check (required vs provided)**")
        oksteel = state['As_long'] >= state['As_req']
        st.write(
            f"Required Ast = **{state['As_req']:.0f} mm²** "
            f"({state['As_reason']}); Provided = **{state['As_long']:.0f} mm²** "
            f"→ {'✅ OK' if oksteel else '❌ Increase steel'}"
        )

    with col_int_out:
        st.subheader("P–M Interaction Curves vs. Demand Point")
        fig_pmx = plot_pm_curve_plotly(b, D, bars, state["fck"], state["fy"], state["Pu"], state["Mux_eff"], 'x')
        chart(fig_pmx, key=kkey("pmx","interaction"))
        fig_pmy = plot_pm_curve_plotly(b, D, bars, state["fck"], state["fy"], state["Pu"], state["Muy_eff"], 'y')
        chart(fig_pmy, key=kkey("pmy","interaction"))

    st.markdown("---")
    st.markdown('<div class="print-break"></div>', unsafe_allow_html=True)

    # --- 5) Shear & ties ---
    st.header("5️⃣ Shear and Tie Design (IS 456 Cl. 40 / IS 13920)")
    col_sh1, col_sh2, col_sh3 = st.columns(3)
    tie_options = [6.0, 8.0, 10.0, 12.0]

    with col_sh1:
        st.subheader("Tie Design Inputs")
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        state["tie_dia"] = st.selectbox("Tie $\\phi$ (mm)", tie_options, index=tie_options.index(state["tie_dia"]), key='in_tdia_shear')
        leg_options_map = {"2-legged": 2, "4-legged": 4, "6-legged": 6}
        current_legs_str = f"{state.get('legs', 4)}-legged"
        if current_legs_str not in leg_options_map:
            current_legs_str = "4-legged"
        selected_legs_str = st.selectbox("Tie Legs ($A_{sv}$)", list(leg_options_map.keys()),
                                         index=list(leg_options_map.keys()).index(current_legs_str), key='in_legs')
        state["legs"] = leg_options_map[selected_legs_str]
        state["tie_spacing"] = st.number_input("Tie Spacing $s$ (mm) Provided", 50.0, 300.0, state["tie_spacing"], 5.0, key='in_ts_prov')
        st.markdown('</div>', unsafe_allow_html=True)

    b, D, cover, bars, Ag = recalculate_properties(state)

    with col_sh2:
        st.subheader("Shear Capacity")
        st.write(f"Axial factor $\\phi_N$ (IS 456 40.2.2.1): **{state['phiN']:.2f}**")
        st.metric("Concrete Capacity $V_c$ (kN)", f"{kN(state['Vc'])}")
        st.metric("Required Steel Capacity $V_{us}$ (kN)", f"{kN(state['Vus'])}")

    with col_sh3:
        st.subheader("Tie Spacing Check")
        st.metric("Shear Required $s$ (mm)", f"{state['s_required_shear']:.0f}")
        st.metric("Detailing Cap $s$ (mm)", f"{state['s_cap_detailing']:.0f}")
        s_status = "✅ Spacing PASS" if state["tie_spacing"] <= state["s_governing_tie"] else "❌ Spacing FAIL"
        st.markdown(f"**Governing $s$ ≤ {state['s_governing_tie']:.0f} mm:** **({s_status})**")

    st.markdown("#### Ductile Detailing Requirements (IS 13920)")
    lo = max(state['D'], state['b'], state['storey_clear']/6.0, 450.0)
    st.write(f"**Confinement Zone Length ($l_0$):** larger of D={D:.0f} mm, b={b:.0f} mm, $L_{{clear}}/6$={state['storey_clear']/6.0:.0f} mm, 450 mm → **{lo:.0f} mm** at each end.")
    st.write(f"**Spacing in Confinement Zone:** ties ≤ **{state['s_13920_confine']:.0f} mm** (min($\\min(b,D)/4$, 100 mm)).")

    st.markdown("---")
    st.markdown('<div class="print-break"></div>', unsafe_allow_html=True)

    # --- 6) Final report summary ---
    st.header("6️⃣ Final Submission Report Summary")
    st.caption("This entire page is designed to be printed as a complete calculation sheet.")

    st.subheader("A. Key Design Inputs and Governing Results")
    As_min = state["As_min"]
    As_governing_req = state["As_req"]
    As_reason = state["As_reason"]
    out = {
        "Cross Section b × D (mm)": f"{b:.0f} × {D:.0f}",
        "Material fck / fy (MPa)": f"{state['fck']:.0f} / {state['fy']:.0f}",
        "Design Pu (kN)": state["Pu"] / 1e3,
        "Design Mux,eff / Muy,eff (kNm)": f"{kNm(state['Mux_eff'])} / {kNm(state['Muy_eff'])}",
        "Slenderness $\\lambda_x/\\lambda_y$": f"{state['lam_x']:.1f} / {state['lam_y']:.1f} ({'Short' if state['short_x'] and state['short_y'] else 'Slender'})",
        "Magnifiers $\\delta_x/\\delta_y$": f"{state['delta_x']:.2f} / {state['delta_y']:.2f}",
        "Uniaxial Capacity $M_{u,lim}$ (kNm)": f"{kNm(state['Mux_lim'])} / {kNm(state['Muy_lim'])}",
        "Biaxial Utilization (≤1)": f"{state['util']:.2f}",
        "Ast Provided / Required (mm²)": f"{state['As_long']:.0f} / {As_governing_req:.0f}",
        "Ast Controlling Criterion": As_reason,
        "Min Ast (0.8%·Ag) (mm²)": f"{As_min:.0f}",
        "Concrete Shear $V_c$ (kN)": f"{kN(state['Vc'])}",
        "Tie Spacing Provided (mm)": f"Ø{state['tie_dia']:.0f} @ {state['tie_spacing']:.0f} (Legs: {state['legs']})",
        "Tie Spacing Governing Req (mm)": f"{state['s_governing_tie']:.0f}",
        "13920 Confined Zone Limit (mm)": f"{state['s_13920_confine']:.0f}"
    }
    df_out = pd.DataFrame({"Parameter": list(out.keys()), "Value": list(out.values())})
    st.dataframe(df_out, use_container_width=True)

    st.markdown("---")
    st.subheader("B. Detailed Calculation Figures")
    fig_sec_report = plotly_cross_section(b, D, cover, bars, state['tie_dia'], state['tie_spacing'])
    chart(fig_sec_report, key=kkey("sec","report"))
    st.markdown('<div class="print-break"></div>', unsafe_allow_html=True)

    col_rep1, col_rep2 = st.columns(2)
    with col_rep1:
        fig_pmx_report = plot_pm_curve_plotly(b, D, bars, state["fck"], state["fy"], state["Pu"], state["Mux_eff"], 'x')
        chart(fig_pmx_report, key=kkey("pmx","report"))
    with col_rep2:
        fig_pmy_report = plot_pm_curve_plotly(b, D, bars, state["fck"], state["fy"], state["Pu"], state["Muy_eff"], 'y')
        chart(fig_pmy_report, key=kkey("pmy","report"))

    st.markdown("---")
    st.caption("Note: For final submissions, adopt a full strain-compatibility PM surface and capacity-compatible shear per IS 456/13920 with clause citations.")

if __name__ == "__main__":
    main()
