# -*- coding: utf-8 -*-
# Column-EK.py — RCC Column (Biaxial) Design Canvas
# Single-page section; safe for Streamlit multipage (no set_page_config here)
# Units: N, mm, MPa, kN·m

import math
import json
import base64
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------- styles (ASCII-only) ----------
st.markdown(
    """
<style>
@media print {
  header, .stToolbar, .stAppDeployButton, .stActionButton, .stDownloadButton, footer, .stSidebar { display: none !important; }
  .block-container { padding: 0.6cm 1.0cm !important; max-width: 100% !important; }
  .print-break { page-break-before: always; }
}
.highlight { background: #fff8e1; border: 2px solid #ffcc80; border-radius: 12px; padding: 10px 15px; margin: 10px 0 20px 0; }
.highlight .stSelectbox, .highlight .stNumberInput, .highlight .stSlider, .highlight .stTextInput { background: #fffceb !important; }
.js-plotly-plot, .plotly, .user-select-none { margin: 0 !important; }
.badge { padding:4px 8px; border-radius:8px; background:#eef7ee; color:#0a7a0a; font-weight:700 }
.badge.bad { background:#fdecee; color:#b00020 }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- constants & helpers ----------
ES = 200000.0
EPS_CU = 0.0035

def bar_area(dia_mm: float) -> float:
    return math.pi * (dia_mm ** 2) / 4.0

def kN(value: float) -> str:
    return f"{value / 1e3:.1f}"

def kNm(value: float) -> str:
    return f"{value / 1e6:.1f}"

def effective_length_factor(restraint: str) -> float:
    return {
        "Fixed-Fixed": 0.65,
        "Fixed-Pinned": 0.80,
        "Pinned-Pinned": 1.00,
        "Fixed-Free (cantilever)": 2.00,
    }.get(restraint, 1.00)

def moment_magnifier(Pu: float, le_mm: float, fck: float, Ic: float, Cm: float = 0.85, sway: bool = False) -> float:
    Ec = 5000.0 * math.sqrt(max(fck, 1e-6))
    Pcr = (math.pi ** 2) * 0.4 * Ec * Ic / (le_mm ** 2 + 1e-9)
    if Pcr <= Pu:
        return 10.0
    delta = 1.0 / max(1e-6, (1.0 - Pu / Pcr))
    if not sway:
        delta = max(1.0, Cm * delta)
        return float(np.clip(delta, 1.0, 2.5))
    return float(np.clip(delta, 1.0, 5.0))

def to_json_serializable(state: dict) -> dict:
    safe_state = {}
    for k, v in state.items():
        if isinstance(v, float):
            safe_state[k] = round(v, 6)
        elif isinstance(v, np.floating):
            safe_state[k] = round(float(v), 6)
        elif isinstance(v, (list, tuple)) and all(isinstance(x, tuple) for x in v):
            safe_state[k] = [list(item) for item in v]
        else:
            safe_state[k] = v
    return safe_state

def get_json_download_link(data_dict: dict, filename: str) -> str:
    json_str = json.dumps(data_dict, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    return f'<a href="data:file/json;base64,{b64}" download="{filename}">Download Design State</a>'

def chart(fig, *, key: str):
    st.plotly_chart(fig, use_container_width=True, key=key)

def kkey(*parts) -> str:
    return "chart_" + "_".join(str(p) for p in parts)

# ---------- rebar layout & capacities ----------
def _linspace_points(a: float, c: float, n: int) -> List[float]:
    if n <= 0:
        return []
    if n == 1:
        return [a + (c - a) / 2.0]
    return [a + i * (c - a) / (n - 1) for i in range(n)]

def _generate_bar_layout(b: float, D: float, cover: float, state: dict) -> List[Tuple[float, float, float]]:
    """
    Perimeter bars with no corner double counting:
    - Top & bottom rows include corners.
    - Side rows exclude corners so plot count equals inputs.
    """
    n_top, n_bot, n_left, n_right = state["n_top"], state["n_bot"], state["n_left"], state["n_right"]
    dia_top, dia_bot, dia_side = state["dia_top"], state["dia_bot"], state["dia_side"]

    bars = []
    max_dia = max(dia_top, dia_bot, dia_side)
    dx = cover + state["tie_dia"] + max_dia / 2.0
    dx = min(dx, min(b, D) / 2 - 5.0)

    x0, x1 = dx, b - dx
    y0, y1 = dx, D - dx

    for x in _linspace_points(x0, x1, n_top):
        bars.append((x, y1, dia_top))
    for x in _linspace_points(x0, x1, n_bot):
        bars.append((x, y0, dia_bot))
    if n_left > 0:
        ys = _linspace_points(y0, y1, n_left + 2)[1:-1]
        for y in ys:
            bars.append((x0, y, dia_side))
    if n_right > 0:
        ys = _linspace_points(y0, y1, n_right + 2)[1:-1]
        for y in ys:
            bars.append((x1, y, dia_side))
    return bars

def _uniaxial_capacity_Mu_for_Pu(b: float, D: float, bars: List[Tuple[float, float, float]],
                                 fck: float, fy: float, Pu: float, axis: str) -> float:
    depth = D if axis == "x" else b
    width = b if axis == "x" else D

    def forces_and_moment(c: float):
        xu = min(c, depth)
        Cc = 0.36 * fck * width * xu
        arm_Cc = 0.5 * depth - 0.42 * xu
        Mc = Cc * arm_Cc
        Fs, Ms = 0.0, 0.0
        for (x_abs, y_abs, dia) in bars:
            As = bar_area(dia)
            y = (D - y_abs) if axis == "x" else x_abs
            strain = EPS_CU * (1.0 - (y / max(c, 1e-9)))
            stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
            force = stress * As
            z = 0.5 * depth - y
            Fs += force
            Ms += force * z
        return Cc + Fs, Mc + Ms

    target = Pu
    cL, cR = 0.01 * depth, 1.50 * depth
    NL, ML = forces_and_moment(cL)
    NR, MR = forces_and_moment(cR)
    if target <= NL:
        return float(ML)
    if target >= NR:
        return float(MR)
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
    Mux_lim = _uniaxial_capacity_Mu_for_Pu(b, D, bars, fck, fy, Pu, axis="x")
    Muy_lim = _uniaxial_capacity_Mu_for_Pu(b, D, bars, fck, fy, Pu, axis="y")
    Mux_lim = max(Mux_lim, 1e-3) if abs(Mux_eff) > 1e-3 else Mux_lim
    Muy_lim = max(Muy_lim, 1e-3) if abs(Muy_eff) > 1e-3 else Muy_lim
    Rx = (abs(Mux_eff) / Mux_lim) ** alpha
    Ry = (abs(Muy_eff) / Muy_lim) ** alpha
    util = Rx + Ry
    return util, Mux_lim, Muy_lim

# ---------- figures ----------
def plotly_cross_section(b: float, D: float, cover: float, bars: List[Tuple[float, float, float]],
                         tie_dia: float, tie_spacing: float) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=b, y1=D, line=dict(color="black", width=2),
                  fillcolor="rgba(240,240,240,0.8)")
    eff_cov = cover + tie_dia
    fig.add_shape(type="rect", x0=eff_cov, y0=eff_cov, x1=b-eff_cov, y1=D-eff_cov,
                  line=dict(color="gray", width=1, dash="dot"), fillcolor="rgba(0,0,0,0)")
    xs, y
