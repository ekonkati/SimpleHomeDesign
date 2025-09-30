import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st

# -----------------------------
# 1. CONSTANTS, UTILITIES, and SECTION DATACLASS
# -----------------------------
ES = 200000.0  # MPa (N/mm2)
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

@dataclass
class Section:
    b: float  
    D: float  
    cover: float  
    bars: List[Tuple[float, float, float]]

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

# -----------------------------
# 2. CORE ENGINEERING LOGIC (Simplified)
# -----------------------------

def linspace_points(a: float, c: float, n: int) -> List[float]:
    if n <= 0: return []
    if n == 1: return [a + (c - a) / 2.0]
    return [a + i * (c - a) / (n - 1) for i in range(n)]

def generate_rectangular_bar_layout(b: float, D: float, cover: float, 
                                    n_top: int, n_bot: int, dia_top: float, dia_bot: float) -> List[Tuple[float, float, float]]:
    """Simplest layout: only top and bottom bars (corners + middle)."""
    bars_list = []
    
    dy_bot = cover + dia_bot / 2.0
    dy_top = D - (cover + dia_top / 2.0)
    
    # Top Row
    if n_top > 0:
        x_coords = linspace_points(cover + dia_top / 2.0, b - (cover + dia_top / 2.0), n_top)
        for x in x_coords:
            bars_list.append((x, dy_top, dia_top))

    # Bottom Row
    if n_bot > 0:
        x_coords = linspace_points(cover + dia_bot / 2.0, b - (cover + dia_bot / 2.0), n_bot)
        for x in x_coords:
            bars_list.append((x, dy_bot, dia_bot))
            
    # Use a set to handle corner overlaps
    unique_bars = []
    for x, y, dia in sorted(bars_list, key=lambda item: (item[1], item[0])):
        if not any(abs(x - ux) < 1e-4 and abs(y - uy) < 1e-4 for ux, uy, _ in unique_bars):
            unique_bars.append((x, y, dia))
            
    return unique_bars


def uniaxial_capacity_Mu_for_Pu(section: Section, fck: float, fy: float, Pu: float, axis: str) -> float:
    dimension = section.D if axis == 'x' else section.b
    
    def forces_and_moment(c: float):
        # IS 456 stress block
        Cc = 0.36 * fck * dimension * min(c, dimension)
        arm_Cc = 0.5 * dimension - 0.42 * min(c, dimension)
        Mc = Cc * arm_Cc
        Fs, Ms = 0.0, 0.0
        
        for (x_abs, y_abs, dia) in section.bars:
            As = bar_area(dia)
            # y is distance from compression face 
            if axis == 'x': y = section.D - y_abs # Compression face at y=D
            else: y = x_abs                     # Compression face at x=0
            
            strain = EPS_CU * (1.0 - (y / max(c, 1e-6)))
            stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
            force = stress * As
            z = 0.5 * dimension - y # lever arm to centroid
            
            Fs += force
            Ms += force * z

        N_res = Cc + Fs
        M_res = Mc + Ms
        return N_res, M_res

    target = Pu
    c_min = 0.05 * dimension
    c_max = 1.50 * dimension

    cL, cR = c_min, c_max
    # Bisection search for c_u that gives N_res = Pu
    for _ in range(50):
        cm = 0.5 * (cL + cR)
        Nm, Mm = forces_and_moment(cm)
        if abs(Nm - target) < 1.0: return float(Mm)
        
        NL, ML = forces_and_moment(cL)
        
        if (NL - target) * (Nm - target) <= 0: cR = cm
        else: cL = cm
        
    return float(forces_and_moment(0.5 * (cL + cR))[1])


def moment_magnifier(Pu: float, le_mm: float, fck: float, Ic: float, Cm: float = 0.85, sway: bool = False) -> float:
    Ec = 5000.0 * math.sqrt(max(fck, 1e-6))
    Pcr = (math.pi ** 2) * 0.4 * Ec * Ic / (le_mm ** 2 + 1e-9)
    if Pcr <= Pu: return 10.0
    delta = 1.0 / max(1e-6, (1.0 - Pu / Pcr))
    if not sway: delta = max(1.0, Cm * delta)
    return float(np.clip(delta, 1.0, 2.5))

def biaxial_utilization(section: Section, fck: float, fy: float, Pu: float, Mux_eff: float, Muy_eff: float, alpha: float) -> Tuple[float, float, float]:
    Mux_lim = uniaxial_capacity_Mu_for_Pu(section, fck, fy, Pu, axis='x')
    Muy_lim = uniaxial_capacity_Mu_for_Pu(section, fck, fy, Pu, axis='y')
    
    Mux_lim = max(Mux_lim, 1e-3) if abs(Mux_eff) > 1e-3 else Mux_lim
    Muy_lim = max(Muy_lim, 1e-3) if abs(Muy_eff) > 1e-3 else Muy_lim
        
    Rx = (abs(Mux_eff) / Mux_lim) ** alpha
    Ry = (abs(Muy_eff) / Muy_lim) ** alpha
    util = Rx + Ry
    return util, Mux_lim, Muy_lim

# ----------------------------
# MAIN APP BODY (Minimal Streamlit)
# ----------------------------

st.set_page_config(page_title="RCC Column Designer", layout="wide")
st.title("🧱 RCC Column Biaxial Design (Minimal Version)")
st.markdown("---")

# --- Initialize State ---
if "state" not in st.session_state:
    st.session_state.state = {
        "b": 450.0, "D": 600.0, "cover": 40.0, "fck": 30.0, "fy": 500.0,
        "Pu": 1200e3, "Mux": 120e6, "Muy": 80e6,
        "storey_clear": 3200.0, "restraint": "Pinned-Pinned", "sway": False,
        "n_top": 3, "n_bot": 3, "dia_top": 16.0, "dia_bot": 16.0, "alpha": 1.0
    }
state = st.session_state.state

# --- 1. Inputs ---
st.header("1️⃣ Design Inputs")
c1, c2, c3, c4 = st.columns(4)
with c1:
    state["b"] = st.number_input("Width $b$ (mm)", 200.0, 2000.0, state["b"], 25.0, key='in_b')
    state["D"] = st.number_input("Depth $D$ (mm)", 200.0, 3000.0, state["D"], 25.0, key='in_D')
    state["cover"] = st.number_input("Clear Cover (mm)", 20.0, 75.0, state["cover"], 5.0, key='in_cover')
with c2:
    state["fck"] = st.number_input("$f_{ck}$ (MPa, M-Grade)", 20.0, 80.0, state["fck"], 1.0, key='in_fck')
    state["fy"] = st.number_input("$f_{y}$ (MPa, Fe-Grade)", 415.0, 600.0, state["fy"], 5.0, key='in_fy')
    state["Pu"] = st.number_input("Axial Load $P_u$ (kN, +comp)", -3000.0, 6000.0, state["Pu"] / 1e3, 10.0, key='in_Pu') * 1e3
with c3:
    state["Mux"] = st.number_input("Moment $M_{ux}$ (kNm)", -2000.0, 2000.0, state["Mux"] / 1e6, 5.0, key='in_Mux') * 1e6
    state["Muy"] = st.number_input("Moment $M_{uy}$ (kNm)", -2000.0, 2000.0, state["Muy"] / 1e6, 5.0, key='in_Muy') * 1e6
    restraint_options = ["Fixed-Fixed", "Pinned-Pinned", "Fixed-Free (cantilever)"]
    state["restraint"] = st.selectbox("End Restraint", restraint_options, index=restraint_options.index(state["restraint"]), key='in_restraint')
with c4:
    state["n_top"] = st.number_input("Top bars ($n_{top}$)", 2, 10, state["n_top"], 1, key='in_ntop')
    state["n_bot"] = st.number_input("Bottom bars ($n_{bot}$)", 2, 10, state["n_bot"], 1, key='in_nbot')
    bar_options = [12.0, 16.0, 20.0, 25.0, 28.0, 32.0]
    state["dia_top"] = st.selectbox("Top bar $\\phi$ (mm)", bar_options, index=bar_options.index(state["dia_top"]), key='in_dtop')
    state["dia_bot"] = st.selectbox("Bottom bar $\\phi$ (mm)", bar_options, index=bar_options.index(state["dia_bot"]), key='in_dbot')

# --- Calculate Derived Properties ---
k_factor = effective_length_factor(state["restraint"])
le_x = k_factor * state["storey_clear"]
le_y = k_factor * state["storey_clear"]

bars = generate_rectangular_bar_layout(state["b"], state["D"], state["cover"], 
                                      state["n_top"], state["n_bot"], state["dia_top"], state["dia_bot"])
section = Section(b=state["b"], D=state["D"], cover=state["cover"], bars=bars)

# --- 2. Slenderness & Magnification ---
st.header("2️⃣ Slenderness & Moment Magnification")
Mux_eff = state["Mux"]
Muy_eff = state["Muy"]

with st.expander("Slender Column Check and P-Delta Magnification"):
    # Slenderness check
    lam_x = le_x / max(section.rx, 1e-6)
    lam_y = le_y / max(section.ry, 1e-6)
    short_x, short_y = lam_x <= 12.0, lam_y <= 12.0
    st.write(f"$\lambda_x = {lam_x:.1f}$ ({'Short' if short_x else 'Slender'}), $\lambda_y = {lam_y:.1f}$ ({'Short' if short_y else 'Slender'})")

    # Magnification
    delta_x = moment_magnifier(state["Pu"], le_x, state["fck"], section.Ic_x, sway=state["sway"]) if not short_x else 1.0
    delta_y = moment_magnifier(state["Pu"], le_y, state["fck"], section.Ic_y, sway=state["sway"]) if not short_y else 1.0
    
    Mux_eff = state["Mux"] * delta_x
    Muy_eff = state["Muy"] * delta_y
    
    st.markdown(f"**$\delta_x$**: {delta_x:.2f}, $M_{{ux}}'$: {kNm(Mux_eff)} kNm")
    st.markdown(f"**$\delta_y$**: {delta_y:.2f}, $M_{{uy}}'$: {kNm(Muy_eff)} kNm")


# --- 3. Biaxial Interaction ---
st.header("3️⃣ Biaxial Interaction Check (Strength)")

st.latex(r"\left(\frac{|M_{ux}'|}{M_{ux,lim}}\right)^{\alpha} + \left(\frac{|M_{uy}'|}{M_{uy,lim}}\right)^{\alpha} \le 1.0")

state["alpha"] = st.slider("Interaction Exponent $\\alpha$", 0.8, 2.0, state["alpha"], 0.1, key='in_alpha')

# Core Calculation
with st.spinner("Calculating Uniaxial Capacities..."):
    util, Mux_lim, Muy_lim = biaxial_utilization(section, state["fck"], state["fy"], state["Pu"], Mux_eff, Muy_eff, state["alpha"])

c_m_lim, c_util = st.columns(2)
with c_m_lim:
    st.metric("Uniaxial $M_{ux,lim}$ (kNm)", f"{kNm(Mux_lim)}")
    st.metric("Uniaxial $M_{uy,lim}$ (kNm)", f"{kNm(Muy_lim)}")
with c_util:
    st.metric("Biaxial Utilization (Σ ≤ 1)", f"**{util:.2f}**")
    if util <= 1.0:
        st.success("✅ Interaction PASS.")
    else:
        st.error("❌ Interaction FAIL. Increase steel.")

st.markdown("---")
st.caption("Minimal code focusing only on the core strength calculation to bypass the Streamlit AST parsing error.")
