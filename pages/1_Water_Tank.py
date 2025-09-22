# -*- coding: utf-8 -*-
import math, io
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# ===============================
# Data classes
# ===============================
@dataclass
class Materials:
    fck: float
    fy: float
    gamma_conc: float
    Ec: float
    exposure: str

@dataclass
class Geometry:
    H: float
    L: float
    B: float
    t_wall: float
    t_base: float
    freeboard: float
    tank_type: str  # "Ground" or "Elevated"

@dataclass
class Loads:
    gamma_w: float
    gamma_s: float
    K0: float
    phi: float
    mu_base: float

# ===============================
# General helpers
# ===============================
def triangular_resultant(gamma: float, H: float) -> Tuple[float, float]:
    R = 0.5 * gamma * H**2
    zbar = H/3.0
    return R, zbar

def cantilever_arrays(gamma: float, H: float, n: int = 101) -> pd.DataFrame:
    x = np.linspace(0, H, n)  # from free surface down
    p = gamma * x                 # kPa
    V = 0.5 * gamma * x**2        # kN/m
    M = (gamma * x**3) / 6.0      # kNm/m
    return pd.DataFrame({
        "Level from Top x (m)": x,
        "Water Pressure p(x) (kPa)": p,
        "Shear V(x) (kN/m)": V,
        "Moment M(x) (kNm/m)": M
    })

def base_values(gamma: float, H: float) -> Tuple[float, float]:
    return 0.5 * gamma * H**2, gamma * H**3 / 6.0

def demand_ast_from_M(Mu_kNm: float, d_eff_mm: float, fy_MPa: float) -> float:
    z = 0.9 * max(d_eff_mm, 1.0)
    Mu_Nmm = Mu_kNm * 1e6
    Ast = Mu_Nmm / (0.87 * fy_MPa * z)
    return max(Ast, 0.0)

def pick_bar_spacing(Ast_req_mm2_per_m: float,
                     pref_dias: List[int], s_min: int = 75, s_max: int = 300) -> Tuple[int, int, float]:
    for dia in pref_dias:
        area = 0.25 * math.pi * dia**2
        for s in range(s_max, s_min-1, -5):
            Ast = 1000.0 * area / s
            if Ast >= Ast_req_mm2_per_m:
                return dia, s, Ast
    dia = pref_dias[0] if pref_dias else 12
    s = s_min
    Ast = 1000.0 * (0.25 * math.pi * dia**2) / s
    return dia, s, Ast

# ===============================
# IS 3370 (Part 2) — Tables & Checks (from code)
# ===============================
# Table 1: load combos and γf (rows 1..5)
TABLE1_ROWS = {
    1: {"state":"ULS","DL":1.5,"IL":1.5,"EP":1.5,"FL":1.5,"WL_EL":0.0},
    2: {"state":"ULS","DL":1.2,"IL":0.0,"EP":1.0,"FL":1.2,"WL_EL":1.4},
    3: {"state":"ULS","DL":0.9,"IL":0.0,"EP":1.0,"FL":1.0,"WL_EL":1.4},
    4: {"state":"ULS","DL":1.4,"IL":0.0,"EP":1.0,"FL":0.0,"WL_EL":1.4},
    5: {"state":"ULS","DL":1.2,"IL":1.2,"EP":1.2,"FL":1.2,"WL_EL":1.2},
    # service
    101: {"state":"SLS","DL":1.0,"IL":1.0,"EP":1.0,"FL":1.0,"WL_EL":0.0},
    102: {"state":"SLS","DL":1.0,"IL":0.0,"EP":0.7,"FL":1.0,"WL_EL":0.3},
}

# Table 2: σs limits vs crack width (HS deformed)
TABLE2_SIGMA_LIMIT = {0.10:100.0, 0.20:130.0}  # MPa

# Table 3: (dia, spacing) -> σs,max (MPa)
TABLE3_SIGMA_BY_DETAIL = {
    (10,75):155,(12,75):155,
    (12,100):150,(16,100):150,
    (16,125):148,(20,125):148,
    (20,150):145,(25,150):145,
    (20,175):142,(25,175):142,
    (25,200):140,(28,200):140,(32,200):135,
}

# Table 4: Tightness classes → suggested crack width
TIGHTNESS_TO_W = {"1":0.20,"2":0.20,"2 (strict)":0.10,"3":0.10}

# Table 5: Minimum steel % per surface zone (by tank type & length band)
# rows are (grade, tank_type, band) -> percent
TABLE5_MIN_PERC = {
    ("Fe250","Elevated","<=14m"):0.44, ("Fe250","Elevated",">=28m"):0.66,
    ("Fe250","Ground","<=14m"):0.40,   ("Fe250","Ground",">=22m"):0.60,
    ("Fe415/500","Elevated","<=14m"):0.28, ("Fe415/500","Elevated",">=28m"):0.42,
    ("Fe415/500","Ground","<=14m"):0.24,   ("Fe415/500","Ground",">=22m"):0.36,
}

def interpolate_table5(grade: str, tank_type: str, length_m: float) -> float:
    # linear interpolation between band endpoints per Table 5 notes
    if tank_type == "Elevated":
        x0,x1 = 14.0, 28.0
        lo = TABLE5_MIN_PERC[(grade,"Elevated","<=14m")]
        hi = TABLE5_MIN_PERC[(grade,"Elevated",">=28m")]
    else:
        x0,x1 = 14.0, 22.0
        lo = TABLE5_MIN_PERC[(grade,"Ground","<=14m")]
        hi = TABLE5_MIN_PERC[(grade,"Ground",">=22m")]
    if length_m <= x0: return lo
    if length_m >= x1: return hi
    t = (length_m - x0)/(x1 - x0)
    return lo*(1-t) + hi*t

def sigma_allow_for_detailing(dia_mm: int, spacing_mm: int, w_lim: float) -> float:
    row3 = TABLE3_SIGMA_BY_DETAIL.get((int(dia_mm), int(spacing_mm)))
    row2 = TABLE2_SIGMA_LIMIT.get(round(w_lim,2))
    if row3 is None and row2 is None: return 130.0
    if row3 is None: return row2
    if row2 is None: return row3
    return min(row3, row2)

def steel_stress_sls(Ms_kNm_per_m: float, d_eff_mm: float, As_mm2_per_m: float) -> float:
    z = 0.9 * max(d_eff_mm, 1.0)
    T_N = (Ms_kNm_per_m * 1e6) / z
    return T_N / max(As_mm2_per_m, 1e-6)

# ===============================
# IS 3370 (Part 4/Sec 2) bilinear interpolation for kM (optional CSV)
# ===============================
def bilinear_kM(df: pd.DataFrame, case: str, side: str, position: str,
                b_over_a: float, c_over_a: float) -> Optional[float]:
    d = df.copy()
    d["case"]=d["case"].astype(str)
    d = d[(d["case"]==str(case))&(d["side"]==side)&(d["position"]==position)]
    if d.empty: return None
    bs = np.sort(d["b_over_a"].unique()); cs = np.sort(d["c_over_a"].unique())
    b0 = bs[bs<=b_over_a].max(initial=None); b1 = bs[bs>=b_over_a].min(initial=None)
    c0 = cs[cs<=c_over_a].max(initial=None); c1 = cs[cs>=c_over_a].min(initial=None)
    if None in (b0,b1,c0,c1):
        d["score"]=(d["b_over_a"]-b_over_a).abs()+(d["c_over_a"]-c_over_a).abs()
        return float(d.sort_values("score").iloc[0]["kM"])
    def get(bb,cc):
        row=d[(d["b_over_a"]==bb)&(d["c_over_a"]==cc)]
        return None if row.empty else float(row.iloc[0]["kM"])
    k00,k10,k01,k11 = get(b0,c0),get(b1,c0),get(b0,c1),get(b1,c1)
    if None in (k00,k10,k01,k11):
        d["score"]=(d["b_over_a"]-b_over_a).abs()+(d["c_over_a"]-c_over_a).abs()
        return float(d.sort_values("score").iloc[0]["kM"])
    tb = (b_over_a - b0)/(b1-b0) if (b1-b0)!=0 else 0
    tc = (c_over_a - c0)/(c1-c0) if (c1-c0)!=0 else 0
    k_b0 = k00*(1-tb)+k10*tb
    k_b1 = k01*(1-tb)+k11*tb
    return float(k_b0*(1-tc)+k_b1*tc)

# ===============================
# EQ hydrodynamic actions (IS 1893-2 style inputs)
# ===============================
def ah_from_ZIR(Z: float, I: float, R: float, Sa_over_g: float) -> float:
    """Design horizontal acceleration coefficient Ah ~ (Z/2)*(I/R)*(Sa/g)."""
    return (Z/2.0) * (I/R) * Sa_over_g

def hydrodynamic_components(gamma_w: float, L: float, B: float, H: float,
                            Ci: float, Cc: float, hi: float, hc: float) -> dict:
    """
    Compute impulsive & convective 'effective' masses (per unit plan area * H) and
    return simple triangular/sinusoidal pressure shapes scaled to match base shear
    post-combination. This keeps graphs meaningful even without full 1893-2 tables.
    """
    V = L*B
    # Effective masses (per unit plan area * H), scale by fluid density (γw/g ≈ 1000 kg/m³)
    # We will carry forces directly in kN using γw.
    mi = Ci * gamma_w * H    # kN/m² equivalent "weight" scaling
    mc = Cc * gamma_w * H
    return {"mi":mi,"mc":mc,"hi":hi,"hc":hc}

def eq_pressures_profile(H: float, Pi_base: float, Pc_base: float, hi: float, hc: float, n=51):
    """
    Build vertical distributions (kPa) for graphing:
    - impulsive: linear (triangular) reaching max at base portion near hi
    - convective: half-cosine 'sloshing' with zero at base and max at free-surface vicinity (hc)
    Scaled so ∫p dz produces base shear proportional to Pi_base/Pc_base.
    """
    z = np.linspace(0, H, n)           # 0 at base, H at free surface (for plotting convenience)
    # Impulsive triangular with peak near base:
    p_i = (Pi_base / max(H,1e-6)) * (z / H)  # simple linear rise; will annotate hi
    # Convective half-wave (max near free surface):
    p_c = (Pc_base / max(H,1e-6)) * (1 - np.cos(np.pi * z/H)) / 2.0
    return z, p_i*1e3, p_c*1e3  # return kPa

# ===============================
# Streamlit APP
# ===============================
#st.set_page_config(page_title="RCC Water Tank — EP/FL/EQ + IS 3370/1893-2", layout="wide")
st.title("RCC Water Tank Design — EP • FL • EQ(slosh) + IS 3370-2 Tables + Detailed PDF")

# ---- SIDEBAR INPUTS ----
st.sidebar.header("Geometry")
H = st.sidebar.number_input("Water depth H (m)", 1.0, 25.0, 4.0, 0.1)
freeboard = st.sidebar.number_input("Freeboard (m)", 0.0, 1.0, 0.3, 0.05)
L = st.sidebar.number_input("Internal Length b (m)", 1.0, 100.0, 10.0, 0.1)
B = st.sidebar.number_input("Internal Width c (m)", 1.0, 100.0, 6.0, 0.1)
t_wall = st.sidebar.number_input("Wall thickness (m)", 0.15, 1.0, 0.25, 0.01)
t_base = st.sidebar.number_input("Base slab thickness (m)", 0.20, 2.0, 0.35, 0.01)
tank_type = st.sidebar.selectbox("Tank type (Table 5)", ["Ground","Elevated"], index=0)

st.sidebar.header("Materials")
fck = st.sidebar.selectbox("Concrete fck (MPa)", [20,25,30,35,40], index=2)
fy  = st.sidebar.selectbox("Steel fy (MPa)", [250,415,500], index=1)
gamma_conc = st.sidebar.number_input("Concrete unit weight (kN/m³)", 22.0, 26.0, 24.0, 0.1)
Ec = st.sidebar.number_input("Ec (MPa)", 25000.0, 40000.0, 27386.0, 10.0)
exposure = st.sidebar.selectbox("Exposure", ["Mild","Moderate","Severe","Very Severe"], index=1)

st.sidebar.header("Loads")
gamma_w = st.sidebar.number_input("Water unit weight (kN/m³)", 9.5, 10.5, 9.81, 0.01)
design_condition = st.sidebar.selectbox("Backfill condition", ["On-ground (no backfill)", "Underground / Backfilled"], index=0)
gamma_s = st.sidebar.number_input("Soil unit weight (kN/m³)", 16.0, 22.0, 18.0, 0.1)
K0 = st.sidebar.number_input("K0 (at-rest earth pressure coeff.)", 0.3, 1.0, 0.5, 0.01)
phi = st.sidebar.number_input("Soil friction angle φ (deg)", 20.0, 40.0, 30.0, 0.5)
mu_base = st.sidebar.number_input("Base friction coefficient μ", 0.3, 0.8, 0.5, 0.01)

st.sidebar.header("Detailing & Serviceability")
cover = st.sidebar.number_input("Nominal cover (mm)", 25.0, 60.0, 40.0, 1.0)
pref_dias = st.sidebar.multiselect("Preferred bar diameters (mm)", [10,12,16,20,25], default=[12,16,20])
tight_class = st.sidebar.selectbox("Tightness Class (Table 4)", ["1","2","2 (strict)","3"], index=1)
w_lim = float(TIGHTNESS_TO_W[tight_class])
st.sidebar.caption(f"Crack width limit from Table 4 → {w_lim:.2f} mm (used with Tables 2/3).")

st.sidebar.header("Uplift / Flotation")
gw_height_above_base = st.sidebar.number_input("GW level above slab underside (m) — empty tank", 0.0, 10.0, 2.0, 0.1)
soil_cover_above_slab = st.sidebar.number_input("Soil cover above slab (m)", 0.0, 5.0, 0.0, 0.1)

# IS 3370-4/Sec2 optional CSV for kM
st.sidebar.header("IS 3370-4/Sec2 (optional)")
mode_tables = st.sidebar.radio("Wall moment source", ["Cantilever (closed-form)","Table-based CSV (bilinear interp)"], index=0)
coef_file = st.sidebar.file_uploader("Upload kM CSV (case,side,position,b_over_a,c_over_a,kM)")

# EQ hydrodynamics (IS 1893-2 style)
st.sidebar.header("EQ Hydrodynamics (IS 1893-2 style)")
Z = st.sidebar.number_input("Zone factor Z", 0.06, 0.36, 0.16, 0.02)
I_fac = st.sidebar.number_input("Importance factor I", 1.0, 1.5, 1.2, 0.1)
R = st.sidebar.number_input("Response reduction R", 1.0, 6.0, 3.0, 0.5)
Sa_i = st.sidebar.number_input("Sa/g (impulsive) at Ti", 0.2, 3.0, 2.5, 0.1)
Sa_c = st.sidebar.number_input("Sa/g (convective) at Tc", 0.05, 3.0, 1.5, 0.05)
coeff_file = st.sidebar.file_uploader("Upload hydrodynamic coeff CSV (Ci,Cc,hi,hc,Ti,Tc) for your H/b")
Ci = st.sidebar.number_input("Ci (fallback)", 0.3, 1.0, 0.6, 0.05)
Cc = st.sidebar.number_input("Cc (fallback)", 0.1, 0.8, 0.3, 0.05)
hi = st.sidebar.number_input("hi/H (fallback)", 0.3, 0.9, 0.4, 0.05)
hc = st.sidebar.number_input("hc/H (fallback)", 0.3, 1.2, 0.8, 0.05)

# Base plan extras
st.sidebar.header("Base Plan (key + anchors)")
key_width = st.sidebar.number_input("Shear key width (m)", 0.20, 1.00, 0.40, 0.01)
key_depth = st.sidebar.number_input("Shear key depth (m)", 0.10, 1.00, 0.30, 0.01)
anchor_dia = st.sidebar.number_input("Anchor dia (mm)", 12.0, 32.0, 20.0, 1.0)
anchor_sx = st.sidebar.number_input("Anchor spacing along L (m)", 0.25, 3.0, 1.50, 0.05)
anchor_sy = st.sidebar.number_input("Anchor spacing along B (m)", 0.25, 3.0, 1.50, 0.05)

# Pack
mats = Materials(fck=fck, fy=fy, gamma_conc=gamma_conc, Ec=Ec, exposure=exposure)
geom = Geometry(H=H, L=L, B=B, t_wall=t_wall, t_base=t_base, freeboard=freeboard, tank_type=tank_type)
loads = Loads(gamma_w=gamma_w, gamma_s=gamma_s, K0=K0, phi=phi, mu_base=mu_base)

# ===============================
# TABS
# ===============================
t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
    "1) Hydrostatics & EP/FL",
    "2) Shear/Moment + Table-based option",
    "3) EQ Hydrodynamic (impulsive/convective)",
    "4) ULS/SLS Combo & Bearing",
    "5) Detailing & Min Steel (Table 5)",
    "6) Crack Control SLS (Tables 2/3/4)",
    "7) Base Plan (key+anchors)",
    "8) PDF Export"
])

# ---- TAB 1: Hydrostatics & Earth Pressure
with t1:
    st.subheader("Hydrostatic FL + Earth Pressure EP (if backfilled)")
    df_h = cantilever_arrays(loads.gamma_w, geom.H, n=41)
    st.dataframe(df_h, use_container_width=True, height=240)

    colA, colB = st.columns(2)
    with colA:
        figP, axP = plt.subplots()
        axP.plot(df_h["Water Pressure p(x) (kPa)"], df_h["Level from Top x (m)"])
        axP.invert_yaxis(); axP.set_xlabel("FL pressure p(x) [kPa]"); axP.set_ylabel("Level from Top x [m]")
        axP.set_title("Liquid pressure (FL)")
        st.pyplot(figP)
    with colB:
        if design_condition == "Underground / Backfilled":
            # earth pressure (at rest) from ground to full height, acting outside (opposing FL)
            z = np.linspace(0, geom.H, 41)
            p_e = loads.K0 * loads.gamma_s * z  # kPa
            figEP, axEP = plt.subplots()
            axEP.plot(p_e, geom.H - z)
            axEP.invert_yaxis(); axEP.set_xlabel("EP [kPa]"); axEP.set_ylabel("Height above base [m]")
            axEP.set_title("At-rest earth pressure (EP)")
            st.pyplot(figEP)
            st.session_state["figEP"] = figEP
        else:
            st.info("On-ground tank: external earth pressure not considered.")
    st.session_state["figP"] = figP

# ---- TAB 2: Shear/Moment + table-based
with t2:
    st.subheader("Shear V(x), Moment M(x)")
    df_vm = cantilever_arrays(loads.gamma_w, geom.H, n=121)
    st.dataframe(df_vm, use_container_width=True, height=320)
    st.download_button("Download V/M Table (CSV)", df_vm.to_csv(index=False), "shear_moment_table.csv", "text/csv")

    col1, col2 = st.columns(2)
    with col1:
        figV, axV = plt.subplots()
        axV.plot(df_vm["Shear V(x) (kN/m)"], df_vm["Level from Top x (m)"])
        axV.invert_yaxis(); axV.set_xlabel("Shear [kN/m]"); axV.set_ylabel("Level from Top x [m]"); axV.set_title("Shear Diagram")
        st.pyplot(figV)
    with col2:
        figM, axM = plt.subplots()
        axM.plot(df_vm["Moment M(x) (kNm/m)"], df_vm["Level from Top x (m)"])
        axM.invert_yaxis(); axM.set_xlabel("Moment [kNm/m]"); axM.set_ylabel("Level from Top x [m]"); axM.set_title("Bending Moment Diagram")
        st.pyplot(figM)

    Vb_cant, Mb_cant = base_values(loads.gamma_w, geom.H)
    Mb_design = Mb_cant
    calc_source = "Cantilever (closed-form)"
    if mode_tables == "Table-based CSV (bilinear interp)" and coef_file is not None:
        try:
            cdf = pd.read_csv(coef_file)
            case = st.selectbox("Case (per your CSV)", sorted(cdf["case"].astype(str).unique()))
            side = st.selectbox("Side", ["long","short"])
            position = st.selectbox("Position", ["base","mid-height","top","critical"])
            kM = bilinear_kM(cdf, case, side, position, geom.L/geom.H, geom.B/geom.H)
            if kM is not None:
                Mb_design = kM * loads.gamma_w * geom.H**3
                calc_source = f"IS 3370-4/Sec2 (kM={kM:.4f})"
                st.info(f"Table-based base moment used: M = {Mb_design:.2f} kNm/m ({calc_source}).")
        except Exception as e:
            st.error(f"CSV read/interp error: {e}")
    st.success(f"Base: Vb={Vb_cant:.2f} kN/m, M (design)={Mb_design:.2f} kNm/m  — {calc_source}")

    st.session_state["df_vm"] = df_vm
    st.session_state["figV"] = figV
    st.session_state["figM"] = figM
    st.session_state["Mb_design"] = Mb_design

# ---- TAB 3: EQ hydrodynamics
with t3:
    st.subheader("Earthquake hydrodynamic actions (impulsive + convective with sloshing)")
    # coefficients: CSV overrides
    Ci_use, Cc_use, hi_use, hc_use, Ti_use, Tc_use = Ci, Cc, hi*H, hc*H, None, None
    if coeff_file is not None:
        try:
            hdf = pd.read_csv(coeff_file)
            # allow choosing a row (say, by H/b band)
            st.caption("Pick row from uploaded 1893-2 coefficients")
            idx = st.selectbox("Row", list(range(len(hdf))))
            row = hdf.iloc[int(idx)]
            Ci_use = float(row.get("Ci", Ci_use))
            Cc_use = float(row.get("Cc", Cc_use))
            hi_use = float(row.get("hi", hi_use))
            hc_use = float(row.get("hc", hc_use))
            Ti_use = float(row.get("Ti", 0.0)) or None
            Tc_use = float(row.get("Tc", 0.0)) or None
        except Exception as e:
            st.error(f"Could not read coeff CSV: {e}")

    comp = hydrodynamic_components(loads.gamma_w, geom.L, geom.B, geom.H, Ci_use, Cc_use, hi_use, hc_use)
    Ah_i = ah_from_ZIR(Z, I_fac, R, Sa_i)
    Ah_c = ah_from_ZIR(Z, I_fac, R, Sa_c)

    # Effective liquid weights over plan footprint (kN): use γw*H per m²; scale by Ci/Cc
    Wi_eff = comp["mi"] * geom.L * geom.B / loads.gamma_w  # back to "kN" scale via ratios
    Wc_eff = comp["mc"] * geom.L * geom.B / loads.gamma_w

    # Base shears (kN)
    Vi = Ah_i * comp["mi"] * geom.L * geom.B
    Vc = Ah_c * comp["mc"] * geom.L * geom.B
    st.write(f"Impulsive base shear Vi ≈ **{Vi:.1f} kN**, Convective base shear Vc ≈ **{Vc:.1f} kN** "
             f"(Ah_i={Ah_i:.3f}, Ah_c={Ah_c:.3f})")

    # Build notional pressure profiles that integrate to these base shears
    z, p_i_kPa, p_c_kPa = eq_pressures_profile(H=geom.H,
                                               Pi_base=Vi/(max(geom.L,geom.B)),  # distribute per m of wall notionally
                                               Pc_base=Vc/(max(geom.L,geom.B)),
                                               hi=comp["hi"], hc=comp["hc"])

    col1, col2 = st.columns(2)
    with col1:
        figEi, axEi = plt.subplots()
        axEi.plot(p_i_kPa, z)
        axEi.invert_yaxis(); axEi.set_xlabel("Impulsive pressure [kPa]"); axEi.set_ylabel("z from base [m]")
        axEi.set_title("EQ — impulsive component")
        st.pyplot(figEi)
    with col2:
        figEc, axEc = plt.subplots()
        axEc.plot(p_c_kPa, z)
        axEc.invert_yaxis(); axEc.set_xlabel("Convective pressure [kPa]"); axEc.set_ylabel("z from base [m]")
        axEc.set_title("EQ — convective (sloshing)")
        st.pyplot(figEc)

    st.session_state["eq"] = {"Vi":Vi,"Vc":Vc,"Ah_i":Ah_i,"Ah_c":Ah_c,
                              "Ci":Ci_use,"Cc":Cc_use,"hi":hi_use,"hc":hc_use,
                              "figEi":figEi,"figEc":figEc}

# ---- TAB 4: ULS/SLS combination & bearing
with t4:
    st.subheader("ULS/SLS load combinations (IS 3370-2 Table 1) and bearing")
    # pick Table 1 row
    table1_keys = list(TABLE1_ROWS.keys())
    labels = [f"Row {k} — {TABLE1_ROWS[k]['state']}" for k in table1_keys]
    pick = st.selectbox("Pick Table 1 row", table1_keys, format_func=lambda k: f"Row {k} — {TABLE1_ROWS[k]['state']}")
    gam = TABLE1_ROWS[pick]
    state = gam["state"]

    # Optional “WL/EL” base shear or base moment from wind/quake; we will use Vi+Vc as WL/EL (auto)
    eq = st.session_state.get("eq", None)
    WLEL_force = (eq["Vi"] + eq["Vc"]) if eq else 0.0  # kN
    st.caption(f"WL/EL auto-taken = Vi+Vc = {WLEL_force:.1f} kN (IS 3370-2 Table 1 Note 3 guidance)")

    # Base weights
    W_base = mats.gamma_conc * geom.t_base * (geom.L * geom.B)
    W_walls = mats.gamma_conc * geom.t_wall * (2*geom.H*(geom.L + geom.B))
    W_water = loads.gamma_w * geom.L * geom.B * geom.H

    # Liquid thrust (unfactored)
    Rw_long = 0.5 * loads.gamma_w * geom.H**2 * geom.L
    Rw_short= 0.5 * loads.gamma_w * geom.H**2 * geom.B
    M_liq = max(Rw_long, Rw_short) * (geom.H/3.0)

    # Factored vertical
    W_tot = gam["DL"]*(W_base+W_walls) + gam["FL"]*W_water

    # Driving (horizontal): FL thrust factored + WL/EL factor*auto EQ (conservatively add)
    Driving_FL = gam["FL"] * max(Rw_long, Rw_short)
    Driving_EQ = gam["WL_EL"] * WLEL_force
    Driving_total = Driving_FL + Driving_EQ

    # Sliding
    Resisting = loads.mu_base * W_tot
    FS_slide = Resisting / Driving_total if Driving_total>0 else float("inf")

    # Bearing & eccentricity (use governing base width across which pressure varies)
    A = geom.L * geom.B
    B_dir = geom.B if (Rw_long >= Rw_short) else geom.L
    # Overturning moment: factored FL + add factored EQ moment approx as Driving_EQ * H/3
    M_total = gam["FL"]*M_liq + Driving_EQ * (geom.H/3.0)
    q0 = W_tot / A if A>0 else float("inf")
    e  = M_total / W_tot if W_tot>0 else 0.0

    st.write(f"**{state} Row {pick}** → W={W_tot:.1f} kN, Driving={Driving_total:.1f} kN (FL {Driving_FL:.1f} + WL/EL {Driving_EQ:.1f}), "
             f"FS_slide={FS_slide:.2f}")
    st.write(f"Bearing across B'={B_dir:.2f} m:  e={e:.3f} m;  q_avg={q0:.2f} kPa")
    if e <= B_dir/6:
        qmax = q0*(1 + 6*e/B_dir); qmin = q0*(1 - 6*e/B_dir)
        st.success(f"No tension (e ≤ B'/6).  q_min={qmin:.2f} kPa, q_max={qmax:.2f} kPa")
    else:
        st.error("Tension likely (e > B'/6). Consider thicker base, key/anchors, or reduce overturning.")

    st.session_state["uls"] = {"row":pick,"state":state,"W":W_tot,"Driving":Driving_total,
                               "FS_slide":FS_slide,"e":e,"q0":q0,"Bdir":B_dir,"M_total":M_total,
                               "Driving_FL":Driving_FL,"Driving_EQ":Driving_EQ}

# ---- TAB 5: Detailing & Table 5 minimum steel
with t5:
    st.subheader("Main steel at base + Table 5 minimum steel (per face surface zone)")
    cover_mm = cover
    Mb_design = st.session_state.get("Mb_design", base_values(loads.gamma_w, geom.H)[1])
    d_eff_mm = geom.t_wall*1000.0 - cover_mm - 0.5*max(pref_dias+[12])

    Ast_req = demand_ast_from_M(Mb_design, d_eff_mm, mats.fy)
    dia_v, s_v, Ast_v = pick_bar_spacing(Ast_req, pref_dias)

    st.write(f"Design base moment **M = {Mb_design:.2f} kNm/m**, effective depth **d ≈ {d_eff_mm:.0f} mm**")
    st.write(f"Required Ast (tension) ≈ **{Ast_req:.0f} mm²/m** → Adopt **{dia_v} mm @ {s_v} mm** (Ast≈{Ast_v:.0f} mm²/m)")

    # Table 5 minimum % check
    grade_label = "Fe250" if mats.fy<=300 else "Fe415/500"
    length_for_table5 = max(geom.L, geom.B)  # main bar direction length between movement joints (approx)
    perc_min = interpolate_table5(grade_label, geom.tank_type, length_for_table5)  # %
    # Required per face (surface zone) area over 1 m width:
    A_conc_face = 1000.0 * geom.t_wall*1000.0  # mm² per 1 m strip (gross); codes speak of surface zones — conservative use gross/face
    Ast_min_face = perc_min/100.0 * A_conc_face
    Ast_prov_face = Ast_v  # using vertical main face as the critical direction

    ok_min = Ast_prov_face >= Ast_min_face
    st.write(f"**Table 5 min steel (grade {grade_label}, {geom.tank_type}, length≈{length_for_table5:.1f} m)** → "
             f"ρ_min ≈ {perc_min:.2f}% → Ast_min(face)≈ **{Ast_min_face:.0f} mm²/m**")
    if ok_min:
        st.success("Meets Table 5 minimum reinforcement (per face).")
    else:
        st.error("Does **NOT** meet Table 5 minimum — increase bar area / reduce spacing / use two layers.")

    # stash for SLS & PDF
    st.session_state["detailing"] = {
        "Mb": Mb_design, "d_eff_mm": d_eff_mm,
        "Vert": {"dia_mm": dia_v, "spacing_mm": s_v, "Ast_mm2_per_m": Ast_v},
        "Table5": {"grade": grade_label, "tank_type": geom.tank_type,
                   "length_m": length_for_table5, "rho_min_percent": perc_min,
                   "Ast_min_face": Ast_min_face, "ok": ok_min}
    }

# ---- TAB 6: Crack control SLS (Tables 2/3/4)
with t6:
    st.subheader("SLS crack control per IS 3370-2 Tables 2/3 + Tightness Class (Table 4)")
    det = st.session_state.get("detailing", {})
    if not det:
        st.warning("Compute detailing first in Tab 5.")
    else:
        dia_v = det["Vert"]["dia_mm"]; s_v = det["Vert"]["spacing_mm"]; As_v = det["Vert"]["Ast_mm2_per_m"]
        d_eff_mm = det["d_eff_mm"]
        # Use service base moment = cantilever base (or table-based) without ULS factors
        M_service = st.session_state.get("Mb_design", base_values(loads.gamma_w, geom.H)[1])
        sigma_s = steel_stress_sls(M_service, d_eff_mm, As_v)
        sigma_allow = sigma_allow_for_detailing(dia_v, s_v, w_lim)

        colA,colB,colC = st.columns(3)
        colA.metric("M_service (kNm/m)", f"{M_service:.2f}")
        colB.metric("σs calc (MPa)", f"{sigma_s:.0f}")
        colC.metric("σs,max by code (MPa)", f"{sigma_allow:.0f}")
        if sigma_s <= sigma_allow:
            st.success(f"Crack control OK for Tightness Class {tight_class} (w_lim={w_lim:.2f} mm).")
        else:
            st.error("Crack control NOT OK — tighten spacing/use smaller bars/increase thickness.")

        st.session_state["sls"] = {"tight_class": tight_class, "w_lim": w_lim,
                                   "sigma_s": sigma_s, "sigma_allow": sigma_allow,
                                   "table2_row": (w_lim, TABLE2_SIGMA_LIMIT.get(w_lim, None)),
                                   "table3_row": (dia_v, s_v, TABLE3_SIGMA_BY_DETAIL.get((dia_v,s_v), None))}

# ---- TAB 7: Base Plan (key + anchors)
with t7:
    st.subheader("Base Plan Schematic (not to scale)")
    figPlan, axPl = plt.subplots(figsize=(6.0,4.6))
    axPl.plot([0, geom.L, geom.L, 0, 0], [0, 0, geom.B, geom.B, 0], lw=1.2, color="k", label="Base footprint")
    inset = key_width
    axPl.plot([inset, geom.L-inset, geom.L-inset, inset, inset],
              [inset, inset, geom.B-inset, geom.B-inset, inset],
              lw=1.2, ls="--", color="gray", label=f"Shear key (w={key_width:.2f} m, d={key_depth:.2f} m)")
    nx = max(1, int(geom.L/anchor_sx)); ny = max(1, int(geom.B/anchor_sy))
    xs = np.linspace(anchor_sx/2, geom.L-anchor_sx/2, nx)
    ys = np.linspace(anchor_sy/2, geom.B-anchor_sy/2, ny)
    for xi in xs:
        for yi in ys:
            axPl.plot(xi, yi, marker="o", ms=3)
    axPl.set_aspect('equal'); axPl.set_xlim(-0.1, geom.L+0.1); axPl.set_ylim(-0.1, geom.B+0.1)
    axPl.set_xlabel("Length b [m]"); axPl.set_ylabel("Width c [m]"); axPl.legend(loc="upper right")
    st.pyplot(figPlan)
    st.info(f"Anchors: ∅{anchor_dia:.0f} mm @ {anchor_sx:.2f} m × {anchor_sy:.2f} m (~{nx*ny} anchors).")
    st.session_state["figPlan"] = figPlan

# ---- TAB 8: PDF Export with table-row disclosures
with t8:
    st.subheader("Detailed PDF (with selected code table rows)")
    pdf_name = st.text_input("PDF file name", "water_tank_IS3370_report.pdf")
    if st.button("Create PDF"):
        df_vm = st.session_state.get("df_vm", cantilever_arrays(loads.gamma_w, geom.H, n=61))
        figP = st.session_state.get("figP", None)
        figEP = st.session_state.get("figEP", None)
        figV = st.session_state.get("figV", None)
        figM = st.session_state.get("figM", None)
        eq = st.session_state.get("eq", {})
        figEi = eq.get("figEi", None); figEc = eq.get("figEc", None)
        figPlan = st.session_state.get("figPlan", None)
        det = st.session_state.get("detailing", {})
        sls = st.session_state.get("sls", {})
        uls = st.session_state.get("uls", {})

        buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
        Wp, Hp = A4
        def para(lines, x=15*mm, y_start=Hp-25*mm, lead=5.2*mm, font="Helvetica", size=10):
            c.setFont(font, size); y = y_start
            for t in lines:
                c.drawString(x, y, t); y -= lead
                if y < 20*mm:
                    c.showPage(); c.setFont(font, size); y = Hp-20*mm
            return y

        # Cover & inputs
        c.setFont("Helvetica-Bold", 14); c.drawString(15*mm, Hp-18*mm, "RCC Water Tank — IS 3370-2 & IS 1893-2 Design Sheet")
        y = para([
            "1. INPUTS",
            f"Geometry: H={geom.H:.2f} m, Freeboard={geom.freeboard:.2f} m, b={geom.L:.2f} m, c={geom.B:.2f} m, "
            f"t_wall={geom.t_wall*1000:.0f} mm, t_base={geom.t_base*1000:.0f} mm, Tank={geom.tank_type}",
            f"Materials: fck={mats.fck} MPa, fy={mats.fy} MPa, γc={mats.gamma_conc:.2f} kN/m³, Ec≈{mats.Ec:.0f} MPa",
            f"Loads: γw={loads.gamma_w:.2f} kN/m³, EP via K0={loads.K0:.2f} and γs={loads.gamma_s:.1f} kN/m³",
            ""
        ])

        # Hydrostatics / EP plots
        c.showPage()
        c.setFont("Helvetica-Bold", 12); c.drawString(15*mm, Hp-18*mm, "2. FL & EP Distributions")
        for cap, fig in [("Liquid pressure (FL)", figP), ("Earth pressure (EP)", figEP)]:
            if fig is None: continue
            bio = io.BytesIO(); fig.savefig(bio, format="png", dpi=150, bbox_inches="tight"); bio.seek(0)
            c.drawString(15*mm, Hp-30*mm, cap)
            c.drawImage(ImageReader(bio), 15*mm, Hp-120*mm, width=85*mm, height=85*mm, preserveAspectRatio=True)

        # Shear/Moment
        c.showPage()
        c.setFont("Helvetica-Bold", 12); c.drawString(15*mm, Hp-18*mm, "3. V/M Derivation & Table")
        Mb = st.session_state.get("Mb_design", base_values(loads.gamma_w, geom.H)[1])
        y = para([
            "V(x)=½γw x² ; M(x)=γw x³/6   (cantilever, free top, fixed base)",
            f"Base Vb={0.5*loads.gamma_w*geom.H**2:.2f} kN/m ; Base M={Mb:.2f} kNm/m"
        ], y_start=Hp-28*mm)
        # small table
        show = df_vm.iloc[::3].head(40).copy()
        cols = ["Level from Top x (m)","Water Pressure p(x) (kPa)","Shear V(x) (kN/m)","Moment M(x) (kNm/m)"]
        data = [cols] + show[cols].round(3).values.tolist()
        y = Hp-48*mm; x = 15*mm; colw = [34*mm,38*mm,34*mm,38*mm]
        c.setFont("Helvetica-Bold", 9)
        for j,h in enumerate(data[0]): c.drawString(x+sum(colw[:j]), y, h)
        c.setFont("Helvetica", 9); y -= 6*mm
        for row in data[1:]:
            for j,v in enumerate(row): c.drawString(x+sum(colw[:j]), y, str(v))
            y -= 5.0*mm
            if y < 25*mm:
                c.showPage(); c.setFont("Helvetica-Bold", 9)
                for j,h in enumerate(data[0]): c.drawString(x+sum(colw[:j]), Hp-20*mm, h)
                c.setFont("Helvetica", 9); y = Hp-26*mm

        # EQ hydrodynamics plots + parameters
        if figEi or figEc:
            c.showPage()
            c.setFont("Helvetica-Bold", 12); c.drawString(15*mm, Hp-18*mm, "4. Earthquake Hydrodynamic Actions (IS 1893-2 style)")
            if figEi:
                bio = io.BytesIO(); figEi.savefig(bio, format="png", dpi=150, bbox_inches="tight"); bio.seek(0)
                c.drawString(15*mm, Hp-30*mm, "Impulsive pressure profile")
                c.drawImage(ImageReader(bio), 15*mm, Hp-120*mm, width=85*mm, height=85*mm, preserveAspectRatio=True)
            if figEc:
                bio = io.BytesIO(); figEc.savefig(bio, format="png", dpi=150, bbox_inches="tight"); bio.seek(0)
                c.drawString(110*mm, Hp-30*mm, "Convective (sloshing) profile")
                c.drawImage(ImageReader(bio), 110*mm, Hp-120*mm, width=85*mm, height=85*mm, preserveAspectRatio=True)
            lines = [
                f"Z={Z:.2f}, I={I_fac:.2f}, R={R:.2f}, Sa/g (imp)={Sa_i:.2f}, Sa/g (conv)={Sa_c:.2f}",
                f"Coefficients: Ci≈{eq.get('Ci','-')}, Cc≈{eq.get('Cc','-')}, hi≈{eq.get('hi','-'):.2f} m, hc≈{eq.get('hc','-'):.2f} m",
                f"Base shears: Vi≈{eq.get('Vi',0):.1f} kN, Vc≈{eq.get('Vc',0):.1f} kN"
            ]
            para(lines, y_start=Hp-130*mm)

        # ULS/SLS selection (Table 1)
        c.showPage()
        c.setFont("Helvetica-Bold", 12); c.drawString(15*mm, Hp-18*mm, "5. Load Combination & Bearing (IS 3370-2 Table 1)")
        if uls:
            row = uls["row"]; state = uls["state"]
            lines = [
                f"Selected: Table 1 — Row {row} ({state})  → γf: DL={TABLE1_ROWS[row]['DL']} IL={TABLE1_ROWS[row]['IL']} EP={TABLE1_ROWS[row]['EP']} FL={TABLE1_ROWS[row]['FL']} WL/EL={TABLE1_ROWS[row]['WL_EL']}",
                f"W={uls['W']:.1f} kN; Driving={uls['Driving']:.1f} kN (FL {uls['Driving_FL']:.1f} + WL/EL {uls['Driving_EQ']:.1f}); FS_slide={uls['FS_slide']:.2f}",
                f"Bearing: q_avg={uls['q0']:.2f} kPa; e={uls['e']:.3f} m against B'={uls['Bdir']:.2f} m"
            ]
            para(lines, y_start=Hp-28*mm)

        # Detailing + Table 5 + Table 2/3/4
        c.showPage()
        c.setFont("Helvetica-Bold", 12); c.drawString(15*mm, Hp-18*mm, "6. Detailing, Minimum Steel (Table 5) & Crack Control (Tables 2/3/4)")
        if det:
            t5 = det.get("Table5", {})
            dlines = [
                f"Main bars (water face): {det['Vert']['dia_mm']} mm @ {det['Vert']['spacing_mm']} mm (Ast≈{det['Vert']['Ast_mm2_per_m']:.0f} mm²/m)",
                f"Table 5 — grade={t5.get('grade','?')}, tank={t5.get('tank_type','?')}, length≈{t5.get('length_m',0):.1f} m",
                f"ρ_min={t5.get('rho_min_percent',0):.2f}% → Ast_min(face)≈{t5.get('Ast_min_face',0):.0f} mm²/m → {'OK' if t5.get('ok',False) else 'NOT OK'}"
            ]
            para(dlines, y_start=Hp-28*mm)
        if sls:
            s2 = sls.get("table2_row", (None,None))
            s3 = sls.get("table3_row", (None,None,None))
            slines = [
                f"Tightness Class={sls['tight_class']} → w_lim={sls['w_lim']:.2f} mm",
                f"Table 2 row: w={s2[0]} mm → σs,max={s2[1]} MPa",
                f"Table 3 row: dia={s3[0]} mm, spacing={s3[1]} mm → σs,max={s3[2]} MPa",
                f"Calculated σs={sls['sigma_s']:.0f} MPa {'≤' if sls['sigma_s']<=sls['sigma_allow'] else '>'} σs,max={sls['sigma_allow']:.0f} MPa"
            ]
            para(slines, y_start=Hp-60*mm)

        # Base plan
        if figPlan:
            c.showPage()
            c.setFont("Helvetica-Bold", 12); c.drawString(15*mm, Hp-18*mm, "7. Base Plan (shear key & anchors) — schematic")
            bio = io.BytesIO(); figPlan.savefig(bio, format="png", dpi=150, bbox_inches="tight"); bio.seek(0)
            c.drawImage(ImageReader(bio), 15*mm, 28*mm, width=175*mm, height=125*mm, preserveAspectRatio=True, anchor='sw')

        c.save(); buf.seek(0)
        st.download_button("Download PDF", data=buf, file_name=pdf_name, mime="application/pdf")
        st.success("PDF ready.")
