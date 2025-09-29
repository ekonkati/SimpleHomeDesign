
# app.py — Biaxial Footing Designer (Streamlit Skeleton, Patched)
# ---------------------------------------------------------------
# Tabs: Inputs → Bearing/Contact → Stability → Shear → Flexure → Pedestal → Anchors → Detailing → Output
# Units (default): kN, m, MPa (N/mm^2). Convert carefully inside calc functions.

import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Helpers & Data Structures
# -----------------------------

@dataclass
class Materials:
    fck: float     # MPa
    fy: float      # MPa
    gamma_c: float # kN/m^3
    mu_base: float # friction coefficient at base

@dataclass
class Geometry:
    B: float  # footing width (m, local X)
    L: float  # footing length (m, local Y)
    D: float  # footing thickness (m)
    cover_bot: float  # m
    cover_top: float  # m

@dataclass
class Pedestal:
    bp: float  # m
    lp: float  # m
    hp: float  # m

@dataclass
class Loads:
    N_serv: float  # kN (service)
    Mx_serv: float # kN·m (service)
    My_serv: float # kN·m (service)
    Hx_serv: float # kN (service)
    Hy_serv: float # kN (service)

    N_uls: float   # kN (ULS)
    Mx_uls: float  # kN·m (ULS)
    My_uls: float  # kN·m (ULS)
    Hx_uls: float  # kN (ULS)
    Hy_uls: float  # kN (ULS)

@dataclass
class Soil:
    qall_sls: float      # kPa (kN/m^2) allowable service bearing
    gamma_soil: float    # kN/m^3
    water_table_at_base: bool

BAR_DIAMETERS = [12, 16, 20, 25, 28, 32]  # mm

# -----------------------------
# Core Calculations
# -----------------------------

def ecc(N: float, Mx: float, My: float) -> Tuple[float, float]:
    """Return eccentricities ex, ey (m). Handle N≈0 safely."""
    if abs(N) < 1e-6:
        return float('inf'), float('inf')
    return Mx / N, My / N

def full_contact_pressures(N: float, Mx: float, My: float, B: float, L: float) -> Dict[str, Any]:
    """Full-contact biaxial linear pressure distribution. Returns qmax, qmin (kPa), ex, ey, within_kern."""
    ex, ey = ecc(N, Mx, My)
    within_kern = (abs(ex) <= B/6.0) and (abs(ey) <= L/6.0)
    q0 = N / (B*L)  # kN/m^2
    qmax = q0 * (1 + 6*abs(ex)/B + 6*abs(ey)/L)
    qmin = q0 * (1 - 6*abs(ex)/B - 6*abs(ey)/L)
    return {"ex": ex, "ey": ey, "within_kern": within_kern, "qmax": qmax, "qmin": qmin}

def partial_contact_effective_dims(N: float, Mx: float, My: float, B: float, L: float) -> Dict[str, Any]:
    """Conservative partial-contact approach: reduced dimensions Bc, Lc; qmax ≈ 4N/(Bc*Lc)."""
    ex, ey = ecc(N, Mx, My)
    Bc = B - 6.0*abs(ex)
    Lc = L - 6.0*abs(ey)
    feasible = (Bc > 0) and (Lc > 0)
    qmax = 4.0 * N / max(1e-6, (Bc * Lc))  # kN/m^2
    qavg = N / max(1e-6, (Bc * Lc))
    return {"ex": ex, "ey": ey, "Bc": Bc, "Lc": Lc, "feasible": feasible, "qmax": qmax, "qavg": qavg}

def base_weight(geom: Geometry, mat: Materials) -> float:
    """Self-weight of footing block (kN), ignoring pedestal for now (added separately)."""
    return geom.B * geom.L * geom.D * mat.gamma_c

def stability_checks_service(loads: Loads, geom: Geometry, soil: Soil, mat: Materials) -> Dict[str, Any]:
    """Simple sliding/uplift snapshot at service. Overturning by bearing kernel test already."""
    Wf = base_weight(geom, mat)
    Hres = (loads.Hx_serv**2 + loads.Hy_serv**2) ** 0.5
    N_base = loads.N_serv + Wf
    R_cap = mat.mu_base * max(0.0, N_base)
    return {"Wf": Wf, "Hres": Hres, "N_base": N_base, "R_cap": R_cap,
            "sliding_ok": Hres <= R_cap, "uplift_ok": (loads.N_serv + Wf) > 0.0}

def eff_depth(geom: Geometry, bar_d_mm: float) -> float:
    """Effective depth d (m) using bottom cover and half bar dia (simple)."""
    d = geom.D - (geom.cover_bot + (bar_d_mm/1000.0)/2.0)
    return max(0.0, d)

def one_way_shear_ULS(q_design: float, span_m: float, d: float, strip_width: float=1.0) -> float:
    """Very simplified one-way shear demand Vu on a strip of unit width."""
    return q_design * span_m * strip_width  # kN

def punching_perimeter(bp: float, lp: float, d: float) -> float:
    """Critical punching perimeter b0 at distance d from pedestal loaded area (m)."""
    return 2.0 * ((bp + 2*d) + (lp + 2*d))

def design_flexure_As(Mu_kNm: float, d_m: float, fy_MPa: float, fck_MPa: float) -> float:
    """Quick As (mm^2/m) approximation; enforce simple min steel placeholder."""
    if d_m <= 0:
        return 0.0
    Mu_Nmm = Mu_kNm * 1e6
    z_mm = 0.9 * d_m * 1000.0
    As_mm2_per_m = Mu_Nmm / max(1e-6, (0.87 * fy_MPa * z_mm))
    d_mm = d_m * 1000.0
    As_min = 0.0012 * (1000.0 * d_mm)
    return max(As_mm2_per_m, As_min)

def bar_schedule_from_As(As_req_mm2_per_m: float, preferred_dia_mm_list=BAR_DIAMETERS, max_spacing_mm=250) -> Dict[str, Any]:
    """Suggest bar dia and spacing for a target As per meter."""
    best = None
    for dia in preferred_dia_mm_list:
        area = math.pi * (dia**2) / 4.0
        spacing_mm = min(max_spacing_mm, (area * 1000.0) / max(1e-6, As_req_mm2_per_m))
        spacing_mm = max(75.0, round(spacing_mm / 10.0) * 10.0)
        As_prov = area * (1000.0 / spacing_mm)
        ok = As_prov >= As_req_mm2_per_m
        candidate = {"dia_mm": dia, "spacing_mm": spacing_mm, "As_prov_mm2_per_m": As_prov, "meets": ok}
        if ok and (best is None or As_prov < best["As_prov_mm2_per_m"]):
            best = candidate
    if best is None:
        dia = preferred_dia_mm_list[-1]
        area = math.pi * (dia**2) / 4.0
        spacing_mm = 75.0
        As_prov = area * (1000.0 / spacing_mm)
        best = {"dia_mm": dia, "spacing_mm": spacing_mm, "As_prov_mm2_per_m": As_prov, "meets": As_prov >= As_req_mm2_per_m}
    return best

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Biaxial Footing Designer (Skeleton)", layout="wide")

st.title("🧱 Biaxial Footing Designer — Streamlit Skeleton (Patched)")
st.caption("Tabs: Inputs → Bearing/Contact → Stability → Shear → Flexure → Pedestal → Anchors → Detailing → Output")

tabs = st.tabs([
    "Inputs", "Bearing/Contact", "Stability", "Shear", "Flexure", "Pedestal", "Anchors", "Detailing", "Output"
])

# ---------- Inputs Tab ----------
with tabs[0]:
    st.subheader("1) Inputs")

    st.markdown("**Units**: kN, m, MPa (N/mm²).")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Geometry**")
        B = st.number_input("Footing width B (m, local X)", min_value=0.1, value=2.0, step=0.1)
        L = st.number_input("Footing length L (m, local Y)", min_value=0.1, value=2.5, step=0.1)
        D = st.number_input("Footing thickness D (m)", min_value=0.2, value=0.6, step=0.05)
        cover_bot = st.number_input("Bottom cover (m)", min_value=0.03, value=0.06, step=0.01)
        cover_top = st.number_input("Top cover (m)", min_value=0.03, value=0.05, step=0.01)

    with col2:
        st.markdown("**Materials & Soil**")
        fck = st.selectbox("Concrete grade fck (MPa)", [20, 25, 30, 35, 40, 50], index=2)
        fy = st.selectbox("Steel grade fy (MPa)", [415, 500], index=1)
        gamma_c = st.number_input("Concrete unit weight γc (kN/m³)", min_value=20.0, value=25.0, step=0.5)
        mu_base = st.number_input("Base friction coeff μ", min_value=0.2, value=0.5, step=0.05)
        qall_sls = st.number_input("Allowable bearing q_allow (kPa = kN/m²)", min_value=50.0, value=200.0, step=10.0)
        gamma_soil = st.number_input("Soil unit weight γsoil (kN/m³)", min_value=16.0, value=18.0, step=0.5)
        wtb = st.checkbox("Water table at base level", value=False)

    with col3:
        st.markdown("**Loads**")
        N_serv = st.number_input("N (service) kN (compression +)", value=1500.0, step=10.0)
        Mx_serv = st.number_input("Mx (service) kN·m", value=100.0, step=5.0)
        My_serv = st.number_input("My (service) kN·m", value=80.0, step=5.0)
        Hx_serv = st.number_input("Hx (service) kN", value=0.0, step=1.0)
        Hy_serv = st.number_input("Hy (service) kN", value=0.0, step=1.0)
        st.divider()
        N_uls = st.number_input("N (ULS) kN", value=1800.0, step=10.0)
        Mx_uls = st.number_input("Mx (ULS) kN·m", value=130.0, step=5.0)
        My_uls = st.number_input("My (ULS) kN·m", value=110.0, step=5.0)
        Hx_uls = st.number_input("Hx (ULS) kN", value=0.0, step=1.0)
        Hy_uls = st.number_input("Hy (ULS) kN", value=0.0, step=1.0)

    st.markdown("**Pedestal (for punching & bearing)**")
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        bp = st.number_input("Pedestal bp (m) along B", min_value=0.2, value=0.6, step=0.05)
    with colp2:
        lp = st.number_input("Pedestal lp (m) along L", min_value=0.2, value=0.6, step=0.05)
    with colp3:
        hp = st.number_input("Pedestal height hp (m)", min_value=0.2, value=0.6, step=0.05)

    st.session_state["geom"] = Geometry(B, L, D, cover_bot, cover_top)
    st.session_state["mat"] = Materials(fck, fy, gamma_c, mu_base)
    st.session_state["soil"] = Soil(qall_sls, gamma_soil, wtb)
    st.session_state["loads"] = Loads(N_serv, Mx_serv, My_serv, Hx_serv, Hy_serv, N_uls, Mx_uls, My_uls, Hx_uls, Hy_uls)
    st.session_state["ped"] = Pedestal(bp, lp, hp)

    st.info("➡ Proceed to **Bearing/Contact** tab to evaluate full vs partial contact pressures.")

# ---------- Bearing/Contact Tab ----------
with tabs[1]:
    st.subheader("2) Bearing / Contact (Service)")
    geom: Geometry = st.session_state["geom"]
    soil: Soil = st.session_state["soil"]
    loads: Loads = st.session_state["loads"]

    res = full_contact_pressures(loads.N_serv, loads.Mx_serv, loads.My_serv, geom.B, geom.L)
    st.write(f"Eccentricities: e_x = **{res['ex']:.3f} m**, e_y = **{res['ey']:.3f} m**")
    st.write(f"Kern check: within kern? **{res['within_kern']}**")
    st.write(f"Full-contact pressures: q_max = **{res['qmax']:.1f} kPa**, q_min = **{res['qmin']:.1f} kPa**")
    ok_full = (res["within_kern"] and (res["qmax"] <= soil.qall_sls) and (res["qmin"] >= 0.0))
    if ok_full:
        st.success("Full contact OK against allowable bearing.")
    else:
        st.warning("Full contact NOT OK → consider partial contact / resizing.")

    with st.expander("Partial Contact (no-tension) — Conservative"):
        par = partial_contact_effective_dims(loads.N_serv, loads.Mx_serv, loads.My_serv, geom.B, geom.L)
        st.write(f"Effective dims: Bc = **{par['Bc']:.3f} m**, Lc = **{par['Lc']:.3f} m**, feasible: **{par['feasible']}**")
        st.write(f"Conservative q_max ≈ **{par['qmax']:.1f} kPa**, q_avg ≈ **{par['qavg']:.1f} kPa**")
        if par["feasible"] and (par["qmax"] <= soil.qall_sls):
            st.success("Partial contact OK against allowable bearing.")
        else:
            st.error("Partial contact NOT OK → increase plan size or improve ground.")

# ---------- Stability Tab ----------
with tabs[2]:
    st.subheader("3) Stability (Service)")
    geom: Geometry = st.session_state["geom"]
    soil: Soil = st.session_state["soil"]
    mat: Materials = st.session_state["mat"]
    loads: Loads = st.session_state["loads"]

    stab = stability_checks_service(loads, geom, soil, mat)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Self-weight Wf (kN)", f"{stab['Wf']:.1f}")
    c2.metric("Horiz. resultants H (kN)", f"{stab['Hres']:.1f}")
    c3.metric("Friction cap μ·N_base (kN)", f"{stab['R_cap']:.1f}")
    c4.metric("N_base (kN)", f"{stab['N_base']:.1f}")

    st.write("Sliding OK?" , stab["sliding_ok"])
    st.write("Uplift OK?" , stab["uplift_ok"])
    st.info("Refinements: include soil overburden, passive resistance, shear key if adopted; document assumptions.")

# ---------- Shear Tab ----------
with tabs[3]:
    st.subheader("4) Shear (ULS) — Skeleton")
    geom: Geometry = st.session_state["geom"]
    ped: Pedestal = st.session_state["ped"]
    mat: Materials = st.session_state["mat"]
    loads: Loads = st.session_state["loads"]

    q_uls = loads.N_uls / (geom.B * geom.L)  # kPa
    st.write(f"Assumed ULS soil pressure (placeholder): **{q_uls:.1f} kPa**")

    bar_d = st.selectbox("Trial main bar dia (mm) for d calc", BAR_DIAMETERS, index=2)
    d_m = eff_depth(geom, bar_d)

    st.write(f"Effective depth d ≈ **{d_m:.3f} m** (with {bar_d} mm bars)")

    cant_B = (geom.B - ped.bp) / 2.0
    cant_L = (geom.L - ped.lp) / 2.0
    Vu_x = one_way_shear_ULS(q_uls, cant_L, d_m)
    Vu_y = one_way_shear_ULS(q_uls, cant_B, d_m)

    c1, c2 = st.columns(2)
    c1.metric("One-way shear Vu_x (kN/m strip)", f"{Vu_x:.1f}")
    c2.metric("One-way shear Vu_y (kN/m strip)", f"{Vu_y:.1f}")

    b0 = punching_perimeter(ped.bp, ped.lp, d_m)
    st.write(f"Punching critical perimeter b0 ≈ **{b0:.3f} m**")
    st.info("👉 Finalize τ_v vs τ_c checks per IS 456 (punching & beam shear).")

# ---------- Flexure Tab ----------
with tabs[4]:
    st.subheader("5) Flexure (ULS) — Skeleton & Auto Bar-Schedule")
    geom: Geometry = st.session_state["geom"]
    ped: Pedestal = st.session_state["ped"]
    mat: Materials = st.session_state["mat"]
    loads: Loads = st.session_state["loads"]

    q_uls = loads.N_uls / (geom.B * geom.L)  # kPa
    cant_B = (geom.B - ped.bp) / 2.0
    cant_L = (geom.L - ped.lp) / 2.0

    Mu_x = q_uls * (cant_L**2) / 2.0  # kN·m/m
    Mu_y = q_uls * (cant_B**2) / 2.0  # kN·m/m

    bar_d = st.selectbox("Design bar dia (mm)", BAR_DIAMETERS, index=2, key="flex_bar_d")
    d_m = eff_depth(geom, bar_d)

    As_x = design_flexure_As(Mu_x, d_m, mat.fy, mat.fck)  # mm²/m
    As_y = design_flexure_As(Mu_y, d_m, mat.fy, mat.fck)  # mm²/m

    st.write(f"Design moments (placeholder): Mu_x = **{Mu_x:.2f} kN·m/m**, Mu_y = **{Mu_y:.2f} kN·m/m**")
    c1, c2 = st.columns(2)
    c1.metric("As_req (X-bottom) mm²/m", f"{As_x:.0f}")
    c2.metric("As_req (Y-bottom) mm²/m", f"{As_y:.0f}")

    sched_x = bar_schedule_from_As(As_x)
    sched_y = bar_schedule_from_As(As_y)

    st.markdown("**Auto bar-schedule suggestions (bottom):**")
    c3, c4 = st.columns(2)
    with c3:
        st.write(f"X-direction: {sched_x['dia_mm']} mm @ {sched_x['spacing_mm']} mm  (As_prov ≈ {sched_x['As_prov_mm2_per_m']:.0f} mm²/m)")
    with c4:
        st.write(f"Y-direction: {sched_y['dia_mm']} mm @ {sched_y['spacing_mm']} mm  (As_prov ≈ {sched_y['As_prov_mm2_per_m']:.0f} mm²/m)")

    st.info("Add top mats where uplift/negative moments under pedestal or anchor forces occur. Replace placeholders with full IS 456 flexural design.")

# ---------- Pedestal Tab ----------
with tabs[5]:
    st.subheader("6) Pedestal — Bearing & Ties (Skeleton)")
    st.write("**Concrete bearing under pedestal on footing (IS 456 enhancement):**")
    st.latex(r"\sigma_{c,allow} \le 0.45 f_{ck} \sqrt{A_1/A_2}")
    st.write("Define A1 (loaded area) and A2 (supporting area within permitted dispersion). Provide vertical bars and ties as needed.")

# ---------- Anchors Tab (Patched) ----------
with tabs[6]:
    st.subheader("7) Anchors — Layout & Checks (Skeleton)")
    st.markdown("If anchors are required (uplift/overturning), compute bolt-group tensions from N and Mx, My.")
    st.code(
        '''
# Example: tension in bolts of a rectangular group about centroid
# (Only bolts in tension considered)
T_i = N_t/n_t + (Mx*y_i)/sum(y**2) + (My*x_i)/sum(x**2)

# Then check (per ACI 318-19 Chapter 17 or project spec):
# - Steel tension/yield
# - Concrete breakout (tension), pull-out, side-face blowout
# - Shear breakout / pry-out and steel shear
# - Edge distances, spacing, embedment
# Provide hairpins/ties around anchor group to control splitting.
        ''',
        language="python"
    )
    st.info("Provide base plate geometry, grout thickness, and minimum edge cover in concrete.")

# ---------- Detailing Tab ----------
with tabs[7]:
    st.subheader("8) Detailing — Mesh & Notes (Skeleton)")
    st.write("**Bottom mesh:** from Flexure tab (X & Y may differ).")
    st.write("**Top mesh:** provide under pedestal / uplift zones; ensure code minimums and spacing limits.")
    st.write("**Cover & laps:** respect clear cover and Ld per IS 456; stagger laps away from peak moments.")
    st.write("**Shear reinforcement:** if adopted for one-way shear or punching (stud rails/links).")

# ---------- Output Tab ----------
with tabs[8]:
    st.subheader("9) Output — Summary (Skeleton)")
    geom: Geometry = st.session_state["geom"]
    ped: Pedestal = st.session_state["ped"]
    mat: Materials = st.session_state["mat"]
    soil: Soil = st.session_state["soil"]
    loads: Loads = st.session_state["loads"]

    res = full_contact_pressures(loads.N_serv, loads.Mx_serv, loads.My_serv, geom.B, geom.L)
    par = partial_contact_effective_dims(loads.N_serv, loads.Mx_serv, loads.My_serv, geom.B, geom.L)

    df = pd.DataFrame({
        "Item": ["B (m)", "L (m)", "D (m)",
                 "Ped bp (m)", "Ped lp (m)", "Ped hp (m)",
                 "fck (MPa)", "fy (MPa)",
                 "q_allow (kPa)",
                 "e_x (m)", "e_y (m)",
                 "Full contact?", "qmax_full (kPa)", "qmin_full (kPa)",
                 "Bc (m)", "Lc (m)", "qmax_partial (kPa)"
                ],
        "Value": [geom.B, geom.L, geom.D,
                  ped.bp, ped.lp, ped.hp,
                  mat.fck, mat.fy,
                  soil.qall_sls,
                  res["ex"], res["ey"],
                  res["within_kern"], res["qmax"], res["qmin"],
                  par["Bc"], par["Lc"], par["qmax"]
                 ]
    })
    st.dataframe(df, use_container_width=True)
    st.success("Patched version: removed any accidental execution/doc rendering in code blocks.")
