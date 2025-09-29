# RCC Column Design App (Biaxial + Axial ± Shear) — IS 456 + IS 13920 oriented
# Single-file Streamlit app with tabs:
# Inputs → Slenderness → Moments → Interaction → Shear → Detailing → Output
# Includes: SVG cross-section drawing (bars + ties + leader IDs)

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st

# -----------------------------
# Constants & Utility Functions
# -----------------------------
ES = 200000.0  # MPa (N/mm2)
EPS_CU = 0.0035  # Ultimate concrete compressive strain
FY_DEFAULT = 500.0  # MPa


def Ec_from_fck(fck: float) -> float:
    """IS 456: Ec = 5000 * sqrt(fck) (MPa)."""
    return 5000.0 * math.sqrt(max(fck, 1e-6))


@dataclass
class Section:
    b: float  # mm (width along local x-axis)
    D: float  # mm (depth along local y-axis)
    cover: float  # mm (clear cover to main bars)
    bars: List[Tuple[float, float, float]]  # list of (x, y, dia_mm)

    @property
    def Ag(self) -> float:
        return self.b * self.D

    @property
    def Ic_x(self) -> float:
        # About local x-axis (neutral axis // to x, bending about x means compression at top/bottom along y)
        return self.b * self.D**3 / 12.0

    @property
    def Ic_y(self) -> float:
        # About local y-axis (neutral axis // to y, bending about y means compression at left/right along x)
        return self.D * self.b**3 / 12.0

    @property
    def rx(self) -> float:
        return math.sqrt(self.Ic_x / self.Ag)

    @property
    def ry(self) -> float:
        return math.sqrt(self.Ic_y / self.Ag)


# -----------------------------
# Rebar geometry helpers
# -----------------------------

def bar_area(dia_mm: float) -> float:
    return math.pi * (dia_mm ** 2) / 4.0


def generate_rectangular_bar_layout(b: float, D: float, cover: float, 
                                    n_top: int, n_bot: int, n_left: int, n_right: int, 
                                    dia_top: float, dia_bot: float, dia_side: float) -> List[Tuple[float, float, float]]:
    """
    Returns list of (x, y, dia) for a rectangular tied column.
    Coordinate system:
      - origin at top-left corner of concrete rectangle
      - x to right (0..b), y downward (0..D)
    Bars placed at cover + 0.5*dia from faces.
    """
    bars = []
    # helper for linear spacing
    def linspace(a, c, n):
        if n == 1:
            return [0.5 * (a + c)]
        return [a + i * (c - a) / (n - 1) for i in range(n)]

    # Top row
    y_top = cover + dia_top / 2.0
    x_span_left = cover + dia_side / 2.0
    x_span_right = b - (cover + dia_side / 2.0)
    if n_top > 0:
        for x in linspace(x_span_left, x_span_right, n_top):
            bars.append((x, y_top, dia_top))

    # Bottom row
    y_bot = D - (cover + dia_bot / 2.0)
    if n_bot > 0:
        for x in linspace(x_span_left, x_span_right, n_bot):
            bars.append((x, y_bot, dia_bot))

    # Left column (excluding top/bottom corners to avoid duplicates)
    x_left = cover + dia_side / 2.0
    y_span_top = cover + dia_top / 2.0
    y_span_bot = D - (cover + dia_bot / 2.0)
    if n_left > 2:
        for y in linspace(y_span_top, y_span_bot, n_left)[1:-1]:
            bars.append((x_left, y, dia_side))

    # Right column
    x_right = b - (cover + dia_side / 2.0)
    if n_right > 2:
        for y in linspace(y_span_top, y_span_bot, n_right)[1:-1]:
            bars.append((x_right, y, dia_side))

    return bars


# -----------------------------
# Slenderness & 2nd-order
# -----------------------------

def effective_length_factor(restraint: str) -> float:
    """Very simplified k-factor pick for demonstration (engineer to override)."""
    mapping = {
        "Fixed-Fixed": 0.65,
        "Fixed-Pinned": 0.80,
        "Pinned-Pinned": 1.00,
        "Fixed-Free (cantilever)": 2.0,
    }
    return mapping.get(restraint, 1.0)


def moment_magnifier(Pu: float, le_mm: float, Ec: float, Ic: float, Cm: float = 0.85, sway: bool = False) -> float:
    """Compute simple moment magnifier δ = 1/(1 - Pu/Pcr). Caps δ between 1 and 2.5 by default.
    This is a teaching/preview implementation; use code-accurate magnifiers for production.
    """
    Pcr = (math.pi ** 2) * Ec * Ic / (le_mm ** 2 + 1e-9)
    if Pcr <= 0:
        return 1.0
    delta = 1.0 / max(1e-6, (1.0 - Pu / Pcr))
    # sway frames typically larger magnifier; apply Cm reduction for non-sway
    if not sway:
        delta = max(1.0, Cm * delta)
    delta = float(np.clip(delta, 1.0, 2.5))
    return delta


# -----------------------------
# Strain compatibility (uniaxial capacity)
# -----------------------------

def uniaxial_capacity_Mu_for_Pu(section: Section, fck: float, fy: float, Pu: float, axis: str) -> float:
    """
    Compute ultimate uniaxial moment capacity Mu,lim for a given factored axial Pu (N) and bending axis.
    axis: 'x' (bending about x-axis, compression at top) or 'y' (bending about y-axis, compression at left).
    Assumptions: IS 456 rectangular stress block (0.36 fck*b*xu at 0.42 xu), EPS_CU at extreme fiber.
    Bars: elastic-perfectly plastic with 0.87*fy cap, ES = 200000 MPa.
    Returns Mu in Nmm.
    """
    b, D = section.b, section.D
    Ec = Ec_from_fck(fck)

    # Iterate neutral axis depth c (mm) to satisfy axial equilibrium for given Pu.
    # Search over reasonable range of c.
    c_min = 0.05 * (D if axis == 'x' else b)
    c_max = 1.50 * (D if axis == 'x' else b)

    def forces_and_moment(c: float):
        # Concrete block contribution
        if axis == 'x':
            xu = min(c, D)  # limit to section depth
            Cc = 0.36 * fck * b * xu  # N/mm2 * mm * mm = N
            arm_Cc = (D / 2.0) - (0.42 * xu)  # lever arm to section centroid (y measured downwards)
            Mc = Cc * arm_Cc
        else:  # axis == 'y'
            xu = min(c, b)
            Cc = 0.36 * fck * D * xu
            arm_Cc = (b / 2.0) - (0.42 * xu)
            Mc = Cc * arm_Cc

        # Steel bars
        Fs = 0.0
        Ms = 0.0
        for (x, y, dia) in section.bars:
            As = bar_area(dia)
            if axis == 'x':
                # compression at top y=0, neutral axis at y=c
                strain = EPS_CU * (1.0 - (y / max(c, 1e-6)))
                # positive strain -> compression if y < c (above NA); negative -> tension
                stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
                force = stress * As
                # lever arm to centroid at D/2
                z = (D / 2.0) - y
            else:
                # compression at left x=0
                strain = EPS_CU * (1.0 - (x / max(c, 1e-6)))
                stress = np.clip(ES * strain, -0.87 * fy, 0.87 * fy)
                force = stress * As
                z = (b / 2.0) - x

            Fs += force
            Ms += force * z

        N_res = Cc + Fs  # resultant axial (compression +)
        M_res = Mc + Ms  # about centroidal axis
        return N_res, M_res

    # Binary search on c to match Pu (target axial). We look for N_res ≈ Pu.
    target = Pu
    cL, cR = c_min, c_max
    NL, ML = forces_and_moment(cL)
    NR, MR = forces_and_moment(cR)

    # If target outside bracket, clamp
    if (NL - target) * (NR - target) > 0:
        # fallback: choose c giving closer N
        candidates = [(abs(NL - target), ML), (abs(NR - target), MR)]
        Mu = min(candidates, key=lambda t: t[0])[1]
        return float(Mu)

    for _ in range(60):
        cm = 0.5 * (cL + cR)
        Nm, Mm = forces_and_moment(cm)
        if (Nm - target) == 0:
            return float(Mm)
        if (NL - target) * (Nm - target) <= 0:
            cR, NR, MR = cm, Nm, Mm
        else:
            cL, NL, ML = cm, Nm, Mm
    # After iterations, take midpoint
    Mu = 0.5 * (ML + MR)
    return float(Mu)


# -----------------------------
# SVG Cross-section drawing
# -----------------------------

def svg_cross_section(section: Section, tie_dia: float, tie_spacing: float) -> str:
    """Return an SVG string showing the column, bars, and a sample tie rectangle with leader labels."""
    b, D = section.b, section.D
    scale = 1.0  # 1 mm = 1 px (adjust if large)
    pad = 40  # px around
    W = int(b * scale + 2 * pad)
    H = int(D * scale + 2 * pad)

    def px(val_mm):
        return val_mm * scale + pad

    # SVG header
    parts = [
        f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
        '<style> .txt{font: 12px sans-serif;} .lbl{font: 11px sans-serif;}</style>'
    ]

    # Concrete rectangle
    parts.append(f'<rect x="{px(0)}" y="{px(0)}" width="{b*scale}" height="{D*scale}" fill="#f2f2f2" stroke="#555" stroke-width="1"/>')

    # Tie (one pitch shown as a rectangle inset by cover) – simplified visual
    cov = section.cover
    parts.append(
        f'<rect x="{px(cov)}" y="{px(cov)}" width="{(b-2*cov)*scale}" height="{(D-2*cov)*scale}" fill="none" stroke="#888" stroke-dasharray="4,3" stroke-width="1"/>'
    )

    # Bars
    for i, (x, y, dia) in enumerate(section.bars, start=1):
        cx, cy, r = px(x), px(y), (dia * scale) / 2.0
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#0a66c2" stroke="#043f77" stroke-width="1"/>')
        # leader
        lx, ly = cx + 30, cy - 10
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{lx}" y2="{ly}" stroke="#333" stroke-width="1"/>')
        parts.append(f'<text x="{lx+4}" y="{ly-4}" class="lbl">Bar {i}: {int(dia)}mm</text>')

    # Labels
    parts.append(f'<text x="{px(b/2)}" y="{px(-10)}" class="txt" text-anchor="middle">b = {b:.0f} mm</text>')
    parts.append(f'<text x="{px(-10)}" y="{px(D/2)}" class="txt" text-anchor="end" transform="rotate(-90 {px(-10)} {px(D/2)})">D = {D:.0f} mm</text>')
    parts.append(f'<text x="{px(b+5)}" y="{px(15)}" class="lbl">Tie: {int(tie_dia)} mm @ {int(tie_spacing)} mm c/c</text>')

    parts.append('</svg>')
    return "\n".join(parts)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RCC Column (Biaxial) Designer", layout="wide")
st.title("RCC Column Design — Biaxial Moments ± Axial ± Shear (IS 456/13920)")

with st.sidebar:
    st.header("Global Settings")
    code_context = st.selectbox("Design Context", ["IS 456 + IS 13920 (India)", "Generic Euro/ACI-like"], index=0)
    st.caption("This app uses teaching-grade formulas; verify with your code provisions/design aids.")

# Tabs
T1, T2, T3, T4, T5, T6, T7 = st.tabs([
    "Inputs", "Slenderness", "Moments", "Interaction", "Shear", "Detailing", "Output"
])

# Shared state
if "state" not in st.session_state:
    st.session_state.state = {}

state = st.session_state.state

# -----------------------------
# Tab 1: Inputs
# -----------------------------
with T1:
    st.subheader("1) Inputs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        b = st.number_input("Width b (mm)", 200.0, 2000.0, 450.0, 25.0)
        fck = st.number_input("fck (MPa)", 20.0, 80.0, 30.0, 1.0)
        sway = st.checkbox("Sway Frame?", value=False)
    with c2:
        D = st.number_input("Depth D (mm)", 200.0, 3000.0, 600.0, 25.0)
        fy = st.number_input("fy (MPa)", 415.0, 600.0, 500.0, 5.0)
        restraint = st.selectbox("End Restraint (k-factor)", ["Fixed-Fixed", "Fixed-Pinned", "Pinned-Pinned", "Fixed-Free (cantilever)"])
    with c3:
        cover = st.number_input("Clear cover (mm)", 20.0, 75.0, 40.0, 5.0)
        storey_clear = st.number_input("Clear storey height l0 (mm)", 2000.0, 6000.0, 3200.0, 50.0)
        kx = effective_length_factor(restraint)
        ky = kx
    with c4:
        Pu = st.number_input("Factored Axial Pu (kN, +comp, −tension)", -3000.0, 6000.0, 1200.0, 10.0) * 1e3
        Mux = st.number_input("Factored Mux (kNm)", -2000.0, 2000.0, 120.0, 5.0) * 1e6
        Muy = st.number_input("Factored Muy (kNm)", -2000.0, 2000.0, 80.0, 5.0) * 1e6
        Vu = st.number_input("Factored Shear Vu (kN)", 0.0, 5000.0, 150.0, 5.0) * 1e3

    st.markdown("---")
    st.markdown("### Longitudinal Bar Layout")
    cL1, cL2, cL3, cL4 = st.columns(4)
    with cL1:
        n_top = st.number_input("Top row bars", 0, 10, 3, 1)
        dia_top = st.selectbox("Top bar dia (mm)", [12, 16, 20, 25, 28, 32], index=1)
    with cL2:
        n_bot = st.number_input("Bottom row bars", 0, 10, 3, 1)
        dia_bot = st.selectbox("Bottom bar dia (mm)", [12, 16, 20, 25, 28, 32], index=1)
    with cL3:
        n_left = st.number_input("Left column bars", 0, 10, 2, 1)
        dia_side = st.selectbox("Side bar dia (mm)", [12, 16, 20, 25, 28], index=0)
    with cL4:
        n_right = st.number_input("Right column bars", 0, 10, 2, 1)
        tie_dia = st.selectbox("Tie dia (mm)", [6, 8, 10], index=1)

    tie_spacing = st.number_input("Tie spacing s (mm)", 50.0, 300.0, 150.0, 5.0)

    # Build bars
    bars = generate_rectangular_bar_layout(b, D, cover, n_top, n_bot, n_left, n_right, float(dia_top), float(dia_bot), float(dia_side))
    section = Section(b=b, D=D, cover=cover, bars=bars)

    # Save to state
    state.update(dict(b=b, D=D, cover=cover, fck=fck, fy=fy, Pu=Pu, Mux=Mux, Muy=Muy, Vu=Vu,
                      kx=kx, ky=ky, storey_clear=storey_clear, sway=sway,
                      tie_dia=tie_dia, tie_spacing=tie_spacing, bars=bars))

    # Quick visualization
    st.markdown("#### Cross-section Preview (SVG)")
    svg = svg_cross_section(section, tie_dia=float(tie_dia), tie_spacing=float(tie_spacing))
    st.components.v1.html(svg, height=min(600, int(D + 100)))

# -----------------------------
# Tab 2: Slenderness
# -----------------------------
with T2:
    st.subheader("2) Slenderness & Effective Length")
    b = state["b"]; D = state["D"]; storey_clear = state["storey_clear"]
    kx = state["kx"]; ky = state["ky"]
    section = Section(b=b, D=D, cover=state["cover"], bars=state["bars"])

    le_x = kx * storey_clear
    le_y = ky * storey_clear

    lam_x = le_x / max(section.rx, 1e-6)
    lam_y = le_y / max(section.ry, 1e-6)

    st.write(f"Effective length le,x = **{le_x:.0f} mm**, le,y = **{le_y:.0f} mm**")
    st.write(f"Radius of gyration: r_x = **{section.rx:.1f} mm**, r_y = **{section.ry:.1f} mm**")
    st.write(f"Slenderness: λx = **{lam_x:.1f}**, λy = **{lam_y:.1f}** (Short if ≤ ~12)")

    short_x = lam_x <= 12.0
    short_y = lam_y <= 12.0
    st.info(f"Classification → About x: {'Short' if short_x else 'Slender'}, About y: {'Short' if short_y else 'Slender'}")

    state.update(dict(le_x=le_x, le_y=le_y, lam_x=lam_x, lam_y=lam_y, short_x=short_x, short_y=short_y))

# -----------------------------
# Tab 3: Moments (2nd-order magnification)
# -----------------------------
with T3:
    st.subheader("3) First-Order → Magnified Moments")
    Pu = state["Pu"]; Mux = state["Mux"]; Muy = state["Muy"]
    fck = state["fck"]; fy = state["fy"]; sway = state["sway"]

    Ec = Ec_from_fck(fck)
    EI_eff_x = 0.4 * Ec * section.Ic_x
    EI_eff_y = 0.4 * Ec * section.Ic_y

    delta_x = moment_magnifier(Pu, state["le_x"], Ec, section.Ic_x, Cm=0.85, sway=sway) if not state["short_x"] else 1.0
    delta_y = moment_magnifier(Pu, state["le_y"], Ec, section.Ic_y, Cm=0.85, sway=sway) if not state["short_y"] else 1.0

    Mux_eff = Mux * delta_x
    Muy_eff = Muy * delta_y

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"EI_eff,x ≈ 0.4EcIc = **{EI_eff_x/1e6:.2f}×10^6 N·mm²**, δx = **{delta_x:.2f}** → Mux′ = **{Mux_eff/1e6:.1f} kNm**")
    with c2:
        st.write(f"EI_eff,y ≈ 0.4EcIc = **{EI_eff_y/1e6:.2f}×10^6 N·mm²**, δy = **{delta_y:.2f}** → Muy′ = **{Muy_eff/1e6:.1f} kNm**")

    state.update(dict(Mux_eff=Mux_eff, Muy_eff=Muy_eff, delta_x=delta_x, delta_y=delta_y))

# -----------------------------
# Tab 4: Interaction (Biaxial)
# -----------------------------
with T4:
    st.subheader("4) Biaxial Interaction Check")
    fck = state["fck"]; fy = state["fy"]; Pu = state["Pu"]
    alpha = st.slider("α (interaction exponent)", 0.8, 1.8, 1.0, 0.05)

    # Compute uniaxial Mu,lim for the given Pu
    with st.spinner("Computing uniaxial capacities via strain compatibility..."):
        Mux_lim = uniaxial_capacity_Mu_for_Pu(section, fck, fy, Pu, axis='x')
        Muy_lim = uniaxial_capacity_Mu_for_Pu(section, fck, fy, Pu, axis='y')

    Mux_eff = state["Mux_eff"]; Muy_eff = state["Muy_eff"]

    Rx = (Mux_eff / max(Mux_lim, 1e-6)) ** alpha
    Ry = (Muy_eff / max(Muy_lim, 1e-6)) ** alpha
    util = Rx + Ry

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Mux,lim (kNm)", f"{Mux_lim/1e6:.1f}")
    with c2:
        st.metric("Muy,lim (kNm)", f"{Muy_lim/1e6:.1f}")
    with c3:
        st.metric("Utilization (Σ ≤ 1)", f"{util:.2f}")

    if util <= 1.0:
        st.success("Biaxial interaction PASS.")
    else:
        st.error("Biaxial interaction FAIL — increase section/steel or revise layout.")

    state.update(dict(Mux_lim=Mux_lim, Muy_lim=Muy_lim, util=util, alpha=alpha))

# -----------------------------
# Tab 5: Shear
# -----------------------------
with T5:
    st.subheader("5) Shear Design (Ties)")
    Vu = state["Vu"]
    b = state["b"]; D = state["D"]; fck = state["fck"]; fy = state["fy"]

    d_eff = D - (state["cover"] + 0.5 * max([bar[2] for bar in state["bars"]], default=16.0))
    # Simplified concrete shear capacity with axial load influence factor φN (clamped 0.5..1.5)
    Ag = b * D
    phiN = 1.0 + (state["Pu"] / max(1.0, (0.25 * fck * Ag)))
    phiN = float(np.clip(phiN, 0.5, 1.5))
    tau_c = 0.62 * math.sqrt(fck) / 1.0  # MPa (teaching simplification)
    Vc = tau_c * b * d_eff * phiN  # N

    if Vu <= Vc:
        st.info("Vu ≤ Vc → Provide minimum transverse reinforcement per IS 456/13920.")
        Vus = 0.0
    else:
        Vus = Vu - Vc

    # Tie design: Vus = 0.87 fy Asv d / s → choose 2-legged 8/10 mm ties typical
    leg_options = st.selectbox("Tie leg configuration", ["2-legged", "4-legged"], index=0)
    legs = 2 if leg_options == "2-legged" else 4
    tie_dia = float(state["tie_dia"])  # mm
    Asv_one_leg = bar_area(tie_dia)
    Asv = legs * Asv_one_leg

    s_try = state["tie_spacing"]
    if Vus > 0:
        s_required = (0.87 * fy * Asv * d_eff) / max(Vus, 1e-6)
        s_prov = min(s_try, s_required)
    else:
        s_required = 300.0
        s_prov = s_try

    st.write(f"d (effective) ≈ **{d_eff:.0f} mm**, τ_c ≈ **{tau_c:.2f} MPa**, φN = **{phiN:.2f}** → Vc = **{Vc/1e3:.1f} kN**")
    st.write(f"Tie area Asv (provided) = **{Asv:.1f} mm²** ({legs} legs × {int(tie_dia)} mm)")
    if Vus > 0:
        st.warning(f"Vus = Vu − Vc = **{Vus/1e3:.1f} kN** → s_required = **{s_required:.0f} mm**; provide s ≤ min(code cap, this)")
    st.write(f"Provide ties @ **{s_prov:.0f} mm** c/c (or as per ductile limits if governing).")

    state.update(dict(Vc=Vc, Vus=Vus, s_required=s_required, s_prov=s_prov, d_eff=d_eff, legs=legs))

# -----------------------------
# Tab 6: Detailing
# -----------------------------
with T6:
    st.subheader("6) Detailing Checks")
    b = state["b"]; D = state["D"]; cover = state["cover"]
    fy = state["fy"]; fck = state["fck"]

    # Longitudinal steel totals
    As_long = sum([bar_area(d) for (_, _, d) in state["bars"]])
    rho_long = 100.0 * As_long / (b * D)

    st.write(f"Total As (longitudinal) = **{As_long:.0f} mm²** → ρ_l = **{rho_long:.2f}%** of gross")

    # Code-ish limits (indicative)
    As_min = 0.008 * b * D
    As_max = 0.06 * b * D
    st.write(f"Indicative limits: As_min ≈ **{As_min:.0f} mm²** (0.8%), As_max ≈ **{As_max:.0f} mm²** (6%)")

    # Tie spacing limits (ductile hint values; engineer to verify exact code clause)
    s_lim1 = 16.0 * max([bar[2] for bar in state["bars"]], default=16.0)
    s_lim2 = 0.75 * D
    s_lim3 = 300.0
    s_cap = min(s_lim1, s_lim2, s_lim3)

    st.write(f"Tie spacing cap (indicative) ≤ min(16·db, 0.75D, 300) = **{s_cap:.0f} mm**")

    if state.get("s_prov", 300.0) <= s_cap:
        st.success("Tie spacing within indicative cap.")
    else:
        st.error("Reduce tie spacing to meet cap (or apply ductile detailing rules if applicable).")

    # SVG again with labels
    st.markdown("#### Cross-section Detailing (SVG)")
    svg = svg_cross_section(Section(b, D, cover, state["bars"]), tie_dia=float(state["tie_dia"]), tie_spacing=float(state["tie_spacing"]))
    st.components.v1.html(svg, height=min(600, int(D + 100)))

# -----------------------------
# Tab 7: Output Summary
# -----------------------------
with T7:
    st.subheader("7) Output Summary")
    def kNm(val_Nmm):
        return val_Nmm / 1e6

    out = {
        "b (mm)": state["b"],
        "D (mm)": state["D"],
        "cover (mm)": state["cover"],
        "fck (MPa)": state["fck"],
        "fy (MPa)": state["fy"],
        "Pu (kN)": state["Pu"] / 1e3,
        "Mux (kNm)": state["Mux"] / 1e6,
        "Muy (kNm)": state["Muy"] / 1e6,
        "Vu (kN)": state["Vu"] / 1e3,
        "le,x (mm)": state.get("le_x", float('nan')),
        "le,y (mm)": state.get("le_y", float('nan')),
        "λx": state.get("lam_x", float('nan')),
        "λy": state.get("lam_y", float('nan')),
        "δx": state.get("delta_x", 1.0),
        "δy": state.get("delta_y", 1.0),
        "Mux′ (kNm)": kNm(state.get("Mux_eff", 0.0)),
        "Muy′ (kNm)": kNm(state.get("Muy_eff", 0.0)),
        "Mux,lim (kNm)": kNm(state.get("Mux_lim", 0.0)),
        "Muy,lim (kNm)": kNm(state.get("Muy_lim", 0.0)),
        "Utilization (≤1)": state.get("util", float('nan')),
        "Vc (kN)": state.get("Vc", 0.0) / 1e3,
        "Vus (kN)": state.get("Vus", 0.0) / 1e3,
        "Tie spacing provided (mm)": state.get("s_prov", float('nan')),
        "Tie legs": state.get("legs", 2),
        "As(long) (mm²)": sum([bar_area(d) for (_,_,d) in state["bars"]]),
    }
    st.dataframe({"Parameter": list(out.keys()), "Value": list(out.values())})

    st.caption(
        "Notes: Teaching-grade formulas for slenderness and shear; uniaxial capacities via simplified strain compatibility.\n"
        "Verify with SP-16/design charts and apply full IS 456/13920 clauses in production.")
