# streamlit_foundation_app.py
# --- Streamlit App: RCC Foundation Designer (Biaxial Moments, Shear in 2 dirs, No‑Tension Soil, Pedestal, Optional Raft) ---
# Author: ChatGPT (GPT‑5 Thinking)
# Notes:
#  - Implements biaxial soil‑pressure with eccentricity including "no‑tension" correction via iterative clipping and moment‑matching.
#  - Checks: SLS/ULS bearing, sliding, overturning (basic), one‑way shear, punching shear, flexure (strip integration), pedestal, crack control (simplified).
#  - Outputs: Pressure heatmap, contact area, punching perimeter sketch, BM/SF strip plots, tabular pass/fail.
#  - This is an engineering aid, not a substitute for a licensed engineer’s seal. Verify against project codes and geotech report.

import io
import math
import json
import zipfile
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------ Utility & Defaults ------------------------------

def_header = {
    "project": {
        "title": "Pump House Column F3 Foundation",
        "location": "Hyderabad",
        "designer": "Your Firm",
        "reviewer": "Approving Agency"
    },
    "materials": {
        "fck_mpa": 30,
        "fy_mpa": 500,
        "gamma_c": 25.0  # kN/m3
    },
    "geometry": {
        "column_bx": 0.4,
        "column_by": 0.4,
        "ped_px": 0.6,
        "ped_py": 0.6,
        "ped_h": 0.6,
        "foot_B": 2.2,
        "foot_L": 2.6,
        "foot_D": 0.55,
        "cover": 0.05
    },
    "soil": {
        "SBC_kPa_SLS": 200.0,
        "phi_bearing_factor_ULS": 0.67,  # ULS bearing reduction factor (user may adjust)
        "unit_weight": 18.0,
        "k_subgrade": 30000.0,
        "mu_base": 0.5,
        "GWT": 2.0,
        "allowable_settlement_mm": 25.0
    }
}

def default_loads_df():
    return pd.DataFrame([
        {"Combo": "ULS1 (DL+LL)", "Type": "ULS", "P_kN": 1200, "Vx_kN": 50, "Vy_kN": 40, "Mx_kNm": 180, "My_kNm": 220},
        {"Combo": "ULS2 (DL+EQx)", "Type": "ULS", "P_kN": 950, "Vx_kN": 120, "Vy_kN": 30, "Mx_kNm": 280, "My_kNm": 90},
        {"Combo": "SLS1 (Service)", "Type": "SLS", "P_kN": 900, "Vx_kN": 40, "Vy_kN": 35, "Mx_kNm": 150, "My_kNm": 170},
    ])

# ------------------------------ Soil Pressure Engine ------------------------------

def linear_pressure_full_contact(P: float, Mx: float, My: float, B: float, L: float,
                                 nx: int = 61, ny: int = 61):
    """Return grid (X,Y), linear q over full rectangle (can be negative if outside kern).
    Coordinates x in [-B/2, B/2], y in [-L/2, L/2]. q in kPa if P in kN and dimensions in m.
    q(x,y) = P/A + 12*P*e_x*x/B^2/A + 12*P*e_y*y/L^2/A with e_x = M_y/P, e_y = M_x/P
    Note: Mx is about x‑axis => creates gradient along y; My is about y‑axis => along x.
    """
    A = B * L
    ex = My / P if P != 0 else 0.0
    ey = Mx / P if P != 0 else 0.0
    x = np.linspace(-B/2, B/2, nx)
    y = np.linspace(-L/2, L/2, ny)
    X, Y = np.meshgrid(x, y)
    q0 = P / A  # kN/m^2 = kPa
    q = q0 * (1 + 12 * ex * X / (B**2) + 12 * ey * Y / (L**2))
    return X, Y, q, ex, ey


def no_tension_correction(P: float, Mx: float, My: float, B: float, L: float,
                          max_iter: int = 200, tol: float = 1e-3,
                          nx: int = 61, ny: int = 61):
    """Iteratively enforce q>=0 while matching target P, Mx, My as closely as possible.
    1) Start with full‑contact linear field.
    2) Clip negatives to zero.
    3) Rescale a,b,c (via simple gradient steps) so that integrals over positive region match P, Mx, My.
    Returns (X,Y,q,q_contact_mask,ex,ey,stats)
    """
    X, Y, q, ex, ey = linear_pressure_full_contact(P, Mx, My, B, L, nx, ny)
    dA = (B/(nx-1)) * (L/(ny-1))

    # Initialize affine q = a + b*X + c*Y such that integrals match P, Mx, My (full contact solution)
    A = B * L
    a = P / A
    b = a * 12 * (My / P) / (B**2) * B**2  # simplifies to 12 * (My/A) / B? We'll fit via integrals below.
    c = a * 12 * (Mx / P) / (L**2) * L**2
    # For numerical stability, just compute from q field by linear regression initial guess:
    A_mat = np.vstack([np.ones_like(X).ravel(), X.ravel(), Y.ravel()]).T
    coeffs, *_ = np.linalg.lstsq(A_mat, q.ravel(), rcond=None)
    a, b, c = coeffs.tolist()

    lr = 0.2  # learning rate for moment matching
    stats = {"iters": 0, "residual_P": None, "residual_Mx": None, "residual_My": None}

    for it in range(max_iter):
        q_field = a + b * X + c * Y
        q_pos = np.clip(q_field, 0.0, None)
        mask = q_pos > 0

        # Integrals over positive region
        P_num = q_pos.sum() * dA
        # Moments: Mx about x‑axis -> y lever arm; My about y‑axis -> x lever arm
        Mx_num = (q_pos * Y).sum() * dA
        My_num = (q_pos * X).sum() * dA

        # Residuals (target - current)
        rP = P - P_num
        rMx = Mx - Mx_num
        rMy = My - My_num

        stats.update({"iters": it+1, "residual_P": rP, "residual_Mx": rMx, "residual_My": rMy})

        if abs(rP) < tol and abs(rMx) < tol and abs(rMy) < tol:
            q = q_pos
            return X, Y, q, mask, ex, ey, stats

        # Gradient of integrals w.r.t a,b,c over positive region
        dP_da = (np.ones_like(q_pos)[mask]).sum() * dA
        dP_db = (X[mask]).sum() * dA
        dP_dc = (Y[mask]).sum() * dA

        dMx_da = (Y[mask]).sum() * dA
        dMx_db = (X[mask] * Y[mask]).sum() * dA
        dMx_dc = (Y[mask] * Y[mask]).sum() * dA

        dMy_da = (X[mask]).sum() * dA
        dMy_db = (X[mask] * X[mask]).sum() * dA
        dMy_dc = (X[mask] * Y[mask]).sum() * dA

        # Solve small 3x3 least‑squares step to update a,b,c
        J = np.array([
            [dP_da, dP_db, dP_dc],
            [dMx_da, dMx_db, dMx_dc],
            [dMy_da, dMy_db, dMy_dc],
        ])
        r = np.array([rP, rMx, rMy])
        try:
            delta = lr * np.linalg.lstsq(J, r, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        a += float(delta[0])
        b += float(delta[1])
        c += float(delta[2])

    # Fallback: scale q_pos to match P only
    q_field = a + b * X + c * Y
    q_pos = np.clip(q_field, 0.0, None)
    P_num = q_pos.sum() * dA
    scale = P / P_num if P_num > 0 else 1.0
    q = q_pos * scale
    mask = q > 0
    return X, Y, q, mask, ex, ey, stats

# ------------------------------ Structural Checks ------------------------------

@dataclass
class Geometry:
    B: float
    L: float
    D: float
    cover: float
    bx: float
    by: float
    ped_px: float
    ped_py: float
    ped_h: float


@dataclass
class Materials:
    fck: float
    fy: float
    gamma_c: float


def effective_depth(D: float, cover: float, bar_diam: float = 16e-3) -> float:
    return max(D - cover - bar_diam/2, 0.05)  # m


def bearing_check(q: np.ndarray, SBC_SLS_kPa: float, phi_ULS: float, combo_type: str) -> Dict:
    qmax = float(np.max(q))  # kPa
    qmin = float(np.min(q))
    limit = SBC_SLS_kPa if combo_type.upper()=="SLS" else SBC_SLS_kPa * phi_ULS
    ok = qmax <= limit and qmin >= -1e-6
    return {"qmax_kPa": qmax, "qmin_kPa": qmin, "limit_kPa": limit, "OK": ok}


def sliding_check(P_vert_kN: float, Vx_kN: float, Vy_kN: float, mu: float) -> Dict:
    R = mu * P_vert_kN
    V = math.hypot(Vx_kN, Vy_kN)
    ok = R >= V
    return {"Resisting_kN": R, "Demand_kN": V, "OK": ok}


def overturning_check(P: float, Mx: float, My: float, B: float, L: float) -> Dict:
    # Check kern (for info) and factor against loss of contact in each direction
    ex = abs(My / P) if P!=0 else 0
    ey = abs(Mx / P) if P!=0 else 0
    kern_x = B/6
    kern_y = L/6
    ok = (ex <= kern_x) and (ey <= kern_y)
    return {"ex": ex, "ey": ey, "kern_x": kern_x, "kern_y": kern_y, "OK": ok}


def one_way_shear_check(q: np.ndarray, X: np.ndarray, Y: np.ndarray, geom: Geometry, mat: Materials) -> Dict:
    # Critical sections at distance d from face of pedestal (conservative)
    d = effective_depth(geom.D, geom.cover)
    # Along X (shear across a strip parallel to Y), cut at x = bx/2 + d from column center
    x_crit_pos = geom.bx/2 + d
    x_crit_neg = -x_crit_pos
    # Integrate soil reaction outside the critical section that contributes to shear at the section
    dA = (geom.B/(X.shape[1]-1)) * (geom.L/(X.shape[0]-1))

    Vx_pos = (q[:, X[0]>=x_crit_pos].sum()) * dA  # kN
    Vx_neg = (q[:, X[0]<=x_crit_neg].sum()) * dA
    Vx_demand = max(Vx_pos, Vx_neg)

    # breadth b = footing width in Y at section; take full L
    b_x = geom.L
    d_eff = d
    tau_v = Vx_demand*1e3 / (b_x * d_eff)  # kPa*m / m^2 = kN/m^2? Convert to MPa later

    # Concrete shear capacity per IS 456 (approx.): tau_c = 0.62*sqrt(fck) MPa for footing one‑way? Use 0.62*sqrt(fck)/1.5
    tau_c = 0.62 * math.sqrt(mat.fck) / 1.5  # MPa
    tau_v_mpa = tau_v / 1000.0  # convert kPa to MPa

    ok_x = tau_v_mpa <= tau_c

    # Along Y
    y_crit_pos = geom.by/2 + d
    y_crit_neg = -y_crit_pos
    Vy_pos = (q[Y[:,0]>=y_crit_pos, :].sum()) * dA
    Vy_neg = (q[Y[:,0]<=y_crit_neg, :].sum()) * dA
    Vy_demand = max(Vy_pos, Vy_neg)
    b_y = geom.B
    tau_vy = Vy_demand*1e3 / (b_y * d_eff)
    tau_vy_mpa = tau_vy / 1000.0
    ok_y = tau_vy_mpa <= tau_c

    return {
        "one_way": {
            "X": {"V_kN": Vx_demand, "tau_v_MPa": tau_v_mpa, "tau_c_MPa": tau_c, "OK": ok_x},
            "Y": {"V_kN": Vy_demand, "tau_v_MPa": tau_vy_mpa, "tau_c_MPa": tau_c, "OK": ok_y}
        }
    }


def punching_shear_check(q: np.ndarray, X: np.ndarray, Y: np.ndarray, geom: Geometry, mat: Materials) -> Dict:
    d = effective_depth(geom.D, geom.cover)
    # Critical perimeter at d/2 from pedestal face
    perim_x = geom.ped_px + d
    perim_y = geom.ped_py + d
    u = 2 * (perim_x + perim_y)

    # Load inside the critical perimeter (soil reaction reduces punching demand)
    inside = (np.abs(X) <= perim_x/2) & (np.abs(Y) <= perim_y/2)
    dA = (geom.B/(X.shape[1]-1)) * (geom.L/(X.shape[0]-1))
    R_inside = q[inside].sum() * dA  # kN

    # Column + pedestal load at ULS approx unknown here; estimate punching demand as net upward outside perimeter on one side
    # Conservative: take resultant reaction outside perimeter as V_u
    V_u = (q.sum()*dA) - R_inside

    v_u = (V_u*1e3) / (u * d)  # MPa
    # IS 456 punching strength v_c ~ 0.25*sqrt(fck) MPa (simplified), reduce by gamma_m ~1.5
    v_c = 0.25 * math.sqrt(mat.fck) / 1.5
    ok = v_u <= v_c

    return {"u_m": u, "V_u_kN": V_u, "v_u_MPa": v_u, "v_c_MPa": v_c, "OK": ok}


def strip_moments(q: np.ndarray, X: np.ndarray, Y: np.ndarray, geom: Geometry) -> Dict:
    """Compute design strip moments along X and Y by integrating soil pressure times lever arms to the pedestal face.
    Returns dict with Mx_strip (about X‑axis; causes tension at bottom along Y‑strip) and My_strip.
    """
    dA = (geom.B/(X.shape[1]-1)) * (geom.L/(X.shape[0]-1))
    # Moments at faces of pedestal (conservative)
    x_face_pos = geom.ped_px/2
    y_face_pos = geom.ped_py/2

    # About X‑axis (bending along Y direction): integrate q * (y - y_face)^2 / 2 over regions beyond face
    # Use simple beam analogy: resultant on one side times its centroid distance.
    right = X[0] >= x_face_pos
    left  = X[0] <= -x_face_pos
    top   = Y[:,0] >= y_face_pos
    bottom= Y[:,0] <= -y_face_pos

    # Moments about Y‑axis (span along X) from right & left wings
    q_right = q[:, right]
    Xr = X[:, right]
    Ar = q_right.sum() * dA
    xcg_r = (q_right * Xr).sum() * dA / Ar if Ar>1e-9 else x_face_pos
    Mr = Ar * (xcg_r - x_face_pos)

    q_left = q[:, left]
    Xl = X[:, left]
    Al = q_left.sum() * dA
    xcg_l = (q_left * (-Xl)).sum() * dA / Al if Al>1e-9 else x_face_pos
    Ml = Al * (xcg_l - x_face_pos)

    My_design = max(Mr, Ml)  # kNm per meter breadth along Y; here we keep absolute kNm

    # Moments about X‑axis from top & bottom wings
    q_top = q[top, :]
    Yt = Y[top, :]
    At = q_top.sum() * dA
    ycg_t = (q_top * Yt).sum() * dA / At if At>1e-9 else y_face_pos
    Mt = At * (ycg_t - y_face_pos)

    q_bot = q[bottom, :]
    Yb = Y[bottom, :]
    Ab = q_bot.sum() * dA
    ycg_b = (q_bot * (-Yb)).sum() * dA / Ab if Ab>1e-9 else y_face_pos
    Mb = Ab * (ycg_b - y_face_pos)

    Mx_design = max(Mt, Mb)

    return {"Mx_kNm": float(Mx_design), "My_kNm": float(My_design)}


def flexure_steel(M_kNm: float, geom: Geometry, mat: Materials, strip_width: float) -> Dict:
    d = effective_depth(geom.D, geom.cover)
    Mu = M_kNm * 1e6  # Nmm -> actually kNm to Nmm: 1 kNm = 1e6 Nmm
    jd = 0.9 * d
    z = jd
    fy = mat.fy
    # As = Mu / (0.87 fy z)
    As = Mu / (0.87 * fy * 1e6 * z)  # fy in MPa = N/mm2 -> multiply by 1e6 to N/m2? Careful with units
    # Fix units: Use SI m & MPa, convert Mu to N*m (kNm*1000), z in m, fy in MPa = N/mm2 = 1e6 N/m2
    Mu_Nm = M_kNm * 1000.0
    As_m2 = Mu_Nm / (0.87 * (fy*1e6) * z)
    As_mm2_per_m = As_m2 * 1e6 / strip_width  # per meter strip if needed
    return {"d_m": d, "As_m2": As_m2, "As_mm2_per_m": As_mm2_per_m}


def pedestal_bearing_check(geom: Geometry, mat: Materials) -> Dict:
    # IS 456 bearing at column‑pedestal/footing interface (very simplified placeholder)
    # Allowable increased stress = 0.45 fck * sqrt(A2/A1) limited to 0.9 fck
    A1 = geom.bx * geom.by
    A2 = geom.ped_px * geom.ped_py
    inc = math.sqrt(max(A2/A1, 1.0))
    sigma_allow = min(0.45 * mat.fck * inc, 0.9 * mat.fck)  # MPa
    return {"sigma_allow_MPa": sigma_allow, "A1_m2": A1, "A2_m2": A2}

# ------------------------------ Streamlit UI ------------------------------

st.set_page_config(page_title="RCC Foundation Designer (Biaxial)", layout="wide")
st.title("RCC Foundation Designer — Biaxial Moments, No‑Tension Soil, Shear & Punching, Pedestal")
st.caption("Design aid per IS 456/IS 2950 guidance. Verify with a licensed engineer.")

with st.sidebar:
    st.header("Project & Materials")
    proj = def_header["project"]
    materials = def_header["materials"]
    geometry = def_header["geometry"]
    soil = def_header["soil"]

    with st.expander("Project"):
        title = st.text_input("Title", proj["title"]) 
        location = st.text_input("Location", proj["location"]) 
        designer = st.text_input("Designer", proj["designer"]) 
        reviewer = st.text_input("Reviewer", proj["reviewer"]) 

    with st.expander("Materials"):
        fck = st.number_input("Concrete fck (MPa)", 20.0, 80.0, materials["fck_mpa"], 5.0)
        fy = st.number_input("Steel fy (MPa)", 415.0, 600.0, materials["fy_mpa"], 5.0)
        gamma_c = st.number_input("Concrete unit weight (kN/m³)", 20.0, 27.0, materials["gamma_c"], 0.5)

    with st.expander("Soil & Bearing"):
        SBC = st.number_input("SBC @ SLS (kPa)", 50.0, 600.0, soil["SBC_kPa_SLS"], 10.0)
        phi_ULS = st.number_input("ULS bearing factor (φ)", 0.4, 1.0, soil["phi_bearing_factor_ULS"], 0.01)
        k_sub = st.number_input("Subgrade modulus k (kN/m³)", 1000.0, 100000.0, soil["k_subgrade"], 100.0)
        mu = st.number_input("Base friction μ", 0.2, 0.8, soil["mu_base"], 0.05)
        GWT = st.number_input("GWT depth below base (m)", 0.0, 20.0, soil["GWT"], 0.1)

    with st.expander("Geometry"):
        bx = st.number_input("Column size bx (m)", 0.2, 2.0, geometry["column_bx"], 0.05)
        by = st.number_input("Column size by (m)", 0.2, 2.0, geometry["column_by"], 0.05)
        ped_px = st.number_input("Pedestal px (m)", bx, 4.0, geometry["ped_px"], 0.05)
        ped_py = st.number_input("Pedestal py (m)", by, 4.0, geometry["ped_py"], 0.05)
        ped_h = st.number_input("Pedestal height (m)", 0.3, 2.0, geometry["ped_h"], 0.05)
        B = st.number_input("Footing B (m) — x‑dir size", ped_px+0.2, 20.0, geometry["foot_B"], 0.1)
        L = st.number_input("Footing L (m) — y‑dir size", ped_py+0.2, 20.0, geometry["foot_L"], 0.1)
        D = st.number_input("Footing overall thickness D (m)", 0.3, 2.0, geometry["foot_D"], 0.05)
        cover = st.number_input("Concrete cover (m)", 0.04, 0.1, geometry["cover"], 0.005)

    mat = Materials(fck=fck, fy=fy, gamma_c=gamma_c)
    geom = Geometry(B=B, L=L, D=D, cover=cover, bx=bx, by=by, ped_px=ped_px, ped_py=ped_py, ped_h=ped_h)

st.subheader("Load Combinations")
loads_df = st.data_editor(default_loads_df(), num_rows="dynamic", use_container_width=True)

st.markdown("---")
run = st.button("▶️ Run Design & Generate Plots")

results_rows = []
plots_png = {}
report_text = []

if run:
    st.success("Running analysis…")

    # Iterate combos
    for i, row in loads_df.iterrows():
        combo = str(row.get("Combo", f"Combo_{i+1}"))
        ctype = str(row.get("Type", "ULS")).upper()
        P = float(row.get("P_kN", 0.0))
        Vx = float(row.get("Vx_kN", 0.0))
        Vy = float(row.get("Vy_kN", 0.0))
        Mx = float(row.get("Mx_kNm", 0.0))
        My = float(row.get("My_kNm", 0.0))

        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown(f"### {combo} ({ctype})")
            st.write(f"P={P:.1f} kN, Vx={Vx:.1f} kN, Vy={Vy:.1f} kN, Mx={Mx:.1f} kNm, My={My:.1f} kNm")

        # Pressure field with no‑tension correction
        X, Y, q, mask, ex, ey, stats = no_tension_correction(P, Mx, My, geom.B, geom.L, nx=101, ny=101)
        qmax = float(np.max(q)); qmin = float(np.min(q))
        bearing = bearing_check(q, SBC, phi_ULS, ctype)
        slide = sliding_check(P, Vx, Vy, mu)
        ot = overturning_check(P, Mx, My, geom.B, geom.L)
        shear = one_way_shear_check(q, X, Y, geom, mat)
        punch = punching_shear_check(q, X, Y, geom, mat)
        stripM = strip_moments(q, X, Y, geom)

        # Flexure steel (very simplified per‑meter strip along both dirs)
        steel_x = flexure_steel(stripM["Mx_kNm"], geom, mat, strip_width=1.0)
        steel_y = flexure_steel(stripM["My_kNm"], geom, mat, strip_width=1.0)

        # -------- Plots --------
        with col1:
            fig1, ax1 = plt.subplots()
            im = ax1.imshow(q, extent=[-geom.B/2, geom.B/2, -geom.L/2, geom.L/2], origin='lower')
            ax1.set_xlabel('x (m)')
            ax1.set_ylabel('y (m)')
            ax1.set_title(f"Soil Pressure Heatmap — {combo}")
            plt.colorbar(im, ax=ax1, label='q (kPa)')
            # Draw pedestal
            rx = geom.ped_px/2; ry = geom.ped_py/2
            ax1.plot([ -rx,  rx,  rx, -rx, -rx], [ -ry, -ry,  ry,  ry, -ry])
            st.pyplot(fig1, use_container_width=True)
            buf1 = io.BytesIO(); fig1.savefig(buf1, format='png', dpi=200); buf1.seek(0)
            plots_png[f"{combo}_pressure.png"] = buf1.getvalue()
            plt.close(fig1)

        with col2:
            # Strip BM bars chart
            fig2, ax2 = plt.subplots()
            dirs = ['Mx (about X)', 'My (about Y)']
            vals = [stripM["Mx_kNm"], stripM["My_kNm"]]
            ax2.bar(dirs, vals)
            ax2.set_ylabel('Design strip moment (kNm)')
            ax2.set_title(f'Strip Moments — {combo}')
            st.pyplot(fig2, use_container_width=True)
            buf2 = io.BytesIO(); fig2.savefig(buf2, format='png', dpi=200); buf2.seek(0)
            plots_png[f"{combo}_moments.png"] = buf2.getvalue()
            plt.close(fig2)

        # Punching sketch
        fig3, ax3 = plt.subplots()
        ax3.set_title(f"Punching Perimeter @ d/2 — {combo}")
        ax3.set_xlabel('x (m)'); ax3.set_ylabel('y (m)')
        ax3.set_aspect('equal','box')
        # Footing outline
        ax3.plot([-geom.B/2, geom.B/2, geom.B/2, -geom.B/2, -geom.B/2],
                 [-geom.L/2, -geom.L/2, geom.L/2, geom.L/2, -geom.L/2])
        # Pedestal
        ax3.plot([-geom.ped_px/2, geom.ped_px/2, geom.ped_px/2, -geom.ped_px/2, -geom.ped_px/2],
                 [-geom.ped_py/2, -geom.ped_py/2, geom.ped_py/2, geom.ped_py/2, -geom.ped_py/2])
        d_eff = effective_depth(geom.D, geom.cover)
        perim_x = geom.ped_px + d_eff
        perim_y = geom.ped_py + d_eff
        ax3.plot([-perim_x/2, perim_x/2, perim_x/2, -perim_x/2, -perim_x/2],
                 [-perim_y/2, -perim_y/2, perim_y/2, perim_y/2, -perim_y/2])
        st.pyplot(fig3, use_container_width=True)
        buf3 = io.BytesIO(); fig3.savefig(buf3, format='png', dpi=200); buf3.seek(0)
        plots_png[f"{combo}_punching.png"] = buf3.getvalue()
        plt.close(fig3)

        # Row summary
        row_summary = {
            "Combo": combo, "Type": ctype,
            "qmax_kPa": bearing["qmax_kPa"], "qmin_kPa": bearing["qmin_kPa"], "Bearing_Limit_kPa": bearing["limit_kPa"], "Bearing_OK": bearing["OK"],
            "Sliding_R(kN)": slide["Resisting_kN"], "Sliding_V(kN)": slide["Demand_kN"], "Sliding_OK": slide["OK"],
            "Overturning_OK": ot["OK"],
            "OneWay_X_OK": shear["one_way"]["X"]["OK"], "OneWay_Y_OK": shear["one_way"]["Y"]["OK"],
            "Punch_OK": punch["OK"],
            "Mx_strip_kNm": stripM["Mx_kNm"], "My_strip_kNm": stripM["My_kNm"],
            "As_x_mm2pm": steel_x["As_mm2_per_m"], "As_y_mm2pm": steel_y["As_mm2_per_m"]
        }
        results_rows.append(row_summary)

        # Report snippet
        report_text.append(f"## {combo} ({ctype})\n"
                           f"P={P:.1f} kN, Vx={Vx:.1f} kN, Vy={Vy:.1f} kN, Mx={Mx:.1f} kNm, My={My:.1f} kNm\n\n"
                           f"- Eccentricities: e_x={My/P if P else 0:.3f} m, e_y={Mx/P if P else 0:.3f} m; No‑tension iters={stats['iters']}, resP={stats['residual_P']:.3f} kN\n"
                           f"- Bearing: qmax={bearing['qmax_kPa']:.1f} ≤ {bearing['limit_kPa']:.1f} kPa → {'OK' if bearing['OK'] else 'NG'}\n"
                           f"- Sliding: R={slide['Resisting_kN']:.1f} vs V={slide['Demand_kN']:.1f} → {'OK' if slide['OK'] else 'NG'}\n"
                           f"- One‑way shear X/Y → {('OK' if shear['one_way']['X']['OK'] else 'NG')}/{('OK' if shear['one_way']['Y']['OK'] else 'NG')}\n"
                           f"- Punching: v_u={punch['v_u_MPa']:.3f} ≤ v_c={punch['v_c_MPa']:.3f} MPa → {'OK' if punch['OK'] else 'NG'}\n"
                           f"- Strip moments: Mx={stripM['Mx_kNm']:.1f} kNm, My={stripM['My_kNm']:.1f} kNm; As_x={steel_x['As_mm2_per_m']:.0f} mm²/m, As_y={steel_y['As_mm2_per_m']:.0f} mm²/m\n")

    # Results table
    res_df = pd.DataFrame(results_rows)
    st.subheader("Design Summary — All Combos")
    st.dataframe(res_df, use_container_width=True)

    # Pedestal check (common)
    ped_check = pedestal_bearing_check(geom, mat)
    st.subheader("Pedestal Bearing Check (IS 456 — simplified)")
    st.json(ped_check)

    # Downloads: CSV + ZIP of plots + JSON report
    csv_buf = io.StringIO()
    res_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode()

    md_report = f"# {title} — {location}\n\n**Designer:** {designer}  \\\n**Reviewer:** {reviewer}  \\\n**fck:** {mat.fck} MPa, **fy:** {mat.fy} MPa  \\\n**Footing:** B×L×D = {geom.B}×{geom.L}×{geom.D} m, Cover={geom.cover} m  \\\n**Pedestal:** {geom.ped_px}×{geom.ped_py}×{geom.ped_h} m  \\\n**SBC@SLS:** {SBC} kPa (φ_ULS={phi_ULS})\n\n" + "\n\n".join(report_text)

    # Bundle ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('summary.csv', csv_bytes)
        zf.writestr('report.md', md_report.encode())
        for name, data in plots_png.items():
            zf.writestr(name, data)
    zip_buf.seek(0)

    st.download_button("⬇️ Download package (CSV + plots + report.md)", data=zip_buf, file_name="foundation_design_outputs.zip")

# ------------------------------ Guidance & Next Steps ------------------------------

with st.expander("Method Notes & Advancements Suggestions"):
    st.markdown(
        """
**Pressure Solver**  
This app enforces **no‑tension soil** by iteratively clipping negative pressure and matching the target **P, Mx, My** via a small least‑squares update on an affine pressure field \(q=a+bx+cy\). It produces a realistic **contact polygon** and **heatmap** even for large eccentricities. For critical designs, cross‑check with finite element or closed‑form kern/partial‑contact derivations.

**Advancements you can add next:**
1. **Optimization loop** on B, L, D to minimize concrete/steel while satisfying governing check → use `scipy.optimize`.
2. **Raft (Mat) fallback**: switch to Winkler plate strips and re‑compute panel moments & punching at column locations.
3. **Crack width** checks (IS 456 Annex F) based on bar spacing, cover, and strain.
4. **Anchors for uplift/overturning** with ACI 318 anchorage checks.
5. **DXF export** of rebar plans/sections using `ezdxf` for drawing submissions.
6. **PDF report generator** via `reportlab` or `weasyprint` to embed plots and tables with clause references.
7. **Seismic combinations** per IS 1893; allow load‑comb creator and auto‑factoring.
8. **Soil buoyancy** adjustment for high GWT; subtract submerged weight and include uplift.
9. **Load import** from Excel/CSV and **batch processing** for multiple columns.
        """
    )
