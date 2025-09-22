# -*- coding: utf-8 -*-
# streamlit_foundation_app_ascii.py
# Streamlit App: RCC Foundation Designer (Biaxial Moments, Shear in 2 directions, No-Tension Soil, Pedestal)
# Author: ChatGPT (GPT-5 Thinking)
# NOTE: ASCII-safe version for Python 3.13 / Streamlit AST parser
#  - Removed non-ASCII characters (em dashes, symbols)
#  - Replaced Greek letters and superscripts with ASCII (phi, m^3, etc.)
#  - Avoided fancy quotes

import io
import math
import zipfile
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------ Defaults ------------------------------

def_header = {
    "project": {
        "title": "Pump House Column F3 Foundation",
        "location": "Hyderabad",
        "designer": "Your Firm",
        "reviewer": "Approving Agency"
    },
    "materials": {
        "fck_mpa": 30.0,
        "fy_mpa": 500.0,
        "gamma_c": 25.0  # kN/m^3
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
        "phi_bearing_factor_ULS": 0.67,
        "unit_weight": 18.0,
        "k_subgrade": 30000.0,
        "mu_base": 0.5,
        "GWT": 2.0,
        "allowable_settlement_mm": 25.0
    }
}

def default_loads_df():
    return pd.DataFrame([
        {"Combo": "ULS1 (DL+LL)", "Type": "ULS", "P_kN": 1200.0, "Vx_kN": 50.0, "Vy_kN": 40.0, "Mx_kNm": 180.0, "My_kNm": 220.0},
        {"Combo": "ULS2 (DL+EQx)", "Type": "ULS", "P_kN": 950.0, "Vx_kN": 120.0, "Vy_kN": 30.0, "Mx_kNm": 280.0, "My_kNm": 90.0},
        {"Combo": "SLS1 (Service)", "Type": "SLS", "P_kN": 900.0, "Vx_kN": 40.0, "Vy_kN": 35.0, "Mx_kNm": 150.0, "My_kNm": 170.0},
    ])

# ------------------------------ Streamlit helper ------------------------------

def num_input_float(label, minv, maxv, val, step, **kwargs):
    return st.number_input(
        label,
        min_value=float(minv),
        max_value=float(maxv),
        value=float(val),
        step=float(step),
        **kwargs,
    )

# ------------------------------ Pressure Solver ------------------------------

def linear_pressure_full_contact(P, Mx, My, B, L, nx=61, ny=61):
    A = B * L
    ex = My / P if P != 0 else 0.0
    ey = Mx / P if P != 0 else 0.0
    x = np.linspace(-B/2.0, B/2.0, nx)
    y = np.linspace(-L/2.0, L/2.0, ny)
    X, Y = np.meshgrid(x, y)
    q0 = P / A
    q = q0 * (1 + 12.0 * ex * X / (B**2) + 12.0 * ey * Y / (L**2))
    return X, Y, q, ex, ey


def no_tension_correction(P, Mx, My, B, L, max_iter=200, tol=1e-3, nx=61, ny=61):
    X, Y, q_init, ex, ey = linear_pressure_full_contact(P, Mx, My, B, L, nx, ny)
    dA = (B/(nx-1)) * (L/(ny-1))

    A_mat = np.vstack([np.ones_like(X).ravel(), X.ravel(), Y.ravel()]).T
    coeffs, *_ = np.linalg.lstsq(A_mat, q_init.ravel(), rcond=None)
    a, b, c = coeffs.tolist()

    lr = 0.2
    stats = {"iters": 0, "residual_P": None, "residual_Mx": None, "residual_My": None}

    for it in range(max_iter):
        q_field = a + b * X + c * Y
        q_pos = np.clip(q_field, 0.0, None)
        mask = q_pos > 0

        P_num = q_pos.sum() * dA
        Mx_num = (q_pos * Y).sum() * dA
        My_num = (q_pos * X).sum() * dA

        rP = P - P_num
        rMx = Mx - Mx_num
        rMy = My - My_num

        stats.update({"iters": it+1, "residual_P": float(rP), "residual_Mx": float(rMx), "residual_My": float(rMy)})

        if abs(rP) < tol and abs(rMx) < tol and abs(rMy) < tol:
            return X, Y, q_pos, mask, ex, ey, stats

        Xm = X[mask]; Ym = Y[mask]
        ones = np.ones_like(Xm)
        dP = np.array([ones.sum(), Xm.sum(), Ym.sum()]) * dA
        dMx = np.array([Ym.sum(), (Xm*Ym).sum(), (Ym*Ym).sum()]) * dA
        dMy = np.array([Xm.sum(), (Xm*Xm).sum(), (Xm*Ym).sum()]) * dA
        J = np.vstack([dP, dMx, dMy])
        r = np.array([rP, rMx, rMy])
        try:
            delta = lr * np.linalg.lstsq(J, r, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        a += float(delta[0]); b += float(delta[1]); c += float(delta[2])

    q_field = a + b * X + c * Y
    q_pos = np.clip(q_field, 0.0, None)
    P_num = q_pos.sum() * dA
    scale = P / P_num if P_num > 0 else 1.0
    return X, Y, q_pos * scale, (q_pos>0), ex, ey, stats

# ------------------------------ Checks ------------------------------

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

def effective_depth(D, cover, bar_diam=16e-3):
    return max(D - cover - bar_diam/2.0, 0.05)

def bearing_check(q, SBC_SLS_kPa, phi_ULS, combo_type):
    qmax = float(np.max(q))
    qmin = float(np.min(q))
    limit = SBC_SLS_kPa if combo_type.upper()=="SLS" else SBC_SLS_kPa * phi_ULS
    ok = (qmax <= limit) and (qmin >= -1e-6)
    return {"qmax_kPa": qmax, "qmin_kPa": qmin, "limit_kPa": float(limit), "OK": ok}

def sliding_check(P_vert_kN, Vx_kN, Vy_kN, mu):
    R = mu * max(P_vert_kN, 0.0)
    V = math.hypot(Vx_kN, Vy_kN)
    ok = R >= V
    return {"Resisting_kN": float(R), "Demand_kN": float(V), "OK": ok}

def overturning_check(P, Mx, My, B, L):
    ex = abs(My / P) if P!=0 else 0.0
    ey = abs(Mx / P) if P!=0 else 0.0
    kern_x = B/6.0
    kern_y = L/6.0
    ok = (ex <= kern_x) and (ey <= kern_y)
    return {"ex": float(ex), "ey": float(ey), "kern_x": float(kern_x), "kern_y": float(kern_y), "OK": ok}

def one_way_shear_check(q, X, Y, geom: Geometry, mat: Materials):
    d = effective_depth(geom.D, geom.cover)
    dA = (geom.B/(X.shape[1]-1)) * (geom.L/(X.shape[0]-1))

    x_crit = geom.bx/2.0 + d
    Vx_pos = (q[:, X[0]>=x_crit].sum()) * dA
    Vx_neg = (q[:, X[0]<=-x_crit].sum()) * dA
    Vx_demand = max(Vx_pos, Vx_neg)
    b_x = geom.L
    tau_vx_MPa = (Vx_demand*1e3) / (b_x * d) / 1000.0

    y_crit = geom.by/2.0 + d
    Vy_pos = (q[Y[:,0]>=y_crit, :].sum()) * dA
    Vy_neg = (q[Y[:,0]<=-y_crit, :].sum()) * dA
    Vy_demand = max(Vy_pos, Vy_neg)
    b_y = geom.B
    tau_vy_MPa = (Vy_demand*1e3) / (b_y * d) / 1000.0

    tau_c = 0.62 * math.sqrt(mat.fck) / 1.5

    return {
        "one_way": {
            "X": {"V_kN": float(Vx_demand), "tau_v_MPa": float(tau_vx_MPa), "tau_c_MPa": float(tau_c), "OK": tau_vx_MPa <= tau_c},
            "Y": {"V_kN": float(Vy_demand), "tau_v_MPa": float(tau_vy_MPa), "tau_c_MPa": float(tau_c), "OK": tau_vy_MPa <= tau_c},
        }
    }

def punching_shear_check(q, X, Y, geom: Geometry, mat: Materials):
    d = effective_depth(geom.D, geom.cover)
    perim_x = geom.ped_px + d
    perim_y = geom.ped_py + d
    u = 2.0 * (perim_x + perim_y)

    dA = (geom.B/(X.shape[1]-1)) * (geom.L/(X.shape[0]-1))
    inside = (np.abs(X) <= perim_x/2.0) & (np.abs(Y) <= perim_y/2.0)
    R_inside = q[inside].sum() * dA

    V_u = (q.sum()*dA) - R_inside
    v_u_MPa = (V_u*1e3) / (u * d) / 1000.0
    v_c_MPa = 0.25 * math.sqrt(mat.fck) / 1.5
    ok = v_u_MPa <= v_c_MPa

    return {"u_m": float(u), "V_u_kN": float(V_u), "v_u_MPa": float(v_u_MPa), "v_c_MPa": float(v_c_MPa), "OK": ok}

def strip_moments(q, X, Y, geom: Geometry):
    dA = (geom.B/(X.shape[1]-1)) * (geom.L/(X.shape[0]-1))
    x_face = geom.ped_px/2.0
    y_face = geom.ped_py/2.0

    right = X[0] >= x_face
    left  = X[0] <= -x_face

    q_right = q[:, right]; Xr = X[:, right]
    Ar = q_right.sum() * dA
    xcg_r = (q_right * Xr).sum() * dA / Ar if Ar>1e-9 else x_face
    Mr = Ar * max(xcg_r - x_face, 0.0)

    q_left = q[:, left]; Xl = X[:, left]
    Al = q_left.sum() * dA
    xcg_l = (q_left * (-Xl)).sum() * dA / Al if Al>1e-9 else x_face
    Ml = Al * max(xcg_l - x_face, 0.0)

    My_design = max(Mr, Ml)

    top   = Y[:,0] >= y_face
    bottom= Y[:,0] <= -y_face

    q_top = q[top, :]; Yt = Y[top, :]
    At = q_top.sum() * dA
    ycg_t = (q_top * Yt).sum() * dA / At if At>1e-9 else y_face
    Mt = At * max(ycg_t - y_face, 0.0)

    q_bot = q[bottom, :]; Yb = Y[bottom, :]
    Ab = q_bot.sum() * dA
    ycg_b = (q_bot * (-Yb)).sum() * dA / Ab if Ab>1e-9 else y_face
    Mb = Ab * max(ycg_b - y_face, 0.0)

    Mx_design = max(Mt, Mb)

    return {"Mx_kNm": float(Mx_design), "My_kNm": float(My_design)}

def flexure_steel(M_kNm, geom: Geometry, mat: Materials, strip_width):
    d = effective_depth(geom.D, geom.cover)
    Mu_Nm = M_kNm * 1000.0
    fy = mat.fy * 1e6  # MPa -> N/m^2
    z = 0.9 * d
    As_m2 = Mu_Nm / (0.87 * fy * z) if z>0 else 0.0
    As_mm2_per_m = As_m2 * 1e6 / max(strip_width,1e-6)
    return {"d_m": float(d), "As_m2": float(As_m2), "As_mm2_per_m": float(As_mm2_per_m)}

def pedestal_bearing_check(geom: Geometry, mat: Materials):
    A1 = geom.bx * geom.by
    A2 = geom.ped_px * geom.ped_py
    inc = math.sqrt(max(A2/A1, 1.0)) if A1>0 else 1.0
    sigma_allow = min(0.45 * mat.fck * inc, 0.9 * mat.fck)
    return {"sigma_allow_MPa": float(sigma_allow), "A1_m2": float(A1), "A2_m2": float(A2)}

# ------------------------------ Streamlit UI ------------------------------

try:
    st.set_page_config(page_title="RCC Foundation Designer (Biaxial)", layout="wide")
except Exception:
    pass

st.title("RCC Foundation Designer - Biaxial Moments, No-Tension Soil, Shear and Punching, Pedestal")
st.caption("Design aid per IS 456 / IS 2950 guidance. Verify with a licensed engineer.")

with st.sidebar:
    st.header("Project & Materials")
    proj = def_header["project"]
    materials = def_header["materials"]
    geometry = def_header["geometry"]
    soil = def_header["soil"]

    with st.expander("Project"):
        title = st.text_input("Title", str(proj["title"]))
        location = st.text_input("Location", str(proj["location"]))
        designer = st.text_input("Designer", str(proj["designer"]))
        reviewer = st.text_input("Reviewer", str(proj["reviewer"]))

    with st.expander("Materials"):
        fck = num_input_float("Concrete fck (MPa)", 20.0, 80.0, materials["fck_mpa"], 5.0)
        fy = num_input_float("Steel fy (MPa)", 415.0, 600.0, materials["fy_mpa"], 5.0)
        gamma_c = num_input_float("Concrete unit weight (kN/m^3)", 20.0, 27.0, materials["gamma_c"], 0.5)

    with st.expander("Soil & Bearing"):
        SBC = num_input_float("SBC @ SLS (kPa)", 50.0, 600.0, soil["SBC_kPa_SLS"], 10.0)
        phi_ULS = num_input_float("ULS bearing factor (phi)", 0.4, 1.0, soil["phi_bearing_factor_ULS"], 0.01)
        k_sub = num_input_float("Subgrade modulus k (kN/m^3)", 1000.0, 100000.0, soil["k_subgrade"], 100.0)
        mu = num_input_float("Base friction mu", 0.2, 0.8, soil["mu_base"], 0.05)
        GWT = num_input_float("GWT depth below base (m)", 0.0, 20.0, soil["GWT"], 0.1)

    with st.expander("Geometry"):
        bx = num_input_float("Column size bx (m)", 0.2, 2.0, geometry["column_bx"], 0.05)
        by = num_input_float("Column size by (m)", 0.2, 2.0, geometry["column_by"], 0.05)
        ped_px = num_input_float("Pedestal px (m)", bx, 4.0, geometry["ped_px"], 0.05)
        ped_py = num_input_float("Pedestal py (m)", by, 4.0, geometry["ped_py"], 0.05)
        ped_h = num_input_float("Pedestal height (m)", 0.3, 2.0, geometry["ped_h"], 0.05)
        B = num_input_float("Footing B (m) - x dir size", float(ped_px+0.2), 20.0, geometry["foot_B"], 0.1)
        L = num_input_float("Footing L (m) - y dir size", float(ped_py+0.2), 20.0, geometry["foot_L"], 0.1)
        D = num_input_float("Footing overall thickness D (m)", 0.3, 2.0, geometry["foot_D"], 0.05)
        cover = num_input_float("Concrete cover (m)", 0.04, 0.1, geometry["cover"], 0.005)

    mat = Materials(fck=fck, fy=fy, gamma_c=gamma_c)
    geom = Geometry(B=B, L=L, D=D, cover=cover, bx=bx, by=by, ped_px=ped_px, ped_py=ped_py, ped_h=ped_h)

st.subheader("Load Combinations")
loads_df = st.data_editor(default_loads_df(), num_rows="dynamic", use_container_width=True)

st.markdown("---")
run = st.button("Run Design and Generate Plots")

results_rows: List[Dict] = []
plots_png: Dict[str, bytes] = {}
report_text: List[str] = []

if run:
    st.success("Running analysis...")

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

        X, Y, q, mask, ex, ey, stats = no_tension_correction(P, Mx, My, geom.B, geom.L, nx=101, ny=101)
        bearing = bearing_check(q, SBC, phi_ULS, ctype)
        slide = sliding_check(P, Vx, Vy, mu)
        ot = overturning_check(P, Mx, My, geom.B, geom.L)
        shear = one_way_shear_check(q, X, Y, geom, mat)
        punch = punching_shear_check(q, X, Y, geom, mat)
        stripM = strip_moments(q, X, Y, geom)

        steel_x = flexure_steel(stripM["Mx_kNm"], geom, mat, strip_width=1.0)
        steel_y = flexure_steel(stripM["My_kNm"], geom, mat, strip_width=1.0)

        with col1:
            fig1, ax1 = plt.subplots()
            im = ax1.imshow(q, extent=[-geom.B/2.0, geom.B/2.0, -geom.L/2.0, geom.L/2.0], origin='lower')
            ax1.set_xlabel('x (m)')
            ax1.set_ylabel('y (m)')
            ax1.set_title(f"Soil Pressure Heatmap - {combo}")
            plt.colorbar(im, ax=ax1, label='q (kPa)')
            rx = geom.ped_px/2.0; ry = geom.ped_py/2.0
            ax1.plot([-rx, rx, rx, -rx, -rx], [-ry, -ry, ry, ry, -ry])
            st.pyplot(fig1, use_container_width=True)
            buf1 = io.BytesIO(); fig1.savefig(buf1, format='png', dpi=200); buf1.seek(0)
            plots_png[f"{combo}_pressure.png"] = buf1.getvalue()
            plt.close(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            dirs = ['Mx (about X)', 'My (about Y)']
            vals = [stripM["Mx_kNm"], stripM["My_kNm"]]
            ax2.bar(dirs, vals)
            ax2.set_ylabel('Design strip moment (kNm)')
            ax2.set_title(f'Strip Moments - {combo}')
            st.pyplot(fig2, use_container_width=True)
            buf2 = io.BytesIO(); fig2.savefig(buf2, format='png', dpi=200); buf2.seek(0)
            plots_png[f"{combo}_moments.png"] = buf2.getvalue()
            plt.close(fig2)

        fig3, ax3 = plt.subplots()
        ax3.set_title(f"Punching Perimeter @ d/2 - {combo}")
        ax3.set_xlabel('x (m)'); ax3.set_ylabel('y (m)')
        ax3.set_aspect('equal','box')
        ax3.plot([-geom.B/2.0, geom.B/2.0, geom.B/2.0, -geom.B/2.0, -geom.B/2.0],
                 [-geom.L/2.0, -geom.L/2.0, geom.L/2.0, geom.L/2.0, -geom.L/2.0])
        ax3.plot([-geom.ped_px/2.0, geom.ped_px/2.0, geom.ped_px/2.0, -geom.ped_px/2.0, -geom.ped_px/2.0],
                 [-geom.ped_py/2.0, -geom.ped_py/2.0, geom.ped_py/2.0, geom.ped_py/2.0, -geom.ped_py/2.0])
        d_eff = effective_depth(geom.D, geom.cover)
        perim_x = geom.ped_px + d_eff
        perim_y = geom.ped_py + d_eff
        ax3.plot([-perim_x/2.0, perim_x/2.0, perim_x/2.0, -perim_x/2.0, -perim_x/2.0],
                 [-perim_y/2.0, -perim_y/2.0, perim_y/2.0, perim_y/2.0, -perim_y/2.0])
        st.pyplot(fig3, use_container_width=True)
        buf3 = io.BytesIO(); fig3.savefig(buf3, format='png', dpi=200); buf3.seek(0)
        plots_png[f"{combo}_punching.png"] = buf3.getvalue()
        plt.close(fig3)

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

        report_text.append(
            f"## {combo} ({ctype})
"
            f"P={P:.1f} kN, Vx={Vx:.1f} kN, Vy={Vy:.1f} kN, Mx={Mx:.1f} kNm, My={My:.1f} kNm

"
            f"- Eccentricities: e_x={My/P if P else 0:.3f} m, e_y={Mx/P if P else 0:.3f} m; No-tension iters={stats['iters']}, resP={stats['residual_P'] if stats['residual_P'] is not None else 0:.3f} kN
"
            f"- Bearing: qmax={bearing['qmax_kPa']:.1f} <= {bearing['limit_kPa']:.1f} kPa -> {'OK' if bearing['OK'] else 'NG'}
"
            f"- Sliding: R={slide['Resisting_kN']:.1f} vs V={slide['Demand_kN']:.1f} -> {'OK' if slide['OK'] else 'NG'}
"
            f"- One-way shear X/Y -> {('OK' if shear['one_way']['X']['OK'] else 'NG')}/{('OK' if shear['one_way']['Y']['OK'] else 'NG')}
"
            f"- Punching: v_u={punch['v_u_MPa']:.3f} <= v_c={punch['v_c_MPa']:.3f} MPa -> {'OK' if punch['OK'] else 'NG'}
"
            f"- Strip moments: Mx={stripM['Mx_kNm']:.1f} kNm, My={stripM['My_kNm']:.1f} kNm; As_x={steel_x['As_mm2_per_m']:.0f} mm^2/m, As_y={steel_y['As_mm2_per_m']:.0f} mm^2/m
"
        )

    res_df = pd.DataFrame(results_rows)
    st.subheader("Design Summary - All Combos")
    st.dataframe(res_df, use_container_width=True)

    ped_check = pedestal_bearing_check(geom, mat)
    st.subheader("Pedestal Bearing Check (IS 456 - simplified)")
    st.json(ped_check)

    csv_buf = io.StringIO(); res_df.to_csv(csv_buf, index=False); csv_bytes = csv_buf.getvalue().encode()

    md_report = (
        f"# {title} - {location}

"
        f"Designer: {designer}  
"
        f"Reviewer: {reviewer}  
"
        f"fck: {mat.fck} MPa, fy: {mat.fy} MPa  
"
        f"Footing: BxLxD = {geom.B}x{geom.L}x{geom.D} m, Cover={geom.cover} m  
"
        f"Pedestal: {geom.ped_px}x{geom.ped_py}x{geom.ped_h} m  
"
        f"SBC@SLS: {SBC} kPa (phi_ULS={phi_ULS})

"
        + "

".join(report_text)
    )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('summary.csv', csv_bytes)
        zf.writestr('report.md', md_report.encode('utf-8'))
        for name, data in plots_png.items():
            zf.writestr(name, data)
    zip_buf.seek(0)

    st.download_button("Download package (CSV + plots + report.md)", data=zip_buf, file_name="foundation_design_outputs.zip")

with st.expander("Method Notes and Next Steps"):
    st.markdown(
        """
Pressure Solver  
This app enforces no-tension soil by iteratively clipping negative pressure and matching target P, Mx, My via a small least-squares update on an affine pressure field q = a + b x + c y. It produces a realistic contact polygon and heatmap even for large eccentricities. Cross-check with FEM or closed-form methods for critical designs.

Advancements to add next:
1) Optimization loop on B, L, D to minimize concrete/steel while satisfying governing check.
2) Raft (Mat) fallback: Winkler strips and punching at column nodes.
3) Crack width checks (IS 456 Annex F) based on bar spacing and cover.
4) Anchors for uplift/overturning with ACI 318 anchorage checks.
5) DXF export of rebar plans/sections via ezdxf.
6) PDF report generator (ReportLab/WeasyPrint) with clause references.
7) Seismic combinations per IS 1893; load-combination generator and auto-factoring.
8) Buoyancy for high GWT; subtract submerged weight and include uplift.
9) Excel/CSV import and batch design for multiple supports.
        """
    )
