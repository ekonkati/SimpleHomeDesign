# -*- coding: utf-8 -*-
# Streamlit App: RCC Foundation Designer (Biaxial Moments, Shear, No-Tension Soil, Pedestal)
# - Interactive charts (Plotly)
# - Separate top/bottom pad reinforcement (X & Y) with adequacy checks
# - Narrative calculations (stability, base pressure, flexure, shear, punching)
# - Pedestal design narrative
# - Plan + Cross-section rebar sketches
# - PDF export via reportlab

import io
import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


# ------------------------------ Defaults ------------------------------
def_header = {
    "project": {"title": "Pump House Column F3 Foundation", "location": "Hyderabad", "designer": "Your Firm", "reviewer": "Approving Agency"},
    "materials": {"fck_mpa": 30.0, "fy_mpa": 500.0, "gamma_c": 25.0},
    "geometry": {"column_bx": 0.4, "column_by": 0.4, "ped_px": 0.6, "ped_py": 0.6, "ped_h": 0.6, "foot_B": 2.2, "foot_L": 2.6, "foot_D": 0.55, "cover": 0.05},
    "soil": {"SBC_kPa_SLS": 200.0, "phi_bearing_factor_ULS": 0.67, "unit_weight": 18.0, "k_subgrade": 30000.0, "mu_base": 0.5, "GWT": 2.0, "allowable_settlement_mm": 25.0}
}
BAR_DIAM_MM_OPTIONS = [8, 10, 12, 16, 20, 25, 32]


def default_loads_df():
    return pd.DataFrame([
        {"Combo": "ULS1 (DL+LL)", "Type": "ULS", "P_kN": 1200.0, "Vx_kN": 50.0, "Vy_kN": 40.0, "Mx_kNm": 180.0, "My_kNm": 220.0},
        {"Combo": "ULS2 (DL+EQx)", "Type": "ULS", "P_kN": 950.0, "Vx_kN": 120.0, "Vy_kN": 30.0, "Mx_kNm": 280.0, "My_kNm": 90.0},
        {"Combo": "SLS1 (Service)", "Type": "SLS", "P_kN": 900.0, "Vx_kN": 40.0, "Vy_kN": 35.0, "Mx_kNm": 150.0, "My_kNm": 170.0},
    ])


# ------------------------------ Helpers ------------------------------
def num_input_float(label, minv, maxv, val, step, **kwargs):
    return st.number_input(label, min_value=float(minv), max_value=float(maxv), value=float(val), step=float(step), **kwargs)


@dataclass
class Geometry:
    B: float; L: float; D: float; cover: float; bx: float; by: float; ped_px: float; ped_py: float; ped_h: float


@dataclass
class Materials:
    fck: float; fy: float; gamma_c: float


# ------------------------------ Pressure solver ------------------------------
def linear_pressure_full_contact(P, Mx, My, B, L, nx=81, ny=81):
    A = B * L
    ex = My / P if P != 0 else 0.0
    ey = Mx / P if P != 0 else 0.0
    x = np.linspace(-B/2.0, B/2.0, nx)
    y = np.linspace(-L/2.0, L/2.0, ny)
    X, Y = np.meshgrid(x, y)
    q0 = P / A
    q = q0 * (1 + 12.0 * ex * X / (B**2) + 12.0 * ey * Y / (L**2))
    return X, Y, q, ex, ey


def no_tension_correction(P, Mx, My, B, L, max_iter=200, tol=1e-3, nx=81, ny=81):
    X, Y, q_init, ex, ey = linear_pressure_full_contact(P, Mx, My, B, L, nx, ny)
    dA = (B/(nx-1)) * (L/(ny-1))

    A_mat = np.vstack([np.ones_like(X).ravel(), X.ravel(), Y.ravel()]).T
    coeffs, *_ = np.linalg.lstsq(A_mat, q_init.ravel(), rcond=None)
    a, b, c = coeffs.tolist()

    lr = 0.2
    for _ in range(max_iter):
        q_field = a + b * X + c * Y
        q_pos = np.clip(q_field, 0.0, None)
        mask = q_pos > 0

        P_num = q_pos.sum() * dA
        Mx_num = (q_pos * Y).sum() * dA
        My_num = (q_pos * X).sum() * dA

        rP = P - P_num
        rMx = Mx - Mx_num
        rMy = My - My_num
        if max(abs(rP), abs(rMx), abs(rMy)) < tol:
            return X, Y, q_pos, mask, ex, ey

        Xm = X[mask]; Ym = Y[mask]
        if Xm.size == 0:
            break
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
    return X, Y, q_pos*scale, (q_pos > 0), ex, ey


# ------------------------------ Checks & design ------------------------------
def effective_depth(D, cover, bar_diam=16e-3):
    return max(D - cover - bar_diam/2.0, 0.05)


def bearing_check(q, SBC_SLS_kPa, phi_ULS, combo_type):
    qmax = float(np.max(q)); qmin = float(np.min(q))
    limit = SBC_SLS_kPa if combo_type.upper() == "SLS" else SBC_SLS_kPa * phi_ULS
    return {"qmax_kPa": qmax, "qmin_kPa": qmin, "limit_kPa": float(limit), "OK": (qmax <= limit and qmin >= -1e-6)}


def sliding_check(P_vert_kN, Vx_kN, Vy_kN, mu):
    R = mu * max(P_vert_kN, 0.0); V = math.hypot(Vx_kN, Vy_kN)
    return {"Resisting_kN": float(R), "Demand_kN": float(V), "OK": R >= V}


def overturning_check(P, Mx, My, B, L):
    ex = abs(My/P) if P else 0.0
    ey = abs(Mx/P) if P else 0.0
    return {"ex": float(ex), "ey": float(ey), "kern_x": B/6.0, "kern_y": L/6.0, "OK": (ex <= B/6.0 and ey <= L/6.0)}


def one_way_shear_check(q, X, Y, geom: Geometry, mat: Materials):
    d = effective_depth(geom.D, geom.cover); dA = (geom.B/(X.shape[1]-1))*(geom.L/(X.shape[0]-1))
    x_crit = geom.bx/2.0 + d
    Vx_pos = (q[:, X[0] >= x_crit].sum())*dA; Vx_neg = (q[:, X[0] <= -x_crit].sum())*dA
    Vx = max(Vx_pos, Vx_neg); b_x = geom.L; tau_vx = (Vx*1e3)/(b_x*d)/1000.0
    y_crit = geom.by/2.0 + d
    Vy_pos = (q[Y[:, 0] >= y_crit, :].sum())*dA; Vy_neg = (q[Y[:, 0] <= -y_crit, :].sum())*dA
    Vy = max(Vy_pos, Vy_neg); b_y = geom.B; tau_vy = (Vy*1e3)/(b_y*d)/1000.0
    tau_c = 0.62*math.sqrt(mat.fck)/1.5
    return {"one_way": {"X": {"V_kN": float(Vx), "tau_v_MPa": float(tau_vx), "tau_c_MPa": float(tau_c), "OK": tau_vx <= tau_c},
                        "Y": {"V_kN": float(Vy), "tau_v_MPa": float(tau_vy), "tau_c_MPa": float(tau_c), "OK": tau_vy <= tau_c}}}


def punching_shear_check(q, X, Y, geom: Geometry, mat: Materials):
    d = effective_depth(geom.D, geom.cover); perim_x = geom.ped_px + d; perim_y = geom.ped_py + d; u = 2.0*(perim_x+perim_y)
    dA = (geom.B/(X.shape[1]-1))*(geom.L/(X.shape[0]-1))
    inside = (np.abs(X) <= perim_x/2.0) & (np.abs(Y) <= perim_y/2.0)
    R_inside = q[inside].sum()*dA; V_u = (q.sum()*dA) - R_inside
    v_u = (V_u*1e3)/(u*d)/1000.0; v_c = 0.25*math.sqrt(mat.fck)/1.5
    return {"u_m": float(u), "V_u_kN": float(V_u), "v_u_MPa": float(v_u), "v_c_MPa": float(v_c), "OK": v_u <= v_c}


def strip_moments(q, X, Y, geom: Geometry):
    dA = (geom.B/(X.shape[1]-1))*(geom.L/(X.shape[0]-1))
    x_face = geom.ped_px/2.0; y_face = geom.ped_py/2.0
    right = X[0] >= x_face; left = X[0] <= -x_face
    q_right = q[:, right]; Xr = X[:, right]; Ar = q_right.sum()*dA
    xcg_r = (q_right*Xr).sum()*dA/Ar if Ar > 1e-9 else x_face; Mr = Ar*max(xcg_r - x_face, 0.0)
    q_left  = q[:, left];  Xl = X[:, left];  Al = q_left.sum()*dA
    xcg_l = (q_left*(-Xl)).sum()*dA/Al if Al > 1e-9 else x_face; Ml = Al*max(xcg_l - x_face, 0.0)
    My = max(Mr, Ml)
    top = Y[:, 0] >= y_face; bottom = Y[:, 0] <= -y_face
    q_top = q[top, :]; Yt = Y[top, :]; At = q_top.sum()*dA
    ycg_t = (q_top*Yt).sum()*dA/At if At > 1e-9 else y_face; Mt = At*max(ycg_t - y_face, 0.0)
    q_bot = q[bottom, :]; Yb = Y[bottom, :]; Ab = q_bot.sum()*dA
    ycg_b = (q_bot*(-Yb)).sum()*dA/Ab if Ab > 1e-9 else y_face; Mb = Ab*max(ycg_b - y_face, 0.0)
    Mx = max(Mt, Mb)
    return {"Mx_kNm": float(Mx), "My_kNm": float(My)}


def flexure_As_req_per_m(M_kNm, geom: Geometry, mat: Materials):
    d = effective_depth(geom.D, geom.cover); Mu = M_kNm*1000.0; fy = mat.fy*1e6; z = 0.9*max(d, 1e-6)
    As_m2 = Mu/(0.87*fy*z) if Mu > 0 else 0.0
    return As_m2*1e6  # mm^2/m


def bar_area_mm2(phi_mm: float) -> float:
    return math.pi*(phi_mm**2)/4.0


def As_prov_mm2pm_from(phi_mm: float, spacing_mm: float) -> float:
    return bar_area_mm2(phi_mm)*(1000.0/max(spacing_mm, 1e-6))


# ------------------------------ UI ------------------------------
try:
    st.set_page_config(page_title="RCC Foundation Designer (Biaxial)", layout="wide")
except Exception:
    pass

st.title("RCC Foundation Designer - Biaxial, No-Tension, Shear, Punching, Pedestal")
st.caption("Design aid per IS 456 / IS 2950 guidance. Verify with a licensed engineer.")

with st.sidebar:
    st.header("Project & Materials")
    proj, materials, geometry, soil = def_header["project"], def_header["materials"], def_header["geometry"], def_header["soil"]

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
        mu = num_input_float("Base friction mu", 0.2, 0.8, soil["mu_base"], 0.05)

    with st.expander("Geometry"):
        bx = num_input_float("Column size bx (m)", 0.2, 2.0, geometry["column_bx"], 0.05)
        by = num_input_float("Column size by (m)", 0.2, 2.0, geometry["column_by"], 0.05)
        ped_px = num_input_float("Pedestal px (m)", bx, 4.0, geometry["ped_px"], 0.05)
        ped_py = num_input_float("Pedestal py (m)", by, 4.0, geometry["ped_py"], 0.05)
        ped_h = num_input_float("Pedestal height (m)", 0.3, 2.0, geometry["ped_h"], 0.05)
        B = num_input_float("Footing B (m) - x dir", float(ped_px+0.2), 20.0, geometry["foot_B"], 0.1)
        L = num_input_float("Footing L (m) - y dir", float(ped_py+0.2), 20.0, geometry["foot_L"], 0.1)
        D = num_input_float("Footing thickness D (m)", 0.3, 2.0, geometry["foot_D"], 0.05)
        cover = num_input_float("Concrete cover (m)", 0.04, 0.1, geometry["cover"], 0.005)

mat = Materials(fck=fck, fy=fy, gamma_c=gamma_c)
geom = Geometry(B=B, L=L, D=D, cover=cover, bx=bx, by=by, ped_px=ped_px, ped_py=ped_py, ped_h=ped_h)

st.subheader("Load Combinations")
loads_df = st.data_editor(default_loads_df(), num_rows="dynamic", use_container_width=True)

# Pad reinforcement (provided) - top and bottom separate, X & Y each
st.subheader("Pad Reinforcement (Provided)")
colA, colB, colC, colD = st.columns(4)
with colA:
    phi_bx = st.selectbox("Bottom X dia (mm)", BAR_DIAM_MM_OPTIONS, index=BAR_DIAM_MM_OPTIONS.index(16))
    s_bx = num_input_float("Bottom X spacing (mm)", 50.0, 400.0, 150.0, 5.0)
with colB:
    phi_by = st.selectbox("Bottom Y dia (mm)", BAR_DIAM_MM_OPTIONS, index=BAR_DIAM_MM_OPTIONS.index(16))
    s_by = num_input_float("Bottom Y spacing (mm)", 50.0, 400.0, 150.0, 5.0)
with colC:
    phi_tx = st.selectbox("Top X dia (mm)", BAR_DIAM_MM_OPTIONS, index=BAR_DIAM_MM_OPTIONS.index(12))
    s_tx = num_input_float("Top X spacing (mm)", 50.0, 400.0, 150.0, 5.0)
with colD:
    phi_ty = st.selectbox("Top Y dia (mm)", BAR_DIAM_MM_OPTIONS, index=BAR_DIAM_MM_OPTIONS.index(12))
    s_ty = num_input_float("Top Y spacing (mm)", 50.0, 400.0, 150.0, 5.0)

ext_top = num_input_float("Top band extension from pedestal edge (m)", 0.1, 2.0, 0.5, 0.05)

# Pedestal reinforcement
st.subheader("Pedestal Reinforcement (Provided)")
colP1, colP2, colP3, colP4 = st.columns(4)
with colP1:
    ped_vert_phi = st.selectbox("Pedestal vertical dia (mm)", BAR_DIAM_MM_OPTIONS, index=BAR_DIAM_MM_OPTIONS.index(16))
    ped_vert_qty = st.number_input("Vertical bar qty", min_value=4, max_value=40, value=8, step=1)
with colP2:
    ped_tie_phi = st.selectbox("Pedestal tie dia (mm)", BAR_DIAM_MM_OPTIONS, index=BAR_DIAM_MM_OPTIONS.index(8))
    ped_tie_spacing = num_input_float("Tie spacing (mm)", 75.0, 300.0, 150.0, 5.0)
with colP3:
    ped_cover = num_input_float("Pedestal cover (m)", 0.04, 0.10, 0.05, 0.005)
with colP4:
    Ld_top = num_input_float("Ld top (mm)", 20.0, 2000.0, 47.0*float(ped_vert_phi), 10.0)
    Ld_bot = num_input_float("Ld bottom (mm)", 20.0, 2000.0, 47.0*float(ped_vert_phi), 10.0)

st.markdown("---")

run = st.button("Run Design (Interactive)")

results_rows: List[Dict] = []
narratives: List[str] = []

if run:
    st.success("Running analysis...")

    for i, row in loads_df.iterrows():
        combo = str(row.get("Combo", f"Combo_{i+1}"))
        ctype = str(row.get("Type", "ULS")).upper()
        P = float(row.get("P_kN", 0.0)); Vx = float(row.get("Vx_kN", 0.0)); Vy = float(row.get("Vy_kN", 0.0))
        Mx = float(row.get("Mx_kNm", 0.0)); My = float(row.get("My_kNm", 0.0))

        # Pressure & checks
        X, Y, q, mask, ex, ey = no_tension_correction(P, Mx, My, geom.B, geom.L, nx=121, ny=121)
        bearing = bearing_check(q, SBC, phi_ULS, ctype)
        slide = sliding_check(P, Vx, Vy, mu)
        ot = overturning_check(P, Mx, My, geom.B, geom.L)
        shear = one_way_shear_check(q, X, Y, geom, mat)
        punch = punching_shear_check(q, X, Y, geom, mat)
        strips = strip_moments(q, X, Y, geom)

        # Interactive pressure heatmap
        fig_hm = px.imshow(q, origin="lower", aspect="equal", labels=dict(color="q (kPa)"))
        fig_hm.update_layout(title=f"Soil Pressure Heatmap - {combo}")
        st.plotly_chart(fig_hm, use_container_width=True)

        # Interactive moment bars
        fig_bar = go.Figure(data=[go.Bar(x=['Mx (about X)', 'My (about Y)'], y=[strips['Mx_kNm'], strips['My_kNm']])])
        fig_bar.update_layout(title=f"Design Strip Moments - {combo}", yaxis_title="kNm")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Flexure requirements at pedestal face (per meter)
        Asx_req = flexure_As_req_per_m(strips["Mx_kNm"], geom, mat)
        Asy_req = flexure_As_req_per_m(strips["My_kNm"], geom, mat)

        # Provided (user) at bottom; top bands informational
        Asx_prov = As_prov_mm2pm_from(phi_bx, s_bx)
        Asy_prov = As_prov_mm2pm_from(phi_by, s_by)
        Asx_top_prov = As_prov_mm2pm_from(phi_tx, s_tx)
        Asy_top_prov = As_prov_mm2pm_from(phi_ty, s_ty)

        ok_Asx = Asx_prov >= Asx_req
        ok_Asy = Asy_prov >= Asy_req

        d_eff = effective_depth(geom.D, geom.cover)
        nar = []
        nar.append(f"### {combo} ({ctype})")
        nar.append(f"Loads: P={P:.2f} kN, Vx={Vx:.2f} kN, Vy={Vy:.2f} kN, Mx={Mx:.2f} kNm, My={My:.2f} kNm")
        nar.append(f"Eccentricities: e_x = My/P = {My:.2f}/{P:.2f} = {My/P if P else 0:.3f} m; e_y = Mx/P = {Mx:.2f}/{P:.2f} = {Mx/P if P else 0:.3f} m")
        nar.append(f"No-tension pressure solution gives q_max={bearing['qmax_kPa']:.1f} kPa, q_min={bearing['qmin_kPa']:.1f} kPa")
        nar.append(f"Bearing check: limit={bearing['limit_kPa']:.1f} -> {'OK' if bearing['OK'] else 'NG'}")
        nar.append(f"Sliding: mu*P={mu:.2f}*{P:.1f}={mu*P:.1f} kN vs V={math.hypot(Vx,Vy):.1f} kN -> {'OK' if slide['OK'] else 'NG'}")
        nar.append(f"Overturning: kern(B/6={geom.B/6:.3f}, L/6={geom.L/6:.3f}) -> {'OK' if ot['OK'] else 'NG'}")
        nar.append(f"One-way shear: X tau_v={shear['one_way']['X']['tau_v_MPa']:.3f} <= {shear['one_way']['X']['tau_c_MPa']:.3f} -> {'OK' if shear['one_way']['X']['OK'] else 'NG'}; "
                   f"Y tau_v={shear['one_way']['Y']['tau_v_MPa']:.3f} <= {shear['one_way']['Y']['tau_c_MPa']:.3f} -> {'OK' if shear['one_way']['Y']['OK'] else 'NG'}")
        nar.append(f"Punching: v_u={punch['v_u_MPa']:.3f} <= v_c={punch['v_c_MPa']:.3f} -> {'OK' if punch['OK'] else 'NG'} (u={punch['u_m']:.3f} m)")
        nar.append(f"Flexure strips at pedestal face: Mx={strips['Mx_kNm']:.1f} kNm, My={strips['My_kNm']:.1f} kNm; d={d_eff:.3f} m")
        nar.append(f"Required As per meter: X={Asx_req:.0f} mm^2/m, Y={Asy_req:.0f} mm^2/m")
        nar.append(f"Provided bottom: X T{int(phi_bx)} @ {s_bx:.0f} -> {Asx_prov:.0f} mm^2/m ({'OK' if ok_Asx else 'NG'}); "
                   f"Y T{int(phi_by)} @ {s_by:.0f} -> {Asy_prov:.0f} mm^2/m ({'OK' if ok_Asy else 'NG'})")
        nar.append(f"Top over pedestal (info): X T{int(phi_tx)} @ {s_tx:.0f} -> {Asx_top_prov:.0f} mm^2/m; "
                   f"Y T{int(phi_ty)} @ {s_ty:.0f} -> {Asy_top_prov:.0f} mm^2/m (band ext {ext_top:.2f} m)")
        narratives.append("\n".join(nar))

        results_rows.append({
            "Combo": combo, "Type": ctype,
            "qmax_kPa": bearing["qmax_kPa"], "qmin_kPa": bearing["qmin_kPa"], "Bearing_OK": bearing["OK"],
            "Sliding_OK": slide["OK"], "OT_OK": ot["OK"],
            "OW_X_OK": shear['one_way']['X']['OK'], "OW_Y_OK": shear['one_way']['Y']['OK'], "Punch_OK": punch['OK'],
            "Mx_kNm": strips['Mx_kNm'], "My_kNm": strips['My_kNm'],
            "Asx_req": Asx_req, "Asy_req": Asy_req,
            "Asx_prov": Asx_prov, "Asy_prov": Asy_prov,
            "Asx_OK": ok_Asx, "Asy_OK": ok_Asy
        })

    # Summary
    res_df = pd.DataFrame(results_rows)
    st.subheader("Design Summary")
    st.dataframe(res_df, use_container_width=True)

    # Plan (interactive): show band rectangle + indicative bar traces (schematic)
    st.subheader("Rebar Layout - Plan (interactive, schematic)")
    fig_plan = go.Figure()
    fig_plan.add_shape(type='rect', x0=-geom.B/2, x1=geom.B/2, y0=-geom.L/2, y1=geom.L/2, line=dict(width=2))
    fig_plan.add_shape(type='rect', x0=-geom.ped_px/2, x1=geom.ped_px/2, y0=-geom.ped_py/2, y1=geom.ped_py/2, line=dict(width=1))
    bx0 = -(geom.ped_px/2 + ext_top); bx1 = (geom.ped_px/2 + ext_top)
    by0 = -(geom.ped_py/2 + ext_top); by1 = (geom.ped_py/2 + ext_top)
    fig_plan.add_shape(type='rect', x0=bx0, x1=bx1, y0=by0, y1=by1, line=dict(dash='dot'))
    fig_plan.update_layout(title="Plan with Pedestal and Top Band", xaxis_title="x (m)", yaxis_title="y (m)", yaxis_scaleanchor="x", yaxis_scaleratio=1)
    st.plotly_chart(fig_plan, use_container_width=True)

    # Cross section (schematic)
    st.subheader("Rebar Cross Section (schematic, interactive)")
    fig_sec = go.Figure()
    fig_sec.add_shape(type='rect', x0=-geom.B/2, x1=geom.B/2, y0=0, y1=geom.D, line=dict(width=2))
    y_bot = cover; y_top = geom.D - cover
    # draw a few circles to indicate layers (schematic only)
    for x in np.linspace(-geom.B/2+cover, geom.B/2-cover, 8):
        fig_sec.add_shape(type='circle', x0=x-0.01, x1=x+0.01, y0=y_bot-0.01, y1=y_bot+0.01)
        fig_sec.add_shape(type='circle', x0=x-0.01, x1=x+0.01, y0=y_top-0.01, y1=y_top+0.01)
    fig_sec.update_layout(title="Section Through Footing", xaxis_title="x (m)", yaxis_title="depth (m)", yaxis_scaleanchor="x", yaxis_scaleratio=1)
    st.plotly_chart(fig_sec, use_container_width=True)

    # Pedestal narrative (simplified baseline)
    st.subheader("Pedestal Design - Narrative")
    A1 = geom.bx*geom.by; A2 = geom.ped_px*geom.ped_py
    inc = math.sqrt(max(A2/A1, 1.0)) if A1 > 0 else 1.0
    sigma_allow = min(0.45*mat.fck*inc, 0.9*mat.fck)
    Ast_vert = bar_area_mm2(ped_vert_phi)*ped_vert_qty
    ped_lines = [
        "### Pedestal",
        f"Bearing enhancement factor sqrt(A2/A1) = {inc:.3f}; allowable bearing ~ {sigma_allow:.2f} MPa (info).",
        f"Provide verticals: {int(ped_vert_qty)}-T{int(ped_vert_phi)} (Ast = {Ast_vert:.0f} mm^2).",
        f"Ties: T{int(ped_tie_phi)} @ {ped_tie_spacing:.0f} mm; cover {ped_cover:.2f} m; development: top {Ld_top:.0f} mm, bottom {Ld_bot:.0f} mm."
    ]
    st.markdown("\n".join(ped_lines))

    # Detailed narratives (all combos)
    st.subheader("Detailed Design Calculations (Narrative)")
    st.markdown("\n\n".join(narratives))

    # PDF export
    st.subheader("PDF Export")
    if not REPORTLAB_OK:
        st.info("Install reportlab to enable PDF export: pip install reportlab")
    else:
        pdf_buf = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buf, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph(f"<b>{title}</b> - {location}", styles["Title"]))
        story.append(Paragraph(f"Designer: {designer}  Reviewer: {reviewer}", styles["Normal"]))
        story.append(Paragraph(f"fck {mat.fck} MPa, fy {mat.fy} MPa; Footing {geom.B}x{geom.L}x{geom.D} m; Cover {geom.cover} m; "
                               f"Pedestal {geom.ped_px}x{geom.ped_py}x{geom.ped_h} m.", styles["Normal"]))
        story.append(Spacer(1, 8))
        story.append(Paragraph("<b>Detailed Calculations</b>", styles["Heading2"]))
        for blk in narratives:
            for line in blk.split("\n"):
                story.append(Paragraph(line.replace("<=", "&le;"), styles["Normal"]))
            story.append(Spacer(1, 6))
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Summary Table</b>", styles["Heading3"]))
        if not res_df.empty:
            cols = ["Combo","Bearing_OK","Sliding_OK","OT_OK","OW_X_OK","OW_Y_OK","Punch_OK","Asx_req","Asx_prov","Asy_req","Asy_prov"]
            data = [cols] + [[str(res_df.at[i,c]) for c in cols] for i in range(len(res_df))]
            t = Table(data, hAlign="LEFT")
            t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.black), ("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
            story.append(t)
        story.append(Spacer(1, 8))
        story.append(Paragraph("<b>Pedestal</b>", styles["Heading3"]))
        for line in ped_lines:
            story.append(Paragraph(line, styles["Normal"]))
        doc.build(story)
        pdf_buf.seek(0)
        st.download_button("Download PDF report", data=pdf_buf, file_name="foundation_design_report.pdf")
