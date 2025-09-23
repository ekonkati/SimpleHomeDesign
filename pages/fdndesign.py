# -*- coding: utf-8 -*-
# streamlit_foundation_app_ascii.py
# Streamlit App: RCC Foundation Designer (Biaxial Moments, Shear in 2 directions, No-Tension Soil, Pedestal)
# Author: ChatGPT (GPT-5 Thinking)
# Notes:
#  - ASCII-safe
#  - Float-safe number_input
#  - No-tension soil-pressure solver
#  - Interactive charts with Plotly
#  - Separate top/bottom pad reinforcement selection, adequacy checks
#  - Rebar layout (plan) and cross-section sketches
#  - BBS and PDF export with detailed narrative calculations

import io
import math
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Interactive charts
import plotly.express as px
import plotly.graph_objects as go

# Optional dependency for PDF export
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
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

# ------------------------------ Streamlit helpers ------------------------------

def num_input_float(label, minv, maxv, val, step, **kwargs):
    return st.number_input(label, min_value=float(minv), max_value=float(maxv), value=float(val), step=float(step), **kwargs)

# ------------------------------ Structural engine ------------------------------

@dataclass
class Geometry:
    B: float; L: float; D: float; cover: float; bx: float; by: float; ped_px: float; ped_py: float; ped_h: float

@dataclass
class Materials:
    fck: float; fy: float; gamma_c: float

# Pressure solver

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
    stats = {"iters": 0, "residual_P": None, "residual_Mx": None, "residual_My": None}
    for it in range(max_iter):
        q_field = a + b * X + c * Y
        q_pos = np.clip(q_field, 0.0, None)
        mask = q_pos > 0
        P_num = q_pos.sum() * dA
        Mx_num = (q_pos * Y).sum() * dA
        My_num = (q_pos * X).sum() * dA
        r = np.array([P - P_num, Mx - Mx_num, My - My_num])
        stats.update({"iters": it+1, "residual_P": float(r[0]), "residual_Mx": float(r[1]), "residual_My": float(r[2])})
        if np.all(np.abs(r) < tol):
            return X, Y, q_pos, mask, ex, ey, stats
        Xm = X[mask]; Ym = Y[mask]
        if Xm.size == 0: break
        ones = np.ones_like(Xm)
        dP  = np.array([ones.sum(), Xm.sum(), Ym.sum()]) * dA
        dMx = np.array([Ym.sum(), (Xm*Ym).sum(), (Ym*Ym).sum()]) * dA
        dMy = np.array([Xm.sum(), (Xm*Xm).sum(), (Xm*Ym).sum()]) * dA
        J = np.vstack([dP, dMx, dMy])
        try:
            delta = lr * np.linalg.lstsq(J, r, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        a += float(delta[0]); b += float(delta[1]); c += float(delta[2])
    q_field = a + b * X + c * Y
    q_pos = np.clip(q_field, 0.0, None)
    P_num = q_pos.sum() * dA
    scale = P / P_num if P_num > 0 else 1.0
    return X, Y, q_pos*scale, (q_pos>0), ex, ey, stats

# Checks & design helpers

def effective_depth(D, cover, bar_diam=16e-3):
    return max(D - cover - bar_diam/2.0, 0.05)

def bearing_check(q, SBC_SLS_kPa, phi_ULS, combo_type):
    qmax = float(np.max(q)); qmin = float(np.min(q))
    limit = SBC_SLS_kPa if combo_type.upper()=="SLS" else SBC_SLS_kPa * phi_ULS
    return {"qmax_kPa": qmax, "qmin_kPa": qmin, "limit_kPa": float(limit), "OK": (qmax<=limit and qmin>=-1e-6)}

def sliding_check(P_vert_kN, Vx_kN, Vy_kN, mu):
    R = mu * max(P_vert_kN, 0.0); V = math.hypot(Vx_kN, Vy_kN)
    return {"Resisting_kN": float(R), "Demand_kN": float(V), "OK": R>=V}

def overturning_check(P, Mx, My, B, L):
    ex = abs(My/P) if P else 0.0; ey = abs(Mx/P) if P else 0.0
    return {"ex": float(ex), "ey": float(ey), "kern_x": B/6.0, "kern_y": L/6.0, "OK": (ex<=B/6.0 and ey<=L/6.0)}

def one_way_shear_check(q, X, Y, geom: Geometry, mat: Materials):
    d = effective_depth(geom.D, geom.cover); dA = (geom.B/(X.shape[1]-1))*(geom.L/(X.shape[0]-1))
    x_crit = geom.bx/2.0 + d
    Vx_pos = (q[:, X[0]>=x_crit].sum())*dA; Vx_neg=(q[:, X[0]<=-x_crit].sum())*dA
    Vx = max(Vx_pos, Vx_neg); b_x = geom.L; tau_vx = (Vx*1e3)/(b_x*d)/1000.0
    y_crit = geom.by/2.0 + d
    Vy_pos = (q[Y[:,0]>=y_crit, :].sum())*dA; Vy_neg=(q[Y[:,0]<=-y_crit, :].sum())*dA
    Vy = max(Vy_pos, Vy_neg); b_y = geom.B; tau_vy = (Vy*1e3)/(b_y*d)/1000.0
    tau_c = 0.62*math.sqrt(mat.fck)/1.5
    return {"one_way": {"X": {"V_kN": float(Vx), "tau_v_MPa": float(tau_vx), "tau_c_MPa": float(tau_c), "OK": tau_vx<=tau_c},
                         "Y": {"V_kN": float(Vy), "tau_v_MPa": float(tau_vy), "tau_c_MPa": float(tau_c), "OK": tau_vy<=tau_c}}}

def punching_shear_check(q, X, Y, geom: Geometry, mat: Materials):
    d = effective_depth(geom.D, geom.cover); perim_x = geom.ped_px + d; perim_y = geom.ped_py + d; u = 2.0*(perim_x+perim_y)
    dA = (geom.B/(X.shape[1]-1))*(geom.L/(X.shape[0]-1))
    inside = (np.abs(X)<=perim_x/2.0) & (np.abs(Y)<=perim_y/2.0)
    R_inside = q[inside].sum()*dA; V_u = (q.sum()*dA)-R_inside
    v_u = (V_u*1e3)/(u*d)/1000.0; v_c = 0.25*math.sqrt(mat.fck)/1.5
    return {"u_m": float(u), "V_u_kN": float(V_u), "v_u_MPa": float(v_u), "v_c_MPa": float(v_c), "OK": v_u<=v_c}

def strip_moments(q, X, Y, geom: Geometry):
    dA = (geom.B/(X.shape[1]-1))*(geom.L/(X.shape[0]-1))
    x_face=geom.ped_px/2.0; y_face=geom.ped_py/2.0
    right=X[0]>=x_face; left=X[0]<=-x_face
    q_right=q[:,right]; Xr=X[:,right]; Ar=q_right.sum()*dA; xcg_r=(q_right*Xr).sum()*dA/Ar if Ar>1e-9 else x_face; Mr=Ar*max(xcg_r-x_face,0.0)
    q_left=q[:,left];  Xl=X[:,left];  Al=q_left.sum()*dA;  xcg_l=(q_left*(-Xl)).sum()*dA/Al if Al>1e-9 else x_face; Ml=Al*max(xcg_l-x_face,0.0)
    My=max(Mr,Ml)
    top=Y[:,0]>=y_face; bottom=Y[:,0]<=-y_face
    q_top=q[top,:]; Yt=Y[top,:]; At=q_top.sum()*dA; ycg_t=(q_top*Yt).sum()*dA/At if At>1e-9 else y_face; Mt=At*max(ycg_t-y_face,0.0)
    q_bot=q[bottom,:]; Yb=Y[bottom,:]; Ab=q_bot.sum()*dA; ycg_b=(q_bot*(-Yb)).sum()*dA/Ab if Ab>1e-9 else y_face; Mb=Ab*max(ycg_b-y_face,0.0)
    Mx=max(Mt,Mb)
    return {"Mx_kNm": float(Mx), "My_kNm": float(My)}

def flexure_As_req_per_m(M_kNm, geom: Geometry, mat: Materials):
    d = effective_depth(geom.D, geom.cover); Mu = M_kNm*1000.0; fy=mat.fy*1e6; z=0.9*d
    As_m2 = Mu/(0.87*fy*max(z,1e-6)) if Mu>0 else 0.0
    return As_m2*1e6  # mm^2/m

# Reinforcement utilities

def bar_area_mm2(phi_mm: float) -> float: return math.pi*(phi_mm**2)/4.0

def bar_unit_weight_kgpm(phi_mm: float) -> float: return (phi_mm**2)/162.0

def As_prov_mm2pm_from(phi_mm: float, spacing_mm: float) -> float:
    return bar_area_mm2(phi_mm)*(1000.0/max(spacing_mm,1e-6))

# Sketching helpers (plan and section)

def rebar_layout_plan_plotly(geom: Geometry, cover_m: float, s_bx: float, s_by: float, s_tx: float, s_ty: float, top_ext_m: float) -> go.Figure:
    fig = go.Figure()
    # Footprint
    fig.add_shape(type='rect', x0=-geom.B/2, x1=geom.B/2, y0=-geom.L/2, y1=geom.L/2, line=dict(width=2))
    # Pedestal
    fig.add_shape(type='rect', x0=-geom.ped_px/2, x1=geom.ped_px/2, y0=-geom.ped_py/2, y1=geom.ped_py/2, line=dict(width=1))
    x0=-geom.B/2+cover_m; x1=geom.B/2-cover_m; y0=-geom.L/2+cover_m; y1=geom.L/2-cover_m
    # Bottom X bars (horizontal lines)
    y=y0
    while y<=y1+1e-9 and s_bx>0:
        fig.add_shape(type='line', x0=x0, x1=x1, y0=y, y1=y)
        y+=s_bx/1000.0
    # Bottom Y bars (vertical)
    x=x0
    while x<=x1+1e-9 and s_by>0:
        fig.add_shape(type='line', x0=x, x1=x, y0=y0, y1=y1)
        x+=s_by/1000.0
    # Top band ext envelope
    bx0=- (geom.ped_px/2 + top_ext_m); bx1=(geom.ped_px/2 + top_ext_m)
    by0=- (geom.ped_py/2 + top_ext_m); by1=(geom.ped_py/2 + top_ext_m)
    fig.add_shape(type='rect', x0=bx0, x1=bx1, y0=by0, y1=by1, line=dict(dash='dot'))
    # Top X bars in y-band
    y=max(by0,y0)
    while y<=min(by1,y1)+1e-9 and s_tx>0:
        fig.add_shape(type='line', x0=x0, x1=x1, y0=y, y1=y)
        y+=s_tx/1000.0
    # Top Y bars in x-band
    x=max(bx0,x0)
    while x<=min(bx1,x1)+1e-9 and s_ty>0:
        fig.add_shape(type='line', x0=x, x1=x, y0=y0, y1=y1)
        x+=s_ty/1000.0
    fig.update_layout(title='Rebar Layout (Plan) - bottom solid, top within dotted band', xaxis_title='x (m)', yaxis_title='y (m)', yaxis_scaleanchor="x", yaxis_scaleratio=1, dragmode='pan')
    return fig


def section_plotly(geom: Geometry, cover_m: float, phi_bx: float, s_bx: float, phi_by: float, s_by: float, phi_tx: float, s_tx: float):
    """Simple cross-section along short direction (B) showing bottom and top bars."""
    fig = go.Figure()
    # Footing slab
    fig.add_shape(type='rect', x0=-geom.B/2, x1=geom.B/2, y0=0, y1=geom.D, line=dict(width=2))
    # Ground line
    fig.add_shape(type='line', x0=-geom.B/2, x1=geom.B/2, y0=0, y1=0)
    # Bottom layer bars (draw as circles along width)
    y_bot = cover_m
    x0=-geom.B/2+cover_m; x1=geom.B/2-cover_m
    x=x0
    dia_bx = phi_bx/1000.0
    while x<=x1+1e-9 and s_bx>0:
        fig.add_shape(type='circle', x0=x-dia_bx/2, x1=x+dia_bx/2, y0=y_bot-dia_bx/2, y1=y_bot+dia_bx/2)
        x+=s_bx/1000.0
    # Top layer bars (over pedestal) shown centrally for simplicity
    y_top = geom.D - cover_m
    x=x0
    dia_tx = phi_tx/1000.0
    while x<=x1+1e-9 and s_tx>0:
        fig.add_shape(type='circle', x0=x-dia_tx/2, x1=x+dia_tx/2, y0=y_top-dia_tx/2, y1=y_top+dia_tx/2)
        x+=s_tx/1000.0
    fig.update_layout(title='Cross Section (schematic)', xaxis_title='x (m)', yaxis_title='Depth (m)', yaxis_scaleanchor="x", yaxis_scaleratio=1, dragmode='pan')
    return fig

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

# Reinforcement selection panel (Pad Top & Bottom separate, X & Y each)
st.subheader("Pad Reinforcement Selection (Provided)")
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

# Pedestal reinforcement selection
st.subheader("Pedestal Reinforcement Selection (Provided)")
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

run = st.button("Run Design and Plots (Interactive)")

results_rows: List[Dict] = []
plots_png: Dict[str, bytes] = {}
report_narrative: List[str] = []

if run:
    st.success("Running analysis...")

    for i, row in loads_df.iterrows():
        combo = str(row.get("Combo", f"Combo_{i+1}"))
        ctype = str(row.get("Type", "ULS")).upper()
        P = float(row.get("P_kN", 0.0)); Vx = float(row.get("Vx_kN", 0.0)); Vy = float(row.get("Vy_kN", 0.0)); Mx = float(row.get("Mx_kNm", 0.0)); My = float(row.get("My_kNm", 0.0))
        X, Y, q, mask, ex, ey, stats = no_tension_correction(P, Mx, My, geom.B, geom.L, nx=121, ny=121)

        # Interactive heatmap (kPa)
        fig_hm = px.imshow(q, origin='lower', aspect='equal', labels=dict(color='q (kPa)'))
        fig_hm.update_layout(title=f"Soil Pressure Heatmap - {combo}")
        fig_hm.update_xaxes(title_text='x index'); fig_hm.update_yaxes(title_text='y index')
        st.plotly_chart(fig_hm, use_container_width=True)

        # Moment strips
        stripM = strip_moments(q, X, Y, geom)
        fig_bar = go.Figure(data=[go.Bar(x=['Mx (about X)','My (about Y)'], y=[stripM['Mx_kNm'], stripM['My_kNm']])])
        fig_bar.update_layout(title=f"Design Strip Moments - {combo}", yaxis_title='kNm')
        st.plotly_chart(fig_bar, use_container_width=True)

        # Checks
        bearing = bearing_check(q, SBC, phi_ULS, ctype)
        slide = sliding_check(P, Vx, Vy, mu)
        ot = overturning_check(P, Mx, My, geom.B, geom.L)
        shear = one_way_shear_check(q, X, Y, geom, mat)
        punch = punching_shear_check(q, X, Y, geom, mat)

        # As requirements per meter (for bottom design):
        Asx_req = flexure_As_req_per_m(stripM["Mx_kNm"], geom, mat)
        Asy_req = flexure_As_req_per_m(stripM["My_kNm"], geom, mat)

        # Provided from user selections
        Asx_prov = As_prov_mm2pm_from(phi_bx, s_bx)
        Asy_prov = As_prov_mm2pm_from(phi_by, s_by)
        Asx_top_prov = As_prov_mm2pm_from(phi_tx, s_tx)
        Asy_top_prov = As_prov_mm2pm_from(phi_ty, s_ty)

        # Adequacy flags
        ok_Asx = Asx_prov >= Asx_req
        ok_Asy = Asy_prov >= Asy_req

        # Narrative calculation block
        d_eff = effective_depth(geom.D, geom.cover)
        A_foot = geom.B*geom.L; qavg = P/A_foot if A_foot>0 else 0.0
        nar = []
        nar.append(f"### {combo} ({ctype})")
        nar.append(f"Loads: P={P:.2f} kN, Vx={Vx:.2f} kN, Vy={Vy:.2f} kN, Mx={Mx:.2f} kNm, My={My:.2f} kNm")
        nar.append(f"Eccentricities: e_x=My/P={My:.2f}/{P:.2f}={My/P if P else 0:.3f} m, e_y=Mz/P={Mx:.2f}/{P:.2f}={Mx/P if P else 0:.3f} m")
        nar.append(f"Base pressure computed with no-tension correction to satisfy P, Mx, My with q>=0. q_max={bearing['qmax_kPa']:.1f} kPa, q_min={bearing['qmin_kPa']:.1f} kPa")
        nar.append(f"Bearing check: limit={bearing['limit_kPa']:.1f} kPa -> {'OK' if bearing['OK'] else 'NG'}")
        nar.append(f"Sliding: mu*P={mu:.2f}*{P:.1f}={mu*P:.1f} kN vs resultant V={math.hypot(Vx,Vy):.1f} kN -> {'OK' if slide['OK'] else 'NG'}")
        nar.append(f"Overturning (kern): ex<={geom.B/6:.3f} m, ey<={geom.L/6:.3f} m -> {'OK' if ot['OK'] else 'NG'}")
        nar.append(f"One-way shear: tau_vx={shear['one_way']['X']['tau_v_MPa']:.3f} MPa <= tau_c={shear['one_way']['X']['tau_c_MPa']:.3f} MPa -> {'OK' if shear['one_way']['X']['OK'] else 'NG'}; "
                   f"tau_vy={shear['one_way']['Y']['tau_v_MPa']:.3f} MPa <= tau_c={shear['one_way']['Y']['tau_c_MPa']:.3f} MPa -> {'OK' if shear['one_way']['Y']['OK'] else 'NG'}")
        nar.append(f"Punching: v_u={punch['v_u_MPa']:.3f} MPa <= v_c={punch['v_c_MPa']:.3f} MPa -> {'OK' if punch['OK'] else 'NG'} (u={punch['u_m']:.3f} m around pedestal)")
        nar.append(f"Flexure strips at pedestal face: Mx={stripM['Mx_kNm']:.1f} kNm, My={stripM['My_kNm']:.1f} kNm; d={d_eff:.3f} m")
        nar.append(f"Required steel per meter: Asx={Asx_req:.0f} mm^2/m, Asy={Asy_req:.0f} mm^2/m")
        nar.append(f"Provided bottom steel: X T{int(phi_bx)} @ {s_bx:.0f} -> {Asx_prov:.0f} mm^2/m ({'OK' if ok_Asx else 'NG'}); "
                   f"Y T{int(phi_by)} @ {s_by:.0f} -> {Asy_prov:.0f} mm^2/m ({'OK' if ok_Asy else 'NG'})")
        nar.append(f"Top bands provided over pedestal: X T{int(phi_tx)} @ {s_tx:.0f} -> {Asx_top_prov:.0f} mm^2/m; Y T{int(phi_ty)} @ {s_ty:.0f} -> {Asy_top_prov:.0f} mm^2/m")
        report_narrative.append("
".join(nar))

        # Results row
        results_rows.append({
            "Combo": combo, "Type": ctype,
            "qmax_kPa": bearing["qmax_kPa"], "qmin_kPa": bearing["qmin_kPa"], "Bearing_OK": bearing["OK"],
            "Sliding_OK": slide["OK"], "OT_OK": ot["OK"],
            "OW_X_OK": shear['one_way']['X']['OK'], "OW_Y_OK": shear['one_way']['Y']['OK'], "Punch_OK": punch['OK'],
            "Mx_kNm": stripM['Mx_kNm'], "My_kNm": stripM['My_kNm'],
            "Asx_req": Asx_req, "Asy_req": Asy_req,
            "Asx_prov": Asx_prov, "Asy_prov": Asy_prov,
            "Asx_OK": ok_Asx, "Asy_OK": ok_Asy
        })

    # Summary table
    res_df = pd.DataFrame(results_rows)
    st.subheader("Design Summary (per combo)")
    st.dataframe(res_df, use_container_width=True)

    # Plan layout (interactive) and cross section
    st.subheader("Rebar Layout - Plan (interactive)")
    fig_plan = rebar_layout_plan_plotly(geom, geom.cover, s_bx, s_by, s_tx, s_ty, ext_top)
    st.plotly_chart(fig_plan, use_container_width=True)

    st.subheader("Rebar Cross Section (schematic, interactive)")
    st.plotly_chart(section_plotly(geom, geom.cover, phi_bx, s_bx, phi_by, s_by, phi_tx, s_tx), use_container_width=True)

    # BBS (simple)
    st.subheader("Adequacy Flags")
    st.write(f"Bottom X steel OK: {res_df['Asx_OK'].all() if len(res_df)>0 else True}")
    st.write(f"Bottom Y steel OK: {res_df['Asy_OK'].all() if len(res_df)>0 else True}")

    # Pedestal detailed design (narrative)
    st.subheader("Pedestal Design - Detailed Narrative")
    # Bearing enhancement and vertical steel area check (simplified)
    A1 = geom.bx*geom.by; A2 = geom.ped_px*geom.ped_py; inc = math.sqrt(max(A2/A1,1.0)) if A1>0 else 1.0
    sigma_allow = min(0.45*mat.fck*inc, 0.9*mat.fck)
    Ast_vert_mm2 = bar_area_mm2(ped_vert_phi)*ped_vert_qty
    min_pct = 0.8  # promille (0.8%) placeholder; adjust to code requirements
    Ast_min_mm2 = min_pct/100.0 * (geom.ped_px*1000.0)*(geom.ped_py*1000.0)  # gross approx for narrative
    ped_nar = []
    ped_nar.append("### Pedestal")
    ped_nar.append(f"Bearing enhancement: inc=sqrt(A2/A1)={inc:.3f}, allowable bearing={sigma_allow:.2f} MPa (limit)")
    ped_nar.append(f"Provided verticals: {int(ped_vert_qty)}-T{int(ped_vert_phi)} -> Ast={Ast_vert_mm2:.0f} mm^2")
    ped_nar.append(f"Minimum steel (approx narrative): {Ast_min_mm2:.0f} mm^2 -> {'OK' if Ast_vert_mm2>=Ast_min_mm2 else 'NG'}")
    ped_nar.append(f"Ties: T{int(ped_tie_phi)} @ {ped_tie_spacing:.0f} mm within {geom.ped_h:.2f} m height; cover {ped_cover:.2f} m")
    st.markdown("
".join(ped_nar))

    # Narrative block (all combos)
    st.subheader("Detailed Design Calculations (Narrative)")
    st.markdown("

".join(report_narrative))

    # Export PDF
    st.subheader("PDF Export")
    if not REPORTLAB_OK:
        st.info("Install reportlab to enable PDF export: pip install reportlab")
    else:
        pdf_buf = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buf, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
        styles = getSampleStyleSheet(); story = []
        story.append(Paragraph(f"<b>{title}</b> - {location}", styles['Title']))
        story.append(Paragraph(f"Designer: {designer} &nbsp; Reviewer: {reviewer}", styles['Normal']))
        story.append(Paragraph(f"Concrete fck {mat.fck} MPa, Steel fy {mat.fy} MPa, Footing {geom.B}x{geom.L}x{geom.D} m, Cover {geom.cover} m", styles['Normal']))
        story.append(Spacer(1,8))
        story.append(Paragraph("<b>Detailed Calculations</b>", styles['Heading2']))
        for blk in report_narrative:
            for para in blk.split("
"):
                story.append(Paragraph(para.replace('<=','&le;'), styles['Normal']))
            story.append(Spacer(1,6))
        # Adequacy
        story.append(Paragraph("<b>Pad Reinforcement Provided</b>", styles['Heading3']))
        story.append(Paragraph(f"Bottom: X T{int(phi_bx)} @ {s_bx:.0f} mm; Y T{int(phi_by)} @ {s_by:.0f} mm", styles['Normal']))
        story.append(Paragraph(f"Top over pedestal: X T{int(phi_tx)} @ {s_tx:.0f} mm; Y T{int(phi_ty)} @ {s_ty:.0f} mm", styles['Normal']))
        # Simple table for summary
        if not res_df.empty:
            cols = ["Combo","Bearing_OK","Sliding_OK","OT_OK","OW_X_OK","OW_Y_OK","Punch_OK","Asx_req","Asx_prov","Asy_req","Asy_prov"]
            data_table = [cols] + [[str(res_df.at[i,c]) for c in cols] for i in range(len(res_df))]
            t = Table(data_table, hAlign='LEFT')
            t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.black), ('BACKGROUND',(0,0),(-1,0),colors.lightgrey)]))
            story.append(t)
        story.append(Spacer(1,8))
        # Pedestal summary
        story.append(Paragraph("<b>Pedestal Design</b>", styles['Heading3']))
        for para in ped_nar:
            story.append(Paragraph(para.replace('<=','&le;'), styles['Normal']))
        doc.build(story)
        pdf_buf.seek(0)
        st.download_button("Download PDF report", data=pdf_buf, file_name="foundation_design_report.pdf")

# Footer notes
with st.expander("Notes and Next Steps"):
    st.markdown(
        """
- Interactive charts use Plotly; pan/zoom and hover to inspect pressures and strip moments.
- Adequacy checks compare required As (from strip moments at pedestal face) against user-provided bottom mesh per meter.
- For production: enforce IS 456 limits for max spacing based on slab depth and environment; compute Ld from bond stress tables; generate DXF drawings and shape-code exact BBS lengths.
        """
    )
