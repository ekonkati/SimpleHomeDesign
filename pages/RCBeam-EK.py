import math
from dataclasses import dataclass
import io
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json

st.set_page_config(page_title="Printable RCC Beam Designer (IS Codes)", layout="wide")
st.title("RCC Beam Designer – Printable Narrative Report")
st.caption("Single-page design report based on IS 456 and IS 13920 (Advisory). Units: length in m/mm, loads in kN.")

# ---------- CONSTANTS AND HELPERS ----------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

@dataclass
class Materials:
    fck: int  # N/mm2
    fy: int   # N/mm2 (Fe415/500)
    density: float = 25.0  # kN/m3

@dataclass
class Section:
    b: float  # mm
    D: float  # mm overall
    cover: float  # mm to main tension steel

    @property
    def d_eff(self):
        # Assuming 8mm link bar and 16mm main bar for d_eff calculation
        return max(self.D - (self.cover + 8 + 0.5*16), 0.0)

# IS Code Tables (Simplified)
TAU_BD_PLAIN = {20:1.2, 25:1.4, 30:1.5, 35:1.7, 40:1.9, 45:2.0, 50:2.2}
TAU_C_MAX = {20:2.8, 25:3.1, 30:3.5, 35:3.7, 40:4.0}
TC_TABLE = {
    20: [(0.25,0.28),(0.50,0.38),(0.75,0.46),(1.00,0.62)],
    25: [(0.25,0.29),(0.50,0.40),(0.75,0.48),(1.00,0.62)],
    30: [(0.25,0.29),(0.50,0.40),(0.75,0.48),(1.00,0.62)],
    35: [(0.25,0.30),(0.50,0.42),(0.75,0.50),(1.00,0.62)],
    40: [(0.25,0.31),(0.50,0.44),(0.75,0.52),(1.00,0.62)],
}
LD_LIMITS = {"Simply Supported":20.0, "Continuous":26.0, "Cantilever":7.0}

def interp_xy(table, x):
    xs = [a for a,_ in table]
    ys = [b for _,b in table]
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(len(xs)-1):
        if xs[i] <= x <= xs[i+1]:
            x0,y0 = xs[i],ys[i]
            x1,y1 = xs[i+1],ys[i+1]
            t = (x - x0) / (x1 - x0)
            return y0 + t*(y1-y0)

def tau_c(fck, p_t):
    fck_key = min(TC_TABLE.keys(), key=lambda k: abs(k - fck))
    return interp_xy(TC_TABLE[fck_key], clamp(p_t, 0.25, 1.0))

def mu_lim_rect(fck, b_mm, d_mm, fy):
    k = 0.138 if fy <= 415 else 0.133 
    MuNmm = k * fck * b_mm * (d_mm**2)
    return MuNmm / 1e6 

def ast_singly(Mu_kNm, fy, d_mm, jd_ratio=0.9):
    Mu_Nmm = Mu_kNm * 1e6
    jd = jd_ratio * d_mm
    return Mu_Nmm / (0.87 * fy * jd)

def ld_required(fck, fy, bar_dia_mm, deformed=True, tension=True):
    fck_key = min(TAU_BD_PLAIN.keys(), key=lambda k: abs(k - fck))
    tau_bd = TAU_BD_PLAIN[fck_key]
    if deformed: tau_bd *= 1.6
    if not tension: tau_bd *= 1.25 
    Ld = (bar_dia_mm * fy) / (4.0 * tau_bd)
    return Ld

def shear_design(Vu_kN_eff, b_mm, d_mm, fck, fy, Ast_mm2):
    tau_v = (Vu_kN_eff * 1e3) / (b_mm * d_mm)
    p_t = 100.0 * Ast_mm2 / (b_mm * d_mm)
    tc = tau_c(fck, p_t)
    tc_max = TAU_C_MAX[min(TAU_C_MAX.keys(), key=lambda k: abs(k - fck))]
    
    Vus_kN = Vu_kN_eff - tc * b_mm * d_mm / 1e3
    
    phi, legs = 8.0, 2
    Asv_default = legs * math.pi * (phi**2) / 4.0
    s_v_min = (0.87 * fy * Asv_default) / (0.4 * b_mm) 
    s_v_max_limit = min(0.75 * d_mm, 300.0)
    
    return {"tau_v": tau_v, "p_t": p_t, "tau_c": tc, "tau_c_max": tc_max,
            "Vus_kN": Vus_kN, 
            "s_v_min_control": min(s_v_min, s_v_max_limit),
            "min_shear_needed": Vus_kN <= 0,
            "d_eff": d_mm}

def get_fsc(fy, d_prime_over_d):
    fsc_vals = {
        415: [(0.05, 355), (0.10, 353), (0.15, 342), (0.20, 329)],
        500: [(0.05, 412), (0.10, 395), (0.15, 379), (0.20, 368)]
    }
    
    if d_prime_over_d >= 0.20: return fsc_vals[fy][-1][1]
    if d_prime_over_d <= 0.05: return fsc_vals[fy][0][1]

    for i in range(len(fsc_vals[fy])-1):
        d0, f0 = fsc_vals[fy][i]
        d1, f1 = fsc_vals[fy][i+1]
        if d0 < d_prime_over_d <= d1:
            return f0 + (f1 - f0) * (d_prime_over_d - d0) / (d1 - d0)
    return 300 

def ast_asc_doubly(Mu_kNm, Mu_lim_kNm, fck, fy, d_mm, d_prime_mm, b_mm):
    
    Mu_Nmm = Mu_kNm * 1e6
    Mu_lim_Nmm = Mu_lim_kNm * 1e6
    
    k_mulim = 0.138 if fy <= 415 else 0.133
    Ast_lim = (0.36 * fck * 0.48 * d_mm * b_mm) / (0.87 * fy) # 0.48*d_mm is xu,max for Fe500
    
    delta_Mu_Nmm = Mu_Nmm - Mu_lim_Nmm
    
    d_prime_over_d = d_prime_mm / d_mm
    fsc = get_fsc(fy, d_prime_over_d)
    
    Asc_req = delta_Mu_Nmm / (fsc * (d_mm - d_prime_mm))
    Ast2 = delta_Mu_Nmm / (0.87 * fy * (d_mm - d_prime_mm))

    Ast_total_req = Ast_lim + Ast2

    return Ast_total_req, Asc_req, Ast_lim


# ---------- JSON IMPORT/EXPORT FUNCTIONS ----------

# Initialize session state for all inputs if they don't exist
# This allows the app to maintain state during JSON load
DEFAULT_STATE = {
    "span": 6.0, "support": "Simply Supported", "b": 300, "D": 500, "cover": 25,
    "fck": 30, "fy": 500, "t_bar_d_default": 16, "c_bar_d_default": 12,
    "finishes": 2.0, "ll": 5.0, "use_wall": False, "wall_thk": 115, "wall_h": 3.0, "wall_density": 19.0,
    "include_eq": False, "eq_coeff": 0.0, "ductile": True, 
    "action_mode": 0, "Mu_in": 120.0, "Vu_in": 180.0, "Tu_in": 0.0, "Nu_in": 0.0,
    # Provided Reinforcement (design checks)
    "t_bar_d": 20, "t_bar_count": 3,
    "c_bar_d": 12, "c_bar_count": 2, # Note: c_bar_d_default used here for initial sync
    "stirrup_d": 8, "stirrup_legs": 2, "s_v_prov": 150,
    # Drawing table (must be synced after flexure inputs)
    "rebar_df_draw": None 
}

for key, default_val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

def export_design_to_json():
    # Collect all input values from session state
    export_data = {key: st.session_state[key] for key in DEFAULT_STATE.keys()}
    
    # Handle the DataFrame specifically
    if st.session_state.rebar_df_draw is not None:
        export_data["rebar_df_draw"] = st.session_state.rebar_df_draw.to_dict('records')
    
    json_string = json.dumps(export_data, indent=4)
    return json_string

def import_design_from_json(json_file):
    try:
        data = json.load(json_file)
        # Update session state with imported values
        for key, val in data.items():
            if key in DEFAULT_STATE:
                if key == "rebar_df_draw" and isinstance(val, list):
                    st.session_state[key] = pd.DataFrame(val)
                else:
                    st.session_state[key] = val
        st.toast("✅ Design successfully loaded! Re-run the app to see all changes.", icon='💾')
        st.experimental_rerun() # Rerun to update all widgets
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")


# ---------- UI INPUTS (GENERAL) ----------

with st.sidebar:
    st.header("Save/Load Design")
    
    # Export button
    json_export = export_design_to_json()
    st.download_button(
        "⬇️ Export Design State (JSON)",
        data=json_export,
        file_name="beam_design_state.json",
        mime="application/json",
        help="Download the current design inputs as a JSON file."
    )
    
    # Import uploader
    uploaded_file = st.file_uploader(
        "⬆️ Import Design State (JSON)",
        type="json",
        help="Upload a previously saved design state JSON file.",
        accept_multiple_files=False
    )
    if uploaded_file is not None:
        import_design_from_json(uploaded_file)
    
    st.markdown("---")
    st.header("Quick Toggles")
    st.session_state.use_wall = st.checkbox("Add wall load", value=st.session_state.use_wall)
    st.session_state.include_eq = st.checkbox("Include E/W coeff", value=st.session_state.include_eq)
    st.session_state.eq_coeff = st.number_input("Eq. coeff (×wL²)", value=st.session_state.eq_coeff, step=0.05, disabled=not st.session_state.include_eq)
    st.session_state.ductile = st.checkbox("Apply IS 13920 checks", value=st.session_state.ductile)
    st.caption("⚠️ Use the 'Print' function of your browser (Ctrl+P / Cmd+P) for the final report.")
    st.markdown("---")


st.header("1. Project Inputs")
st.markdown("---")
colA, colB, colC = st.columns(3)

with colA:
    st.subheader("Geometry")
    st.session_state.span = st.number_input("Clear span $L$ (m)", value=st.session_state.span, min_value=0.5, step=0.1)
    st.session_state.support = st.selectbox("End condition", ["Simply Supported","Continuous","Cantilever"], index=["Simply Supported","Continuous","Cantilever"].index(st.session_state.support))
    st.session_state.b = st.number_input("Beam width $b$ (mm)", value=st.session_state.b, step=10, min_value=150)
    st.session_state.D = st.number_input("Overall depth $D$ (mm)", value=st.session_state.D, step=10, min_value=200)
    st.session_state.cover = st.number_input("Clear cover (mm)", value=st.session_state.cover, step=5, min_value=20)

with colB:
    st.subheader("Materials")
    st.session_state.fck = st.selectbox("Concrete $f_{ck}$ (N/mm²)", [20,25,30,35,40], index=[20,25,30,35,40].index(st.session_state.fck))
    st.session_state.fy = st.selectbox("Steel $f_y$ (N/mm²)", [415,500], index=[415,500].index(st.session_state.fy))
    st.session_state.t_bar_d_default = st.selectbox("Tension Bar $\phi$ for $L_d$ (mm)", [12,16,20,25,28,32], index=[12,16,20,25,28,32].index(st.session_state.t_bar_d_default))
    st.session_state.c_bar_d_default = st.selectbox("Compression Bar $\phi$ for $L_d$ (mm)", [12,16,20,25], index=[12,16,20,25].index(st.session_state.c_bar_d_default))

with colC:
    st.subheader("Loads")
    st.session_state.finishes = st.number_input("Finishes (kN/m)", value=st.session_state.finishes, step=0.1, min_value=0.0)
    st.session_state.ll = st.number_input("Live load (kN/m)", value=st.session_state.ll, step=0.5, min_value=0.0)
    st.session_state.wall_thk = st.number_input("Wall thk (mm)", value=st.session_state.wall_thk, step=115, min_value=0, disabled=not st.session_state.use_wall)
    st.session_state.wall_h = st.number_input("Wall height (m)", value=st.session_state.wall_h, step=0.1, min_value=0.0, disabled=not st.session_state.use_wall)
    st.session_state.wall_density = st.number_input("Masonry density (kN/m³)", value=st.session_state.wall_density, step=0.5, disabled=not st.session_state.use_wall)

st.subheader("Design Action Source")
action_options = ["Derive from loads", "Direct design actions"]
st.session_state.action_mode = st.radio("Use:", action_options, index=st.session_state.action_mode if isinstance(st.session_state.action_mode, int) else action_options.index(st.session_state.action_mode), horizontal=True)

Mu_in, Vu_in, Tu_in, Nu_in = 0.0, 0.0, 0.0, 0.0
if st.session_state.action_mode == "Direct design actions":
    st.info("Enter factored design actions (ULS) at the critical section.")
    colX,colY = st.columns(2)
    with colX:
        st.session_state.Mu_in = st.number_input("Design bending moment $M_u$ (kN·m)", value=st.session_state.Mu_in, step=5.0, min_value=0.0)
        st.session_state.Vu_in = st.number_input("Design shear $V_u$ (kN)", value=st.session_state.Vu_in, step=5.0, min_value=0.0)
    with colY:
        st.session_state.Tu_in = st.number_input("Design torsion $T_u$ (kN·m)", value=st.session_state.Tu_in, step=1.0, min_value=0.0)
        st.session_state.Nu_in = st.number_input("Design axial $N_u$ (kN) (+compression)", value=st.session_state.Nu_in, step=5.0)


# ---------- CORE DESIGN CALCULATIONS (Before Results) ----------

mat, sec = Materials(st.session_state.fck, st.session_state.fy), Section(st.session_state.b, st.session_state.D, st.session_state.cover)
d = sec.d_eff
L = st.session_state.span
d_prime_approx = sec.cover + 8.0 + 0.5 * st.session_state.c_bar_d_default 

self_wt = mat.density * (st.session_state.b/1000.0) * (st.session_state.D/1000.0)
wall_kNpm = mat.density * (st.session_state.wall_thk/1000.0) * st.session_state.wall_h if st.session_state.use_wall and st.session_state.wall_thk>0 and st.session_state.wall_h>0 else 0.0
w_DL = self_wt + st.session_state.finishes + wall_kNpm
w_LL = st.session_state.ll

if st.session_state.action_mode == "Direct design actions":
    Mu_kNm = float(abs(st.session_state.Mu_in))
    Vu_kN  = float(abs(st.session_state.Vu_in))
    Tu_kNm = float(abs(st.session_state.Tu_in))
    Nu_kN  = float(st.session_state.Nu_in)
    w_ULS_15 = 0.0 
    w_service = w_DL + w_LL
else:
    w_ULS_15 = 1.5 * (w_DL + w_LL) 
    w_service = w_DL + w_LL
    
    if st.session_state.support=="Simply Supported": kM, kV = 1/8, 0.5
    elif st.session_state.support=="Cantilever": kM, kV = 1/2, 1.0
    else: kM, kV = 1/12, 0.6 

    Mu_kNm = kM * w_ULS_15 * (L**2)
    Vu_kN  = kV * w_ULS_15 * L

    if st.session_state.include_eq and st.session_state.eq_coeff > 0:
        w_ULS_12 = 1.2 * (w_DL + w_LL)
        Mu_kNm += st.session_state.eq_coeff * w_ULS_12 * (L**2) 
    
    Tu_kNm, Nu_kN = 0.0, 0.0

Mu_lim = mu_lim_rect(mat.fck, sec.b, d, mat.fy)
Ast_min = (0.85 * st.session_state.b * d) / st.session_state.fy 
Ast_max = 0.04 * st.session_state.b * st.session_state.D 

if Mu_kNm <= Mu_lim:
    Ast_req = ast_singly(Mu_kNm, mat.fy, d)
    Asc_req = 0.0
    moment_type = "Singly"
else:
    Ast_req, Asc_req, Ast_lim_req = ast_asc_doubly(Mu_kNm, Mu_lim, mat.fck, mat.fy, d, d_prime_approx, sec.b)
    moment_type = "Doubly"

Vu_eff_kN = Vu_kN + (1.6 * Tu_kNm * 1000.0 / sec.b) if Tu_kNm > 0 else Vu_kN


# ---------- RESULTS AND NARRATIVE REPORT ----------

st.header("2. Design Calculations and Checks")
st.markdown("---")

st.subheader("2.1 Load and Factored Action Summary (IS 456: Cl. 18.2)")
st.write(f"**Section**: $b={int(st.session_state.b)}$ mm, $D={int(st.session_state.D)}$ mm, $d$ (effective depth) $= **{d:.0f}**$ mm. $d'$ (comp. cover) $\\approx **{d_prime_approx:.0f}**$ mm.")
st.write(f"**Factored UDL ($w_u$)**: **{w_ULS_15:.2f}** kN/m.")
st.write(f"**Design Moment ($M_u$)**: **{Mu_kNm:.1f} kN·m**. **Design Shear ($V_u$)**: **{Vu_kN:.1f} kN**.")
st.write(f"**Ultimate Moment Capacity ($M_{{u,lim}}$)**: **{Mu_lim:.1f}** kN·m.")

if Mu_kNm > Mu_lim:
    st.error(f"❌ $M_u$ ({Mu_kNm:.1f} kN·m) > $M_{{u,lim}}$ ({Mu_lim:.1f} kN·m). **DOUBLY REINFORCED SECTION REQUIRED.**")
else:
    st.success("✅ Section size is adequate for flexure ($M_u \leq M_{{u,lim}}$).")

st.subheader("2.2 Provided Flexural Reinforcement (Tension) $\leftarrow$ Customize Here")

st.markdown(f"**Required Tension Steel ($A_{{st, req}}$)**: **{Ast_req:.0f}** mm².")
st.markdown(f"**Minimum $A_{{st}}$ (Cl 26.5.1.1)**: {Ast_min:.0f} mm². **Maximum $A_{{st}}$ (Cl 26.5.1.2)**: {Ast_max:.0f} mm².")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.session_state.t_bar_d = st.selectbox("Bottom Bar $\phi$ (mm)", [12, 16, 20, 25, 28, 32], index=[12, 16, 20, 25, 28, 32].index(st.session_state.t_bar_d))
with col_t2:
    st.session_state.t_bar_count = st.number_input("Bottom Bar Count (Total)", value=max(2, math.ceil(Ast_req / (math.pi*st.session_state.t_bar_d**2/4.0))) if not st.session_state.action_mode else st.session_state.t_bar_count, min_value=2, step=1)

Ast_prov = (math.pi*(st.session_state.t_bar_d**2)/4.0) * st.session_state.t_bar_count 

st.write(f"**Provided Tension Steel ($A_{{st, prov}}$)**: **{Ast_prov:.0f}** mm² (from {st.session_state.t_bar_count} $\\times \phi{st.session_state.t_bar_d}$ bars).")

if Ast_prov < Ast_req:
    st.error("❌ $A_{{st, prov}} < A_{{st, req}}$. **INCREASE BOTTOM REINFORCEMENT.**")
elif Ast_prov < Ast_min:
    st.warning("⚠️ $A_{{st, prov}} < A_{{st, min}}$. Provide minimum reinforcement to control cracking.")
elif Ast_prov > Ast_max:
    st.error("❌ $A_{{st, prov}} > A_{{st, max}}$. **SECTION IS OVER-REINFORCED**.")
else:
    st.success("✅ Provided Tension Reinforcement check passed.")
    
# --- Compression Reinforcement Check ---
if moment_type == "Doubly":
    st.subheader("2.2.5 Provided Flexural Reinforcement (Compression) $\leftarrow$ Customize Here")

    st.markdown(f"**Required Compression Steel ($A_{{sc, req}}$)**: **{Asc_req:.0f}** mm².")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.session_state.c_bar_d = st.selectbox("Top Bar $\phi$ (mm)", [12, 16, 20, 25], index=[12, 16, 20, 25].index(st.session_state.c_bar_d))
    with col_c2:
        st.session_state.c_bar_count = st.number_input("Top Bar Count (Total)", value=max(2, math.ceil(Asc_req / (math.pi*st.session_state.c_bar_d**2/4.0))) if not st.session_state.action_mode else st.session_state.c_bar_count, min_value=2, step=1)

    Asc_prov = (math.pi*(st.session_state.c_bar_d**2)/4.0) * st.session_state.c_bar_count 
    st.write(f"**Provided Compression Steel ($A_{{sc, prov}}$)**: **{Asc_prov:.0f}** mm² (from {st.session_state.c_bar_count} $\\times \phi{st.session_state.c_bar_d}$ bars).")
    
    if Asc_prov < Asc_req:
        st.error("❌ $A_{{sc, prov}} < A_{{sc, req}}$. **INCREASE TOP REINFORCEMENT.**")
    elif Asc_prov > Ast_max:
        st.error("❌ $A_{{sc, prov}} > A_{{st, max}}$. Compression steel limit violated.")
    else:
        st.success("✅ Provided Compression Reinforcement check passed.")
else:
    st.session_state.c_bar_d = st.session_state.c_bar_d_default
    st.session_state.c_bar_count = 2 # Minimum two top bars for stirrups
    Asc_prov = (math.pi*(st.session_state.c_bar_d**2)/4.0) * st.session_state.c_bar_count 

st.subheader("2.3 Shear Design (IS 456: Cl. 40 & 41)")

shear_res = shear_design(Vu_eff_kN, sec.b, d, mat.fck, mat.fy, Ast_prov)

st.write(f"**Equivalent Shear ($V_{{e}}$)**: $V_u + 1.6 T_u/b = **{Vu_eff_kN:.1f}**$ kN.")
st.write(f"**Nominal Shear Stress ($\tau_v$)**: $V_e / (b d) = **{shear_res['tau_v']:.3f}**$ N/mm².")
st.write(f"**Concrete Shear Strength ($\tau_c$)**: **{shear_res['tau_c']:.3f}** N/mm².")

if shear_res["tau_v"] > shear_res["tau_c_max"]:
    st.error("❌ **SECTION FAILURE**: $\\tau_v > \\tau_{{c,max}}$. Increase section size or $f_{{ck}}$.")
else:
    Vus_req_kN = max(0.0, shear_res['Vus_kN'])
    if Vus_req_kN == 0.0:
        st.success("✅ Concrete carries all shear. Provide minimum shear reinforcement.")
    else:
        st.warning(f"⚠️ Shear links required. $V_{{us,req}} = {Vus_req_kN:.1f}$ kN.")


st.subheader("2.4 Provided Shear Reinforcement (Stirrups) $\leftarrow$ Customize Here")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.session_state.stirrup_d = st.selectbox("Stirrup $\phi$ (mm)", [6, 8, 10, 12], index=[6, 8, 10, 12].index(st.session_state.stirrup_d))
with col_s2:
    st.session_state.stirrup_legs = st.number_input("Stirrup Legs (n)", value=st.session_state.stirrup_legs, min_value=2, max_value=4, step=2)
with col_s3:
    Asv_prov = st.session_state.stirrup_legs * math.pi * (st.session_state.stirrup_d**2) / 4.0
    
    s_v_req_calc = (0.87 * st.session_state.fy * Asv_prov * d) / (Vus_req_kN * 1e3) if Vus_req_kN > 0 else float('inf')
    s_v_min_calc = (0.87 * st.session_state.fy * Asv_prov) / (0.4 * st.session_state.b)
    s_v_max_abs = min(0.75 * d, 300.0)

    control_spacing = min(s_v_req_calc, s_v_min_calc, s_v_max_abs)
    
    default_spacing = int(clamp(control_spacing, 50, s_v_max_abs))
    st.session_state.s_v_prov = st.number_input("Provided Spacing $s_v$ (mm)", value=default_spacing if not st.session_state.action_mode else st.session_state.s_v_prov, min_value=50, step=10)

Vus_prov_kN = (0.87 * st.session_state.fy * Asv_prov * d) / (st.session_state.s_v_prov * 1e3) 

if Vus_prov_kN >= Vus_req_kN and st.session_state.s_v_prov <= s_v_max_abs:
    st.success(f"✅ Shear Reinforcement passed: $\\phi{st.session_state.stirrup_d}$ mm {st.session_state.stirrup_legs}-leg @ **{st.session_state.s_v_prov} mm** c/c.")
    st.write(f"Capacity $V_{{us,prov}}$ = **{Vus_prov_kN:.1f} kN** (Required {Vus_req_kN:.1f} kN).")
    st.write(f"Maximum Allowed Spacing (Cl 26.5.1.5): **{s_v_max_abs:.0f} mm**.")
else:
    st.error("❌ Shear Reinforcement failed: Spacing is too large or stirrup capacity is insufficient.")
    st.write(f"Max Allowed Spacing for design: **{control_spacing:.0f} mm** (Controls).")
    st.write(f"Absolute Max Spacing: **{s_v_max_abs:.0f} mm**.")


# Serviceability (L/d) and Development Length
base_Ld = LD_LIMITS[st.session_state.support]
p_t_service = 100.0 * Ast_prov / (sec.b * d)
mod = clamp(1.0 + 0.15 * (p_t_service - 1.0), 0.8, 1.3) 
allowable_L_over_d = base_Ld * mod
actual_L_over_d = (L * 1000.0) / d

Ld_tension = ld_required(mat.fck, mat.fy, st.session_state.t_bar_d_default, tension=True)
Ld_comp = ld_required(mat.fck, mat.fy, st.session_state.c_bar_d_default, tension=False) 

st.subheader("2.5 Serviceability Check (Deflection Control - L/d)")
st.write(f"**Allowable $L/d$**: **{allowable_L_over_d:.1f}**. **Actual $L/d$**: **{actual_L_over_d:.1f}**.")

if actual_L_over_d <= allowable_L_over_d:
    st.success("✅ **Deflection Control Passed**.")
else:
    st.error("❌ **L/d Check Failed**. Increase depth $D$ or provide compression steel.")

st.subheader("2.6 Development Length ($L_d$) and Ductile Detailing (IS 13920)")
st.write(f"**Required Development Length $L_d$ (Tension)**: **{Ld_tension:.0f}** mm (for $\\phi={st.session_state.t_bar_d_default}$ mm).")
st.write(f"**Required Development Length $L_d$ (Compression)**: **{Ld_comp:.0f}** mm (for $\\phi={st.session_state.c_bar_d_default}$ mm).")

if st.session_state.ductile:
    hinge_len_mm = max(2*d, 600)
    phi_main = max(12, st.session_state.t_bar_d) 
    max_hoop = min(0.25*d, 8*phi_main, 100)

    st.warning("⚠️ **IS 13920 Advisory Checks**")
    st.write(f"**Confinement Length** at each end $\\geq 2d$ or $600\\text{{ mm}} = **{hinge_len_mm:.0f}**$ mm.")
    st.write(f"**Hoop Spacing in Hinge Zone** $\\leq \min(0.25d, 8\\phi_{{main}}, 100) = **{max_hoop:.0f}**$ mm.")
    st.caption("Detailed design shear based on probable moments and joint checks must be verified separately.")


# ---------- VISUALIZATIONS AND EXPORT ----------

st.header("3. Beam Diagrams and Drawings")
st.markdown("---")

st.subheader("3.1 Factored Bending Moment, Shear Force, and Deflection Diagrams")

xs = np.linspace(0, L, 50)
if st.session_state.action_mode == "Derive from loads":
    if st.session_state.support=="Simply Supported": 
        M = [-1 * w_ULS_15 * x * (L - x) / 2 for x in xs] 
        V = [w_ULS_15 * (L / 2 - x) for x in xs]
    elif st.session_state.support=="Cantilever":
        M = [0.5 * w_ULS_15 * (L - x)**2 for x in xs] 
        V = [w_ULS_15 * (L - x) for x in xs]
    else: 
        M = [-1 * w_ULS_15 * x * (L - x) / 8 for x in xs] 
        V = [w_ULS_15 * (L / 2 - x) for x in xs]
else:
    M = [-Mu_kNm * np.sin(np.pi * x / L) for x in xs] 
    V = [Vu_kN * np.cos(np.pi * x / L) for x in xs] 


dfM = pd.DataFrame({"x": xs, "M (kN·m)": M}).set_index("x")
dfV = pd.DataFrame({"x": xs, "V (kN)": V}).set_index("x")

fig_M = px.line(dfM, y="M (kN·m)", title="Factored Bending Moment Diagram (Sagging Downward)")
fig_M.update_traces(fill='tozeroy', line_color='rgb(30, 144, 255)')
fig_M.update_yaxes(autorange='reversed') 
st.plotly_chart(fig_M, use_container_width=True)

fig_V = px.line(dfV, y="V (kN)", title="Factored Shear Force Diagram (kN)")
fig_V.update_traces(line_color='rgb(255, 69, 0)')
st.plotly_chart(fig_V, use_container_width=True)

Ec = 5000 * math.sqrt(st.session_state.fck)
Igross = (st.session_state.b * st.session_state.D**3) / 12 

deflection = [0] * len(xs)
if st.session_state.support=="Simply Supported":
    w_service_Nmm = w_service * 1000.0 / 1000.0 
    L_mm = L * 1000.0
    for i, x in enumerate(xs):
        x_mm = x * 1000.0
        deflection[i] = -(w_service_Nmm * x_mm * (L_mm**3 - 2*L_mm*x_mm**2 + x_mm**3)) / (24 * Ec * Igross)
elif st.session_state.support=="Cantilever":
    w_service_Nmm = w_service * 1000.0 / 1000.0 
    L_mm = L * 1000.0
    for i, x in enumerate(xs):
        x_mm = x * 1000.0
        deflection[i] = (w_service_Nmm * x_mm**2 * (6*L_mm**2 - 4*L_mm*x_mm + x_mm**2)) / (24 * Ec * Igross)

dfD = pd.DataFrame({"x": xs, "Deflection (mm)": deflection}).set_index("x")

fig_D = px.line(dfD, y="Deflection (mm)", title="Approximate Deflection Profile (mm)")
fig_D.update_traces(line_color='rgb(0, 180, 0)')
fig_D.update_yaxes(title_text="Deflection (mm) [Negative is Up, Positive is Down]")
st.plotly_chart(fig_D, use_container_width=True)

st.subheader("3.2 Rebar Layout and Curtailment for Drawing $\leftarrow$ Customize Here")
st.caption("Use this table **only** to define the visual layout and curtailment. Lengths are in **meters**.")

# Initialize or re-sync the drawing table with the current design inputs
if st.session_state.rebar_df_draw is None or len(st.session_state.rebar_df_draw) == 0:
    st.session_state.rebar_df_draw = pd.DataFrame([
        {"position":"bottom","dia_mm":int(st.session_state.t_bar_d),"count":2,"start_m":0.0,"end_m":L},
        {"position":"bottom","dia_mm":int(st.session_state.t_bar_d),"count":max(0, st.session_state.t_bar_count-2),"start_m":L/8,"end_m":L-L/8},
        {"position":"top","dia_mm":int(st.session_state.c_bar_d),"count":st.session_state.c_bar_count,"start_m":0.0,"end_m":L},
    ])

rebar_df_edited = st.data_editor(st.session_state.rebar_df_draw, num_rows="dynamic")
st.session_state.rebar_df_draw = rebar_df_edited

st.subheader("3.3 Cross and Longitudinal Drawings")

def draw_cross_section_plotly(b_mm, D_mm, cover_mm, df_rebar):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=b_mm, y1=D_mm, line=dict(color="black", width=2), fillcolor="rgba(192, 192, 192, 0.5)")
    cc = float(cover_mm)
    fig.add_shape(type="rect", x0=cc, y0=cc, x1=b_mm-cc, y1=D_mm-cc, line=dict(color="black", width=1), fillcolor="rgba(0,0,0,0)")

    bottom_bars = df_rebar[df_rebar["position"].str.lower()=="bottom"].groupby("dia_mm")["count"].sum().reset_index()
    top_bars    = df_rebar[df_rebar["position"].str.lower()=="top"].groupby("dia_mm")["count"].sum().reset_index()

    def place_bars(rowset, y_ref, is_bottom):
        bars = []
        for _,r in rowset.iterrows():
            bars += [int(r["dia_mm"]) for _ in range(int(r["count"]))]
        bars.sort(reverse=True)
        if not bars: return
        n = len(bars)
        
        if n == 1: xs = [b_mm / 2.0]
        else:
            span_avail = b_mm - 2*cc
            step = span_avail / (n - 1)
            xs = [cc + i * step for i in range(n)]

        for i, phi in enumerate(bars):
            r = phi / 2.0
            y = y_ref + r if is_bottom else y_ref - r
            fig.add_shape(type="circle", x0=xs[i]-r, y0=y-r, x1=xs[i]+r, y1=y+r,
                          line=dict(color="blue", width=1), fillcolor="blue")
    
    y_ref_bottom = cc
    y_ref_top = D_mm - cc
    place_bars(bottom_bars, y_ref_bottom, is_bottom=True)
    place_bars(top_bars, y_ref_top, is_bottom=False)

    fig.update_layout(title=f"Cross Section $b={int(st.session_state.b)}$mm x $D={int(st.session_state.D)}$mm", height=400, width=500)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)

def draw_longitudinal_section_plotly(L_m, D_mm, cover_mm, df_rebar):
    fig = go.Figure()
    L_mm = L_m * 1000
    
    fig.add_shape(type="rect", x0=0, y0=0, x1=L_mm, y1=D_mm, line=dict(color="black", width=2), fillcolor="rgba(192, 192, 192, 0.5)")
    
    for _, row in df_rebar.iterrows():
        start_x = row['start_m'] * 1000
        end_x = row['end_m'] * 1000
        dia = row['dia_mm']
        count = row['count']
        pos = row['position'].lower()
        
        y_center = cover_mm + dia / 2.0
        if pos == 'top':
            y_center = D_mm - y_center
        
        fig.add_trace(go.Scatter(
            x=[start_x, end_x], y=[y_center, y_center],
            mode='lines',
            line=dict(color='blue', width=count * 2, dash='solid'),
            name=f"{count}-Ø{dia} {pos.capitalize()}"
        ))
        
        label = f"{count}-Ø{dia} {pos.capitalize()}"
        mid_x = (start_x + end_x) / 2
        fig.add_annotation(x=mid_x, y=y_center + 20, text=label, showarrow=False, font=dict(size=10))

    fig.add_shape(type="rect", x0=-20, y0=0, x1=20, y1=-D_mm/4, fillcolor="black")
    fig.add_shape(type="rect", x0=L_mm - 20, y0=0, x1=L_mm + 20, y1=-D_mm/4, fillcolor="black")
    
    fig.update_layout(title="Longitudinal Elevation (Beam)", height=400)
    st.plotly_chart(fig, use_container_width=True)

col_d1, col_d2 = st.columns(2)
with col_d1:
    draw_cross_section_plotly(st.session_state.b, st.session_state.D, st.session_state.cover, rebar_df_edited)
with col_d2:
    draw_longitudinal_section_plotly(L, st.session_state.D, st.session_state.cover, rebar_df_edited)


st.header("4. Export Options")
st.markdown("---")

summary = {
    "span": [L], "support": [st.session_state.support], "b_mm": [st.session_state.b], "D_mm": [st.session_state.D], "d_eff_mm": [d],
    "Mu_kNm": [Mu_kNm], "Vu_kN": [Vu_kN], "Tu_kNm": [Tu_kNm], "Nu_kN": [Nu_kN],
    "Ast_req_mm2": [Ast_req], "Ast_prov_mm2": [Ast_prov],
    "Asc_req_mm2": [Asc_req], "Asc_prov_mm2": [Asc_prov if moment_type == "Doubly" else Asc_prov],
    "Shear_Stirrup": [f"{st.session_state.stirrup_legs}-leg $\\phi{st.session_state.stirrup_d}$"],
    "Shear_Spacing_mm": [st.session_state.s_v_prov],
    "Ld_tension_mm": [Ld_tension], "Ld_comp_mm": [Ld_comp],
    "Allowable_L/d": [allowable_L_over_d], "Actual_L/d": [actual_L_over_d],
}
df_summary = pd.DataFrame(summary)
buf = io.StringIO()
df_summary.to_csv(buf, index=False)

st.download_button("⬇️ Download Design Summary CSV", data=buf.getvalue(), file_name="beam_summary.csv")
