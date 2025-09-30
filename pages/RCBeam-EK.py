import math
from dataclasses import dataclass
import io
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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

# ---------- UI INPUTS (GENERAL) ----------

with st.sidebar:
    st.header("Quick Toggles")
    use_wall = st.checkbox("Add wall load", value=False)
    include_eq = st.checkbox("Include E/W coeff", value=False)
    eq_coeff = st.number_input("Eq. coeff (×wL²)", value=0.0, step=0.05, disabled=not include_eq)
    ductile = st.checkbox("Apply IS 13920 checks", value=True)
    st.caption("⚠️ Use the 'Print' function of your browser (Ctrl+P / Cmd+P) for the final report.")

st.header("1. Project Inputs")
st.markdown("---")
colA, colB, colC = st.columns(3)

with colA:
    st.subheader("Geometry")
    span = st.number_input("Clear span $L$ (m)", value=6.0, min_value=0.5, step=0.1)
    support = st.selectbox("End condition", ["Simply Supported","Continuous","Cantilever"], index=0)
    b = st.number_input("Beam width $b$ (mm)", value=300, step=10, min_value=150)
    D = st.number_input("Overall depth $D$ (mm)", value=500, step=10, min_value=200)
    cover = st.number_input("Clear cover (mm)", value=25, step=5, min_value=20)

with colB:
    st.subheader("Materials")
    fck = st.selectbox("Concrete $f_{ck}$ (N/mm²)", [20,25,30,35,40], index=2)
    fy = st.selectbox("Steel $f_y$ (N/mm²)", [415,500], index=1)
    # Default bar size for Ld check
    t_bar_d_default = st.selectbox("Tension Bar $\phi$ for $L_d$ (mm)", [12,16,20,25,28,32], index=1)
    c_bar_d_default = st.selectbox("Compression Bar $\phi$ for $L_d$ (mm)", [12,16,20,25], index=0)

with colC:
    st.subheader("Loads")
    finishes = st.number_input("Finishes (kN/m)", value=2.0, step=0.1, min_value=0.0)
    ll = st.number_input("Live load (kN/m)", value=5.0, step=0.5, min_value=0.0)
    wall_thk = st.number_input("Wall thk (mm)", value=115, step=115, min_value=0, disabled=not use_wall)
    wall_h = st.number_input("Wall height (m)", value=3.0, step=0.1, min_value=0.0, disabled=not use_wall)
    wall_density = st.number_input("Masonry density (kN/m³)", value=19.0, step=0.5, disabled=not use_wall)

st.subheader("Design Action Source")
action_mode = st.radio("Use:", ["Derive from loads", "Direct design actions"], index=0, horizontal=True)

Mu_in, Vu_in, Tu_in, Nu_in = 0.0, 0.0, 0.0, 0.0
if action_mode == "Direct design actions":
    st.info("Enter factored design actions (ULS) at the critical section.")
    colX,colY = st.columns(2)
    with colX:
        Mu_in = st.number_input("Design bending moment $M_u$ (kN·m)", value=120.0, step=5.0, min_value=0.0)
        Vu_in = st.number_input("Design shear $V_u$ (kN)", value=180.0, step=5.0, min_value=0.0)
    with colY:
        Tu_in = st.number_input("Design torsion $T_u$ (kN·m)", value=0.0, step=1.0, min_value=0.0)
        Nu_in = st.number_input("Design axial $N_u$ (kN) (+compression)", value=0.0, step=5.0)


# ---------- CORE DESIGN CALCULATIONS (Before Results) ----------

mat, sec = Materials(fck,fy), Section(b,D,cover)
d = sec.d_eff
L = span

self_wt = mat.density * (b/1000.0) * (D/1000.0)
wall_kNpm = wall_density * (wall_thk/1000.0) * wall_h if use_wall and wall_thk>0 and wall_h>0 else 0.0
w_DL = self_wt + finishes + wall_kNpm
w_LL = ll

if action_mode == "Direct design actions":
    Mu_kNm = float(abs(Mu_in))
    Vu_kN  = float(abs(Vu_in))
    Tu_kNm = float(abs(Tu_in))
    Nu_kN  = float(Nu_in)
    w_ULS_15 = 0.0 
    w_service = w_DL + w_LL
else:
    w_ULS_15 = 1.5 * (w_DL + w_LL) 
    w_service = w_DL + w_LL
    
    # Coefficients (IS 456: Table 12 & 13)
    if support=="Simply Supported": kM, kV = 1/8, 0.5
    elif support=="Cantilever": kM, kV = 1/2, 1.0
    else: kM, kV = 1/12, 0.6 

    Mu_kNm = kM * w_ULS_15 * (L**2)
    Vu_kN  = kV * w_ULS_15 * L

    if include_eq and eq_coeff > 0:
        w_ULS_12 = 1.2 * (w_DL + w_LL)
        Mu_kNm += eq_coeff * w_ULS_12 * (L**2) 
    
    Tu_kNm, Nu_kN = 0.0, 0.0

Mu_lim = mu_lim_rect(mat.fck, sec.b, d, mat.fy)
Ast_req = ast_singly(Mu_kNm, mat.fy, d)
Ast_min = (0.85 * b * d) / fy 
Ast_max = 0.04 * b * D 

Vu_eff_kN = Vu_kN + (1.6 * Tu_kNm * 1000.0 / sec.b) if Tu_kNm > 0 else Vu_kN


# ---------- RESULTS AND NARRATIVE REPORT ----------

st.header("2. Design Calculations and Checks")
st.markdown("---")

st.subheader("2.1 Load and Factored Action Summary (IS 456: Cl. 18.2)")
st.write(f"**Section**: $b={int(b)}$ mm, $D={int(D)}$ mm, $d$ (effective depth) $= **{d:.0f}**$ mm.")
st.write(f"**Factored UDL ($w_u$)**: **{w_ULS_15:.2f}** kN/m.")
st.write(f"**Design Moment ($M_u$)**: **{Mu_kNm:.1f} kN·m**. **Design Shear ($V_u$)**: **{Vu_kN:.1f} kN**.")
st.write(f"**Ultimate Moment Capacity ($M_{{u,lim}}$)**: **{Mu_lim:.1f}** kN·m.")

if Mu_kNm > Mu_lim:
    st.error("❌ $M_u$ > $M_{{u,lim}}$. **SECTION MUST BE INCREASED IN SIZE.**")
else:
    st.success("✅ Section size is adequate for flexure ($M_u \leq M_{{u,lim}}$).")
    st.markdown(f"**Required Tension Steel ($A_{{st, req}}$)**: **{Ast_req:.0f}** mm².")
    st.markdown(f"**Minimum $A_{{st}}$ (Cl 26.5.1.1)**: {Ast_min:.0f} mm². **Maximum $A_{{st}}$ (Cl 26.5.1.2)**: {Ast_max:.0f} mm².")

st.subheader("2.2 Provided Flexural Reinforcement (Tension) $\leftarrow$ Customize Here")

col_t1, col_t2 = st.columns(2)
with col_t1:
    t_bar_d = st.selectbox("Bottom Bar $\phi$ (mm)", [12, 16, 20, 25, 28, 32], index=2 if Ast_req > 1000 else 1)
with col_t2:
    t_bar_count = st.number_input("Bottom Bar Count (Total)", value=math.ceil(Ast_req / (math.pi*t_bar_d**2/4.0)), min_value=2, step=1)

Ast_prov = (math.pi*(t_bar_d**2)/4.0) * t_bar_count 

st.write(f"**Provided Tension Steel ($A_{{st, prov}}$)**: **{Ast_prov:.0f}** mm² (from {t_bar_count} $\\times \phi{t_bar_d}$ bars).")

if Ast_prov < Ast_req:
    st.error("❌ $A_{{st, prov}} < A_{{st, req}}$. **INCREASE BOTTOM REINFORCEMENT.**")
elif Ast_prov < Ast_min:
    st.warning("⚠️ $A_{{st, prov}} < A_{{st, min}}$. Provide minimum reinforcement to control cracking.")
elif Ast_prov > Ast_max:
    st.error("❌ $A_{{st, prov}} > A_{{st, max}}$. **SECTION IS OVER-REINFORCED**.")
else:
    st.success("✅ Provided Flexural Reinforcement check passed.")


st.subheader("2.3 Shear Design (IS 456: Cl. 40 & 41)")

shear_res = shear_design(Vu_eff_kN, sec.b, d, mat.fck, mat.fy, Ast_prov)

st.write(f"**Equivalent Shear ($V_{{e}}$)**: $V_u + 1.6 T_u/b = **{Vu_eff_kN:.1f}**$ kN.")
st.write(f"**Nominal Shear Stress ($\tau_v$)**: $V_e / (b d) = **{shear_res['tau_v']:.3f}**$ N/mm².")
st.write(f"**Concrete Shear Strength ($\tau_c$)**: **{shear_res['tau_c']:.3f}** N/mm².")

if shear_res["tau_v"] > shear_res["tau_c_max"]:
    st.error("❌ **SECTION FAILURE**: $\\tau_v > \\tau_{{c,max}}$. Increase section size or $f_{{ck}}$.")
else:
    if shear_res["min_shear_needed"]:
        Vus_req_kN = 0.0
        st.success("✅ Concrete carries all shear. Provide minimum shear reinforcement.")
    else:
        Vus_req_kN = shear_res['Vus_kN']
        st.warning(f"⚠️ Shear links required. $V_{{us,req}} = {Vus_req_kN:.1f}$ kN.")


st.subheader("2.4 Provided Shear Reinforcement (Stirrups) $\leftarrow$ Customize Here")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    stirrup_d = st.selectbox("Stirrup $\phi$ (mm)", [6, 8, 10, 12], index=1)
with col_s2:
    stirrup_legs = st.number_input("Stirrup Legs (n)", value=2, min_value=2, max_value=4, step=2)
with col_s3:
    Asv_prov = stirrup_legs * math.pi * (stirrup_d**2) / 4.0
    s_v_req_calc = (0.87 * fy * Asv_prov * d) / (Vus_req_kN * 1e3) if Vus_req_kN > 0 else float('inf')
    s_v_min_calc = (0.87 * fy * Asv_prov) / (0.4 * b)
    s_v_min_control = min(shear_res['s_v_max_limit'], s_v_min_calc)

    default_spacing = int(min(s_v_req_calc, s_v_min_control))
    s_v_prov = st.number_input("Provided Spacing $s_v$ (mm)", value=clamp(default_spacing, 50, 300), min_value=50, step=10)

Vus_prov_kN = (0.87 * fy * Asv_prov * d) / (s_v_prov * 1e3) 

if Vus_prov_kN >= Vus_req_kN and s_v_prov <= shear_res['s_v_max_limit'] and s_v_prov <= s_v_min_calc:
    st.success(f"✅ Shear Reinforcement passed: $\\phi{stirrup_d}$ mm {stirrup_legs}-leg @ **{s_v_prov} mm** c/c.")
    st.write(f"Capacity $V_{{us,prov}}$ = **{Vus_prov_kN:.1f} kN** (Required {Vus_req_kN:.1f} kN).")
else:
    st.error("❌ Shear Reinforcement failed: Spacing is too large or stirrup capacity is insufficient.")
    st.write(f"Max Allowed Spacing: **{min(s_v_req_calc, shear_res['s_v_max_limit']):.0f} mm** (Controls).")
    st.write(f"Max Spacing (Min Steel): **{s_v_min_calc:.0f} mm**.")

# Serviceability (L/d) and Development Length
base_Ld = LD_LIMITS[support]
p_t_service = 100.0 * Ast_prov / (sec.b * d)
mod = clamp(1.0 + 0.15 * (p_t_service - 1.0), 0.8, 1.3) 
allowable_L_over_d = base_Ld * mod
actual_L_over_d = (L * 1000.0) / d

Ld_tension = ld_required(mat.fck, mat.fy, t_bar_d_default, tension=True)
Ld_comp = ld_required(mat.fck, mat.fy, c_bar_d_default, tension=False) 

st.subheader("2.5 Serviceability Check (Deflection Control - L/d)")
st.write(f"**Allowable $L/d$**: **{allowable_L_over_d:.1f}**. **Actual $L/d$**: **{actual_L_over_d:.1f}**.")

if actual_L_over_d <= allowable_L_over_d:
    st.success("✅ **Deflection Control Passed**.")
else:
    st.error("❌ **L/d Check Failed**. Increase depth $D$ or provide compression steel.")

st.subheader("2.6 Development Length ($L_d$) and Ductile Detailing (IS 13920)")
st.write(f"**Required Development Length $L_d$ (Tension)**: **{Ld_tension:.0f}** mm.")


# ---------- VISUALIZATIONS AND EXPORT ----------

st.header("3. Beam Diagrams and Drawings")
st.markdown("---")

st.subheader("3.1 Factored Bending Moment and Shear Force Diagrams")

# M and V calculations for plotting
xs = np.linspace(0, L, 50)
if action_mode == "Derive from loads":
    if support=="Simply Supported": 
        M = [-1 * w_ULS_15 * x * (L - x) / 2 for x in xs] # Negative for plotting downwards (sagging)
        V = [w_ULS_15 * (L / 2 - x) for x in xs]
    elif support=="Cantilever":
        M = [0.5 * w_ULS_15 * (L - x)**2 for x in xs] # Positive for plotting upwards (hogging)
        V = [w_ULS_15 * (L - x) for x in xs]
    else: # Continuous: assume hogging at supports and sagging at midspan
        M = [-1 * w_ULS_15 * x * (L - x) / 8 for x in xs] # Simplified parabolic 
        V = [w_ULS_15 * (L / 2 - x) for x in xs]
else:
    # Use approximate shapes for visualization if direct input is used
    M = [-Mu_kNm * np.sin(np.pi * x / L) for x in xs] # Sagging always plotted down
    V = [Vu_kN * np.cos(np.pi * x / L) for x in xs] 


dfM = pd.DataFrame({"x": xs, "M (kN·m)": M}).set_index("x")
dfV = pd.DataFrame({"x": xs, "V (kN)": V}).set_index("x")

# Bending Moment Diagram (Inverted)
fig_M = px.line(dfM, y="M (kN·m)", title="Factored Bending Moment Diagram (Sagging Downward)")
fig_M.update_traces(fill='tozeroy', line_color='rgb(30, 144, 255)')
fig_M.update_yaxes(autorange='reversed') # Ensures sagging (negative y-value) is plotted downward
st.plotly_chart(fig_M, use_container_width=True)

# Shear Force Diagram
fig_V = px.line(dfV, y="V (kN)", title="Factored Shear Force Diagram (kN)")
fig_V.update_traces(line_color='rgb(255, 69, 0)')
st.plotly_chart(fig_V, use_container_width=True)

# Deflection Diagram
Ec = 5000 * math.sqrt(fck) # N/mm2
Igross = (b * D**3) / 12 # mm4
EIkNmm2 = (Ec * 1e-3) * (Igross * 1e-6) # kN.m2 (Placeholder for approximate I)

if support=="Simply Supported":
    k_def = 5.0 / 384.0
    # Deflection equation for simply supported beam under UDL (sign convention for plotting downwards)
    deflection = [k_def * w_service * L**4 * (24*(x/L)**4 - 40*(x/L)**3 + 16*(x/L)) * 10**6 / (Ec * Igross) for x in xs] # in mm
elif support=="Cantilever":
    k_def = 1.0 / 8.0
    deflection = [-k_def * w_service * L**4 * (x/L)**2 * 10**6 / (Ec * Igross) for x in xs] # in mm (negative sign for downward)
else: # Continuous - use SS for simple approximation
    k_def = 1.0 / 384.0
    deflection = [k_def * w_service * L**4 * (24*(x/L)**4 - 40*(x/L)**3 + 16*(x/L)) * 10**6 / (Ec * Igross) for x in xs]

dfD = pd.DataFrame({"x": xs, "Deflection (mm)": deflection}).set_index("x")

fig_D = px.line(dfD, y="Deflection (mm)", title="Approximate Deflection Profile (mm)")
fig_D.update_traces(line_color='rgb(0, 180, 0)')
fig_D.update_yaxes(title_text="Deflection (mm) [Negative is Down]")
st.plotly_chart(fig_D, use_container_width=True)

st.subheader("3.2 Rebar Layout and Curtailment for Drawing $\leftarrow$ Customize Here")
st.caption("Use this table **only** to define the visual layout and curtailment. $A_{st, prov}$ for design is from Section 2.2.")

# Simplified Rebar Table for Drawing (Initialised based on Section 2.2 input)
if "rebar_df_draw" not in st.session_state:
    st.session_state.rebar_df_draw = pd.DataFrame([
        # Main Bottom Steel (Continuous)
        {"position":"bottom","dia_mm":int(t_bar_d_default),"count":2,"start_m":0.0,"end_m":span},
        # Additional Bottom Steel (Curtailed)
        {"position":"bottom","dia_mm":int(t_bar_d_default),"count":t_bar_count-2,"start_m":L/8,"end_m":span-L/8},
        # Top Steel (Continuous)
        {"position":"top","dia_mm":int(c_bar_d_default),"count":2,"start_m":0.0,"end_m":span},
    ])
rebar_df_edited = st.data_editor(st.session_state.rebar_df_draw, num_rows="dynamic")
st.session_state.rebar_df_draw = rebar_df_edited

st.subheader("3.3 Cross and Longitudinal Drawings")

def draw_cross_section_plotly(b_mm, D_mm, cover_mm, df_rebar):
    # This function uses the current state of the drawing table to place circles
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

    fig.update_layout(title=f"Cross Section $b={int(b)}$mm x $D={int(D)}$mm", height=400, width=500)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)

def draw_longitudinal_section_plotly(L_m, D_mm, cover_mm, df_rebar):
    # This function respects the start_m and end_m for curtailment
    fig = go.Figure()
    L_mm = L_m * 1000
    
    fig.add_shape(type="rect", x0=0, y0=0, x1=L_mm, y1=D_mm, line=dict(color="black", width=2), fillcolor="rgba(192, 192, 192, 0.5)")
    
    for i, row in df_rebar.iterrows():
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
    draw_cross_section_plotly(b, D, cover, rebar_df_edited)
with col_d2:
    draw_longitudinal_section_plotly(L, D, cover, rebar_df_edited)


st.header("4. Export Options")
st.markdown("---")

summary = {
    "span": [L], "support": [support], "b_mm": [b], "D_mm": [D], "d_eff_mm": [d],
    "Mu_kNm": [Mu_kNm], "Vu_kN": [Vu_kN], "Tu_kNm": [Tu_kNm], "Nu_kN": [Nu_kN],
    "Ast_req_mm2": [Ast_req], "Ast_prov_mm2": [Ast_prov],
    "Shear_Stirrup": [f"{stirrup_legs}-leg $\\phi{stirrup_d}$"],
    "Shear_Spacing_mm": [s_v_prov],
    "Ld_tension_mm": [Ld_tension], "Ld_comp_mm": [Ld_comp],
    "Allowable_L/d": [allowable_L_over_d], "Actual_L/d": [actual_L_over_d],
}
df_summary = pd.DataFrame(summary)
buf = io.StringIO()
df_summary.to_csv(buf, index=False)

st.download_button("⬇️ Download Design Summary CSV", data=buf.getvalue(), file_name="beam_summary.csv")
