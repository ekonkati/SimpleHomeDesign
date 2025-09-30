import math
from dataclasses import dataclass
import io
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Ensure ezdxf is not required for execution
ezdxf = None

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
        # Assuming a default of 8mm link bar and 16mm main bar for d_eff calculation
        return max(self.D - (self.cover + 8 + 0.5*16), 0.0)

# τ_bd table (design bond stress) – plain bars per IS456 Table 26; deformed +60%
TAU_BD_PLAIN = {20:1.2, 25:1.4, 30:1.5, 35:1.7, 40:1.9, 45:2.0, 50:2.2}

# τ_c(max) approximate per IS456 Table 20 (concrete shear strength max), N/mm2
TAU_C_MAX = {20:2.8, 25:3.1, 30:3.5, 35:3.7, 40:4.0}

# τ_c interpolation tables for p_t (partial from source)
TC_TABLE = {
    20: [(0.25,0.28),(0.50,0.38),(0.75,0.46),(1.00,0.62)],
    25: [(0.25,0.29),(0.50,0.40),(0.75,0.48),(1.00,0.62)],
    30: [(0.25,0.29),(0.50,0.40),(0.75,0.48),(1.00,0.62)],
    35: [(0.25,0.30),(0.50,0.42),(0.75,0.50),(1.00,0.62)],
    40: [(0.25,0.31),(0.50,0.44),(0.75,0.52),(1.00,0.62)],
}

LD_LIMITS = {"Simply Supported":20.0, "Continuous":26.0, "Cantilever":7.0}

def interp_xy(table, x):
    # Function to interpolate values
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
    # Calculates design shear strength of concrete τ_c
    fck_key = min(TC_TABLE.keys(), key=lambda k: abs(k - fck))
    return interp_xy(TC_TABLE[fck_key], clamp(p_t, 0.25, 1.0))

def mu_lim_rect(fck, b_mm, d_mm, fy):
    # Calculates ultimate moment capacity (Mu_lim) for balanced section (singly reinforced)
    k = 0.138 if fy <= 415 else 0.133 # xu/d = 0.48 for Fe415, 0.46 for Fe500
    MuNmm = k * fck * b_mm * (d_mm**2)
    return MuNmm / 1e6 # Return in kN·m

def ast_singly(Mu_kNm, fy, d_mm, jd_ratio=0.9):
    # Calculates required steel area using simplified lever arm (jd=0.9d)
    Mu_Nmm = Mu_kNm * 1e6
    jd = jd_ratio * d_mm
    return Mu_Nmm / (0.87 * fy * jd)

def ld_required(fck, fy, bar_dia_mm, deformed=True, tension=True):
    # Calculates development length Ld (IS 456: Cl. 26.2)
    fck_key = min(TAU_BD_PLAIN.keys(), key=lambda k: abs(k - fck))
    tau_bd = TAU_BD_PLAIN[fck_key]
    if deformed: tau_bd *= 1.6
    if not tension: tau_bd *= 1.25 # Compression Ld is 1.25 times tension Ld
    Ld = (bar_dia_mm * fy) / (4.0 * tau_bd)
    return Ld

# ---------- UI INPUTS (Refactored to single column for printing) ----------

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
    t_bar = st.selectbox("Tension bar dia default (mm)", [12,16,20,25,28,32], index=1)
    c_bar = st.selectbox("Compression bar dia default (mm)", [12,16,20,25], index=0)

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
    st.info("Enter factored design actions (ULS) at the critical section. Signs are ignored for moment/shear/torsion.")
    colX,colY = st.columns(2)
    with colX:
        Mu_in = st.number_input("Design bending moment $M_u$ (kN·m)", value=120.0, step=5.0, min_value=0.0)
        Vu_in = st.number_input("Design shear $V_u$ (kN)", value=180.0, step=5.0, min_value=0.0)
    with colY:
        Tu_in = st.number_input("Design torsion $T_u$ (kN·m)", value=0.0, step=1.0, min_value=0.0)
        Nu_in = st.number_input("Design axial $N_u$ (kN) (+compression)", value=0.0, step=5.0)

st.subheader("Reinforcement Layout")
st.caption("Define bar groups below for $A_{st, prov}$ and drawing. **Length in m.**")
if "rebar_df" not in st.session_state:
    st.session_state.rebar_df = pd.DataFrame([
        {"id":"B1","position":"bottom","zone":"span","dia_mm":int(t_bar),"count":2,"continuity":"continuous","start_m":0.0,"end_m":span},
        {"id":"T1","position":"top","zone":"continuous","dia_mm":int(c_bar),"count":2,"continuity":"continuous","start_m":0.0,"end_m":span},
        {"id":"B2","position":"bottom","zone":"span","dia_mm":16,"count":2,"continuity":"curtailed","start_m":0.2,"end_m":span-0.2},
    ])
rebar_df_edited = st.data_editor(st.session_state.rebar_df, num_rows="dynamic")
st.session_state.rebar_df = rebar_df_edited

# ---------- CORE DESIGN CALCULATIONS ----------

mat, sec = Materials(fck,fy), Section(b,D,cover)
d = sec.d_eff
L = span

# Loads
self_wt = mat.density * (b/1000.0) * (D/1000.0)
wall_kNpm = wall_density * (wall_thk/1000.0) * wall_h if use_wall and wall_thk>0 and wall_h>0 else 0.0
w_DL = self_wt + finishes + wall_kNpm
w_LL = ll

# Determine Factored Actions (ULS)
if action_mode == "Direct design actions":
    Mu_kNm = float(abs(Mu_in))
    Vu_kN  = float(abs(Vu_in))
    Tu_kNm = float(abs(Tu_in))
    Nu_kN  = float(Nu_in)
    w_ULS_15 = 0.0 
else:
    w_ULS_15 = 1.5 * (w_DL + w_LL) 
    
    if support=="Simply Supported": kM, kV = 1/8, 0.5
    elif support=="Cantilever": kM, kV = 1/2, 1.0
    else: kM, kV = 1/12, 0.6 

    Mu_kNm = kM * w_ULS_15 * (L**2)
    Vu_kN  = kV * w_ULS_15 * L

    if include_eq and eq_coeff > 0:
        w_ULS_12 = 1.2 * (w_DL + w_LL)
        Mu_kNm += eq_coeff * w_ULS_12 * (L**2) 
    
    Tu_kNm = 0.0
    Nu_kN  = 0.0

def ast_from_rows(rows, pos_filter):
    sel = rows[rows["position"].str.lower()==pos_filter]
    areas = (math.pi*(sel["dia_mm"]**2)/4.0) * sel["count"]
    return float(areas.sum())

Ast_bottom_span = ast_from_rows(rebar_df_edited, "bottom")
Ast_prov = max(Ast_bottom_span, 1.0) 

# Flexure Calculations
Mu_lim = mu_lim_rect(mat.fck, sec.b, d, mat.fy)
Ast_req = ast_singly(Mu_kNm, mat.fy, d)
Ast_min = (0.85 * b * d) / fy 
Ast_max = 0.04 * b * D 

# Shear Calculations
Vu_eff_kN = Vu_kN + (1.6 * Tu_kNm * 1000.0 / sec.b) if Tu_kNm > 0 else Vu_kN

def shear_design(Vu_kN_eff, b_mm, d_mm, fck, fy, Ast_mm2):
    tau_v = (Vu_kN_eff * 1e3) / (b_mm * d_mm)
    p_t = 100.0 * Ast_mm2 / (b_mm * d_mm)
    tc = tau_c(fck, p_t)
    tc_max = TAU_C_MAX[min(TAU_C_MAX.keys(), key=lambda k: abs(k - fck))]
    
    Vus_kN = Vu_kN_eff - tc * b_mm * d_mm / 1e3
    
    phi, legs = 8.0, 2
    Asv = legs * math.pi * (phi**2) / 4.0
    s_v_req = (0.87 * fy * Asv * d_mm) / (Vus_kN * 1e3) if Vus_kN > 0 else float('inf')
    s_v_min = (0.87 * fy * Asv) / (0.4 * b_mm) 
    
    s_v_max_limit = min(0.75 * d_mm, 300.0)
    
    return {"tau_v": tau_v, "p_t": p_t, "tau_c": tc, "tau_c_max": tc_max,
            "Vus_kN": Vus_kN, "s_v_req": s_v_req, "s_v_max_limit": s_v_max_limit,
            "s_v_min_spacing": s_v_min,
            "min_shear_needed": Vus_kN <= 0,
            "stirrup_size": f"{int(phi)} mm {legs}-leg"}

shear_res = shear_design(Vu_eff_kN, sec.b, d, mat.fck, mat.fy, Ast_prov)

# Serviceability (L/d)
Ld_tension = ld_required(mat.fck, mat.fy, t_bar, tension=True)
Ld_comp = ld_required(mat.fck, mat.fy, c_bar, tension=False) 

base_Ld = LD_LIMITS[support]
p_t_service = 100.0 * Ast_prov / (sec.b * d)
mod = clamp(1.0 + 0.15 * (p_t_service - 1.0), 0.8, 1.3) 
allowable_L_over_d = base_Ld * mod
actual_L_over_d = (L * 1000.0) / d

# ---------- RESULTS AND NARRATIVE REPORT ----------

st.header("2. Design Calculations and Checks")
st.markdown("---")

st.subheader("2.1 Geometric and Material Properties")
st.write(f"**Section**: $b={int(b)}$ mm, $D={int(D)}$ mm, $d$ (effective depth) $= **{d:.0f}**$ mm.")
st.write(f"**Materials**: $f_{{ck}}={fck}$ N/mm² (M{fck}), $f_y={fy}$ N/mm² (Fe{fy}).")

st.subheader("2.2 Load and Factored Action Summary (IS 456: Cl. 18.2)")
st.write(f"**Self-Weight**: {self_wt:.2f} kN/m. **Wall Load**: {wall_kNpm:.2f} kN/m.")
st.write(f"**Total DL**: {w_DL:.2f} kN/m. **Total LL**: {w_LL:.2f} kN/m.")

if action_mode == "Derive from loads":
    st.write(f"**Factored UDL ($w_u$)**: $1.5(DL+LL) = 1.5({w_DL:.2f} + {w_LL:.2f}) = **{w_ULS_15:.2f}**$ kN/m.")
    st.write(f"**Maximum Factored Moment ($M_u$)**: **{Mu_kNm:.1f} kN·m** (Calculated using $w_u L^2/k$).")
    st.write(f"**Maximum Factored Shear ($V_u$)**: **{Vu_kN:.1f} kN**.")
else:
    st.info("Using directly entered design actions (ULS).")
    st.write(f"**Design Moment ($M_u$)**: **{Mu_kNm:.1f} kN·m**.")
    st.write(f"**Design Shear ($V_u$)**: **{Vu_kN:.1f} kN**.")
    st.write(f"**Design Torsion ($T_u$)**: **{Tu_kNm:.1f} kN·m**.")
    st.write(f"**Design Axial ($N_u$)**: **{Nu_kN:.1f} kN**.")

st.subheader("2.3 Flexural Design (IS 456: Cl. 26.5)")

st.markdown(f"**Ultimate Moment Capacity ($M_{{u,lim}}$)**: $0.133(0.138)f_{{ck}}bd^2 = **{Mu_lim:.1f}**$ kN·m.")
if Mu_kNm > Mu_lim:
    st.error("❌ $M_u$ > $M_{{u,lim}}$. **SECTION MUST BE INCREASED IN SIZE.**")
else:
    st.success("✅ Section size is adequate for flexure ($M_u \leq M_{{u,lim}}$).")
    
    st.markdown(f"**Required Tension Steel ($A_{{st, req}}$)**: $M_u / (0.87 f_y j d) = **{Ast_req:.0f}**$ mm².")
    st.markdown(f"**Provided Tension Steel ($A_{{st, prov}}$)**: **{Ast_prov:.0f}** mm² (from Rebar Layout table).")
    
    st.markdown(f"**Minimum $A_{{st}}$ (Cl 26.5.1.1)**: {Ast_min:.0f} mm². **Maximum $A_{{st}}$ (Cl 26.5.1.2)**: {Ast_max:.0f} mm².")
    
    if Ast_prov < Ast_req:
        st.error("❌ $A_{{st, prov}} < A_{{st, req}}$. **INCREASE BOTTOM REINFORCEMENT.**")
    elif Ast_prov < Ast_min:
        st.warning("⚠️ $A_{{st, prov}} < A_{{st, min}}$. Provide minimum reinforcement to control cracking.")
    elif Ast_prov > Ast_max:
        st.error("❌ $A_{{st, prov}} > A_{{st, max}}$. **SECTION IS OVER-REINFORCED** (Brittle Failure possible).")
    else:
        st.success("✅ Flexural Reinforcement check passed.")

st.subheader("2.4 Shear Design (IS 456: Cl. 40 & 41)")

st.write(f"**Equivalent Shear ($V_{{e}}$)**: $V_u + 1.6 T_u/b = **{Vu_eff_kN:.1f}**$ kN.")
st.write(f"**Nominal Shear Stress ($\tau_v$)**: $V_e / (b d) = **{shear_res['tau_v']:.3f}**$ N/mm².")
st.write(f"**Concrete Shear Strength ($\tau_c$)**: **{shear_res['tau_c']:.3f}** N/mm² (for $p_t={shear_res['p_t']:.2f}$).")
st.write(f"**Max Shear Strength ($\tau_{{c,max}}$)**: **{shear_res['tau_c_max']:.2f}** N/mm² (Cl 40.2).")

if shear_res["tau_v"] > shear_res["tau_c_max"]:
    st.error("❌ **SECTION FAILURE**: $\\tau_v > \\tau_{{c,max}}$. Increase section size or $f_{{ck}}$.")
elif shear_res["min_shear_needed"]:
    st.success("✅ Concrete carries all shear. Provide minimum shear reinforcement.")
    s_v_final = min(shear_res['s_v_max_limit'], shear_res['s_v_min_spacing']) 
    st.write(f"**Stirrup Spacing**: Provide **{shear_res['stirrup_size']} @ {s_v_final:.0f} mm c/c** (Minimum requirement, Cl 26.5.1.6).")
else:
    st.warning("⚠️ Shear links required. Concrete capacity $V_{uc} = {Vu_eff_kN - shear_res['Vus_kN']:.1f}$ kN.")
    s_v_final = clamp(shear_res['s_v_req'], 50, shear_res['s_v_max_limit'])
    st.write(f"**Required Spacing $s_v$**: **{s_v_final:.0f} mm c/c** (for {shear_res['stirrup_size']}).")
    st.write(f"**Maximum Allowed Spacing (Cl 26.5.1.5)**: **{shear_res['s_v_max_limit']:.0f} mm**.")

st.subheader("2.5 Serviceability Check (Deflection Control - L/d)")

st.write(f"**Basic $L/d$ Ratio (Cl 23.2)**: **{base_Ld:.1f}**.")
st.write(f"**Modification Factor ($k_t$)**: **{mod:.2f}** (based on tension steel $p_t$).")
st.write(f"**Allowable $L/d$**: $L/d_{{basic}} \times k_t = **{allowable_L_over_d:.1f}**$.")
st.write(f"**Actual $L/d$**: $L / d = **{actual_L_over_d:.1f}**$.")

if actual_L_over_d <= allowable_L_over_d:
    st.success("✅ **Deflection Control Passed** (Span/Depth Ratio Check).")
else:
    st.error("❌ **L/d Check Failed**. Increase depth $D$ or provide compression steel.")

st.subheader("2.6 Development Length ($L_d$) and Ductile Detailing (IS 13920)")

st.write(f"**Required Development Length $L_d$ (Tension)**: **{Ld_tension:.0f}** mm (for $\\phi={t_bar}$ mm).")
st.write(f"**Required Development Length $L_d$ (Compression)**: **{Ld_comp:.0f}** mm (for $\\phi={c_bar}$ mm).")

if ductile:
    hinge_len_mm = max(2*d, 600)
    phi_main = max(12, int(st.session_state.rebar_df['dia_mm'].max()) if not st.session_state.rebar_df.empty else 12)
    max_hoop = min(0.25*d, 8*phi_main, 100)

    st.warning("⚠️ **IS 13920 Advisory Checks**")
    # Corrected line 334: removed spurious character
    st.write(f"**Confinement Length** at each end $\\geq 2d$ or $600\\text{ mm} = **{hinge_len_mm:.0f}**$ mm.") 
    st.write(f"**Hoop Spacing in Hinge Zone** $\\leq \min(0.25d, 8\\phi_{{main}}, 100) = **{max_hoop:.0f}**$ mm.")
    st.caption("Detailed design shear based on probable moments and joint checks must be verified separately.")

# ---------- VISUALIZATIONS AND EXPORT ----------

st.header("3. Beam Diagrams and Drawings")
st.markdown("---")

st.subheader("3.1 Factored Bending Moment and Shear Force Diagrams")

xs = np.linspace(0, L, 50)
if action_mode == "Derive from loads":
    if support != "Cantilever":
        # Simplified parabolic M for UDL
        M = [w_ULS_15 * x * (L - x) / 2 for x in xs] if support == "Simply Supported" else [w_ULS_15 * x * (L - x) / 8 for x in xs] 
        V = [w_ULS_15 * (L / 2 - x) for x in xs]
    else:
        M = [-0.5 * w_ULS_15 * (x ** 2) for x in xs]
        V = [-w_ULS_15 * x for x in xs]
else:
    # Use approximate shapes for visualization if direct input is used
    M = [Mu_kNm * np.sin(np.pi * x / L) / np.sin(np.pi / 2) for x in xs] if support == "Simply Supported" else [Mu_kNm * (x / L) for x in xs] 
    V = [Vu_kN * np.cos(np.pi * x / L) for x in xs] if support == "Simply Supported" else [Vu_kN * (1 - 2*x/L) for x in xs]

dfM = pd.DataFrame({"x": xs, "M (kN·m)": M}).set_index("x")
dfV = pd.DataFrame({"x": xs, "V (kN)": V}).set_index("x")

fig_M = px.line(dfM, y="M (kN·m)", title="Factored Bending Moment Diagram (kN·m)")
fig_M.update_traces(fill='tozeroy', line_color='rgb(30, 144, 255)')
st.plotly_chart(fig_M, use_container_width=True)

fig_V = px.line(dfV, y="V (kN)", title="Factored Shear Force Diagram (kN)")
fig_V.update_traces(line_color='rgb(255, 69, 0)')
st.plotly_chart(fig_V, use_container_width=True)

st.subheader("3.2 Cross-Section Drawing (Schematic)")

def draw_cross_section_plotly(b_mm, D_mm, cover_mm, df_rebar):
    fig = go.Figure()
    
    fig.add_shape(type="rect", x0=0, y0=0, x1=b_mm, y1=D_mm, 
                  line=dict(color="black", width=2), fillcolor="rgba(192, 192, 192, 0.5)")
    
    cc = float(cover_mm)
    fig.add_shape(type="rect", x0=cc, y0=cc, x1=b_mm-cc, y1=D_mm-cc, 
                  line=dict(color="black", width=1), fillcolor="rgba(0,0,0,0)")

    bottom_bars = df_rebar[df_rebar["position"].str.lower()=="bottom"].groupby("dia_mm")["count"].sum().reset_index()
    top_bars    = df_rebar[df_rebar["position"].str.lower()=="top"].groupby("dia_mm")["count"].sum().reset_index()

    def place_bars(rowset, y_ref, is_bottom):
        bars = []
        for _,r in rowset.iterrows():
            bars += [int(r["dia_mm"]) for _ in range(int(r["count"]))]
        bars.sort(reverse=True)
        if not bars: return
        
        n = len(bars)
        
        if n == 1:
            xs = [b_mm / 2.0]
        else:
            span_avail = b_mm - 2*cc
            step = span_avail / (n - 1)
            xs = [cc + i * step for i in range(n)]

        for i, phi in enumerate(bars):
            r = phi / 2.0
            
            y = y_ref + r if is_bottom else y_ref - r
                
            fig.add_shape(type="circle", x0=xs[i]-r, y0=y-r, x1=xs[i]+r, y1=y+r,
                          line=dict(color="blue", width=1), fillcolor="blue")
            
        label = "+".join([f"{bars.count(d)}-Ø{d}" for d in sorted(set(bars), reverse=True)])
        pos_txt = "TENSION" if is_bottom else "COMPRESSION"
        fig.add_annotation(x=b_mm + 50, y=y, text=label, showarrow=False, font=dict(size=10))

    y_ref_bottom = cc
    y_ref_top = D_mm - cc
    
    place_bars(bottom_bars, y_ref_bottom, is_bottom=True)
    place_bars(top_bars, y_ref_top, is_bottom=False)

    fig.update_layout(
        title=f"Cross Section $b={int(b)}$mm x $D={int(D)}$mm",
        xaxis_title="Width (mm)",
        yaxis_title="Depth (mm)",
        height=400, width=500,
        yaxis_range=[0, D_mm],
        xaxis_range=[-50, b_mm+150],
        showlegend=False,
        plot_bgcolor='white',
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)

draw_cross_section_plotly(b, D, cover, rebar_df_edited)

st.subheader("3.3 Longitudinal Section Drawing (Schematic)")

def draw_longitudinal_section_plotly(L_m, D_mm, cover_mm, df_rebar):
    fig = go.Figure()
    L_mm = L_m * 1000
    
    # 1. Concrete outline (beam elevation)
    fig.add_shape(type="rect", x0=0, y0=0, x1=L_mm, y1=D_mm, 
                  line=dict(color="black", width=2), fillcolor="rgba(192, 192, 192, 0.5)")
    
    # 2. Main Reinforcement
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

    # 3. Supports (triangles/rects)
    fig.add_shape(type="rect", x0=-20, y0=0, x1=20, y1=-D_mm/4, fillcolor="black")
    fig.add_shape(type="rect", x0=L_mm - 20, y0=0, x1=L_mm + 20, y1=-D_mm/4, fillcolor="black")
    
    fig.update_layout(
        title="Longitudinal Elevation (Beam)",
        xaxis_title="Length (mm)",
        yaxis_title="Depth (mm)",
        height=400,
        yaxis_range=[ -D_mm/2, D_mm*1.2], 
        xaxis_range=[-100, L_mm + 100],
        showlegend=False,
        plot_bgcolor='white',
    )
    st.plotly_chart(fig, use_container_width=True)

draw_longitudinal_section_plotly(L, D, cover, rebar_df_edited)


st.header("4. Export Options")
st.markdown("---")

summary = {
    "span": [L], "support": [support], "b_mm": [b], "D_mm": [D], "d_eff_mm": [d],
    "mode": [action_mode],
    "Mu_kNm": [Mu_kNm], "Vu_kN": [Vu_kN], "Tu_kNm": [Tu_kNm], "Nu_kN": [Nu_kN],
    "Ast_req_mm2": [Ast_req], "Ast_prov_mid_mm2": [Ast_prov],
    "Ld_tension_mm": [Ld_tension], "Ld_comp_mm": [Ld_comp],
    "Allowable_L/d": [allowable_L_over_d], "Actual_L/d": [actual_L_over_d],
    "Shear_Spacing_mm": [s_v_final]
}
df_summary = pd.DataFrame(summary)
buf = io.StringIO()
df_summary.to_csv(buf, index=False)

st.download_button("⬇️ Download Design Summary CSV", data=buf.getvalue(), file_name="beam_summary.csv")
