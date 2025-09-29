# Streamlit RCC Beam Designer (IS 456) – Mobile‑friendly single file app
# --------------------------------------------------------------
# Removed matplotlib dependency to avoid ModuleNotFoundError.
# Uses Streamlit's built‑in charting (line_chart) instead of matplotlib.
# --------------------------------------------------------------

import math
from dataclasses import dataclass
import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RCC Beam Designer (IS 456 + IS 13920)", layout="centered")
st.title("RCC Beam Designer – IS 456 + IS 13920")
st.caption("Mobile‑friendly Streamlit app. Units: length in m/mm, loads in kN. Now with tabs, ductile detailing checks, rebar table, and cross‑section SVG.")

# ---------- Helpers ----------

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
        return max(self.D - (self.cover + 8 + 0.5*16), 0.0)

# τ_bd table (design bond stress) – plain bars per IS456 Table 26; deformed +60%
TAU_BD_PLAIN = {20:1.2, 25:1.4, 30:1.5, 35:1.7, 40:1.9, 45:2.0, 50:2.2}

# τ_c(max) approximate per IS456 Table 20 (concrete shear strength max), N/mm2
TAU_C_MAX = {20:2.8, 25:3.1, 30:3.5, 35:3.7, 40:4.0}

# τ_c interpolation tables for p_t
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
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
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

def shear_design(Vu_kN, b_mm, d_mm, fck, fy, Ast_mm2):
    tau_v = (Vu_kN * 1e3) / (b_mm * d_mm)
    p_t = 100.0 * Ast_mm2 / (b_mm * d_mm)
    tc = tau_c(fck, p_t)
    tc_max = TAU_C_MAX[min(TAU_C_MAX.keys(), key=lambda k: abs(k - fck))]
    result = {"tau_v": tau_v, "p_t": p_t, "tau_c": tc, "tau_c_max": tc_max,
              "ok_concrete": tau_v <= tc, "exceeds_tcmax": tau_v > tc_max}
    if tau_v <= tc:
        result.update({"stirrups": "Provide minimum shear reinforcement", "s_v_mm": None})
    else:
        Vus_kN = Vu_kN - tc * b_mm * d_mm / 1e3
        phi, legs = 8.0, 2
        Asv = legs * math.pi * (phi**2) / 4.0
        s_v = (0.87 * fy * Asv * d_mm) / (Vus_kN * 1e3)
        result.update({"s_v_mm": s_v,
                       "stirrups": f"{int(phi)} mm {legs}-leg stirrups @ {max(50, min(300, round(s_v/10)*10))} mm c/c"})
    return result

def ld_required(fck, fy, bar_dia_mm, deformed=True, tension=True):
    fck_key = min(TAU_BD_PLAIN.keys(), key=lambda k: abs(k - fck))
    tau_bd = TAU_BD_PLAIN[fck_key]
    if deformed: tau_bd *= 1.6
    if not tension: tau_bd *= 1.25
    Ld = bar_dia_mm * fy / (4.0 * tau_bd)
    return Ld, tau_bd

# ---------- UI (with TABS) ----------
# Initialize default rebar table
if "rebar_df" not in st.session_state:
    st.session_state.rebar_df = pd.DataFrame([
        {"id":"B1","position":"bottom","zone":"span","dia_mm":20,"count":2,"continuity":"continuous","start_m":0.0,"end_m":None},
        {"id":"B2","position":"bottom","zone":"span","dia_mm":16,"count":1,"continuity":"curtailed","start_m":0.2,"end_m":0.8},
        {"id":"T1","position":"top","zone":"support","dia_mm":16,"count":2,"continuity":"end_zones","start_m":0.0,"end_m":0.25},
    ])

# Sidebar kept minimal for mobile toggles
with st.sidebar:
    st.header("Quick Toggles")
    use_wall = st.checkbox("Add wall load", value=False)
    include_eq = st.checkbox("Include E/W coeff", value=False)
    eq_coeff = st.number_input("Eq. coeff (×wL²)", value=0.0, step=0.05)
    seismic_zone = st.selectbox("Seismic zone (IS 1893)", ["II","III","IV","V"], index=0)
    ductile = st.checkbox("Apply IS 13920 checks", value=True)

# Tabs
tab_inputs, tab_rebar, tab_results, tab_section, tab_ductile = st.tabs([
    "Inputs","Rebar Layout","Results","Cross‑Section","IS 13920"
])

with tab_inputs:
    st.subheader("Geometry & Materials")
    colA,colB = st.columns(2)
    with colA:
        span = st.number_input("Clear span L (m)", value=6.0, min_value=0.5, step=0.1)
        support = st.selectbox("End condition", ["Simply Supported","Continuous","Cantilever"], index=0)
        b = st.number_input("Beam width b (mm)", value=300, step=10, min_value=150)
        D = st.number_input("Overall depth D (mm)", value=500, step=10, min_value=200)
        cover = st.number_input("Clear cover (mm)", value=25, step=5, min_value=20)
    with colB:
        fck = st.selectbox("Concrete fck (N/mm²)", [20,25,30,35,40], index=2)
        fy = st.selectbox("Steel fy (N/mm²)", [415,500], index=1)
        finishes = st.number_input("Finishes (kN/m)", value=2.0, step=0.1, min_value=0.0)
        ll = st.number_input("Live load (kN/m)", value=5.0, step=0.5, min_value=0.0)
        wall_thk = st.number_input("Wall thk (mm)", value=115, step=115, min_value=0)
        wall_h = st.number_input("Wall height (m)", value=3.0, step=0.1, min_value=0.0)
        wall_density = st.number_input("Masonry density (kN/m³)", value=19.0, step=0.5)

    st.subheader("Action Source")
    action_mode = st.radio("Use:", ["Derive from loads", "Direct design actions"], index=0, horizontal=True)

    if action_mode == "Direct design actions":
        st.info("Enter factored design actions (ULS) at the critical section. Signs are ignored; absolute values are used for sizing.")
        colX,colY = st.columns(2)
        with colX:
            Mu_in = st.number_input("Design bending moment Mu (kN·m)", value=120.0, step=5.0, min_value=0.0)
            Vu_in = st.number_input("Design shear Vu (kN)", value=180.0, step=5.0, min_value=0.0)
        with colY:
            Tu_in = st.number_input("Design torsion Tu (kN·m)", value=0.0, step=1.0, min_value=0.0)
            Nu_in = st.number_input("Design axial N_u (kN) (+compression)", value=0.0, step=5.0)

    st.subheader("Bar Defaults")
    colC,colD = st.columns(2)
    with colC:
        t_bar = st.selectbox("Main bar dia default (mm)", [12,16,20,25,28,32], index=1)
    with colD:
        c_bar = st.selectbox("Compression bar dia (mm)", [12,16,20,25], index=0)

# Compute common values
mat, sec = Materials(fck,fy), Section(b,D,cover)
d = sec.d_eff
self_wt = mat.density * (b/1000.0) * (D/1000.0)
wall_kNpm = wall_density * (wall_thk/1000.0) * wall_h if use_wall and wall_thk>0 and wall_h>0 else 0.0
w_DL, w_LL = self_wt + finishes + wall_kNpm, ll
w_ULS_15 = 1.5*(w_DL+w_LL)
w_ULS_12 = 1.2*(w_DL+w_LL)

if support=="Simply Supported": kM,kV=1/8,0.5
elif support=="Cantilever": kM,kV=1/2,1.0
else: kM,kV=1/12,0.6

L=span
# Determine design actions
if 'action_mode' in locals() and action_mode == "Direct design actions":
    Mu_kNm = float(abs(Mu_in))
    Vu_kN  = float(abs(Vu_in))
    Tu_kNm = float(abs(Tu_in))
    Nu_kN  = float(Nu_in)  # compression positive
else:
    Mu_kNm = kM*w_ULS_15*(L**2)
    Vu_kN  = kV*w_ULS_15*L
    if include_eq and eq_coeff>0:
        Mu_kNm += eq_coeff * w_ULS_12 * (L**2)
    Tu_kNm = 0.0
    Nu_kN  = 0.0

Mu_lim=mu_lim_rect(mat.fck,sec.b,d,mat.fy)
Ast_req=ast_singly(Mu_kNm,mat.fy,d)

# ---- Rebar Table (mix & match) ----
with tab_rebar:
    st.write("Define bar groups (rows). Positions: bottom/top/side. Zones: span/support/end_zones. Continuity: continuous/curtailed/end_zones.")
    edited = st.data_editor(st.session_state.rebar_df, num_rows="dynamic")
    st.session_state.rebar_df = edited

    # Compute Ast by zone
    def ast_from_rows(rows, pos_filter, zone_filter=None):
        sel = rows[rows["position"].str.lower()==pos_filter]
        if zone_filter:
            sel = sel[sel["zone"].str.lower().isin(zone_filter)]
        areas = (math.pi*(sel["dia_mm"]**2)/4.0) * sel["count"]
        return float(areas.sum())

    Ast_bottom_span = ast_from_rows(edited, "bottom", ["span","all"])
    Ast_top_support = ast_from_rows(edited, "top", ["support","end_zones","all"])
    Ast_bottom_support = ast_from_rows(edited, "bottom", ["support","end_zones"])

    st.markdown(f"**Ast provided (bottom @ midspan)**: {Ast_bottom_span:.0f} mm² — **Ast required**: {Ast_req:.0f} mm²")
    need_more = Ast_bottom_span < Ast_req
    if need_more:
        st.warning("Bottom Ast at midspan < required. Increase bars or depth.")

# Use provided Ast for shear p_t etc.
Ast_prov = max(Ast_bottom_span, 1.0)

# If torsion present, use equivalent shear per IS 456 (advisory): Ve = Vu + 1.6*Tu/b
Vu_eff_kN = Vu_kN + (1.6*Tu_kNm*1000.0/ sec.b) if Tu_kNm>0 else Vu_kN
shear = shear_design(Vu_eff_kN, sec.b, d, mat.fck, mat.fy, Ast_prov)
Ld_tension,_ = ld_required(mat.fck, mat.fy, t_bar)
Ld_comp,_ = ld_required(mat.fck, mat.fy, c_bar, tension=False)

base_Ld=LD_LIMITS[support]
p_t=100.0*Ast_prov/(sec.b*d)
mod=clamp(1.0+0.15*(p_t-1.0),0.8,1.3)
allowable_L_over_d=base_Ld*mod
actual_L_over_d=(L*1000.0)/d

with tab_results:
    st.subheader("Summary")
    col1,col2 = st.columns(2)
    with col1:
        st.metric("Mu (ULS)", f"{Mu_kNm:.1f} kN·m")
        st.metric("Vu (ULS)", f"{Vu_kN:.1f} kN")
        st.metric("Tu (ULS)", f"{Tu_kNm:.1f} kN·m")
        st.metric("Nu (ULS)", f"{Nu_kN:.1f} kN")
        if action_mode == "Derive from loads":
            st.write(f"Self‑wt {self_wt:.2f}, Wall {wall_kNpm:.2f}, Finishes {finishes:.2f}, LL {ll:.2f} kN/m")
        else:
            st.info("Using directly entered design actions.")
    with col2:
        st.metric("d (eff)", f"{d:.0f} mm")
        st.metric("Mu,lim", f"{Mu_lim:.1f} kN·m")
        st.write(f"Ast_req midspan: **{Ast_req:.0f} mm²**, Ast_prov(mid): **{Ast_prov:.0f} mm²**")
        st.write("Bottom midspan OK" if Ast_prov>=Ast_req else "Increase bottom Ast or depth")

    st.subheader("Shear / Torsion")
    st.write(f"Using equivalent shear **V_e** = {Vu_eff_kN:.1f} kN (Vu + 1.6·Tu/b).")
    st.write(f"τ_v = {shear['tau_v']:.3f}, τ_c ≈ {shear['tau_c']:.3f}, τ_c,max ≈ {shear['tau_c_max']:.2f}")
    if shear["exceeds_tcmax"]:
        st.error("τ_v > τ_c,max → Increase b/d or fck.")
    elif shear["ok_concrete"]:
        st.success("Concrete can carry shear; provide minimum links.")
    else:
        st.warning("Provide shear reinforcement (compute spacing per site preferences).")
        # quick spacing with user default stirrup dia from table? approximate 8mm 2‑leg
        phi = 8.0; legs = 2
        Asv = legs * math.pi * (phi**2) / 4.0
        Vus_kN = Vu_kN - shear['tau_c'] * sec.b * d / 1e3
        s_v = (0.87 * mat.fy * Asv * d) / (Vus_kN * 1e3)
        s_v_final = max(50, min(300, round(s_v/10)*10))
        st.write(f"Try {int(phi)} mm {legs}-leg @ **{s_v_final} mm c/c** (cap by min{{0.75d, 300}} = {min(0.75*d,300):.0f} mm)")

    st.subheader("Serviceability (Span/Depth)")
    st.write(f"Basic L/d={base_Ld:.1f}, modifier={mod:.2f} → allowable **{allowable_L_over_d:.1f}**; actual **{actual_L_over_d:.1f}** → {'OK' if actual_L_over_d<=allowable_L_over_d else 'Increase depth/comp steel'}.")

    # Diagrams without matplotlib (kept inside Results tab to avoid indentation issues)
    import numpy as np
    xs = np.linspace(0, L, 50)
    if action_mode == "Derive from loads":
        if support != "Cantilever":
            M = [kM * w_ULS_15 * (L * x - x * x) for x in xs]
            V = [w_ULS_15 * (L / 2 - x) for x in xs]
        else:
            M = [-0.5 * w_ULS_15 * (x ** 2) for x in xs]
            V = [-w_ULS_15 * x for x in xs]
    else:
        # For directly entered actions, draw a simple representative shape
        M = [Mu_kNm * (x / L) for x in xs]
        V = [Vu_kN for _ in xs]

    import pandas as _pd
    dfM = _pd.DataFrame({"x": xs, "M (kN·m)": M}).set_index("x")
    dfV = _pd.DataFrame({"x": xs, "V (kN)": V}).set_index("x")
    st.line_chart(dfM)
    st.line_chart(dfV)

    # CSV download
    summary = {
        "span": [L], "support": [support],
        "mode": [action_mode],
        "Mu_kNm": [Mu_kNm], "Vu_kN": [Vu_kN], "Tu_kNm": [Tu_kNm], "Nu_kN": [Nu_kN],
        "Ast_req": [Ast_req], "Ast_prov_mid": [Ast_prov],
        "Ld_tension": [Ld_tension], "Ld_comp": [Ld_comp],
        "allowable_L/d": [allowable_L_over_d], "actual_L/d": [actual_L_over_d]
    }
    df = pd.DataFrame(summary)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Download CSV", data=buf.getvalue(), file_name="beam_summary.csv")

# ---- Cross Section SVG (tab) ----
with tab_section:
    st.subheader("Cross‑section (schematic)")
    bs, Ds = float(b), float(D)
    padding = 30
    svg_w, svg_h = bs + 2*padding, Ds + 2*padding
    cc = float(cover)
    stirrup_x0 = padding + 5
    stirrup_y0 = padding + 5
    stirrup_w = bs - 10
    stirrup_h = Ds - 10

    df_re = st.session_state.rebar_df.copy()
    bottom = df_re[df_re["position"].str.lower()=="bottom"].groupby("dia_mm")["count"].sum().reset_index()
    top    = df_re[df_re["position"].str.lower()=="top"].groupby("dia_mm")["count"].sum().reset_index()

    def bars_to_svg(rowset, y_from_top):
        elems=[]
        if rowset.empty: return elems
        total_bars = int(rowset["count"].sum())
        if total_bars<=0: return elems
        gap = (stirrup_w - 2*cc) / (max(total_bars-1,1))
        x = stirrup_x0 + cc
        y = stirrup_y0 + y_from_top
        # flatten list of bars by dia
        bars=[]
        for _,r in rowset.iterrows():
            bars += [int(r["dia_mm"]) for _ in range(int(r["count"]))]
        bars.sort(reverse=True)
        for i,phi in enumerate(bars):
            cx = x + i*gap
            r = phi/2.0
            elems.append(f'<circle cx="{cx:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="none" stroke="black" stroke-width="1" />')
        return elems

    bottom_circles = bars_to_svg(bottom, stirrup_h-cc)
    top_circles    = bars_to_svg(top, cc)

    def leader_text(x,y,text):
        x2,y2 = x+40,y-20
        return (f'<line x1="{x}" y1="{y}" x2="{x2}" y2="{y2}" stroke="black" stroke-width="1" />'
                f'<text x="{x2+4}" y="{y2-4}" font-size="12">{text}</text>')

    leader_elems=[]
    if not bottom.empty:
        txt = "+".join([f"{int(r['count'])}‑{int(r['dia_mm'])}" for _,r in bottom.iterrows()]) + " bottom"
        leader_elems.append(leader_text(stirrup_x0+stirrup_w*0.6, stirrup_y0+stirrup_h-cc, txt))
    if not top.empty:
        txt = "+".join([f"{int(r['count'])}‑{int(r['dia_mm'])}" for _,r in top.iterrows()]) + " top"
        leader_elems.append(leader_text(stirrup_x0+stirrup_w*0.6, stirrup_y0+cc, txt))

    svg = f'''
    <svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg">
      <rect x="{padding}" y="{padding}" width="{bs}" height="{Ds}" fill="white" stroke="black" stroke-width="2" />
      <rect x="{stirrup_x0}" y="{stirrup_y0}" width="{stirrup_w}" height="{stirrup_h}" fill="none" stroke="black" stroke-width="1" />
      {''.join(bottom_circles)}
      {''.join(top_circles)}
      {''.join(leader_elems)}
      <text x="{padding}" y="{padding-8}" font-size="12">b={int(b)} mm</text>
      <text x="{padding+bs-80}" y="{padding-8}" font-size="12">D={int(D)} mm</text>
    </svg>
    '''
    st.markdown(svg, unsafe_allow_html=True)
    st.caption("Schematic only. Bar sizes to scale; spacing schematic. Adjust rebar table to see updates.")

# ---- IS 13920 advisory checks (tab) ----
with tab_ductile:
    st.subheader("IS 13920 – Ductile Detailing (Advisory)")
    if not ductile:
        st.info("Ductile detailing not selected (toggle in sidebar).")
    # Confinement length at each end
    hinge_len_mm = max(2*d, 600)
    # Maximum hoop spacing in hinge zones
    phi_main = max(12, int(st.session_state.rebar_df['dia_mm'].max()) if not st.session_state.rebar_df.empty else 12)
    max_hoop = min(0.25*d, 8*phi_main, 100)
    st.write(f"Confinement length ≥ **{hinge_len_mm:.0f} mm** at each end.")
    st.write(f"Hoop spacing in confinement zones ≤ **{max_hoop:.0f} mm** (and ≤ 100 mm). Use 135° hooks with 10d extension.")
    st.write("Avoid lap splices in hinge regions; if unavoidable, provide closely spaced hoops over splice length.")
    st.write("Provide crossties (α‑ties) for wider beams; ensure anchorage beyond support face ≥ Ld.")
    st.warning("Full IS 13920 design shear based on probable moments and joint checks are not automated here; verify in detailed design.")

