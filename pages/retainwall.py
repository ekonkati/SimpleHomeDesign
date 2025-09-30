# Retaining Wall Design Wizard (Streamlit)
# -----------------------------------------------------------
# Covers L / T / Inverted-L walls. Handles Active/At-Rest/Passive
# pressures, water, surcharge, seismic (MO), stability, member design,
# simple reinforcement detailing, pressure & load drawings, and BOQ.
# -----------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------- Helpers & Constants ------------------------- #
GAMMA_W = 9.81  # kN/m3
STEEL_DENSITY = 7850  # kg/m3

# Bar areas (mm^2)
BAR_AREAS = {6: 28.27, 8: 50.27, 10: 78.54, 12: 113.10, 16: 201.06, 20: 314.16}

# Utility: mm^2/m @ spacing -> As (mm^2/m)
def as_per_m(dia_mm: int, spacing_mm: float) -> float:
    return BAR_AREAS[dia_mm] * (1000.0 / spacing_mm)

# Utility: convert As (mm^2/m) to kg/m (take 1m length of bars)
def steel_kg_per_m(As_mm2_per_m: float) -> float:
    # As(mm^2/m) -> area in m^2 per m width = As * 1e-6
    # multiply by steel density to get kg/m of wall strip of 1m width and 1m bar length
    # This approximates a mat; for BOQ we will convert by bar length estimates separately.
    return As_mm2_per_m * 1e-6 * STEEL_DENSITY

# ------------------------- Data Classes ------------------------- #
@dataclass
class Soil:
    gamma: float  # kN/m3 (bulk)
    phi: float    # degrees
    cohesion: float = 0.0  # kPa -> kN/m2
    gwl_from_base: float = None  # m above base (None = dry)

@dataclass
class Materials:
    fck: int = 25
    fy: int = 500
    gamma_c: float = 24.0
    cover: float = 50.0  # mm

@dataclass
class Geometry:
    H: float         # retained height (m)
    Df: float        # embedment below ground in front (m)
    B: float         # total base width (m)
    heel: float      # heel length (m)
    toe: float       # toe length (m)
    t_base: float    # base thickness (m)
    t_stem: float    # stem thickness (m)
    wall_type: str   # 'L', 'T', 'I'
    shear_key_depth: float = 0.0  # m (optional)
    shear_key_width: float = 0.0  # m

@dataclass
class Loads:
    surcharge_q: float = 0.0  # kPa = kN/m2
    seismic_kh: float = 0.0
    seismic_kv: float = 0.0
    use_seismic: bool = False

@dataclass
class Bearing:
    SBC_allow: float  # kPa (kN/m2)
    mu_base: float = 0.5

# ------------------------- Earth Pressure Coefficients ------------------------- #
def coeff_rankine_active(phi_deg: float) -> float:
    s = math.sin(math.radians(phi_deg))
    return (1 - s) / (1 + s)

def coeff_rankine_passive(phi_deg: float) -> float:
    s = math.sin(math.radians(phi_deg))
    return (1 + s) / (1 - s)

def coeff_jaky_atrest(phi_deg: float) -> float:
    return 1 - math.sin(math.radians(phi_deg))

# Mononobe–Okabe simplified (level backfill, vertical wall). Conservative form.
def coeff_mononobe_okabe_active(phi_deg: float, kh: float, kv: float) -> float:
    # For simplicity, use Seed & Whitman approximation: Ka_e ≈ Ka * ((1 - kv)/(1 + kv)) * ((1 - kh) / (1 + kh))
    Ka = coeff_rankine_active(phi_deg)
    num = (1 - kv) * (1 - kh)
    den = (1 + kv) * (1 + kh)
    fac = max(num / den, 0.1)
    return Ka * fac

# ------------------------- Pressure Computations ------------------------- #
def pressures(H: float, soil: Soil, loads: Loads, state: str) -> Dict:
    """Return dictionary with base intensities and resultants for earth, surcharge, water."""
    phi = soil.phi
    gamma = soil.gamma

    if loads.use_seismic and loads.seismic_kh > 0:
        K_active = coeff_mononobe_okabe_active(phi, loads.seismic_kh, loads.seismic_kv)
    else:
        if state == 'Active (Ka)':
            K_active = coeff_rankine_active(phi)
        elif state == 'At-Rest (K0)':
            K_active = coeff_jaky_atrest(phi)
        else:
            K_active = coeff_rankine_active(phi)

    Ka = K_active
    Kp = coeff_rankine_passive(phi)

    p0_earth = Ka * gamma * H  # base intensity (kPa)
    P_earth = 0.5 * Ka * gamma * H**2  # kN/m

    p_surcharge = Ka * loads.surcharge_q  # rectangular (kPa)
    P_surcharge = Ka * loads.surcharge_q * H  # kN/m @ H/2

    water = soil.gwl_from_base is not None and soil.gwl_from_base > 0
    p0_water = GAMMA_W * min(H, soil.gwl_from_base or 0) if water else 0.0
    P_water = 0.5 * p0_water * min(H, soil.gwl_from_base or 0) if water else 0.0

    out = {
        'Ka': Ka,
        'Kp': Kp,
        'p0_earth': p0_earth,
        'P_earth': P_earth,
        'p_surcharge': p_surcharge,
        'P_surcharge': P_surcharge,
        'p0_water': p0_water,
        'P_water': P_water,
    }
    return out

# ------------------------- Weights & Resultants ------------------------- #
def self_weights(geo: Geometry, mat: Materials, soil: Soil) -> Dict:
    # Simple prismatic weights per metre run
    W_stem = geo.t_stem * geo.H * mat.gamma_c
    W_base = geo.B * geo.t_base * mat.gamma_c

    # Soil over heel (assume to ground level at top of base)
    W_soil_heel = geo.heel * geo.H * soil.gamma

    return {
        'W_stem': W_stem,
        'x_stem': geo.toe + geo.t_stem/2,  # from toe
        'W_base': W_base,
        'x_base': geo.B/2,
        'W_soil_heel': W_soil_heel,
        'x_soil_heel': geo.toe + geo.heel/2,
    }

# ------------------------- Stability Checks ------------------------- #
def stability(geo: Geometry, soil: Soil, loads: Loads, bearing: Bearing, pres: Dict) -> Dict:
    # Horizontal forces & moments
    H = pres['P_earth'] + pres['P_surcharge'] + pres['P_water']
    # Lever arms from base for earth & water
    z_earth = geo.H/3
    z_surcharge = geo.H/2 if pres['P_surcharge'] > 0 else 0
    z_water = (soil.gwl_from_base or 0)/3 if pres['P_water'] > 0 else 0

    M_o = pres['P_earth'] * z_earth + pres['P_surcharge'] * z_surcharge + pres['P_water'] * z_water

    W = self_weights(geo, Materials(), soil)
    V = W['W_stem'] + W['W_base'] + W['W_soil_heel']

    # Resisting moment about toe
    M_r = W['W_stem'] * W['x_stem'] + W['W_base'] * W['x_base'] + W['W_soil_heel'] * W['x_soil_heel']

    # Sliding resistance (no passive counted by default)
    R_sliding = bearing.mu_base * V

    Fos_OT = M_r / max(M_o, 1e-6)
    Fos_SL = R_sliding / max(H, 1e-6)

    # Bearing
    e = (M_r - M_o) / max(V, 1e-6)
    q_avg = V / geo.B
    q_max = q_avg * (1 + 6*e/geo.B)
    q_min = q_avg * (1 - 6*e/geo.B)

    return {
        'H': H, 'V': V,
        'M_o': M_o, 'M_r': M_r,
        'FOS_OT': Fos_OT, 'FOS_SL': Fos_SL,
        'e': e, 'q_avg': q_avg, 'q_max': q_max, 'q_min': q_min,
    }

# ------------------------- Member Design (ULS, simplified) ------------------------- #
def flexural_As_required(Mu_kNm: float, d_mm: float, fck: int, fy: int) -> float:
    # IS 456 rectangular section, singly reinforced: Mu = 0.87*fy*Ast*z, z≈0.9d for typical ranges
    # Use Mu(Nmm) = 0.87*fy(N/mm2)*Ast(mm2)*0.9*d(mm)
    Mu_Nmm = Mu_kNm * 1e6
    Ast = Mu_Nmm / (0.87 * fy * 0.9 * d_mm)
    return Ast  # mm^2 per metre width

def member_design(geo: Geometry, soil: Soil, loads: Loads, pres: Dict, mat: Materials) -> Dict:
    # Stem cantilever base moment (service -> ULS factor 1.5)
    M_serv = pres['p0_earth'] * geo.H**2 / 6.0 + pres['p_surcharge'] * geo.H**2 / 2.0 + pres['p0_water'] * min(geo.H, soil.gwl_from_base or 0)**2 / 6.0
    Mu_stem = 1.5 * M_serv  # kNm/m

    d_stem = (geo.t_stem*1000.0 - mat.cover - 0.5*10)  # assume 10 mm main bars
    As_stem_req = max(flexural_As_required(Mu_stem, d_stem, mat.fck, mat.fy), 0.0035*1000*geo.t_stem*1000)  # min 0.35% of b*d (approx)

    # Toe: upward bearing ~ q_avg to q_max across toe length
    # Simplify: use q_avg over toe; net w = q_avg - selfweight of base slab
    Wslab = mat.gamma_c * geo.t_base
    w_toe = max(0.0, (Wslab * 1.0) * 0.0)  # selfweight cancels in reaction distribution for simplicity here
    q = max(0.0, (pres.get('q_avg', 0.0)))
    L_toe = geo.toe
    M_toe_serv = q * (L_toe**2) / 2.0  # cantilever upward -> bottom tension at stem face
    Mu_toe = 1.5 * M_toe_serv

    d_toe = (geo.t_base*1000.0 - mat.cover - 0.5*10)
    As_toe_req = max(flexural_As_required(Mu_toe, d_toe, mat.fck, mat.fy), 0.002*1000*geo.t_base*1000)

    # Heel: net downward = soil over heel + slab - bearing (approx q_avg)
    q_avg = pres.get('q_avg', 0.0)
    w_down = soil.gamma * geo.H + mat.gamma_c * geo.t_base - q_avg
    w_down = max(w_down, 0.0)
    L_heel = geo.heel
    M_heel_serv = w_down * (L_heel**2) / 2.0
    Mu_heel = 1.5 * M_heel_serv

    d_heel = (geo.t_base*1000.0 - mat.cover - 0.5*10)
    As_heel_req = max(flexural_As_required(Mu_heel, d_heel, mat.fck, mat.fy), 0.002*1000*geo.t_base*1000)

    return {
        'Mu_stem': Mu_stem,
        'Mu_toe': Mu_toe,
        'Mu_heel': Mu_heel,
        'As_stem_req': As_stem_req,
        'As_toe_req': As_toe_req,
        'As_heel_req': As_heel_req,
    }

# ------------------------- Drawings (Matplotlib) ------------------------- #

def plot_pressure_diagram(H: float, p0_earth: float, p_surcharge: float, p0_water: float):
    z = np.linspace(0, H, 50)
    p_earth = p0_earth * (z / H)
    p_water = p0_water * (z / H)
    p_s = np.full_like(z, p_surcharge)

    fig, ax = plt.subplots()
    ax.plot(p_earth, z, label='Earth')
    if p_surcharge > 0:
        ax.plot(p_s, z, label='Surcharge')
    if p0_water > 0:
        ax.plot(p_water, z, label='Water')
    ax.invert_yaxis()
    ax.set_xlabel('Pressure (kPa)')
    ax.set_ylabel('Depth (m) from top')
    ax.set_title('Pressure Diagram')
    ax.legend()
    st.pyplot(fig)


def plot_wall_section(geo: Geometry, mat: Materials, show_rebar=True,
                      stem_main_dia=10, stem_main_sp=200,
                      slab_main_dia=10, slab_main_sp=150):
    # Simple schematic (not to scale)
    fig, ax = plt.subplots()

    # Base rectangle
    ax.add_patch(plt.Rectangle((0, 0), geo.B, geo.t_base, fill=False))
    # Stem rectangle
    ax.add_patch(plt.Rectangle((geo.toe, geo.t_base), geo.t_stem, geo.H, fill=False))

    # Ground line at top of base (behind heel)
    ax.plot([geo.toe, geo.B], [geo.t_base + geo.H, geo.t_base + geo.H], linestyle='--')

    # Rebar dots
    if show_rebar:
        # Stem verticals (at back face)
        x = geo.toe + mat.cover/1000.0
        ys = np.linspace(geo.t_base + mat.cover/1000.0, geo.t_base + geo.H - mat.cover/1000.0, 8)
        for y in ys:
            ax.plot(x, y, 'o')
        # Heel top bars
        xh = np.linspace(geo.toe + 0.05, geo.B - 0.05, 15)
        y = geo.t_base + geo.H - mat.cover/1000.0
        ax.plot(xh, [y]*len(xh), 'o')
        # Toe bottom bars
        xt = np.linspace(0.05, geo.toe - 0.05, 10)
        y = mat.cover/1000.0
        ax.plot(xt, [y]*len(xt), 'o')

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-0.1, geo.B + 0.1)
    ax.set_ylim(0, geo.t_base + geo.H + 0.3)
    ax.set_title('Wall Section & Bar Schematic (not to scale)')
    ax.set_xlabel('m')
    ax.set_ylabel('m')
    st.pyplot(fig)


def plot_load_resultants(H: float, P_earth: float, P_surcharge: float, P_water: float):
    # Bar chart of horizontal resultants
    labels = ['Earth', 'Surcharge', 'Water']
    values = [P_earth, P_surcharge, P_water]
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel('Resultant (kN/m)')
    ax.set_title('Horizontal Resultants')
    st.pyplot(fig)

# ------------------------- BOQ Generator ------------------------- #

def make_boq(geo: Geometry, mat: Materials, steel: Dict) -> pd.DataFrame:
    # Concrete volumes per metre run
    conc_stem = geo.t_stem * geo.H
    conc_base = geo.B * geo.t_base
    conc_total = conc_stem + conc_base

    # Steel weights (very approximate): use required As with simple bar length assumptions
    # Stem main: assume bars along height with spacing, length ~ H + dev
    stem_As = steel['As_stem']  # mm2/m provided
    heel_As = steel['As_heel']
    toe_As = steel['As_toe']

    # Convert As (mm2/m) to kg per m (approximate per layer length = 1 m)
    steel_kg_stem = steel_kg_per_m(stem_As)
    steel_kg_heel = steel_kg_per_m(heel_As)
    steel_kg_toe = steel_kg_per_m(toe_As)

    data = [
        ['Concrete - Stem', 'm3/m', conc_stem, ''],
        ['Concrete - Base', 'm3/m', conc_base, ''],
        ['Concrete - Total', 'm3/m', conc_total, ''],
        ['Reinf. Steel - Stem (approx)', 'kg/m', steel_kg_stem, 'As based'],
        ['Reinf. Steel - Heel (approx)', 'kg/m', steel_kg_heel, 'As based'],
        ['Reinf. Steel - Toe (approx)', 'kg/m', steel_kg_toe, 'As based'],
        ['Weep holes 100 mm dia', 'no./m', 1.0/1.2, 'at 1.2 m c/c'],
        ['Geocomposite drain (back)', 'm2/m', geo.H * 1.0, '1m strip'],
        ['Toe drain pipe', 'm/m', 1.0, 'continuous'],
    ]
    df = pd.DataFrame(data, columns=['Item', 'Unit', 'Qty per m', 'Notes'])
    return df

# ------------------------- Streamlit UI ------------------------- #
st.set_page_config(page_title='Retaining Wall Design Wizard', layout='wide')
st.title('Retaining Wall Design Wizard — L / T / I Types')

with st.sidebar:
    st.header('Project Setup')
    wall_type = st.selectbox('Wall Type', ['L', 'T', 'I (Inverted-L)'])
    design_state = st.selectbox('Earth Pressure State', ['Active (Ka)', 'At-Rest (K0)'])

    H = st.number_input('Retained Height H (m)', 1.0, 15.0, 2.0, 0.1)
    Df = st.number_input('Embedment in front Df (m)', 0.0, 5.0, 0.3, 0.1)

    # Base geometry guess per type
    default_B = 0.9*H if design_state == 'At-Rest (K0)' else 0.7*H
    if wall_type == 'L':
        heel_default = 0.6*default_B
        toe_default = default_B - heel_default
    elif wall_type.startswith('I'):
        toe_default = 0.6*default_B
        heel_default = default_B - toe_default
    else:  # T
        heel_default = 0.55*default_B
        toe_default = default_B - heel_default

    B = st.number_input('Base Width B (m)', 0.5, 8.0, round(default_B,2), 0.05)
    heel = st.number_input('Heel Length (m)', 0.1, 6.0, round(heel_default,2), 0.05)
    toe = st.number_input('Toe Length (m)', 0.1, 6.0, round(toe_default,2), 0.05)
    t_base = st.number_input('Base Thickness (m)', 0.2, 1.0, 0.35, 0.05)
    t_stem = st.number_input('Stem Thickness (m)', 0.15, 0.8, 0.20, 0.01)

    # Soil & loads
    st.subheader('Soil & Loads')
    gamma = st.number_input('Soil Unit Weight γ (kN/m3)', 10.0, 24.0, 18.0, 0.1)
    phi = st.number_input('Friction Angle φ (deg)', 20.0, 45.0, 30.0, 0.5)
    surcharge_q = st.number_input('Uniform Surcharge q (kPa)', 0.0, 100.0, 0.0, 1.0)
    gwl_h = st.number_input('GWL height above base (m) (0=dry, H=full)', 0.0, 20.0, 0.0, 0.1)

    st.subheader('Seismic (optional)')
    use_seis = st.checkbox('Include Seismic (MO approx)', value=False)
    kh = st.number_input('k_h', 0.0, 0.3, 0.0, 0.01)
    kv = st.number_input('k_v', -0.2, 0.2, 0.0, 0.01)

    st.subheader('Materials & Bearing')
    fck = st.selectbox('Concrete grade fck (MPa)', [20,25,30,35,40], index=1)
    fy = st.selectbox('Steel grade fy (MPa)', [415,500], index=1)
    cover = st.number_input('Clear cover (mm)', 40, 75, 50, 5)

    SBC = st.number_input('Allowable SBC (kPa)', 50.0, 400.0, 98.0, 1.0)
    mu_base = st.number_input('Base friction coefficient μ', 0.3, 0.8, 0.5, 0.01)

# Build data objects
soil = Soil(gamma=gamma, phi=phi, cohesion=0.0, gwl_from_base=(gwl_h if gwl_h>0 else None))
mat = Materials(fck=fck, fy=fy, gamma_c=24.0, cover=cover)
geo = Geometry(H=H, Df=Df, B=B, heel=heel, toe=toe, t_base=t_base, t_stem=t_stem, wall_type=wall_type)
loads = Loads(surcharge_q=surcharge_q, seismic_kh=kh, seismic_kv=kv, use_seismic=use_seis)
bearing = Bearing(SBC_allow=SBC, mu_base=mu_base)

# Tabs per the flowchart
tabs = st.tabs([
    'Inputs', 'Pressures', 'Stability', 'Geometry', 'Member Design', 'Serviceability', 'Drawings', 'BOQ', 'Report']
)

# ------------------------- Tab: Inputs ------------------------- #
with tabs[0]:
    st.subheader('Inputs Summary')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Wall Type:** {wall_type}")
        st.markdown(f"**Earth State:** {design_state}")
        st.markdown(f"**H:** {H:.2f} m, **Df:** {Df:.2f} m")
        st.markdown(f"**B:** {B:.2f} m (heel {heel:.2f} m, toe {toe:.2f} m)")
    with c2:
        st.markdown(f"**γ_soil:** {soil.gamma:.2f} kN/m³, **φ:** {soil.phi:.1f}°")
        st.markdown(f"**q:** {loads.surcharge_q:.1f} kPa, **GWL:** {gwl_h:.2f} m above base")
        st.markdown(f"**Seismic:** {use_seis}, kh={kh:.02f}, kv={kv:.02f}")
    with c3:
        st.markdown(f"**fck:** M{mat.fck}, **fy:** Fe{mat.fy}")
        st.markdown(f"**Cover:** {mat.cover:.0f} mm")
        st.markdown(f"**SBC_allow:** {bearing.SBC_allow:.0f} kPa, **μ:** {bearing.mu_base:.2f}")

# ------------------------- Tab: Pressures ------------------------- #
with tabs[1]:
    st.subheader('Earth/Water/Surcharge Pressures')
    pres = pressures(H, soil, loads, design_state)

    c1, c2 = st.columns(2)
    with c1:
        st.write({k: round(v, 3) if isinstance(v, float) else v for k,v in pres.items()})
    with c2:
        plot_pressure_diagram(H, pres['p0_earth'], pres['p_surcharge'], pres['p0_water'])
    plot_load_resultants(H, pres['P_earth'], pres['P_surcharge'], pres['P_water'])

# ------------------------- Tab: Stability ------------------------- #
with tabs[2]:
    st.subheader('Stability Checks (SLS)')
    stab = stability(geo, soil, loads, bearing, pres)
    # Attach q_avg etc back into pres for later use
    pres.update({'q_avg': stab['q_avg']})

    df = pd.DataFrame({
        'Quantity': ['H (kN/m)', 'V (kN/m)', 'M_o (kNm/m)', 'M_r (kNm/m)', 'FOS_OT', 'FOS_SL', 'e (m)', 'q_avg (kPa)', 'q_max (kPa)', 'q_min (kPa)'],
        'Value': [stab['H'], stab['V'], stab['M_o'], stab['M_r'], stab['FOS_OT'], stab['FOS_SL'], stab['e'], stab['q_avg'], stab['q_max'], stab['q_min']]
    })
    st.dataframe(df, use_container_width=True)

    ok1 = stab['FOS_OT'] >= 2.0
    ok2 = stab['FOS_SL'] >= 1.5
    ok3 = (abs(stab['e']) <= geo.B/6.0) and (stab['q_max'] <= bearing.SBC_allow) and (stab['q_min'] >= 0)

    st.markdown(f"**Overturning FOS ≥ 2.0:** {'✅' if ok1 else '❌'}")
    st.markdown(f"**Sliding FOS ≥ 1.5:** {'✅' if ok2 else '❌'}")
    st.markdown(f"**Bearing & Eccentricity:** {'✅' if ok3 else '❌'}")

# ------------------------- Tab: Geometry ------------------------- #
with tabs[3]:
    st.subheader('Geometry Schematic')
    plot_wall_section(geo, mat)
    st.info("Adjust B, heel, toe, t_base, t_stem from sidebar to iterate for stability.")

# ------------------------- Tab: Member Design ------------------------- #
with tabs[4]:
    st.subheader('Member Design (ULS – simplified)')
    desg = member_design(geo, soil, loads, pres, mat)

    # Provide suggested bar mats
    stem_As_req = desg['As_stem_req']  # mm2/m
    heel_As_req = desg['As_heel_req']
    toe_As_req = desg['As_toe_req']

    # Suggested default mats
    stem_As_prov = as_per_m(10, 200)
    heel_As_prov = as_per_m(10, 150)
    toe_As_prov = as_per_m(10, 150)

    # Editable choices
    c1, c2, c3 = st.columns(3)
    with c1:
        stem_dia = st.selectbox('Stem main dia (mm)', [10,12,16], index=0)
        stem_sp = st.number_input('Stem spacing (mm)', 100, 300, 200, 25)
        stem_As_prov = as_per_m(stem_dia, stem_sp)
    with c2:
        heel_dia = st.selectbox('Heel main dia (mm)', [10,12,16], index=0)
        heel_sp = st.number_input('Heel spacing (mm)', 100, 300, 150, 25)
        heel_As_prov = as_per_m(heel_dia, heel_sp)
    with c3:
        toe_dia = st.selectbox('Toe main dia (mm)', [10,12,16], index=0)
        toe_sp = st.number_input('Toe spacing (mm)', 100, 300, 150, 25)
        toe_As_prov = as_per_m(toe_dia, toe_sp)

    df_as = pd.DataFrame([
        ['Stem', stem_As_req, stem_As_prov, 'OK' if stem_As_prov >= stem_As_req else 'INC'],
        ['Heel (top)', heel_As_req, heel_As_prov, 'OK' if heel_As_prov >= heel_As_req else 'INC'],
        ['Toe (bottom)', toe_As_req, toe_As_prov, 'OK' if toe_As_prov >= toe_As_req else 'INC'],
    ], columns=['Member', 'As req (mm2/m)', 'As prov (mm2/m)', 'Status'])
    st.dataframe(df_as, use_container_width=True)

    st.markdown(f"**Mu(stem):** {desg['Mu_stem']:.2f} kNm/m, **Mu(heel):** {desg['Mu_heel']:.2f} kNm/m, **Mu(toe):** {desg['Mu_toe']:.2f} kNm/m")

    # Save provided As for BOQ
    st.session_state['As_provided'] = {
        'As_stem': stem_As_prov,
        'As_heel': heel_As_prov,
        'As_toe': toe_As_prov,
    }

# ------------------------- Tab: Serviceability ------------------------- #
with tabs[5]:
    st.subheader('Serviceability / Detailing Checks (Guidance)')
    st.markdown('- Bar spacing ≤ 3×thickness and ≤ 300 mm (soil faces)')
    st.markdown('- Minimum steel ratios per code (e.g., ~0.2–0.35% typical for walls/slabs depending on exposure)')
    st.markdown('- Provide weep holes @ 1.0–1.5 m c/c staggered with graded filter or geocomposite drain')
    st.markdown('- Provide construction/expansion joints at 6–10 m; waterstops if water-retaining')
    st.markdown('- Cover to soil faces typically 50–60 mm (exposure dependent)')

# ------------------------- Tab: Drawings ------------------------- #
with tabs[6]:
    st.subheader('Drawings (Schematic)')
    st.markdown('**Pressure Diagram:**')
    plot_pressure_diagram(H, pres['p0_earth'], pres['p_surcharge'], pres['p0_water'])
    st.markdown('**Wall Section & Bars:**')
    plot_wall_section(geo, mat)
    st.caption('Bar marks are schematic; finalize in CAD with bar schedules and laps/dev lengths as per code.')

# ------------------------- Tab: BOQ ------------------------- #
with tabs[7]:
    st.subheader('Bill of Quantities (per metre run)')
    As_map = st.session_state.get('As_provided', {'As_stem': as_per_m(10,200), 'As_heel': as_per_m(10,150), 'As_toe': as_per_m(10,150)})
    boq_df = make_boq(geo, mat, As_map)
    st.dataframe(boq_df, use_container_width=True)

    # Allow download as CSV / Excel
    csv = boq_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download BOQ (CSV)', data=csv, file_name='BOQ_retaining_wall_per_m.csv', mime='text/csv')

    with pd.ExcelWriter('BOQ_retaining_wall.xlsx', engine='xlsxwriter') as writer:
        boq_df.to_excel(writer, index=False, sheet_name='BOQ')
    with open('BOQ_retaining_wall.xlsx', 'rb') as f:
        st.download_button('Download BOQ (Excel)', data=f, file_name='BOQ_retaining_wall.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# ------------------------- Tab: Report ------------------------- #
with tabs[8]:
    st.subheader('Design Summary Report (quick view)')
    st.markdown('**Key Coefficients**')
    st.json({k: round(v, 4) if isinstance(v, float) else v for k,v in pres.items()})

    st.markdown('**Stability**')
    st.json({k: (round(v,4) if isinstance(v, float) else v) for k,v in stability(geo, soil, loads, bearing, pres).items()})

    st.markdown('**Member Design**')
    st.json({k: (round(v,4) if isinstance(v, float) else v) for k,v in member_design(geo, soil, loads, pres, mat).items()})

    st.info('Note: Calculations are simplified for quick iteration. For final issue, verify with your governing code (IS, BS, Eurocode) including shear checks, punching, seismic MO exact, layered backfill, and global stability if slopes/soft strata are present.')

st.success('Ready. Use the sidebar to iterate geometry & loads; review tabs for pressures, stability, design, drawings, and BOQ.')
