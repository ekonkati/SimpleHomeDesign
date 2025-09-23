
import streamlit as st
import pandas as pd
import math
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
from io import BytesIO

st.set_page_config(page_title="Landfill Levels – Exact Offsets", layout="wide")
st.title("Landfill Levels – Exact BBL/BL/ABL Offsets (Length/Width & KML)")

# ---------------- Helpers ----------------
EARTH_R = 6371000.0

def vh_to_h_per_v(vh: str) -> float:
    try:
        v,h = vh.split(":")
        return float(h)/float(v)
    except Exception:
        return 3.0

def frustum_volume(A1: float, A2: float, H: float) -> float:
    return H/3.0 * (A1 + A2 + math.sqrt(max(A1*A2,0.0)))

def parse_kml_polygon(file_bytes: bytes):
    try:
        root = ET.fromstring(file_bytes)
    except Exception:
        return None
    ns = {"kml":"http://www.opengis.net/kml/2.2"}
    coords = root.findall(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)
    if not coords:
        coords = root.findall(".//Polygon/outerBoundaryIs/LinearRing/coordinates")
        if not coords: return None
    text = (coords[0].text or "").strip()
    pts = []
    for tok in text.replace("\n"," ").split():
        parts = tok.split(",")
        if len(parts)>=2:
            try:
                lon=float(parts[0]); lat=float(parts[1]); pts.append((lat,lon))
            except: pass
    return pts if len(pts)>=3 else None

def latlon_to_xy(lat, lon, lat0, lon0):
    x = (math.radians(lon-lon0)*math.cos(math.radians(lat0)))*EARTH_R
    y = (math.radians(lat-lat0))*EARTH_R
    return x,y

def polygon_area_perimeter_xy(pts):
    if pts[0]!=pts[-1]: pts = pts+[pts[0]]
    area2=0.0; perim=0.0
    for i in range(len(pts)-1):
        x1,y1 = pts[i]; x2,y2 = pts[i+1]
        area2 += x1*y2 - x2*y1
        perim += math.hypot(x2-x1, y2-y1)
    return abs(area2)/2.0, perim

def eq_rect_from_A_P(A: float, P: float):
    S = P/2.0
    disc = (S/2.0)**2 - A
    if disc < 0:
        s = math.sqrt(max(A,0.0))
        return s,s
    root = math.sqrt(disc)
    W = S/2.0 - root
    L = S - W
    if L < W: L,W = W,L
    return L,W

def fmt(x, unit=""):
    if x is None: return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return ""
    return f"{x:,.2f} {unit}" if unit else f"{x:,.2f}"

# ---------------- Inputs ----------------
with st.sidebar:
    st.header("Footprint Source")
    mode = st.radio("Use", ["Length & Width", "Area & Aspect", "KML Polygon"])
    if mode == "Length & Width":
        L0 = st.number_input("Site overall Length at GL (m)", 1.0, value=600.0, step=1.0)
        W0 = st.number_input("Site overall Width at GL (m)",  1.0, value=400.0, step=1.0)
    elif mode == "Area & Aspect":
        A0 = st.number_input("Site overall Area at GL (m²)", 1000.0, value=240000.0, step=1000.0)
        asp = st.slider("Assumed L:W", 1.0, 5.0, 1.5, 0.1)
        W0 = math.sqrt(A0/asp); L0 = asp*W0
    else:
        kml = st.file_uploader("Upload KML (GL outer footprint)", type=["kml"])
        L0=W0=0.0
        if kml is not None:
            latlon = parse_kml_polygon(kml.read())
            if latlon:
                lat0 = sum(p[0] for p in latlon)/len(latlon)
                lon0 = sum(p[1] for p in latlon)/len(latlon)
                xy = [latlon_to_xy(lat,lon,lat0,lon0) for lat,lon in latlon]
                A,P = polygon_area_perimeter_xy(xy)
                L0,W0 = eq_rect_from_A_P(A,P)
            else:
                st.error("Could not parse polygon from KML.")

    st.header("Bund & Ancillary Offsets (per side)")
    bund_top_w = st.number_input("Bund top width (m)", 0.0, value=5.0, step=0.5)
    bund_out_vert = st.number_input("Bund OUTER vertical height (m)", 0.0, value=2.0, step=0.5)
    bund_out_slope = st.selectbox("Bund OUTER slope (V:H)", ["1:1","1:2","1:3","1:4"], index=3)
    m_out = vh_to_h_per_v(bund_out_slope)

    bund_in_vert = st.number_input("Bund INNER vertical height (m)", 0.0, value=4.0, step=0.5)
    bund_in_slope = st.selectbox("Bund INNER slope (V:H)", ["1:2","1:2.5","1:3"], index=2)
    m_in_bund = vh_to_h_per_v(bund_in_slope)

    drain_w = st.number_input("Drain width (m)", 0.0, value=1.0, step=0.5)
    gap_w   = st.number_input("Gap between drain & bund (m)", 0.0, value=1.0, step=0.5)

    st.header("Below-Ground Excavation")
    depth_bg = st.number_input("Depth below GL (m)", 0.0, value=2.0, step=0.5)
    exc_slope = st.selectbox("Excavation side slope (V:H)", ["1:2","1:2.5","1:3"], index=2)
    m_exc = vh_to_h_per_v(exc_slope)

    st.header("ABL (Above BL) Lifts")
    lift_h = st.number_input("Lift height (m)", 0.5, value=5.0, step=0.5)
    inside_slope = st.selectbox("Inside slope above BL (V:H)", ["1:2","1:2.5","1:3"], index=2)
    m_inside = vh_to_h_per_v(inside_slope)
    berm_w_per_side = st.number_input("Berm width (m) per side", 0.0, value=4.0, step=0.5)
    max_lifts = st.number_input("Max number of lifts", 1, value=10, step=1)
    top_limit_pct = st.slider("Stop when top area ≤ (% of BL area)", 5, 100, 30, 5)

    st.header("Ops")
    density = st.number_input("Density (t/m³)", 0.3, value=1.0, step=0.05)
    tpa     = st.number_input("Waste generation (TPA)", 1000.0, value=365000.0, step=5000.0)

# Validate footprint
if L0<=0 or W0<=0:
    st.warning("Provide a valid outer GL footprint (Length/Width or KML-derived).")
    st.stop()

# ---------------- Exact Offset Logic ----------------
bund_base_w_per_side = bund_top_w + m_out*bund_out_vert + m_in_bund*bund_in_vert

def bbl_dim(D0: float) -> float:
    return max(D0 - 2*(bund_base_w_per_side + depth_bg*m_exc + drain_w + gap_w), 0.0)

L_bbl = bbl_dim(L0)
W_bbl = bbl_dim(W0)
A_bbl = L_bbl * W_bbl

def bl_dim(D_bbl: float) -> float:
    return max(D_bbl + 2*((bund_in_vert + depth_bg) * m_exc), 0.0)

L_bl = bl_dim(L_bbl)
W_bl = bl_dim(W_bbl)
A_bl = L_bl * W_bl

rows = []
cumV = 0.0

V_bbl = frustum_volume(A_bbl, A_bl, depth_bg) if depth_bg>0 else 0.0
cumV += V_bbl
rows.append({"Level":"BBL","Length (m)":L_bbl,"Width (m)":W_bbl,"Area (sqm)":A_bbl,"Height (m)":depth_bg if depth_bg>0 else None,"Volume (Cum)":V_bbl})
rows.append({"Level":"BL","Length (m)":L_bl,"Width (m)":W_bl,"Area (sqm)":A_bl,"Height (m)":None,"Volume (Cum)":None})

L_prev = L_bl; W_prev = W_bl; A_prev = A_bl

for n in range(1, max_lifts+1):
    L_n = max(L_prev - 2*(m_inside*lift_h), 0.0)
    W_n = max(W_prev - 2*(m_inside*lift_h), 0.0)
    A_n = L_n * W_n
    if A_n <= 0: break
    rows.append({"Level":f"ABL {n}","Length (m)":L_n,"Width (m)":W_n,"Area (sqm)":A_n,"Height (m)":lift_h,"Volume (Cum)":None})
    V_n = frustum_volume(A_prev, A_n, lift_h) if lift_h>0 else 0.0
    cumV += V_n
    L_berm = max(L_n - 2*berm_w_per_side, 0.0)
    W_berm = max(W_n - 2*berm_w_per_side, 0.0)
    A_berm = L_berm * W_berm
    rows.append({"Level":f"ABL {n} Berm","Length (m)":L_berm,"Width (m)":W_berm,"Area (sqm)":A_berm,"Height (m)":lift_h,"Volume (Cum)":V_n})
    if A_n <= (A_bl * top_limit_pct/100.0):
        break
    L_prev, W_prev, A_prev = L_berm, W_berm, A_berm

df = pd.DataFrame(rows)

total_height = (depth_bg if depth_bg>0 else 0.0) + (df["Level"].str.startswith("ABL ").sum())*lift_h
total_vol = cumV
total_tons = total_vol * density
duration_years = (total_tons / tpa) if tpa>0 else None

disp = df.copy()
for col in ["Length (m)","Width (m)"]:
    disp[col] = disp[col].map(lambda x: fmt(x,"m"))
disp["Area (sqm)"] = disp["Area (sqm)"].map(lambda x: fmt(x,"sqm"))
disp["Height (m)"] = disp["Height (m)"].map(lambda x: fmt(x,"m") if x is not None else "")
disp["Volume (Cum)"] = disp["Volume (Cum)"].map(lambda x: fmt(x,"Cum") if x is not None else "")

st.subheader("Level-wise Table")
st.dataframe(disp, use_container_width=True, hide_index=True)

st.subheader("Totals")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Height", fmt(total_height,"m"))
c2.metric("Total Volume", fmt(total_vol,"Cum"))
c3.metric("Density", fmt(density,"Ton/cum"))
c4.metric("Total Quantity", f"{total_tons:,.2f} Tons")
c5,c6 = st.columns(2)
c5.metric("Waste Generation (TPA)", f"{tpa:,.0f} TPA")
c6.metric("Duration", f"{duration_years:,.2f} Years" if duration_years is not None else "")

buf = BytesIO()
with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="LevelsRaw", index=False)
    disp.to_excel(writer, sheet_name="LevelsFormatted", index=False)
    pd.DataFrame({
        "Total_Height_m":[total_height],
        "Total_Volume_Cum":[total_vol],
        "Density_t_per_m3":[density],
        "Total_Quantity_Tons":[total_tons],
        "TPA":[tpa],
        "Duration_Years":[duration_years if duration_years is not None else float('nan')],
        "L0_m":[L0],"W0_m":[W0],
        "BundTop_m":[bund_top_w],
        "BundOutVert_m":[bund_out_vert],"BundOutSlope_H_per_V":[m_out],
        "BundInVert_m":[bund_in_vert],"BundInSlope_H_per_V":[m_in_bund],
        "Drain_m":[drain_w],"Gap_m":[gap_w],
        "DepthBG_m":[depth_bg],"ExcSlope_H_per_V":[m_exc],
        "LiftH_m":[lift_h],"InsideSlope_H_per_V":[m_inside],
        "Berm_m_per_side":[berm_w_per_side],
        "TopLimit_pct_of_BL_area":[top_limit_pct]
    }).to_excel(writer, sheet_name="Inputs&Totals", index=False)

st.download_button("Download Excel (Levels + Totals + Inputs)", buf.getvalue(),
                   file_name="landfill_levels_exact_offsets.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Formulas used:\n"
           "BBL = GL outer − 2 × [ BundBase + (depth×m_exc) + drain + gap ], "
           "where BundBase = BundTop + (m_out×BundOuterVert) + (m_in_bund×BundInnerVert).\n"
           "BL  = BBL + 2 × [ (BundInnerVert + depth) × m_exc ].\n"
           "ABLₙ = prev_berm − 2 × (m_inside × lift_h).  ABLₙ Berm = ABLₙ − 2 × berm_width_per_side.\n"
           "Lift volume recorded on the Berm row: V = H/3 × (A_prev + A_curr + √(A_prev×A_curr)).")
