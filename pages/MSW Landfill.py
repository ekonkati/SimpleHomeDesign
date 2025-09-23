
import streamlit as st
import pandas as pd
import math
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
from io import BytesIO

st.set_page_config(page_title="Landfill Levels (BBL/BL/ABL Berms)", layout="wide")
st.title("Landfill Levels – BBL, BL, and ABL (with Berms)")

# --------------- Helpers ---------------
EARTH_R = 6371000.0

def vh_to_h_per_v(vh: str) -> float:
    try:
        v, h = vh.split(":")
        return float(h) / float(v)
    except Exception:
        return 3.0

def frustum_volume(A1: float, A2: float, H: float) -> float:
    return H/3.0 * (A1 + A2 + math.sqrt(max(A1*A2, 0.0)))

def parse_kml_polygon(file_bytes: bytes) -> Optional[List[Tuple[float,float]]]:
    try:
        tree = ET.fromstring(file_bytes)
    except Exception:
        return None
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    coords_elems = tree.findall(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)
    if not coords_elems:
        coords_elems = tree.findall(".//Polygon/outerBoundaryIs/LinearRing/coordinates")
        if not coords_elems:
            return None
    text = coords_elems[0].text or ""
    pts = []
    for token in text.replace("\\n", " ").split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0]); lat = float(parts[1])
                pts.append((lat, lon))
            except Exception:
                pass
    return pts if len(pts) >= 3 else None

def latlon_to_xy(lat, lon, lat0, lon0):
    x = (math.radians(lon - lon0) * math.cos(math.radians(lat0))) * EARTH_R
    y = (math.radians(lat - lat0)) * EARTH_R
    return x, y

def polygon_area_perimeter_xy(xy_pts: List[Tuple[float,float]]) -> Tuple[float,float]:
    if xy_pts[0] != xy_pts[-1]:
        pts = xy_pts + [xy_pts[0]]
    else:
        pts = xy_pts
    area2 = 0.0
    perim = 0.0
    for i in range(len(pts)-1):
        x1,y1 = pts[i]; x2,y2 = pts[i+1]
        area2 += (x1*y2 - x2*y1)
        perim += math.hypot(x2-x1, y2-y1)
    return abs(area2)/2.0, perim

def kml_to_area_perimeter(file) -> Tuple[float, float]:
    latlon = parse_kml_polygon(file.read())
    if not latlon:
        return 0.0, 0.0
    lat0 = sum(p[0] for p in latlon)/len(latlon)
    lon0 = sum(p[1] for p in latlon)/len(latlon)
    xy = [latlon_to_xy(lat, lon, lat0, lon0) for lat,lon in latlon]
    A, P = polygon_area_perimeter_xy(xy)
    return A, P

def rect_from_A_P(A: float, P: float) -> Tuple[float, float]:
    # Solve for rectangle L,W given area A and perimeter P:
    # L+W = P/2 = S, L*W = A. W = S/2 - sqrt((S/2)^2 - A)   and L = S - W
    S = P/2.0
    disc = (S/2.0)**2 - A
    if disc < 0:
        # fallback: square
        s = math.sqrt(max(A, 0.0))
        return s, s
    root = math.sqrt(disc)
    W = S/2.0 - root
    L = S - W
    if L < W:
        L, W = W, L
    return L, W

def fmt(x, unit=""):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    if unit:
        return f"{x:,.2f} {unit}"
    return f"{x:,.2f}"

# --------------- Sidebar Inputs ---------------
with st.sidebar:
    st.header("Input Mode")
    mode = st.radio("Footprint source", ["Length & Width", "Area & Aspect", "KML Polygon"])

    st.header("Geometry")
    if mode == "Length & Width":
        L_bl = st.number_input("BL Length (m)", 10.0, value=567.0, step=1.0)
        W_bl = st.number_input("BL Width (m)", 10.0, value=566.15, step=1.0)
    elif mode == "Area & Aspect":
        A_bl = st.number_input("BL Area (m²)", 1000.0, value=321008.457, step=1000.0, format="%.3f")
        aspect = st.slider("Assumed BL L:W", 1.0, 5.0, 1.0, 0.05)
        W_bl = math.sqrt(A_bl/aspect); L_bl = aspect*W_bl
    else:
        kml = st.file_uploader("Upload KML (BL footprint)", type=["kml"])
        L_bl = W_bl = 0.0
        if kml is not None:
            A, P = kml_to_area_perimeter(kml)
            if A > 0 and P > 0:
                L_bl, W_bl = rect_from_A_P(A, P)
            else:
                st.error("Could not derive area/perimeter from KML.")
    A_bl = L_bl * W_bl
    P_bl = 2*(L_bl + W_bl)

    st.header("Levels & Slopes")
    depth_bg = st.number_input("BBL depth (m)", 0.0, value=7.0, step=0.5)
    exc_slope = st.selectbox("Excavation side slope (V:H)", ["1:1", "1:1.5", "1:2", "1:2.5", "1:3"], index=4)
    m_exc = vh_to_h_per_v(exc_slope)

    lift_h = st.number_input("Lift height (m)", 1.0, value=5.0, step=0.5)
    inside_slope = st.selectbox("Inside slope (V:H)", ["1:2", "1:2.5", "1:3"], index=2)
    m_in = vh_to_h_per_v(inside_slope)
    bench_w = st.number_input("Bench width (m, per side)", 0.0, value=4.0, step=0.5)
    top_area_cap_pct = st.slider("Stop when top area ≤ (% of BL area)", 5, 100, 30, 5)

    st.header("Operations")
    density = st.number_input("Density (t/m³)", 0.3, value=1.0, step=0.05)
    tpa = st.number_input("Waste generation (TPA)", 1000.0, value=365000.0, step=5000.0)

# --------------- Compute Levels ---------------
if L_bl <= 0 or W_bl <= 0:
    st.warning("Provide a valid BL footprint.")
    st.stop()

rows = []
cum_vol = 0.0

# BBL dimensions from excavation slope & depth
L_bbl = max(L_bl - 2*m_exc*depth_bg, 0.0)
W_bbl = max(W_bl - 2*m_exc*depth_bg, 0.0)
A_bbl = max(L_bbl*W_bbl, 0.0)
V_bbl = frustum_volume(A_bbl, A_bl, depth_bg) if depth_bg > 0 else 0.0
cum_vol += V_bbl
rows.append({
    "Level": "BBL",
    "Length (m)": L_bbl,
    "Width (m)": W_bbl,
    "Area (sqm)": A_bbl,
    "Height (m)": depth_bg if depth_bg>0 else None,
    "Volume (Cum)": V_bbl
})

# BL row (reference, no height/volume)
rows.append({
    "Level": "BL",
    "Length (m)": L_bl,
    "Width (m)": W_bl,
    "Area (sqm)": A_bl,
    "Height (m)": None,
    "Volume (Cum)": None
})

# Iterative ABL with berms
L_prev_berm = L_bl
W_prev_berm = W_bl
A_prev_berm = L_prev_berm * W_prev_berm

i = 1
while True:
    # slope up one lift
    L_i = max(L_prev_berm - 2*m_in*lift_h, 0.0)
    W_i = max(W_prev_berm - 2*m_in*lift_h, 0.0)
    A_i = max(L_i*W_i, 0.0)

    if A_i <= 0:
        break

    # stop if top area cap reached next (will still show the last ABL row)
    if A_i <= (A_bl * top_area_cap_pct/100.0):
        stop_after = True
    else:
        stop_after = False

    # Volume for this lift = between previous berm area and ABL i area, over lift_h
    V_i = frustum_volume(A_prev_berm, A_i, lift_h) if lift_h>0 else 0.0
    cum_vol += V_i

    # Add ABL i (no volume shown in this row to match screenshot)
    rows.append({
        "Level": f"ABL {i}",
        "Length (m)": L_i,
        "Width (m)": W_i,
        "Area (sqm)": A_i,
        "Height (m)": lift_h,
        "Volume (Cum)": None
    })

    # Berm at top of lift (zero-height), record the volume on this row
    L_berm = max(L_i - 2*bench_w, 0.0)
    W_berm = max(W_i - 2*bench_w, 0.0)
    A_berm = max(L_berm*W_berm, 0.0)

    rows.append({
        "Level": f"ABL {i} Berm",
        "Length (m)": L_berm,
        "Width (m)": W_berm,
        "Area (sqm)": A_berm,
        "Height (m)": lift_h,  # display height here like the screenshot
        "Volume (Cum)": V_i
    })

    # Prepare for next lift
    L_prev_berm, W_prev_berm, A_prev_berm = L_berm, W_berm, A_berm
    if stop_after:
        break
    i += 1
    if i > 100:
        break

df = pd.DataFrame(rows)

# Totals
total_height = (depth_bg if depth_bg>0 else 0.0) + max(0, i)*lift_h
total_volume = cum_vol
total_tons = total_volume * density
years = (total_tons / tpa) if tpa>0 else None

# Display table with formatting similar to screenshot
def _fmt(x, unit=""):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    if unit:
        return f"{x:,.2f} {unit}"
    return f"{x:,.2f}"

display_df = df.copy()
for col in ["Length (m)", "Width (m)"]:
    display_df[col] = display_df[col].map(lambda x: _fmt(x, "m"))
display_df["Area (sqm)"] = display_df["Area (sqm)"].map(lambda x: _fmt(x, "sqm"))
display_df["Height (m)"] = display_df["Height (m)"].map(lambda x: _fmt(x, "m") if x is not None else "")
display_df["Volume (Cum)"] = display_df["Volume (Cum)"].map(lambda x: _fmt(x, "Cum") if x is not None else "")

st.dataframe(display_df, use_container_width=True, hide_index=True)

# Totals panel
st.markdown("### Totals")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Height", _fmt(total_height, "m"))
c2.metric("Total Volume", _fmt(total_volume, "Cum"))
c3.metric("Density", _fmt(density, "Ton/cum"))
c4.metric("Total Quantity", f"{total_tons:,.2f} Tons")

c5, c6 = st.columns(2)
c5.metric("Waste Generation (TPA)", f"{tpa:,.0f} TPA")
c6.metric("Duration", f"{years:,.2f} Years" if years is not None else "")

# Export
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="LevelsRaw", index=False)
    display_df.to_excel(writer, sheet_name="LevelsFormatted", index=False)
    pd.DataFrame({
        "Total_Height_m":[total_height],
        "Total_Volume_Cum":[total_volume],
        "Density_t_per_m3":[density],
        "Total_Quantity_Tons":[total_tons],
        "TPA":[tpa],
        "Duration_Years":[years if years is not None else float('nan')]
    }).to_excel(writer, sheet_name="Totals", index=False)

st.download_button("Download Excel (Levels + Totals)", buffer.getvalue(),
                   file_name="landfill_levels_bbl_bl_abl.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Notes: BL = Bund Level (starting footprint). BBL computed using excavation slope & depth. "
           "Each ABL lift shows two rows: the sloped top (ABL n) and the post-berm plan (ABL n Berm). "
           "Lift volume is recorded against the Berm row to match your Excel layout.")
