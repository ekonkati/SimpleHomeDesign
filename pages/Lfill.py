import streamlit as st
import pandas as pd
import math
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict 
from io import BytesIO

# --- Optional Imports for Plotting & Geometry ---
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
    
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    PLOTLY_AVAILABLE = False
    
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False
    
try:
    from shapely.geometry import Polygon, MultiPolygon
    SHAPELY_AVAILABLE = True
except ImportError:
    Polygon = None
    MultiPolygon = None
    SHAPELY_AVAILABLE = False

st.set_page_config(page_title="Landfill Levels (BBL/BL/ABL Berms)", layout="wide")
st.title("Landfill Levels – BBL, BL, and ABL (with Berms)")

# --------------- Helpers ---------------
EARTH_R = 6371000.0

def vh_to_h_per_v(vh: str) -> float:
    """Converts a V:H slope string (e.g., '1:3') to H/V (e.g., 3.0)."""
    try:
        v, h = vh.split(":")
        return float(h) / float(v)
    except Exception:
        return 3.0

def frustum_volume(A1: float, A2: float, H: float) -> float:
    # Formula for frustum volume: H/3 * (A1 + A2 + sqrt(A1*A2))
    return H/3.0 * (A1 + A2 + math.sqrt(max(A1*A2, 0.0)))

def rectangle_polygon(width: float, length: float) -> List[Tuple[float, float]]:
    """Generates 5 XY points for a closed rectangular footprint centered at (0,0)."""
    half_w, half_l = width / 2.0, length / 2.0
    return [(-half_w, -half_l), (half_w, -half_l), (half_w, half_l), 
            (-half_w, half_l), (-half_w, -half_l)]

# --- Geometry Functions for Polygon Mode (KML) ---

def polygon_buffer(xy_coords: List[Tuple[float, float]], offset_dist: float) -> Optional[List[Tuple[float, float]]]:
    """Generates a new polygon by offsetting (buffering) the original xy_coords using Shapely."""
    if not SHAPELY_AVAILABLE or len(xy_coords) < 3:
        return None
        
    try:
        # Ensure the polygon is closed for Shapely (remove closing point if present for internal geometry)
        if xy_coords and xy_coords[0] == xy_coords[-1]:
            poly = Polygon(xy_coords[:-1])
        else:
            poly = Polygon(xy_coords)

        # Negative offset for shrinking, positive for expanding.
        # join_style=2 is mitered join
        new_poly = poly.buffer(offset_dist, join_style=2, mitre_limit=5.0) 
        
        if isinstance(new_poly, MultiPolygon):
            if not new_poly.geoms: return None
            new_poly = max(new_poly.geoms, key=lambda p: p.area) 
        
        if new_poly.area < 1e-6: return None
           
        # Extract the exterior coordinates and close the loop for plotting
        new_coords = list(new_poly.exterior.coords)
        if new_coords and new_coords[0] != new_coords[-1]:
            new_coords.append(new_coords[0])
            
        return new_coords
    except Exception:
        return None

def parse_kml_polygon(file_bytes: bytes) -> Optional[List[Tuple[float,float]]]:
    """Parses KML file bytes to extract (Lat, Lon) coordinates."""
    try:
        tree = ET.fromstring(file_bytes)
    except Exception:
        return None
    
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    # Standard KML path or fallback without namespace
    coords_elems = tree.findall(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)
    if not coords_elems:
        coords_elems = tree.findall(".//Polygon/outerBoundaryIs/LinearRing/coordinates")
        if not coords_elems:
            return None
            
    text = coords_elems[0].text or ""
    # Replace the escaped newline that sometimes occurs
    pts = []
    for token in text.replace("\n", " ").replace("\\n", " ").split(): 
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                # KML uses Lon, Lat, Alt. We need Lat, Lon
                lon = float(parts[0])
                lat = float(parts[1])
                pts.append((lat, lon))
            except Exception:
                pass
          
    return pts if len(pts) >= 3 else None

def latlon_to_xy(lat, lon, lat0, lon0):
    """Converts Lat/Lon to XY coordinates (meters) relative to a centroid (lat0, lon0)."""
    x = (math.radians(lon - lon0) * math.cos(math.radians(lat0))) * EARTH_R
    y = (math.radians(lat - lat0)) * EARTH_R
    return x, y

def polygon_area_perimeter_xy(xy_pts: List[Tuple[float,float]]) -> Tuple[float,float]:
    """Calculates Area and Perimeter from XY coordinates using the Shoelace formula and distance formula."""
    if not xy_pts: return 0.0, 0.0
    
    # Ensure the list is closed for calculation by duplicating the first point
    pts = xy_pts
    if pts[0] != pts[-1]:
        pts = xy_pts + [xy_pts[0]]

    area2 = 0.0
    perim = 0.0
    for i in range(len(pts)-1):
        x1,y1 = pts[i]
        x2,y2 = pts[i+1]
        area2 += (x1*y2 - x2*y1)
        perim += math.hypot(x2-x1, y2-y1)
    return abs(area2)/2.0, perim

def kml_to_area_perimeter_and_xy(file) -> Tuple[float, float, Optional[List[Tuple[float, float]]]]:
    """Returns Area, Perimeter, and XY coordinates from KML."""
    file.seek(0)
    latlon = parse_kml_polygon(file.read())
    if not latlon:
        return 0.0, 0.0, None
        
    lat0 = sum(p[0] for p in latlon)/len(latlon)
    lon0 = sum(p[1] for p in latlon)/len(latlon)
    xy = [latlon_to_xy(lat, lon, lat0, lon0) for lat,lon in latlon]
    
    A, P = polygon_area_perimeter_xy(xy)
    
    # Ensure the XY list is closed for plotting (first point equals last point)
    if xy and xy[0] != xy[-1]:
        xy.append(xy[0])
        
    return A, P, xy

def rect_from_A_P(A: float, P: float) -> Tuple[float, float]:
    """Calculates Length and Width of an equivalent rectangle from Area and Perimeter."""
    S = P/2.0
    disc = (S/2.0)**2 - A
    if disc < 0:
        s = math.sqrt(max(A, 0.0))
        return s, s
    root = math.sqrt(disc)
    W = S/2.0 - root
    L = S - W
    if L < W: L, W = W, L
    return L, W

def fmt(x, unit=""):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    if unit:
        return f"{x:,.2f} {unit}"
    return f"{x:,.2f}"

# --------------- Sidebar Inputs ---------------
if 'bl_coords_xy' not in st.session_state:
    st.session_state.bl_coords_xy = []

with st.sidebar:
    st.header("Input Mode")
    mode = st.radio("Footprint source", ["Length & Width", "Area & Aspect", "KML Polygon"])
    
    if mode == "KML Polygon" and not SHAPELY_AVAILABLE:
        st.error("⚠️ **Shapely** library is required for KML polygon modeling. Defaulting to rectangular approximation.")
        
    avg_ground_rl = st.number_input("Avg. ground RL (m)", 0.0, value=100.0, step=1.0)

    st.header("Geometry")
    
    L_bl, W_bl, A_bl = 0.0, 0.0, 0.0
    st.session_state.bl_coords_xy = []
    
    if mode == "Length & Width":
        L_bl = st.number_input("BL Length (m)", 10.0, value=567.0, step=1.0)
        W_bl = st.number_input("BL Width (m)", 10.0, value=566.15, step=1.0)
    elif mode == "Area & Aspect":
        A_bl = st.number_input("BL Area (m²)", 1000.0, value=321008.457, step=1000.0, format="%.3f")
        aspect = st.slider("Assumed BL L:W", 1.0, 5.0, 1.0, 0.05)
        W_bl = math.sqrt(A_bl/aspect)
        L_bl = aspect*W_bl
    else: # KML Polygon
        kml = st.file_uploader("Upload KML (BL footprint)", type=["kml"])
        if kml is not None and SHAPELY_AVAILABLE:
            A_bl, P_bl, xy_coords = kml_to_area_perimeter_and_xy(kml)
            if A_bl > 0:
                L_bl, W_bl = rect_from_A_P(A_bl, P_bl)
                st.session_state.bl_coords_xy = xy_coords
            else:
                st.error("Could not derive area/perimeter from KML.")
        elif kml is not None and not SHAPELY_AVAILABLE:
            st.warning("Cannot process KML shape without Shapely. Using rectangular L/W approximation.")
            
    # Finalize BL footprint and coords
    if mode != "KML Polygon" or not st.session_state.bl_coords_xy:
        A_bl = L_bl * W_bl
        P_bl = 2*(L_bl + W_bl)
        # Generate rectangular corners for all non-KML modes and as a fallback
        st.session_state.bl_coords_xy = rectangle_polygon(W_bl, L_bl) if A_bl > 0 else []

    st.header("Levels & Slopes")
    depth_bg = st.number_input("BBL depth (m)", 0.0, value=7.0, step=0.5)
    exc_slope = st.selectbox("Excavation side slope (V:H)", ["1:1", "1:1.5", "1:2", "1:2.5", "1:3"], index=4)
    m_exc = vh_to_h_per_v(exc_slope)

    # NEW INPUTS FOR 2D PLOT FIX
    outer_slope = st.selectbox("Outer slope (V:H) (Bund)", ["1:2", "1:2.5", "1:3", "1:4"], index=2)
    m_out = vh_to_h_per_v(outer_slope)
    crest_w = st.number_input("Crest Width (m, per side)", 0.0, value=2.0, step=0.5)
    # END NEW INPUTS

    lift_h = st.number_input("Lift height (m)", 1.0, value=5.0, step=0.5)
    inside_slope = st.selectbox("Inside slope (V:H)", ["1:2", "1:2.5", "1:3"], index=2)
    m_in = vh_to_h_per_v(inside_slope)
    bench_w = st.number_input("Bench width (m, per side)", 0.0, value=4.0, step=0.5)
    top_area_cap_pct = st.slider("Stop when top area ≤ (% of BL area)", 5, 100, 30, 5)
    
    st.header("Operations")
    density = st.number_input("Density (t/m³)", 0.3, value=1.0, step=0.05)
    tpa = st.number_input("Waste generation (TPA)", 1000.0, value=365000.0, step=5000.0)
    
    st.header("Visualization Options")
    cross_section_axis = st.radio("2D Section View:", ["Width (W)", "Length (L)"])

# --------------- Compute Levels ---------------
if L_bl <= 0 or W_bl <= 0:
    st.warning("Provide a valid BL footprint.")
    st.stop()

rows = []
cum_vol = 0.0
is_polygon_mode = mode == "KML Polygon" and SHAPELY_AVAILABLE and st.session_state.bl_coords_xy

# --- BBL CALCULATION ---
xy_bl = st.session_state.bl_coords_xy

if is_polygon_mode:
    offset_bbl = m_exc * depth_bg 
    xy_bbl = polygon_buffer(xy_bl, -offset_bbl) # Negative offset to shrink
    A_bbl, P_bbl = polygon_area_perimeter_xy(xy_bbl) if xy_bbl else (0.0, 0.0)
    L_bbl, W_bbl = rect_from_A_P(A_bbl, P_bbl)
    
else: # Rectangular Mode
    L_bbl = max(L_bl - 2*m_exc*depth_bg, 0.0)
    W_bbl = max(W_bl - 2*m_exc*depth_bg, 0.0)
    A_bbl = max(L_bbl*W_bbl, 0.0)
    xy_bbl = rectangle_polygon(W_bbl, L_bbl) if A_bbl > 0 else []

V_bbl = frustum_volume(A_bbl, A_bl, depth_bg) if depth_bg > 0 else 0.0
cum_vol += V_bbl
rows.append({
    "Level": "BBL", "Length (m)": L_bbl, "Width (m)": W_bl, "Area (sqm)": A_bbl,
    "Height (m)": depth_bg if depth_bg > 0 else None, "Volume (Cum)": V_bbl, 
    "Coords_XY": xy_bbl, "Lift_ID": 0 # BBL is lift 0
})

# BL row (reference, Z_rel=0)
rows.append({
    "Level": "BL", "Length (m)": L_bl, "Width (m)": W_bl, "Area (sqm)": A_bl,
    "Height (m)": None, "Volume (Cum)": None, "Coords_XY": xy_bl, "Lift_ID": -1 # Use -1 for BL as separator
})

# --- ABL ITERATION ---
if is_polygon_mode:
    xy_prev_berm = xy_bl
    A_prev_berm = A_bl
else:
    L_prev_berm = L_bl; W_prev_berm = W_bl; A_prev_berm = A_bl
    
i = 1 # Lift ID starts at 1 for ABL lifts
while True:
    
    if is_polygon_mode:
        # 1. Slope up one lift
        offset_abl = m_in * lift_h 
        xy_i = polygon_buffer(xy_prev_berm, -offset_abl)
        A_i, P_i = polygon_area_perimeter_xy(xy_i) if xy_i else (0.0, 0.0)
        L_i, W_i = rect_from_A_P(A_i, P_i)
        
        # 2. Berm at top of lift
        offset_berm = bench_w 
        xy_berm = polygon_buffer(xy_i, -offset_berm)
        A_berm, P_berm = polygon_area_perimeter_xy(xy_berm) if xy_berm else (0.0, 0.0)
        L_berm, W_berm = rect_from_A_P(A_berm, P_berm)
        
    else:
        # 1. Slope up one lift (Rectangular)
        L_i = max(L_prev_berm - 2*m_in*lift_h, 0.0)
        W_i = max(W_prev_berm - 2*m_in*lift_h, 0.0)
        A_i = max(L_i*W_i, 0.0)
        xy_i = rectangle_polygon(W_i, L_i) if A_i > 0 else []
        
        # 2. Berm at top of lift (Rectangular)
        L_berm = max(L_i - 2*bench_w, 0.0)
        W_berm = max(W_i - 2*bench_w, 0.0)
        A_berm = max(L_berm*W_berm, 0.0)
        xy_berm = rectangle_polygon(W_berm, L_berm) if A_berm > 0 else []
        
    if A_i <= 0 or A_berm <= 0:
        break

    stop_after = False
    if A_i <= (A_bl * top_area_cap_pct/100.0):
        stop_after = True

    V_i = frustum_volume(A_prev_berm, A_i, lift_h) if lift_h > 0 else 0.0
    cum_vol += V_i

    # ABL i (sloped top)
    rows.append({
        "Level": f"ABL {i}", "Length (m)": L_i, "Width (m)": W_i, "Area (sqm)": A_i,
        "Height (m)": lift_h, "Volume (Cum)": None, "Coords_XY": xy_i, "Lift_ID": None
    })
    
    # Berm at top of lift
    rows.append({
        "Level": f"ABL {i} Berm", "Length (m)": L_berm, "Width (m)": W_berm, "Area (sqm)": A_berm,
        "Height (m)": lift_h, "Volume (Cum)": V_i, "Coords_XY": xy_berm, "Lift_ID": i
    })

    # Prepare for next lift
    if is_polygon_mode:
        xy_prev_berm = xy_berm
    else:
        L_prev_berm, W_prev_berm = L_berm, W_berm
        
    A_prev_berm = A_berm
    
    if stop_after:
        break
    i += 1
    if i > 100: break

df = pd.DataFrame(rows)

# Totals
total_height = (depth_bg if depth_bg > 0 else 0.0) + ((i-1) * lift_h) # Corrected height accumulation
total_volume = cum_vol
total_tons = total_volume * density
years = (total_tons / tpa) if tpa > 0 else None

st.subheader("Landfill Level Summary Table")

# Display table
def _fmt_display(x, unit=""):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)) or x == 0.0):
        return ""
    if unit:
        return f"{x:,.2f} {unit}"
    return f"{x:,.2f}"

display_df = df.copy()
display_df = display_df.drop(columns=['Coords_XY', 'Lift_ID'])
for col in ["Length (m)", "Width (m)"]:
    display_df[col] = display_df[col].map(lambda x: _fmt_display(x, "m"))
display_df["Area (sqm)"] = display_df["Area (sqm)"].map(lambda x: _fmt_display(x, "sqm"))
display_df["Height (m)"] = display_df["Height (m)"].map(lambda x: _fmt_display(x, "m") if x is not None else "")
display_df["Volume (Cum)"] = display_df["Volume (Cum)"].map(lambda x: _fmt_display(x, "Cum") if x is not None else "")

st.dataframe(display_df, use_container_width=True, hide_index=True)

# Totals panel
st.markdown("### Totals")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Height", fmt(total_height, "m"))
c2.metric("Total Volume", fmt(total_volume, "Cum"))
c3.metric("Density", fmt(density, "Ton/cum"))
c4.metric("Total Quantity", f"{total_tons:,.2f} Tons")

c5, c6 = st.columns(2)
c5.metric("Waste Generation (TPA)", f"{tpa:,.0f} TPA")
c6.metric("Duration", f"{years:,.2f} Years" if years is not None else "")

st.markdown("---")
st.subheader("2D and 3D Visualizations")

# ----------------- 3D Plotting Functions -----------------

def get_footprint_corners(W: float, L: float, Z_rel: float, RL_ref: float) -> List[Tuple[float, float, float]]:
    """Returns the 4 corner (X, Y, Z_abs) coordinates for a rectangular footprint."""
    half_W, half_L = W / 2.0, L / 2.0
    RL = Z_rel + RL_ref
    return [
        (-half_W, -half_L, RL), (half_W, -half_L, RL), 
        (half_W, half_L, RL), (-half_W, half_L, RL)
    ]

def frustum_mesh_rect(W1, L1, Z1_rel, W2, L2, Z2_rel, RL_ref):
    """Generates mesh vertices and faces for a single rectangular frustum section (waste profile)."""
    if not NUMPY_AVAILABLE: return [], [], [], []
    
    corners1 = get_footprint_corners(W1, L1, Z1_rel, RL_ref)
    corners2 = get_footprint_corners(W2, L2, Z2_rel, RL_ref)
    verts = corners1 + corners2 # 8 vertices in total

    faces = []
    # Side walls
    for i in range(4):
        i1, i2 = i, (i + 1) % 4
        i3, i4 = i + 4, (i + 1) % 4 + 4

        # Face 1 (Triangle 1): i1, i2, i4
        faces.append([i1, i2, i4])
        # Face 2 (Triangle 2): i1, i4, i3
        faces.append([i1, i4, i3])

    X = [v[0] for v in verts]
    Y = [v[1] for v in verts]
    Z_abs = [v[2] for v in verts]
    
    return X, Y, Z_abs, faces 

def top_surface_mesh_rect(W_top, L_top, Z_top_rel, RL_ref):
    """Generates a top mesh (4 vertices, 2 faces) for the final lift."""
    if not NUMPY_AVAILABLE: return [], [], [], []
    
    corners = get_footprint_corners(W_top, L_top, Z_top_rel, RL_ref)
    verts = corners # 4 vertices
    # Top face triangles (relative indices)
    faces = [[0, 1, 2], [0, 2, 3]] 
    
    X = [v[0] for v in verts]
    Y = [v[1] for v in verts]
    Z_abs = [v[2] for v in verts]
    return X, Y, Z_abs, faces

def bund_mesh_rect(W_bl, L_bl, W_out, L_out, H_out, RL_ref):
    """Generates mesh vertices and faces for the rectangular Outer Bund."""
    if not NUMPY_AVAILABLE: return [], [], [], []
    
    # Footprint 1: BL (Z_rel=0)
    corners1 = get_footprint_corners(W_bl, L_bl, 0.0, RL_ref)
    # Footprint 2: Outer Crest (Z_rel=H_out)
    corners2 = get_footprint_corners(W_out, L_out, H_out, RL_ref)
    
    verts = corners1 + corners2 # 8 vertices in total

    faces = []
    for i in range(4):
        i1, i2 = i, (i + 1) % 4
        i3, i4 = i + 4, (i + 1) % 4 + 4

        # Face 1 (Triangle 1) - Side wall
        faces.append([i1, i2, i4])
        # Face 2 (Triangle 2) - Side wall
        faces.append([i1, i4, i3])

    X = [v[0] for v in verts]
    Y = [v[1] for v in verts]
    Z_abs = [v[2] for v in verts]
    
    return X, Y, Z_abs, faces 


def plotly_3d_full_stack(df: pd.DataFrame, avg_ground_rl: float, total_height: float, m_out: float, crest_w: float, is_polygon_plot: bool):
    if not PLOTLY_AVAILABLE or not NUMPY_AVAILABLE: return go.Figure()
    
    RL_ref = avg_ground_rl 
    traces = []
    
    # Define a color palette for the individual lifts
    COLORS = ['#3366cc', '#dc3912', '#ff9900', '#109618', '#990099', '#0099c6', '#dd4477', '#66aa00', '#b82e2e', '#314798']
    
    # Filter to the distinct segments defined by the BBL level and ABL Berm levels
    # The segment is defined from (Level 1) to (Level 2/Berm). The volume and Lift_ID are on the Level 2/Berm row.
    df_geom = df[df['Level'] == 'BBL'].copy()
    df_geom = pd.concat([df_geom, df[df['Level'].str.endswith('Berm')]], ignore_index=True)
    df_geom.sort_values(by='Lift_ID', inplace=True)
    df_geom.reset_index(drop=True, inplace=True)
    
    # Create a list of levels that define the top and bottom of each frustum segment
    frustum_segments = []
    
    # BBL segment (Bottom is BBL, Top is BL)
    if not df_geom.empty and df_geom.iloc[0]['Lift_ID'] == 0:
        row_bbl = df_geom.iloc[0]
        row_bl = df[df['Level'] == 'BL'].iloc[0]
        frustum_segments.append({
            'bottom': row_bbl, 
            'top': row_bl, 
            'Z1_rel': -row_bbl['Height (m)'], 
            'Z2_rel': 0.0,
            'Lift_ID': 0 # BBL
        })

    # ABL segments (Bottom is previous Berm/BL, Top is current Berm)
    df_berms = df[df['Level'].str.endswith('Berm')].sort_values(by='Lift_ID')
    
    Z_cum_h = 0.0 # Height of previous berm (start at BL/GL)
    
    for i in range(len(df_berms)):
        row_current_berm = df_berms.iloc[i]
        
        if i == 0:
            row_bottom = df[df['Level'] == 'BL'].iloc[0]
        else:
            row_bottom = df_berms.iloc[i-1]
        
        Z_cum_h += row_current_berm['Height (m)'] # Total height of waste up to this berm's top
        
        frustum_segments.append({
            'bottom': row_bottom,
            'top': row_current_berm, 
            'Z1_rel': Z_cum_h - row_current_berm['Height (m)'], # Z level of the bench below
            'Z2_rel': Z_cum_h, # Z level of the current bench
            'Lift_ID': int(row_current_berm['Lift_ID'])
        })
    
    # --- 1. Waste Profile Meshes (Per Segment) ---
    for i, seg in enumerate(frustum_segments):
        
        Z1_rel = seg['Z1_rel']
        Z2_rel = seg['Z2_rel']
        lift_id = seg['Lift_ID']

        if Z2_rel <= Z1_rel: continue

        # Assign color based on Lift_ID
        if lift_id == 0: # BBL is lift 0
            color = 'rgba(0, 100, 200, 0.7)' # Darker blue for excavation pit
            name = "BBL Excavation Pit"
        else:
            color = COLORS[(lift_id - 1) % len(COLORS)]
            name = f"ABL Lift {lift_id}"

        # --- Geometry Calculation ---
        X_seg, Y_seg, Z_seg, faces_seg = [], [], [], []

        # Rectangular Mode Mesh Generation (Frustum side walls)
        W1, L1 = seg['bottom']['Width (m)'], seg['bottom']['Length (m)']
        W2, L2 = seg['top']['Width (m)'], seg['top']['Length (m)']
        X_seg, Y_seg, Z_seg, faces_seg = frustum_mesh_rect(W1, L1, Z1_rel, W2, L2, Z2_rel, RL_ref)


        if faces_seg:
            I_seg = [f[0] for f in faces_seg]
            J_seg = [f[1] for f in faces_seg]
            K_seg = [f[2] for f in faces_seg]
            
            traces.append(
                go.Mesh3d(
                    x=X_seg, y=Y_seg, z=Z_seg,
                    i=I_seg, j=J_seg, k=K_seg,
                    opacity=0.8, color=color, 
                    name=name, hoverinfo='name', showlegend=True
                )
            )
            
        # --- Add Top Surface Mesh for the Final Lift ---
        # The top surface is defined by the coordinates of the current berm (the top of the segment)
        if i == len(frustum_segments) - 1: # Final ABL lift segment
            W_top, L_top = seg['top']['Width (m)'], seg['top']['Length (m)']
            X_top, Y_top, Z_top, faces_top = top_surface_mesh_rect(W_top, L_top, Z2_rel, RL_ref)
            
            if faces_top:
                I_top = [f[0] for f in faces_top]
                J_top = [f[1] for f in faces_top]
                K_top = [f[2] for f in faces_top]
                
                traces.append(
                    go.Mesh3d(
                        x=X_top, y=Y_top, z=Z_top,
                        i=I_top, j=J_top, k=K_top,
                        opacity=0.8, color=color, # Same color as the final lift
                        name=f"ABL {lift_id} Final Surface", hoverinfo='name', showlegend=False
                    )
                )

    
    # --- 2. Outer Bund Mesh (Exterior Shell) ---
    bl_row = df[df['Level'] == 'BL'].iloc[0]
    W_bl_ref, L_bl_ref = bl_row[['Width (m)', 'Length (m)']]
    final_H_abl = total_height - (depth_bg if depth_bg > 0 else 0.0) # Total height of ABL lifts
    
    # Calculate outer crest dimensions
    W_out = W_bl_ref + 2 * m_out * final_H_abl + 2 * crest_w
    L_out = L_bl_ref + 2 * m_out * final_H_abl + 2 * crest_w
    
    # The Bund mesh will represent the area from the BL edge outward and up
    if not is_polygon_plot and final_H_abl > 0:
        
        X_bund, Y_bund, Z_bund, faces_bund = bund_mesh_rect(W_bl_ref, L_bl_ref, W_out, L_out, final_H_abl, RL_ref)
        
        I_bund = [f[0] for f in faces_bund]
        J_bund = [f[1] for f in faces_bund]
        K_bund = [f[2] for f in faces_bund]
        
        traces.append(
            go.Mesh3d(
                x=X_bund, y=Y_bund, z=Z_bund,
                i=I_bund, j=J_bund, k=K_bund,
                opacity=0.15, color='rgba(0, 0, 0, 0.2)', 
                name='Outer Bund (Exterior Shell)', hoverinfo='name', showlegend=True
            )
        )

    
    # --- 3. Ground Level Plane ---
    # Use the Bund's outer bounds for the ground plane extent if available, otherwise BL
    max_dim_x = max(L_out, L_bl_ref) * 1.05 / 2.0
    max_dim_y = max(W_out, W_bl_ref) * 1.05 / 2.0
            
    gl_z = RL_ref
    traces.append(
        go.Surface(
            x=np.linspace(-max_dim_x, max_dim_x, 2),
            y=np.linspace(-max_dim_y, max_dim_y, 2), 
            z=np.full((2, 2), gl_z),
            colorscale=[[0, 'rgba(0, 128, 0, 0.2)'], [1, 'rgba(0, 128, 0, 0.2)']],
            showscale=False, 
            opacity=0.3, 
            name='Ground Level',
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"3D Landfill Model ({'KML Polygon Approx.' if is_polygon_plot else 'Rectangular Profile'}) - Lifts Colored", 
        height=700,
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="RL (m)", 
            aspectmode="data", 
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True
    )
    return fig


# ----------------- 2D Plotting (No Change) -----------------

def generate_section_profile(df: pd.DataFrame, depth_bg: float, m_exc: float, m_out: float, crest_w: float, total_height: float, axis: str = "W") -> dict:
    """Calculates all profile points for the 2D cross-section."""
    
    df_rect_metrics = df[['Level', 'Length (m)', 'Width (m)', 'Height (m)', 'Volume (Cum)']].copy()
    
    if axis == "W": 
        def get_dim(row): return row["Width (m)"]
        bl_row = df_rect_metrics[df_rect_metrics['Level'] == 'BL'].iloc[0]
        bl_dim = get_dim(bl_row)
    else: 
        def get_dim(row): return row["Length (m)"]
        bl_row = df_rect_metrics[df_rect_metrics['Level'] == 'BL'].iloc[0]
        bl_dim = get_dim(bl_row)
    
    # --- 1. Waste Profile (Inner/Blue Line) ---
    x_in_right = [] ; z_in_right = []
    Z_current = 0.0 # Height accumulation (relative to BL)

    # BBL point (Inner base)
    bbl_row = df_rect_metrics[df_rect_metrics['Level'] == 'BBL'].iloc[0]
    W_bbl_ref = get_dim(bbl_row)
    x_in_right.append(W_bbl_ref / 2.0)
    z_in_right.append(-depth_bg)
    
    # BL Point (Inner) - waste profile starts here
    W_bl_ref = get_dim(bl_row)
    x_in_right.append(W_bl_ref / 2.0)
    z_in_right.append(0.0) # BL is Z=0
    
    Z_current = 0.0
    for i in range(len(df_rect_metrics)):
        row = df_rect_metrics.iloc[i]
        # Only process ABL Berm rows which contain the volume and final berm dimensions
        if row['Level'].startswith('ABL') and row['Volume (Cum)'] is not None:
            # Find the sloped top dimensions (ABL n)
            abl_top_row = df_rect_metrics[df_rect_metrics['Level'] == row['Level'].replace(' Berm', '')].iloc[0]
            W_top_ref = get_dim(abl_top_row)
            
            Z_lift_top = Z_current + row['Height (m)'] 
            
            # Point 1: Sloped top (start of berm bench)
            x_in_right.append(W_top_ref / 2.0)
            z_in_right.append(Z_lift_top)
        
            # Point 2: Berm edge (end of berm bench)
            W_berm_ref = get_dim(row)
            x_in_right.append(W_berm_ref / 2.0)
            z_in_right.append(Z_lift_top)
            
            Z_current = Z_lift_top # New Z for the next lift start
           
    W_top_final = get_dim(df_rect_metrics.iloc[-1])
    final_H_waste = Z_current

    x_in_left = [-x for x in x_in_right]; z_in_left = z_in_right
    x_top_plateau = [-W_top_final / 2.0, W_top_final / 2.0]
    z_top_plateau = [final_H_waste, final_H_waste]


    # --- 2. Excavation Profile (Red/Yellow Line) ---
    x_excav_right = [W_bl_ref / 2.0] ; z_excav_right = [0.0] # BL point
    W_Excav_Base = W_bbl_ref # BBL Width
    x_excav_right.append(W_Excav_Base / 2.0); z_excav_right.append(-depth_bg) # BBL point
    
    x_excav_base_plateau = [-W_Excav_Base / 2.0, W_Excav_Base / 2.0]
    z_excav_base_plateau = [-depth_bg, -depth_bg]
    x_excav_left = [-x for x in x_excav_right]; z_excav_left = z_excav_right

    # --- 3. Outer Profile / Bund (Black Line) ---
    final_H_abl = total_height - (depth_bg if depth_bg > 0 else 0.0)
    
    x_out_right = [W_bl_ref / 2.0] ; z_out_right = [0.0] # Start at BL edge (Z=0)
    
    # Outer slope up to TOB
    W_top_outer = W_bl_ref + 2 * m_out * final_H_abl
    x_out_right.append(W_top_outer / 2.0)
    z_out_right.append(final_H_abl)
    
    # Crest width (TOB)
    W_crest = W_top_outer + 2 * crest_w # Distance to the edge of the crest
    x_out_right.append(W_crest / 2.0)
    z_out_right.append(final_H_abl) # TOL is at final ABL lift height
    
    x_out_left = [-x for x in x_out_right]; z_out_left = z_out_right
    
    # TOL line span (from one side of the crest to the other)
    x_tol_span = [-W_crest / 2.0, W_crest / 2.0]
    z_tol_span = [final_H_abl, final_H_abl]
    
    return {
        "x_in_left": x_in_left, "z_in_left": z_in_left, "x_in_right": x_in_right, "z_in_right": z_in_right,
        "x_top_plateau": x_top_plateau, "z_top_plateau": z_top_plateau,
        "x_excav_right": x_excav_right, "z_excav_right": z_excav_right,
        "x_excav_left": x_excav_left, "z_excav_left": z_excav_left,
        "x_excav_base_plateau": x_excav_base_plateau, "z_excav_base_plateau": z_excav_base_plateau, 
        "x_out_left": x_out_left, "z_out_left": z_out_left, "x_out_right": x_out_right, "z_out_right": z_out_right,
        "x_tol_span": x_tol_span, "z_tol_span": z_tol_span,
        "axis": axis,
        "max_z": final_H_abl 
    }

def plot_cross_section(section: dict, title: str) -> Optional[bytes]:
    if not MATPLOTLIB_AVAILABLE or not plt: return None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 1. Outer Profile (Bund/Black Line - includes crest)
    ax.plot(section["x_out_left"], section["z_out_left"], marker='.', linestyle='-', color='k', linewidth=2, label="Outer Profile (Bund)")
    ax.plot(section["x_out_right"], section["z_out_right"], marker='.', linestyle='-', color='k', linewidth=2)
    
    # 2. Top of Landfill (TOL) - Spans the full crest width
    ax.plot(section["x_tol_span"], section["z_tol_span"], color='r', linewidth=1.5, linestyle='--', label="Top of Landfill (TOL)")

    # 3. Waste Profile (Inner/Blue Line - stepped profile)
    ax.plot(section["x_in_left"], section["z_in_left"], marker='o', linestyle='-', color='b', label="Waste Profile")
    ax.plot(section["x_in_right"], section["z_in_right"], marker='o', linestyle='-', color='b')
    ax.plot(section["x_top_plateau"], section["z_top_plateau"], linestyle='-', color='b')
    ax.plot(section["x_excav_base_plateau"], section["z_excav_base_plateau"], linestyle='-', color='b')
    
    # 4. Excavation Profile (Pit Boundary - Red/Yellow Line)
    ax.plot(section["x_excav_right"], section["z_excav_right"], linestyle='-', color='r', linewidth=3, label="Excavation Pit Wall")
    ax.plot(section["x_excav_left"], section["z_excav_left"], linestyle='-', color='r', linewidth=3)
    
    # 5. Reference Line
    ax.axhline(0, color='g', linewidth=1.5, linestyle=':', label="Ground Level (BL/GL)")
    
    # 6. Final Plot Setup
    ax.set_xlabel(f"{section['axis']}-axis distance (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.axis('equal')
    
    buf = BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=150); plt.close(fig)
    return buf.getvalue()


# ----------------- Display Plots -----------------

# 2. 3D Plot
fig3d = plotly_3d_full_stack(df, avg_ground_rl, total_height, m_out, crest_w, is_polygon_mode)
if fig3d:
    st.plotly_chart(fig3d, use_container_width=True)
else:
    st.warning("⚠️ Cannot generate 3D plot. Plotly, Numpy, and/or Shapely must be available.")
    
# 1. 2D Cross-Section Plot
section = generate_section_profile(df, depth_bg, m_exc, m_out, crest_w, total_height, axis=cross_section_axis[0])
title_caption = f"Cross-section along {cross_section_axis} Axis (Based on Equivalent Rectangular Dimensions)"
img = plot_cross_section(section, title=title_caption)
if img:
    st.image(img, caption=f"2D Cross-section: Waste Profile (Blue), Outer Bund (Black), Excavation Pit (Red).")
else:
    st.warning("⚠️ 2D plot function (Matplotlib) is missing/not working.")


# Export
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
    df.drop(columns=['Coords_XY', 'Lift_ID'], errors='ignore').to_excel(writer, sheet_name="LevelsRaw", index=False)
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

st.caption(f"Notes: The 3D model now colors each lift/segment individually. The **BBL Excavation Pit** (Lift 0) is dark blue, and subsequent **ABL Lifts** (Lift 1, Lift 2, etc.) cycle through distinct colors. A **Final Surface** mesh has been added to the top lift for a solid look.")
