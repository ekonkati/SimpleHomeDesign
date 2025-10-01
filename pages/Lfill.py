import streamlit as st
import pandas as pd
import math
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict
from io import BytesIO

# --- Optional Imports ---
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    np = None
    NUMPY_AVAILABLE = False
    
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    PLOTLY_AVAILABLE = False
    
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False
    
try:
    from shapely.geometry import Polygon, MultiPolygon
    SHAPELY_AVAILABLE = True
except Exception:
    Polygon = None
    MultiPolygon = None
    SHAPELY_AVAILABLE = False

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
    # Formula for frustum volume: H/3 * (A1 + A2 + sqrt(A1*A2))
    return H/3.0 * (A1 + A2 + math.sqrt(max(A1*A2, 0.0)))

def rectangle_polygon(width: float, length: float) -> List[Tuple[float, float]]:
    """Generates 5 XY points for a closed rectangular footprint centered at (0,0)."""
    half_w, half_l = width / 2.0, length / 2.0
    return [(-half_w, -half_l), (half_w, -half_l), (half_w, half_l), (-half_w, half_l), (-half_w, -half_l)]

# --- Geometry Functions for Polygon Mode (KML) ---

def polygon_buffer(xy_coords: List[Tuple[float, float]], offset_dist: float) -> Optional[List[Tuple[float, float]]]:
    """Generates a new polygon by offsetting (buffering) the original xy_coords using Shapely."""
    if not SHAPELY_AVAILABLE or len(xy_coords) < 3:
        return None
        
    try:
        # Ensure the polygon is closed for Shapely
        if xy_coords[0] != xy_coords[-1]:
            poly = Polygon(xy_coords + [xy_coords[0]])
        else:
            poly = Polygon(xy_coords)

        # Negative offset for shrinking (BBL, ABL top)
        new_poly = poly.buffer(offset_dist, join_style=2, mitre_limit=5.0) 
        
        # Handle cases where buffering results in multiple polygons
        if isinstance(new_poly, MultiPolygon):
            if not new_poly.geoms: return None
            new_poly = max(new_poly.geoms, key=lambda p: p.area)
        
        # If the resulting polygon is too small (area is zero), return None
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
    coords_elems = tree.findall(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)
    if not coords_elems:
        coords_elems = tree.findall(".//Polygon/outerBoundaryIs/LinearRing/coordinates")
        if not coords_elems:
            return None
            
    text = coords_elems[0].text or ""
    pts = []
    for token in text.replace("\n", " ").split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
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
    """Calculates Area and Perimeter from XY coordinates using the Shoelace formula."""
    if not xy_pts: return 0.0, 0.0
    
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
    """Main KML loader function."""
    file.seek(0)
    latlon = parse_kml_polygon(file.read())
    if not latlon:
        return 0.0, 0.0, None
        
    lat0 = sum(p[0] for p in latlon)/len(latlon)
    lon0 = sum(p[1] for p in latlon)/len(latlon)
    xy = [latlon_to_xy(lat, lon, lat0, lon0) for lat,lon in latlon]
    
    A, P = polygon_area_perimeter_xy(xy)
    
    # Ensure the XY list is closed
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

# --- Geometry Functions for Rectangular Mode ---

def get_footprint_corners(W: float, L: float, Z: float, RL_ref: float) -> List[Tuple[float, float, float]]:
    """Returns the 4 corner (X, Y, Z_abs) coordinates for a rectangular footprint."""
    half_W, half_L = W / 2.0, L / 2.0
    RL = Z + RL_ref
    return [
        (-half_W, -half_L, RL), (half_W, -half_L, RL), 
        (half_W, half_L, RL), (-half_W, half_L, RL)
    ]

def frustum_mesh(W1, L1, Z1, W2, L2, Z2, RL_ref, start_v_index):
    """Generates mesh vertices and faces for a rectangular frustum section."""
    corners1 = get_footprint_corners(W1, L1, Z1, RL_ref)
    corners2 = get_footprint_corners(W2, L2, Z2, RL_ref)
    verts = corners1 + corners2 # 8 vertices in total

    faces = []
    for i in range(4):
        i1, i2 = i, (i + 1) % 4
        i3, i4 = i + 4, (i + 1) % 4 + 4

        # Face 1: (Z1_i, Z1_{i+1}, Z2_{i+1})
        faces.append([i1, i2, i4])
        # Face 2: (Z1_i, Z2_{i+1}, Z2_i)
        faces.append([i1, i4, i3])

    faces_np = np.array(faces) + start_v_index
    
    X = [v[0] for v in verts]
    Y = [v[1] for v in verts]
    Z_abs = [v[2] for v in verts]
    
    return X, Y, Z_abs, faces_np.tolist(), 8 

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
        st.error("⚠️ **Shapely** library is required for KML polygon modeling.")
        
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
        if kml is not None:
            A_bl, P_bl, xy_coords = kml_to_area_perimeter_and_xy(kml)
            if A_bl > 0:
                L_bl, W_bl = rect_from_A_P(A_bl, P_bl)
                st.session_state.bl_coords_xy = xy_coords
            else:
                st.error("Could not derive area/perimeter from KML.")
                
    # Finalize BL footprint and coords
    if mode != "KML Polygon" or not st.session_state.bl_coords_xy:
        A_bl = L_bl * W_bl
        P_bl = 2*(L_bl + W_bl)
        st.session_state.bl_coords_xy = rectangle_polygon(W_bl, L_bl) if A_bl > 0 else []

    st.header("Levels & Slopes")
    depth_bg = st.number_input("BBL depth (m)", 0.0, value=7.0, step=0.5)
    exc_slope = st.selectbox("Excavation side slope (V:H)", ["1:1", "1:1.5", "1:2", "1:2.5", "1:3"], index=4)
    m_exc = vh_to_h_per_v(exc_slope)

    lift_h = st.number_input("Lift height (m)", 1.0, value=5.0, step=0.5)
    inside_slope = st.selectbox("Inside slope (V:H)", ["1:2", "1:2.5", "1:3"], index=2)
    m_in = vh_to_h_per_v(inside_slope)
    bench_w = st.number_input("Bench width (m, per side)", 0.0, value=4.0, step=0.5)
    top_area_cap_pct = st.slider("Stop when top area ≤ (% of BL area)", 5, 100, 30, 5)
    
    st.header("Visualization Options")
    cross_section_axis = st.radio("2D Section View:", ["Width (W)", "Length (L)"])

    st.header("Operations")
    density = st.number_input("Density (t/m³)", 0.3, value=1.0, step=0.05)
    tpa = st.number_input("Waste generation (TPA)", 1000.0, value=365000.0, step=5000.0)

# --------------- Compute Levels ---------------
if L_bl <= 0 or W_bl <= 0:
    st.warning("Provide a valid BL footprint.")
    st.stop()

rows = []
cum_vol = 0.0
is_polygon_mode = mode == "KML Polygon" and SHAPELY_AVAILABLE and st.session_state.bl_coords_xy

# --- BBL CALCULATION ---
if is_polygon_mode:
    # Polygon Mode: Use buffer to shrink footprint
    offset_bbl = -m_exc * depth_bg
    xy_bbl = polygon_buffer(st.session_state.bl_coords_xy, offset_bbl)
    
    A_bbl, P_bbl = polygon_area_perimeter_xy(xy_bbl) if xy_bbl else (0.0, 0.0)
    L_bbl, W_bbl = rect_from_A_P(A_bbl, P_bbl)
    
    V_bbl = frustum_volume(A_bbl, A_bl, depth_bg) if depth_bg > 0 else 0.0
    cum_vol += V_bbl
    rows.append({
        "Level": "BBL", "Length (m)": L_bbl, "Width (m)": W_bbl, "Area (sqm)": A_bbl,
        "Height (m)": depth_bg if depth_bg > 0 else None, "Volume (Cum)": V_bbl, 
        "Coords_XY": xy_bbl
    })
else:
    # Rectangular Mode: Use simple subtraction
    L_bbl = max(L_bl - 2*m_exc*depth_bg, 0.0)
    W_bbl = max(W_bl - 2*m_exc*depth_bg, 0.0)
    A_bbl = max(L_bbl*W_bbl, 0.0)
    xy_bbl = rectangle_polygon(W_bbl, L_bbl) if A_bbl > 0 else []
    
    V_bbl = frustum_volume(A_bbl, A_bl, depth_bg) if depth_bg > 0 else 0.0
    cum_vol += V_bbl
    rows.append({
        "Level": "BBL", "Length (m)": L_bbl, "Width (m)": W_bbl, "Area (sqm)": A_bbl,
        "Height (m)": depth_bg if depth_bg > 0 else None, "Volume (Cum)": V_bbl, 
        "Coords_XY": xy_bbl
    })

# BL row (reference, GL = 0)
rows.append({
    "Level": "BL", "Length (m)": L_bl, "Width (m)": W_bl, "Area (sqm)": A_bl,
    "Height (m)": None, "Volume (Cum)": None, "Coords_XY": st.session_state.bl_coords_xy
})

# --- ABL ITERATION ---
if is_polygon_mode:
    xy_prev_berm = st.session_state.bl_coords_xy
    A_prev_berm = A_bl
else:
    L_prev_berm = L_bl; W_prev_berm = W_bl; A_prev_berm = A_bl
    
i = 1
while True:
    
    if is_polygon_mode:
        # 1. Slope up one lift (Polygon)
        offset_abl = -m_in * lift_h
        xy_i = polygon_buffer(xy_prev_berm, offset_abl)
        A_i, P_i = polygon_area_perimeter_xy(xy_i) if xy_i else (0.0, 0.0)
        L_i, W_i = rect_from_A_P(A_i, P_i)
        
        # 2. Berm at top of lift (Polygon)
        offset_berm = -bench_w
        xy_berm = polygon_buffer(xy_i, offset_berm)
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

    # stop if top area cap reached next
    stop_after = False
    if A_i <= (A_bl * top_area_cap_pct/100.0):
        stop_after = True

    V_i = frustum_volume(A_prev_berm, A_i, lift_h) if lift_h > 0 else 0.0
    cum_vol += V_i

    # ABL i (sloped top)
    rows.append({
        "Level": f"ABL {i}", "Length (m)": L_i, "Width (m)": W_i, "Area (sqm)": A_i,
        "Height (m)": lift_h, "Volume (Cum)": None, "Coords_XY": xy_i
    })
    
    # Berm at top of lift
    rows.append({
        "Level": f"ABL {i} Berm", "Length (m)": L_berm, "Width (m)": W_berm, "Area (sqm)": A_berm,
        "Height (m)": lift_h, "Volume (Cum)": V_i, "Coords_XY": xy_berm
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
total_height = (depth_bg if depth_bg > 0 else 0.0) + (i-1)*lift_h
if total_height < 0: total_height = 0.0

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
display_df = display_df.drop(columns=['Coords_XY'])
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

# ----------------- 3D Plotting -----------------

def plotly_3d_full_stack(df: pd.DataFrame, avg_ground_rl: float):
    if not PLOTLY_AVAILABLE or not NUMPY_AVAILABLE: return go.Figure()
    
    RL_ref = avg_ground_rl 
    traces = []
    
    is_polygon_plot = 'Coords_XY' in df.columns and df['Coords_XY'].iloc[0] is not None
    
    # Filter to only the distinct levels (BBL bottom, BL, ABL Berms)
    df_geom = df[df['Volume (Cum)'].notnull() | (df['Level'] == 'BL') | (df['Level'] == 'BBL')].copy().reset_index(drop=True)
    
    # Z-level pre-calculation: Calculate absolute RL for each distinct level row
    Z_abs_map = {}
    cum_H = 0.0
    for idx, row in df_geom.iterrows():
        if row['Level'] == 'BBL':
             Z_abs_map[idx] = RL_ref - row['Height (m)'] 
        elif row['Level'] == 'BL':
             Z_abs_map[idx] = RL_ref 
             cum_H = 0.0 # Reset cumulative height at GL
        else: # ABL Berm level
            cum_H += row['Height (m)']
            Z_abs_map[idx] = RL_ref + cum_H
    
    # Build Mesh
    all_X, all_Y, all_Z, all_faces_waste = [], [], [], []
    v_count = 0 
        
    for i in range(len(df_geom) - 1):
        row1 = df_geom.iloc[i] # Bottom level of frustum
        row2 = df_geom.iloc[i+1] # Top level of frustum
        
        # Determine Z levels (Absolute RL)
        Z1_abs = Z_abs_map.get(i, RL_ref)
        Z2_abs = Z_abs_map.get(i+1, RL_ref)
        
        if Z2_abs <= Z1_abs: continue

        if is_polygon_plot:
            # Polygon Mode Mesh Generation
            verts1 = row1['Coords_XY'] 
            verts2 = row2['Coords_XY'] 
            
            if not verts1 or not verts2: continue
            
            # --- FIX: Ensure we use the actual count of vertices for each level ---
            v1_count = len(verts1)
            v2_count = len(verts2)

            # Polygons must have at least 3 vertices to form a face
            if v1_count < 3 or v2_count < 3: continue 

            # Add vertices to the master list
            # Vertices for Frustum Bottom (verts1)
            for k in range(v1_count):
                all_X.append(verts1[k][0])
                all_Y.append(verts1[k][1])
                all_Z.append(Z1_abs)
            
            # Vertices for Frustum Top (verts2)
            for k in range(v2_count):
                all_X.append(verts2[k][0])
                all_Y.append(verts2[k][1])
                all_Z.append(Z2_abs)

            # Faces (connect bottom to top)
            # Only connect faces if the number of vertices is the same for a clean connection
            if v1_count == v2_count:
                for k in range(v1_count):
                    i1, i2 = k, (k + 1) % v1_count      # Bottom corners
                    i3, i4 = k + v1_count, (k + 1) % v1_count + v1_count # Top corners

                    # Face 1 (Triangle 1)
                    all_faces_waste.append([i1 + v_count, i2 + v_count, i4 + v_count])
                    # Face 2 (Triangle 2)
                    all_faces_waste.append([i1 + v_count, i4 + v_count, i3 + v_count])
                
                v_count += v1_count + v2_count # Total vertices added in this segment
            else:
                 # If counts are different, skip the face generation but ensure v_count is correct.
                 v_count += v1_count + v2_count
                 continue

        else:
            # Rectangular Mode Mesh Generation (Fallback)
            W1, L1 = row1['Width (m)'], row1['Length (m)']
            W2, L2 = row2['Width (m)'], row2['Length (m)']
            
            # Z1 and Z2 must be relative to GL=0 for the helper function
            X_seg, Y_seg, Z_seg, faces_seg, v_num_seg = frustum_mesh(W1, L1, Z1_abs-RL_ref, W2, L2, Z2_abs-RL_ref, RL_ref, v_count)
            
            all_X.extend(X_seg)
            all_Y.extend(Y_seg)
            all_Z.extend(Z_seg)
            all_faces_waste.extend(faces_seg)
            v_count += v_num_seg

    if all_faces_waste:
        I_waste = [f[0] for f in all_faces_waste]
        J_waste = [f[1] for f in all_faces_waste]
        K_waste = [f[2] for f in all_faces_waste]
        
        traces.append(
            go.Mesh3d(
                x=all_X, y=all_Y, z=all_Z,
                i=I_waste, j=J_waste, k=K_waste,
                opacity=0.8, color='rgba(0, 0, 255, 0.6)', 
                name='Waste (Landfill Profile)', hoverinfo='name'
            )
        )
    
    # Ground Level Plane
    if is_polygon_plot and st.session_state.bl_coords_xy:
        poly_bl = Polygon(st.session_state.bl_coords_xy)
        minx, miny, maxx, maxy = poly_bl.bounds
    else:
        minx, miny = -L_bl/2, -W_bl/2
        maxx, maxy = L_bl/2, W_bl/2

    gl_z = RL_ref
    traces.append(
        go.Surface(
            x=np.linspace(minx * 1.05, maxx * 1.05, 2),
            y=np.linspace(miny * 1.05, maxy * 1.05, 2), 
            z=np.full((2, 2), gl_z),
            colorscale=[[0, 'rgba(0, 128, 0, 0.2)'], [1, 'rgba(0, 128, 0, 0.2)']],
            showscale=False, 
            opacity=0.3, 
            name='Ground Level',
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"3D Landfill Model ({'KML Polygon' if is_polygon_plot else 'Rectangular Profile'})", 
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


# ----------------- 2D Plotting -----------------
# 2D Plotting is based on the equivalent rectangular dimensions (L_bl, W_bl) for simplification.

def generate_section_profile(df: pd.DataFrame, depth_bg: float, m_exc: float, axis: str = "W") -> dict:
    
    df_rect_metrics = df[['Level', 'Length (m)', 'Width (m)', 'Height (m)']].copy()
    
    if axis == "W": 
        def get_dim(row): return row["Width (m)"]
    else: 
        def get_dim(row): return row["Length (m)"]
    
    x_in_right = [] ; z_in_right = []
    Z_current = 0.0

    # BBL point
    bbl_row = df_rect_metrics[df_rect_metrics['Level'] == 'BBL'].iloc[0]
    W_bbl_ref = get_dim(bbl_row)
    x_in_right.append(W_bbl_ref / 2.0)
    z_in_right.append(-depth_bg)
    
    # BL Point
    bl_row = df_rect_metrics[df_rect_metrics['Level'] == 'BL'].iloc[0]
    W_bl_ref = get_dim(bl_row)
    x_in_right.append(W_bl_ref / 2.0)
    z_in_right.append(0.0)
    
    Z_current = 0.0
    for i in range(len(df_rect_metrics)):
        row = df_rect_metrics.iloc[i]
        if row['Level'].startswith('ABL') and row['Volume (Cum)'] is not None:
            # Look up the dimensions from the corresponding ABL sloped top
            if 'Berm' in row['Level']:
                # For a Berm row, we need the dimensions of the sloped top *above* it
                abl_top_row = df_rect_metrics[df_rect_metrics['Level'] == row['Level'].replace(' Berm', '')].iloc[0]
                W_top_ref = get_dim(abl_top_row)
                
                # Point 1: Top of the sloped lift (where the berm starts)
                # Need to use the Z value calculated from the previous berm's Z + lift height
                Z_lift_top = Z_current + row['Height (m)'] 
                x_in_right.append(W_top_ref / 2.0)
                z_in_right.append(Z_lift_top)
                
                # Point 2: End of the berm (start of the next lift)
                W_berm_ref = get_dim(row)
                x_in_right.append(W_berm_ref / 2.0)
                z_in_right.append(Z_lift_top)
                
                Z_current = Z_lift_top
            
    W_top_ref = get_dim(df_rect_metrics.iloc[-1]) # Final top dimension
    
    x_in_left = [-x for x in x_in_right]; z_in_left = z_in_right
    x_top_plateau = [-W_top_ref / 2.0, W_top_ref / 2.0]
    z_top_plateau = [z_in_right[-1], z_in_right[-1]]

    # Excavation Profile (Pit Boundary) - Red Line (Below GL only)
    x_excav_right = [W_bl_ref / 2.0] ; z_excav_right = [0.0]
    W_Excav_Base = W_bbl_ref
    x_excav_right.append(W_Excav_Base / 2.0); z_excav_right.append(-depth_bg)
    
    x_excav_base_plateau = [-W_Excav_Base / 2.0, W_Excav_Base / 2.0]
    z_excav_base_plateau = [-depth_bg, -depth_bg]

    x_excav_left = [-x for x in x_excav_right]; z_excav_left = z_excav_right
    
    return {
        "x_in_left": x_in_left, "z_in_left": z_in_left, "x_in_right": x_in_right, "z_in_right": z_in_right,
        "x_top_plateau": x_top_plateau, "z_top_plateau": z_top_plateau,
        "x_base_plateau": x_excav_base_plateau, "z_base_plateau": z_excav_base_plateau, 
        "x_excav_right": x_excav_right, "z_excav_right": z_excav_right,
        "x_excav_left": x_excav_left, "z_excav_left": z_excav_left,
        "axis": axis,
        "max_z": z_in_right[-1]
    }

def plot_cross_section(section: dict, title: str = "Cross-Section") -> Optional[bytes]:
    if not MATPLOTLIB_AVAILABLE or not plt: return None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 1. Inner Profile (Landfill Waste - Blue line)
    ax.plot(section["x_in_left"], section["z_in_left"], marker='o', linestyle='-', color='b', label="Waste Profile")
    ax.plot(section["x_in_right"], section["z_in_right"], marker='o', linestyle='-', color='b')
    ax.plot(section["x_top_plateau"], section["z_top_plateau"], linestyle='-', color='b')
    ax.plot(section["x_base_plateau"], section["z_base_plateau"], linestyle='-', color='b')
    
    # 2. Excavation Profile (Pit Boundary) - Red Line (Below GL only)
    ax.plot(section["x_excav_right"], section["z_excav_right"], linestyle='-', color='r', linewidth=3, label="Excavation Pit Wall")
    ax.plot(section["x_excav_left"], section["z_excav_left"], linestyle='-', color='r', linewidth=3)
    ax.plot(section["x_base_plateau"], section["z_base_plateau"], linestyle='-', color='r', linewidth=3)
    
    # 3. Reference Lines
    ax.axhline(0, color='g', linewidth=1.5, linestyle=':', label="Ground Level (BL/GL)")
    ax.axhline(section["max_z"], color='r', linewidth=0.8, linestyle='--', label="Top of Landfill (TOL)")
    
    # 4. Final Plot Setup
    ax.set_xlabel(f"{section['axis']}-axis distance (m)"); ax.set_ylabel("z (m)"); ax.set_title(title); ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left'); ax.axis('equal')
    buf = BytesIO();
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150); plt.close(fig)
    return buf.getvalue()

# ----------------- Display Plots -----------------

# 2. 3D Plot
fig3d = plotly_3d_full_stack(df, avg_ground_rl)
if fig3d:
    st.plotly_chart(fig3d, use_container_width=True)
else:
    st.warning("⚠️ Cannot generate 3D plot. Plotly, Numpy, and/or Shapely may be missing or the geometry is invalid.")
    
# 1. 2D Cross-Section Plot
section = generate_section_profile(df, depth_bg, m_exc, axis=cross_section_axis[0])
title_caption = f"Cross-section along {cross_section_axis} Axis (Based on Equivalent Rectangular Dimensions)"
img = plot_cross_section(section, title=title_caption)
if img:
    st.image(img, caption=f"2D Cross-section of Stepped Landfill Profile (Waste in Blue, Excavation in Red).")
else:
    st.warning("⚠️ Cannot generate 2D plot. Matplotlib may be missing.")


# Export
buffer = BytesIO()
with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
    df.drop(columns=['Coords_XY'], errors='ignore').to_excel(writer, sheet_name="LevelsRaw", index=False)
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

st.caption("Notes: The 3D model uses the actual KML polygon shape. The 2D cross-section is plotted using the equivalent rectangular Length/Width derived from the KML's Area/Perimeter.")
