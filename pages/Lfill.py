import streamlit as st
import pandas as pd
import math
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict
from io import BytesIO
import io
# Plotting Imports
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

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

def rectangle_polygon(width: float, length: float) -> List[Tuple[float, float]]:
    # Corrected function body
    half_w, half_l = width / 2.0, length / 2.0
    return [(-half_w, -half_l), (half_w, -half_l), (half_w, half_l), (half_w, half_l), (-half_w, half_l), (-half_w, -half_l)]

def get_footprint_corners(W: float, L: float, Z: float, RL_ref: float) -> List[Tuple[float, float, float]]:
    """Returns the 4 corner (X, Y, Z_abs) coordinates for a rectangular footprint."""
    half_W, half_L = W / 2.0, L / 2.0 # Corrected L instead of length
    RL = Z + RL_ref
    # Order: [(-W/2, -L/2), (W/2, -L/2), (W/2, L/2), (-W/2, L/2)]
    return [
        (-half_W, -half_L, RL),
        (half_W, -half_L, RL),
        (half_W, half_L, RL),
        (-half_W, half_L, RL)
    ]

def frustum_mesh(W1, L1, Z1, W2, L2, Z2, RL_ref, start_v_index):
    """Generates mesh vertices and faces for a rectangular frustum section."""
    corners1 = get_footprint_corners(W1, L1, Z1, RL_ref)
    corners2 = get_footprint_corners(W2, L2, Z2, RL_ref)
    verts = corners1 + corners2 # 8 vertices in total

    faces = []
    for i in range(4):
        i1, i2 = i, (i + 1) % 4  # Bottom corners
        i3, i4 = i + 4, (i + 1) % 4 + 4 # Top corners

        # Face 1: (Z1_i, Z1_{i+1}, Z2_{i+1})
        faces.append([i1, i2, i4])
        # Face 2: (Z1_i, Z2_{i+1}, Z2_i)
        faces.append([i1, i4, i3])

    faces_np = np.array(faces) + start_v_index
    
    X = [v[0] for v in verts]
    Y = [v[1] for v in verts]
    Z_abs = [v[2] for v in verts]
    
    return X, Y, Z_abs, faces_np.tolist(), 8 # Return the vertex count

# New function to convert the calculated levels into a list of geometric frustums for plotting
def generate_frustum_data(df: pd.DataFrame, depth_bg: float, avg_ground_rl: float) -> Tuple[List[Dict], float, float, float, float]:
    """Converts the levels DataFrame into a list of geometric frustums for 3D plotting."""
    frustum_list = []
    RL_ref = avg_ground_rl
    Z_BL = 0.0 # BL is at Z=0 relative to ground
    Z_Base = -depth_bg

    # Get BL dimensions
    bl_row = df[df['Level'] == 'BL'].iloc[0]
    L_bl, W_bl = bl_row["Length (m)"], bl_row["Width (m)"]

    # 1. BBL Frustum (Base to BL)
    if 'BBL' in df['Level'].values:
        bbl_row = df[df['Level'] == 'BBL'].iloc[0]
        L_bbl, W_bbl = bbl_row["Length (m)"], bbl_row["Width (m)"]
        
        if L_bbl > 0 and W_bbl > 0 and L_bl > 0 and W_bl > 0 and depth_bg > 0:
            frustum_list.append({
                "W1": W_bbl, "L1": L_bbl, "Z1": Z_Base,
                "W2": W_bl, "L2": L_bl, "Z2": Z_BL,
                "Type": "Waste/Liner", "Color": "rgba(0, 0, 255, 0.6)" # Blue
            })
        L_current_top, W_current_top = L_bl, W_bl
    else:
        L_bbl, W_bbl = L_bl, W_bl # No BBL means base is at BL
        L_current_top, W_current_top = L_bl, W_bl

    # 2. ABL Frustums
    df_abl_berm = df[df['Level'].str.contains('Berm')].copy().reset_index()
    Z_current = Z_BL
    L_top, W_top = L_bl, W_bl
        
    for i, row in df_abl_berm.iterrows():
        lift_h = row["Height (m)"]
        
        # ABL row immediately preceding the Berm row is the top of the slope
        try:
             abl_row = df[df['Level'] == f'ABL {i+1}'].iloc[0]
             L_top_slope, W_top_slope = abl_row["Length (m)"], abl_row["Width (m)"]
        except IndexError:
             # This happens if the last ABL lift was cut short by the area cap
             L_top_slope, W_top_slope = row["Length (m)"], row["Width (m)"]

        if L_current_top > 0 and W_current_top > 0 and L_top_slope > 0 and W_top_slope > 0 and lift_h > 0:
            Z_next = Z_current + lift_h
            frustum_list.append({
                "W1": W_current_top, "L1": L_current_top, "Z1": Z_current,
                "W2": W_top_slope, "L2": L_top_slope, "Z2": Z_next,
                "Type": "Waste/Fill", "Color": "rgba(0, 0, 255, 0.6)" # Blue
            })
            Z_current = Z_next
            W_top, L_top = W_top_slope, L_top_slope # Update max dimensions
            
        # Update current top to the post-berm dimensions for the next lift's base
        L_current_top, W_current_top = row["Length (m)"], row["Width (m)"]
        
    # The final top is the last W_top, L_top from the loop
    return frustum_list, L_bl, W_bl, L_bbl, W_bbl, L_top, W_top

def plotly_3d_full_stack(frustum_data: List[Dict], L_bl: float, W_bl: float, L_top: float, W_top: float, avg_ground_rl: float):
    
    RL_ref = avg_ground_rl 
    traces = []
    v_count = 0 
    all_X_waste, all_Y_waste, all_Z_waste, all_faces_waste = [], [], [], []
    
    # --- 1. Waste Mass (BBL and ABL) (Blue) ---
    for f_data in frustum_data:
        W1, L1, Z1, W2, L2, Z2 = f_data["W1"], f_data["L1"], f_data["Z1"], f_data["W2"], f_data["L2"], f_data["Z2"]
        
        X_seg, Y_seg, Z_seg, faces_seg, v_num_seg = frustum_mesh(W1, L1, Z1, W2, L2, Z2, RL_ref, v_count)
        
        all_X_waste.extend(X_seg)
        all_Y_waste.extend(Y_seg)
        all_Z_waste.extend(Z_seg)
        all_faces_waste.extend(faces_seg)
        v_count += v_num_seg

    I_waste = [f[0] for f in all_faces_waste]
    J_waste = [f[1] for f in all_faces_waste]
    K_waste = [f[2] for f in all_faces_waste]
    
    # Waste Mesh Trace (Blue)
    traces.append(
        go.Mesh3d(
            x=all_X_waste, y=all_Y_waste, z=all_Z_waste,
            i=I_waste, j=J_waste, k=K_waste,
            opacity=0.8, color='rgba(0, 0, 255, 0.6)', 
            name='Waste (Landfill Profile)', hoverinfo='name'
        )
    )
    
    # --- 2. GL Plane ---
    # We use a bounding box slightly larger than the BL footprint for the GL plane
    gl_corners = get_footprint_corners(W_bl * 1.1, L_bl * 1.1, 0.0, RL_ref)
    
    X_coords = np.unique([c[0] for c in gl_corners[:4]])
    Y_coords = np.unique([c[1] for c in gl_corners[:4]])
    RL_val = gl_corners[0][2] # This is avg_ground_rl
    Z_matrix = np.full((len(Y_coords), len(X_coords)), RL_val)
    
    traces.append(
        go.Surface(
            x=X_coords,
            y=Y_coords, 
            z=Z_matrix,
            colorscale=[[0, 'rgba(0, 128, 0, 0.2)'], [1, 'rgba(0, 128, 0, 0.2)']],
            showscale=False, 
            opacity=0.3, 
            name='Ground Level',
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="3D Landfill Model (BBL/ABL Stepped Profile)", 
        height=700,
        scene=dict(
            xaxis_title="Width (m)", yaxis_title="Length (m)", zaxis_title="RL (m)", 
            aspectmode="data", 
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.25, y=1.25, z=1.25)
            )
        ),
        showlegend=True
    )
    return fig

# New function to generate the 2D cross-section profile
def generate_section_profile(frustum_data: List[Dict], L_bl: float, W_bl: float, L_bbl: float, W_bbl: float, L_top: float, W_top: float, depth_bg: float, m_exc: float, axis: str = "W") -> dict:
    
    if axis == "W": 
        W_bl_ref, W_bbl_ref, W_top_ref = W_bl, W_bbl, W_top
        def get_w(data): return data["W1"], data["W2"]
    else: 
        W_bl_ref, W_bbl_ref, W_top_ref = L_bl, L_bbl, L_top
        def get_w(data): return data["L1"], data["L2"]
    
    # 1. Inner Profile (Landfill Waste - Blue Line)
    x_in_right = [] ; z_in_right = []
    
    # Start at Base (BBL)
    if W_bbl_ref > 0 and depth_bg > 0:
        x_in_right.append(W_bbl_ref / 2.0)
        z_in_right.append(-depth_bg)
    
    # BL Point (GL)
    x_in_right.append(W_bl_ref / 2.0)
    z_in_right.append(0.0)
    
    # ABL Steps
    for f_data in frustum_data:
        if "Waste" in f_data["Type"]: # Only consider sloped waste sections
            W_base, W_top_w = get_w(f_data)
            Z_base, Z_top_w = f_data["Z1"], f_data["Z2"]
            
            # Base of the sloped part
            if not (abs(x_in_right[-1] - W_base / 2.0) < 1e-3 and abs(z_in_right[-1] - Z_base) < 1e-3):
                 x_in_right.append(W_base / 2.0)
                 z_in_right.append(Z_base)
                 
            # Top of the sloped part
            x_in_right.append(W_top_w / 2.0)
            z_in_right.append(Z_top_w)

    # Final Crest Plateau (from the last sloped top to the final top width)
    if W_top_ref / 2.0 != x_in_right[-1]:
        x_in_right.append(W_top_ref / 2.0)
        z_in_right.append(z_in_right[-1])
          
    x_in_left = [-x for x in x_in_right]; z_in_left = z_in_right
    x_top_plateau = [-W_top_ref / 2.0, W_top_ref / 2.0]
    z_top_plateau = [z_in_right[-1], z_in_right[-1]]

    # 2. Excavation Profile (Pit Boundary) - Red Line (Below GL only)
    x_excav_right = [W_bl_ref / 2.0] 
    z_excav_right = [0.0]
    W_Excav_Base = W_bbl_ref # The BBL width is the waste base, also the base of the excavation pit
    
    x_excav_right.append(W_Excav_Base / 2.0)
    z_excav_right.append(-depth_bg)
    
    # Horizontal Excavation Base Plateau
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

def plot_cross_section(section: dict, title: str = "Cross-Section") -> bytes:
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
    buf = io.BytesIO();
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150); plt.close(fig)
    return buf.getvalue()

# ... (rest of the original helper functions: parse_kml_polygon, latlon_to_xy, etc.)

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
                lon = float(parts[0]);
                lat = float(parts[1])
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
        x1,y1 = pts[i];
        x2,y2 = pts[i+1]
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
    S = P/2.0
    disc = (S/2.0)**2 - A
    if disc < 0:
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
    
    # New Input for absolute RL reference
    avg_ground_rl = st.number_input("Avg. ground RL (m)", 0.0, value=100.0, step=1.0)

    st.header("Geometry")
    if mode == "Length & Width":
        L_bl = st.number_input("BL Length (m)", 10.0, value=567.0, step=1.0)
        W_bl = st.number_input("BL Width (m)", 10.0, value=566.15, step=1.0)
    elif mode == "Area & Aspect":
        A_bl = st.number_input("BL Area (m²)", 1000.0, value=321008.457, step=1000.0, format="%.3f")
        aspect = st.slider("Assumed BL L:W", 1.0, 5.0, 1.0, 0.05)
        W_bl = math.sqrt(A_bl/aspect);
        L_bl = aspect*W_bl
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
    stop_after = False
    if A_i <= (A_bl * top_area_cap_pct/100.0):
        stop_after = True

    # Volume for this lift = between previous berm area and ABL i area, over lift_h
    V_i = frustum_volume(A_prev_berm, A_i, lift_h) if lift_h>0 else 0.0
    cum_vol += V_i

    # Add ABL i (sloped top)
    rows.append({
        "Level": f"ABL {i}",
        "Length (m)": L_i,
        "Width (m)": W_i,
        "Area (sqm)": A_i,
        "Height (m)": lift_h,
        "Volume (Cum)": None
    })

    # Berm at top of lift (zero-height transition)
    L_berm = max(L_i - 2*bench_w, 0.0)
    W_berm = max(W_i - 2*bench_w, 0.0)
    A_berm = max(L_berm*W_berm, 0.0)
    
    if L_berm <= 0 or W_berm <= 0:
        # If the berm width makes the footprint zero or negative, stop here.
        break

    rows.append({
        "Level": f"ABL {i} Berm",
        "Length (m)": L_berm,
        "Width (m)": W_berm,
        "Area (sqm)": A_berm,
        "Height (m)": lift_h, # display height here like the screenshot
        "Volume (Cum)": V_i
    })

    # Prepare for next lift
    L_prev_berm, W_prev_berm, A_prev_berm = L_berm, W_berm, A_berm
    if stop_after:
        break
    i += 1
    if i > 100: # Safety break
        break

df = pd.DataFrame(rows)

# Totals
total_height = (depth_bg if depth_bg>0 else 0.0) + i*lift_h
total_volume = cum_vol
total_tons = total_volume * density
years = (total_tons / tpa) if tpa>0 else None

st.subheader("Landfill Level Summary Table")

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

st.markdown("---")
st.subheader("2D and 3D Visualizations")

# --------------- Visualization ---------------

# 1. 2D Cross-Section Plot
frustum_list, L_bl, W_bl, L_bbl, W_bbl, L_top, W_top = generate_frustum_data(df, depth_bg, avg_ground_rl)

section = generate_section_profile(frustum_list, L_bl, W_bl, L_bbl, W_bbl, L_top, W_top, depth_bg, m_exc, axis=cross_section_axis[0])
title_caption = f"Cross-section along {cross_section_axis} Axis"
img = plot_cross_section(section, title=title_caption)
st.image(img, caption=f"2D Cross-section of Stepped Landfill Profile (Waste in Blue, Excavation in Red).")

# 2. 3D Plot
fig3d = plotly_3d_full_stack(frustum_list, L_bl, W_bl, L_top, W_top, avg_ground_rl)
if fig3d:
    st.plotly_chart(fig3d, use_container_width=True)

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
