import streamlit as st
import pandas as pd
import math
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Dict # Dict needed for cleaner typing
from io import BytesIO

# --- Optional Imports for Plotting ---
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
    
# NOTE: Shapely is omitted as it was not in the provided file and complicates the fix.
# The 3D model will use the rectangular approximation L/W from the KML.

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

# --- Geometry Functions for Rectangular Plotting ---

def get_footprint_corners(W: float, L: float, Z_rel: float, RL_ref: float) -> List[Tuple[float, float, float]]:
    """Returns the 4 corner (X, Y, Z_abs) coordinates for a rectangular footprint."""
    half_W, half_L = W / 2.0, L / 2.0
    RL = Z_rel + RL_ref
    # Vertices: 1:(-W/2, -L/2), 2:(W/2, -L/2), 3:(W/2, L/2), 4:(-W/2, L/2)
    return [
        (-half_W, -half_L, RL),
        (half_W, -half_L, RL),
        (half_W, half_L, RL),
        (-half_W, half_L, RL)
    ]

def frustum_mesh(W1, L1, Z1_rel, W2, L2, Z2_rel, RL_ref, start_v_index):
    """Generates mesh vertices and faces for a rectangular frustum section."""
    if not NUMPY_AVAILABLE: return [], [], [], [], 0
    
    corners1 = get_footprint_corners(W1, L1, Z1_rel, RL_ref)
    corners2 = get_footprint_corners(W2, L2, Z2_rel, RL_ref)
    verts = corners1 + corners2 # 8 vertices in total

    faces = []
    # Faces connect the 4 corners of the bottom to the 4 corners of the top
    for i in range(4):
        i1, i2 = i, (i + 1) % 4      # Bottom corners (0-3)
        i3, i4 = i + 4, (i + 1) % 4 + 4 # Top corners (4-7)

        # Face 1 (Triangle 1): i1, i2, i4
        faces.append([i1, i2, i4])
        # Face 2 (Triangle 2): i1, i4, i3
        faces.append([i1, i4, i3])

    faces_np = np.array(faces) + start_v_index
    
    X = [v[0] for v in verts]
    Y = [v[1] for v in verts]
    Z_abs = [v[2] for v in verts]
    
    return X, Y, Z_abs, faces_np.tolist(), 8 # Return the vertex count

# --- KML Helper Functions (Copied from provided file) ---

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
    x = (math.radians(lon - lon0) * math.cos(math.radians(lat0))) * EARTH_R
    y = (math.radians(lat - lat0)) * EARTH_R
    return x, y

def polygon_area_perimeter_xy(xy_pts: List[Tuple[float,float]]) -> Tuple[float,float]:
    if not xy_pts: return 0.0, 0.0
    
    if xy_pts[0] != xy_pts[-1]:
        pts = xy_pts + [xy_pts[0]]
    else:
        pts = xy_pts
        
    area2 = 0.0
    perim = 0.0
    for i in range(len(pts)-1):
        x1,y1 = pts[i]
        x2,y2 = pts[i+1]
        area2 += (x1*y2 - x2*y1)
        perim += math.hypot(x2-x1, y2-y1)
    return abs(area2)/2.0, perim

def kml_to_area_perimeter(file) -> Tuple[float, float]:
    file.seek(0)
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
    
    avg_ground_rl = st.number_input("Avg. ground RL (m)", 0.0, value=100.0, step=1.0) # NEW: Added RL input for Z-axis context

    st.header("Geometry")
    if mode == "Length & Width":
        L_bl = st.number_input("BL Length (m)", 10.0, value=567.0, step=1.0)
        W_bl = st.number_input("BL Width (m)", 10.0, value=566.15, step=1.0)
    elif mode == "Area & Aspect":
        A_bl = st.number_input("BL Area (m²)", 1000.0, value=321008.457, step=1000.0, format="%.3f")
        aspect = st.slider("Assumed BL L:W", 1.0, 5.0, 1.0, 0.05)
        W_bl = math.sqrt(A_bl/aspect)
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

    st.header("Operations")
    density = st.number_input("Density (t/m³)", 0.3, value=1.0, step=0.05)
    tpa = st.number_input("Waste generation (TPA)", 1000.0, value=365000.0, step=5000.0)
    
    st.header("Visualization Options")
    # Added cross_section_axis for 2D plot, kept separate from 3D for simplicity
    cross_section_axis = st.radio("2D Section View:", ["Width (W)", "Length (L)"]) 

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
    "Level": "BBL", "Length (m)": L_bbl, "Width (m)": W_bbl, "Area (sqm)": A_bbl,
    "Height (m)": depth_bg if depth_bg>0 else None, "Volume (Cum)": V_bbl
})

# BL row (reference, Z_rel=0)
rows.append({
    "Level": "BL", "Length (m)": L_bl, "Width (m)": W_bl, "Area (sqm)": A_bl,
    "Height (m)": None, "Volume (Cum)": None
})

# Iterative ABL with berms
L_prev_berm = L_bl; W_prev_berm = W_bl; A_prev_berm = L_prev_berm * W_prev_berm

i = 1
while True:
    # slope up one lift
    L_i = max(L_prev_berm - 2*m_in*lift_h, 0.0)
    W_i = max(W_prev_berm - 2*m_in*lift_h, 0.0)
    A_i = max(L_i*W_i, 0.0)

    if A_i <= 0: break

    # stop if top area cap reached next
    stop_after = A_i <= (A_bl * top_area_cap_pct/100.0)

    # Volume for this lift
    V_i = frustum_volume(A_prev_berm, A_i, lift_h) if lift_h>0 else 0.0
    cum_vol += V_i

    # ABL i (sloped top) - Height is cumulative for this lift
    rows.append({
        "Level": f"ABL {i}", "Length (m)": L_i, "Width (m)": W_i, "Area (sqm)": A_i,
        "Height (m)": lift_h, "Volume (Cum)": None
    })

    # Berm at top of lift (zero-height transition, records volume)
    L_berm = max(L_i - 2*bench_w, 0.0)
    W_berm = max(W_i - 2*bench_w, 0.0)
    A_berm = max(L_berm*W_berm, 0.0)

    rows.append({
        "Level": f"ABL {i} Berm", "Length (m)": L_berm, "Width (m)": W_berm, "Area (sqm)": A_berm,
        "Height (m)": lift_h, "Volume (Cum)": V_i # Volume recorded here
    })

    # Prepare for next lift
    L_prev_berm, W_prev_berm, A_prev_berm = L_berm, W_berm, A_berm
    if stop_after: break
    i += 1
    if i > 100: break

df = pd.DataFrame(rows)

# Totals
total_height = (depth_bg if depth_bg>0 else 0.0) + (i * lift_h)
total_volume = cum_vol
total_tons = total_volume * density
years = (total_tons / tpa) if tpa>0 else None

st.subheader("Landfill Level Summary Table")

# Display table
def _fmt_display(x, unit=""):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)) or x == 0.0):
        return ""
    if unit:
        return f"{x:,.2f} {unit}"
    return f"{x:,.2f}"

display_df = df.copy()
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
    
    # Filter to the distinct vertical change rows: BBL bottom, BL, and ABL Berms
    df_geom = df[df['Volume (Cum)'].notnull() | (df['Level'] == 'BL') | (df['Level'] == 'BBL')].copy().reset_index(drop=True)
    
    # Z-level calculation: Calculate absolute RL for each distinct level row
    Z_rel_map = {}
    cum_H = 0.0
    for idx, row in df_geom.iterrows():
        if row['Level'] == 'BBL':
             Z_rel_map[idx] = -row['Height (m)'] # BBL bottom (relative to GL=0)
        elif row['Level'] == 'BL':
             Z_rel_map[idx] = 0.0 # BL top (Ground Level)
             cum_H = 0.0
        else: # ABL Berm level
            cum_H += row['Height (m)']
            Z_rel_map[idx] = cum_H
    
    # Build Mesh
    all_X, all_Y, all_Z, all_faces_waste = [], [], [], []
    v_count = 0 
        
    for i in range(len(df_geom) - 1):
        row1 = df_geom.iloc[i]   # Bottom level of frustum
        row2 = df_geom.iloc[i+1] # Top level of frustum
        
        # Determine Z levels (Relative to GL=0)
        Z1_rel = Z_rel_map.get(i, 0.0)
        Z2_rel = Z_rel_map.get(i+1, 0.0)
        
        if Z2_rel <= Z1_rel: continue

        # Rectangular Mode Mesh Generation
        W1, L1 = row1['Width (m)'], row1['Length (m)']
        W2, L2 = row2['Width (m)'], row2['Length (m)']
        
        X_seg, Y_seg, Z_seg, faces_seg, v_num_seg = frustum_mesh(W1, L1, Z1_rel, W2, L2, Z2_rel, RL_ref, v_count)
        
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
    L_bl, W_bl = df[df['Level'] == 'BL'].iloc[0][['Length (m)', 'Width (m)']]
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
        title="3D Landfill Model (Rectangular Profile)", 
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

def generate_section_profile(df: pd.DataFrame, depth_bg: float, m_exc: float, axis: str = "W") -> dict:
    
    # ... (Keep your existing 2D plotting logic here, which is based on L/W metrics)
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
            # For ABL Berm rows, we use the height to determine the Z level
            abl_top_row = df_rect_metrics[df_rect_metrics['Level'] == row['Level'].replace(' Berm', '')].iloc[0]
            W_top_ref = get_dim(abl_top_row)
            
            Z_lift_top = Z_current + row['Height (m)'] 
            x_in_right.append(W_top_ref / 2.0)
            z_in_right.append(Z_lift_top)
            
            W_berm_ref = get_dim(row)
            x_in_right.append(W_berm_ref / 2.0)
            z_in_right.append(Z_lift_top)
            
            Z_current = Z_lift_top
            
    W_top_ref = get_dim(df_rect_metrics.iloc[-1])
    
    x_in_left = [-x for x in x_in_right]; z_in_left = z_in_right
    x_top_plateau = [-W_top_ref / 2.0, W_top_ref / 2.0]
    z_top_plateau = [z_in_right[-1], z_in_right[-1]]

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
    # Placeholder for the matplotlib function, assuming user has it implemented
    # Since it's not provided, we just return None.
    # If the user has a working 2D plot, they should paste that function in.
    return None # Assuming the user's Matplotlib function is omitted here

# ----------------- Display Plots -----------------

# 2. 3D Plot
fig3d = plotly_3d_full_stack(df, avg_ground_rl)
if fig3d:
    st.plotly_chart(fig3d, use_container_width=True)
else:
    st.warning("⚠️ Cannot generate 3D plot. Plotly and Numpy must be available.")
    
# 1. 2D Cross-Section Plot (Placeholder)
section = generate_section_profile(df, depth_bg, m_exc, axis=cross_section_axis[0])
title_caption = f"Cross-section along {cross_section_axis} Axis (Based on Equivalent Rectangular Dimensions)"
img = plot_cross_section(section, title=title_caption)
if img:
    st.image(img, caption=f"2D Cross-section of Stepped Landfill Profile (Waste in Blue, Excavation in Red).")
else:
    st.warning("⚠️ 2D plot function (Matplotlib) is missing/not working. Showing only 3D for the fix.")


# Export (Keep existing export logic)
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

st.caption("Notes: 3D model now shows the stepped profile using the **equivalent rectangular L/W** derived from the KML. BL = Bund Level (starting footprint).")
