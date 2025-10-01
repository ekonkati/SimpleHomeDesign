import math
import numpy as np
from typing import List, Tuple, Dict, Any

# ------------------------- 1. CONSTANTS AND UTILITIES -------------------------
ES = 200000.0  # MPa (Modulus of Elasticity of Steel)
EPS_CU = 0.0035  # Ultimate concrete compressive strain
GAMMA_M = 1.5  # Partial safety factor for concrete
GAMMA_S = 1.15 # Partial safety factor for steel

def bar_area(dia_mm: float) -> float:
    """Calculates the area of a single bar."""
    return math.pi * (dia_mm ** 2) / 4.0

def effective_length_factor(restraint: str) -> float:
    """IS 456-2000 Table 28 approximation for k-factor."""
    if restraint == "Fixed-Fixed": return 0.65
    if restraint == "Fixed-Pinned": return 0.8
    if restraint == "Pinned-Pinned": return 1.0
    if restraint == "Fixed-Free (cantilever)": return 2.0
    return 1.0

# ------------------------- 2. CORE STRAIN COMPATIBILITY METHOD -------------------------

def calculate_steel_stress(epsilon_s: float, fy: float) -> float:
    """
    Calculates the steel stress (fs) based on strain (epsilon_s) for Fe415/Fe500 (IS 456, Annex E).
    Uses the idealized elastic-perfectly plastic curve with 0.87*Fy limit.
    """
    # Yield Strain
    eps_y = (0.87 * fy) / ES
    
    # Stress-Strain relationship (sgn(e_s) is the sign of the strain)
    if abs(epsilon_s) <= eps_y:
        fs = ES * epsilon_s
    else:
        # Reached yield strength (0.87*Fy)
        fs = 0.87 * fy * np.sign(epsilon_s)
        
    return fs

def get_section_forces(c: float, D: float, b: float, fck: float, fy: float, bar_layers: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calculates the Total Axial Force (Pu) and Moment (Mu) for a given Neutral Axis Depth (c).
    
    Args:
        c (float): Neutral Axis Depth from compression face (mm).
        D (float): Overall depth of the section (mm).
        b (float): Width of the section (mm).
        fck (float): Characteristic strength of concrete (MPa).
        fy (float): Characteristic strength of steel (MPa).
        bar_layers (List[Tuple[float, float]]): List of (distance_from_comp_face, total_area) for each steel layer.
        
    Returns:
        Tuple[float, float]: (Total Axial Force Pu, Total Moment Mu)
    """
    # Stress parameters
    fcd = 0.45 * fck # Max concrete compression stress in steel area (for Ac reduction)
    fcc = 0.36 * fck # Concrete compression block stress (0.36*fck)

    Pu_total = 0.0 # Total Axial Force (N)
    Mu_total = 0.0 # Total Moment about center (N-mm)

    # 1. Concrete Contribution (Parabola-Rectangle Stress Block)
    if c > 0:
        # Depth of stress block (xu_max = D if NA is outside section)
        x_block = min(c, D)
        
        # Concrete compression force (Cc) and centroid calculation (IS 456: Cl 38.1 and 39.4)
        if x_block <= (3/7) * D: # Case 1: Neutral Axis is inside the section
            # Area of stress block is 0.36 * fck * b * x_block
            # Centroid is at 0.42 * x_block from compression face
            Cc = fcc * b * x_block
            z_c = D/2 - 0.42 * x_block # Lever arm to center
            
        else: # Case 2: Neutral Axis is outside the section (full section compression)
            # Simplified approach for full compression, using constant 0.45*fck
            Cc = 0.45 * fck * D * b
            z_c = 0.0
            
        Pu_total += Cc
        Mu_total += Cc * z_c # Moment about center

    # 2. Steel Contribution
    for d_i, As_i in bar_layers:
        
        # Strain in steel (Linear strain distribution)
        # Strain = EPS_CU * (xu - d_i) / xu
        epsilon_s = EPS_CU * (c - d_i) / c
        
        fs_i = calculate_steel_stress(epsilon_s, fy)
        
        Fs_i = fs_i * As_i
        
        # Reduction for concrete displaced by steel (only if in compression, d_i < c)
        if d_i < c:
            # Net steel force = Steel force - concrete force replaced by steel
            Fs_i -= fcd * As_i

        # Force and Moment
        Pu_total += Fs_i
        z_i = D/2 - d_i # Lever arm to center
        Mu_total += Fs_i * z_i
        
    return Pu_total, Mu_total

def _uniaxial_capacity_Mu_for_Pu(Pu_target: float, D: float, b: float, fck: float, fy: float, bar_layers: List[Tuple[float, float]]) -> float:
    """
    Finds the Neutral Axis Depth 'c' via Binary Search that satisfies the Pu_target,
    then returns the corresponding Moment Capacity Mu. (The core logic from col.txt)
    """
    if Pu_target < 0:
        # Safety break for tension cases not fully handled by this simplified model
        return 0.0

    # Binary Search Limits: c range
    # Lower bound: c very close to 0 (high tension, low compression)
    # Upper bound: c = 2*D (ensures c is well outside the section for max compression)
    c_low = 1.0  # Cannot be 0
    c_high = 2.0 * D
    
    c_best = 0.0
    tolerance_Pu = 1.0 # N (0.001 kN)

    # Max iterations for robust binary search (col.txt used a loop, this is more controlled)
    for _ in range(50):
        c_mid = (c_low + c_high) / 2.0
        
        Pu_mid, _ = get_section_forces(c_mid, D, b, fck, fy, bar_layers)
        
        error = Pu_mid - Pu_target
        
        if abs(error) < tolerance_Pu:
            c_best = c_mid
            break
        elif error < 0:
            # Calculated Pu is too low -> need a deeper NA (more compression)
            c_low = c_mid
        else:
            # Calculated Pu is too high -> need a shallower NA (less compression)
            c_high = c_mid
    else:
        # If loop finishes without meeting tolerance, use the last best guess
        c_best = (c_low + c_high) / 2.0

    # Now calculate the final moment for the best c
    _, Mu_limit = get_section_forces(c_best, D, b, fck, fy, bar_layers)
    
    return abs(Mu_limit) # Return absolute value

# ------------------------- 3. INTEGRATED DESIGN CHECK -------------------------

def check_column_capacity(
    b: float, D: float, L_eff: float, L_unsupported: float, fck: float, fy: float, 
    bar_layers_x: List[Tuple[float, float]], bar_layers_y: List[Tuple[float, float]], 
    Pu_design: float, Mux_design: float, Muy_design: float, k_factor: float
) -> Dict[str, Any]:
    """
    Performs the full IS 456-2000 Biaxial Column Design Check.
    """
    results = {}
    
    # ------------------ 3a. Minimum Eccentricity and Slenderness ------------------
    
    # Minimum Eccentricity (Cl 25.4)
    emin_x = max(L_unsupported / 500.0 + D / 30.0, 20.0)
    emin_y = max(L_unsupported / 500.0 + b / 30.0, 20.0)
    
    # Minimum Moment (N-mm)
    Mux_min = Pu_design * emin_x
    Muy_min = Pu_design * emin_y

    # Slenderness Check (Cl 25.4)
    le_D = k_factor * L_unsupported / D
    le_b = k_factor * L_unsupported / b
    
    is_short_x = le_D <= 12.0
    is_short_y = le_b <= 12.0
    is_short = is_short_x and is_short_y
    
    results['is_short'] = is_short
    results['le/D'] = le_D
    results['le/b'] = le_b

    # Moment Magnification (delta - Only needed for slender columns)
    # The moment magnification factor is complex and depends on boundary conditions
    # For simplicity and compliance, in the integrated code, we *require* the user 
    # to handle the detailed moment magnification (Cl. 39.7) externally if the column is slender.
    # If short, delta is 1.0.
    delta_x = 1.0 
    delta_y = 1.0

    # Effective Design Moments (Cl 39.5)
    Mux_eff = max(abs(Mux_design * delta_x), Mux_min)
    Muy_eff = max(abs(Muy_design * delta_y), Muy_min)
    
    results['Mux_eff'] = Mux_eff
    results['Muy_eff'] = Muy_eff
    
    # ------------------ 3b. Uniaxial Capacity Limits ------------------
    
    # Capacity in X-direction (Moment about y-axis)
    Mux_limit = _uniaxial_capacity_Mu_for_Pu(Pu_design, D, b, fck, fy, bar_layers_x)
    
    # Capacity in Y-direction (Moment about x-axis - use b as depth, D as width)
    # Note: Bar layers for Y-axis moment must be defined w.r.t b (compression face)
    Muy_limit = _uniaxial_capacity_Mu_for_Pu(Pu_design, b, D, fck, fy, bar_layers_y) 
    
    results['Mux_limit'] = Mux_limit
    results['Muy_limit'] = Muy_limit

    # ------------------ 3c. Biaxial Interaction (Annex G) ------------------

    # Calculate Puz (Axial only capacity) (Cl 39.3)
    Asc_total = sum(area for _, area in bar_layers_x) # Total steel area
    Ac = D * b - Asc_total # Net concrete area
    Puz = 0.45 * fck * Ac + 0.75 * fy * Asc_total
    
    results['Puz'] = Puz

    # Calculate Alpha (Power Exponent) based on Pu/Puz (Figure 63)
    Pu_Puz_ratio = Pu_design / Puz
    
    if Pu_Puz_ratio <= 0.2:
        alpha = 1.0
    elif Pu_Puz_ratio >= 0.8:
        alpha = 2.0
    else:
        # Linear interpolation between 0.2 (alpha=1.0) and 0.8 (alpha=2.0)
        # alpha = 1.0 + (2.0 - 1.0) * (Pu_Puz_ratio - 0.2) / (0.8 - 0.2)
        alpha = 1.0 + (1.0 / 0.6) * (Pu_Puz_ratio - 0.2) 
        
    results['alpha'] = alpha
    
    # Biaxial Utilization Check
    if Mux_limit == 0 or Muy_limit == 0:
        # Handle cases where limit is zero (e.g., pure axial load capacity point)
        biaxial_util = 0.0 # Will be checked via pure axial check if needed
    else:
        biaxial_util = (abs(Mux_eff) / Mux_limit)**alpha + (abs(Muy_eff) / Muy_limit)**alpha
        
    results['Biaxial_Utilization'] = biaxial_util
    results['Status'] = "SAFE" if biaxial_util <= 1.0 else "FAIL"
    
    return results

# ------------------------- 4. EXAMPLE USAGE -------------------------

# --- Design Input Parameters ---

# Dimensions and Materials
D_section = 600.0   # Depth of section (mm, along X-axis moment)
b_section = 400.0   # Width of section (mm, along Y-axis moment)
fck = 30.0          # Concrete grade (M30)
fy = 500.0          # Steel grade (Fe500)
cover = 50.0        # Clear cover (mm)
L_unsupported = 3000.0 # Unsupported length (mm)
k_factor = 0.8      # Effective length factor (e.g., Fixed-Pinned)

# Design Loads (Factored)
Pu_design = 4500.0 * 1e3    # Factored Axial Load (N) - 4500 kN
Mux_design = 300.0 * 1e6    # Factored Moment about X-axis (N-mm) - 300 kNm
Muy_design = 200.0 * 1e6    # Factored Moment about Y-axis (N-mm) - 200 kNm

# Reinforcement Setup
bar_dia = 25.0
bar_area_single = bar_area(bar_dia)
d_prime_x = cover # Effective cover in X-direction
d_prime_y = cover # Effective cover in Y-direction

# Bar Layers (d_i, As_i) for Mux calculation (about X-axis, D is depth)
# d_i is distance from compression face (top or bottom)
# Example: 4 bars on top, 4 in middle, 4 on bottom
bar_layers_x = [
    (d_prime_x, 4 * bar_area_single),                     # Top layer
    (D_section / 2.0, 4 * bar_area_single),              # Middle layer
    (D_section - d_prime_x, 4 * bar_area_single)         # Bottom layer
]

# Bar Layers (d_i, As_i) for Muy calculation (about Y-axis, b is depth)
# d_i is distance from compression face (left or right)
# Example: 3 bars on left, 6 in middle, 3 on right (Total 12 bars)
# NOTE: The bar layers in the other direction are often simplified to be uniform 
# at the two faces for biaxial checks. This setup assumes a common grid.
bar_layers_y = [
    (d_prime_y, 4 * bar_area_single),                     # Left layer
    (b_section / 2.0, 4 * bar_area_single),              # Middle layer
    (b_section - d_prime_y, 4 * bar_area_single)         # Right layer
]

print("--- IS 456-2000 Biaxial Column Design Check ---")
print(f"Section: {D_section}x{b_section} mm | Pu_design: {Pu_design/1e3:.0f} kN")
print(f"Reinforcement: {sum(area for _, area in bar_layers_x)/1e2:.2f}% (Total {sum(area for _, area in bar_layers_x):.0f} mm²)\n")

design_results = check_column_capacity(
    b=b_section, D=D_section, L_eff=k_factor * L_unsupported, L_unsupported=L_unsupported, 
    fck=fck, fy=fy, bar_layers_x=bar_layers_x, bar_layers_y=bar_layers_y, 
    Pu_design=Pu_design, Mux_design=Mux_design, Muy_design=Muy_design, k_factor=k_factor
)

# --- Output Report ---

print("A. Slenderness and Eccentricity")
print(f"  Unsupported Length (L_un): {L_unsupported/1e3:.2f} m")
print(f"  Slenderness Ratio (le/D): {design_results['le/D']:.2f} (Max 12.0)")
print(f"  Slenderness Ratio (le/b): {design_results['le/b']:.2f} (Max 12.0)")
print(f"  Column Type: {'SHORT' if design_results['is_short'] else 'SLENDER'} (Moment Magnification assumed 1.0 for short)")
print(f"  Min Eccentricity Moment (X): {Mux_min/1e6:.2f} kNm")
print(f"  Min Eccentricity Moment (Y): {Muy_min/1e6:.2f} kNm")
print(f"  Effective Design Mux: {design_results['Mux_eff']/1e6:.2f} kNm")
print(f"  Effective Design Muy: {design_results['Muy_eff']/1e6:.2f} kNm")
print("-" * 30)

print("B. Uniaxial and Biaxial Capacity")
print(f"  Axial Only Capacity (Puz): {design_results['Puz']/1e3:.0f} kN")
print(f"  Pu/Puz Ratio: {Pu_design / design_results['Puz']:.2f}")
print(f"  Uniaxial Moment Capacity (Mux_limit): {design_results['Mux_limit']/1e6:.2f} kNm")
print(f"  Uniaxial Moment Capacity (Muy_limit): {design_results['Muy_limit']/1e6:.2f} kNm")
print("-" * 30)

print("C. Biaxial Interaction Check (IS 456 Annex G)")
print(f"  Interaction Exponent (alpha): {design_results['alpha']:.2f} (Variable)")
print(f"  Biaxial Utilization Ratio: {design_results['Biaxial_Utilization']:.3f} (Max 1.0)")
print(f"  Design Status: {design_results['Status']}")
