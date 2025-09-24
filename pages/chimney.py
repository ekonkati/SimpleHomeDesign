
# Landfill Sizing, Capacity, Cost & Leachate Calculator (Streamlit)
# ---------------------------------------------------------------
# Single-file app implementing:
# - Inputs for waste/time, site footprint, bund/embankment, lift-step geometry, costing, leachate
# - Two calculation modes: Quick Height model and Lift-step model
# - Validations, tier-wise table, charts (matplotlib), narrative text
# - Export: Excel (inputs+results+tier table), HTML report, best-effort PDF (if reportlab present)

import io
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Optional dependency for PDF export (fallback to HTML if missing)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.units import cm
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# ----------------------------
# Utility & Data Structures
# ----------------------------

ACRE_IN_M2 = 4046.8564224

@dataclass
class WasteTimeInputs:
    tpd: float = 1000.0
    working_days: int = 365
    link_tpa: bool = True
    tpa_override: float = 365000.0
    design_years: float = 25.0
    density_t_per_m3: float = 1.0

@dataclass
class SiteInputs:
    site_area_acres: float = 89.0
    plan_width_m: float = 600.15248
    plan_length_m: float = 601.0
    solve_dimension: bool = False
    aspect_ratio_w_over_l: float = 1.0

@dataclass
class BundInputs:
    bund_width_m: float = 5.0
    bund_height_m: float = 5.0
    external_slope_h: float = 2.0
    external_slope_v: float = 1.0
    internal_slope_h: float = 3.0
    internal_slope_v: float = 1.0

@dataclass
class LiftInputs:
    use_lift_model: bool = True
    lift_height_m: float = 5.0
    berm_width_m: float = 4.0
    waste_slope_h: float = 3.0
    waste_slope_v: float = 1.0
    avg_height_quick_m: float = 25.0
    include_cap: bool = False
    cap_slope_h: float = 3.0
    cap_slope_v: float = 1.0

@dataclass
class CostInputs:
    basis_per_m2: bool = True  # else per m3
    rate_per_m2: float = 3850.0
    rate_per_m3: float = 0.0
    currency: str = "₹"

@dataclass
class LeachateInputs:
    pct_of_tpa: float = 10.0
    one_ton_equals_one_kl: bool = True


# ----------------------------
# Core Calculations
# ----------------------------

def compute_tpa(w: WasteTimeInputs) -> float:
    if w.link_tpa:
        return w.tpd * w.working_days
    return w.tpa_override

def compute_total_quantity_tons(tpa: float, years: float) -> float:
    return tpa * years

def compute_required_volume_m3(total_tons: float, density: float) -> float:
    return total_tons / max(density, 1e-9)

def site_area_m2(s: SiteInputs) -> float:
    return s.site_area_acres * ACRE_IN_M2

def plan_area_m2(s: SiteInputs) -> float:
    return s.plan_width_m * s.plan_length_m

def inside_bund_dimensions(s: SiteInputs, b: BundInputs) -> Tuple[float, float]:
    """
    Compute the inside-bund rectangle by offsetting inward from plan dimensions:
    inward offset per side = bund_width + (internal slope run) = wb + (H/V * height)
    """
    offset = b.bund_width_m + (b.internal_slope_h / max(b.internal_slope_v, 1e-9)) * b.bund_height_m
    wi = s.plan_width_m - 2.0 * offset
    li = s.plan_length_m - 2.0 * offset
    return max(wi, 0.0), max(li, 0.0)

def tier_geometry_and_volume(
    base_w: float,
    base_l: float,
    lift: LiftInputs,
) -> Tuple[pd.DataFrame, float]:
    """
    Given base (inside-bund) rectangle and lift parameters, compute tier-by-tier
    rectangles and volumes using frustum of right prism approximation.
    Returns (df, total_volume_m3).
    """
    # Safety: no negative base
    if base_w <= 0 or base_l <= 0:
        return pd.DataFrame(columns=["tier","Wi","Li","Ai","Wi_next","Li_next","Ai_next","setback","berm","Vi"]), 0.0

    # Assume we keep adding tiers until geometry collapses or until N chosen by avg height proxy.
    # We can estimate a max theoretical number of tiers from a guessed peak height.
    # Instead, we iterate until Wi_next or Li_next becomes non-positive.
    rows = []
    total_v = 0.0

    Wi = base_w
    Li = base_l
    Ai = Wi * Li

    tier = 0
    while True:
        tier += 1
        # Horizontal setback per side due to waste slope on this lift
        setback = (lift.waste_slope_h / max(lift.waste_slope_v, 1e-9)) * lift.lift_height_m
        # Total offset per side for the *next* rectangle = slope setback + berm width
        total_offset = setback + lift.berm_width_m

        Wi_next = Wi - 2.0 * total_offset
        Li_next = Li - 2.0 * total_offset
        if Wi_next <= 0 or Li_next <= 0:
            # No complete next tier—stop before creating non-positive top
            break

        Ai_next = Wi_next * Li_next
        # Frustum volume between Ai (bottom) and Ai_next (top) of height = lift_height
        Vi = ((Ai + Ai_next) / 2.0) * lift.lift_height_m

        rows.append({
            "tier": tier,
            "Wi": Wi,
            "Li": Li,
            "Ai": Ai,
            "Wi_next": Wi_next,
            "Li_next": Li_next,
            "Ai_next": Ai_next,
            "setback": setback,
            "berm": lift.berm_width_m,
            "Vi": Vi
        })
        total_v += Vi

        # advance
        Wi, Li, Ai = Wi_next, Li_next, Ai_next

        # Optional cap (single, small shell) – implemented as a no-height geometric check (skip volume)
        if lift.include_cap:
            # we simply allow one more "geometry" shrink for viewing if needed (no volume)
            pass

        # Guard against runaway loop
        if tier > 200:
            break

    df = pd.DataFrame(rows)
    return df, total_v

def quick_model_volume(area_m2: float, avg_height_m: float) -> float:
    return max(area_m2, 0.0) * max(avg_height_m, 0.0)

def cost_calculation(cost: CostInputs, charge_area_m2: float, volume_m3: float) -> Tuple[float, float]:
    if cost.basis_per_m2:
        total = charge_area_m2 * cost.rate_per_m2
        per_m3 = total / max(volume_m3, 1e-9) if volume_m3 > 0 else 0.0
    else:
        total = volume_m3 * cost.rate_per_m3
        per_m3 = cost.rate_per_m3
    return total, per_m3

def leachate_calc(leach: LeachateInputs, tpa: float) -> Tuple[float, float]:
    annual_tons = tpa * (leach.pct_of_tpa / 100.0)
    kld = (annual_tons / 365.0) if leach.one_ton_equals_one_kl else 0.0
    return annual_tons, kld


# ----------------------------
# Report Generators
# ----------------------------

def make_excel_download(inputs: Dict[str, Dict], results: Dict[str, float], tier_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Inputs sheet
        inp_rows = []
        for group, vals in inputs.items():
            for k, v in vals.items():
                inp_rows.append({"Group": group, "Parameter": k, "Value": v})
        pd.DataFrame(inp_rows).to_excel(writer, sheet_name="Inputs", index=False)

        # Results sheet
        pd.DataFrame([results]).T.reset_index().rename(columns={"index": "Metric", 0: "Value"}).to_excel(
            writer, sheet_name="Results", index=False
        )

        # Tier table
        if not tier_df.empty:
            tier_df.to_excel(writer, sheet_name="Tiers", index=False)

    return output.getvalue()

def make_html_report(title: str, inputs: Dict[str, Dict], results: Dict[str, float], tier_df: pd.DataFrame, charts_png: Dict[str, bytes]) -> str:
    def fmt_row(k, v):
        return f"<tr><td>{k}</td><td>{v}</td></tr>"

    inputs_html = []
    for group, vals in inputs.items():
        rows = "".join(fmt_row(k, v) for k, v in vals.items())
        inputs_html.append(f"<h3>{group}</h3><table border='1' cellspacing='0' cellpadding='4'>{rows}</table>")
    inputs_html = "".join(inputs_html)

    results_rows = "".join(fmt_row(k, v) for k, v in results.items())
    results_html = f"<h3>Results</h3><table border='1' cellspacing='0' cellpadding='4'>{results_rows}</table>"

    tiers_html = ""
    if not tier_df.empty:
        tiers_html = "<h3>Tiers</h3>" + tier_df.to_html(index=False)

    charts_html = ""
    for name, img in charts_png.items():
        b64 = base64.b64encode(img).decode("utf-8")
        charts_html += f"<h3>{name}</h3><img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto;'/>"

    html = f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>{title}</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        h1 {{ margin-bottom: 6px; }}
        h3 {{ margin-top: 24px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        td, th {{ border: 1px solid #999; padding: 6px 8px; }}
      </style>
    </head>
    <body>
      <h1>{title}</h1>
      {results_html}
      {tiers_html}
      <h2>Inputs</h2>
      {inputs_html}
      <h2>Charts</h2>
      {charts_html}
    </body>
    </html>
    """
    return html

def make_pdf_from_text(title: str, narrative: str, results: Dict[str, float], tier_df: pd.DataFrame, charts_png: Dict[str, bytes]) -> bytes:
    """
    Best-effort simple PDF using reportlab if available.
    If reportlab is missing, returns empty bytes.
    """
    if not REPORTLAB_AVAILABLE:
        return b""

    buffer = io.BytesIO()
    c = rl_canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    def draw_multiline(text, x, y, max_width):
        # Very simple word wrap
        lines = []
        for para in text.split("\n"):
            words = para.split(" ")
            line = ""
            for w in words:
                test = (line + " " + w).strip()
                if c.stringWidth(test, "Helvetica", 9) <= max_width:
                    line = test
                else:
                    lines.append(line)
                    line = w
            lines.append(line)
        # draw from top to bottom
        cur_y = y
        for line in lines:
            c.drawString(x, cur_y, line)
            cur_y -= 10
            if cur_y < 2*cm:
                c.showPage(); c.setFont("Helvetica", 9); cur_y = height - 2*cm
        return cur_y

    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, height - 2*cm, title)

    c.setFont("Helvetica", 9)
    y = height - 3*cm
    y = draw_multiline("Narrative:\n" + narrative, 2*cm, y, width - 4*cm)

    y -= 10
    y = draw_multiline("Key Results:", 2*cm, y, width - 4*cm)
    for k, v in results.items():
        line = f"- {k}: {v}"
        c.drawString(2*cm, y, line)
        y -= 10
        if y < 2*cm:
            c.showPage(); c.setFont("Helvetica", 9); y = height - 2*cm

    # Simple tier table first rows
    if not tier_df.empty:
        y -= 10
        c.drawString(2*cm, y, "Tier Table (first 20 rows):")
        y -= 12
        cols = ["tier","Wi","Li","Ai","Wi_next","Li_next","Ai_next","Vi"]
        c.setFont("Helvetica", 7)
        for _, row in tier_df.head(20).iterrows():
            line = " | ".join(f"{col}:{round(row[col],3) if isinstance(row[col], (int,float,np.floating)) else row[col]}" for col in cols if col in row)
            c.drawString(2*cm, y, line[:180])
            y -= 9
            if y < 2*cm:
                c.showPage(); c.setFont("Helvetica", 7); y = height - 2*cm

    # Charts
    if charts_png:
        c.showPage()
        c.setFont("Helvetica", 9)
        y = height - 2*cm
        for name, img in charts_png.items():
            # Place one chart per page for simplicity
            c.drawString(2*cm, y, name)
            y -= 12
            try:
                from reportlab.lib.utils import ImageReader
                ir = ImageReader(io.BytesIO(img))
                img_w, img_h = ir.getSize()
                # scale to fit
                max_w = width - 4*cm
                max_h = height - 6*cm
                scale = min(max_w / img_w, max_h / img_h)
                c.drawImage(ir, 2*cm, 3*cm, img_w*scale, img_h*scale, preserveAspectRatio=True, mask='auto')
            except Exception:
                c.drawString(2*cm, y, "(Chart render error)")
            c.showPage()
    c.save()
    return buffer.getvalue()


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Landfill Capacity & Leachate Calculator", layout="wide")

st.title("Landfill Sizing, Capacity, Cost & Leachate Calculator")
st.caption("Single-file Streamlit app with per-tier geometry, costing, and leachate.")

with st.sidebar:
    st.header("A) Waste & Time")
    w = WasteTimeInputs()
    w.tpd = st.number_input("Capacity (TPD)", min_value=0.0, value=w.tpd, step=50.0)
    w.working_days = st.number_input("Working days/year", min_value=1, value=w.working_days, step=1)
    w.link_tpa = st.toggle("Link TPA = TPD × days", value=w.link_tpa)
    if not w.link_tpa:
        w.tpa_override = st.number_input("TPA (override)", min_value=0.0, value=w.tpa_override, step=500.0)
    w.design_years = st.number_input("Design duration (years)", min_value=0.0, value=w.design_years, step=1.0)
    w.density_t_per_m3 = st.number_input("Waste density (t/m³)", min_value=0.001, value=w.density_t_per_m3, step=0.1)

    st.header("B) Site / Footprint")
    s = SiteInputs()
    s.site_area_acres = st.number_input("Site area (acres)", min_value=0.0, value=s.site_area_acres, step=0.5)
    s.solve_dimension = st.toggle("Solve one plan dimension from site area + aspect ratio", value=s.solve_dimension)
    if s.solve_dimension:
        s.aspect_ratio_w_over_l = st.number_input("Assumed aspect ratio (W/L)", min_value=0.01, value=s.aspect_ratio_w_over_l, step=0.05)
        area_m2_val = site_area_m2(s)
        # Solve W and L given area and ratio r = W/L -> A = W*L = r*L^2 => L = sqrt(A/r), W = r*L
        L = math.sqrt(max(area_m2_val, 0.0) / max(s.aspect_ratio_w_over_l, 1e-9))
        W = s.aspect_ratio_w_over_l * L
        s.plan_width_m = st.number_input("Plan width W (m)", min_value=0.0, value=float(W), step=1.0, help="Auto-computed; editable")
        s.plan_length_m = st.number_input("Plan length L (m)", min_value=0.0, value=float(L), step=1.0, help="Auto-computed; editable")
    else:
        s.plan_width_m = st.number_input("Plan width W (m)", min_value=0.0, value=s.plan_width_m, step=0.5)
        s.plan_length_m = st.number_input("Plan length L (m)", min_value=0.0, value=s.plan_length_m, step=0.5)

    st.header("C) Bund / Embankment")
    b = BundInputs()
    b.bund_width_m = st.number_input("Bund width (m)", min_value=0.0, value=b.bund_width_m, step=0.5)
    b.bund_height_m = st.number_input("Bund height (m)", min_value=0.0, value=b.bund_height_m, step=0.5)
    b.external_slope_h = st.number_input("External slope H (H:V)", min_value=0.0, value=b.external_slope_h, step=0.5)
    b.external_slope_v = st.number_input("External slope V (H:V)", min_value=0.1, value=b.external_slope_v, step=0.1)
    b.internal_slope_h = st.number_input("Internal slope H (H:V)", min_value=0.0, value=b.internal_slope_h, step=0.5)
    b.internal_slope_v = st.number_input("Internal slope V (H:V)", min_value=0.1, value=b.internal_slope_v, step=0.1)

    st.header("D) Geometry (Lifts & Cover)")
    l = LiftInputs()
    l.use_lift_model = st.toggle("Use Lift-step model (recommended)", value=l.use_lift_model)
    l.lift_height_m = st.number_input("Lift height (m)", min_value=0.0, value=l.lift_height_m, step=0.5)
    l.berm_width_m = st.number_input("Berm width per tier (m)", min_value=0.0, value=l.berm_width_m, step=0.5)
    l.waste_slope_h = st.number_input("Waste slope H (H:V)", min_value=0.0, value=l.waste_slope_h, step=0.5)
    l.waste_slope_v = st.number_input("Waste slope V (H:V)", min_value=0.1, value=l.waste_slope_v, step=0.1)
    l.include_cap = st.toggle("Include cap (visual only; no extra volume)", value=l.include_cap)
    if not l.use_lift_model:
        l.avg_height_quick_m = st.number_input("Avg height (Quick model, m)", min_value=0.0, value=l.avg_height_quick_m, step=0.5)

    st.header("E) Costing")
    cst = CostInputs()
    cst.basis_per_m2 = st.toggle("Cost basis: ₹/m² (else ₹/m³)", value=cst.basis_per_m2)
    if cst.basis_per_m2:
        cst.rate_per_m2 = st.number_input("Rate (₹/m²)", min_value=0.0, value=cst.rate_per_m2, step=50.0)
    else:
        cst.rate_per_m3 = st.number_input("Rate (₹/m³)", min_value=0.0, value=cst.rate_per_m3, step=10.0)
    cst.currency = st.text_input("Currency symbol/text", value=cst.currency)

    st.header("F) Leachate")
    le = LeachateInputs()
    le.pct_of_tpa = st.number_input("Leachate as % of TPA", min_value=0.0, value=le.pct_of_tpa, step=1.0)
    le.one_ton_equals_one_kl = st.toggle("Assume 1 ton ≈ 1 kL", value=le.one_ton_equals_one_kl)


# ----------------------------
# Calculations
# ----------------------------

tpa = compute_tpa(w)
total_tons = compute_total_quantity_tons(tpa, w.design_years)
required_volume = compute_required_volume_m3(total_tons, w.density_t_per_m3)

site_area_val = site_area_m2(s)
plan_area_val = plan_area_m2(s)

Wi_in, Li_in = inside_bund_dimensions(s, b)
inside_bund_area = Wi_in * Li_in

# Lift-step or Quick model
tier_df = pd.DataFrame()
volume_achievable = 0.0
if l.use_lift_model:
    tier_df, volume_achievable = tier_geometry_and_volume(Wi_in, Li_in, l)
else:
    # Quick model uses plan area (user can decide to use site area instead by toggling solve behavior)
    volume_achievable = quick_model_volume(plan_area_val, l.avg_height_quick_m)

# Convert achievable volume to tons & life
achievable_tons = volume_achievable * w.density_t_per_m3
life_years_from_footprint = achievable_tons / tpa if tpa > 0 else 0.0

# Costing (default chargeable area = plan area; users often use site area—provide a selector)
charge_area_choice = st.selectbox("Select chargeable area for ₹/m² costing", ["Plan area (W×L)", "Site area (acres→m²)"], index=0)
charge_area = plan_area_val if charge_area_choice.startswith("Plan") else site_area_val
total_amount, per_m3_rate = cost_calculation(cst, charge_area, volume_achievable)
per_ton_cost = total_amount / max(achievable_tons, 1e-9) if achievable_tons > 0 else 0.0

# Leachate
annual_leachate_tons, leachate_kld = leachate_calc(le, tpa)


# ----------------------------
# Validation Warnings
# ----------------------------

warnings = []
if s.plan_width_m <= 0 or s.plan_length_m <= 0:
    warnings.append("Plan dimensions must be > 0.")
if Wi_in <= 0 or Li_in <= 0:
    warnings.append("Inside-bund footprint is non-positive. Bund width + internal slope may be too large for plan.")
if l.use_lift_model and tier_df.empty:
    warnings.append("No tiers could be formed with current lift/berm/slope parameters. Reduce berm width or slope, or increase base footprint.")
if w.density_t_per_m3 <= 0:
    warnings.append("Density must be > 0.")
if w.design_years < 0:
    warnings.append("Design duration must be non-negative.")

for msg in warnings:
    st.warning(msg)


# ----------------------------
# At-a-glance Cards
# ----------------------------

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("TPA (tons/yr)", f"{tpa:,.0f}")
col2.metric("Total (tons) over design", f"{total_tons:,.0f}")
col3.metric("Required volume (m³)", f"{required_volume:,.0f}")
col4.metric("Achievable volume (m³)", f"{volume_achievable:,.0f}")
col5.metric("Achievable life (years)", f"{life_years_from_footprint:,.2f}")
col6.metric(f"Total Cost ({cst.currency})", f"{total_amount:,.0f}")

col7, col8, col9 = st.columns(3)
col7.metric("₹/ton", f"{per_ton_cost:,.2f}")
col8.metric("Chargeable area (m²)", f"{charge_area:,.0f}")
col9.metric("Leachate (KLD)", f"{leachate_kld:,.1f}")


# ----------------------------
# Details & Tables
# ----------------------------

with st.expander("Waste & Time Summary", expanded=False):
    st.write(pd.DataFrame({
        "Metric": ["TPD", "Working days/yr", "TPA", "Design years", "Density (t/m³)", "Total tons", "Required volume (m³)"],
        "Value": [w.tpd, w.working_days, tpa, w.design_years, w.density_t_per_m3, total_tons, required_volume]
    }))

with st.expander("Site & Bund Geometry", expanded=False):
    st.write(pd.DataFrame({
        "Metric": ["Site area (m²)", "Site area (acres)", "Plan width W (m)", "Plan length L (m)", "Plan area (m²)",
                   "Inside-bund W (m)", "Inside-bund L (m)", "Inside-bund area (m²)"],
        "Value": [site_area_val, s.site_area_acres, s.plan_width_m, s.plan_length_m, plan_area_val, Wi_in, Li_in, inside_bund_area]
    }))

if l.use_lift_model:
    with st.expander("Tier-wise Geometry & Volume (Lift-step model)", expanded=True):
        st.write(tier_df)
else:
    st.info("Quick Height model is ON (no tier table).")

# ----------------------------
# Charts (matplotlib)
# ----------------------------

charts_png: Dict[str, bytes] = {}

if l.use_lift_model and not tier_df.empty:
    # Chart 1: Cumulative Volume vs Tier
    fig1, ax1 = plt.subplots()
    cum_vol = tier_df["Vi"].cumsum()
    ax1.plot(tier_df["tier"], cum_vol, marker="o")
    ax1.set_xlabel("Tier")
    ax1.set_ylabel("Cumulative Volume (m³)")
    ax1.set_title("Cumulative Volume vs Tier")
    st.pyplot(fig1)
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", bbox_inches="tight")
    charts_png["Cumulative Volume vs Tier"] = buf1.getvalue()

    # Chart 2: Area shrinkage Wi×Li per tier (top areas)
    fig2, ax2 = plt.subplots()
    ax2.plot(tier_df["tier"], tier_df["Ai_next"], marker="o")
    ax2.set_xlabel("Tier")
    ax2.set_ylabel("Top Area (m²) of Each Tier")
    ax2.set_title("Plan Area Shrinkage per Tier (Top Areas)")
    st.pyplot(fig2)
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches="tight")
    charts_png["Plan Area Shrinkage per Tier"] = buf2.getvalue()


# ----------------------------
# Narrative (auto-generated)
# ----------------------------

narrative_lines = []
narrative_lines.append("This report computes landfill capacity, life, costing, and leachate based on user inputs.")
narrative_lines.append(f"Total waste per annum (TPA) is computed as TPD × working days if linked; here TPA = {tpa:,.0f} t/yr.")
narrative_lines.append(f"Over {w.design_years} years, the total waste equals {total_tons:,.0f} tons requiring {required_volume:,.0f} m³ at a density of {w.density_t_per_m3} t/m³.")
narrative_lines.append(f"Plan footprint {s.plan_width_m:.2f} m × {s.plan_length_m:.2f} m; inside-bund footprint is {Wi_in:.2f} m × {Li_in:.2f} m based on bund width and internal slopes.")
if l.use_lift_model:
    narrative_lines.append("Capacity is evaluated by a lift-step model using frustum volumes between successive tier rectangles defined by slope setback and berm width.")
    narrative_lines.append(f"The achievable infill volume is {volume_achievable:,.0f} m³ corresponding to {achievable_tons:,.0f} tons and an estimated life of {life_years_from_footprint:,.2f} years at the given TPA.")
else:
    narrative_lines.append("Capacity is evaluated using a Quick Height model (plan area × average height).")
    narrative_lines.append(f"The achievable infill volume is {volume_achievable:,.0f} m³ corresponding to {achievable_tons:,.0f} tons and an estimated life of {life_years_from_footprint:,.2f} years.")

narrative_lines.append(f"Costing uses {'area' if cst.basis_per_m2 else 'volume'} basis with rate {cst.currency} {cst.rate_per_m2 if cst.basis_per_m2 else cst.rate_per_m3:,.2f} per {'m²' if cst.basis_per_m2 else 'm³'}, chargeable area = {charge_area:,.0f} m².")
narrative_lines.append(f"Total amount: {cst.currency} {total_amount:,.0f}; Unit cost ≈ {cst.currency} {per_ton_cost:,.2f} per ton (at achievable capacity).")
narrative_lines.append(f"Leachate assumed at {le.pct_of_tpa:.1f}% of TPA yields {annual_leachate_tons:,.0f} t/yr; reported as {leachate_kld:,.1f} kL/day when 1 t≈1 kL.")

narrative = "\n".join(narrative_lines)

# ----------------------------
# Exports
# ----------------------------

inputs_dict = {
    "Waste & Time": {
        "TPD": w.tpd, "Working days": w.working_days, "TPA (linked?)": w.link_tpa,
        "TPA value used": tpa, "Design years": w.design_years, "Density (t/m³)": w.density_t_per_m3
    },
    "Site / Footprint": {
        "Site area (acres)": s.site_area_acres, "Plan W (m)": s.plan_width_m, "Plan L (m)": s.plan_length_m,
        "Plan area (m²)": plan_area_val, "Inside-bund W (m)": Wi_in, "Inside-bund L (m)": Li_in
    },
    "Bund / Embankment": asdict(b),
    "Geometry (Lifts)": asdict(l),
    "Costing": asdict(cst),
    "Leachate": asdict(le),
}

results_dict = {
    "TPA": tpa,
    "Total tons over design": total_tons,
    "Required volume (m³)": required_volume,
    "Achievable volume (m³)": volume_achievable,
    "Achievable tons": achievable_tons,
    "Achievable life (years)": life_years_from_footprint,
    f"Total Cost ({cst.currency})": total_amount,
    "₹/ton": per_ton_cost,
    "Leachate annual (t/yr)": annual_leachate_tons,
    "Leachate (kL/day)": leachate_kld,
}

excel_bytes = make_excel_download(inputs_dict, results_dict, tier_df)

st.subheader("Export")
st.download_button("Download Excel (inputs + results + tiers)", data=excel_bytes, file_name="landfill_capacity_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# HTML report
import base64
html_report = make_html_report("Landfill Capacity Report", inputs_dict, results_dict, tier_df, charts_png)
st.download_button("Download HTML Report", data=html_report, file_name="landfill_report.html", mime="text/html")

# PDF report (best-effort)
pdf_bytes = make_pdf_from_text("Landfill Capacity Report", narrative, results_dict, tier_df, charts_png)
if pdf_bytes:
    st.download_button("Download PDF Report", data=pdf_bytes, file_name="landfill_report.pdf", mime="application/pdf")
else:
    st.info("PDF export uses reportlab; if not available, use the HTML report or install `reportlab`.")

# Show Narrative
with st.expander("Narrative (copyable)", expanded=False):
    st.text(narrative)

st.caption("Tip: Toggle between Lift-step and Quick model in the sidebar. Adjust slopes/berms if tiers collapse.")
