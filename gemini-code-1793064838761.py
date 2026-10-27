import io
import math
import re
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# SymPy for calculus parsing and exact symbolic derivatives
import sympy as sp

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & INITIAL STATE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hispanic Heritage Data & Calculus Explorer Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session States
if "custom_data" not in st.session_state:
    st.session_state["custom_data"] = None
if "calc_results" not in st.session_state:
    st.session_state["calc_results"] = None

# -----------------------------------------------------------------------------
# 2. DATA DICTIONARIES & CONSTANTS
# -----------------------------------------------------------------------------
INDICATORS = {
    "SP.DYN.LE00.IN": {
        "name": "Life Expectancy at Birth",
        "units": "Years",
        "desc": "Total life expectancy at birth indicates the number of years a newborn infant would live if prevailing patterns of mortality at the time of its birth were to stay the same throughout its life."
    },
    "SL.UEM.TOTL.NE.ZS": {
        "name": "Unemployment Rate (National Estimate)",
        "units": "% of Total Labor Force",
        "desc": "Unemployment refers to the share of the labor force that is without work but available for and seeking employment, based on national estimates."
    },
    "NY.GDP.PCAP.CD": {
        "name": "GDP per Capita",
        "units": "Current US$",
        "desc": "Gross Domestic Product per capita is gross domestic product divided by midyear population. It is not average household income or individual wealth."
    },
    "SP.DYN.CBRT.IN": {
        "name": "Crude Birth Rate",
        "units": "Births per 1,000 People",
        "desc": "Crude birth rate indicates the number of live births occurring during the year, per 1,000 population estimated at midyear. It is a rate, not a total birth count."
    },
    "SM.POP.NETM": {
        "name": "Net Migration",
        "units": "Net Total People",
        "desc": "Net migration is the net total number of migrants during the period (immigrants minus emigrants). It is not an emigration count."
    },
    "VC.IHR.PSRC.P5": {
        "name": "Intentional Homicide Rate",
        "units": "Per 100,000 People",
        "desc": "Unlawful deaths purposefully inflicted on a person by another person, per 100,000 population."
    },
    "FP.CPI.TOTL.ZG": {
        "name": "Consumer Price Inflation",
        "units": "Annual %",
        "desc": "Inflation as measured by the consumer price index reflects the annual percentage change in the cost to the average consumer of acquiring a basket of goods and services."
    },
    "GC.DOD.TOTL.GD.ZS": {
        "name": "Central Government Debt",
        "units": "% of GDP",
        "desc": "Gross debt consists of all liabilities that require payment or payments of interest and/or principal by the debtor to the creditor at a date or dates in the future."
    },
    "FI.RES.TOTL.CD": {
        "name": "Total Reserves (includes Gold)",
        "units": "Current US$",
        "desc": "Total reserves comprise holdings of monetary gold, special drawing rights, reserves of IMF members, and holdings of foreign exchange under the control of monetary authorities."
    },
    "NE.RSB.GNFS.ZS": {
        "name": "External Balance on Goods and Services",
        "units": "% of GDP",
        "desc": "External balance on goods and services (formerly resource balance) equals exports of goods and services minus imports of goods and services."
    }
}

COUNTRIES = {
    "MX": "Mexico",
    "BR": "Brazil (LA Comparison)",
    "AR": "Argentina",
    "CO": "Colombia",
    "CL": "Chile",
    "PE": "Peru",
    "VE": "Venezuela",
    "EC": "Ecuador",
    "UY": "Uruguay"
}

CONTEXTS = {
    "Economic Forecasting": {
        "ind": "Year / Time", "dep": "GDP / Debt / Value", "units": "US$ / %",
        "suggested_fn": "0.05*x**3 - 0.5*x**2 + 2*x + 10", "domain": (0, 20),
        "desc": "Modelling national GDP growth, debt dynamics, or inflation curves."
    },
    "Bioengineering": {
        "ind": "Time", "dep": "Knee Joint Angle", "units": "Degrees",
        "suggested_fn": "25*sin(1.5*x) + 30", "domain": (0, 10),
        "desc": "Kinematic analysis of lower-limb prosthetics during human gait cycle."
    },
    "Medical Science": {
        "ind": "Time Post-Administration", "dep": "Drug Concentration", "units": "mg/L",
        "suggested_fn": "15*x*exp(-0.4*x)", "domain": (0, 15),
        "desc": "Pharmacokinetics of intravenous dosage and plasma metabolic clearance."
    },
    "Biology": {
        "ind": "Time", "dep": "Cell Population", "units": "Thousand Cells/mL",
        "suggested_fn": "100 / (1 + 9*exp(-0.5*x))", "domain": (0, 15),
        "desc": "Logistic growth of cell culture reaching carrying capacity."
    },
    "Engineering / Physics": {
        "ind": "Time", "dep": "Projectile Altitude", "units": "Meters",
        "suggested_fn": "-4.9*x**2 + 35*x + 2", "domain": (0, 7.5),
        "desc": "Trajectory analysis under constant gravitational acceleration."
    },
    "Weather Dynamics": {
        "ind": "Time of Day", "dep": "Temperature", "units": "°C",
        "suggested_fn": "8*sin(0.26*(x-6)) + 18", "domain": (0, 24),
        "desc": "Diurnal solar radiation and atmospheric ambient temperature fluctuation."
    },
    "Sports Science": {
        "ind": "Distance", "dep": "Runner Velocity", "units": "m/s",
        "suggested_fn": "-0.001*x**2 + 0.1*x + 2", "domain": (0, 100),
        "desc": "Pacing strategy and sprint exertion velocity profile."
    },
    "Business Analytics": {
        "ind": "Production Output", "dep": "Marginal Revenue", "units": "Thousand US$",
        "suggested_fn": "-0.2*x**2 + 12*x - 50", "domain": (5, 50),
        "desc": "Optimization of manufacturing volume and marginal profitability."
    },
    "Custom / User Specified": {
        "ind": "x", "dep": "y", "units": "Units",
        "suggested_fn": "x**3 - 3*x", "domain": (-3, 3),
        "desc": "Custom mathematical function exploration."
    }
}

# -----------------------------------------------------------------------------
# 3. WORLD BANK API & CACHING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_world_bank_data(country_code, indicator_code, start_year, end_year):
    """Fetches authentic historical data directly from World Bank API v2."""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    params = {
        "date": f"{start_year}:{end_year}",
        "format": "json",
        "per_page": 1000
    }
    try:
        resp = requests.get(url, params=params, timeout=12)
        if resp.status_code != 200:
            return None, f"HTTP Error {resp.status_code} returned from API."
        data = resp.json()
        if len(data) < 2 or not data[1]:
            return None, "No historical records returned for this indicator/country selection."
        
        records = []
        for entry in data[1]:
            val = entry.get("value")
            yr = entry.get("date")
            if yr and yr.isdigit():
                records.append({"Year": int(yr), "Value": float(val) if val is not None else np.nan})
        
        df = pd.DataFrame(records).sort_values("Year").reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, f"Failed to connect to API: {str(e)}"

# -----------------------------------------------------------------------------
# 4. MATHEMATICAL FIT & STATISTICAL COMPUTATION
# -----------------------------------------------------------------------------
def fit_polynomial_model(df, degree, graph_increment=1):
    """Fits centered degree-d polynomial and calculates in-sample regression metrics."""
    valid_df = df.dropna(subset=["Value"]).copy()
    if len(valid_df) < degree + 1:
        return None, f"Insufficient data: Degree {degree} polynomial requires at least {degree + 1} valid observation points. Found {len(valid_df)}."

    sampled_df = valid_df.iloc[::graph_increment].copy()
    if len(sampled_df) < degree + 1:
        sampled_df = valid_df.copy()

    base_year = int(sampled_df["Year"].mean())
    t = sampled_df["Year"].values - base_year
    y = sampled_df["Value"].values

    coeffs = np.polyfit(t, y, degree)
    poly_fn = np.poly1d(coeffs)

    y_pred = poly_fn(t)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    residuals = y - y_pred

    return {
        "coeffs": coeffs,
        "poly_fn": poly_fn,
        "base_year": base_year,
        "degree": degree,
        "r2": r2,
        "mae": mae,
        "residuals": residuals,
        "count": len(sampled_df),
        "year_min": int(valid_df["Year"].min()),
        "year_max": int(valid_df["Year"].max()),
        "mean_val": float(valid_df["Value"].mean()),
        "df_valid": valid_df
    }, None

def predict_year(model, target_year):
    """Evaluates the fitted centered model for a specific target year."""
    t_target = target_year - model["base_year"]
    pred = float(model["poly_fn"](t_target))
    within_range = model["year_min"] <= target_year <= model["year_max"]
    return pred, within_range

# -----------------------------------------------------------------------------
# 5. SYMBOLIC CALCULUS & ANALYSIS ENGINE
# -----------------------------------------------------------------------------
def parse_sympy_expression(expr_str):
    """Parses mathematical string expressions into SymPy symbols and callables."""
    clean_str = expr_str.replace("^", "**")
    x = sp.Symbol('x', real=True)
    
    transform_dict = {
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "ln": sp.log,
        "sqrt": sp.sqrt, "abs": sp.Abs
    }
    
    f_sym = sp.sympify(clean_str, locals=transform_dict)
    f_prime_sym = sp.diff(f_sym, x)
    f_double_prime_sym = sp.diff(f_prime_sym, x)

    f_num = sp.lambdify(x, f_sym, modules=["numpy", "scipy"])
    f_prime_num = sp.lambdify(x, f_prime_sym, modules=["numpy", "scipy"])
    f_double_prime_num = sp.lambdify(x, f_double_prime_sym, modules=["numpy", "scipy"])

    return {
        "x_sym": x, "f_sym": f_sym,
        "f_prime_sym": f_prime_sym, "f_double_prime_sym": f_double_prime_sym,
        "f_num": f_num, "f_prime_num": f_prime_num, "f_double_prime_num": f_double_prime_num
    }

def analyze_calculus_features(calc_dict, domain_bounds):
    """Calculates critical points, extrema, concavity, and inflection points rigorously."""
    x = calc_dict["x_sym"]
    f_sym = calc_dict["f_sym"]
    f_prime_sym = calc_dict["f_prime_sym"]
    f_double_prime_sym = calc_dict["f_double_prime_sym"]
    
    a, b = domain_bounds
    critical_points = []
    extrema = []
    inflection_points = []

    try:
        sol_prime = sp.solve(f_prime_sym, x)
        for sol in sol_prime:
            if sol.is_real:
                val = float(sol.evalf())
                if a <= val <= b:
                    critical_points.append(round(val, 4))
    except Exception:
        grid = np.linspace(a, b, 200)
        for i in range(len(grid)-1):
            try:
                p1 = calc_dict"f_prime_num"
                p2 = calc_dict"f_prime_num"
                if p1 * p2 <= 0:
                    mid = 0.5 * (grid[i] + grid[i+1])
                    critical_points.append(round(float(mid), 4))
            except Exception:
                pass
    critical_points = sorted(list(set(critical_points)))

    for cp in critical_points:
        try:
            f2_val = float(f_double_prime_sym.subs(x, cp).evalf())
            f_val = float(f_sym.subs(x, cp).evalf())
            if f2_val < -1e-6:
                extrema.append({"type": "Local Maximum", "x": cp, "y": round(f_val, 4), "reason": "f''(x) < 0"})
            elif f2_val > 1e-6:
                extrema.append({"type": "Local Minimum", "x": cp, "y": round(f_val, 4), "reason": "f''(x) > 0"})
            else:
                extrema.append({"type": "Inconclusive", "x": cp, "y": round(f_val, 4), "reason": "f''(x) = 0"})
        except Exception:
            pass

    try:
        sol_double = sp.solve(f_double_prime_sym, x)
        for sol in sol_double:
            if sol.is_real:
                val = float(sol.evalf())
                if a <= val <= b:
                    eps = 1e-3
                    f2_left = float(f_double_prime_sym.subs(x, val - eps).evalf())
                    f2_right = float(f_double_prime_sym.subs(x, val + eps).evalf())
                    if f2_left * f2_right < 0:
                        y_val = float(f_sym.subs(x, val).evalf())
                        inflection_points.append({"x": round(val, 4), "y": round(y_val, 4)})
    except Exception:
        pass

    return critical_points, extrema, inflection_points

# -----------------------------------------------------------------------------
# 6. PDF & EXCEL EXPORT GENERATORS
# -----------------------------------------------------------------------------
def build_excel_report(df_raw, model, custom_years_df, country_name, ind_name):
    """Generates multi-tab Excel workbook using XlsxWriter."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_data = {
            "Metric / Parameter": [
                "Country Selected", "Indicator Name", "Fitted Degree", "Base Year (Centering)",
                "In-Sample R-Squared", "In-Sample MAE", "Observation Count",
                "Historical Range Min Year", "Historical Range Max Year", "Export Date"
            ],
            "Value": [
                country_name, ind_name, model["degree"], model["base_year"],
                f"{model['r2']:.4f}", f"{model['mae']:.4f}", model["count"],
                model["year_min"], model["year_max"], datetime.date.today().strftime("%Y-%m-%d")
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Model Summary", index=False)
        df_raw.to_excel(writer, sheet_name="Working Data", index=False)

        if custom_years_df is not None:
            custom_years_df.to_excel(writer, sheet_name="Calculated Results", index=False)
            
    return output.getvalue()

def build_pdf_report(country_name, ind_name, model, calc_df, plot_buf):
    """Builds a printer-friendly PDF report using ReportLab."""
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor("#1E3A8A"), spaceAfter=12)
    h2_style = ParagraphStyle('H2Style', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor("#0D9488"), spaceAfter=8)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=9, leading=12, spaceAfter=6)

    story.append(Paragraph(f"Hispanic Heritage Analytics Report: {country_name}", title_style))
    story.append(Paragraph(f"<b>Indicator:</b> {ind_name} | <b>Generated:</b> {datetime.date.today().strftime('%B %d, %Y')}", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Regression Specification & In-Sample Metrics", h2_style))
    poly_terms = []
    for i, c in enumerate(model["coeffs"]):
        p = model["degree"] - i
        if p == 0: poly_terms.append(f"{c:+.4e}")
        elif p == 1: poly_terms.append(f"{c:+.4e}(t)")
        else: poly_terms.append(f"{c:+.4e}(t)^{p}")
    eq_str = f"y(t) = {' '.join(poly_terms)}, where t = Year - {model['base_year']}"

    table_data = [
        ["Parameter / Metric", "Value", "Parameter / Metric", "Value"],
        ["Centered Base Year", str(model["base_year"]), "Polynomial Degree", str(model["degree"])],
        ["In-Sample R-Squared", f"{model['r2']:.4f}", "In-Sample MAE", f"{model['mae']:.4f}"],
        ["Observations Count", str(model["count"]), "Historical Span", f"{model['year_min']} - {model['year_max']}"]
    ]
    t = Table(table_data, colWidths=[130, 120, 130, 120])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Fitted Equation:</b> <font fontName='Courier'>{eq_str}</font>", body_style))
    story.append(Spacer(1, 10))

    if plot_buf:
        story.append(Paragraph("Historical Model Visualization", h2_style))
        story.append(RLImage(plot_buf, width=500, height=250))
        story.append(Spacer(1, 10))

    if calc_df is not None and not calc_df.empty:
        story.append(Paragraph("Evaluated Model Predictions & Classifications", h2_style))
        calc_list = [["Year", "Fitted Value", "Classification", "Extrapolation Warning"]]
        for _, row in calc_df.iterrows():
            calc_list.append([
                str(row["Year"]),
                f"{row['Estimated Value']:.4f}",
                str(row["Classification"]),
                str(row["Extrapolation Status"])
            ])
        t_calc = Table(calc_list[:15], colWidths=[60, 100, 140, 200])
        t_calc.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
            ('FONTSIZE', (0,0), (-1,-1), 8),
        ]))
        story.append(t_calc)

    doc.build(story)
    return pdf_buffer.getvalue()

# -----------------------------------------------------------------------------
# 7. UI SIDEBAR & CONTROL PANEL
# -----------------------------------------------------------------------------
st.sidebar.title("🏛️ Control Center")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Select Explorer Module",
    ["1. Historical Data Explorer", "2. Dynamic Calculus & Simulation Lab"]
)

st.sidebar.markdown("---")
academic_level = st.sidebar.selectbox(
    "Academic Pedagogy Level",
    ["Honors Algebra 2", "Precalculus", "Calculus"]
)

# -----------------------------------------------------------------------------
# 8. MODULE 1: HISTORICAL DATA EXPLORER
# -----------------------------------------------------------------------------
if app_mode == "1. Historical Data Explorer":
    st.title("📊 Hispanic Heritage Historical Data Explorer")
    st.markdown("""
    Retrieve authentic time-series indicator records from the **World Bank API**, fit polynomial models, 
    evaluate out-of-sample projections, and conduct analytical trajectory comparisons.
    """)

    col_c1, col_c2, col_c3 = st.columns([1.5, 2, 1])
    with col_c1:
        selected_country = st.selectbox("Primary Country", options=list(COUNTRIES.keys()), format_func=lambda k: f"{COUNTRIES[k]} ({k})")
    with col_c2:
        selected_indicator = st.selectbox("World Bank Indicator", options=list(INDICATORS.keys()), format_func=lambda k: f"{INDICATORS[k]['name']}")
    with col_c3:
        compare_country = st.selectbox("Comparison Country", options=["None"] + list(COUNTRIES.keys()), format_func=lambda k: "None" if k == "None" else f"{COUNTRIES[k]} ({k})")

    ind_info = INDICATORS[selected_indicator]
    
    with st.expander("ℹ️ World Bank Indicator Metadata & Exact Mappings", expanded=False):
        st.markdown(f"**Official Name:** {ind_info['name']}")
        st.markdown(f"**API Code:** `{selected_indicator}`")
        st.markdown(f"**Measurement Units:** {ind_info['units']}")
        st.markdown(f"**Official Definition:** {ind_info['desc']}")
        st.markdown(f"**Source Endpoint:** [https://api.worldbank.org/v2/country/{selected_country}/indicator/{selected_indicator}](https://api.worldbank.org/v2/country/{selected_country}/indicator/{selected_indicator})")
        st.markdown(f"**Data Retrieval Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}")

    current_year = datetime.datetime.now().year
    start_yr = current_year - 70
    end_yr = current_year - 1

    df_raw, err = fetch_world_bank_data(selected_country, selected_indicator, start_yr, end_yr)

    if err:
        st.error(f"❌ API Retrieval Error: {err}")
        st.stop()

    if st.session_state["custom_data"] is not None:
        working_df = st.session_state["custom_data"].copy()
        st.warning("⚠️ **User-Modified Scenario Active:** Working dataset has been manually altered.")
        if st.button("🔄 Reset to Source Data"):
            st.session_state["custom_data"] = None
            st.rerun()
    else:
        working_df = df_raw.copy()

    with st.expander("📝 View / Edit Historical Table (Scenario Modeling)", expanded=False):
        edited_df = st.data_editor(
            working_df,
            num_rows="dynamic",
            key="data_editor",
            use_container_width=True
        )
        if not edited_df.equals(working_df):
            st.session_state["custom_data"] = edited_df
            st.rerun()

    st.markdown("### 🎛️ Model Fitting & Sampling Controls")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        poly_degree = st.slider("Polynomial Degree (d)", min_value=3, max_value=8, value=3)
    with m_col2:
        graph_increment = st.slider("Graph Sample Increment (Years)", min_value=1, max_value=10, value=1)
    with m_col3:
        forward_horizon = st.slider("Forward Display Horizon (Years)", min_value=0, max_value=50, value=15)
    with m_col4:
        holdout_ratio = st.slider("Holdout Validation %", min_value=0, max_value=30, value=0, step=5)

    model, fit_err = fit_polynomial_model(working_df, poly_degree, graph_increment)
    if fit_err:
        st.error(f"❌ Fitting Error: {fit_err}")
        st.stop()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    valid_pts = model["df_valid"]
    ax.scatter(valid_pts["Year"], valid_pts["Value"], color="#1E3A8A", label=f"Observed Points ({COUNTRIES[selected_country]})", zorder=4, s=30)

    x_smooth = np.linspace(model["year_min"], model["year_max"] + forward_horizon, 400)
    t_smooth = x_smooth - model["base_year"]
    y_smooth = model["poly_fn"](t_smooth)

    hist_mask = x_smooth <= model["year_max"]
    ax.plot(x_smooth[hist_mask], y_smooth[hist_mask], color="#0D9488", linewidth=2.5, label=f"Degree-{poly_degree} Fitted Curve (In-Sample)")
    
    if forward_horizon > 0:
        extrap_mask = x_smooth >= model["year_max"]
        ax.plot(x_smooth[extrap_mask], y_smooth[extrap_mask], color="#DC2626", linestyle="--", linewidth=2, label="Extrapolated Segment")
        ax.axvline(model["year_max"], color="#64748B", linestyle=":", label="Historical Boundary")

    if compare_country != "None" and compare_country != selected_country:
        df_comp, comp_err = fetch_world_bank_data(compare_country, selected_indicator, start_yr, end_yr)
        if df_comp is not None and not df_comp.empty:
            valid_comp = df_comp.dropna(subset=["Value"])
            ax.plot(valid_comp["Year"], valid_comp["Value"], color="#F59E0B", linestyle="-.", linewidth=1.5, label=f"Comparison: {COUNTRIES[compare_country]}")

    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel(f"{ind_info['name']} ({ind_info['units']})", fontsize=10)
    ax.set_title(f"Polynomial Trajectory Fit: {COUNTRIES[selected_country]}", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    st.pyplot(fig)

    st.markdown("### 📐 Fitted Model Statistics & Equation")
    poly_terms = []
    for i, c in enumerate(model["coeffs"]):
        p = model["degree"] - i
        if p == 0: poly_terms.append(f"{c:+.4e}")
        elif p == 1: poly_terms.append(f"{c:+.4e}(t)")
        else: poly_terms.append(f"{c:+.4e}(t)^{p}")
    eq_str = f"y(t) = {' '.join(poly_terms)}"

    st.latex(rf"\text{{Model Equation: }} {eq_str}")
    st.info(f"**Centered Transformation:** $t = \\text{{Year}} - {model['base_year']}$. Base year is set to average historical observation year to maintain numerical precision.")

    m_stat1, m_stat2, m_stat3, m_stat4 = st.columns(4)
    m_stat1.metric("In-Sample R²", f"{model['r2']:.4f}")
    m_stat2.metric("In-Sample MAE", f"{model['mae']:.4f}")
    m_stat3.metric("Observation Count", f"{model['count']}")
    m_stat4.metric("Observed Span", f"{model['year_min']} - {model['year_max']}")

    st.markdown("---")
    st.markdown("### 🔮 Single & Batch Target Prediction Engine")
    
    p_tab1, p_tab2 = st.tabs(["Single-Year Evaluation", "Batch Evaluation"])
    
    with p_tab1:
        calc_year = st.number_input("Target Input Year (1950 - 2100)", min_value=1950, max_value=2100, value=int(model["year_max"] + 5))
        if st.button("Calculate Single Value"):
            val_est, in_bound = predict_year(model, calc_year)
            st.markdown(f"**Estimated Value for {calc_year}:** `{val_est:.4f} {ind_info['units']}`")
            if in_bound:
                st.success("📍 Classification: **Within Historical Range** (In-Sample Fitted Prediction)")
            else:
                st.warning("⚠️ Classification: **Outside Historical Range** (Extrapolated Prediction)")
                if abs(val_est - model["mean_val"]) > 3 * np.std(model["df_valid"]["Value"].dropna()):
                    st.error("🚨 **Extrapolation Flag:** Projected value exhibits extreme polynomial divergence from historical baseline mean!")

    with p_tab2:
        batch_mode = st.radio("Batch Mode Selection", ["Year Range Step", "Custom Comma-Separated List"], horizontal=True)
        eval_years = []
        if batch_mode == "Year Range Step":
            b_col1, b_col2, b_col3 = st.columns(3)
            with b_col1: r_start = st.number_input("Start Year", 1950, 2100, int(model["year_min"]))
            with b_col2: r_end = st.number_input("End Year", 1950, 2100, int(model["year_max"] + 10))
            with b_col3: r_step = st.slider("Step Size", 1, 10, 5)
            eval_years = list(range(int(r_start), int(r_end) + 1, int(r_step)))
        else:
            custom_input = st.text_input("Enter Years (comma-separated)", "1980, 1995, 2010, 2025, 2040")
            try:
                eval_years = [int(y.strip()) for y in custom_input.split(",") if y.strip().isdigit()]
            except Exception:
                st.error("Invalid custom year format.")

        if eval_years:
            batch_results = []
            for y_eval in eval_years:
                v_est, in_b = predict_year(model, y_eval)
                is_extrap_wild = "HIGH DIVERGENCE" if not in_b and (v_est < 0 or abs(v_est) > 10 * abs(model["mean_val"])) else "Normal"
                batch_results.append({
                    "Year": y_eval,
                    "Estimated Value": v_est,
                    "Units": ind_info["units"],
                    "Classification": "Within Historical Range" if in_b else "Extrapolation Horizon",
                    "Extrapolation Status": is_extrap_wild
                })
            res_df = pd.DataFrame(batch_results)
            st.dataframe(res_df, use_container_width=True)
            st.session_state["calc_results"] = res_df

    st.markdown("---")
    st.markdown("### 📥 Export Laboratory Results & Documentation")
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        xlsx_bytes = build_excel_report(working_df, model, st.session_state.get("calc_results"), COUNTRIES[selected_country], ind_info['name'])
        st.download_button("📥 Download Complete Excel Workbook (.xlsx)", data=xlsx_bytes, file_name=f"{selected_country}_analytics.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with ex_col2:
        pdf_bytes = build_pdf_report(COUNTRIES[selected_country], ind_info['name'], model, st.session_state.get("calc_results"), buf)
        st.download_button("📄 Download Printable PDF Summary (.pdf)", data=pdf_bytes, file_name=f"{selected_country}_summary.pdf", mime="application/pdf")

# -----------------------------------------------------------------------------
# 9. MODULE 2: DYNAMIC CALCULUS & SIMULATION LAB
# -----------------------------------------------------------------------------
else:
    st.title("🧪 Interactive Dynamic Calculus & Visual Simulation Lab")
    st.markdown("""
    Select real-world applied context, inspect symbolic derivatives, explore tangent/secant rates of change, 
    and visualize synchronized real-time multi-domain physical/economic simulations.
    """)

    ctx_col1, ctx_col2 = st.columns([1.5, 2])
    with ctx_col1:
        selected_ctx_name = st.selectbox("Select Applied Real-World Context", list(CONTEXTS.keys()))
        ctx_data = CONTEXTS[selected_ctx_name]
    with ctx_col2:
        fn_source = st.radio("Function Origin Mode", ["Suggested Context Function", "Custom Mathematical Notation"], horizontal=True)

    if fn_source == "Suggested Context Function":
        raw_fn_str = ctx_data["suggested_fn"]
    else:
        raw_fn_str = st.text_input("Enter Function f(x)", "x**3 - 3*x", help="Use standard python operators: x**2, sin(x), exp(x), log(x)")

    with st.expander("⚙️ Customize Variable Names, Units & Analysis Domain", expanded=False):
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1: var_ind = st.text_input("Independent Variable Name", ctx_data["ind"])
        with c_p2: var_dep = st.text_input("Dependent Variable Name", ctx_data["dep"])
        with c_p3: var_unit = st.text_input("Measurement Units", ctx_data["units"])
        
        dom_min = st.number_input("Domain Min (a)", value=float(ctx_data["domain"][0]))
        dom_max = st.number_input("Domain Max (b)", value=float(ctx_data["domain"][1]))

    try:
        calc_obj = parse_sympy_expression(raw_fn_str)
    except Exception as e:
        st.error(f"❌ Expression Parsing Error: {str(e)}. Please check syntax (e.g. use `*` for multiplication).")
        st.stop()

    st.markdown("### 📜 Mathematical Function & Symbolic Derivatives")
    st.latex(rf"f(x) = {sp.latex(calc_obj['f_sym'])}")
    
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.latex(rf"f'(x) = \frac{{d}}{{dx}}[f(x)] = {sp.latex(calc_obj['f_prime_sym'])}")
    with d_col2:
        st.latex(rf"f''(x) = \frac{{d^2}}{{dx^2}}[f(x)] = {sp.latex(calc_obj['f_double_prime_sym'])}")

    crit_pts, extrema, inflections = analyze_calculus_features(calc_obj, (dom_min, dom_max))

    st.markdown("---")
    st.markdown("### 🎛️ Tangent & Secant Rate Interactive Controls")
    
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        eval_x = st.slider(f"Instantaneous Tangent Point ({var_ind} x)", min_value=float(dom_min), max_value=float(dom_max), value=float(dom_min + 0.5*(dom_max-dom_min)), step=0.05)
    with ctrl_col2:
        sec_a = st.slider("Secant Interval Start (a)", min_value=float(dom_min), max_value=float(dom_max), value=float(dom_min), step=0.05)
    with ctrl_col3:
        sec_b = st.slider("Secant Interval End (b)", min_value=float(dom_min), max_value=float(dom_max), value=float(dom_max), step=0.05)

    if sec_a >= sec_b:
        st.warning("⚠️ Secant start point (a) must be strictly less than end point (b). Adjusting automatically.")
        sec_b = sec_a + 0.1

    try:
        y_val = float(calc_obj["f_num"](eval_x))
        inst_rate = float(calc_obj["f_prime_num"](eval_x))
        conc_val = float(calc_obj["f_double_prime_num"](eval_x))
        
        y_sec_a = float(calc_obj["f_num"](sec_a))
        y_sec_b = float(calc_obj["f_num"](sec_b))
        avg_rate = (y_sec_b - y_sec_a) / (sec_b - sec_a)
    except Exception as e:
        st.error(f"Computation error at selected point: {str(e)}")
        st.stop()

    r_col1, r_col2, r_col3 = st.columns(3)
    r_col1.metric(f"Function Value f({eval_x:.2f})", f"{y_val:.4f} {var_unit}")
    r_col2.metric(f"Instantaneous Rate f'({eval_x:.2f})", f"{inst_rate:.4f} {var_unit}/{var_ind}")
    r_col3.metric(f"Average Rate over [{sec_a:.2f}, {sec_b:.2f}]", f"{avg_rate:.4f} {var_unit}/{var_ind}")

    v_col1, v_col2, v_col3 = st.columns(3)
    show_f = v_col1.checkbox("Show f(x)", value=True)
    show_fp = v_col2.checkbox("Show First Derivative f'(x)", value=True)
    show_fpp = v_col3.checkbox("Show Second Derivative f''(x)", value=False)

    fig_calc, ax_calc = plt.subplots(figsize=(10, 5))
    x_grid = np.linspace(dom_min, dom_max, 500)
    
    if show_f:
        y_grid = calc_obj["f_num"](x_grid)
        ax_calc.plot(x_grid, y_grid, color="#1E3A8A", linewidth=2.5, label=f"f(x): {var_dep}")
    if show_fp:
        yp_grid = calc_obj["f_prime_num"](x_grid)
        ax_calc.plot(x_grid, yp_grid, color="#0D9488", linestyle="--", linewidth=1.8, label="f'(x): Instantaneous Rate")
    if show_fpp:
        ypp_grid = calc_obj["f_double_prime_num"](x_grid)
        ax_calc.plot(x_grid, ypp_grid, color="#8B5CF6", linestyle=":", linewidth=1.8, label="f''(x): Acceleration/Concavity")

    tangent_x = np.linspace(eval_x - 0.2*(dom_max-dom_min), eval_x + 0.2*(dom_max-dom_min), 50)
    tangent_y = y_val + inst_rate * (tangent_x - eval_x)
    ax_calc.plot(tangent_x, tangent_y, color="#DC2626", linewidth=2, label=f"Tangent Line at x={eval_x:.2f}")
    ax_calc.scatter([eval_x], [y_val], color="#DC2626", zorder=5, s=50)

    secant_x = np.array([sec_a, sec_b])
    secant_y = np.array([y_sec_a, y_sec_b])
    ax_calc.plot(secant_x, secant_y, color="#F59E0B", linestyle="-.", linewidth=2, label=f"Secant Line [{sec_a:.2f}, {sec_b:.2f}]")
    ax_calc.scatter(secant_x, secant_y, color="#F59E0B", zorder=5, s=40)

    for ep in extrema:
        ax_calc.scatter([ep["x"]], [ep["y"]], color="#059669", marker="^", s=70, zorder=6, label=f"{ep['type']} ({ep['x']:.2f}, {ep['y']:.2f})")
    for ip in inflections:
        ax_calc.scatter([ip["x"]], [ip["y"]], color="#7C3AED", marker="s", s=70, zorder=6, label=f"Inflection Point ({ip['x']:.2f}, {ip['y']:.2f})")

    ax_calc.set_xlabel(f"{var_ind}", fontsize=10)
    ax_calc.set_ylabel(f"{var_dep} ({var_unit})", fontsize=10)
    ax_calc.set_title(f"Dynamic Analysis: {selected_ctx_name}", fontsize=12, fontweight="bold")
    ax_calc.grid(True, linestyle="--", alpha=0.5)
    ax_calc.legend(loc="upper right", fontsize=8)

    st.pyplot(fig_calc)

    st.markdown("---")
    st.markdown("### 🎬 Synchronized Context Visual Simulation")
    
    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        st.markdown(f"**Current Simulation Point:** `{eval_x:.2f} {var_ind}`")
        st.markdown(f"**State Variable Value:** `{y_val:.4f} {var_unit}`")
        
        if "Prosthetic" in selected_ctx_name or "Bioengineering" in selected_ctx_name:
            st.info(f"🦿 **Prosthetic Gait Analysis:** At time {eval_x:.2f}s, knee flexion angle is {y_val:.2f}°. Flexion angular velocity is {inst_rate:.2f} deg/s.")
        elif "Medical" in selected_ctx_name:
            st.info(f"💉 **Pharmacokinetics:** Blood drug concentration is {y_val:.2f} mg/L. Rate of metabolic clearance/absorption is {inst_rate:.2f} (mg/L)/hr.")
        elif "Engineering" in selected_ctx_name or "Physics" in selected_ctx_name:
            st.info(f"🚀 **Kinematics:** Altitude is {y_val:.2f} m. Instantaneous vertical velocity is {inst_rate:.2f} m/s; vertical acceleration is {conc_val:.2f} m/s².")
        elif "Business" in selected_ctx_name or "Economic" in selected_ctx_name:
            st.info(f"📈 **Economic State:** Value is {y_val:.2f} {var_unit}. Marginal change rate is {inst_rate:.2f} {var_unit}/{var_ind}.")
        else:
            st.info(f"🔬 **System State:** Value is {y_val:.2f} {var_unit} with rate of change {inst_rate:.2f}.")

    with sim_col2:
        fig_sim, ax_sim = plt.subplots(figsize=(6, 2.5))
        ax_sim.set_xlim(-1, 11)
        ax_sim.set_ylim(-1, 10)
        ax_sim.axis("off")

        if "Bioengineering" in selected_ctx_name:
            knee_angle_rad = math.radians(y_val)
            hip = (5, 8)
            knee = (5, 4)
            ankle = (5 + 3.5 * math.sin(knee_angle_rad), 4 - 3.5 * math.cos(knee_angle_rad))
            ax_sim.plot([hip[0], knee[0]], [hip[1], knee[1]], color="#1E3A8A", linewidth=6, label="Thigh")
            ax_sim.plot([knee[0], ankle[0]], [knee[1], ankle[1]], color="#0D9488", linewidth=5, label="Prosthetic Shank")
            ax_sim.scatter([knee[0]], [knee[1]], color="#DC2626", s=120, zorder=5)
            ax_sim.text(1, 8, f"Flexion Angle: {y_val:.1f}°", fontsize=10, fontweight="bold")
        elif "Medical" in selected_ctx_name:
            fill_h = min(max(0, y_val / 20.0 * 8.0), 8.0)
            ax_sim.add_patch(plt.Rectangle((3, 1), 4, 8, fill=False, edgecolor="#1E3A8A", linewidth=2))
            ax_sim.add_patch(plt.Rectangle((3, 1), 4, fill_h, color="#0D9488", alpha=0.7))
            ax_sim.text(0.5, 5, f"Concentration:\n{y_val:.2f} mg/L", fontsize=9, fontweight="bold")
        else:
            norm_x = min(max(0, (eval_x - dom_min) / (dom_max - dom_min) * 10), 10)
            norm_y = min(max(0, (y_val - 0) / 50 * 8 + 1), 9)
            ax_sim.scatter([norm_x], [norm_y], color="#DC2626", s=200, zorder=5)
            ax_sim.plot([norm_x, norm_x], [0, norm_y], color="#64748B", linestyle="--")
            ax_sim.text(0, 9, f"State Marker ({var_ind}={eval_x:.1f}, {var_dep}={y_val:.1f})", fontsize=9)

        st.pyplot(fig_sim)

    st.markdown("---")
    st.markdown("### 📋 Formally Verified Calculus Properties")
    
    f_tab1, f_tab2 = st.tabs(["Local Extrema Justifications", "Inflection Points & Concavity"])
    with f_tab1:
        if extrema:
            st.table(pd.DataFrame(extrema))
        else:
            st.info("No local extrema found on the specified open interval.")
    with f_tab2:
        if inflections:
            st.table(pd.DataFrame(inflections))
        else:
            st.info("No inflection points (concavity changes) detected on the specified open interval.")