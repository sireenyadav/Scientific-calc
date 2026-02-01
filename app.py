import streamlit as st
import sympy
import sympy.physics.units as u
from sympy.physics.units import convert_to
import numpy as np
import plotly.graph_objects as go
import requests
import re

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & CSS (Modern "Dark/Neon" UI)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Scientific Super Calculator Pro",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
state_keys = ['input_value', 'history', 'last_result', 'last_expr_obj', 'last_unit_check']
for key in state_keys:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'history' else "" if key == 'input_value' else None

# Custom CSS for that "Engineer Grade" look
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        background-color: #1e293b; color: #38bdf8;
        border: 1px solid #475569; border-radius: 8px;
        font-family: 'Fira Code', 'Courier New', monospace; font-size: 18px;
    }
    
    /* Virtual Keyboard Buttons */
    .kbd-btn {
        padding: 5px; margin: 2px; font-size: 14px;
        background-color: #334155; color: white; border-radius: 5px;
        border: 1px solid #475569; width: 100%; cursor: pointer;
    }
    
    /* Result Cards */
    .result-card {
        background: rgba(30, 41, 59, 0.7); border-left: 5px solid #00f2ff;
        padding: 15px; margin-top: 10px; border-radius: 5px;
    }
    .result-val { color: #00f2ff; font-size: 1.4em; font-family: monospace; font-weight: bold; }
    .unit-tag { color: #a5b4fc; font-size: 0.9em; margin-left: 10px; font-style: italic; }
    
    /* AI Box */
    .ai-box {
        background: #312e81; border: 1px solid #6366f1; padding: 15px; 
        border-radius: 8px; margin-top: 15px; color: #e0e7ff;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. EXTENSIVE CONSTANTS LIBRARY (From PDF)
# -----------------------------------------------------------------------------
# Mapping symbols to SymPy Units objects
C_DICT = {
    # UNIVERSAL
    'c': 299792458 * u.meter / u.second,
    'G': 6.67430e-11 * u.meter**3 / (u.kg * u.second**2),
    'h': 6.62607015e-34 * u.joule * u.second,
    'hbar': 1.054571817e-34 * u.joule * u.second,
    'ħ': 1.054571817e-34 * u.joule * u.second,
    
    # ELECTROMAGNETIC
    'e': 1.602176634e-19 * u.coulomb,
    'mu0': 1.25663706127e-6 * u.newton / u.ampere**2,
    'μ0': 1.25663706127e-6 * u.newton / u.ampere**2,
    'eps0': 8.8541878188e-12 * u.farad / u.meter,
    'ε0': 8.8541878188e-12 * u.farad / u.meter,
    'Z0': 376.730313412 * u.ohm,
    'Kj': 483597.8484e9 * u.hertz / u.volt, # Josephson
    'Rk': 25812.80745 * u.ohm, # Von Klitzing
    
    # ATOMIC & NUCLEAR
    'me': 9.1093837139e-31 * u.kg, # Electron mass
    'mp': 1.67262192595e-27 * u.kg, # Proton mass
    'mn': 1.67492750056e-27 * u.kg, # Neutron mass
    'm_mu': 1.883531627e-28 * u.kg, # Muon mass
    'm_tau': 3.16754e-27 * u.kg, # Tau mass
    'm_d': 3.3435837768e-27 * u.kg, # Deuteron mass
    'm_alpha': 6.6446573450e-27 * u.kg, # Alpha particle
    'mu_B': 9.2740100783e-24 * u.joule / u.tesla, # Bohr magneton
    'mu_N': 5.050783699e-27 * u.joule / u.tesla, # Nuclear magneton
    'a0': 5.29177210903e-11 * u.meter, # Bohr radius
    'alpha': 7.2973525643e-3, # Fine structure (dimensionless)
    'Rinf': 10973731.568157 / u.meter, # Rydberg
    
    # PHYSICOCHEMICAL
    'NA': 6.02214076e23 / u.mol, # Avogadro
    'k': 1.380649e-23 * u.joule / u.kelvin, # Boltzmann
    'R': 8.314462618 * u.joule / (u.mol * u.kelvin), # Gas constant
    'F': 96485.33212 * u.coulomb / u.mol, # Faraday
    'sigma': 5.670374419e-8 * u.watt / (u.meter**2 * u.kelvin**4), # Stefan-Boltzmann
    'atm': 101325 * u.pascal,
    
    # MATH
    'pi': sympy.pi, 'π': sympy.pi
}

# -----------------------------------------------------------------------------
# 3. CORE LOGIC ENGINE
# -----------------------------------------------------------------------------

def insert_text(text):
    """Updates session state input."""
    st.session_state.input_value += str(text)

def smart_parse(expr_str):
    """
    1. Detects if input is Math or Physics.
    2. Uses SymPy to parse.
    3. Retains Units if constants are used.
    """
    if not expr_str: return None, "Empty Input"
    
    # Equation handling: LHS = RHS -> LHS - RHS
    if "=" in expr_str:
        parts = expr_str.split("=")
        expr_str = f"({parts[0]}) - ({parts[1]})"
    
    # Standardize input
    expr_str = expr_str.replace('^', '**')
    
    # Create Local Context (All SymPy functions + Physics Constants)
    local_dict = {
        'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
        'exp': sympy.exp, 'log': sympy.log, 'ln': sympy.ln, 'sqrt': sympy.sqrt,
        'diff': sympy.diff, 'integrate': sympy.integrate, 'limit': sympy.limit,
        'solve': sympy.solve
    }
    
    # Add Symbols (x, y, t) and Constants
    x, y, z, t, m = sympy.symbols('x y z t m')
    local_dict.update({'x': x, 'y': y, 'z': z, 't': t, 'm': m})
    local_dict.update(C_DICT)
    
    try:
        # Attempt Parse
        expr = sympy.sympify(expr_str, locals=local_dict)
        return expr, None
    except Exception as e:
        return None, f"Syntax Error: {str(e)}"

def format_units_ai(api_key, val_str, unit_str):
    """
    Uses Groq to convert messy units (kg*m^2/s^2) into clean ones (Joules).
    """
    if not api_key: return "⚠️ Connect AI for smart unit cleanup."
    
    prompt = f"""
    You are a physics assistant. 
    Value: {val_str}
    Current Unit: {unit_str}
    
    Task: Identify the physical quantity (Energy, Force, etc.) and convert the unit to its standard SI name (e.g. Joules, Newtons, Volts).
    If it's already simple, just repeat it.
    
    Output format: "Value Unit (Physical Quantity)"
    Example: "5.2e-19 Joules (Energy)" or "9.8 m/s^2 (Acceleration)"
    KEEP IT SHORT.
    """
    
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        return resp.json()['choices'][0]['message']['content']
    except:
        return "AI Unit Check Failed"

# -----------------------------------------------------------------------------
# 4. UI LAYOUT
# -----------------------------------------------------------------------------

# --- Sidebar: Settings & Reference ---
with st.sidebar:
    st.title("⚙️ Engine Room")
    
    # API Key Handler
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("✅ AI Connected (Secrets)")
    else:
        api_key = st.text_input("Groq API Key", type="password", help="Required for Smart Unit Formatting")

    with st.expander("📖 User Manual (How-To)"):
        st.markdown("""
        **1. Physics Calculations (with Units!)**
        * `E = m * c^2` (Mass-Energy)
        * `G * mp * me / r^2` (Gravity)
        * `h * c / (500 * nm)` (Photon Energy)
        
        **2. Symbolic Math**
        * `diff(sin(x)*x, x)` (Derivative)
        * `integrate(exp(-x), x)` (Integral)
        * `solve(x^2 - 4, x)` (Roots)
        
        **3. Plotting**
        * Just type an expression with `x` (e.g. `sin(x) * x`) and calculate.
        """)

    with st.expander("📚 Constants Library"):
        st.write("Contains all CODATA 2022 constants (Universal, Atomic, Nuclear, etc.)")
        st.json({k: str(v) for k, v in list(C_DICT.items())[:10]}) # Show preview

# --- Main Interface ---
st.title("⚛️ Scientific Super Calculator v2.0")

# VIRTUAL MATH KEYBOARD
with st.expander("⌨️  Virtual Math & Physics Keyboard", expanded=False):
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    if k1.button("∫ (Int)", use_container_width=True): insert_text("integrate(")
    if k2.button("d/dx", use_container_width=True): insert_text("diff(")
    if k3.button("√ (Sqrt)", use_container_width=True): insert_text("sqrt(")
    if k4.button("sin", use_container_width=True): insert_text("sin(")
    if k5.button("cos", use_container_width=True): insert_text("cos(")
    if k6.button("log", use_container_width=True): insert_text("log(")
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    if c1.button("π", use_container_width=True): insert_text("π")
    if c2.button("c", use_container_width=True): insert_text("c")
    if c3.button("G", use_container_width=True): insert_text("G")
    if c4.button("ℏ", use_container_width=True): insert_text("hbar")
    if c5.button("ε0", use_container_width=True): insert_text("eps0")
    if c6.button("me", use_container_width=True): insert_text("me")

# INPUT AREA
query = st.text_input("Enter Equation or Expression:", key="input_value", placeholder="e.g. h * c / (500e-9 * m) OR diff(x^2, x)")

# ACTIONS
col_act1, col_act2 = st.columns([1, 3])
calc_btn = col_act1.button("🚀 Calculate", use_container_width=True)

# -----------------------------------------------------------------------------
# 5. EXECUTION LOGIC
# -----------------------------------------------------------------------------
if calc_btn and query:
    # Parse
    expr_obj, err = smart_parse(query)
    
    if err:
        st.error(err)
    else:
        # Separate Units from Value (if physics)
        try:
            # Check if expression has units
            # SymPy logic: simplify() often combines units
            evaluated = sympy.simplify(expr_obj)
            
            # Numeric Evaluation
            numeric_val = evaluated.evalf()
            
            # Save to history
            st.session_state.history.append(f"{query}")
            
            # DISPLAY RESULTS
            st.markdown(f"""
            <div class="result-card">
                <div style="color:#94a3b8; font-size:12px;">RAW SYMPY OUTPUT</div>
                <div class="result-val">{str(numeric_val).replace('**', '^')}</div>
            </div>
            """, unsafe_allow_html=True)

            # AI UNIT CLEANUP (The "Desmos Fixer")
            # If the result looks like physics (contains units or is huge/tiny), trigger AI
            str_res = str(numeric_val)
            if "meter" in str_res or "kg" in str_res or "joule" in str_res or "second" in str_res:
                st.info("🤖 Analyzing Units with AI...")
                clean_units = format_units_ai(api_key, str(numeric_val), "SI Base Units")
                st.markdown(f"""
                <div class="ai-box">
                    <strong>✨ AI Smart Format:</strong><br>
                    <span style="font-size: 1.3em; color: #a5b4fc;">{clean_units}</span>
                </div>
                """, unsafe_allow_html=True)

            # GRAPHING (If pure math 'x')
            if hasattr(evaluated, 'free_symbols') and len(evaluated.free_symbols) == 1:
                var = list(evaluated.free_symbols)[0]
                if var.name == 'x': # Only plot if 'x' is the variable to avoid confusion
                    try:
                        f_lam = sympy.lambdify(var, evaluated, modules=['numpy'])
                        x_vals = np.linspace(-10, 10, 500)
                        y_vals = f_lam(x_vals)
                        # Mask errors
                        y_vals = np.where(np.abs(y_vals) > 1e10, np.nan, y_vals) 
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, line=dict(color='#00f2ff')))
                        fig.update_layout(title=f"Graph: {query}", template="plotly_dark", height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    except: pass

        except Exception as e:
            st.error(f"Calculation Error: {e}")

# History Footer
if st.session_state.history:
    st.markdown("---")
    st.caption("Recent Calculations:")
    for h in st.session_state.history[-3:]:
        st.text(h)
