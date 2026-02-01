import streamlit as st
import sympy
import sympy.physics.units as u
import numpy as np
import plotly.graph_objects as go
import requests
import re

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Omni-Calc AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed for cleaner "App-like" feel
)

# Custom CSS for the "Futuristic/Responsive" feel
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp { background-color: #000000; color: #e0e0e0; }
    
    /* The "Omnibar" Input */
    .stTextInput > div > div > input {
        background-color: #1a1a1a; 
        color: #00ffcc;
        border: 2px solid #333; 
        border-radius: 50px; /* Pill shape */
        padding: 15px 25px;
        font-family: 'Inter', sans-serif;
        font-size: 20px;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.1);
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00ffcc;
        box-shadow: 0 0 30px rgba(0, 255, 204, 0.3);
    }
    
    /* Result Display */
    .result-card {
        background: linear-gradient(145deg, #111, #0a0a0a);
        border: 1px solid #333;
        border-left: 6px solid #00ffcc;
        padding: 30px;
        border-radius: 20px;
        margin-top: 30px;
        animation: fadeIn 0.5s ease-in-out;
    }
    .main-result {
        font-size: 3em;
        font-weight: 700;
        color: #00ffcc;
        font-family: 'Courier New', monospace;
        margin-bottom: 10px;
    }
    .sub-result {
        color: #888;
        font-size: 1.1em;
        font-family: sans-serif;
    }
    
    /* AI Tag */
    .ai-badge {
        background: #333; color: #fff; padding: 4px 10px;
        border-radius: 12px; font-size: 0.8em; margin-bottom: 10px;
        display: inline-block;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. EXTENSIVE CONSTANTS LIBRARY
# -----------------------------------------------------------------------------
CONSTANTS = {
    'c': 299792458 * u.meter / u.second,
    'G': 6.67430e-11 * u.meter**3 / (u.kg * u.second**2),
    'h': 6.62607015e-34 * u.joule * u.second,
    'hbar': 1.054571817e-34 * u.joule * u.second,
    'e': 1.602176634e-19 * u.coulomb,
    'me': 9.1093837139e-31 * u.kg,
    'mp': 1.67262192595e-27 * u.kg,
    'NA': 6.02214076e23 / u.mol,
    'k': 1.380649e-23 * u.joule / u.kelvin,
    'eps0': 8.8541878188e-12 * u.farad / u.meter,
    'mu0': 1.25663706127e-6 * u.newton / u.ampere**2,
    'pi': sympy.pi, 'π': sympy.pi, 'i': sympy.I,
    'g': 9.80665 * u.meter / u.second**2, # Earth Gravity
    'atm': 101325 * u.pascal
}

# -----------------------------------------------------------------------------
# 3. DUAL-ENGINE LOGIC
# -----------------------------------------------------------------------------

def engine_1_regex(text):
    """
    Fast, offline conversion for common patterns.
    Returns: (expression_string, was_matched)
    """
    text = text.lower()
    patterns = [
        (r"derivative of (.+) with respect to (\w+)", r"diff(\1, \2)"),
        (r"derivative of (.+)", r"diff(\1, x)"),
        (r"d/dx (.+)", r"diff(\1, x)"),
        (r"integrate (.+) from ([\d\.]+) to ([\d\.]+)", r"integrate(\1, (x, \2, \3))"),
        (r"integrate (.+)", r"integrate(\1, x)"),
        (r"solve (.+) for (\w+)", r"solve(\1, \2)"),
        (r"solve (.+)", r"solve(\1, x)"),
        (r"roots of (.+)", r"solve(\1, x)"),
        (r"(.+) squared", r"(\1)**2"),
        (r"(.+) cubed", r"(\1)**3"),
        (r"sqrt (.+)", r"sqrt(\1)"),
    ]
    
    original = text
    for p, r in patterns:
        text = re.sub(p, r, text)
    
    # If text changed, we matched a pattern
    return text, text != original

def engine_2_ai_translation(api_key, text):
    """
    Uses LLM to convert complex natural language to SymPy syntax.
    Example: "Escape velocity of earth" -> "sqrt(2 * G * 5.97e24 / 6.37e6)"
    """
    if not api_key: return None
    
    prompt = f"""
    You are a Math Translator. Convert this English query into a valid Python/SymPy expression.
    
    Query: "{text}"
    
    Rules:
    1. Use these constants if needed: c, G, h, e, me, mp, k, eps0, mu0, g (9.8), pi.
    2. If the user asks for a known value (e.g. "mass of earth"), write the number in scientific notation (e.g. 5.97e24).
    3. Return ONLY the code. No markdown, no "Here is the code".
    
    Examples:
    "Kinetic energy of 5kg at 10m/s" -> "0.5 * 5 * 10**2"
    "50 fahrenheit in celsius" -> "(50 - 32) * 5/9"
    "sin squared x" -> "sin(x)**2"
    """
    
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": "llama3-70b-8192", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
        
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content'].strip()
    except:
        return None
    return None

def omni_solve(query, api_key):
    # 1. Try Engine 1 (Regex)
    expr_str, matched = engine_1_regex(query)
    
    # Context
    local_dict = CONSTANTS.copy()
    local_dict.update({
        'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
        'exp': sympy.exp, 'log': sympy.log, 'ln': sympy.ln, 
        'sqrt': sympy.sqrt, 'diff': sympy.diff, 'integrate': sympy.integrate,
        'solve': sympy.solve, 'Matrix': sympy.Matrix
    })
    x, y, z, t = sympy.symbols('x y z t')
    local_dict.update({'x': x, 'y': y, 'z': z, 't': t})
    
    try:
        # 2. Try to parse the Regex result
        # Handle "Equation" input (LHS = RHS)
        if "=" in expr_str and "solve" not in expr_str:
            parts = expr_str.split("=")
            expr_str = f"solve({parts[0]} - ({parts[1]}), x)"
            
        res = sympy.sympify(expr_str, locals=local_dict)
        return res, "⚡ Instant Compute"
        
    except:
        # 3. If Regex failed, use Engine 2 (AI Translation)
        if api_key:
            ai_expr = engine_2_ai_translation(api_key, query)
            if ai_expr:
                try:
                    res = sympy.sympify(ai_expr, locals=local_dict)
                    return res, f"🧠 AI Interpreted: `{ai_expr}`"
                except Exception as e:
                    return None, f"AI Translation Error: {e}"
        
        return None, "Could not understand input."

# -----------------------------------------------------------------------------
# 4. UI & MAIN LOOP
# -----------------------------------------------------------------------------

# API Key Handling (Silent)
api_key = st.secrets.get("GROQ_API_KEY", "")
if not api_key:
    # Fallback for demo if not in secrets
    api_key = st.sidebar.text_input("Groq API Key (for 'Type Anything' mode)", type="password")

# APP HEADER
c1, c2 = st.columns([1, 6])
with c1: st.markdown("## ⚛️") 
with c2: st.markdown("## Omni-Calc")

# THE OMNIBAR
query = st.text_input("", placeholder="Type anything... 'derivative of sin x', '50C to F', 'kinetic energy of 5kg'", key="omnibar")

if query:
    with st.spinner("Processing..."):
        result, source_tag = omni_solve(query, api_key)
    
    if result is not None:
        # Format Result
        try:
            # Try to get numeric float
            numeric = result.evalf()
            
            # Check for Graphing (if it's a function of x)
            show_graph = False
            if hasattr(result, 'free_symbols') and len(result.free_symbols) == 1:
                show_graph = True

            st.markdown(f"""
            <div class="result-card">
                <div class="ai-badge">{source_tag}</div>
                <div class="main-result">{str(result).replace('**', '^')}</div>
                <div class="sub-result">
                    Decimal: {float(numeric):.6g}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive Graphing
            if show_graph:
                var = list(result.free_symbols)[0]
                f_lam = sympy.lambdify(var, result, modules=['numpy'])
                x_vals = np.linspace(-10, 10, 500)
                try:
                    y_vals = f_lam(x_vals)
                    y_vals = np.where(np.abs(y_vals) > 1e6, np.nan, y_vals) # Trim asymptotes
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, line=dict(color='#00ffcc', width=3)))
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(20,20,20,0.5)',
                        xaxis=dict(showgrid=False, color='#555'),
                        yaxis=dict(showgrid=True, gridcolor='#222', color='#555'),
                        margin=dict(l=0,r=0,t=30,b=0),
                        height=250
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except: pass

        except Exception as e:
            # Fallback for symbolic results (like 'x + y')
            st.markdown(f"""
            <div class="result-card">
                <div class="ai-badge">{source_tag}</div>
                <div class="main-result">{str(result)}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error(f"❌ {source_tag}")

# Quick Helper
st.markdown("---")
st.caption("Try: `solve x^2 + 5x + 6`, `force of 10kg at 9.8m/s^2`, `d/dx sin(x)*x`")
