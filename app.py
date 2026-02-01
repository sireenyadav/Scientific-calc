import streamlit as st
import sympy
import numpy as np
import plotly.graph_objects as go
import requests
import re

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & STATE INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Scientific Super Calculator",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for Input Management
if 'input_value' not in st.session_state:
    st.session_state['input_value'] = ""
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None
if 'last_expr_obj' not in st.session_state:
    st.session_state['last_expr_obj'] = None

# Custom CSS
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: #1e293b; color: #00f2ff;
        border: 1px solid #334155; border-radius: 10px;
        font-family: 'Courier New', monospace; font-size: 18px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #3b82f6, #8b5cf6);
        color: white; border: none; border-radius: 8px;
        font-weight: bold; transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.05); box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
    }
    
    /* Result Box */
    .result-container {
        background-color: #1e293b; border-left: 5px solid #00f2ff;
        padding: 20px; border-radius: 5px; margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .result-title {
        color: #94a3b8; font-size: 0.9em; margin-bottom: 5px;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .result-value {
        color: #00f2ff; font-size: 1.5em;
        font-family: 'Courier New', monospace; font-weight: bold;
    }
    
    /* AI Explanation Box */
    .ai-box {
        background-color: #312e81; border: 1px solid #6366f1;
        padding: 15px; border-radius: 10px; margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. PHYSICS CONSTANTS DATA (CODATA 2022)
# -----------------------------------------------------------------------------
CONSTANTS = {
    # [cite_start]Universal [cite: 2]
    'c': 299792458, 'G': 6.67430e-11, 'h': 6.62607015e-34, 'hbar': 1.054571817e-34,
    # [cite_start]Electromagnetic [cite: 2]
    'e': 1.602176634e-19, 'mu0': 1.25663706127e-6, 'eps0': 8.8541878188e-12,
    # [cite_start]Atomic [cite: 5, 8, 11]
    'me': 9.1093837139e-31, 'mp': 1.67262192595e-27, 'mn': 1.67492750056e-27,
    # [cite_start]Physicochemical [cite: 17]
    'NA': 6.02214076e23, 'k': 1.380649e-23, 'R': 8.314462618, 'sigma': 5.670374419e-8
}
# Aliases
CONSTANTS.update({'pi': sympy.pi, 'π': sympy.pi, 'ħ': CONSTANTS['hbar'], 
                 'ε0': CONSTANTS['eps0'], 'μ0': CONSTANTS['mu0'], 'σ': CONSTANTS['sigma']})

# -----------------------------------------------------------------------------
# 3. CORE LOGIC ENGINE
# -----------------------------------------------------------------------------

def insert_symbol(symbol):
    """Callback to insert text into the input field."""
    st.session_state.input_value += str(symbol)

def safe_parse_expression(expr_str):
    """Sanitizes, handles equations, and parses into SymPy."""
    if not expr_str: return "Empty Input"
    
    # FIX 1: Handle Equations (LHS = RHS -> LHS - RHS)
    if "=" in expr_str:
        parts = expr_str.split("=")
        if len(parts) == 2:
            expr_str = f"({parts[0]}) - ({parts[1]})"
    
    expr_str = expr_str.replace('^', '**')
    
    # Setup SymPy context
    x, y, z, t, r, m = sympy.symbols('x y z t r m')
    local_dict = CONSTANTS.copy()
    local_dict.update({
        'x': x, 'y': y, 'z': z, 't': t, 'r': r, 'm': m,
        'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
        'exp': sympy.exp, 'log': sympy.log, 'ln': sympy.ln, 'sqrt': sympy.sqrt,
        'diff': sympy.diff, 'integrate': sympy.integrate, 'solve': sympy.solve
    })

    try:
        return sympy.sympify(expr_str, locals=local_dict)
    except Exception as e:
        return f"Parse Error: {e}"

def plot_expression(expr_obj):
    """Generates a robust Plotly graph with error masking."""
    free_symbols = list(expr_obj.free_symbols)
    if len(free_symbols) != 1:
        return None, "Plotting requires exactly one variable (e.g., 'x')."
    
    var = free_symbols[0]
    try:
        f = sympy.lambdify(var, expr_obj, modules=['numpy'])
        x_vals = np.linspace(-10, 10, 1000)
        
        # FIX 4: Handle Domain Errors (1/0, sqrt(-1))
        with np.errstate(divide='ignore', invalid='ignore'):
            y_vals = f(x_vals)
        
        # Handle complex numbers
        if np.iscomplexobj(y_vals):
            y_vals = np.real_if_close(y_vals)
            
        y_vals = np.array(y_vals, dtype=float)
        
        # Filter NaNs and Infinities for plotting
        mask = np.isfinite(y_vals)
        x_plot = x_vals[mask]
        y_plot = y_vals[mask]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', name=str(expr_obj),
                                line=dict(color='#00f2ff', width=3)))
        
        fig.update_layout(
            title=f"y = {str(expr_obj)}",
            xaxis_title=str(var), yaxis_title="f(x)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30, 41, 59, 0.5)',
            hovermode="x"
        )
        return fig, None
    except Exception as e:
        return None, f"Plotting Error: {e}"

def query_groq_ai(api_key, expression, result):
    if not api_key: return "⚠️ Missing API Key"
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = f"Explain this Physics/Math formula: {expression}. Result: {result}. Identify variables, constants, and meaning. Be concise."
    
    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, 
                             json={"model": "llama3-70b-8192", "messages": [{"role": "user", "content": prompt}]})
        return resp.json()['choices'][0]['message']['content'] if resp.status_code == 200 else f"Error: {resp.text}"
    except Exception as e: return f"Error: {e}"

# -----------------------------------------------------------------------------
# 4. UI LAYOUT & INTERACTION
# -----------------------------------------------------------------------------

# Sidebar
with st.sidebar:
    st.title("⚙️ Control Panel")
    groq_api_key = st.text_input("Groq API Key", type="password")
    with st.expander("📚 Constants (Quick Ref)"):
        st.write(CONSTANTS)
    
    st.subheader("📜 History")
    for item in st.session_state['history'][-5:]:
        st.code(f"{item['expr']} = {item['res']}", language="python")

# Main Area
st.title("⚛️ Scientific Super Calculator")

# FIX 2: Functional Quick Buttons
col1, col2, col3, col4, col5 = st.columns(5)
col1.button("c", on_click=insert_symbol, args=("c",))
col2.button("G", on_click=insert_symbol, args=("G",))
col3.button("ħ", on_click=insert_symbol, args=("ħ",))
col4.button("π", on_click=insert_symbol, args=("π",))
col5.button("ε0", on_click=insert_symbol, args=("ε0",))

# Main Input (Linked to Session State)
user_input = st.text_input("Expression / Equation:", key="input_value", placeholder="e.g., E = m*c^2 or diff(sin(x), x)")

# Action Buttons
b1, b2, b3 = st.columns([1, 1, 2])
calc_clicked = b1.button("🚀 Calculate")
# FIX 3: Disable AI button if no result
explain_clicked = b3.button("🤖 Explain with AI", disabled=st.session_state['last_result'] is None)

# Execution Logic
if calc_clicked:
    res_obj = safe_parse_expression(st.session_state.input_value)
    
    if isinstance(res_obj, str) and ("Error" in res_obj):
        st.error(res_obj)
    else:
        # Display Logic
        if hasattr(res_obj, 'free_symbols') and not res_obj.free_symbols:
            display_val = f"{float(res_obj.evalf()):.6g}" # Numeric
        else:
            display_val = str(res_obj).replace('**', '^') # Symbolic

        st.session_state['last_result'] = display_val
        st.session_state['last_expr_obj'] = res_obj
        st.session_state['history'].append({'expr': st.session_state.input_value, 'res': display_val})

# Result Display
if st.session_state['last_result']:
    st.markdown(f'<div class="result-container"><div class="result-title">Result</div><div class="result-value">{st.session_state["last_result"]}</div></div>', unsafe_allow_html=True)
    
    obj = st.session_state['last_expr_obj']
    
    # FIX 5: Symbolic Substitution & Numeric Evaluation
    if hasattr(obj, 'free_symbols') and len(obj.free_symbols) > 0:
        st.latex(sympy.latex(obj))
        
        with st.expander("🔢 Substitute Values (Calculate Numeric)"):
            subs_dict = {}
            cols = st.columns(len(obj.free_symbols))
            for idx, sym in enumerate(obj.free_symbols):
                val = cols[idx].text_input(f"{sym} =", key=f"sub_{sym}")
                if val:
                    try:
                        subs_dict[sym] = float(val)
                    except: pass
            
            if len(subs_dict) == len(obj.free_symbols):
                try:
                    numeric_res = obj.subs(subs_dict).evalf()
                    st.success(f"Numeric Value: {float(numeric_res):.6e}")
                except Exception as e:
                    st.warning(f"Calculation error: {e}")

    # Plotting Logic
    if hasattr(obj, 'free_symbols') and len(obj.free_symbols) == 1:
        fig, err = plot_expression(obj)
        if fig: st.plotly_chart(fig, use_container_width=True)
        elif err: st.warning(err)

# AI Logic
if explain_clicked and st.session_state['last_result']:
    with st.spinner("Analyzing..."):
        expl = query_groq_ai(groq_api_key, st.session_state.input_value, st.session_state['last_result'])
        st.markdown(f'<div class="ai-box"><h3>🤖 Analysis</h3>{expl}</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("v1.0.1 | Engineered with SymPy, Plotly, & Streamlit")
