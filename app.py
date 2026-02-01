import streamlit as st
import sympy
import sympy.physics.units as u
import numpy as np
import plotly.graph_objects as go
import requests
import re

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & STATE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Scientific Super Calculator v3.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
state_defaults = {
    'input_value': "",
    'history': [],
    'last_result_obj': None,
    'last_numeric': None,
    'ans': 0
}
for k, v in state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS (Cyberpunk/Pro Look)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    
    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #1e293b; color: #38bdf8;
        border: 1px solid #475569; font-family: 'Fira Code', monospace;
    }
    
    /* Buttons */
    .stButton > button {
        background: #334155; border: 1px solid #475569; color: white;
        border-radius: 6px; font-size: 14px;
    }
    .stButton > button:hover {
        border-color: #38bdf8; color: #38bdf8;
    }
    
    /* Result Card */
    .result-box {
        background: rgba(30, 41, 59, 0.5); border-left: 4px solid #38bdf8;
        padding: 20px; border-radius: 8px; margin: 10px 0;
    }
    .result-main { font-size: 1.8em; font-weight: bold; color: #38bdf8; font-family: monospace; }
    .result-sub { font-size: 0.9em; color: #94a3b8; margin-top: 5px; }
    
    /* AI Box */
    .ai-panel {
        background: #312e81; border: 1px solid #4f46e5; 
        padding: 15px; border-radius: 8px; margin-top: 15px;
    }
    
    /* History Items */
    .history-btn { text-align: left; padding: 5px; cursor: pointer; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. CONSTANTS & PRESETS
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
    'pi': sympy.pi, 'π': sympy.pi, 'i': sympy.I
}

TEMPLATES = {
    "Select Template...": "",
    "Kinetic Energy": "0.5 * m * v^2",
    "Gravitational Force": "G * m1 * m2 / r^2",
    "Quadratic Root": "solve(a*x^2 + b*x + c, x)",
    "Derivative (Def)": "diff(f, x)",
    "Definite Integral": "integrate(f, (x, 0, 10))",
    "Matrix Mult": "Matrix([[1,2],[3,4]]) * Matrix([[x],[y]])"
}

# -----------------------------------------------------------------------------
# 4. LOGIC ENGINE (NLP + MATH)
# -----------------------------------------------------------------------------

def parse_natural_language(text):
    """Translates English to SymPy syntax."""
    text = text.lower()
    patterns = [
        (r"derivative of (.+) with respect to (\w+)", r"diff(\1, \2)"),
        (r"derivative of (.+)", r"diff(\1, x)"), # assume x
        (r"integrate (.+) from ([\d\.]+) to ([\d\.]+)", r"integrate(\1, (x, \2, \3))"),
        (r"integrate (.+)", r"integrate(\1, x)"),
        (r"solve (.+) for (\w+)", r"solve(\1, \2)"),
        (r"solve (.+)", r"solve(\1, x)"),
        (r"(.+) squared", r"(\1)**2"),
        (r"root of (.+)", r"sqrt(\1)"),
    ]
    for p, r in patterns:
        text = re.sub(p, r, text)
    return text

def smart_solver(expr_str, ans_val):
    """
    1. Pre-process NLP
    2. Inject 'ans' variable
    3. Handle Equations
    4. Parse & Solve
    """
    # 1. NLP Processing
    expr_str = parse_natural_language(expr_str)
    
    # 2. Handle 'Ans' (Previous Result)
    if 'ans' in expr_str.lower() and ans_val is not None:
        expr_str = expr_str.lower().replace('ans', str(ans_val))
    
    # 3. Equation Handling (LHS = RHS -> LHS - RHS)
    if "=" in expr_str and "solve" not in expr_str:
        parts = expr_str.split("=")
        expr_str = f"solve(({parts[0]}) - ({parts[1]}), x)"

    # Standardize
    expr_str = expr_str.replace('^', '**')

    # 4. Context Setup
    local_dict = CONSTANTS.copy()
    # Add common functions
    local_dict.update({
        'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
        'exp': sympy.exp, 'log': sympy.log, 'ln': sympy.ln, 
        'sqrt': sympy.sqrt, 'diff': sympy.diff, 'integrate': sympy.integrate,
        'solve': sympy.solve, 'limit': sympy.limit, 'Matrix': sympy.Matrix
    })
    # Add symbols
    vars_ = sympy.symbols('x y z t a b c m v r m1 m2 f')
    local_dict.update({k.name: k for k in vars_})

    try:
        res = sympy.sympify(expr_str, locals=local_dict)
        return res, expr_str, None
    except Exception as e:
        return None, expr_str, str(e)

def get_ai_explanation(api_key, expr, res, history_ctx):
    """Rich context AI explanation."""
    if not api_key: return "⚠️ Please add API Key for AI insights."
    
    prompt = f"""
    Role: Expert Math/Physics Tutor.
    Current Input: {expr}
    Computed Result: {res}
    Session History: {history_ctx[-3:]}
    
    Task:
    1. Explain what this calculation represents (Physics/Math concept).
    2. Break down the result (units, significance).
    3. Suggest a relevant "Next Step" calculation.
    4. Keep it concise (bullet points).
    """
    
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"AI Error: {e}"

# -----------------------------------------------------------------------------
# 5. UI LAYOUT
# -----------------------------------------------------------------------------

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key Logic
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password")
    else:
        st.success("✅ AI Connected")

    # Features
    auto_explain = st.toggle("🤖 Auto-Explain", value=True, help="Get AI insights automatically")
    
    st.divider()
    
    # Template Loader
    st.subheader("📑 Smart Templates")
    selected_template = st.selectbox("Load Formula:", list(TEMPLATES.keys()))
    if selected_template and selected_template != "Select Template...":
        st.session_state.input_value = TEMPLATES[selected_template]

    # History Panel
    st.divider()
    st.subheader("📜 History (Click to Load)")
    for i, item in enumerate(reversed(st.session_state.history[-10:])):
        # Truncate for display
        lbl = f"{item['expr'][:20]}... = {str(item['res'])[:10]}..."
        if st.button(lbl, key=f"hist_{i}"):
            st.session_state.input_value = item['expr']
            st.rerun()

# --- MAIN PAGE ---
st.title("🧠 Scientific Super Calculator v3.0")

# Tabs for Organization
tab_calc, tab_ref = st.tabs(["🧮 Calculator", "📚 Reference"])

with tab_calc:
    # INPUT SECTION
    c1, c2 = st.columns([4, 1])
    with c1:
        # The main input box
        query = st.text_input("Enter Math, Physics, or Natural Language:", 
                             key="input_value", 
                             placeholder="e.g. 'derivative of sin(x)' or 'G * me * mp / r^2'")
    with c2:
        # Calculate Button
        st.write("") # spacer
        st.write("") 
        calc_btn = st.button("🚀 Solve", use_container_width=True)

    # PROCESS LOGIC
    if query and (calc_btn or query != ""): # Reacts to enter key if logic permits
        if calc_btn: # Explicit trigger
            
            # 1. SOLVE
            result_obj, final_expr, err = smart_solver(query, st.session_state.ans)
            
            if err:
                st.error(f"❌ Syntax Error: {err}")
                st.caption(f"Parsed as: `{final_expr}`")
            else:
                # 2. PROCESS RESULT
                try:
                    # Simplify & Eval
                    simplified = sympy.simplify(result_obj)
                    numeric = simplified.evalf()
                    
                    # Update State
                    st.session_state.ans = numeric
                    st.session_state.last_result_obj = simplified
                    
                    # Add to History
                    st.session_state.history.append({'expr': query, 'res': str(simplified)})
                    
                    # 3. DISPLAY
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-sub">INPUT: {final_expr}</div>
                        <div class="result-main">{str(simplified).replace('**', '^')}</div>
                        <div class="result-sub">≈ {float(numeric):.6e} (Numeric)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 4. PLOTTING (Automatic)
                    if hasattr(simplified, 'free_symbols') and len(simplified.free_symbols) == 1:
                        var = list(simplified.free_symbols)[0]
                        # Only plot if it's a function of 1 variable (math mode)
                        try:
                            f = sympy.lambdify(var, simplified, modules=['numpy'])
                            x = np.linspace(-10, 10, 400)
                            y = f(x)
                            # Clean mess
                            y = np.where(np.abs(y) > 1e10, np.nan, y)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='#38bdf8', width=3)))
                            fig.update_layout(
                                title=f"Graph of {str(simplified)}", 
                                template="plotly_dark", height=300,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15, 23, 42, 0.5)'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except: pass

                    # 5. AI EXPLAINER (Auto or Manual)
                    if auto_explain:
                        with st.spinner("🤖 AI is analyzing context..."):
                            explanation = get_ai_explanation(api_key, query, str(simplified), st.session_state.history)
                            st.markdown(f"""
                            <div class="ai-panel">
                                <strong>💡 AI Insight</strong>
                                <div style="margin-top:10px; font-size: 0.95em;">{explanation}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"Calculation Error: {e}")

with tab_ref:
    st.markdown("### 📘 Constants & Cheat Sheet")
    st.json({k: str(v) for k, v in list(CONSTANTS.items())[:15]})
    st.markdown("""
    **Shortcuts:**
    * `ans` : Use previous result
    * `diff(y, x)` : Derivative
    * `integrate(y, x)` : Integral
    * `solve(eq, x)` : Algebra
    * `Matrix([[a,b],[c,d]])` : Linear Algebra
    """)

# Footer
st.markdown("---")
st.caption(f"v3.0 | 💾 Memory: {len(st.session_state.history)} items | 🔌 AI: {'Connected' if api_key else 'Offline'}")
