import streamlit as st
import pint
import sympy as sp
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from groq import Groq
import math
import re

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="Newton-AI: Physics Engine",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR CALCULATOR LOOK ---
st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-family: 'Courier New', monospace;
        font-size: 1.5rem;
        background-color: #2b2b2b;
        color: #00ff00;
        border-radius: 10px;
    }
    div.stButton > button {
        width: 100%;
        height: 3em;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00ff00;
    }
    .unit-text {
        color: #ffa500;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'input_expression' not in st.session_state:
    st.session_state.input_expression = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# --- ENGINE: UNIT REGISTRY & CONSTANTS ---
ureg = pint.UnitRegistry()
ureg.define('percent = 0.01 = %')

# Define Constants Map for the Calculator
CONSTANTS = {
    'c': ureg.speed_of_light,
    'G': ureg.gravitational_constant,
    'h': ureg.planck_constant,
    'hbar': ureg.planck_constant / (2 * math.pi),
    'k': ureg.boltzmann_constant,
    'e': ureg.elementary_charge,
    'NA': ureg.avogadro_number,
    'g': ureg.standard_gravity,
    'm_e': ureg.electron_mass,
    'm_p': ureg.proton_mass,
}

# SymPy symbols for graphing
x, y, z, t = sp.symbols('x y z t')

# --- ENGINE: CORE LOGIC ---

def safe_evaluate(expression):
    """
    Evaluates an expression supporting physical units and constants.
    """
    try:
        # 1. Pre-process power operator for Python
        expr_clean = expression.replace('^', '**')
        
        # 2. Build the evaluation context
        context = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'sqrt': np.sqrt, 'log': np.log10, 'ln': np.log,
            'pi': np.pi, 'exp': np.exp,
        }
        # Add units and constants to context
        context.update(CONSTANTS)
        
        # 3. Evaluate using Pint
        # We wrap this in ureg.Quantity to ensure it's treated as a physical qty
        result = ureg.parse_expression(expr_clean, values=context)
        
        return result, None
    except Exception as e:
        return None, str(e)

def format_result(quantity):
    """
    Intelligently formats the output unit (e.g., kg*m^2/s^2 -> Joules)
    """
    if isinstance(quantity, (int, float, complex)):
        return quantity, ""
    
    try:
        # Attempt to simplify to compact SI units
        compact = quantity.to_compact()
        
        # Specific overrides for common physics scenarios
        if compact.check('[length] * [mass] / [time]**2'): # Force Newton
            compact = compact.to('newton')
        elif compact.check('[length]**2 * [mass] / [time]**2'): # Force Joule
            compact = compact.to('joule')
        elif compact.check('[length]**2 * [mass] / [time]**3'): # Force Watt
            compact = compact.to('watt')
        
        val = f"{compact.magnitude:.4g}"
        unit = f"{compact.units}"
        return val, unit
    except:
        return f"{quantity.magnitude:.4g}", f"{quantity.units}"

def get_ai_explanation(api_key, expression, result_val, result_unit):
    """
    Calls Groq API to explain the physics.
    """
    if not api_key:
        return "⚠️ Please enter a Groq API Key in the sidebar to enable AI explanations."
    
    client = Groq(api_key=api_key)
    
    prompt = f"""
    Role: You are a Physics Professor and Tutor.
    
    Task: Explain the following calculation to a student.
    
    Input Expression: {expression}
    Calculated Result: {result_val} {result_unit}
    
    Please provide:
    1. The likely Physics formula or law used.
    2. What the variables likely represent (guess based on standard physics notation).
    3. Why the resulting unit ({result_unit}) makes sense conceptually.
    4. A one-sentence real-world analogy.
    
    Keep it concise, encouraging, and educational. Use Markdown.
    """
    
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI Connection Error: {e}"

def generate_graph(expression):
    """
    Generates a Plotly graph if the expression contains 'x'.
    """
    try:
        # Clean expression for SymPy
        expr_clean = expression.replace('^', '**')
        # Define x variable
        x_vals = np.linspace(-10, 10, 400)
        
        # Parse logic
        # We need to map our physical constants to numbers for the plot
        # Sympy parsing
        local_dict = {'x': sp.Symbol('x'), 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp}
        
        # Simplified parser for graphing (ignores units for the plot axis usually)
        # This is a basic implementation. Robust graphing with units is complex.
        # Here we assume y = f(x) where result is dimensionless or standard SI
        
        # Quick hack: Remove units from string for pure mathematical plotting if needed
        # Or, just evaluate f(x) using the safe_evaluate function iteratively
        
        y_vals = []
        for val in x_vals:
            subbed = expr_clean.replace('x', f"({val})")
            res, err = safe_evaluate(subbed)
            if res is not None:
                # If result is quantity, take magnitude
                if hasattr(res, 'magnitude'):
                    y_vals.append(res.to_base_units().magnitude)
                else:
                    y_vals.append(res)
            else:
                y_vals.append(np.nan)
        
        fig = px.line(x=x_vals, y=y_vals, labels={'x': 'x', 'y': 'f(x)'})
        fig.update_layout(title="Function Plot", template="plotly_dark", height=300)
        return fig

    except Exception as e:
        return None

# --- UI LAYOUT ---

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    groq_api_key = st.text_input("Groq API Key (for AI Tutor)", type="password")
    
    st.divider()
    st.subheader("📚 Constant Reference")
    st.markdown("""
    | Key | Constant | Value |
    |---|---|---|
    | `c` | Light Speed | $3 \\times 10^8$ m/s |
    | `G` | Gravity | $6.67 \\times 10^{-11}$ |
    | `h` | Planck | $6.62 \\times 10^{-34}$ |
    | `g` | Earth Gravity | $9.8$ m/s² |
    | `k` | Boltzmann | $1.38 \\times 10^{-23}$ |
    | `NA` | Avogadro | $6.022 \\times 10^{23}$ |
    """)
    
    st.divider()
    st.subheader("History")
    for item in st.session_state.history[-5:]:
        st.caption(f"{item['expr']} = {item['val']} {item['unit']}")

# Main Interface
st.title("⚛️ Newton-AI: Physics Calculator")
st.markdown("*Unit-aware. Physics-smart. AI-explained.*")

# Display Screen (The result)
result_container = st.container()

# Input Area
input_col1, input_col2 = st.columns([4, 1])
with input_col1:
    expression = st.text_input("Enter Physics Expression:", value=st.session_state.input_expression, placeholder="e.g., 5*kg * 9.8*m/s^2 or m*c^2", key="main_input")
with input_col2:
    if st.button("CALCULATE", type="primary", use_container_width=True):
        st.session_state.input_expression = expression # Sync
        
        if expression:
            raw_result, error = safe_evaluate(expression)
            
            if error:
                st.error(f"Syntax Error: {error}")
            else:
                val, unit = format_result(raw_result)
                st.session_state.last_result = {"val": val, "unit": unit, "raw": raw_result, "expr": expression}
                st.session_state.history.append({"val": val, "unit": unit, "expr": expression})

# Render Result
if st.session_state.last_result:
    res = st.session_state.last_result
    with result_container:
        st.markdown(f"""
        <div class="metric-card">
            <h2 style="margin:0; color:#888;">{res['expr']} =</h2>
            <h1 style="margin:0; font-size: 3em;">{res['val']} <span class="unit-text">{res['unit']}</span></h1>
        </div>
        """, unsafe_allow_html=True)

# Helper Function for Buttons
def add_to_input(val):
    # This is a bit tricky in Streamlit. We update state and rerun.
    current = st.session_state.main_input
    st.session_state.input_expression = current + str(val)
    st.rerun()

# Calculator Grid UI
st.divider()
tabs = st.tabs(["⌨️ Keypad", "🧠 AI Tutor", "📈 Graphing"])

with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.button("7", on_click=add_to_input, args=("7",), use_container_width=True)
        st.button("4", on_click=add_to_input, args=("4",), use_container_width=True)
        st.button("1", on_click=add_to_input, args=("1",), use_container_width=True)
        st.button("0", on_click=add_to_input, args=("0",), use_container_width=True)
        st.button("sin(", on_click=add_to_input, args=("sin(",), use_container_width=True)

    with col2:
        st.button("8", on_click=add_to_input, args=("8",), use_container_width=True)
        st.button("5", on_click=add_to_input, args=("5",), use_container_width=True)
        st.button("2", on_click=add_to_input, args=("2",), use_container_width=True)
        st.button(".", on_click=add_to_input, args=(".",), use_container_width=True)
        st.button("cos(", on_click=add_to_input, args=("cos(",), use_container_width=True)

    with col3:
        st.button("9", on_click=add_to_input, args=("9",), use_container_width=True)
        st.button("6", on_click=add_to_input, args=("6",), use_container_width=True)
        st.button("3", on_click=add_to_input, args=("3",), use_container_width=True)
        st.button("^", on_click=add_to_input, args=("^",), use_container_width=True)
        st.button("sqrt(", on_click=add_to_input, args=("sqrt(",), use_container_width=True)

    with col4:
        st.button("DEL", on_click=lambda: st.session_state.update(input_expression=st.session_state.main_input[:-1]), use_container_width=True)
        st.button("AC", on_click=lambda: st.session_state.update(input_expression=""), use_container_width=True)
        st.button("(", on_click=add_to_input, args=("(",), use_container_width=True)
        st.button(")", on_click=add_to_input, args=(")",), use_container_width=True)
        st.button("Ans", on_click=add_to_input, args=(st.session_state.last_result['val'] if st.session_state.last_result else "",), use_container_width=True)

    st.divider()
    st.caption("Quick Constants:")
    cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)
    cc1.button("c", on_click=add_to_input, args=("c",), use_container_width=True, help="Speed of Light")
    cc2.button("G", on_click=add_to_input, args=("G",), use_container_width=True, help="Gravitational Constant")
    cc3.button("h", on_click=add_to_input, args=("h",), use_container_width=True, help="Planck Constant")
    cc4.button("g", on_click=add_to_input, args=("g",), use_container_width=True, help="Standard Gravity")
    cc5.button("m_e", on_click=add_to_input, args=("m_e",), use_container_width=True, help="Electron Mass")
    cc6.button("eV", on_click=add_to_input, args=("eV",), use_container_width=True, help="Electron Volt")

with tabs[1]:
    if st.session_state.last_result:
        st.subheader("Professor AI Explanation")
        if st.button("Explain this Result", type="secondary"):
            with st.spinner("Analyzing physics..."):
                explanation = get_ai_explanation(
                    groq_api_key, 
                    st.session_state.last_result['expr'],
                    st.session_state.last_result['val'],
                    st.session_state.last_result['unit']
                )
                st.markdown(explanation)
    else:
        st.info("Calculate something first to get an explanation.")

with tabs[2]:
    st.subheader("Function Plotter")
    st.caption("Use 'x' as your variable. (e.g., sin(x) or x^2)")
    
    graph_expr = st.text_input("Function of x:", value=st.session_state.input_expression if 'x' in st.session_state.input_expression else "")
    
    if graph_expr:
        fig = generate_graph(graph_expr)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not plot. Ensure expression contains 'x' and valid math.")
