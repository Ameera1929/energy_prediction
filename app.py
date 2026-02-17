import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(page_title="Energy Predictor", layout="wide")

# ======================================
# CUSTOM CSS (Modern Dark UI)
# ======================================
st.markdown("""
<style>
/* ---------- TITLE ---------- */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg, #4f46e5, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* ---------- BUTTON ---------- */
.stButton>button {
    border-radius: 14px;
    padding: 12px 22px;
    font-weight: 600;
    background: linear-gradient(90deg,#6366f1,#22c55e);
    color: white;
    border: none;
    transition: 0.25s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79,70,229,0.35);
}

/* ---------- METRIC / BOX ---------- */
[data-testid="stMetric"], .energy-box, .suggestion-box {
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    text-align: center;
    transition: all 0.3s ease;
    border: 1px solid rgba(0,0,0,0.1);
}

/* BIG PREDICTED VALUE */
[data-testid="stMetricValue"], .energy-box {
    font-size: 36px !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg,#6366f1,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* labels */
[data-testid="stMetricLabel"], .suggestion-box {
    font-size: 18px !important;
    font-weight: 600;
    opacity: 0.85;
}

/* ---------- LIGHT MODE ---------- */
@media (prefers-color-scheme: light) {
.stApp {
    background: linear-gradient(180deg,#f8f7ff,#eef2ff,#f0f9ff);
    color: #111827;
}
[data-testid="stMetric"], .energy-box, .suggestion-box {
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(99,102,241,0.18);
    box-shadow: 0 10px 30px rgba(99,102,241,0.15);
}
[data-testid="stMetricValue"], .energy-box {
    background: linear-gradient(90deg,#7c3aed,#2563eb,#06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
/* ---------- LIGHT MODE INPUTS ---------- */
@media (prefers-color-scheme: light) {

    /* number_input, slider, text_input outer box */
    div.stTextInput>div>input,
    div.stNumberInput>div>input,
    div.stSlider>div>input {
        background: rgba(255,255,255,0.9) !important;
        border: 1px solid rgba(124,58,237,0.25) !important;
        border-radius: 10px;
        color: #111827;
        padding: 6px 10px;
    }

    /* selectbox / dropdown outer box */
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border-radius: 14px !important;
        border: 1px solid rgba(124,58,237,0.25) !important;
        box-shadow: 0 4px 14px rgba(124,58,237,0.15);
    }

    /* dropdown menu panel */
    ul[role="listbox"] {
        background: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(124,58,237,0.25) !important;
        box-shadow: 0 10px 30px rgba(124,58,237,0.25);
    }

    /* dropdown option hover */
    li[role="option"]:hover {
        background: rgba(124,58,237,0.12) !important;
    }

    /* selected option */
    li[aria-selected="true"] {
        background: linear-gradient(90deg,#7c3aed,#2563eb) !important;
        color: white !important;
        font-weight: 600;
    }

}
/* ---------- COST BOXES ---------- */
.cost-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    transition: all 0.3s ease;
}

/* LIGHT MODE */
@media (prefers-color-scheme: light) {
    .cost-box {
        background: rgba(255,255,255,0.85);    /* light background */
        border: 1px solid rgba(99,102,241,0.18);
        box-shadow: 0 10px 30px rgba(99,102,241,0.15);
        color: #111827;                         /* dark text */
    }
}

/* DARK MODE */
@media (prefers-color-scheme: dark) {
    .cost-box {
        background: rgba(0,0,0,0.6);            /* dark background */
        border: 1px solid rgba(255,255,255,0.2);
        color: #ffffff;                          /* white text */
    }
}


/* ---------- DARK MODE ---------- */
@media (prefers-color-scheme: dark) {
.stApp {
    background: linear-gradient(180deg,#020617,#0f172a);
    color: #e5e7eb;
}
[data-testid="stMetric"], .energy-box, .suggestion-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
}
}
</style>
""", unsafe_allow_html=True)

# ======================================
# LOAD MODEL
# ======================================
model = joblib.load("Final_model_Epc.pkl")

# ======================================
# SESSION STATE
# ======================================
if "page" not in st.session_state:
    st.session_state.page = "input"

if "input_data" not in st.session_state:
    st.session_state.input_data = None
# ======================================
# INPUT PAGE
# ======================================
if st.session_state.page == "input":

    st.title("⚡ Smart Energy Consumption Predictor")

    col1, col2 = st.columns(2)

    with col1:
        house_size = st.number_input("🏠 House Size (sqft)", 100, 10000, 1500)
        num_rooms = st.number_input("🛏 Number of Rooms", 1, 20, 3)
        num_occupants = st.number_input("👨‍👩‍👧 Number of Occupants", 1, 15, 4)
        avg_temperature = st.slider("🌡 Average Temperature (°C)",25.0,48.0,37.0)

    with col2:
        ac_hours = st.number_input("❄ AC Hours per Day", 0, 24, 6)
        heater_hours = st.number_input("🔥 Heater Hours per Day", 0, 24, 1)
        fridge_count = st.number_input("🧊 Number of Fridges", 0, 5, 1)
        washing_usage = st.number_input("🧺 Washing Machine Usage (weekly)", 0, 14, 3)
        solar_panel = st.selectbox("☀ Solar Panel Installed?", ["No", "Yes"])

    solar_panel = 1 if solar_panel == "Yes" else 0

    if st.button("🚀 Predict Energy Usage"):

        input_data = np.array([[ 
            house_size,
            num_rooms,
            num_occupants,
            ac_hours,
            heater_hours,
            fridge_count,
            washing_usage,
            avg_temperature,
            solar_panel
        ]])

        st.session_state.input_data = input_data
        st.session_state.page = "result"
        st.rerun()

elif st.session_state.page == "result":

    # ------------------------------
    # SAFETY CHECK
    # ------------------------------
    if "input_data" not in st.session_state or st.session_state.input_data is None:
        st.warning("⚠ No input data found! Please enter inputs first.")
        st.session_state.page = "input"
        st.rerun()  # <- updated for Streamlit >=1.18
        st.stop()

    # ------------------------------
    # SAFE INPUT DATA (2D)
    # ------------------------------
    input_data = np.array(st.session_state.input_data).reshape(1, -1)
    input_data = np.nan_to_num(input_data, nan=0.0)

    # ------------------------------
    # MODEL PREDICTION
    # ------------------------------
    prediction = model.predict(input_data)[0]

    # ------------------------------
    # SIMULATIONS FOR SAVINGS
    # ------------------------------
    input_no_ac = input_data.copy()
    input_no_ac[0][3] = 0
    prediction_no_ac = model.predict(input_no_ac)[0]
    ac_saving = prediction - prediction_no_ac

    input_no_heater = input_data.copy()
    input_no_heater[0][4] = 0
    prediction_no_heater = model.predict(input_no_heater)[0]
    heater_saving = prediction - prediction_no_heater

    input_no_wash = input_data.copy()
    input_no_wash[0][6] = 0
    prediction_no_wash = model.predict(input_no_wash)[0]
    wash_saving = prediction - prediction_no_wash



    # ------------------------------
    # ESTIMATED ENERGY (CENTER)
    # ------------------------------
    st.markdown("<h2 style='text-align:center; margin-top:20px;'>⚡ Estimated Monthly Energy</h2>", unsafe_allow_html=True)
    st.markdown(
       
        f"<div style='display:flex; justify-content:center; align-items:center; margin-bottom:30px;'>"
        f"<div class='energy-box'>{prediction:.2f} kWh</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    
    # ------------------------------
    # POTENTIAL SAVINGS (SIDE BY SIDE)
    # ------------------------------
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("❄ AC Saving", f"{ac_saving:.2f} kWh")
    with c2: st.metric("🔥 Heater Saving", f"{heater_saving:.2f} kWh")
    with c3: st.metric("🧺 Washing Saving", f"{wash_saving:.2f} kWh")

    rate_per_kwh = 8  # ₹8 per kWh
    one_month_cost = prediction * rate_per_kwh
    two_month_cost = one_month_cost * 2
    c4,c5 = st.columns(2)
    box_style = """
       padding: 20px;
       border-radius: 15px;
       text-align: center;
       font-size: 22px;
       font-weight: bold;
       transition: all 0.3s ease;
    """

    with c4:
        st.markdown(
            f"<div class='cost-box'>💰 1 Month Cost<br>₹ {one_month_cost:.2f}</div>",
            unsafe_allow_html=True
        )
    with c5:
        st.markdown(
            f"<div class='cost-box'>💰 2 Months Cost<br>₹ {two_month_cost:.2f}</div>",
            unsafe_allow_html=True
        )
   
    
    # ------------------------------
    # SMART SUGGESTIONS BELOW
    # ------------------------------
    st.subheader("📦 Smart Energy Optimization Suggestions")
    suggestions = []
    if input_data[0][3] > 6:
        suggestions.append("Reduce AC usage by 2-3 hours daily.")
    if input_data[0][4] > 2:
        suggestions.append("Lower heater usage if possible.")
    if input_data[0][6] > 5:
        suggestions.append("Use washing machine only with full loads.")
    if input_data[0][8] == 0:
        suggestions.append("Consider installing solar panels for long-term savings.")

    if len(suggestions) == 0:
        st.success("Your energy usage looks optimized! 👍")
    else:
        for s in suggestions:
            st.markdown(f"<div class='suggestion-box'>⚡ {s}</div>", unsafe_allow_html=True)

    # ------------------------------
    # BACK BUTTON
    # ------------------------------
    if st.button("⬅ Back to Input Page"):
        st.session_state.page = "input"
        st.rerun()
