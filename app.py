import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
 
from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluate import Evaluator
 
# ------------------------------------------------
# Page Configuration & Modern Theme Setup
# ------------------------------------------------
st.set_page_config(
    page_title="FlightPulse | Predictive Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# Global Modern CSS Injection (Clean UI & Micro-interactions)
st.markdown("""
    <style>
    /* Global Background Adjustments */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Header Customization */
    h1, h2, h3, .stSubheader {
        font-family: 'Inter', system-ui, sans-serif !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }
    
    /* Card Container Wrappers */
    div[data-testid="stVerticalBlock"] > div {
        border-radius: 12px;
    }
    
    /* Premium KPI Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #58a6ff !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #8b949e !important;
    }
    .stMetric {
        background: linear-gradient(145deg, #161b22, #0f141c);
        border: 1px solid #30363d;
        padding: 24px 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .stMetric:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    
    /* Unified Dark Modern Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Interactive Dashboard Action Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #1f6feb 0%, #0975E7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        width: 100%;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        height: 3.2em;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 12px rgba(31, 111, 235, 0.3);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(31, 111, 235, 0.5);
        background: linear-gradient(90deg, #388bfd 0%, #1f6feb 100%) !important;
    }
    
    /* Clean Tab Design styling */
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #8b949e !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #58a6ff !important;
    }
    
    /* Divider Customization */
    hr {
        border-color: #30363d !important;
        margin: 2.5rem 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
 
# ------------------------------------------------
# Control Center Sidebar
# ------------------------------------------------
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Safely handle image loading to prevent app crashes
    try:
        st.image("assets/201623.png", width=70)
    except Exception:
        st.markdown("### ✈️ **FlightPulse**") # Fallback text if the image is corrupted
        
    st.markdown("### **Control Console**")
    st.markdown("---")
    
    future_months = st.slider("Forecast Horizon (Months)", 1, 24, 12)
    st.markdown("<small style='color: #8b949e;'>Adjust the window sequence to modify the RNN temporal lookup layer matrix during inference.</small>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("💡 **System Status**: `Operational`")
    st.markdown("⚙️ **Engine**: `Keras / LSTM-RNN`")

 
# ------------------------------------------------
# Data Processing Pipeline Core
# ------------------------------------------------
loader = DataLoader("data/airline-passengers.csv")
df = loader.load_data()
 
# Modern Header Banner Block
st.title("✈️ Airline Passenger Analysis & Forecasting")
st.markdown("<p style='font-size: 1.15rem; color: #8b949e; margin-top: -15px;'>Enterprise Core Engine: Predicting global operational travel density via Recurrent Neural Networks (RNN).</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------
# Analytical Engine & Performance Tabs
# ------------------------------------------------
tab1, tab2 = st.tabs(["📊 Operational Analytics", "🔬 Deep Learning Performance"])
 
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Model Validation Accuracy Metrics")
    mae, mse, rmse = Evaluator().evaluate()
   
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    m2.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    m3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
 
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 2], gap="large")
   
    with col_a:
        st.subheader("Historical Log Data")
        st.dataframe(
            df, 
            height=380, 
            use_container_width=True
        )
   
    with col_b:
        st.subheader("Historical Trend Progression")
        fig = px.line(
            df, x=df.index, y="passengers",
            template="plotly_dark",
            color_discrete_sequence=["#58a6ff"]
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(showgrid=True, gridcolor='#21262d'),
            yaxis=dict(showgrid=True, gridcolor='#21262d')
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
 
# ------------------------------------------------
# Modern Forecasting Engine UI Block
# ------------------------------------------------
st.markdown("---")
st.header("🔮 Generate Future Projections")
st.markdown("<p style='color: #8b949e; margin-top:-10px;'>Initiate a deep state forward pass execution inside the compiled network graph layers.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Execute Predictive Inference Engine"):
    with st.spinner("Executing sequence evaluation layers..."):
        forecaster = Forecaster()
        future = forecaster.forecast(future_months)
       
        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=future_months,
            freq="MS"
        )
 
        forecast_df = pd.DataFrame({
            "Month": future_dates,
            "Predicted Passengers": future.flatten()
        })
 
    st.toast(f"Successfully generated inference for {future_months} months!", icon="✅")
 
    # Operational Workspace Split Layout
    res_col1, res_col2 = st.columns([1, 2], gap="large")
 
    with res_col1:
        st.subheader("Inference Matrix Output")
        st.dataframe(forecast_df, use_container_width=True, height=340)
       
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Inference CSV Data",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv"
        )
 
    with res_col2:
        st.subheader("Unified System Projections")
       
        fig_combined = go.Figure()
       
        # Cleaned Custom Styled Historical Data Line
        fig_combined.add_trace(go.Scatter(
            x=df.index, y=df["passengers"],
            name="Historical Data", 
            line=dict(color="#8b949e", width=2)
        ))
       
        # Highlighted Modern Gradient Forecast Line
        fig_combined.add_trace(go.Scatter(
            x=forecast_df["Month"], y=forecast_df["Predicted Passengers"],
            name="RNN Model Forecast", 
            line=dict(color="#ff7f0e", width=3, dash='dash')
        ))
       
        fig_combined.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor='#21262d'),
            yaxis=dict(showgrid=True, gridcolor='#21262d')
        )
        st.plotly_chart(fig_combined, use_container_width=True, config={'displayModeBar': False})
