import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="Premium Real Estate Analytics",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a beautiful, modern UI
st.markdown("""
<style>
    /* Main Background & Font */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Gradient Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
    }
    .prediction-card h3 {
        color: #E2E8F0;
        font-weight: 400;
        margin-bottom: 0.5rem;
    }
    .prediction-value {
        font-size: 4.5rem !important;
        font-weight: 800;
        color: #4ADE80; /* Nice bright green */
        margin: 0;
        line-height: 1.2;
    }
    
    /* Subtle Metric Cards */
    .metric-card {
        background-color: white;
        border-left: 5px solid #3B82F6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Headers */
    .main-title {
        font-size: 3rem;
        color: #1E293B;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-title {
        color: #64748B;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Data Loading & Caching
# ==========================================
@st.cache_data
def load_data():
    cal_housing = fetch_california_housing()
    df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    df['Price'] = cal_housing.target * 100000 
    return df, cal_housing.feature_names

# ==========================================
# 3. Model Training & Caching
# ==========================================
@st.cache_resource
def train_model(df, features):
    X = df[features]
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return model, mae, r2

with st.spinner('Initializing Real Estate Engine...'):
    df, feature_names = load_data()
    model, mae, r2 = train_model(df, feature_names)

# ==========================================
# 4. Sidebar / User Inputs
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2111/2111320.png", width=60)
    st.title("Property Specs")
    st.markdown("Fine-tune the parameters below to instantly generate a new property valuation.")
    st.divider()

def user_input_features():
    inputs = {}
    
    st.sidebar.subheader("📍 Location")
    # Using San Francisco area defaults for better map initial view
    inputs['Latitude'] = st.sidebar.slider('Latitude', float(df['Latitude'].min()), float(df['Latitude'].max()), 37.77, 0.05)
    inputs['Longitude'] = st.sidebar.slider('Longitude', float(df['Longitude'].min()), float(df['Longitude'].max()), -122.41, 0.05)
    
    st.sidebar.subheader("👥 Demographics")
    inputs['MedInc'] = st.sidebar.slider('Median Income (x$10k)', float(df['MedInc'].min()), float(df['MedInc'].max()), float(df['MedInc'].median()), 0.5)
    inputs['Population'] = st.sidebar.slider('Local Population', float(df['Population'].min()), float(df['Population'].quantile(0.95)), float(df['Population'].median()), 100.0)
    
    st.sidebar.subheader("🏠 Property Details")
    inputs['HouseAge'] = st.sidebar.slider('House Age (Years)', float(df['HouseAge'].min()), float(df['HouseAge'].max()), float(df['HouseAge'].median()), 1.0)
    inputs['AveRooms'] = st.sidebar.slider('Average Rooms', float(df['AveRooms'].min()), 15.0, float(df['AveRooms'].median()), 0.5)
    inputs['AveBedrms'] = st.sidebar.slider('Average Bedrooms', float(df['AveBedrms'].min()), 5.0, float(df['AveBedrms'].median()), 0.1)
    inputs['AveOccup'] = st.sidebar.slider('Average Occupants', float(df['AveOccup'].min()), 10.0, float(df['AveOccup'].median()), 0.5)
    
    input_df = pd.DataFrame(inputs, index=[0])
    input_df = input_df[feature_names] # Prevents ValueError
    return input_df

input_df = user_input_features()

# ==========================================
# 5. Main UI Layout
# ==========================================
st.markdown('<p class="main-title">AI Real Estate Valuator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by Scikit-Learn Machine Learning</p>', unsafe_allow_html=True)

# Generate Prediction
prediction = model.predict(input_df)[0]

# Beautiful Custom Prediction Card
st.markdown(f"""
<div class="prediction-card">
    <h3>Estimated Market Value</h3>
    <p class="prediction-value">${prediction:,.0f}</p>
    <p style="color: #94A3B8; margin-top: 10px;">Based on current dynamic parameters</p>
</div>
""", unsafe_allow_html=True)

# Use Tabs to organize information cleanly
tab1, tab2 = st.tabs(["📊 Property Dashboard", "⚙️ Model Analytics"])

with tab1:
    col_map, col_details = st.columns([2, 1])
    
    with col_map:
        st.subheader("📍 Property Location & Market Map")
        st.markdown("*Hover over the blue market points to see local area valuations.*")
        
        # 1. Background Market Data (to show area values on hover)
        # We take a sample of 2,500 points from our dataset to keep the browser running fast
        market_sample = df.sample(n=2500, random_state=42).copy()
        market_sample['Area Market Value'] = market_sample['Price'].apply(lambda x: f"${x:,.0f}")
        
        # Create the base map with historical area prices
        fig_map = px.scatter_mapbox(
            market_sample, 
            lat="Latitude", 
            lon="Longitude", 
            color="Price",
            color_continuous_scale="Blues", # Darker blue = higher price
            hover_name="Area Market Value",
            hover_data={"Latitude": False, "Longitude": False, "Price": False},
            zoom=5.5, 
            height=450,
            mapbox_style="open-street-map" # Keeps the water/mountains/grass
        )
        
        # 2. Add the User's Target Property as a giant red pin on top
        fig_map.add_scattermapbox(
            lat=[input_df['Latitude'][0]],
            lon=[input_df['Longitude'][0]],
            mode='markers',
            marker=dict(size=16, color='#FF3B30'), # Bright red pin
            hovertext=['🎯 Target Property Location'],
            hoverinfo='text',
            name='Target Property'
        )
        
        # Remove map margins and hide the colorbar to keep it looking clean
        fig_map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_showscale=False, 
            showlegend=False
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
    with col_details:
        st.subheader("📋 Selected Features")
        st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

with tab2:
    st.subheader("🧠 How the AI makes decisions")
    
    col_chart, col_metrics = st.columns([2, 1])
    
    with col_chart:
        # Interactive Plotly Chart for Feature Importance
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=True)
        
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title="Feature Impact on Price",
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
    with col_metrics:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="Mean Absolute Error", value=f"${mae:,.0f}")
        st.markdown("*Average deviation of predictions.*")
        st.markdown("</div><br>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(label="R² Score (Accuracy)", value=f"{r2:.1%}")
        st.markdown("*Percentage of variance explained.*")
        st.markdown("</div>", unsafe_allow_html=True)