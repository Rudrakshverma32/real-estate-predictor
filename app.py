import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import folium
from folium.plugins import MiniMap
from streamlit_folium import st_folium

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="Premium Real Estate Analytics",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a beautiful, modern UI using the 'Inter' font
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* Apply professional font globally */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Main Background */
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
        letter-spacing: 0.02rem;
    }
    .prediction-value {
        font-size: 4.5rem !important;
        font-weight: 800;
        color: #4ADE80; /* Nice bright green */
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.1rem;
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
        color: #0F172A;
        font-weight: 800;
        margin-bottom: 2rem; /* Adjusted for removed subtitle */
        letter-spacing: -0.05rem;
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
# Initialize session state for coordinates if they don't exist
if 'lat' not in st.session_state:
    st.session_state.lat = 37.77
if 'lon' not in st.session_state:
    st.session_state.lon = -122.41

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2111/2111320.png", width=60)
    st.title("Property Specs")
    st.markdown("Fine-tune the parameters below to instantly generate a new property valuation.")
    st.divider()

def user_input_features():
    inputs = {}
    
    st.sidebar.subheader("📍 Location")
    st.sidebar.markdown("*Click anywhere on the main map to auto-update these coordinates!*")
    
    # Notice we removed the "key" argument here so it doesn't conflict with session_state
    inputs['Latitude'] = st.sidebar.slider('Latitude', -90.0, 90.0, float(st.session_state.lat), 0.05)
    inputs['Longitude'] = st.sidebar.slider('Longitude', -180.0, 180.0, float(st.session_state.lon), 0.05)
    
    # Update session state with slider values manually
    st.session_state.lat = inputs['Latitude']
    st.session_state.lon = inputs['Longitude']
    
    st.sidebar.subheader("👥 Demographics")
    inputs['MedInc'] = st.sidebar.slider('Median Income (x$10k)', float(df['MedInc'].min()), float(df['MedInc'].max()), float(df['MedInc'].median()), 0.5)
    inputs['Population'] = st.sidebar.slider('Local Population', float(df['Population'].min()), float(df['Population'].quantile(0.95)), float(df['Population'].median()), 100.0)
    
    st.sidebar.subheader("🏠 Property Details")
    inputs['HouseAge'] = st.sidebar.slider('House Age (Years)', float(df['HouseAge'].min()), float(df['HouseAge'].max()), float(df['HouseAge'].median()), 1.0)
    inputs['AveRooms'] = st.sidebar.slider('Average Rooms', float(df['AveRooms'].min()), 15.0, float(df['AveRooms'].median()), 0.5)
    inputs['AveBedrms'] = st.sidebar.slider('Average Bedrooms', float(df['AveBedrms'].min()), 5.0, float(df['AveBedrms'].median()), 0.1)
    inputs['AveOccup'] = st.sidebar.slider('Average Occupants', float(df['AveOccup'].min()), 10.0, float(df['AveOccup'].median()), 0.5)
    
    input_df = pd.DataFrame(inputs, index=[0])
    input_df = input_df[feature_names] 
    return input_df

input_df = user_input_features()

# ==========================================
# 5. Main UI Layout
# ==========================================
st.markdown('<p class="main-title">AI Real Estate Valuator</p>', unsafe_allow_html=True)

# Generate Prediction
prediction = model.predict(input_df)[0]

# Beautiful Custom Prediction Card
st.markdown(f"""
<div class="prediction-card">
    <h3>Estimated Market Value</h3>
    <p class="prediction-value">${prediction:,.0f}</p>
    <p style="color: #94A3B8; margin-top: 10px;">Based on current dynamic parameters at coordinates ({st.session_state.lat:.2f}, {st.session_state.lon:.2f})</p>
</div>
""", unsafe_allow_html=True)

# Use Tabs to organize information cleanly
tab1, tab2 = st.tabs(["📊 Property Dashboard", "⚙️ Model Analytics"])

with tab1:
    col_map, col_details = st.columns([2, 1])
    
    with col_map:
        st.subheader("📍 Interactive Property Map")
        st.markdown("*Click **anywhere on Earth** to move the target pin or hover over major cities to see hypothetical valuations!*")
        
        # Create Folium Map centered on current target location
        # scrollWheelZoom=False ensures your mouse scrollbar scrolls the PAGE, not just the map zoom
        m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=2, scrollWheelZoom=False)
        
        # Add an interactive MiniMap (acts as a navigational scroll/pan window)
        minimap = MiniMap(toggle_display=True, position="bottomleft")
        minimap.add_to(m)
        
        # Major Global Cities for AI Evaluation Reference
        world_cities = [
            {"name": "New York, USA", "lat": 40.7128, "lon": -74.0060},
            {"name": "London, UK", "lat": 51.5074, "lon": -0.1278},
            {"name": "Tokyo, Japan", "lat": 35.6762, "lon": 139.6503},
            {"name": "Sydney, Australia", "lat": -33.8688, "lon": 151.2093},
            {"name": "Paris, France", "lat": 48.8566, "lon": 2.3522},
            {"name": "Dubai, UAE", "lat": 25.2048, "lon": 55.2708},
            {"name": "Mumbai, India", "lat": 19.0760, "lon": 72.8777},
            {"name": "Cape Town, SA", "lat": -33.9249, "lon": 18.4241},
            {"name": "Rio de Janeiro, BR", "lat": -22.9068, "lon": -43.1729},
            {"name": "Toronto, Canada", "lat": 43.6510, "lon": -79.3470},
        ]
        
        for city in world_cities:
            # Temporarily calculate what the property would cost in this specific city
            city_df = input_df.copy()
            city_df['Latitude'] = city['lat']
            city_df['Longitude'] = city['lon']
            city_pred = model.predict(city_df)[0]
            
            # Add an amber marker for each world city
            folium.CircleMarker(
                location=[city['lat'], city['lon']],
                radius=6,
                color="orange",
                fill=True,
                fill_color="orange",
                fill_opacity=0.7,
                tooltip=f"<b>{city['name']}</b><br>AI Valuation: ${city_pred:,.0f}"
            ).add_to(m)
            
        # Add the giant Red Pin for the User's Target Location
        folium.Marker(
            [st.session_state.lat, st.session_state.lon],
            popup="🎯 Target Property",
            tooltip="🎯 Target Location (Click map to move)",
            icon=folium.Icon(color="red", icon="home")
        ).add_to(m)
        
        # Render the map in Streamlit and listen for clicks
        map_data = st_folium(m, height=450, use_container_width=True)
        
        # If the user clicks somewhere on the map, update session state and rerun!
        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            
            # Update only if coordinates actually changed to prevent infinite loops
            if click_lat != st.session_state.lat or click_lon != st.session_state.lon:
                st.session_state.lat = click_lat
                st.session_state.lon = click_lon
                st.rerun() # Forces the app to instantly refresh with new coordinates
        
    with col_details:
        st.subheader("📋 Selected Features")
        st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)
        
        st.subheader("🌍 Global Valuations")
        st.markdown("*Scroll to view AI predictions for all cities...*")
        
        # This creates a literal UI Scrollbar area for the data!
        with st.container(height=210):
            for city in world_cities:
                # Temporarily swap the lat/lon in the user's current inputs
                city_df = input_df.copy()
                city_df['Latitude'] = city['lat']
                city_df['Longitude'] = city['lon']
                city_pred = model.predict(city_df)[0]
                
                # Display the city and its predicted price
                st.markdown(f"**{city['name']}**: <span style='color:#3B82F6'>${city_pred:,.0f}</span>", unsafe_allow_html=True)
                st.divider()

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