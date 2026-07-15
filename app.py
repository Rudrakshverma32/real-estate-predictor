import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ==========================================
# 1. Page Configuration & Layout
# ==========================================
st.set_page_config(
    page_title="Real Estate Analytics Dashboard",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to inject to make the dashboard look cleaner
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Data Loading & Caching
# ==========================================
# We use @st.cache_data so the data is only downloaded/processed once per session
@st.cache_data
def load_data():
    cal_housing = fetch_california_housing()
    df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    # The target is the median house value in units of 100,000
    df['Price'] = cal_housing.target * 100000 
    return df, cal_housing.feature_names

# ==========================================
# 3. Model Training & Caching
# ==========================================
# We use @st.cache_resource for ML models so it only trains once and persists in memory
@st.cache_resource
def train_model(df, features):
    X = df[features]
    y = df['Price']
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model
    # Using 50 estimators for quick compilation in a web app context while maintaining accuracy
    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate performance metrics
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return model, mae, r2

# Load data and train model
with st.spinner('Loading data and training machine learning model...'):
    df, feature_names = load_data()
    model, mae, r2 = train_model(df, feature_names)

# ==========================================
# 4. User Interface - Sidebar (Inputs)
# ==========================================
st.sidebar.header("🔧 Property Specifications")
st.sidebar.markdown("Adjust the sliders below to see how property features impact the predicted value.")

# Create dynamic sliders based on dataset min/max values
def user_input_features():
    inputs = {}
    
    # Grouping inputs logically for the user
    st.sidebar.subheader("Area Demographics")
    inputs['MedInc'] = st.sidebar.slider(
        'Median Income (x$10,000)', 
        float(df['MedInc'].min()), float(df['MedInc'].max()), float(df['MedInc'].median()), 0.5
    )
    inputs['Population'] = st.sidebar.slider(
        'Population in Block', 
        float(df['Population'].min()), float(df['Population'].quantile(0.95)), float(df['Population'].median()), 100.0
    )
    
    st.sidebar.subheader("House Characteristics")
    inputs['HouseAge'] = st.sidebar.slider(
        'House Age (Years)', 
        float(df['HouseAge'].min()), float(df['HouseAge'].max()), float(df['HouseAge'].median()), 1.0
    )
    inputs['AveRooms'] = st.sidebar.slider(
        'Average Rooms', 
        float(df['AveRooms'].min()), 15.0, float(df['AveRooms'].median()), 0.5
    )
    inputs['AveBedrms'] = st.sidebar.slider(
        'Average Bedrooms', 
        float(df['AveBedrms'].min()), 5.0, float(df['AveBedrms'].median()), 0.1
    )
    inputs['AveOccup'] = st.sidebar.slider(
        'Average Occupants', 
        float(df['AveOccup'].min()), 10.0, float(df['AveOccup'].median()), 0.5
    )
    
    st.sidebar.subheader("Location Coordinates")
    inputs['Latitude'] = st.sidebar.slider(
        'Latitude', 
        float(df['Latitude'].min()), float(df['Latitude'].max()), float(df['Latitude'].median()), 0.1
    )
    inputs['Longitude'] = st.sidebar.slider(
        'Longitude', 
        float(df['Longitude'].min()), float(df['Longitude'].max()), float(df['Longitude'].median()), 0.1
    )
    
    return pd.DataFrame(inputs, index=[0])

input_df = user_input_features()

# ==========================================
# 5. User Interface - Main Canvas
# ==========================================
st.title("🏡 Predictive Real Estate Analytics")
st.markdown("""
This end-to-end machine learning application predicts real estate valuations based on demographic and structural features. 
Adjust the parameters in the sidebar to generate a **real-time prediction**.
""")

st.divider()

# --- Prediction Section ---
st.subheader("💡 Real-Time Valuation Prediction")

# Make prediction based on user input
prediction = model.predict(input_df)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Estimated Value")
    # Displaying the prediction as a massive, visually appealing metric
    st.metric(label="Predicted House Price", value=f"${prediction[0]:,.2f}")

with col2:
    st.markdown("**Current Input Parameters:**")
    st.dataframe(input_df, hide_index=True)

st.divider()

# --- Analytics & Model Insight Section ---
st.subheader("📊 Model Performance & Insights")
st.markdown("Understanding how the `RandomForestRegressor` makes decisions.")

col3, col4 = st.columns(2)

with col3:
    # Feature Importance Plot
    st.markdown("**Top Drivers of Property Value**")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(importances)), importances[indices], align="center", color="#4CAF50")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(np.array(feature_names)[indices], rotation=45, ha='right')
    ax.set_ylabel("Relative Importance")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

with col4:
    # Model Metrics
    st.markdown("**Test Set Accuracy Metrics**")
    st.info(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
    st.markdown("*This means our model's predictions are, on average, off by this amount.*")
    
    st.success(f"**R² Score:** {r2:.3f}")
    st.markdown(f"*This means the model explains **{r2*100:.1f}%** of the variance in housing prices, indicating a strong predictive fit.*")

    st.markdown("""
    **Tech Stack Used:**
    * **Frontend:** Streamlit
    * **Machine Learning:** Scikit-Learn (Random Forest)
    * **Data Processing:** Pandas & NumPy
    """)