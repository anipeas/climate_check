import streamlit as st

st.set_page_config(
    page_title="Climate Check",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌍 Climate Check")
st.markdown("### Explore global temperature trends over time")

st.markdown("---")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📊 Features
    
    - **Interactive Visualizations**: Explore temperature trends across different locations
    - **Historical Data**: Access decades of climate data
    - **Customizable Analysis**: Select specific year ranges and locations
    
    ### 🗺️ How to Use

    1. Navigate to **📊 1 historical** in the sidebar to explore temperature data
    2. Select a location using the interactive map
    3. Choose your desired year range (1900-present)
    4. Click **Fetch Temperature Data** to view trends
    """)

with col2:
    st.markdown("""
    ### 📈 About
    
    This app helps you visualize how **global temperatures have changed over time** 
    using data from the **NOAA Climate Data Online (CDO) API**.
    
    ### 🔗 Data Source
    
    - **GSOM** (Global Summary of the Month) dataset
    - **GHCND** (Global Historical Climatology Network - Daily) as fallback
    
    ### 🚀 Get Started
    
    Use the sidebar to navigate between different pages and explore the features!
    """)

st.markdown("---")
st.markdown("*Select a page from the sidebar to get started.*")
