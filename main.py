import streamlit as st

st.set_page_config(
    page_title="Climate Check",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌍 Climate Check")
st.markdown("### Explore global climate trends over time")

st.markdown("---")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📊 Features

    - **Interactive Visualizations**: Explore climate trends across different locations
    - **Historical Data**: Access decades of climate data from weather stations worldwide
    - **Multiple Climate Metrics**: Analyze temperature, precipitation, and other climate indicators
    - **Customizable Analysis**: Select specific year ranges and locations

    ### 🗺️ How to Use

    1. Navigate to a climate metric page in the sidebar (e.g., Temperature)
    2. Select a location using the interactive map
    3. Customize config
    4. Click **Fetch Data** to view historical trends
    """)

with col2:
    st.markdown("""
    ### 📈 About

    This app helps you visualize how climate patterns have changed over time 
    across different locations worldwide using data from different sources.

    Explore various climate metrics including temperature, precipitation, snowfall,
    and other weather indicators to understand long-term climate trends.

    ### 🔗 Data Sources

    - **[NOAA](https://www.ncdc.noaa.gov/cdo-web/)** 

    ### 🚀 Get Started

    Use the sidebar to navigate between different climate metrics and start exploring!
    """)

st.markdown("---")
st.markdown("*Select a page from the sidebar to get started.*")
