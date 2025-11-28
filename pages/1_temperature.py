import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime, date
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ncei import NCEIClient

st.set_page_config(
    page_title="Temperature Data - Climate Check",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Temperature Data")
st.markdown("Explore temperature trends by location and time period")

# Initialize session state
default_lat = 40.7128  # New York City
default_lon = -74.0060

if 'map_center' not in st.session_state:
    st.session_state.map_center = [default_lat, default_lon]
if 'selected_location' not in st.session_state:
    st.session_state.selected_location = [default_lat, default_lon]

current_year = datetime.now().year
current_date = date.today()
default_start_date = date(current_year - 50, 1, 1)
default_end_date = current_date

if 'start_date' not in st.session_state or 'end_date' not in st.session_state:
    st.session_state.start_date = default_start_date
    st.session_state.end_date = default_end_date

st.divider()

# Location Selection
st.subheader("📍 Select Location")

# Create map with folium
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=6,
    tiles='OpenStreetMap'
)

# Add marker for selected location
folium.Marker(
    location=st.session_state.selected_location,
    popup=f"Selected Location\nLat: {st.session_state.selected_location[0]:.4f}, Lon: {st.session_state.selected_location[1]:.4f}",
    tooltip="Click map to select location",
    icon=folium.Icon(color='red', icon='info-sign')
).add_to(m)

# Render map and capture click events
map_data = st_folium(
    m,
    width=None,
    height=450,
    returned_objects=["last_clicked"]
)

# Update selected location if map was clicked
if map_data["last_clicked"] is not None:
    st.session_state.selected_location = [
        map_data["last_clicked"]["lat"],
        map_data["last_clicked"]["lng"]
    ]
    st.session_state.map_center = st.session_state.selected_location
    st.rerun()

# Display current selection
col1, col2 = st.columns(2)
with col1:
    st.info(f"**Latitude:** {st.session_state.selected_location[0]:.4f}°")
with col2:
    st.info(f"**Longitude:** {st.session_state.selected_location[1]:.4f}°")

# Manual coordinates in expander
with st.expander("🔢 Enter Coordinates Manually"):
    col_lat, col_lon = st.columns(2)

    with col_lat:
        manual_lat = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(st.session_state.selected_location[0]),
            step=0.1,
            format="%.4f"
        )

    with col_lon:
        manual_lon = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(st.session_state.selected_location[1]),
            step=0.1,
            format="%.4f"
        )

    if st.button("📌 Update Location", type="primary"):
        st.session_state.selected_location = [manual_lat, manual_lon]
        st.session_state.map_center = [manual_lat, manual_lon]
        st.rerun()

st.divider()

# Dataset and Time Period Selection (side by side)
col_dataset, col_time = st.columns(2)

with col_dataset:
    st.subheader("📊 Select Dataset")
    dataset_choice = st.radio(
        "Choose data granularity",
        options=['GSOM', 'GHCND'],
        format_func=lambda x: 'Monthly Data' if x == 'GSOM' else 'Daily Data',
        horizontal=True,
        help="GSOM provides monthly summaries. GHCND provides daily measurements with more detail."
    )

    if dataset_choice == 'GHCND':
        st.info("ℹ️ **Daily data**: More detailed, but may take longer to load. ~1700+")
    else:
        st.info("ℹ️ **Monthly data**: Faster loading, good for long-term trends. ~1763+")

with col_time:
    st.subheader("📅 Select Time Period")

    # Adjust min year based on dataset
    min_year = 1763 if dataset_choice == 'GSOM' else 1700
    min_date = date(min_year, 1, 1)

    col_dates = st.columns(2)

    with col_dates[0]:
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.start_date,
            min_value=min_date,
            max_value=current_date,
            help=f"Select start date. Data available from {min_year}."
        )

    with col_dates[1]:
        end_date = st.date_input(
            "End Date",
            value=st.session_state.end_date,
            min_value=min_date,
            max_value=current_date,
            help="Select end date."
        )

    st.session_state.start_date = start_date
    st.session_state.end_date = end_date

    # Extract years for API calls
    start_year = start_date.year
    end_year = end_date.year

    # Calculate time difference in years, months, days
    delta = end_date - start_date
    total_days = delta.days

    # Calculate years, months, days
    years = total_days // 365
    remaining_days = total_days % 365
    months = remaining_days // 30
    days = remaining_days % 30

    # Build duration string
    duration_parts = []
    if years > 0:
        duration_parts.append(f"{years} year{'s' if years != 1 else ''}")
    if months > 0:
        duration_parts.append(f"{months} month{'s' if months != 1 else ''}")
    if days > 0 or len(duration_parts) == 0:
        duration_parts.append(f"{days} day{'s' if days != 1 else ''}")

    duration_str = ", ".join(duration_parts)

    # Display date range info
    st.info(f"📅 **Date range:** {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')} ({duration_str})")

st.divider()

# Fetch button
col_center = st.columns([1, 2, 1])
with col_center[1]:
    go_button = st.button(
        "🚀 Fetch Temperature Data",
        type="primary",
        width="stretch"
    )

# Helper functions
@st.cache_data(ttl=3600)
def find_nearest_station(lat, lon, start_date, end_date, dataset='GSOM'):
    """Find the nearest weather station with temperature data"""
    try:
        client = NCEIClient()

        # Search within a bounding box around the location
        # Approximately 1 degree latitude/longitude (~111km)
        extent_size = 1.0
        extent = f"{lat-extent_size},{lon-extent_size},{lat+extent_size},{lon+extent_size}"

        stations = client.get_stations(
            datasetid=dataset,
            extent=extent,
            startdate=start_date.strftime('%Y-%m-%d'),
            enddate=end_date.strftime('%Y-%m-%d'),
            datatypeid='TMIN',  # Must have temperature data
            limit=100
        )

        if 'results' not in stations or not stations['results']:
            return None, "No stations found in this area with temperature data"

        # Find closest station by calculating distance
        import math

        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Earth's radius in kilometers
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
                math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            return R * c

        # Calculate distance for each station and sort
        for station in stations['results']:
            station['distance'] = haversine_distance(
                lat, lon,
                station.get('latitude', 0),
                station.get('longitude', 0)
            )

        # Sort by data coverage (quality) and distance
        stations_sorted = sorted(
            stations['results'],
            key=lambda x: (-x.get('datacoverage', 0), x['distance'])
        )

        return stations_sorted[0], None

    except Exception as e:
        return None, f"Error finding station: {str(e)}"


@st.cache_data(ttl=3600)
def fetch_temperature_data(station_id, start_date, end_date, dataset='GSOM'):
    """Fetch temperature data for a station"""
    try:
        client = NCEIClient()

        # Fetch each temperature datatype separately to avoid pagination issues
        # GSOM allows max 10 year ranges, GHCND allows max 1 year range
        all_data = []

        # Temperature data types we want
        temp_datatypes = ['TMIN', 'TMAX']

        # Set batch size based on dataset
        batch_size = 10 if dataset == 'GSOM' else 1

        start_year = start_date.year
        end_year = end_date.year

        for datatype in temp_datatypes:
            for year_start in range(start_year, end_year + 1, batch_size):
                year_end = min(year_start + batch_size - 1, end_year)

                # Use actual dates for first and last batch
                batch_start = start_date.strftime('%Y-%m-%d') if year_start == start_year else f'{year_start}-01-01'
                batch_end = end_date.strftime('%Y-%m-%d') if year_end == end_year else f'{year_end}-12-31'

                try:
                    # Fetch data for this batch and datatype
                    data = client.get_all_pages(
                        'get_data',
                        datasetid=dataset,
                        stationid=station_id,
                        datatypeid=datatype,
                        startdate=batch_start,
                        enddate=batch_end,
                        max_results=10000
                    )

                    all_data.extend(data)
                except Exception as e:
                    # Some stations might not have all datatypes, continue anyway
                    continue

        if not all_data:
            return None, "No temperature data found for this station"

        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Convert temperature values to numeric
        df['value'] = pd.to_numeric(df['value'])

        # Pivot to get TMIN, TMAX as separate columns
        df_pivot = df.pivot_table(
            index=['year', 'month', 'date'],
            columns='datatype',
            values='value',
            aggfunc='first'
        ).reset_index()

        return df_pivot, None

    except Exception as e:
        return None, f"Error fetching data: {str(e)}"


def plot_temperature_data(df, station_info, dataset='GSOM'):
    """Create temperature visualization with monthly or daily data points"""
    # Use data directly (no aggregation)
    data = df.copy()

    # Determine granularity for labels
    granularity = "Daily" if dataset == 'GHCND' else "Monthly"

    # Create figure with plotly
    fig = go.Figure()

    # Determine hover template based on granularity
    date_format = '%d %b %Y' if dataset == 'GHCND' else '%b %Y'

    # Add TMAX and TMIN as separate lines
    if 'TMAX' in data.columns and 'TMIN' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['TMAX'],
            mode='lines',
            name='Maximum Temperature',
            line=dict(color='rgba(255, 87, 51, 0.4)', width=1.5),
            showlegend=True,
            hovertemplate=f'%{{x|{date_format}}}<br>Max: %{{y:.1f}}°C<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['TMIN'],
            mode='lines',
            name='Minimum Temperature',
            line=dict(color='rgba(0, 150, 136, 0.4)', width=1.5),
            showlegend=True,
            hovertemplate=f'%{{x|{date_format}}}<br>Min: %{{y:.1f}}°C<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title=f"{granularity} Temperature Data - {station_info['name']}",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode='x unified',
        height=600,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Format x-axis to show years nicely
    num_years = len(data['year'].unique())
    if num_years <= 5:
        # For short periods, show months
        fig.update_xaxes(tickformat='%b %Y', dtick="M3")
    else:
        # For longer periods, show years
        fig.update_xaxes(tickformat='%Y', dtick="M12")

    return fig


# Data visualization section
if go_button:
    st.divider()
    st.subheader("📈 Temperature Data")

    # Step 1: Find nearest station
    with st.spinner("🔍 Finding nearest weather station..."):
        station, error = find_nearest_station(
            st.session_state.selected_location[0],
            st.session_state.selected_location[1],
            start_date,
            end_date,
            dataset_choice
        )

        if error:
            st.error(f"❌ {error}")
            st.info("💡 Try selecting a different location or expanding the date range")
            st.stop()

        # Display station info
        st.success(f"✅ Found station: **{station['name']}**")

        col_info = st.columns(4)
        with col_info[0]:
            st.metric("Station ID", station['id'].split(':')[1][:10])
        with col_info[1]:
            st.metric("Distance", f"{station['distance']:.1f} km")
        with col_info[2]:
            st.metric("Elevation", f"{station.get('elevation', 0):.0f} m")
        with col_info[3]:
            st.metric("Data Coverage", f"{station.get('datacoverage', 0)*100:.0f}%")

    # Step 2: Fetch temperature data
    granularity_text = "daily" if dataset_choice == 'GHCND' else "monthly"
    with st.spinner(f"📊 Fetching {granularity_text} temperature data (TMIN, TMAX)..."):
        df, error = fetch_temperature_data(station['id'], start_date, end_date, dataset_choice)

        if error:
            st.error(f"❌ {error}")
            st.stop()

        # Show what datatypes we got
        available_types = [col for col in ['TMIN', 'TMAX'] if col in df.columns]
        years_with_data = sorted(df['year'].unique())

        data_points_label = "days" if dataset_choice == 'GHCND' else "months"
        st.success(f"✅ Retrieved {len(df)} {data_points_label} of data")
        st.info(f"📊 Available data types: **{', '.join(available_types)}** | Years: **{years_with_data[0]}-{years_with_data[-1]}**")

    # Step 3: Plot the data
    import numpy as np

    fig = plot_temperature_data(df, station, dataset_choice)
    st.plotly_chart(fig, width="stretch")

    # Additional statistics
    st.divider()
    st.subheader("📊 Summary Statistics")

    col_stats = st.columns(3)

    if 'TMAX' in df.columns and 'TMIN' in df.columns:
        # Find warmest and coldest periods
        date_label = "Day" if dataset_choice == 'GHCND' else "Month"
        warmest = df.loc[df['TMAX'].idxmax()]
        coldest = df.loc[df['TMIN'].idxmin()]

        with col_stats[0]:
            date_str = warmest['date'].strftime('%b %Y') if dataset_choice == 'GSOM' else warmest['date'].strftime('%d %b %Y')
            st.metric(
                f"Highest Temperature",
                f"{date_str} ({warmest['TMAX']:.1f}°C)"
            )
        with col_stats[1]:
            date_str = coldest['date'].strftime('%b %Y') if dataset_choice == 'GSOM' else coldest['date'].strftime('%d %b %Y')
            st.metric(
                f"Lowest Temperature",
                f"{date_str} ({coldest['TMIN']:.1f}°C)"
            )
        with col_stats[2]:
            # Calculate overall average from TMIN and TMAX
            avg_temp = (df['TMIN'].mean() + df['TMAX'].mean()) / 2
            data_points_label = "days" if dataset_choice == 'GHCND' else "months"
            st.metric(
                f"Overall Average",
                f"{avg_temp:.1f}°C ({len(df)} {data_points_label})",
            )

    # Show raw data option
    with st.expander("📋 View Raw Data"):
        st.dataframe(df, width="stretch", height=300)
