# Climate Check - Development Guide

## Project Summary

A Streamlit-based web application for visualizing global temperature changes over time across different locations worldwide. This tool helps users explore climate change trends using real historical weather data from NOAA.

## Core Features

### 1. Location Selection
- Type-based search for locations
- Interactive map-based selection
- Support for global locations

### 2. Time Range Selection
- Configurable year range (50-100+ years of historical data)
- Month-by-month granularity
- Year-over-year comparisons

### 3. Temperature Visualization
- Average temperature trends (`TAVG`)
- Temperature ranges (`TMIN`, `TMAX`)
- Interactive charts and graphs
- Time series analysis

## Technology Stack

### Frontend
- **Streamlit**: Web app framework for rapid development
- **Plotly/Matplotlib**: Data visualization libraries
- **Folium/Streamlit-Folium**: Interactive map components

### Backend
- **Python 3.x**: Core language
- **Pandas**: Data manipulation and analysis
- **Requests**: API calls to NOAA

### Data Source
- **NOAA Climate Data Online (CDO) API**
  - Primary: GSOM (Global Summary of the Month)
  - Fallback: GHCND (Global Historical Climatology Network - Daily)
  - API Documentation: https://www.ncdc.noaa.gov/cdo-web/webservices/v2

## Data Points

### Temperature Metrics
- `TAVG`: Mean temperature
- `TMIN`: Minimum temperature
- `TMAX`: Maximum temperature

### Temporal Granularity
- Monthly summaries (GSOM)
- Daily data aggregation (GHCND fallback)

## Architecture

### Application Flow
1. User selects location (map or search)
2. User selects date range
3. App queries NOAA API for station data
4. Data is fetched and processed
5. Visualizations are generated
6. User can interact with charts and explore trends

### Key Components

#### 1. Location Handler
- Geocoding service integration
- NOAA station lookup by coordinates
- Station metadata management

#### 2. Data Fetcher
- NOAA API integration
- Authentication handling
- Rate limiting compliance
- Data caching for performance

#### 3. Data Processor
- Temperature data aggregation
- Missing data handling
- Unit conversions (if needed)
- Statistical calculations

#### 4. Visualization Engine
- Time series plots
- Temperature anomaly charts
- Comparison visualizations
- Interactive controls

## Implementation Plan

### Phase 1: Setup
- [ ] Initialize Streamlit app structure
- [ ] Set up NOAA API credentials
- [ ] Configure environment and dependencies
- [ ] Create basic page layout

### Phase 2: Data Integration
- [ ] Implement NOAA API client
- [ ] Build location search functionality
- [ ] Create station data fetcher
- [ ] Implement data caching

### Phase 3: Visualization
- [ ] Build time range selector
- [ ] Create temperature trend charts
- [ ] Implement interactive features
- [ ] Add map-based location picker

### Phase 4: Enhancement
- [ ] Add error handling and validation
- [ ] Implement loading states
- [ ] Optimize performance
- [ ] Add data export features

## Development Notes

### NOAA API Considerations
- Requires API token (obtain from NOAA CDO)
- Rate limits apply (5 requests/second, 10,000/day)
- Data availability varies by location
- Some stations have incomplete data

### User Experience
- Clear loading indicators for API calls
- Graceful handling of missing data
- Helpful error messages
- Responsive design for mobile devices

### Data Quality
- Handle missing months/years
- Identify and mark data gaps
- Provide data source attribution
- Show confidence intervals where appropriate
