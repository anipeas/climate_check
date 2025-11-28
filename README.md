# 🌍 Climate Check

A **Streamlit-based web application** for visualizing and analyzing global climate trends across different locations and time periods. 

---

## 🌐 Data Source

This app uses the **NOAA Climate Data Online (CDO) API** to fetch historical weather data:

- **[GSOM](https://www.ncdc.noaa.gov/cdo-web/datasets)** (Global Summary of the Month)
  - Monthly summaries from 1763+
  - Faster loading, ideal for long-term trends
  - 10-year batch processing for efficiency

- **[GHCND](https://www.ncdc.noaa.gov/cdo-web/datasets)** (Global Historical Climatology Network - Daily)
  - Daily measurements from 1700+
  - More detailed granularity
  - 1-year batch processing

**Data Types:**
- Temperature: `TMIN` (minimum), `TMAX` (maximum)
- Precipitation: `PRCP` (total precipitation)

**API Documentation:** https://www.ncdc.noaa.gov/cdo-web/webservices/v2

---

## 🚀 Setup

### Prerequisites
- Python 3.11 or higher
- NOAA CDO API token (free)

### 1. Get a NOAA API Token

1. Visit https://www.ncdc.noaa.gov/cdo-web/token
2. Enter your email address
3. Check your email for the API token
4. Save the token - you'll need it for setup

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/climate_check.git
cd climate_check

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 3. Configuration

Create a `.env` file in the project root:

```bash
NCEI_TOKEN=your_api_token_here
```

### 4. Run the Application

```bash
# Using streamlit
streamlit run main.py

# Or using uv
uv run streamlit run main.py
```

The app will open in your default browser at `http://localhost:8501`

---


## 📊 Example Insights

With Climate Check, you can answer questions like:

- "How much warmer are Julys in New York now compared to 50 years ago?"
- "Has precipitation in Seattle increased over the past century?"
- "What was the coldest January on record for my city?"
- "Are winters getting milder in my region?"
- "How variable is rainfall in monsoon months over time?"

---

## 🐛 Known Limitations

- **API Rate Limits**: NOAA CDO API has limits (5 requests/second, 10,000/day)
- **Data Availability**: Not all stations have complete data for all time periods
- **Geographic Coverage**: Station density varies by region
- **Missing Data**: Some weather stations may only have TMIN or TMAX (not both)
- **Batch Failures**: Long date ranges may have some failed batches (warnings displayed)


---

