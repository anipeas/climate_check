# 🌍 climate_check

A simple **Streamlit-based web app** that visualizes how **global temperatures have changed over time** across different locations.

## 💡 Overview

The goal is to build an interactive data visualization tool where users can:
- Select a **location** (either by typing or clicking on a map).
- Choose a **year range** (e.g. past 50–100 years).
- See how **average temperatures** have changed **month-by-month** and **year-by-year** over that period.

The project helps users intuitively explore climate change trends using **real historical weather data**.

---

## 🧩 Data Source

We use the **NOAA Climate Data Online (CDO) API**, specifically:
- **GSOM (Global Summary of the Month)** dataset for monthly temperature summaries.
- Fallback to **GHCND (Global Historical Climatology Network - Daily)** if GSOM data is unavailable.

The app fetches:
- Mean temperature (`TAVG`)
- Minimum/maximum temperatures (`TMIN`, `TMAX`) when needed

NOAA API docs: https://www.ncdc.noaa.gov/cdo-web/webservices/v2

## Project Setup



---

