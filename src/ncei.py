"""
NCEI (NOAA Climate Data Online) API Client

API Documentation: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
Rate Limits: 5 requests/second, 10,000 requests/day
"""

import os
import time
from datetime import datetime
from typing import Optional, Dict, List, Any
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class NCEIClient:
    """Client for NOAA's Climate Data Online (CDO) API v2"""

    BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
    MAX_LIMIT = 1000
    DEFAULT_LIMIT = 25

    def __init__(self, token: Optional[str] = None):
        """
        Initialize NCEI API client

        Args:
            token: NCEI API token. If not provided, will attempt to read from
                   NCEI_TOKEN environment variable
        """
        self.token = token or os.getenv("NCEI_TOKEN")
        if not self.token:
            raise ValueError(
                "NCEI API token required. Provide via constructor or NCEI_TOKEN env var. "
                "Request token at: https://www.ncdc.noaa.gov/cdo-web/token"
            )

        self.session = requests.Session()
        self.session.headers.update({
            "token": self.token,
            "Accept": "application/json"
        })

        # Rate limiting tracking
        self._last_request_time = 0
        self._min_request_interval = 0.2  # 5 requests/second = 0.2s between requests

    def _rate_limit(self):
        """Enforce rate limiting of 5 requests per second"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a request to the NCEI API

        Args:
            endpoint: API endpoint (e.g., 'datasets', 'stations')
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            requests.HTTPError: If request fails
        """
        self._rate_limit()

        url = f"{self.BASE_URL}/{endpoint}"

        # Clean None values from params
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        response = self.session.get(url, params=params)
        response.raise_for_status()

        return response.json()

    # ========================================================================
    # Datasets
    # ========================================================================

    def get_datasets(
        self,
        datasetid: Optional[str] = None,
        datatypeid: Optional[str] = None,
        locationid: Optional[str] = None,
        stationid: Optional[str] = None,
        startdate: Optional[str] = None,
        enddate: Optional[str] = None,
        sortfield: Optional[str] = None,
        sortorder: Optional[str] = "asc",
        limit: int = DEFAULT_LIMIT,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get available datasets

        Common datasets:
        - GHCND: Daily Summaries
        - GSOM: Global Summary of the Month
        - GSOY: Global Summary of the Year

        Args:
            datasetid: Specific dataset ID to retrieve
            datatypeid: Filter by data type
            locationid: Filter by location
            stationid: Filter by station
            startdate: Filter by start date (YYYY-MM-DD)
            enddate: Filter by end date (YYYY-MM-DD)
            sortfield: Sort by (id, name, mindate, maxdate, datacoverage)
            sortorder: Sort order (asc/desc)
            limit: Number of results (max 1000)
            offset: Pagination offset

        Returns:
            Dictionary with 'results' list and 'metadata'
        """
        endpoint = f"datasets/{datasetid}" if datasetid else "datasets"

        params = {
            "datatypeid": datatypeid,
            "locationid": locationid,
            "stationid": stationid,
            "startdate": startdate,
            "enddate": enddate,
            "sortfield": sortfield,
            "sortorder": sortorder,
            "limit": min(limit, self.MAX_LIMIT),
            "offset": offset
        }

        return self._make_request(endpoint, params)

    # ========================================================================
    # Data Categories
    # ========================================================================

    def get_datacategories(
        self,
        datacategoryid: Optional[str] = None,
        datasetid: Optional[str] = None,
        locationid: Optional[str] = None,
        stationid: Optional[str] = None,
        startdate: Optional[str] = None,
        enddate: Optional[str] = None,
        sortfield: Optional[str] = None,
        sortorder: Optional[str] = "asc",
        limit: int = DEFAULT_LIMIT,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get data categories (general groupings like TEMP, PRCP, etc.)

        Args:
            datacategoryid: Specific category ID to retrieve
            datasetid: Filter by dataset
            locationid: Filter by location
            stationid: Filter by station
            startdate: Filter by start date (YYYY-MM-DD)
            enddate: Filter by end date (YYYY-MM-DD)
            sortfield: Sort by (id, name)
            sortorder: Sort order (asc/desc)
            limit: Number of results (max 1000)
            offset: Pagination offset

        Returns:
            Dictionary with 'results' list and 'metadata'
        """
        endpoint = f"datacategories/{datacategoryid}" if datacategoryid else "datacategories"

        params = {
            "datasetid": datasetid,
            "locationid": locationid,
            "stationid": stationid,
            "startdate": startdate,
            "enddate": enddate,
            "sortfield": sortfield,
            "sortorder": sortorder,
            "limit": min(limit, self.MAX_LIMIT),
            "offset": offset
        }

        return self._make_request(endpoint, params)

    # ========================================================================
    # Data Types
    # ========================================================================

    def get_datatypes(
        self,
        datatypeid: Optional[str] = None,
        datasetid: Optional[str] = None,
        locationid: Optional[str] = None,
        stationid: Optional[str] = None,
        datacategoryid: Optional[str] = None,
        startdate: Optional[str] = None,
        enddate: Optional[str] = None,
        sortfield: Optional[str] = None,
        sortorder: Optional[str] = "asc",
        limit: int = DEFAULT_LIMIT,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get data types (specific measurements like TAVG, TMIN, TMAX)

        Common temperature data types:
        - TAVG: Average temperature
        - TMIN: Minimum temperature
        - TMAX: Maximum temperature

        Args:
            datatypeid: Specific data type ID to retrieve
            datasetid: Filter by dataset
            locationid: Filter by location
            stationid: Filter by station
            datacategoryid: Filter by data category
            startdate: Filter by start date (YYYY-MM-DD)
            enddate: Filter by end date (YYYY-MM-DD)
            sortfield: Sort by (id, name, mindate, maxdate, datacoverage)
            sortorder: Sort order (asc/desc)
            limit: Number of results (max 1000)
            offset: Pagination offset

        Returns:
            Dictionary with 'results' list and 'metadata'
        """
        endpoint = f"datatypes/{datatypeid}" if datatypeid else "datatypes"

        params = {
            "datasetid": datasetid,
            "locationid": locationid,
            "stationid": stationid,
            "datacategoryid": datacategoryid,
            "startdate": startdate,
            "enddate": enddate,
            "sortfield": sortfield,
            "sortorder": sortorder,
            "limit": min(limit, self.MAX_LIMIT),
            "offset": offset
        }

        return self._make_request(endpoint, params)

    # ========================================================================
    # Location Categories
    # ========================================================================

    def get_locationcategories(
        self,
        locationcategoryid: Optional[str] = None,
        datasetid: Optional[str] = None,
        startdate: Optional[str] = None,
        enddate: Optional[str] = None,
        sortfield: Optional[str] = None,
        sortorder: Optional[str] = "asc",
        limit: int = DEFAULT_LIMIT,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get location categories (CITY, ST, CNTRY, etc.)

        Args:
            locationcategoryid: Specific location category ID to retrieve
            datasetid: Filter by dataset
            startdate: Filter by start date (YYYY-MM-DD)
            enddate: Filter by end date (YYYY-MM-DD)
            sortfield: Sort by (id, name)
            sortorder: Sort order (asc/desc)
            limit: Number of results (max 1000)
            offset: Pagination offset

        Returns:
            Dictionary with 'results' list and 'metadata'
        """
        endpoint = f"locationcategories/{locationcategoryid}" if locationcategoryid else "locationcategories"

        params = {
            "datasetid": datasetid,
            "startdate": startdate,
            "enddate": enddate,
            "sortfield": sortfield,
            "sortorder": sortorder,
            "limit": min(limit, self.MAX_LIMIT),
            "offset": offset
        }

        return self._make_request(endpoint, params)

    # ========================================================================
    # Locations
    # ========================================================================

    def get_locations(
        self,
        locationid: Optional[str] = None,
        datasetid: Optional[str] = None,
        locationcategoryid: Optional[str] = None,
        datacategoryid: Optional[str] = None,
        startdate: Optional[str] = None,
        enddate: Optional[str] = None,
        sortfield: Optional[str] = None,
        sortorder: Optional[str] = "asc",
        limit: int = DEFAULT_LIMIT,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get locations (countries, states, cities, etc.)

        Args:
            locationid: Specific location ID to retrieve
            datasetid: Filter by dataset
            locationcategoryid: Filter by location category
            datacategoryid: Filter by data category
            startdate: Filter by start date (YYYY-MM-DD)
            enddate: Filter by end date (YYYY-MM-DD)
            sortfield: Sort by (id, name, mindate, maxdate, datacoverage)
            sortorder: Sort order (asc/desc)
            limit: Number of results (max 1000)
            offset: Pagination offset

        Returns:
            Dictionary with 'results' list and 'metadata'
        """
        endpoint = f"locations/{locationid}" if locationid else "locations"

        params = {
            "datasetid": datasetid,
            "locationcategoryid": locationcategoryid,
            "datacategoryid": datacategoryid,
            "startdate": startdate,
            "enddate": enddate,
            "sortfield": sortfield,
            "sortorder": sortorder,
            "limit": min(limit, self.MAX_LIMIT),
            "offset": offset
        }

        return self._make_request(endpoint, params)

    # ========================================================================
    # Stations
    # ========================================================================

    def get_stations(
        self,
        stationid: Optional[str] = None,
        datasetid: Optional[str] = None,
        locationid: Optional[str] = None,
        datacategoryid: Optional[str] = None,
        datatypeid: Optional[str] = None,
        extent: Optional[str] = None,
        startdate: Optional[str] = None,
        enddate: Optional[str] = None,
        sortfield: Optional[str] = None,
        sortorder: Optional[str] = "asc",
        limit: int = DEFAULT_LIMIT,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get weather stations

        Args:
            stationid: Specific station ID to retrieve
            datasetid: Filter by dataset
            locationid: Filter by location
            datacategoryid: Filter by data category
            datatypeid: Filter by data type
            extent: Filter by bounding box (lat1,lon1,lat2,lon2)
            startdate: Filter by start date (YYYY-MM-DD)
            enddate: Filter by end date (YYYY-MM-DD)
            sortfield: Sort by (id, name, mindate, maxdate, datacoverage)
            sortorder: Sort order (asc/desc)
            limit: Number of results (max 1000)
            offset: Pagination offset

        Returns:
            Dictionary with 'results' list and 'metadata'
        """
        endpoint = f"stations/{stationid}" if stationid else "stations"

        params = {
            "datasetid": datasetid,
            "locationid": locationid,
            "datacategoryid": datacategoryid,
            "datatypeid": datatypeid,
            "extent": extent,
            "startdate": startdate,
            "enddate": enddate,
            "sortfield": sortfield,
            "sortorder": sortorder,
            "limit": min(limit, self.MAX_LIMIT),
            "offset": offset
        }

        return self._make_request(endpoint, params)

    # ========================================================================
    # Data (Actual Climate Observations)
    # ========================================================================

    def get_data(
        self,
        datasetid: str,
        startdate: str,
        enddate: str,
        datatypeid: Optional[str] = None,
        locationid: Optional[str] = None,
        stationid: Optional[str] = None,
        units: Optional[str] = "metric",
        sortfield: Optional[str] = None,
        sortorder: Optional[str] = "asc",
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        includemetadata: bool = True
    ) -> Dict[str, Any]:
        """
        Get actual climate data observations

        IMPORTANT: Date range limits:
        - Annual/Monthly data (GSOM, GSOY): Max 10 year range
        - All other data (GHCND): Max 1 year range

        Args:
            datasetid: Dataset ID (REQUIRED) - e.g., 'GSOM', 'GHCND'
            startdate: Start date (REQUIRED) - YYYY-MM-DD format
            enddate: End date (REQUIRED) - YYYY-MM-DD format
            datatypeid: Filter by data type (e.g., 'TAVG', 'TMIN', 'TMAX')
            locationid: Filter by location
            stationid: Filter by station
            units: Units for response (standard/metric)
            sortfield: Sort by (datatype, date, station)
            sortorder: Sort order (asc/desc)
            limit: Number of results (max 1000)
            offset: Pagination offset
            includemetadata: Include metadata in response

        Returns:
            Dictionary with 'results' list and 'metadata'

        Raises:
            ValueError: If required parameters are missing
        """
        if not all([datasetid, startdate, enddate]):
            raise ValueError("datasetid, startdate, and enddate are required")

        params = {
            "datasetid": datasetid,
            "startdate": startdate,
            "enddate": enddate,
            "datatypeid": datatypeid,
            "locationid": locationid,
            "stationid": stationid,
            "units": units,
            "sortfield": sortfield,
            "sortorder": sortorder,
            "limit": min(limit, self.MAX_LIMIT),
            "offset": offset,
            "includemetadata": str(includemetadata).lower()
        }

        return self._make_request("data", params)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def get_all_pages(
        self,
        method_name: str,
        max_results: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Fetch all pages of results for any endpoint

        Args:
            method_name: Name of the method to call (e.g., 'get_stations')
            max_results: Maximum number of results to fetch
            **kwargs: Arguments to pass to the method

        Returns:
            List of all result items across all pages

        Example:
            all_stations = client.get_all_pages(
                'get_stations',
                datasetid='GSOM',
                locationid='CITY:US370003',
                max_results=5000
            )
        """
        method = getattr(self, method_name)
        all_results = []
        offset = kwargs.get('offset', 0)
        limit = min(kwargs.get('limit', self.MAX_LIMIT), self.MAX_LIMIT)

        while True:
            kwargs['offset'] = offset
            kwargs['limit'] = limit

            response = method(**kwargs)

            # Handle single item responses
            if 'results' in response:
                results = response['results']
                all_results.extend(results)

                # Check if we've hit our max or reached the end
                if max_results and len(all_results) >= max_results:
                    return all_results[:max_results]

                metadata = response.get('metadata', {})
                result_count = metadata.get('resultset', {}).get('count', 0)

                if offset + limit >= result_count:
                    break

                offset += limit
            else:
                # Single item response
                return [response]

        return all_results


# Example usage
if __name__ == "__main__":
    # Initialize client (requires NCEI_TOKEN environment variable)
    client = NCEIClient()

    # Get available datasets
    datasets = client.get_datasets()
    print("Available datasets:")
    for ds in datasets.get('results', []):
        print(f"  {ds['id']}: {ds['name']}")

    # Get stations near a location
    print("\nSearching for stations...")
    stations = client.get_stations(
        datasetid='GSOM',
        locationid='CITY:US370003',  # New York City
        startdate='1950-01-01',
        enddate='2024-12-31',
        limit=5
    )

    if 'results' in stations:
        print(f"Found {len(stations['results'])} stations")
        for station in stations['results']:
            print(f"  {station['id']}: {station['name']}")
