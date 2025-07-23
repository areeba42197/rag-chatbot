# ----- In chatbot/locationTool.py -----
import requests

def get_location() -> str:
    """Return the user's current city, region, and country based on public IP."""
    try:
        response = requests.get('https://ipinfo.io/json')
        response.raise_for_status()
        data = response.json()
        city = data.get('city', '')
        region = data.get('region', '')
        country = data.get('country', '')
        location_str = ", ".join(x for x in [city, region, country] if x)
        if location_str.strip():
            return f"📍 Your current location is: {location_str}."
        else:
            return "❌ Unable to retrieve location information."
    except Exception as e:
        return f"❌ Failed to get location: {e}"
