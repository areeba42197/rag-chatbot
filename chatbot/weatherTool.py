from dotenv import load_dotenv
import os
import requests

load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather(location: str) -> str:
    """Returns the current weather in the specified city/location."""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return f"❌ Error: {data.get('message', 'Invalid city name')}"
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"🌤️ Weather in {location.title()}: {weather}, {temp}°C."
    except Exception as e:
        return f"❌ Could not retrieve weather: {e}"
