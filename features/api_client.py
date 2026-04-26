import requests
import os
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
CITY = "lahore"

def fetch_lahore_aqi():
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    
    response = requests.get(url)
    data = response.json()
    
    if data["status"] != "ok":
        raise Exception(f"API error: {data}")
    
    aqi_data = data["data"]
    
    result = {
        "timestamp": aqi_data["time"]["s"],
        "aqi":       aqi_data["aqi"],
        "pm25":      aqi_data["iaqi"].get("pm25", {}).get("v", None),
        "pm10":      aqi_data["iaqi"].get("pm10", {}).get("v", None),
        "no2":       aqi_data["iaqi"].get("no2",  {}).get("v", None),
        "co":        aqi_data["iaqi"].get("co",   {}).get("v", None),
        "humidity":  aqi_data["iaqi"].get("h",    {}).get("v", None),
        "temp":      aqi_data["iaqi"].get("t",    {}).get("v", None),
        "wind":      aqi_data["iaqi"].get("w",    {}).get("v", None),
    }
    
    return result

if __name__ == "__main__":
    data = fetch_lahore_aqi()
    print(data)
    print("✅ Live Lahore AQI Data:")
    for key, value in data.items():
        print(f"  {key}: {value}")