import requests
import csv
from datetime import datetime, timedelta, time
import pytz  # Install with: pip install pytz
import time as pytime  # Import time for sleep function
# Configuration
API_KEY = "AIzaSyBdkiP9IO7GiHaxQJxgKmMRArPu6k8Gs5k"
ORIGIN = "place_id:ChIJe8bJ-E8ZTIYR_uU4YdzLq6w"  # Paxel Financial Consulting, 450 Independence Pkwy Suite 100, Richardson, TX
DESTINATION = "place_id:ChIJZ0aPfAAhTIYROHWBjj3YtnA"  # UTD Parking Strucutre 4, 2520 Drive H, Richardson, TX
TIMEZONE = "America/Chicago"  # Richardson, TX timezone (CDT)
START_DATE = datetime(2026, 1, 19, tzinfo=pytz.timezone(TIMEZONE))  # Adjust year if needed
END_DATE = datetime(2026, 5, 1, 23, 59, tzinfo=pytz.timezone(TIMEZONE))
OUTPUT_FILE = "historical_traffic_data.csv"

def get_historical_traffic(departure_time):
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins": ORIGIN,
        "destinations": DESTINATION,
        "departure_time": int(departure_time.timestamp()),
        "key": API_KEY,
        "mode": "driving",
        "traffic_model": "best_guess"  # Uses historical data
    }
    response = requests.get(url, params=params).json()
    print(response)
    
    if response["status"] == "OK":
        element = response["rows"][0]["elements"][0]
        if element["status"] == "OK":
            return {
                'date': departure_time.strftime("%Y-%m-%d"),
                'time': departure_time.strftime("%H:%M"),
                'duration_sec': element['duration']['value'],
                'traffic_duration_sec': element['duration_in_traffic']['value'],
            }
    return None

def generate_time_intervals():
    current_date = START_DATE.date()
    end_date = END_DATE.date()
    interval = timedelta(minutes=15)
    start_time = time(6, 0)   # 6:00 AM
    end_time = time(22, 0)    # 10:00 PM

    while current_date <= end_date:
        current_dt = datetime.combine(current_date, start_time, tzinfo=START_DATE.tzinfo)
        end_dt = datetime.combine(current_date, end_time, tzinfo=START_DATE.tzinfo)
        while current_dt <= end_dt and current_dt <= END_DATE:
            if current_dt >= START_DATE:
                yield current_dt
            current_dt += interval
        current_date += timedelta(days=1)

def main():
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "time", "duration_sec", "traffic_duration_sec"])
        writer.writeheader()

        for departure_time in generate_time_intervals():
            data = get_historical_traffic(departure_time)
            if data:
                writer.writerow(data)
                print(f"Processed: {data['date']} {data['time']} sec")
            pytime.sleep(.01)  # Avoid hitting rate limits (adjust as needed)
# duration = typical travel time (no live traffic)
# duration_in_traffic = travel time with current/predicted traffic for your departure time

if __name__ == "__main__":
    main()
