import requests
import pytz
from datetime import datetime, timedelta, time
import csv
import time as pytime

API_KEY = "05f281d0f3a9c2f522191de3a0c4ad17"
LAT = 33.44  # UTDallas coordinates
LON = -94.04
TIMEZONE = pytz.timezone('America/Chicago')

def get_historical_weather(target_time):
    utc_time = target_time.astimezone(pytz.utc)
    url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={LAT}&lon={LON}&dt={int(utc_time.timestamp())}&appid={API_KEY}"
    response = requests.get(url).json()
    return response

def generate_time_intervals():
    # start_date = datetime(2025, 1, 20, tzinfo=TIMEZONE)
    start_date = datetime(2025, 2, 24, tzinfo=TIMEZONE)
    end_date = datetime(2025, 5, 2, 22, 0, tzinfo=TIMEZONE)
    current_date = start_date
    
    while current_date <= end_date:
        current_time = datetime.combine(current_date, time(6, 0), tzinfo=TIMEZONE)
        end_time = datetime.combine(current_date, time(22, 0), tzinfo=TIMEZONE)
        
        while current_time <= end_time:
            if current_time >= start_date:
                yield current_time
            current_time += timedelta(hours=2)
        
        current_date += timedelta(days=1)

def main():
    with open('historical_weather_data2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date', 'time', 'temp', 'humidity', 'clouds', 'visibility', 'wind_speed', 'conditions'])
        
        for target_time in generate_time_intervals():
            data = get_historical_weather(target_time)
            print(f"Processed: {target_time.strftime('%Y-%m-%d %H:%M')} -   {data}")
            
            if 'data' in data:
                current = data['data'][0]
                writer.writerow([
                    target_time.strftime("%Y-%m-%d"),
                    target_time.strftime("%H:%M"),
                    current.get('temp', ''),  # Temperature (Kelvin)
                    current.get('humidity', ''),  # Humidity %
                    current.get('clouds', ''),  # Cloud coverage %
                    current.get('visibility', ''),  # Visibility meters
                    current.get('wind_speed', ''),  # Wind speed (m/s)
                    # Handle nested 'weather' field safely
                    current.get('weather', [{}])[0].get('description', '') if current.get('weather') else ''
                ])


            pytime.sleep(.1)  # Rate limit to 1 request/sec

if __name__ == "__main__":
    main()
# https://openweathermap.org/api/one-call-3