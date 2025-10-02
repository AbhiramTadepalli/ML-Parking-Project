import json
from datetime import datetime
import csv
import pandas as pd

def write_format_parking_data(data, writer):
    # Extract and parse timestamp
    timestamp_str = data['UTDParkingPartitionKey']
    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))  # Handle UTC time

    for garage in data['garages']:
        print(garage['id'])
        orange_row = {
            'date': dt.strftime("%Y-%m-%d"),
            'time': dt.strftime("%H:%M"),
            'day_of_week': dt.strftime("%A"),
            'garage': garage['id'],
            'color': 'Orange Permit',
            'spots_open': 0,
        }
        gold_row = {
            'date': dt.strftime("%Y-%m-%d"),
            'time': dt.strftime("%H:%M"),
            'day_of_week': dt.strftime("%A"),
            'garage': garage['id'],
            'color': 'Gold Permit',
            'spots_open': 0,
        }
        pay_by_row = {
            'date': dt.strftime("%Y-%m-%d"),
            'time': dt.strftime("%H:%M"),
            'day_of_week': dt.strftime("%A"),
            'garage': garage['id'],
            'color': 'Pay-By-Space Permit',
            'spots_open': 0,
        }
        purple_row = {
            'date': dt.strftime("%Y-%m-%d"),
            'time': dt.strftime("%H:%M"),
            'day_of_week': dt.strftime("%A"),
            'garage': garage['id'],
            'color': 'Purple Permit',
            'spots_open': 0,
        }
        

        for entry in garage['entries']:
            print(entry['level'], entry['permit'])
            if 'Orange' in entry['permit']:
                orange_row['spots_open'] += int(entry['spots_open'])
            elif 'Gold' in entry['permit']:
                gold_row['spots_open'] += int(entry['spots_open'])
            elif 'Purple' in entry['permit']:
                purple_row['spots_open'] += int(entry['spots_open'])
            elif 'Pay-By-Space' in entry['permit']:
                pay_by_row['spots_open'] += int(entry['spots_open'])
        writer.writerow(orange_row)
        writer.writerow(gold_row)
        writer.writerow(purple_row)
        writer.writerow(pay_by_row)

def merge_weather(weather, parking):
    # Convert to datetime
    weather['datetime'] = pd.to_datetime(weather['date'] + ' ' + weather['time'])
    parking['datetime'] = pd.to_datetime(parking['date'] + ' ' + parking['time'])
    # Drop original date/time columns from weather
    weather = weather.drop(columns=['date', 'time'])
    # Align weather to parking timestamps
    weather.set_index('datetime', inplace=True)
    weather_aligned = weather.reindex(parking['datetime'], method='ffill')
    # Combine data (align by index)
    merged_df = pd.concat([parking.set_index('datetime'), weather_aligned], axis=1).reset_index()
    # Clean up column names
    merged_df = merged_df.rename(columns={'index': 'datetime'})
    merged_df = merged_df.drop(columns=['datetime'])
    merged_df.to_csv('merged_dataset.csv', index=False)
    print(merged_df.iloc[0])
    # Save to CSV
    merged_df.to_csv('merged_dataset.csv', index=False)

def merge_traffic(traffic, parking):
    # Convert to datetime
    traffic['datetime'] = pd.to_datetime(traffic['date'] + ' ' + traffic['time'])
    parking['datetime'] = pd.to_datetime(parking['date'] + ' ' + parking['time'])
    # Align traffic to parking dates: shift back 1 year + add 1 day
    traffic['datetime'] = traffic['datetime'] - pd.DateOffset(years=1) + pd.DateOffset(days=1)
    
    # Prepare traffic data
    traffic = (traffic.drop(columns=['date', 'time'])
              .set_index('datetime')
              .sort_index()
              .resample('1T').interpolate(method='time'))  # Resample to 1-minute intervals
    print(traffic)
    
    # Merge using merge_asof for temporal alignment
    merged_df = pd.merge_asof(
        parking,
        traffic.reset_index(),
        on='datetime',
        direction='nearest'
    )
    
    # Clean up column names
    merged_df = merged_df.rename(columns={'duration_sec': 'drive_duration_sec'})
    merged_df = merged_df.drop(columns=['datetime'])
    print(merged_df.iloc[0])
    # Save to CSV
    merged_df.to_csv('merged_traffic_parking.csv', index=False)

def filtered_date_range(df):
    df['date'] = pd.to_datetime(df['date'])
    start_date = '2025-01-22'
    end_date = '2025-05-02'
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    filtered_df.to_csv('filtered_dataset.csv', index=False)

def main():
    # Load the JSON file
    with open('export.json', 'r') as f:
        data = json.load(f)

    sorted_data = sorted(
        data,
        key=lambda x: datetime.fromisoformat(x['UTDParkingPartitionKey'].replace('Z', '+00:00'))
    )

    # for i in range(9800, 10150):
    if False:
        print(len(data))
        print(sorted_data[1000])
        with open("formatted_parking.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "time", "day_of_week", "garage", "color", "spots_open"])
            writer.writeheader()
            for datum in sorted_data:
                write_format_parking_data(datum, writer)
    if False:
        merge_traffic(pd.read_csv('historical_traffic_data.csv'), pd.read_csv('formatted_parking.csv'))
    if False:
        merge_weather(pd.read_csv('historical_weather_data.csv'), pd.read_csv('merged_traffic_parking.csv'))    
    if False:
        filtered_date_range(pd.read_csv('merged_dataset.csv'))
    if True:
        # get the biggest spots_open
        df = pd.read_csv('formatted_parking.csv')
        max_spots = df.groupby(['color', 'garage'])['spots_open'].max()
        print(max_spots)


if __name__ == "__main__":
    main()