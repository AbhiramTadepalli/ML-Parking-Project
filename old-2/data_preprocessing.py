import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv(r"daytime_filtered_dataset.csv")
data = pd.read_csv(r"daytime_filtered_dataset.csv")


X = data.drop(['spots_open'], axis=1)
data = df

ordinal_weather = {
    'clear sky': 0,
    'few clouds': 1,
    'scattered clouds': 2,
    'broken clouds': 3,
    'overcast clouds': 4,
    'mist': 5,
    'haze': 6,
    'fog': 7,
    'light rain': 8,
    'moderate rain': 9,
    'heavy intensity rain': 10,
    'light snow': 11,
    'thunderstorm': 12,
    'thunderstorm with light rain': 13,
    'thunderstorm with rain': 14,
    'thunderstorm with heavy rain': 15
}
data['conditions'] = data['conditions'].map(ordinal_weather)

ordinal_weekday= {
    'Sunday': 1,
    'Monday': 2,
    'Tuesday': 3,
    'Wednesday': 4,
    'Thursday': 5,
    'Friday': 6,
    'Saturday': 7
}
data['day_of_week'] = df['day_of_week'].map(ordinal_weekday)

ordinal_color= {
    'Orange Permit': 2,
    'Gold Permit': 1,
    'Purple Permit': 3,
    'Pay-By-Space Permit': 0
}
data['color'] = df['color'].map(ordinal_color)

uniqueps = data['garage'].unique()
print(uniqueps)
data = pd.get_dummies(df, columns=['garage']) #one-hot encoding ps

cols = ['garage_ps1', 'garage_ps3', 'garage_ps4']
data[cols] = data[cols].astype(int)

data['time_modified'] = data['time']
def time_to_minutes(t):
    h, m = map(int, t.split(':'))
    return h * 60 + m

data['time_modified'] = data['time_modified'].apply(time_to_minutes)

data['time'] = data['time_modified']
data = data.drop(['time_modified'], axis=1)

import pandas as pd

# Define total spots dictionary for each (color, garage) pair
total_spots = {
    (1, 'ps1'): 120,
    (1, 'ps3'): 344,
    (1, 'ps4'): 488,
    (2, 'ps1'): 321,
    (2, 'ps3'): 242,
    (2, 'ps4'): 438,
    (0, 'ps1'): 101,
    (0, 'ps3'): 41,
    (0, 'ps4'): 138,
    (3, 'ps1'): 141,
    (3, 'ps3'): 65,
    (3, 'ps4'): 36,
}

# Function to determine if the lot is full
def is_full(row):
    key = (row['color'], row['garage'])
    total = total_spots.get(key, None)
    if total is None:
        return None  # or -1 for unknown
    return 1 if (row['spots_open'] / total) < 0.15 else 0

# Apply the function to df to generate the label column
df['label'] = df.apply(is_full, axis=1)

# Now add the label column to your encoded 'data' DataFrame
# Assuming the row order between df and data is still aligned
data['label'] = df['label']
data = data.drop(['spots_open'], axis=1)

# Split the 'date' column into 'month' and 'day'
data['month'] = pd.to_datetime(data['date']).dt.month
data.insert(0, 'month', data.pop('month'))
data['day'] = pd.to_datetime(data['date']).dt.day
data.insert(1, 'day', data.pop('day'))
# Drop the original 'date' column if no longer needed
data = data.drop(['date'], axis=1)

print(data['label'].value_counts().to_dict())
print(data.head())
print(data.columns)

# data.drop(nan_rows.index)
data = data.dropna(how='any') # drop the 5052 rows with NaN visibility

print(data.iloc[0])
data.to_csv('data_processed2.csv', index=False)
