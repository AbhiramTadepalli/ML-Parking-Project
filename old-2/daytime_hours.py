import pandas as pd

# Load the dataset
input_file = "filtered_dataset.csv"
output_file = "daytime_filtered_dataset.csv"

# Read the CSV file
df = pd.read_csv(input_file)

# Filter rows between 7am (07:00:00) and 10pm (22:00:00)
df['time'] = pd.to_datetime(df['time'], format='%H:%M').dt.time
start_time = pd.to_datetime('07:00', format='%H:%M').time()
end_time = pd.to_datetime('22:00', format='%H:%M').time()

filtered_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)].copy()
filtered_df['time'] = filtered_df['time'].apply(lambda t: t.strftime('%H:%M'))

# Save the filtered dataset
filtered_df.to_csv(output_file, index=False)