import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
import os

# Define custom color palette for pitch types
pitch_colors = {
    "Fastball": '#ff007d',
    "Four-Seam": '#ff007d',
    "Sinker": "#98165D",
    "Slider": "#67E18D",
    "Sweeper": "#1BB999",
    "Curveball": '#3025CE',
    "ChangeUp": '#F79E70',
    "Splitter": '#90EE32',
    "Cutter": "#BE5FA0",
    "Unknown": '#9C8975',
    "PitchOut": '#472C30'
}

# Define function to add lines at the origin
def add_origin_lines(ax):
    """Add lines at the origin to the given axes."""
    ax.axhline(0, color='black', linestyle='-', linewidth=0.75)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.75)

# Define function to load data
@st.cache_data
def load_data():
    # Adjust the path for the deployment environment
    data_file = 'VSGA - Sheet1 (1).csv'
    if not os.path.isfile(data_file):
        st.error(f"Data file {data_file} not found.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is not found

    df = pd.read_csv(data_file)

    # Data transformation similar to R code
    df = df.dropna(subset=["HorzBreak"])
    df['PitchType'] = df['TaggedPitchType'].replace({
        'Four-Seam': 'Fastball', 'Fastball': 'Fastball',
        'Sinker': 'Sinker', 'Slider': 'Slider',
        'Sweeper': 'Sweeper', 'Curveball': 'Curveball',
        'ChangeUp': 'ChangeUp', 'Splitter': 'Splitter',
        'Cutter': 'Cutter'
    }).fillna('Unknown')

    # Format pitcher names and create custom columns
    df['Pitcher'] = df['Pitcher'].str.replace(r'(\w+), (\w+)', r'\2 \1')
    df['inZone'] = np.where((df['PlateLocHeight'].between(1.6, 3.4)) &
                            (df['PlateLocSide'].between(-0.71, 0.71)), 1, 0)
    df['Chase'] = np.where((df['inZone'] == 0) &
                           (df['PitchCall'].isin(['FoulBall', 'FoulBallNotFieldable', 'InPlay', 'StrikeSwinging'])), 1,
                           0)
    df['CustomGameID'] = df['Date'] + ": " + df['AwayTeam'].str[:3] + " @ " + df['HomeTeam'].str[:3]

    return df

# Load data
df = load_data()

if df.empty:
    st.stop()  # Stop execution if no data is loaded

# Sidebar filters
pitcher = st.sidebar.selectbox("Select Pitcher", df['Pitcher'].unique())
games = st.sidebar.multiselect("Select Game(s)", df['CustomGameID'].unique(), default=df['CustomGameID'].unique())
batter_hand = st.sidebar.multiselect("Select Batter Hand", df['BatterSide'].unique(), default=df['BatterSide'].unique())

# Filter data based on user inputs
filtered_data = df[(
                           df['Pitcher'] == pitcher) &
                   (df['CustomGameID'].isin(games)) &
                   (df['BatterSide'].isin(batter_hand))
                   ]

# Define the strike zone boundaries and home plate segments
strike_zone = pd.DataFrame({
    'x': [-0.71, 0.71, 0.71, -0.71],
    'y': [1.6, 1.6, 3.5, 3.5]
})

home_plate_segments = pd.DataFrame({
    'x': [-0.71, 0.71, 0.71, -0.71, -0.71],
    'y': [1.6, 1.6, 3.5, 3.5, 1.6],
    'xend': [-0.71, 0.71, 0.71, -0.71, -0.71],
    'yend': [3.5, 3.5, 1.6, 1.6, 1.6]
})

# Display table of key metrics
st.subheader(f"{pitcher}: Pitch Metrics")

# Calculate metrics
metrics = filtered_data.groupby('PitchType').agg({
    'RelSpeed': 'mean',
    'InducedVertBreak': 'mean',
    'HorzBreak': 'mean',
    'SpinRate': 'mean',
    'RelHeight': 'mean',
    'RelSide': 'mean',
    'Extension': 'mean',
    'VertApprAngle': 'mean'
}).round(2).reset_index()

# Calculate Usage%
total_pitches = len(filtered_data)
usage_percentage = filtered_data['PitchType'].value_counts(normalize=True) * 100
metrics['Usage%'] = metrics['PitchType'].map(usage_percentage).round().astype(int)  # Round and convert to integer

# Reorder columns
metrics = metrics[
    ['PitchType', 'Usage%', 'RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide',
     'Extension', 'VertApprAngle']]
# Display metrics with Usage% in front of RelSpeed
st.dataframe(metrics)

# Plotting Pitch Movement
st.subheader(f"{pitcher}: Pitch Movement")
fig, ax = plt.subplots()

# Scatter plot for Pitch Movement
sns.scatterplot(data=filtered_data, x="HorzBreak", y="InducedVertBreak", hue="PitchType", palette=pitch_colors, ax=ax)

# Calculate average breaks for each pitch type
avg_breaks = filtered_data.groupby('PitchType').agg(
    avgHorzBreak=('HorzBreak', 'mean'),
    avgVertBreak=('InducedVertBreak', 'mean')
).reset_index()

# Plot average breaks as larger, lightly shaded circles
for _, row in avg_breaks.iterrows():
    ax.scatter(row['avgHorzBreak'], row['avgVertBreak'],
               color=pitch_colors[row['PitchType']],
               edgecolor='black',
               s=800,  # Increased size of the circles
               alpha=0.35,  # Lightly shaded (semi-transparent)
               )

# Add origin lines
add_origin_lines(ax)

# Set axis limits
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)

# Set title and legend
ax.set_title("Pitch Movement (Horizontal vs Vertical Break)")
ax.legend(title='Pitch Type', bbox_to_anchor=(1.05, 1), loc='upper left')

st.pyplot(fig)

# Plotting Velocity Distribution using Kernel Density Estimate
st.subheader(f"{pitcher}: Velocity Distribution (KDE)")
fig, ax = plt.subplots()

# Create a KDE plot for each PitchType
for pitch_type, color in pitch_colors.items():
    subset = filtered_data[filtered_data['PitchType'] == pitch_type]
    if not subset.empty:
        sns.kdeplot(subset['RelSpeed'], ax=ax, color=color, label=pitch_type, fill=True)

# Set plot title and labels
ax.set_title("Velocity Distribution (Kernel Density Estimate)")
ax.set_xlabel("Release Speed (mph)")
ax.set_ylabel("Density")

# Add legend
ax.legend(title='Pitch Type')

st.pyplot(fig)

# Plot for Pitch Locations
st.subheader(f"{pitcher}: Pitch Locations")
fig, ax = plt.subplots()

# Scatter plot for Pitch Locations
sns.scatterplot(data=filtered_data, x="PlateLocSide", y="PlateLocHeight", hue="PitchType", palette=pitch_colors,
                alpha=0.7, size=2.5, ax=ax)

# Add home plate and strike zone
home_plate = Polygon(home_plate_segments[['x', 'y']].values, closed=True, edgecolor='black', fill=None)
strike_zone_poly = Polygon(strike_zone[['x', 'y']].values, closed=True, edgecolor='black', facecolor='lightblue',
                           alpha=0.3)
ax.add_patch(home_plate)
ax.add_patch(strike_zone_poly)

# Add origin lines
add_origin_lines(ax)

# Set axis limits and labels
ax.set_xlim(-2, 2)
ax.set_ylim(0, 5)
ax.set_xticks(np.arange(-2, 2.5, 0.5))
ax.set_yticks(np.arange(0, 5.5, 1))
ax.set_title(f"{pitcher}: Pitch Locations")
ax.set_xlabel("Horizontal Location")
ax.set_ylabel("Vertical Location")

# Add legend
ax.legend(title='Pitch Type')

st.pyplot(fig)

# Plot for Strike Swinging
st.subheader(f"{pitcher}: Strike Swinging")
strike_swinging_data = filtered_data[filtered_data['PitchCall'] == 'StrikeSwinging']
fig, ax = plt.subplots()

# Scatter plot for Strike Swinging
sns.scatterplot(data=strike_swinging_data, x="PlateLocSide", y="PlateLocHeight", hue="PitchType", palette=pitch_colors,
                alpha=0.7, size=2.5, ax=ax)

# Add home plate and strike zone
home_plate = Polygon(home_plate_segments[['x', 'y']].values, closed=True, edgecolor='black', fill=None)
strike_zone_poly = Polygon(strike_zone[['x', 'y']].values, closed=True, edgecolor='black', facecolor='lightblue',
                           alpha=0.3)
ax.add_patch(home_plate)
ax.add_patch(strike_zone_poly)

# Add origin lines
add_origin_lines(ax)

# Set axis limits and labels
ax.set_xlim(-2, 2)
ax.set_ylim(0, 5)
ax.set_xticks(np.arange(-2, 2.5, 0.5))
ax.set_yticks(np.arange(0, 5.5, 1))
ax.set_title(f"{pitcher}: Strike Swinging")
ax.set_xlabel("Horizontal Location")
ax.set_ylabel("Vertical Location")

# Add legend
ax.legend(title='Pitch Type')

st.pyplot(fig)
