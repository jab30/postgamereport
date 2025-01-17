import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.patches import Ellipse
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
    "Undefined": '#9C8975',
    "PitchOut": '#472C30'
}

# Define function to add lines at the origin
def add_origin_lines(ax):
    """Add lines at the origin to the given axes."""
    ax.axhline(0, color='black', linestyle='-', linewidth=0.75)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.75)

# Strike zone outline function
def add_strike_zone(ax):
    # Strike zone outline
    ax.plot([-10 / 12, 10 / 12], [1.6, 1.6], color='b', linewidth=2)
    ax.plot([-10 / 12, 10 / 12], [3.5, 3.5], color='b', linewidth=2)
    ax.plot([-10 / 12, -10 / 12], [1.6, 3.5], color='b', linewidth=2)
    ax.plot([10 / 12, 10 / 12], [1.6, 3.5], color='b', linewidth=2)

    # Inner Strike zone
    ax.plot([-10 / 12, 10 / 12], [1.5 + 2 / 3, 1.5 + 2 / 3], color='b', linewidth=1)
    ax.plot([-10 / 12, 10 / 12], [1.5 + 4 / 3, 1.5 + 4 / 3], color='b', linewidth=1)
    ax.axvline(10 / 36, ymin=(1.6 - 0) / (5 - 0), ymax=(3.5 - 0) / (5 - 0), color='b', linewidth=1)
    ax.axvline(-10 / 36, ymin=(1.6 - 0) / (5 - 0), ymax=(3.5 - 0) / (5 - 0), color='b', linewidth=1)


# Define function to load data
@st.cache_data
def load_data():
    data_files = [
        '20241004-KennesawWalterKelly-Private-1_unverified.csv',
        '20241005-KennesawWalterKelly-Private-2_unverified.csv',
        '20241010-KennesawWalterKelly-Private-1_unverified.csv',
        '20241011-KennesawWalterKelly-Private-2_unverified.csv',
        '20241101-KennesawWalterKelly-Private-1_unverified.csv',
        '20241031-KennesawWalterKelly-Private-1_unverified.csv',
        '20241018-GeorgiaTech-Private-3_unverified.csv',
        '20241118-KennesawWalterKelly-Private-2_unverified.csv',
        '20241121-KennesawWalterKelly-Private-4_unverified.csv',
        '20250109-KennesawWalterKelly-Private-1_unverified.csv',
        '20250116-KennesawWalterKelly-Private-1_unverified.csv',
        '20250117-KennesawWalterKelly-Private-1_unverified.csv'
    ]

    all_data = pd.DataFrame()

    for data_file in data_files:
        if not os.path.isfile(data_file):
            st.error(f"Data file {data_file} not found.")
            return pd.DataFrame()

        df = pd.read_csv(data_file)

        # Drop rows with missing 'HorzBreak'
        df = df.dropna(subset=["HorzBreak"])
        df['PitchType'] = df['TaggedPitchType'].replace({
            'Four-Seam': 'Fastball', 'Fastball': 'Fastball',
            'Sinker': 'Sinker', 'Slider': 'Slider',
            'Sweeper': 'Sweeper', 'Curveball': 'Curveball',
            'ChangeUp': 'ChangeUp', 'Splitter': 'Splitter',
            'Cutter': 'Cutter'
        }).fillna('Unknown')

        df['Pitcher'] = df['Pitcher'].str.replace(r'(\w+), (\w+)', r'\2 \1')
        df['inZone'] = np.where((df['PlateLocHeight'].between(1.6, 3.4)) &
                                (df['PlateLocSide'].between(-0.71, 0.71)), 1, 0)
        df['Chase'] = np.where((df['inZone'] == 0) &
                               (df['PitchCall'].isin(['FoulBall', 'FoulBallNotFieldable', 'InPlay', 'StrikeSwinging'])),
                               1, 0)
        df['CustomGameID'] = df['Date'] + ": " + df['AwayTeam'].str[:3] + " @ " + df['HomeTeam'].str[:3]

        all_data = pd.concat([all_data, df], ignore_index=True)

    return all_data


# Load data
df = load_data()

if df.empty:
    st.stop()
# Filter the dataset for pitchers with 'KEN_OWL' as their team
ken_owl_pitchers = df[df['PitcherTeam'] == 'KEN_OWL']['Pitcher'].unique()

# Sidebar filter for selecting pitchers
pitcher = st.sidebar.selectbox("Select Pitcher", ken_owl_pitchers)

# Sidebar filters for other parameters
games = st.sidebar.multiselect("Select Game(s)", df['CustomGameID'].unique(), default=df['CustomGameID'].unique())
batter_hand = st.sidebar.multiselect("Select Batter Hand", df['BatterSide'].unique(), default=df['BatterSide'].unique())

# Filter data based on user inputs
filtered_data = df[(df['Pitcher'] == pitcher) &
                   (df['CustomGameID'].isin(games)) &
                   (df['BatterSide'].isin(batter_hand))]
# Display table of key metrics with AvgEV and AvgLaunchAngle
st.subheader(f"{pitcher}: Pitch Metrics")
metrics = filtered_data.groupby('PitchType').agg({
    'RelSpeed': 'mean',
    'InducedVertBreak': 'mean',
    'HorzBreak': 'mean',
    'SpinRate': 'mean',
    'RelHeight': 'mean',
    'RelSide': 'mean',
    'Extension': 'mean',
    'VertApprAngle': 'mean',
    'HorzApprAngle': 'mean',
    'ExitSpeed': 'mean',  # For AvgEV
    'Angle': 'mean'       # For AvgLaunchAngle
}).round(2).reset_index()

# Rename the columns for readability
metrics.rename(columns={'ExitSpeed': 'AvgEV', 'Angle': 'AvgLaunchAngle'}, inplace=True)

# Add Usage Percentage to the metrics table
total_pitches = len(filtered_data)
usage_percentage = filtered_data['PitchType'].value_counts(normalize=True) * 100
metrics['Usage%'] = metrics['PitchType'].map(usage_percentage).round().astype(int)

# Reorder columns to include AvgEV and AvgLaunchAngle
metrics = metrics[
    ['PitchType', 'Usage%', 'RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate',
     'RelHeight', 'RelSide', 'Extension', 'VertApprAngle', 'HorzApprAngle', 'AvgEV', 'AvgLaunchAngle']
]

# Display table of key metrics with AvgEV and AvgLaunchAngle
st.subheader(f"{pitcher}: Pitch Metrics")
metrics = filtered_data.groupby('PitchType').agg({
    'RelSpeed': 'mean',
    'InducedVertBreak': 'mean',
    'HorzBreak': 'mean',
    'SpinRate': 'mean',
    'RelHeight': 'mean',
    'RelSide': 'mean',
    'Extension': 'mean',
    'VertApprAngle': 'mean',
    'HorzApprAngle': 'mean',
    'ExitSpeed': 'mean',  # For AvgEV
    'Angle': 'mean'       # For AvgLaunchAngle
}).round(2).reset_index()

# Rename the columns for readability
metrics.rename(columns={'ExitSpeed': 'AvgEV', 'Angle': 'AvgLaunchAngle'}, inplace=True)

# Add Usage Percentage to the metrics table
total_pitches = len(filtered_data)
usage_percentage = filtered_data['PitchType'].value_counts(normalize=True) * 100
metrics['Usage%'] = metrics['PitchType'].map(usage_percentage).round().astype(int)

# Reorder columns to include AvgEV and AvgLaunchAngle
metrics = metrics[
    ['PitchType', 'Usage%', 'RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate',
     'RelHeight', 'RelSide', 'Extension', 'VertApprAngle', 'HorzApprAngle', 'AvgEV', 'AvgLaunchAngle']
]

# Calculate totals for AvgEV and AvgLaunchAngle, leaving others blank
totals = pd.DataFrame({
    'PitchType': ['Total'],
    'Usage%': [''],
    'RelSpeed': [''],
    'InducedVertBreak': [''],
    'HorzBreak': [''],
    'SpinRate': [''],
    'RelHeight': [''],
    'RelSide': [''],
    'Extension': [''],
    'VertApprAngle': [''],
    'HorzApprAngle': [''],
    'AvgEV': round(metrics['AvgEV'].mean(), 1),  # Rounded to tenths place
    'AvgLaunchAngle': round(metrics['AvgLaunchAngle'].mean(), 1)
}, index=[0])

# Append totals row to the metrics DataFrame
metrics = pd.concat([metrics, totals], ignore_index=True)

# Display the updated metrics table
st.dataframe(metrics)



# Function to draw confidence ellipse
def confidence_ellipse(x, y, ax, edgecolor, n_std=0.5, facecolor='none', **kwargs):
    # Calculate mean and covariance
    cov = np.cov(x, y)
    mean = [np.mean(x), np.mean(y)]

    # Calculate the eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort the eigenvalues and eigenvectors
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # Calculate the angle of the ellipse
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    angle = np.degrees(angle)

    # Calculate the width and height of the ellipse
    width, height = 2 * n_std * np.sqrt(eigvals)

    # Create the ellipse
    ellipse = Ellipse(xy=mean, width=width, height=height,
                      angle=angle, edgecolor=edgecolor, facecolor=facecolor, **kwargs)

    # Add the ellipse to the plot
    ax.add_patch(ellipse)


# Main plotting function
st.subheader(f"{pitcher}: Pitch Movement with Confidence Ellipses")
fig, ax = plt.subplots()

# Scatter plot for actual pitches
sns.scatterplot(data=filtered_data,
                x="HorzBreak",
                y="InducedVertBreak",
                hue="PitchType",
                palette=pitch_colors,
                ax=ax,
                s=15)

# Calculate average breaks
avg_breaks = filtered_data.groupby('PitchType').agg(
    avgHorzBreak=('HorzBreak', 'mean'),
    avgVertBreak=('InducedVertBreak', 'mean'),
    stdHorzBreak=('HorzBreak', 'std'),
    stdVertBreak=('InducedVertBreak', 'std')
).reset_index()

# Plot average breaks and confidence ellipses
for label in avg_breaks['PitchType']:
    subset = filtered_data[filtered_data['PitchType'] == label]
    if len(subset) > 4:
        try:
            if subset['PitcherThrows'].iloc[0] == 'Right':
                confidence_ellipse(subset['HorzBreak'], subset['InducedVertBreak'], ax=ax,
                                   edgecolor=pitch_colors[label], n_std=2, facecolor=pitch_colors[label], alpha=0.2)
            elif subset['PitcherThrows'].iloc[0] == 'Left':
                confidence_ellipse(-subset['HorzBreak']*-1, subset['InducedVertBreak'], ax=ax,
                                   edgecolor=pitch_colors[label], n_std=2, facecolor=pitch_colors[label], alpha=0.2)
        except ValueError:
            continue

# Final plot adjustments
add_origin_lines(ax)
ax.set_xlim(-25, 25)
ax.set_ylim(-25, 25)
ax.set_title("Pitch Movement (Horizontal vs Vertical Break)")
ax.legend(title='Pitch Type', bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)

# Plotting Player Release Metrics
st.subheader(f"{pitcher} Player Release Metrics (Catchers View)")
fig, ax = plt.subplots()

# Scatter plot for RelSide vs. RelHeight, colored by PitchType
sns.scatterplot(data=filtered_data,
                x="RelSide",
                y="RelHeight",
                hue="PitchType",
                palette=pitch_colors,
                ax=ax,
                s=20)  # Smaller size for actual pitches

# Set fixed plot limits and labels
ax.set_xlim(-4, 4)
ax.set_ylim(0, 8)
ax.set_title("Player Release Metrics (Release Side vs Release Height)")
ax.set_xlabel("Release Side (in feet)")
ax.set_ylabel("Release Height (in feet)")

# Add grid lines at each integer
for i in range(-4, 5):
    ax.axvline(x=i, color='gray', linestyle='--', linewidth=0.5)
for j in range(0, 9):
    ax.axhline(y=j, color='gray', linestyle='--', linewidth=0.5)
# Add legend
ax.legend(title='Pitch Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
st.pyplot(fig)



# Plotting Velocity Distribution using KDE
st.subheader(f"{pitcher}: Velocity Distribution (KDE)")
fig, ax = plt.subplots()
for pitch_type, color in pitch_colors.items():
    subset = filtered_data[filtered_data['PitchType'] == pitch_type]
    if not subset.empty:
        sns.kdeplot(subset['RelSpeed'], ax=ax, color=color, label=pitch_type, fill=True)
ax.set_title("Velocity Distribution (Kernel Density Estimate)")
ax.set_xlabel("Release Speed (mph)")
ax.set_ylabel("Density")
ax.legend(title='Pitch Type')
st.pyplot(fig)

# Plot for Pitch Locations
st.subheader(f"{pitcher}: Pitch Locations")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_data, x="PlateLocSide", y="PlateLocHeight", hue="PitchType", palette=pitch_colors,)

# Add home plate and strike zone
add_strike_zone(ax)


# Add the lower plate at plate_y = 0.5
plate_y = 0.5  # Lower plate
ax.plot([-8.5 / 12, 8.5 / 12], [plate_y, plate_y], color='b', linewidth=2)  # Plate top
ax.plot([-8.5 / 12, -8.25 / 12], [plate_y, plate_y + 0.15], color='b', linewidth=2)  # Left side of plate
ax.plot([8.5 / 12, 8.25 / 12], [plate_y, plate_y + 0.15], color='b', linewidth=2)  # Right side of plate
ax.plot([8.28 / 12, 0], [plate_y + 0.15, plate_y + 0.25], color='b', linewidth=2)  # Right triangle of plate
ax.plot([-8.28 / 12, 0], [plate_y + 0.15, plate_y + 0.25], color='b', linewidth=2)  # Left triangle of plate

# Set limits and labels
ax.set_xlim(-2, 2)
ax.set_ylim(0, 5)
ax.set_xticks(np.arange(-2, 2.5, 0.5))
ax.set_yticks(np.arange(0, 5.5, 1))
ax.set_title(f"{pitcher}: Pitch Locations")
ax.set_xlabel("Horizontal Location")
ax.set_ylabel("Vertical Location")
ax.legend(title='Pitch Type')
st.pyplot(fig)


# Plot Strike Swinging Pitches
st.subheader(f"{pitcher}: Strike Swinging Locations")
strike_swinging_data = filtered_data[filtered_data['PitchCall'] == 'StrikeSwinging']
fig, ax = plt.subplots()
sns.scatterplot(data=strike_swinging_data, x="PlateLocSide", y="PlateLocHeight", hue="PitchType", palette=pitch_colors,)

# Add home plate and strike zone
add_strike_zone(ax)


# Add the lower plate at plate_y = 0.5
plate_y = 0.5  # Lower plate
ax.plot([-8.5 / 12, 8.5 / 12], [plate_y, plate_y], color='b', linewidth=2)  # Plate top
ax.plot([-8.5 / 12, -8.25 / 12], [plate_y, plate_y + 0.15], color='b', linewidth=2)  # Left side of plate
ax.plot([8.5 / 12, 8.25 / 12], [plate_y, plate_y + 0.15], color='b', linewidth=2)  # Right side of plate
ax.plot([8.28 / 12, 0], [plate_y + 0.15, plate_y + 0.25], color='b', linewidth=2)  # Right triangle of plate
ax.plot([-8.28 / 12, 0], [plate_y + 0.15, plate_y + 0.25], color='b', linewidth=2)  # Left triangle of plate

# Set limits and labels
ax.set_xlim(-2, 2)
ax.set_ylim(0, 5)
ax.set_xticks(np.arange(-2, 2.5, 0.5))
ax.set_yticks(np.arange(0, 5.5, 1))
ax.set_title(f"{pitcher}: Strike Swinging Locations")
ax.set_xlabel("Horizontal Location")
ax.set_ylabel("Vertical Location")
ax.legend(title='Pitch Type')
st.pyplot(fig)


# Plot Chase Pitches
st.subheader(f"{pitcher}: Chase Pitch Locations")
chase_data = filtered_data[filtered_data['Chase'] == 1]
fig, ax = plt.subplots()
sns.scatterplot(data=chase_data, x="PlateLocSide", y="PlateLocHeight", hue="PitchType", palette=pitch_colors,)

# Add home plate and strike zone
add_strike_zone(ax)


# Add the lower plate at plate_y = 0.5
plate_y = 0.5  # Lower plate
ax.plot([-8.5 / 12, 8.5 / 12], [plate_y, plate_y], color='b', linewidth=2)  # Plate top
ax.plot([-8.5 / 12, -8.25 / 12], [plate_y, plate_y + 0.15], color='b', linewidth=2)  # Left side of plate
ax.plot([8.5 / 12, 8.25 / 12], [plate_y, plate_y + 0.15], color='b', linewidth=2)  # Right side of plate
ax.plot([8.28 / 12, 0], [plate_y + 0.15, plate_y + 0.25], color='b', linewidth=2)  # Right triangle of plate
ax.plot([-8.28 / 12, 0], [plate_y + 0.15, plate_y + 0.25], color='b', linewidth=2)  # Left triangle of plate

# Set limits and labels
ax.set_xlim(-2, 2)
ax.set_ylim(0, 5)
ax.set_xticks(np.arange(-2, 2.5, 0.5))
ax.set_yticks(np.arange(0, 5.5, 1))
ax.set_title(f"{pitcher}: Chase Pitch Locations")
ax.set_xlabel("Horizontal Location")
ax.set_ylabel("Vertical Location")
ax.legend(title='Pitch Type')
st.pyplot(fig)


# Plot Called Strikes
st.subheader(f"{pitcher}: Called Strike Locations")
called_strike_data = filtered_data[filtered_data['PitchCall'] == 'StrikeCalled']
fig, ax = plt.subplots()
sns.scatterplot(data=called_strike_data, x="PlateLocSide", y="PlateLocHeight", hue="PitchType", palette=pitch_colors,)

# Add home plate and strike zone
add_strike_zone(ax)


# Add the lower plate at plate_y = 0.5
plate_y = 0.5  # Lower plate
ax.plot([-8.5 / 12, 8.5 / 12], [plate_y, plate_y], color='b', linewidth=2)  # Plate top
ax.plot([-8.5 / 12, -8.25 / 12], [plate_y, plate_y + 0.15], color='b', linewidth=2)  # Left side of plate
ax.plot([8.5 / 12, 8.25 / 12], [plate_y, plate_y + 0.15], color='b', linewidth=2)  # Right side of plate
ax.plot([8.28 / 12, 0], [plate_y + 0.15, plate_y + 0.25], color='b', linewidth=2)  # Right triangle of plate
ax.plot([-8.28 / 12, 0], [plate_y + 0.15, plate_y + 0.25], color='b', linewidth=2)  # Left triangle of plate

# Set limits and labels
ax.set_xlim(-2, 2)
ax.set_ylim(0, 5)
ax.set_xticks(np.arange(-2, 2.5, 0.5))
ax.set_yticks(np.arange(0, 5.5, 1))
ax.set_title(f"{pitcher}: Called Strike Locations")
ax.set_xlabel("Horizontal Location")
ax.set_ylabel("Vertical Location")
ax.legend(title='Pitch Type')
st.pyplot(fig)
