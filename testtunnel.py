import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
import streamlit as st

# Define pitch type colors
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

# Load data
csv_file_path = 'FullFallData.csv'
df = pd.read_csv(csv_file_path)

# Drop rows with missing values for the required columns
df = df.dropna(subset=['VertRelAngle', 'HorzRelAngle', 'VertApprAngle', 'HorzApprAngle', 'TaggedPitchType', 'Pitcher'])

# Streamlit app layout
st.title("Pitch Movement and KDE Analysis")
st.sidebar.header("Filters")

# Dropdown to select pitcher
selected_pitcher = st.sidebar.selectbox("Select a Pitcher", options=df['Pitcher'].unique())

# Filter data for the selected pitcher
pitcher_data = df[df['Pitcher'] == selected_pitcher]

# Ellipse Plot Section
st.header(f"Pitch Movement Ellipses for {selected_pitcher}")

fig, ax = plt.subplots(figsize=(10, 8))
for pitch_type, group in pitcher_data.groupby('TaggedPitchType'):
    if pitch_type in pitch_colors:
        # Calculate mean and standard deviation for HorzRelAngle and VertRelAngle
        mean_horz = group['HorzRelAngle'].mean()
        mean_vert = group['VertRelAngle'].mean()
        std_horz = group['HorzRelAngle'].std()
        std_vert = group['VertRelAngle'].std()

        # Plot 1 SD ellipse
        ellipse = Ellipse(
            (mean_horz, mean_vert),
            width=2 * std_horz,
            height=2 * std_vert,
            edgecolor=pitch_colors[pitch_type],
            facecolor=pitch_colors[pitch_type],
            alpha=0.3,
            label=pitch_type
        )
        ax.add_patch(ellipse)

        # Plot the centroid as a larger dot
        ax.scatter(mean_horz, mean_vert, color=pitch_colors[pitch_type], edgecolor='black', s=100, label=f"{pitch_type} (Mean)")

# Customize plot
ax.set_title(f"Pitch Movement Ellipses for {selected_pitcher} (No Dots)", fontsize=16)
ax.set_xlabel("Horizontal Release Angle (°)", fontsize=12)
ax.set_ylabel("Vertical Release Angle (°)", fontsize=12)
ax.legend(loc='upper right', title="TaggedPitchType", fontsize=10)
ax.grid(True)

# Display plot
st.pyplot(fig)

# KDE Comparison Section
st.header(f"KDE Analysis for {selected_pitcher}")

# Dropdown to select TaggedPitchType
selected_pitch_type = st.sidebar.selectbox("Select a TaggedPitchType", options=pitcher_data['TaggedPitchType'].unique())

# Filter data by selected TaggedPitchType and the rest of the arsenal
selected_pitch_data = pitcher_data[pitcher_data['TaggedPitchType'] == selected_pitch_type]
other_pitches_data = pitcher_data[pitcher_data['TaggedPitchType'] != selected_pitch_type]

features = ['VertRelAngle', 'HorzRelAngle', 'VertApprAngle', 'HorzApprAngle']
for feature in features:
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute KDE for the selected pitch type
    if len(selected_pitch_data) > 5:
        selected_kde = gaussian_kde(selected_pitch_data[feature])
        x_vals = np.linspace(selected_pitch_data[feature].min(), selected_pitch_data[feature].max(), 100)
        ax.plot(x_vals, selected_kde(x_vals), label=f'{selected_pitch_type}', color=pitch_colors.get(selected_pitch_type, 'black'))

    # Compute KDE for the rest of the arsenal
    if len(other_pitches_data) > 5:
        other_kde = gaussian_kde(other_pitches_data[feature])
        x_vals = np.linspace(other_pitches_data[feature].min(), other_pitches_data[feature].max(), 100)
        ax.plot(x_vals, other_kde(x_vals), label='Rest of Arsenal', color='grey', linestyle='--')

    # Customize the plot
    ax.set_title(f"{feature} Density Comparison", fontsize=14)
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    ax.grid(True)

    # Show plot in Streamlit
    st.pyplot(fig)
