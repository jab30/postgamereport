import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files into Pandas DataFrames
csv_files = [
    '20241004-KennesawWalterKelly-Private-1_unverified.csv',
    '20241005-KennesawWalterKelly-Private-2_unverified.csv',
    '20241010-KennesawWalterKelly-Private-1_unverified.csv',
    '20241011-KennesawWalterKelly-Private-2_unverified.csv',
    '20241018-GeorgiaTech-Private-3_unverified.csv'
]
data_frames = [pd.read_csv(file) for file in csv_files]
data = pd.concat(data_frames, ignore_index=True)

# Unique pitchers
pitchers = data['Pitcher'].unique()

# Pitch type colors
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

# Streamlit app layout
st.title("Rel Angle and Strike Zone App")

# Pitcher selection
selected_pitcher = st.selectbox("Select a Pitcher", options=pitchers)

# Strike zone section selection
st.markdown("### Select a Strike Zone Section")
zone_options = ["All Zones", "Gloveside (Righty)", "Middle (Righty)", "Armside (Righty)"]
selected_zone = st.radio("Select Zone", options=zone_options)

# Define zone boundaries
zone_limits = {
    "Gloveside (Righty)": (-10 / 12, -3 / 12),
    "Middle (Righty)": (-3 / 12, 3 / 12),
    "Armside (Righty)": (3 / 12, 10 / 12)
}

# Display the strike zone and home plate as a separate graph
st.markdown("### Strike Zone with Home Plate")

# Plot the strike zone and home plate
fig, ax = plt.subplots(figsize=(6, 6))

# Strike zone boundaries
ax.plot([-10 / 12, 10 / 12], [1.6, 1.6], color='blue', linewidth=2)
ax.plot([-10 / 12, 10 / 12], [3.5, 3.5], color='blue', linewidth=2)
ax.plot([-10 / 12, -10 / 12], [1.6, 3.5], color='blue', linewidth=2)
ax.plot([10 / 12, 10 / 12], [1.6, 3.5], color='blue', linewidth=2)
# Dividing lines for sections
ax.plot([-3 / 12, -3 / 12], [1.6, 3.5], color='blue', linestyle='--', linewidth=2)
ax.plot([3 / 12, 3 / 12], [1.6, 3.5], color='blue', linestyle='--', linewidth=2)

# Home plate configuration
plate_y = 0.5  # Position for the lower plate
ax.plot([-8.5 / 12, 8.5 / 12], [plate_y, plate_y], color='blue', linewidth=2)  # Plate top
ax.plot([-8.5 / 12, -8.25 / 12], [plate_y, plate_y + 0.15], color='blue', linewidth=2)  # Left side
ax.plot([8.5 / 12, 8.25 / 12], [plate_y, plate_y + 0.15], color='blue', linewidth=2)  # Right side
ax.plot([8.28 / 12, 0], [plate_y + 0.15, plate_y + 0.25], color='blue', linewidth=2)  # Right triangle
ax.plot([-8.28 / 12, 0], [plate_y + 0.15, plate_y + 0.25], color='blue', linewidth=2)  # Left triangle

# Adjust axes to include home plate
ax.set_xlim(-1, 1)
ax.set_ylim(0, 4)
ax.set_xlabel("Plate Location Side")
ax.set_ylabel("Plate Location Height")
ax.set_title("Strike Zone with Home Plate (Pitcher's View)")
st.pyplot(fig)

# Filter data based on pitcher and selected zone
if selected_pitcher:
    filtered_data = data[data['Pitcher'] == selected_pitcher]

    # Apply zone filter if a specific zone is selected
    if selected_zone != "All Zones":
        x_min, x_max = zone_limits[selected_zone]
        filtered_data = filtered_data[(filtered_data['PlateLocSide'] >= x_min) &
                                      (filtered_data['PlateLocSide'] <= x_max) &
                                      (filtered_data['PlateLocHeight'] >= 1.6) &
                                      (filtered_data['PlateLocHeight'] <= 3.5)]

    # Plot the pitches with pitcher-specific axis limits
    fig, ax = plt.subplots(figsize=(8, 6))
    for pitch_type, color in pitch_colors.items():
        pitch_data = filtered_data[filtered_data['TaggedPitchType'] == pitch_type]
        ax.scatter(
            pitch_data['HorzRelAngle'],
            pitch_data['VertRelAngle'],
            color=color,
            label=pitch_type,
            alpha=0.7
        )

    # Set axis labels and title
    ax.set_xlabel('Horizontal Approach Angle')
    ax.set_ylabel('Vertical Approach Angle')
    ax.set_title(f"Pitch Types for {selected_pitcher}")

    # Set dynamic axis limits based on filtered data
    if not filtered_data.empty:
        ax.set_xlim(filtered_data['HorzRelAngle'].min() - 1, filtered_data['HorzRelAngle'].max() + 1)
        ax.set_ylim(filtered_data['VertRelAngle'].min() - 1, filtered_data['VertRelAngle'].max() + 1)

    ax.legend()
    st.pyplot(fig)
