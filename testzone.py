import streamlit as st
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Function to load and concatenate CSV files
def load_data():
    csv_files = [
        '20241004-KennesawWalterKelly-Private-1_unverified.csv',
        '20241005-KennesawWalterKelly-Private-2_unverified.csv',
        '20241010-KennesawWalterKelly-Private-1_unverified.csv',
        '20241011-KennesawWalterKelly-Private-2_unverified.csv',
        '20241018-GeorgiaTech-Private-3_unverified.csv',
    ]

    # Concatenate all CSV files into a single DataFrame
    df_list = [pd.read_csv(csv_file) for csv_file in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    return df


# Load data
df = load_data()

# Streamlit app
st.title('Pitch Location Density Plot (Catchers POV)')
st.sidebar.header('Filter Options')

# Sidebar filters
pitchers = df[df['PitcherTeam'] == 'KEN_OWL']['Pitcher'].unique()
selected_pitcher = st.sidebar.selectbox('Select Pitcher', pitchers)
pitch_types = df[df['Pitcher'] == selected_pitcher]['TaggedPitchType'].unique()
selected_tagged_pitch_type = st.sidebar.selectbox('Select Pitch Type', pitch_types)

# New Date filter with multi-selection
dates = df['Date'].unique()
dates_with_all_option = ['All of Fall'] + list(dates)
selected_dates = st.sidebar.multiselect('Select Date(s)', dates_with_all_option, default='All of Fall')

# Handle 'All of Fall' option
if 'All of Fall' in selected_dates:
    selected_dates = dates

# Filter by selected Pitcher, TaggedPitchType, and Date
filtered_df = df[
    (df['Pitcher'] == selected_pitcher) &
    (df['TaggedPitchType'] == selected_tagged_pitch_type) &
    (df['Date'].isin(selected_dates))
    ]

# Density plot
plt.figure(figsize=(8, 6))
sns.kdeplot(
    data=filtered_df, x='PlateLocSide', y='PlateLocHeight',
    fill=True, cmap='Reds', bw_adjust=0.5
)

# Add strike zone (assuming standard strike zone dimensions)
strike_zone = patches.Rectangle((-0.83, 1.5), 1.66, 2, linewidth=1, edgecolor='blue', facecolor='none')
plt.gca().add_patch(strike_zone)

plt.gca().invert_xaxis()  # To reflect pitcher's perspective
plt.xlim(-3, 3)  # Fix x-axis
plt.ylim(0.5, 5.5)  # Fix y-axis
plt.title(f'Density Plot for {selected_pitcher} - {selected_tagged_pitch_type} on selected dates')
plt.xlabel('Plate Location Side')
plt.ylabel('Plate Location Height')
st.pyplot(plt)
plt.clf()