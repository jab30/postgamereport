import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.colors as mcolors


# Function to load and filter CSV data
@st.cache_data
def load_csv(file_list):
    data = pd.concat([pd.read_csv(file) for file in file_list])
    return data[data['BatterTeam'] == 'KEN_OWL']


# Calculate wOBA based on PlayResult
def calculate_woba(batter_data):
    uBB = len(batter_data[batter_data['PlayResult'] == 'Walk'])
    HBP = len(batter_data[batter_data['PlayResult'] == 'HitByPitch'])
    single = len(batter_data[batter_data['PlayResult'] == 'Single'])
    double = len(batter_data[batter_data['PlayResult'] == 'Double'])
    triple = len(batter_data[batter_data['PlayResult'] == 'Triple'])
    HR = len(batter_data[batter_data['PlayResult'] == 'HomeRun'])
    SF = len(batter_data[batter_data['PlayResult'] == 'SacFly'])
    IBB = len(batter_data[batter_data['PlayResult'] == 'IntentionalWalk'])

    AB = len(batter_data[~batter_data['PlayResult'].isin(['Walk', 'HitByPitch', 'IntentionalWalk', 'SacFly'])])

    numerator = (0.690 * uBB + 0.722 * HBP + 0.888 * single +
                 1.271 * double + 1.616 * triple + 2.101 * HR)
    denominator = (AB + uBB + HBP + SF)

    return numerator / denominator if denominator > 0 else 0


# Calculate Z-Swing% based on zone criteria
def calculate_z_swing(batter_data):
    strike_zone_side = (-1.6, 1.6)
    strike_zone_height = (1.6, 3.5)

    # Filter pitches in the zone
    in_zone = batter_data[
        (batter_data['PlateLocSide'] >= strike_zone_side[0]) &
        (batter_data['PlateLocSide'] <= strike_zone_side[1]) &
        (batter_data['PlateLocHeight'] >= strike_zone_height[0]) &
        (batter_data['PlateLocHeight'] <= strike_zone_height[1])
    ]

    total_in_zone = len(in_zone)
    swings = len(in_zone[in_zone['PitchCall'].isin(['InPlay', 'StrikeSwinging'])])

    return (swings / total_in_zone * 100) if total_in_zone > 0 else 0


# Calculate Z-Contact% based on zone criteria
def calculate_z_contact(batter_data):
    strike_zone_side = (-1.6, 1.6)
    strike_zone_height = (1.6, 3.5)

    # Filter pitches in the zone
    in_zone = batter_data[
        (batter_data['PlateLocSide'] >= strike_zone_side[0]) &
        (batter_data['PlateLocSide'] <= strike_zone_side[1]) &
        (batter_data['PlateLocHeight'] >= strike_zone_height[0]) &
        (batter_data['PlateLocHeight'] <= strike_zone_height[1])
    ]

    # Calculate swings at pitches in the zone
    swings_in_zone = len(in_zone[in_zone['PitchCall'].isin(['InPlay', 'StrikeSwinging'])])

    # Calculate swings that made contact
    contact = len(in_zone[in_zone['PitchCall'] == 'InPlay'])

    return (contact / swings_in_zone * 100) if swings_in_zone > 0 else 0


# Calculate SwStr% based on the total number of pitches
def calculate_swstr(batter_data):
    total_pitches = len(batter_data)
    swinging_strikes = len(batter_data[batter_data['PitchCall'] == 'StrikeSwinging'])

    return (swinging_strikes / total_pitches * 100) if total_pitches > 0 else 0


# Calculate stats for each batter
def calculate_batter_stats(data):
    batters = data['Batter'].unique()
    batter_stats = []

    for batter in batters:
        batter_data = data[data['Batter'] == batter]

        total_pitches = len(batter_data)
        woba = calculate_woba(batter_data)
        z_contact = calculate_z_contact(batter_data)
        z_swing = calculate_z_swing(batter_data)
        swstr = calculate_swstr(batter_data)  # Calculate SwStr%

        avg_ev = batter_data[batter_data['PitchCall'] == 'InPlay']['ExitSpeed'].mean()
        max_ev = batter_data[batter_data['PitchCall'] == 'InPlay']['ExitSpeed'].max()

        batter_stats.append({
            'Batter': batter,
            'Total Pitches': total_pitches,
            'wOBA': round(woba, 3),
            'Z-Contact%': round(100 - z_contact, 1),
            'Z-Swing%': round(100 - z_swing, 1),  # Adjusted to keep the same format
            'SwStr%': round(swstr, 1),  # Added SwStr%
            'Avg EV': round(avg_ev, 2),
            'Max EV': round(max_ev, 2)
        })

    return pd.DataFrame(batter_stats)


# List of CSV files to load
csv_files = [
    '20241004-KennesawWalterKelly-Private-1_unverified.csv',
    '20241005-KennesawWalterKelly-Private-2_unverified.csv',
    '20241010-KennesawWalterKelly-Private-1_unverified.csv',
    '20241011-KennesawWalterKelly-Private-2_unverified.csv',
    '20241018-GeorgiaTech-Private-3_unverified.csv'
]

# Streamlit app layout
st.title("KEN_OWL Batters Stats")

# Load data and filter by KEN_OWL team
data = load_csv(csv_files)

# Sidebar options to select either the leaderboard or search for a player
option = st.sidebar.selectbox("Select an option", ['Leaderboard', 'Search for Player'])

# Display leaderboard
if option == 'Leaderboard':
    st.subheader("Leaderboard: All Batters Stats")
    batter_stats = calculate_batter_stats(data)
    st.dataframe(batter_stats)

# Search for a specific player
elif option == 'Search for Player':
    st.subheader("Search for a Specific Player")

    # Search bar to enter player's name
    search_query = st.text_input("Enter Batter's Name:")

    if search_query:
        batter_data = data[data['Batter'].str.contains(search_query, case=False)]

        if not batter_data.empty:
            batter_stats = calculate_batter_stats(batter_data)
            st.write(f"Stats for: {search_query}")
            st.dataframe(batter_stats)

            # Create the damage heatmap
            in_play_data = batter_data[batter_data['PitchCall'] == 'InPlay']
            if not in_play_data.empty:
                plt.figure(figsize=(8, 6))

                # Density plot for damage heatmap
                sns.kdeplot(
                    data=in_play_data, x='PlateLocSide', y='PlateLocHeight',
                    fill=True, cmap='Reds', bw_adjust=0.5
                )

                # Add strike zone (assuming standard strike zone dimensions)
                strike_zone = patches.Rectangle((-0.83, 1.5), 1.66, 2, linewidth=1, edgecolor='blue', facecolor='none')
                plt.gca().add_patch(strike_zone)

                plt.gca().invert_xaxis()  # To reflect pitcher's perspective
                plt.xlim(-3, 3)  # Fix x-axis
                plt.ylim(0.5, 5.5)  # Fix y-axis
                plt.title(f'Damage Heatmap for {search_query}(Pitchers POV)')
                plt.xlabel('Plate Location Side')
                plt.ylabel('Plate Location Height')
                st.pyplot(plt)
                plt.clf()

                # Create a custom colormap
                colors = ["blue", "white", "#be0000"]  # Define the color gradient
                cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)

                # Scatter plot for Exit Speed
                plt.figure(figsize=(8, 6))
                # Normalize the ExitSpeed values to the range [0, 1] for coloring
                norm = plt.Normalize(in_play_data['ExitSpeed'].min(), in_play_data['ExitSpeed'].max())

                scatter = plt.scatter(
                    in_play_data['PlateLocSide'], in_play_data['PlateLocHeight'],
                    c=in_play_data['ExitSpeed'], cmap=cmap, norm=norm, alpha=0.6
                )
                plt.colorbar(scatter, label='Exit Speed (mph)')

                # Add strike zone (assuming standard strike zone dimensions)
                strike_zone = patches.Rectangle((-0.83, 1.5), 1.66, 2, linewidth=1, edgecolor='blue', facecolor='none')
                plt.gca().add_patch(strike_zone)

                plt.gca().invert_xaxis()  # To reflect pitcher's perspective
                plt.xlim(-3, 3)  # Fix x-axis
                plt.ylim(0.5, 5.5)  # Fix y-axis
                plt.title(f'Exit Velocity Scatter Plot for {search_query} (Pitchers POV)')
                plt.xlabel('Plate Location Side')
                plt.ylabel('Plate Location Height')

                # Add tiny text labels for Exit Speed next to each dot
                for i in range(len(in_play_data)):
                    plt.text(
                        in_play_data['PlateLocSide'].iloc[i],
                        in_play_data['PlateLocHeight'].iloc[i],
                        f"{in_play_data['ExitSpeed'].iloc[i]:.1f}",  # Format to one decimal place
                        fontsize=8,  # Tiny text size
                        ha='right',  # Horizontal alignment
                        va='bottom',  # Vertical alignment
                        alpha=0.0  # Slight transparency
                    )

                st.pyplot(plt)
                plt.clf()


            else:
                st.write("No 'InPlay' data available for the selected batter.")
        else:
            st.write(f"No data found for {search_query}.")
