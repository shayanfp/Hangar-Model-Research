import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import numpy as np
import sys
import os
import argparse

# --- Script Configuration ---
# Set to True to save a PDF of the hangar status at each event time.
# Set to False to only run the interactive animation.
EXPORT_VECTORS = False

# --- Environment Detection ---
# Automatically detect if running in Google Colab to adjust dependencies.
try:
    import google.colab
    RUNNING_IN_COLAB = True
except ImportError:
    RUNNING_IN_COLAB = False
    # Import GUI libraries only for local execution.
    import tkinter as tk
    from tkinter import filedialog

# --- UI and Plotting Style Configuration ---
# Set fonts based on the execution environment for better compatibility.
if RUNNING_IN_COLAB:
    # Use default sans-serif in Colab to prevent font warnings.
    plt.rcParams['font.family'] = ['sans-serif']
else:
    # Use modern, clean fonts for local execution.
    plt.rcParams['font.family'] = ['Open Sans', 'Segoe UI', 'Roboto', 'Arial', 'sans-serif']

plt.rcParams['font.size'] = 11
# Embed fonts as paths in the vector file (PDF) to ensure they render correctly elsewhere.
plt.rcParams['svg.fonttype'] = 'path'

# A modern color palette for all visual elements.
COLORS = {
    "hangar_bg": "#F8F9FA",
    "grid": "#DEE2E6",
    "border": "#495057",
    "airplane_border" : "#007BFF",
    "text": "#212529",
    "aircraft_static_edge": "#007BFF",
    "aircraft_static_face": "#D1E7FF",
    "aircraft_enter_edge": "#28A745",
    "aircraft_enter_face": "#D4EDDA",
    "aircraft_exit_edge": "#DC3545",
    "aircraft_exit_face": "#F8D7DA",
    "table_header": "#E9ECEF",
    "table_highlight": "#BDE0FE",
    "table_past": "#D4D4D4",
    "table_rejected": "#F5C6CB",
    "button_bg": "#F8F9FA",
    "button_hover": "#E9ECEF"
}


def load_and_validate_csv(file_source):
    """
    Loads and validates the solution CSV.
    It checks for required columns, calculates stay time, and exits on error.
    """
    try:
        df = pd.read_csv(file_source)

        # Define the exact columns expected in the input CSV.
        required_new_columns = [
            'Aircraft_ID', 'Accepted', 'Width', 'Length', 'ETA', 'Roll_In', 'X', 'Y',
            'ServT', 'ETD', 'Roll_Out', 'D_Arr', 'D_Dep', 'Hangar_Width', 'Hangar_Length', 'StartDate'
        ]

        # Validate that all required columns are present.
        missing_cols = [col for col in required_new_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: CSV is missing required columns: {', '.join(missing_cols)}.")
            sys.exit()

        # Create the 'Actual_Stay_Time' column required for analysis.
        df['Actual_Stay_Time'] = df['Roll_Out'] - df['Roll_In']

        return df

    except Exception as e:
        print(f"Error reading or validating the CSV file: {e}.")
        sys.exit()

# --- Argument Parsing and Data Loading ---
file_path = None
if not RUNNING_IN_COLAB:
    parser = argparse.ArgumentParser(description="Visualize aircraft hangar scheduling and layout from a solution CSV file.")
    parser.add_argument('--file', type=str, help="Path to the solution CSV file.")
    args = parser.parse_args()
    file_path = args.file

if file_path:
    if not os.path.exists(file_path):
        print(f"Error: The specified file does not exist: {file_path}")
        sys.exit()
    print(f"Loading data from provided file: {file_path}")
    df = load_and_validate_csv(file_path)
    print(f"'{os.path.basename(file_path)}' loaded and validated successfully.")
elif RUNNING_IN_COLAB:
    import io
    from google.colab import files
    print("Please upload your SolutionReport.csv file.")
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded. Exiting.")
        sys.exit()
    file_name = list(uploaded.keys())[0]
    if not file_name.lower().endswith('.csv'):
        print("Error: The uploaded file must be a CSV. Exiting.")
        sys.exit()
    df = load_and_validate_csv(io.BytesIO(uploaded[file_name]))
    print(f"'{file_name}' loaded and validated successfully.")
else:
    root = tk.Tk()
    root.withdraw() # Hide the empty Tkinter root window.
    print("Opening file selector to choose your SolutionReport.csv file...")
    file_path = filedialog.askopenfilename(
        title="Select SolutionReport.csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit()
    df = load_and_validate_csv(file_path)
    print(f"'{os.path.basename(file_path)}' loaded and validated successfully.")


# --- Data Preparation ---
# Extract hangar dimensions and set animation speed.
HANGAR_WIDTH = df['Hangar_Width'].iloc[0]
HANGAR_LENGTH = df['Hangar_Length'].iloc[0]
FPS = 8 # Frames per second for animations.

# Separate accepted and rejected aircraft for processing.
df_accepted = df[df['Accepted'] == 1].copy()
df_rejected = df[df['Accepted'] == 0].copy()

# Count aircraft for display in table titles.
total_aircraft_count = df['Aircraft_ID'].nunique()
num_accepted_aircraft = df_accepted['Aircraft_ID'].nunique()
num_rejected_aircraft = df_rejected['Aircraft_ID'].nunique()

# Create an "events" DataFrame by treating roll-in and roll-out as separate events.
# This makes it easy to step through time chronologically.
df_in = df_accepted.copy()
df_in['Time'] = df_in['Roll_In']
df_in['Action'] = 'Roll In'

df_out = df_accepted.copy()
df_out['Time'] = df_out['Roll_Out']
df_out['Action'] = 'Roll Out'

events_df = pd.concat([df_in, df_out]).sort_values(by='Time').reset_index(drop=True)

# Prepare the data for the "Accepted Aircraft Events" table.
table_data = pd.DataFrame({
    'Time': events_df['Time'],
    'Action No': range(1, len(events_df) + 1),
    'ID': events_df['Aircraft_ID'],
    'Action': events_df['Action'],
    'x': events_df['X'],
    'y': events_df['Y'],
    'Roll In': events_df['Roll_In'],
    'Roll Out': events_df['Roll_Out'],
    'Arr Delay': events_df['D_Arr'],
    'Dep Delay': events_df['D_Dep']
})

# Prepare the data for the "Rejected Aircraft" table.
rejected_table_data = df_rejected.drop_duplicates(subset=['Aircraft_ID'])[[
    'Aircraft_ID', 'ETA', 'ETD', 'ServT'
]].rename(columns={'Aircraft_ID': 'ID'})

# --- Dynamic Table Row Allocation ---
# Dynamically allocate space for the accepted and rejected tables to prevent empty space.
TOTAL_TABLE_ROWS = 24 # Total vertical space available for both tables.
num_rejected = len(rejected_table_data)

# Cap the rejected table at 10 rows, but use the actual count if smaller.
MAX_REJECTED_ROWS = min(num_rejected, 10)
# The events table gets the remaining space.
MAX_EVENTS_ROWS = TOTAL_TABLE_ROWS - MAX_REJECTED_ROWS

# Get a sorted list of unique event times to step through.
time_points = np.unique(pd.concat([df_accepted['Roll_In'], df_accepted['Roll_Out']]).dropna())
current_index = 0 # Tracks the current position in the time_points array.

# --- Drawing and Update Functions ---

def draw_airplane_shape(ax, x, y, w, l, edgecolor='navy', facecolor='slategray', lw=1.5):
    """Draws a schematic 12-sided polygon representing an airplane."""
    # Ratios define the shape of the aircraft polygon.
    body_width_nose_ratio = 0.3
    body_width_tail_ratio = 0.4
    tail_distance_ratio = 0.2
    wing_tip_y_front_ratio = 0.3
    wing_attach_y_ratio = 0.75
    tail_flare_ratio = 1.2

    # Calculate key coordinates based on the aircraft's position (x,y) and dimensions (w,l).
    cx = x + w / 2
    y_nose = y + l
    y_attach = y + wing_attach_y_ratio * l
    y_wing_bottom = y + tail_distance_ratio * l
    nose_half_w = (body_width_nose_ratio * w) / 2
    tail_half_w = (body_width_tail_ratio * w) / 2
    tail_bottom_half_w = tail_half_w * tail_flare_ratio

    # Define the 12 vertices of the polygon.
    vertices = [
        (cx - nose_half_w, y_nose), (cx + nose_half_w, y_nose), (cx + nose_half_w, y_attach),
        (x + w, y + wing_tip_y_front_ratio * l), (x + w, y_wing_bottom), (cx + tail_half_w, y_wing_bottom),
        (cx + tail_bottom_half_w, y), (cx - tail_bottom_half_w, y), (cx - tail_half_w, y_wing_bottom),
        (x, y_wing_bottom), (x, y + wing_tip_y_front_ratio * l), (cx - nose_half_w, y_attach),
    ]
    polygon = patches.Polygon(vertices, closed=True, edgecolor=edgecolor, facecolor='none', lw=lw, zorder=10)
    ax.add_patch(polygon)
    return [polygon]

import numpy as np # Make sure numpy is imported

def get_font_sizes(aircraft_area, hangar_width, hangar_length, is_for_export=False):
    """
    Determines appropriate font sizes based on aircraft area AND overall hangar dimensions.
    """
    BASE_HANGAR_AREA = 65 * 60
    current_hangar_area = hangar_width * hangar_length

    scale_factor = (BASE_HANGAR_AREA / max(current_hangar_area, 1)) ** 0.25

    if is_for_export:
        if aircraft_area < 360:
            base_title_fontsize = 15
            base_details_fontsize = 10
        elif aircraft_area > 550:
            base_title_fontsize = 17
            base_details_fontsize = 13
        else: # Medium size
            base_title_fontsize = 16
            base_details_fontsize = 11
    else: # For interactive animation
        if aircraft_area < 360:
            base_title_fontsize = 11
            base_details_fontsize = 8
        elif aircraft_area > 550:
            base_title_fontsize = 15
            base_details_fontsize = 11
        else: # Medium size
            base_title_fontsize = 13
            base_details_fontsize = 9

    title_fontsize = max(base_title_fontsize * scale_factor, 7)
    details_fontsize = max(base_details_fontsize * scale_factor, 6)
            
    return title_fontsize, details_fontsize

def draw_static_hangar_for_export(ax, time, df_accepted_data):
    """
    Draws a clean, static image of the hangar at a specific time, optimized for export.
    """
    # Configure axes appearance.
    ax.set_facecolor(COLORS["hangar_bg"])
    ax.set_xlim(-5, HANGAR_WIDTH + 5)
    ax.set_ylim(-5, HANGAR_LENGTH + 15)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6, color=COLORS["grid"])
    ax.set_xticks(np.arange(0, HANGAR_WIDTH + 1, 5))
    ax.set_yticks(np.arange(0, HANGAR_LENGTH + 1, 5))

    # Clean up axis spines for a modern look.
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(COLORS["grid"])

    # Draw hangar border.
    hangar_border = patches.Rectangle((0, 0), HANGAR_WIDTH, HANGAR_LENGTH, linewidth=2, edgecolor=COLORS["border"], facecolor='none')
    ax.add_patch(hangar_border)

    # Add a descriptive title with date and time.
    start_datetime = pd.to_datetime(df['StartDate'].iloc[0])
    event_datetime = start_datetime + pd.to_timedelta(time, unit='h')
    title_text = f"Hangar Status at Time: {time:.2f}h ({event_datetime.strftime('%Y-%m-%d %H:%M')})"
    ax.text(HANGAR_WIDTH / 2, HANGAR_LENGTH + 8, title_text,
            ha='center', va='top', color=COLORS["text"],
            fontsize=22, weight='bold', stretch='semi-expanded')

    # Find the next event time to correctly color aircraft that are about to depart.
    current_time_index = np.where(time_points == time)[0]
    next_time = None
    if len(current_time_index) > 0 and current_time_index[0] < len(time_points) - 1:
        next_time = time_points[current_time_index[0] + 1]

    # Draw all aircraft present at the given time.
    for _, ac in df_accepted_data.iterrows():
        if ac['Roll_In'] <= time < ac['Roll_Out']:
            x, y, w, l = ac['X'], ac['Y'], ac['Width'], ac['Length']

            # Determine colors based on status: arriving (green), departing (red), or static (blue).
            edge_color = COLORS["aircraft_static_edge"]
            face_color = COLORS["aircraft_static_face"]
            border_color = COLORS["airplane_border"]

            is_entering = ac['Roll_In'] == time
            is_exiting_next = next_time is not None and ac['Roll_Out'] == next_time

            if is_entering and is_exiting_next: # Enters now and exits at the very next event.
                face_color = COLORS["aircraft_enter_face"]
                edge_color = COLORS["aircraft_exit_edge"]
                border_color = COLORS["aircraft_exit_edge"]
            elif is_entering: # Newly arrived.
                edge_color = COLORS["aircraft_enter_edge"]
                face_color = COLORS["aircraft_enter_face"]
                border_color = COLORS["aircraft_enter_edge"]
            elif is_exiting_next: # About to depart.
                edge_color = COLORS["aircraft_exit_edge"]
                face_color = COLORS["aircraft_exit_face"]
                border_color = COLORS["aircraft_exit_edge"]

            # Draw the aircraft body, text, and schematic outline.
            rect = patches.Rectangle((x, y), w, l, linewidth=3.0, edgecolor=edge_color, facecolor=face_color, alpha=0.7)
            ax.add_patch(rect)

            details_str = (f"{ac['Width']:.0f}x{ac['Length']:.0f}\n"
                           f"ETA: {ac['ETA']:.1f}\nETD: {ac['ETD']:.1f}\n"
                           f"ServT: {ac['ServT']:.1f}")
            text_y_ref = y + l * 0.6

            # Get font sizes from the dedicated function.
            title_fontsize, details_fontsize = get_font_sizes(w * l, HANGAR_WIDTH, HANGAR_LENGTH, is_for_export=True)

            ax.text(x + w / 2, text_y_ref + l * 0.03, f"#{ac['Aircraft_ID']}", ha='center', va='bottom', fontsize=title_fontsize, weight='bold', zorder=11, color=COLORS["text"], stretch='semi-expanded')
            ax.text(x + w / 2, text_y_ref, details_str, ha='center', va='top', fontsize=details_fontsize, zorder=11, linespacing=1.4, color=COLORS["text"])
            draw_airplane_shape(ax, x, y, w, l, edgecolor=border_color, lw=3.0)


def export_hangar_snapshots(time_points, df_accepted_data, output_folder):
    """
    Iterates through all event times and saves a high-quality PDF of the hangar state for each.
    """
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    total_files = len(time_points)
    print(f"\n--- Starting Vector Export ---")
    print(f"Found {total_files} event times to export to '{output_folder}'.")

    for i, time in enumerate(time_points):
        # Create a new, clean figure for each snapshot to prevent state leakage.
        aspect_ratio = (HANGAR_LENGTH + 20) / (HANGAR_WIDTH + 10)
        temp_fig = plt.figure(figsize=(10, 10 * aspect_ratio))
        temp_ax = temp_fig.add_subplot(111)

        # Draw the hangar state using the dedicated static drawing function.
        draw_static_hangar_for_export(temp_ax, time, df_accepted_data)

        # Define a consistent filename and save the figure.
        filename = f"hangar_state_time_{time:.2f}.pdf".replace('.', '_', 1)
        filepath = os.path.join(output_folder, filename)
        # `bbox_inches='tight'` trims whitespace around the plot.
        temp_fig.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300)

        print(f"  ({i+1}/{total_files}) Saved: {filename}")

        # Close the figure to free up memory.
        plt.close(temp_fig)

    print("--- Vector Export Complete ---\n")


def update_table(ax_table, current_time):
    """Clears and redraws the 'Accepted Aircraft Events' table."""
    ax_table.clear()
    ax_table.axis('off')

    if table_data.empty:
        ax_table.text(0.5, 0.5, "No aircraft have been accepted for service.",
                      ha='center', va='center', fontsize=12.5, style='italic', color='gray')
        return

    # Prepare table headers and title.
    col_labels = table_data.columns.drop('Time').tolist()
    num_cols = len(col_labels)
    title_text = f'Accepted Aircraft Events ({num_accepted_aircraft}/{total_aircraft_count})'
    title_row = [''] * num_cols
    title_row[num_cols // 2] = title_text

    # Determine which row to highlight based on the current time.
    total_rows = len(table_data)
    highlight_indices = table_data[table_data['Time'] == current_time].index
    highlight_idx = highlight_indices[0] if len(highlight_indices) > 0 else -1

    # Implement auto-scrolling to keep the highlighted row visible.
    start_row = 0
    if total_rows > MAX_EVENTS_ROWS:
        start_row = highlight_idx - (MAX_EVENTS_ROWS // 2) + 1
        start_row = max(0, start_row)
        start_row = min(start_row, total_rows - MAX_EVENTS_ROWS)

    # Slice the data to display only the visible rows.
    end_row = start_row + MAX_EVENTS_ROWS
    display_data_rows = table_data.iloc[start_row:end_row]

    # Format numerical columns as strings with two decimal places.
    display_values_df = display_data_rows.loc[:, display_data_rows.columns != 'Time'].copy()
    for col in ['x', 'y', 'Roll In', 'Roll Out', 'Arr Delay', 'Dep Delay']:
        if col in display_values_df.columns:
            display_values_df[col] = display_values_df[col].map('{:.2f}'.format)

    final_cell_text = [title_row, col_labels] + display_values_df.values.tolist()

    # Set cell colors: highlight for current event, gray for past events.
    data_colors_map = np.where(display_data_rows['Time'].values == current_time, COLORS["table_highlight"],
                               np.where(display_data_rows['Time'].values < current_time, COLORS["table_past"], 'white'))
    data_cell_colors = [[c] * num_cols for c in data_colors_map]
    final_cell_colors = [['white'] * num_cols, [COLORS["table_header"]] * num_cols] + data_cell_colors

    # Create and style the table.
    table = ax_table.table(cellText=final_cell_text, cellColours=final_cell_colors, loc='center', colLabels=None)
    table[(0, num_cols // 2)].set_text_props(weight='bold', fontsize=14, stretch='semi-expanded')
    for j in range(num_cols):
        table[(0, j)].set_height(0.09)
        table[(0,j)].set_edgecolor('none')
    for j in range(num_cols):
        table[(1, j)].set_text_props(weight='bold', fontsize=11)
        table[(1, j)].set_height(0.09)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)


def draw_rejected_table(ax_rejected_table):
    """Draws the 'Rejected Aircraft' table."""
    ax_rejected_table.clear()
    ax_rejected_table.axis('off')

    if rejected_table_data.empty:
        ax_rejected_table.text(0.5, 0.5, "All aircraft requests were accepted.",
                               ha='center', va='center', fontsize=12.5, style='italic', color='gray')
        return

    # Prepare table headers and title.
    col_labels = rejected_table_data.columns.tolist()
    num_cols = len(col_labels)
    title_text = f'Rejected Aircraft ({num_rejected_aircraft}/{total_aircraft_count})'
    title_row = [''] * num_cols
    title_position_index = (num_cols // 2) - 1 if num_cols > 1 else 0
    title_row[title_position_index] = title_text

    display_data = rejected_table_data.copy()

    # If there are more rejected aircraft than can be displayed, create a summary row.
    more_row_data = None
    summary_position_index = (num_cols // 2)
    if len(rejected_table_data) > MAX_REJECTED_ROWS:
        display_data = display_data.iloc[:MAX_REJECTED_ROWS]
        remaining_count = len(rejected_table_data) - MAX_REJECTED_ROWS
        remaining_ids = rejected_table_data['ID'].iloc[MAX_REJECTED_ROWS:].astype(str).tolist()

        # Truncate the list of IDs if it's too long.
        max_ids_to_show = 18
        ids_str = ", ".join(remaining_ids[:max_ids_to_show])
        if len(remaining_ids) > max_ids_to_show:
            ids_str += ", ..."

        more_items_str = f"and {remaining_count} more (IDs: {ids_str})"
        more_row_data = [''] * num_cols
        summary_position_index = (num_cols // 2) + 1 if num_cols > 2 else (num_cols // 2)
        more_row_data[summary_position_index] = more_items_str

    # Format numerical columns.
    for col in ['ETA', 'ETD', 'ServT']:
        if col in display_data.columns:
            display_data[col] = display_data[col].map('{:.2f}'.format)

    # Combine all text and color data for the table.
    final_cell_text = [title_row, col_labels] + display_data.values.tolist()
    if more_row_data:
        final_cell_text.append(more_row_data)

    final_cell_colors = [['white'] * num_cols, [COLORS["table_header"]] * num_cols] + \
                        [[COLORS["table_rejected"]] * num_cols] * len(display_data)
    if more_row_data:
        final_cell_colors.append(['white'] * num_cols)

    # Create and style the table.
    table = ax_rejected_table.table(cellText=final_cell_text, cellColours=final_cell_colors, loc='center', colLabels=None)
    table[(0, title_position_index)].set_text_props(weight='bold', fontsize=14, stretch='semi-expanded')
    for j in range(num_cols):
        table[(0, j)].set_height(0.09)
        table[(0, j)].set_edgecolor('none')
    for j in range(num_cols):
        table[(1, j)].set_text_props(weight='bold', fontsize=11)
        table[(1, j)].set_height(0.09)

    # Style the summary row if it exists.
    if more_row_data:
        last_row_idx = len(final_cell_text) - 1
        summary_cell = table[(last_row_idx, summary_position_index)]
        summary_cell.set_text_props(fontsize=10.5, color=COLORS["text"], style='italic', ha='right', va='center')
        for j in range(num_cols):
            table[(last_row_idx, j)].set_edgecolor('none')
            table[(last_row_idx, j)].set_height(0.09)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

# --- Main Execution (Interactive Animation for Local Machine) ---
if not RUNNING_IN_COLAB:
    # If vector export is enabled, prompt for a directory and save the files.
    if EXPORT_VECTORS:
        print("Vector export feature is enabled.")
        root_export = tk.Tk()
        root_export.withdraw()
        output_dir = filedialog.askdirectory(title="Please select a folder to save the vector PDF images")
        root_export.destroy() # Clean up the temporary window.

        if output_dir:
            export_hangar_snapshots(time_points, df_accepted, output_dir)
        else:
            print("Vector export cancelled by user. Continuing with interactive animation.")

    # --- Set up the main figure and layout for the animation ---
    fig = plt.figure(figsize=(25, 12))
    fig.patch.set_facecolor('white')

    # Dynamically calculate the height ratios for the table subplots.
    # This ensures the layout adapts to the amount of data in each table.
    HEADER_UNITS = 2.5
    MIN_TABLE_SPACE = 4.0 # Minimum space to prevent headers from being squashed.

    # Calculate height for the 'Accepted' table.
    accepted_height_ratio = min(len(table_data), MAX_EVENTS_ROWS) + HEADER_UNITS if not table_data.empty else 1

    # Calculate height for the 'Rejected' table.
    if not rejected_table_data.empty:
        displayed_rejected_rows = min(len(rejected_table_data), MAX_REJECTED_ROWS)
        if len(rejected_table_data) > MAX_REJECTED_ROWS:
            displayed_rejected_rows += 1 # Add one row for the '... and x more' summary.
        rejected_height_ratio = max(displayed_rejected_rows, MIN_TABLE_SPACE) + HEADER_UNITS
    else:
        rejected_height_ratio = 1

    dynamic_height_ratios = [accepted_height_ratio, rejected_height_ratio]

    # Create the grid layout with the dynamic height ratios.
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 2.5], height_ratios=dynamic_height_ratios)
    ax_hangar = fig.add_subplot(gs[:, 0])
    ax_table = fig.add_subplot(gs[0, 1])
    ax_rejected_table = fig.add_subplot(gs[1, 1])
    plt.subplots_adjust(bottom=0.2, wspace=0.15, hspace=0.25)

    # Configure the hangar plot appearance.
    ax_hangar.set_facecolor(COLORS["hangar_bg"])
    ax_hangar.set_xlim(-5, HANGAR_WIDTH + 5)
    ax_hangar.set_ylim(-5, HANGAR_LENGTH + 15)
    ax_hangar.set_aspect('equal')
    ax_hangar.grid(True, linestyle='--', alpha=0.6, color=COLORS["grid"])
    ax_hangar.set_xticks(np.arange(0, HANGAR_WIDTH + 1, 5))
    ax_hangar.set_yticks(np.arange(0, HANGAR_LENGTH + 1, 5))
    for spine in ['top', 'right']:
        ax_hangar.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax_hangar.spines[spine].set_color(COLORS["grid"])

    ax_table.axis('off')
    ax_rejected_table.axis('off')

    # Initialize lists and state variables for the animation.
    aircraft_artists = [] # Stores all plotted aircraft elements for easy removal.
    hangar_border = patches.Rectangle((0, 0), HANGAR_WIDTH, HANGAR_LENGTH, linewidth=2, edgecolor=COLORS["border"], facecolor='none')
    ax_hangar.add_patch(hangar_border)
    interrupt_animation = False
    animation_running = False

    def animate_multiple_aircrafts(acs, enters):
        """Animates one or more aircraft entering or exiting the hangar."""
        global interrupt_animation, animation_running
        animation_running = True
        total_frames = FPS
        temp_artists_all = [[] for _ in acs] # Store temporary artists for each aircraft.

        for f in range(total_frames):
            if interrupt_animation: break
            # Clear previous frame's artists.
            for temp_artists in temp_artists_all:
                for a in temp_artists: a.remove()
                temp_artists.clear()

            # Draw each aircraft at its new position for the current frame.
            for idx, ac in enumerate(acs):
                enter = enters[idx]
                x, y, w, l = ac['X'], ac['Y'], ac['Width'], ac['Length']
                ANIMATION_DISTANCE = HANGAR_LENGTH
                start_y, end_y = (HANGAR_LENGTH + ANIMATION_DISTANCE, y) if enter else (y, y + ANIMATION_DISTANCE)
                progress = f / total_frames
                current_y_pos = start_y + (end_y - start_y) * progress

                rect_color, face_color = (COLORS["aircraft_enter_edge"], COLORS["aircraft_enter_face"]) if enter else (COLORS["aircraft_exit_edge"], COLORS["aircraft_exit_face"])
                rect = patches.Rectangle((x, current_y_pos), w, l, linewidth=1.5, edgecolor=rect_color, facecolor=face_color, alpha=0.8)
                ax_hangar.add_patch(rect)

                details_str = (f"{ac['Width']:.0f}x{ac['Length']:.0f}\n"
                               f"ETA: {ac['ETA']:.1f}\nETD: {ac['ETD']:.1f}\n"
                               f"ServT: {ac['ServT']:.1f}")
                text_y_ref = current_y_pos + l * 0.6
                
                # Get font sizes from the dedicated function.
                title_fontsize, details_fontsize = get_font_sizes(w * l, HANGAR_WIDTH, HANGAR_LENGTH, is_for_export=False)

                id_text = ax_hangar.text(x + w / 2, text_y_ref + l * 0.03, f"#{ac['Aircraft_ID']}", ha='center', va='bottom', fontsize=title_fontsize, weight='bold', zorder=11, color=COLORS["text"], stretch='semi-expanded')
                details_text = ax_hangar.text(x + w / 2, text_y_ref, details_str, ha='center', va='top', fontsize=details_fontsize, zorder=11, linespacing=1.4, color=COLORS["text"])
                airplane_shape = draw_airplane_shape(ax_hangar, x, current_y_pos, w, l, edgecolor=rect_color, lw=1.8)
                temp_artists_all[idx].extend([rect, id_text, details_text] + airplane_shape)

            plt.pause(1 / FPS)

        # Clean up all temporary animation artists.
        for temp_artists in temp_artists_all:
            for a in temp_artists: a.remove()

        # After animation, draw the final static state for any aircraft that entered.
        for idx, ac in enumerate(acs):
            if enters[idx]:
                x, y, w, l = ac['X'], ac['Y'], ac['Width'], ac['Length']
                rect = patches.Rectangle((x, y), w, l, linewidth=1.5, edgecolor=COLORS["aircraft_static_edge"], facecolor=COLORS["aircraft_static_face"], alpha=0.7)
                ax_hangar.add_patch(rect)
                details_str = (f"{ac['Width']:.0f}x{ac['Length']:.0f}\n"
                               f"ETA: {ac['ETA']:.1f}\nETD: {ac['ETD']:.1f}\n"
                               f"ServT: {ac['ServT']:.1f}")
                text_y_ref = y + l * 0.6

                # Get font sizes from the dedicated function.
                title_fontsize, details_fontsize = get_font_sizes(w * l, HANGAR_WIDTH, HANGAR_LENGTH, is_for_export=False)
                    
                id_text = ax_hangar.text(x + w / 2, text_y_ref + l * 0.03, f"#{ac['Aircraft_ID']}", ha='center', va='bottom', fontsize=title_fontsize, weight='bold', zorder=11, color=COLORS["text"], stretch='semi-expanded')
                details_text = ax_hangar.text(x + w / 2, text_y_ref, details_str, ha='center', va='top', fontsize=details_fontsize, zorder=11, linespacing=1.4, color=COLORS["text"])
                airplane_shape = draw_airplane_shape(ax_hangar, x, y, w, l, edgecolor=COLORS["airplane_border"], lw=1.5)
                aircraft_artists.extend([rect, id_text, details_text] + airplane_shape)

        animation_running = False
        interrupt_animation = False

    def draw_hangar_state(time, animate=False):
        """
        Updates the hangar view to a specific point in time, animating if requested.
        """
        # Clear all existing aircraft from the plot.
        for artist in aircraft_artists: artist.remove()
        aircraft_artists.clear()
        update_table(ax_table, time)

        aircraft_to_animate = []
        # Iterate through aircraft to determine their state (present, arriving, departing).
        for _, ac in df_accepted.iterrows():
            in_time, out_time = ac['Roll_In'], ac['Roll_Out']
            is_present = in_time <= time < out_time
            is_arriving = in_time == time
            is_departing = out_time == time

            if animate and (is_arriving or is_departing):
                # If animating, add to a list to be handled by the animation function.
                aircraft_to_animate.append((ac, 'in' if is_arriving else 'out'))
                continue

            if is_present:
                # If not animating or aircraft is static, draw it directly.
                x, y, w, l = ac['X'], ac['Y'], ac['Width'], ac['Length']
                rect = patches.Rectangle((x, y), w, l, linewidth=1.5, edgecolor=COLORS["aircraft_static_edge"], facecolor=COLORS["aircraft_static_face"], alpha=0.7)
                ax_hangar.add_patch(rect)
                details_str = (f"{ac['Width']:.0f}x{ac['Length']:.0f}\n"
                               f"ETA: {ac['ETA']:.1f}\nETD: {ac['ETD']:.1f}\n"
                               f"ServT: {ac['ServT']:.1f}")
                text_y_ref = y + l * 0.6

                # Get font sizes from the dedicated function.
                title_fontsize, details_fontsize = get_font_sizes(w * l, HANGAR_WIDTH, HANGAR_LENGTH, is_for_export=False)

                id_text = ax_hangar.text(x + w / 2, text_y_ref + l * 0.03, f"#{ac['Aircraft_ID']}", ha='center', va='bottom', fontsize=title_fontsize, weight='bold', zorder=11, color=COLORS["text"], stretch='semi-expanded')
                details_text = ax_hangar.text(x + w / 2, text_y_ref, details_str, ha='center', va='top', fontsize=details_fontsize, zorder=11, linespacing=1.4, color=COLORS["text"])
                airplane_shape = draw_airplane_shape(ax_hangar, x, y, w, l, edgecolor=COLORS["airplane_border"], lw=1.5)
                aircraft_artists.extend([rect, id_text, details_text] + airplane_shape)

        # Trigger the animation function if there are aircraft to animate.
        if aircraft_to_animate:
            acs_data = [ac for ac, typ in aircraft_to_animate]
            enter_flags = [(typ == 'in') for ac, typ in aircraft_to_animate]
            animate_multiple_aircrafts(acs_data, enter_flags)

    def next_event(event):
        """Event handler for the 'Next' button."""
        global current_index, interrupt_animation
        if animation_running:
            interrupt_animation = True; return # Stop any ongoing animation.
        if current_index < len(time_points) - 1:
            current_index += 1
            current_time = time_points[current_index]
            start_datetime = pd.to_datetime(df['StartDate'].iloc[0])
            event_datetime = start_datetime + pd.to_timedelta(current_time, unit='h')
            ax_hangar.set_title(f"Hangar Status at Time: {current_time:.2f}h ({event_datetime.strftime('%Y-%m-%d %H:%M')})", color=COLORS["text"], fontsize=14.5, weight='bold', stretch='semi-expanded')
            draw_hangar_state(current_time, animate=True)
            fig.canvas.draw_idle()

    def prev_event(event):
        """Event handler for the 'Previous' button."""
        global current_index, interrupt_animation
        if animation_running:
            interrupt_animation = True; return # Stop any ongoing animation.
        if current_index > 0:
            current_index -= 1
            current_time = time_points[current_index]
            start_datetime = pd.to_datetime(df['StartDate'].iloc[0])
            event_datetime = start_datetime + pd.to_timedelta(current_time, unit='h')
            ax_hangar.set_title(f"Hangar Status at Time: {current_time:.2f}h ({event_datetime.strftime('%Y-%m-%d %H:%M')})", color=COLORS["text"], fontsize=14.5, weight='bold', stretch='semi-expanded')
            draw_hangar_state(current_time, animate=False) # No animation when going backward.
            fig.canvas.draw_idle()

    # --- Create and style UI buttons ---
    ax_prev_button = plt.axes([0.3, 0.05, 0.1, 0.05])
    ax_next_button = plt.axes([0.6, 0.05, 0.1, 0.05])
    b_next = Button(ax_next_button, 'Next', color=COLORS["button_bg"], hovercolor=COLORS["button_hover"])
    b_next.on_clicked(next_event)
    b_prev = Button(ax_prev_button, 'Previous', color=COLORS["button_bg"], hovercolor=COLORS["button_hover"])
    b_prev.on_clicked(prev_event)

    for button in [b_next, b_prev]:
        button.label.set_color(COLORS["text"])
        button.label.set_fontsize(11.5)
        button.label.set_stretch('semi-expanded')

    # --- Initial Draw ---
    # Draw the initial state of the hangar and tables at time zero.
    initial_time = time_points[0] if len(time_points) > 0 else 0
    start_datetime = pd.to_datetime(df['StartDate'].iloc[0])
    event_datetime = start_datetime + pd.to_timedelta(initial_time, unit='h')
    ax_hangar.set_title(f"Hangar Status at Time: {initial_time:.2f}h ({event_datetime.strftime('%Y-%m-%d %H:%M')})", color=COLORS["text"], fontsize=14.5, weight='bold', stretch='semi-expanded')
    draw_rejected_table(ax_rejected_table)
    draw_hangar_state(initial_time, animate=False)
    plt.show()

# --- Main Execution (Static Plots for Google Colab) ---
else:
    def draw_static_view(ax_hangar, ax_table, ax_rejected_table, time):
        """
        Draws the complete static view (hangar and tables) for a single time point in Colab.
        """
        update_table(ax_table, time)
        draw_rejected_table(ax_rejected_table)

        # Find the next event time to correctly color aircraft that are about to depart.
        current_time_index = np.where(time_points == time)[0]
        next_time = None
        if len(current_time_index) > 0 and current_time_index[0] < len(time_points) - 1:
            next_time = time_points[current_time_index[0] + 1]

        # Draw all aircraft present at the given time.
        for _, ac in df_accepted.iterrows():
            if ac['Roll_In'] <= time < ac['Roll_Out']:
                x, y, w, l = ac['X'], ac['Y'], ac['Width'], ac['Length']

                # Determine colors based on status: arriving, departing, or static.
                edge_color = COLORS["aircraft_static_edge"]
                face_color = COLORS["aircraft_static_face"]
                border_color = COLORS["airplane_border"]

                is_entering = ac['Roll_In'] == time
                is_exiting_next = next_time is not None and ac['Roll_Out'] == next_time

                if is_entering and is_exiting_next:
                    face_color = COLORS["aircraft_enter_face"]
                    edge_color = COLORS["aircraft_exit_edge"]
                    border_color = COLORS["aircraft_exit_edge"]
                elif is_entering:
                    edge_color = COLORS["aircraft_enter_edge"]
                    face_color = COLORS["aircraft_enter_face"]
                    border_color = COLORS["aircraft_enter_edge"]
                elif is_exiting_next:
                    edge_color = COLORS["aircraft_exit_edge"]
                    face_color = COLORS["aircraft_exit_face"]
                    border_color = COLORS["aircraft_exit_edge"]

                # Draw the aircraft elements.
                rect = patches.Rectangle((x, y), w, l, linewidth=3.0, edgecolor=edge_color, facecolor=face_color, alpha=0.7)
                ax_hangar.add_patch(rect)
                details_str = (f"{ac['Width']:.0f}x{ac['Length']:.0f}\n"
                               f"ETA: {ac['ETA']:.1f}\nETD: {ac['ETD']:.1f}\n"
                               f"ServT: {ac['ServT']:.1f}")
                text_y_ref = y + l * 0.6

                # Get font sizes from the dedicated function.
                title_fontsize, details_fontsize = get_font_sizes(w * l, HANGAR_WIDTH, HANGAR_LENGTH, is_for_export=False)
                    
                id_text = ax_hangar.text(x + w / 2, text_y_ref + l * 0.03, f"#{ac['Aircraft_ID']}", ha='center', va='bottom', fontsize=title_fontsize, weight='bold', zorder=11, color=COLORS["text"], stretch='semi-expanded')
                details_text = ax_hangar.text(x + w / 2, text_y_ref, details_str, ha='center', va='top', fontsize=details_fontsize, zorder=11, linespacing=1.4, color=COLORS["text"])
                draw_airplane_shape(ax_hangar, x, y, w, l, edgecolor=border_color, lw=3.0)

    print(f"Generating {len(time_points)} static plots for each event...")
    # Loop through each event time and generate a separate plot.
    for time in time_points:
        fig = plt.figure(figsize=(25, 12))
        fig.patch.set_facecolor('white')

        # Dynamically calculate height ratios for Colab plots.
        HEADER_UNITS = 2.5
        MIN_TABLE_SPACE = 4.0
        accepted_height_ratio = min(len(table_data), MAX_EVENTS_ROWS) + HEADER_UNITS if not table_data.empty else 1
        if not rejected_table_data.empty:
            displayed_rejected_rows = min(len(rejected_table_data), MAX_REJECTED_ROWS)
            if len(rejected_table_data) > MAX_REJECTED_ROWS:
                displayed_rejected_rows += 1
            rejected_height_ratio = max(displayed_rejected_rows, MIN_TABLE_SPACE) + HEADER_UNITS
        else:
            rejected_height_ratio = 1
        dynamic_height_ratios = [accepted_height_ratio, rejected_height_ratio]

        # Set up the grid and subplots for the current figure.
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 2.5], height_ratios=dynamic_height_ratios)
        ax_hangar = fig.add_subplot(gs[:, 0])
        ax_table = fig.add_subplot(gs[0, 1])
        ax_rejected_table = fig.add_subplot(gs[1, 1])
        plt.subplots_adjust(wspace=0.15, hspace=0.25)

        # Configure and style the hangar plot.
        ax_hangar.set_facecolor(COLORS["hangar_bg"])
        ax_hangar.set_xlim(-5, HANGAR_WIDTH + 5)
        ax_hangar.set_ylim(-5, HANGAR_LENGTH + 15)
        ax_hangar.set_aspect('equal')
        ax_hangar.grid(True, linestyle='--', alpha=0.6, color=COLORS["grid"])
        ax_hangar.set_xticks(np.arange(0, HANGAR_WIDTH + 1, 5))
        ax_hangar.set_yticks(np.arange(0, HANGAR_LENGTH + 1, 5))
        for spine in ['top', 'right']:
            ax_hangar.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax_hangar.spines[spine].set_color(COLORS["grid"])

        # Add the hangar border and title.
        hangar_border = patches.Rectangle((0, 0), HANGAR_WIDTH, HANGAR_LENGTH, linewidth=2, edgecolor=COLORS["border"], facecolor='none')
        ax_hangar.add_patch(hangar_border)
        start_datetime = pd.to_datetime(df['StartDate'].iloc[0])
        event_datetime = start_datetime + pd.to_timedelta(time, unit='h')
        ax_hangar.set_title(f"Hangar Status at Time: {time:.2f}h ({event_datetime.strftime('%Y-%m-%d %H:%M')})", color=COLORS["text"], fontsize=14.5, weight='bold', stretch='semi-expanded')

        # Draw the complete view and display the plot.
        draw_static_view(ax_hangar, ax_table, ax_rejected_table, time)
        plt.show()

    print("Done.")
