import pandas as pd
import datetime
import os
import time
import tkinter as tk
from tkinter import filedialog
import logging
import glob
import re

# ==============================================================================
# Configuration
# ==============================================================================

# Set to True to process a predefined list of T3 files automatically.
# Set to False to be prompted for a single T3 file.
BATCH_PROCESSING_MODE = True

# ==============================================================================
# Standard Path Setup
# ==============================================================================
# This section dynamically calculates paths relative to the script's location.

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
except NameError:
    project_root = os.getcwd()

# Define the base results directory where all output folders will be created.
BASE_RESULTS_DIR = os.path.join(project_root, 'results')

# Define the specific directory for summary reports (Excel) and logs.
HEURISTIC_RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, 'heuristic')

REST_INTERVAL_SECONDS = 5  # Pause between processing files in batch mode.

# ==============================================================================
# Hangar and Heuristic Parameters
# ==============================================================================
HW = 65        # Hangar Width
HL = 60        # Hangar Length
BUFFER = 5     # Required safety buffer space around each aircraft.
EPSILON_T = 0.1  # Minimum time gap between any two aircraft movements (roll-in/out).
EPSILON_P = 0.001 # Penalty multiplier for an aircraft's distance from the origin (0,0).

def get_base_data(data_dir):
    """
    Loads the base data (T1 and T2) from the parent 'data' directory
    """
    # The data_dir is the specific instance folder (e.g., '.../data/incremental').
    # The base T1 and T2 files are one level up in the main 'data' folder.
    base_data_path = os.path.dirname(data_dir)

    # --- Load T1: Aircraft Model Specifications ---
    t1_path = os.path.join(base_data_path, 'T1.csv')
    try:
        model_df = pd.read_csv(t1_path)
        model_df = model_df.rename(columns={'m': 'M_ID'})
        model_data = model_df.set_index('M_ID').to_dict('index')
        print(f"Successfully loaded aircraft models from {t1_path}")
    except FileNotFoundError:
        print(f"Error: T1.csv not found at path: {t1_path}")
        return None, None

    # --- Load T2: Aircraft Currently in Hangar ---
    t2_path = os.path.join(base_data_path, 'T2.csv')
    try:
        current_ac_df = pd.read_csv(t2_path)
        current_ac_df = current_ac_df.rename(columns={'c': 'id', 'Init_X': 'X_init', 'Init_Y': 'Y_init'})
        current_ac_df['ETA'] = 0.0
        current_ac_df['P_Rej'] = 0
        current_ac_df['P_Arr'] = 0
        current_aircraft_data = current_ac_df.to_dict('records')
        print(f"Successfully loaded current aircraft data from {t2_path}")
    except FileNotFoundError:
        print(f"Error: T2.csv not found at path: {t2_path}")
        return model_data, None

    # Merge model dimensions (Width, Length) into the current aircraft data.
    for ac in current_aircraft_data:
        model_id = ac['M_ID']
        if model_id in model_data:
            ac['Width'] = model_data[model_id]['W']
            ac['Length'] = model_data[model_id]['L']
        else:
            print(f"Warning: Model ID {model_id} for aircraft {ac['id']} not found in T1.csv. Dimensions set to 0.")
            ac['Width'] = 0
            ac['Length'] = 0

    return model_data, current_aircraft_data

def get_future_aircraft_data(t3_path, model_data, data_dir):
    """
    Loads future aircraft data from a specific T3 file.
    """
    if not t3_path:
        root = tk.Tk()
        root.withdraw()
        t3_path = filedialog.askopenfilename(
            title="Select the Future Aircraft CSV file (T3)",
            initialdir=data_dir,
            filetypes=[("CSV Files", "*.csv"), ("All files", "*.*")]
        )

    if not t3_path:
        print("No file selected for future aircraft.")
        return [], None

    future_aircraft_data = []
    try:
        future_ac_df = pd.read_csv(t3_path)
        future_ac_df = future_ac_df.rename(columns={'f': 'id'})
        future_aircraft_data = future_ac_df.to_dict('records')
        print(f"Successfully loaded future aircraft data from {t3_path}")

        for ac in future_aircraft_data:
            model_id = ac['M_ID']
            if model_id in model_data:
                ac['Width'] = model_data[model_id]['W']
                ac['Length'] = model_data[model_id]['L']
            else:
                print(f"Warning: Model ID {model_id} for aircraft {ac['id']} not found in T1.csv. Dimensions set to 0.")
                ac['Width'] = 0
                ac['Length'] = 0
        return future_aircraft_data, t3_path
    except Exception as e:
        print(f"Error reading or processing the selected T3 file '{t3_path}': {e}")
        return [], t3_path

# ==============================================================================
# Heuristic Algorithm Functions
# ==============================================================================

def is_valid_placement(new_ac, x, y, roll_in, schedule):
    """
    Checks if a new aircraft can be placed at (x, y) at a given roll_in time.
    Considers: Hangar bounds, collision with other AC, time gaps, and blocking rules.
    """
    new_w = new_ac['Width']
    new_l = new_ac['Length']
    new_roll_out = roll_in + new_ac['ServT']

    # 1. Hangar Boundary Check
    if not (x >= BUFFER and y >= BUFFER and x + new_w + BUFFER <= HW and y + new_l + BUFFER <= HL):
        return False

    # 2. Check against all previously scheduled aircraft.
    for placed_ac in schedule:
        placed_x, placed_y = placed_ac['X'], placed_ac['Y']
        placed_w, placed_l = placed_ac['Width'], placed_ac['Length']
        placed_roll_in, placed_roll_out = placed_ac['Roll_In'], placed_ac['Roll_Out']

        # 3. Spatio-Temporal Collision Check
        is_time_overlap = (roll_in < placed_roll_out) and (placed_roll_in < new_roll_out)

        if is_time_overlap:
            # Only check for spatial overlap if time overlap exists
            is_x_overlap_buffer = (x < placed_x + placed_w + BUFFER) and (placed_x < x + new_w + BUFFER)
            is_y_overlap_buffer = (y < placed_y + placed_l + BUFFER) and (placed_y < y + new_l + BUFFER)
            if is_x_overlap_buffer and is_y_overlap_buffer:
                return False

        # 4. Time Separation Check (Epsilon-T)
        if abs(roll_in - placed_roll_in) < EPSILON_T: return False
        if abs(roll_in - placed_roll_out) < EPSILON_T: return False
        if abs(new_roll_out - placed_roll_in) < EPSILON_T: return False
        if abs(new_roll_out - placed_roll_out) < EPSILON_T: return False

        # 5. Blocking Rule Check (assuming hangar door is at high Y)
        is_x_path_overlap = (x < placed_x + placed_w) and (placed_x < x + new_w)

        if is_x_path_overlap:
            # Case A: Entry is Blocked
            if placed_y > y:
                if roll_in < placed_roll_out + EPSILON_T:
                    return False

            # Case B: Exit is Blocked
            elif y > placed_y:
                if new_roll_out > placed_roll_out - EPSILON_T:
                    return False

    return True

def find_best_spot(aircraft, schedule):
    """
    Finds the best valid placement (position and time) for a single aircraft.
    """
    potential_roll_in = aircraft['ETA']

    max_roll_in = float('inf')
    if aircraft['P_Arr'] > 0 and aircraft['P_Rej'] > 0:
        max_roll_in = aircraft['ETA'] + (aircraft['P_Rej'] / aircraft['P_Arr'])

    while potential_roll_in <= max_roll_in:
        possible_placements = []

        for y in range(int(HL)):
            for x in range(int(HW)):
                if is_valid_placement(aircraft, x, y, potential_roll_in, schedule):
                    positioning_cost = (x + y)
                    possible_placements.append({'x': x, 'y': y, 'pos_cost': positioning_cost})

        if possible_placements:
            best_spot = min(possible_placements, key=lambda p: p['pos_cost'])

            aircraft['Accepted'] = 1
            aircraft['X'] = best_spot['x']
            aircraft['Y'] = best_spot['y']
            aircraft['Roll_In'] = potential_roll_in
            aircraft['Roll_Out'] = potential_roll_in + aircraft['ServT']
            aircraft['D_Arr'] = potential_roll_in - aircraft['ETA']
            aircraft['D_Dep'] = max(0, aircraft['Roll_Out'] - aircraft['ETD'])
            return aircraft

        potential_roll_in += EPSILON_T

    aircraft['Accepted'] = 0
    aircraft['X'], aircraft['Y'], aircraft['Roll_In'], aircraft['Roll_Out'] = 0, 0, 0, 0
    aircraft['D_Arr'], aircraft['D_Dep'] = 0, 0
    return aircraft

# ==============================================================================
# Main Execution
# ==============================================================================

def run_heuristic_instance(current_aircraft, future_aircraft, t3_file_path, output_dir_path):
    """
    Executes a single run of the heuristic and saves the result to the specified output path.
    """
    start_time = time.time()

    if not future_aircraft:
        print("No future aircraft data provided. Aborting instance run.")
        return None

    future_aircraft_sorted = sorted(
        future_aircraft,
        key=lambda ac: (-ac['P_Rej'], ac['ETA'], ac['ServT'])
    )

    schedule = []
    all_results = []

    # Initialize with a deep copy of current aircraft to avoid modifying the base list
    if current_aircraft:
        schedule = [ac.copy() for ac in current_aircraft]
        for ac in schedule:
            ac['Accepted'] = 1
            ac['X'] = ac.pop('X_init')
            ac['Y'] = ac.pop('Y_init')
            ac['Roll_In'] = 0.0
            ac['Roll_Out'] = ac['ServT']
            ac['D_Arr'] = 0.0
            ac['D_Dep'] = max(0, ac['Roll_Out'] - ac['ETD'])
        all_results.extend(schedule)

    log_func = logging.info if BATCH_PROCESSING_MODE else print

    log_func("\nProcessing future aircraft...")
    for i, aircraft_to_place in enumerate(future_aircraft_sorted):
        log_func(f"  ({i+1}/{len(future_aircraft_sorted)}) Trying to place Aircraft {aircraft_to_place['id']}...")

        processed_aircraft = find_best_spot(aircraft_to_place.copy(), schedule)

        if processed_aircraft['Accepted'] == 1:
            schedule.append(processed_aircraft)
            log_func(f"    -> Placed at (X={processed_aircraft['X']:.2f}, Y={processed_aircraft['Y']:.2f}) | Roll-In: {processed_aircraft['Roll_In']:.2f}")
        else:
            log_func(f"    -> Could not be placed. Rejected.")

        all_results.append(processed_aircraft)

    if not all_results:
        log_func("\nNo aircraft were processed. No output file generated.")
        return None

    df = pd.DataFrame(all_results)

    total_cost_reject = df[df['Accepted'] == 0]['P_Rej'].sum()
    total_cost_d_arr = (df['D_Arr'] * df['P_Arr']).sum()
    total_cost_d_dep = (df['D_Dep'] * df['P_Dep']).sum()
    total_positioning_penalty = (df.loc[df['Accepted']==1, 'X'].sum() + df.loc[df['Accepted']==1, 'Y'].sum()) * EPSILON_P

    total_cost = total_cost_reject + total_cost_d_arr + total_cost_d_dep
    total_z = total_cost + total_positioning_penalty
    processing_time = time.time() - start_time

    all_aircraft_ids = ([ac['id'] for ac in current_aircraft] if current_aircraft else []) + [ac['id'] for ac in future_aircraft]
    df = df.set_index('id').reindex(all_aircraft_ids).reset_index()

    static_data_map = {ac['id']: ac for ac in ((current_aircraft if current_aircraft else []) + future_aircraft)}
    for i, row in df.iterrows():
        if pd.isna(row['Accepted']) or row['Accepted'] == 0:
            original_ac = static_data_map[row['id']]
            for col in ['Width', 'Length', 'ETA', 'ServT', 'ETD', 'P_Rej', 'P_Arr', 'P_Dep']:
                df.loc[i, col] = original_ac.get(col)

    df['Hangar_Width'] = HW
    df['Hangar_Length'] = HL
    df['StartDate'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

    output_df = df.rename(columns={'id': 'Aircraft_ID', 'P_Arr': 'Penalty_ArrivalDelay', 'P_Dep': 'Penalty_DepartureDelay', 'P_Rej': 'Penalty_Reject'})

    final_columns = [
        'Aircraft_ID', 'Accepted', 'Width', 'Length', 'ETA', 'Roll_In', 'X', 'Y',
        'ServT', 'ETD', 'Roll_Out', 'D_Arr', 'D_Dep', 'Penalty_Reject',
        'Penalty_ArrivalDelay', 'Penalty_DepartureDelay', 'Hangar_Width',
        'Hangar_Length', 'StartDate'
    ]
    output_df = output_df[final_columns].fillna(0)

    t3_basename = os.path.splitext(os.path.basename(t3_file_path))[0]
    instance_name = t3_basename.replace('T3-', '')
    output_filename = f'Heuristic_Solution_{instance_name}.csv'
    full_path = os.path.join(output_dir_path, output_filename)

    os.makedirs(output_dir_path, exist_ok=True)
    output_df.to_csv(full_path, index=False, float_format='%.2f')

    relative_solution_path = os.path.relpath(full_path, project_root)
    log_func(f"\nFinal solution for {t3_basename} saved to '{relative_solution_path}'")

    num_accepted_total = int(df['Accepted'].sum())
    num_total_aircraft = len(all_aircraft_ids)

    return {
        "Instance_Name": instance_name,
        "Objective_Value_Z": total_z,
        "Total_Cost": total_cost,
        "Cost_Rejection": total_cost_reject,
        "Cost_Arrival_Delay": total_cost_d_arr,
        "Cost_Departure_Delay": total_cost_d_dep,
        "Cost_Positioning": total_positioning_penalty,
        "Processing_Time_Sec": processing_time,
        "Accepted_Aircraft": num_accepted_total,
        "Total_Aircraft": num_total_aircraft,
    }

def extract_number_from_filename(filepath):
    """
    Extracts the first integer found in a filename string.
    Used for sorting file paths numerically.
    """
    basename = os.path.basename(filepath)
    # \d+ matches one or more digits
    match = re.search(r'\d+', basename)
    if match:
        return int(match.group(0))
    # Return 0 if no number is found, for safe sorting
    return 0

def run_for_mode(data_folder_name):
    """
    This function contains the main logic for running the heuristic for a single data mode.
    """
    # --- Dynamic Path Definitions based on user choice ---
    DATA_DIR = os.path.join(project_root, 'data', data_folder_name)
    OUTPUT_DIR_PATH = os.path.join(HEURISTIC_RESULTS_DIR, data_folder_name)
    LOG_FILE_PATH = os.path.join(HEURISTIC_RESULTS_DIR, f'log_{data_folder_name}.txt')
    SUMMARY_EXCEL_PATH = os.path.join(HEURISTIC_RESULTS_DIR, f'Heuristic_Summary_Report_{data_folder_name}.xlsx')
    
    # Find all matching files first
    t3_file_paths = glob.glob(os.path.join(DATA_DIR, 'T3-*.csv'))
    # Sort the list using the custom key function for numerical sorting
    T3_FILE_LIST = sorted(t3_file_paths, key=extract_number_from_filename)

    # --- Create all necessary directories ---
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    os.makedirs(HEURISTIC_RESULTS_DIR, exist_ok=True)

    all_run_summaries = []

    if not BATCH_PROCESSING_MODE:
        model_data, current_aircraft_base = get_base_data(DATA_DIR)
        if model_data is None: return

        future_aircraft, t3_path = get_future_aircraft_data(None, model_data, DATA_DIR)
        if not t3_path: return

        results = run_heuristic_instance(current_aircraft_base, future_aircraft, t3_path, OUTPUT_DIR_PATH)

        if results:
            all_run_summaries.append(results)
            print("\n" + "="*50)
            print("Final Heuristic Solution Summary:")
            print(f"Total Objective Value (Z): {results['Objective_Value_Z']:,.2f}")
            print(f"Total Cost (Rej+Arr+Dep): {results['Total_Cost']:,.2f}")
            label_width = 28
            print(f"  - {'Total Rejection Cost:'.ljust(label_width)} {results['Cost_Rejection']:,.2f}")
            print(f"  - {'Total Arrival Delay Cost:'.ljust(label_width)} {results['Cost_Arrival_Delay']:,.2f}")
            print(f"  - {'Total Departure Delay Cost:'.ljust(label_width)} {results['Cost_Departure_Delay']:,.2f}")
            print(f"  - {'Total Positioning Cost:'.ljust(label_width)} {results['Cost_Positioning']:,.2f}")
            print("-" * 50)
            print(f"  - {'Algorithm Processing Time:'.ljust(label_width)} {results['Processing_Time_Sec']:.4f} seconds")
            print("="*50 + "\n")
    else:
        # Clear previous handlers to avoid duplicate logging in "Both" mode
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.INFO, format='%(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE_PATH, mode='w'),
                logging.StreamHandler()
            ]
        )

        relative_log_path = os.path.relpath(LOG_FILE_PATH, project_root)

        logging.info("="*80)
        logging.info(f"BATCH PROCESSING MODE ACTIVATED")
        logging.info(f"Dataset selected: {data_folder_name.upper()}")
        logging.info(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Log file will be saved to: {relative_log_path}")
        logging.info("="*80 + "\n")

        model_data, current_aircraft_base = get_base_data(DATA_DIR)
        if model_data is None:
            logging.error("Could not load base T1/T2 data. Aborting batch process.")
            return

        for i, t3_file in enumerate(T3_FILE_LIST):
            logging.info(f"--- RUN {i+1}/{len(T3_FILE_LIST)}: PROCESSING FILE: {os.path.basename(t3_file)} ---")

            future_aircraft, t3_path = get_future_aircraft_data(t3_file, model_data, DATA_DIR)

            if not future_aircraft:
                logging.error(f"Could not load or process future aircraft from {t3_file}. Skipping.")
                logging.info("\n" + "="*80 + "\n")
                continue

            results = run_heuristic_instance(current_aircraft_base, future_aircraft, t3_path, OUTPUT_DIR_PATH)

            if results:
                all_run_summaries.append(results)
                logging.info("\n" + "-"*50)
                logging.info("Heuristic Solution Summary:")
                logging.info(f"Total Objective Value (Z): {results['Objective_Value_Z']:,.2f}")
                logging.info(f"Total Cost (Rej+Arr+Dep): {results['Total_Cost']:,.2f}")
                label_width = 28
                logging.info(f"  - {'Total Rejection Cost:'.ljust(label_width)} {results['Cost_Rejection']:,.2f}")
                logging.info(f"  - {'Total Arrival Delay Cost:'.ljust(label_width)} {results['Cost_Arrival_Delay']:,.2f}")
                logging.info(f"  - {'Total Departure Delay Cost:'.ljust(label_width)} {results['Cost_Departure_Delay']:,.2f}")
                logging.info(f"  - {'Total Positioning Cost:'.ljust(label_width)} {results['Cost_Positioning']:,.2f}")
                logging.info("-" * 50)
                logging.info(f"  - {'Algorithm Processing Time:'.ljust(label_width)} {results['Processing_Time_Sec']:.4f} seconds")
                logging.info("-" * 50)

            logging.info(f"--- END OF RUN {i+1}/{len(T3_FILE_LIST)} ---")

            if i < len(T3_FILE_LIST) - 1:
                logging.info(f"\nPausing for {REST_INTERVAL_SECONDS} seconds...")
                time.sleep(REST_INTERVAL_SECONDS)

            logging.info("\n" + "="*80 + "\n")

        logging.info("Batch processing finished.")

    if all_run_summaries:
        log_func = logging.info if BATCH_PROCESSING_MODE else print
        try:
            new_summary_df = pd.DataFrame(all_run_summaries)

            relative_summary_path = os.path.relpath(SUMMARY_EXCEL_PATH, project_root)

            # Check if the summary file already exists to append data
            if os.path.exists(SUMMARY_EXCEL_PATH):
                log_func(f"\nAppending results to existing summary file: '{relative_summary_path}'")
                existing_summary_df = pd.read_excel(SUMMARY_EXCEL_PATH)
                combined_summary_df = pd.concat([existing_summary_df, new_summary_df], ignore_index=True)
            else:
                log_func("\nCreating new summary report.")
                combined_summary_df = new_summary_df

            combined_summary_df.to_excel(SUMMARY_EXCEL_PATH, index=False, engine='openpyxl')

            log_func("\n" + "="*80)
            log_func(f"SUCCESS: Summary report has been saved/updated in:")
            log_func(f"'{relative_summary_path}'")
            log_func("="*80)
        except Exception as e:
            log_func(f"\nERROR: Could not save the summary Excel report. Reason: {e}")
            log_func("Please ensure you have 'openpyxl' installed (`pip install openpyxl`).")

def main():
    """
    Main function to orchestrate the script's execution.
    """
    # --- Interactive Data Selection ---
    choice = ''
    while choice not in ['1', '2', '3']:
        choice = input("Please select the dataset to run:\n1: incremental\n2: random\n3: Both (run 1, then 2)\nEnter choice (1, 2, or 3): ")

    modes_to_run = []
    if choice == '1':
        modes_to_run.append('incremental')
    elif choice == '2':
        modes_to_run.append('random')
    elif choice == '3':
        modes_to_run.append('incremental')
        modes_to_run.append('random')

    for i, mode in enumerate(modes_to_run):
        # Run the main logic for the selected mode
        run_for_mode(mode)
        
        # If running both modes, pause for 10 seconds between them
        if len(modes_to_run) > 1 and i < len(modes_to_run) - 1:
            print(f"\n{'='*80}\nMODE '{mode.upper()}' COMPLETE. Pausing for 10 seconds before starting the next mode...\n{'='*80}\n")
            time.sleep(10)

    print(f"\n{'='*80}\nALL REQUESTED TASKS ARE COMPLETE.\n{'='*80}")


if __name__ == '__main__':
    main()
