# An Optimization Model for Integrated Aircraft Hangar Scheduling and Layout

This repository contains the data and source code for the paper **"An Efficient Continuous-Time Optimization Model for Integrated Aircraft Hangar Scheduling and Layout"**. It includes an implementation of our MILP model using GAMS, a Python implementation of an ACH heuristic, and a visualization tool.

The model was developed as part of a research paper submitted in **2025**.

This project can be accessed on GitHub at: <https://github.com/shayanfp/HangarModelResearch>

### Table of Contents

* [Repository Structure](#-repository-structure)
* [Visualizing the Experiment Results](#-visualizing-the-experiment-results)
* [Getting Started](#-getting-started)
* [How to Run the Code](#-how-to-run-the-code)
* [License](#-license)

## 📂 Repository Structure

The repository is organized as follows:

* **`/data/`**: Contains all input data instances required to run the models, categorized into subdirectories such as `random`, `incremental`, `congestion`, and `case15`.

* **`/src/`**: Contains all source code:

  * `gams_model.gms`: The GAMS implementation of the exact optimization model.

  * `heuristic_ACH.py`: The Python script for our ACH heuristic.

  * `visualization_tool.py`: A Python tool to visualize solution files.

* **`/results/`**: Contains the pre-computed solution files (`.csv`) for all instances.

  * The results for our MILP model are organized in subdirectories that mirror the structure of the `/data` folder (e.g., `/results/random/`, `/results/incremental/`).

  * The results for the heuristic are located under `/results/heuristic/`, which is also subdivided into `random` and `incremental` folders.

## ✅ Visualizing the Experiment Results

**This is the most important section for anyone looking to explore the outcomes of our research. This repository's primary goal is to provide easy access to the pre-computed results from our experiments.**

You can use the `visualization_tool.py` to view any solution `.csv` file from either our MILP model or the ACH heuristic. This tool provides a visual representation of the hangar layout and scheduling over time.

* **Run by providing a file path (recommended):**

  ```bash
  # Example: Visualize an MILP solution for a 'random' instance
  python src/visualization_tool.py --file results/random/SolutionReport_N20_S01.csv
  
  # Example: Visualize a Heuristic solution for an 'incremental' instance
  python src/visualization_tool.py --file results/heuristic/incremental/Heuristic_Solution_INC-20.csv
  ```

* **Run without arguments:**
  This will open a graphical file selector for you to choose the solution file manually.

  ```bash
  python src/visualization_tool.py
  ```

## 🚀 Getting Started

### Prerequisites

Ensure you have the following software installed:

1. **GAMS**: The General Algebraic Modeling System, with the **CPLEX** solver, is required to run our MILP model.

2. **Python 3.13**: Along with the `pip` package manager.

### Installation

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/shayanfp/HangarModelResearch.git](https://github.com/shayanfp/HangarModelResearch.git)
   cd HangarModelResearch
   ```

2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ How to Run the Code

### 1. Running the MILP Model

The MILP model, implemented in GAMS, can be executed directly from the command line by passing arguments. This avoids the need to manually edit the `.gms` file for each run.

**Example Command:**
To run instance `T3-22-01.csv` from the `random` dataset with a 1-hour time limit and 0% optimality gap:

```bash
gams src/gams_model.gms --NVAL=22 --TOTAL_N=22 --FVAL=22 --SAMPLE_ID=1 --FILEPATH="data/random/" --FILENAME="T3-22-01.csv" --reslim=3600 --optcr=0 o="results/gams_temp/output.lst" lf="results/gams_temp/output.log" lo=2
```

**Key Arguments:**

* `--NVAL`, `--TOTAL_N`, `--FVAL`: Instance-specific numeric identifiers.

* `--SAMPLE_ID`: The sample number for the instance (e.g., 1, 2, or 3).

* `--FILEPATH`: The relative path to the directory containing the data files (e.g., `"data/random/"`).

* `--FILENAME`: The full name of the `T3` data file (e.g., `"T3-22-01.csv"`).

* `--reslim`: The solver time limit in seconds (e.g., `3600` for 1 hour).

* `--optcr`: The relative optimality gap (e.g., `0.0` for proving optimality).

* `o`: Specifies the output path for the list (`.lst`) file.

* `lf`: Specifies the output path for the log (`.log`) file.

* `lo=2`: This option is recommended as it ensures a comprehensive log file is generated, which is useful for debugging and reviewing the solver's progress.

### 2. Running the Python Heuristic (ACH)

The heuristic script runs in batch mode, processing all instances found in the `/data` directory.

```bash
python src/heuristic_ACH.py
```

The output solution files (`Heuristic_Solution_*.csv`) and logs will be saved in the relevant subdirectories under `/results/heuristic/`.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.