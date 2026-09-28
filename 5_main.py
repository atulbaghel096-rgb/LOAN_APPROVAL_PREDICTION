"""
======================================================================
LOAN APPROVAL PREDICTION PROJECT
File: 5_main.py

Purpose:
    Main controller file for the complete Loan Approval Prediction
    Machine Learning project.

Workflow:
    1. Data Preprocessing
    2. EDA and Statistical Analysis
    3. Model Training and Evaluation
    4. New Applicant Prediction
    5. Project Status / Output Verification

This file provides a simple menu-based interface so that the complete
project can be operated from one Python file.

======================================================================
"""

# ====================================================================
# 1. IMPORT LIBRARIES
# ====================================================================

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime


# ====================================================================
# 2. PROJECT CONFIGURATION
# ====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = PROJECT_ROOT / "loan_data.csv"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"

PREDICTION_DIR = PROJECT_ROOT / "predictions"

MODEL_FILE = MODEL_DIR / "best_loan_model.pkl"

EDA_REPORT = REPORT_DIR / "eda_report.txt"

MODEL_COMPARISON = REPORT_DIR / "model_comparison.csv"

CONFUSION_MATRIX = REPORT_DIR / "confusion_matrix.png"

PREDICTION_RESULTS = (
    PREDICTION_DIR / "prediction_results.csv"
)


# ====================================================================
# 3. PYTHON FILE CONFIGURATION
# ====================================================================

PREPROCESSING_FILE = (
    PROJECT_ROOT / "1_data_preprocessing.py"
)

EDA_FILE = (
    PROJECT_ROOT / "2_eda_analysis.py"
)

TRAINING_FILE = (
    PROJECT_ROOT / "3_model_training.py"
)

PREDICTION_FILE = (
    PROJECT_ROOT / "4_prediction.py"
)


# ====================================================================
# 4. PROJECT HEADER
# ====================================================================

def display_project_header():
    """
    Display the main project heading.
    """

    print("\n")

    print("=" * 78)

    print(
        "             LOAN APPROVAL PREDICTION SYSTEM"
    )

    print("=" * 78)

    print(
        "Professional Machine Learning Project"
    )

    print(
        "Data Analytics + Machine Learning"
    )

    print(
        "Version: 1.0"
    )

    print("=" * 78)

    print()


# ====================================================================
# 5. PROJECT INFORMATION
# ====================================================================

def display_project_information():
    """
    Display basic information about the project.
    """

    print("\n" + "=" * 78)

    print(
        "PROJECT INFORMATION"
    )

    print("=" * 78)

    print(
        "\nProject Name:"
    )

    print(
        "Loan Approval Prediction"
    )

    print(
        "\nProject Type:"
    )

    print(
        "Data Analytics + Machine Learning"
    )

    print(
        "\nProgramming Language:"
    )

    print(
        "Python"
    )

    print(
        "\nMain Libraries:"
    )

    print(
        "Pandas, NumPy, Matplotlib, "
        "Seaborn, Scikit-learn"
    )

    print(
        "\nMachine Learning Models:"
    )

    print(
        "Logistic Regression"
    )

    print(
        "Decision Tree"
    )

    print(
        "Random Forest"
    )

    print(
        "Gradient Boosting"
    )

    print(
        "K-Nearest Neighbors"
    )

    print(
        "\nProject Workflow:"
    )

    print(
        "Data → Cleaning → EDA → "
        "Preprocessing → Training → "
        "Evaluation → Prediction"
    )

    print("=" * 78)


# ====================================================================
# 6. CREATE PROJECT DIRECTORIES
# ====================================================================

def create_project_directories():
    """
    Create required project folders if they do not exist.
    """

    directories = [

        MODEL_DIR,

        REPORT_DIR,

        PREDICTION_DIR
    ]

    print(
        "\nChecking project directories..."
    )

    for directory in directories:

        try:

            directory.mkdir(
                parents=True,
                exist_ok=True
            )

            print(
                f"[OK] {directory.name}/"
            )

        except Exception as error:

            print(
                f"[ERROR] Could not create "
                f"{directory}: {error}"
            )


# ====================================================================
# 7. CHECK DATASET
# ====================================================================

def check_dataset():
    """
    Check whether the loan dataset exists.
    """

    print("\n" + "-" * 78)

    print(
        "DATASET CHECK"
    )

    print("-" * 78)

    if DATA_FILE.exists():

        print(
            "[OK] loan_data.csv found."
        )

        try:

            size = DATA_FILE.stat().st_size

            print(
                f"File size: {size:,} bytes"
            )

        except OSError:

            pass

        return True

    print(
        "[ERROR] loan_data.csv was not found."
    )

    print(
        f"Expected location:\n{DATA_FILE}"
    )

    return False


# ====================================================================
# 8. CHECK PYTHON FILES
# ====================================================================

def check_python_files():
    """
    Check whether all five Python project files exist.
    """

    print("\n" + "-" * 78)

    print(
        "PYTHON FILE CHECK"
    )

    print("-" * 78)

    files = {

        "Data Preprocessing":
            PREPROCESSING_FILE,

        "EDA Analysis":
            EDA_FILE,

        "Model Training":
            TRAINING_FILE,

        "Prediction":
            PREDICTION_FILE,

        "Main Controller":
            PROJECT_ROOT / "5_main.py"
    }

    all_available = True

    for name, file_path in files.items():

        if file_path.exists():

            print(
                f"[OK] {name}: "
                f"{file_path.name}"
            )

        else:

            print(
                f"[MISSING] {name}: "
                f"{file_path.name}"
            )

            all_available = False

    return all_available


# ====================================================================
# 9. CHECK PROJECT OUTPUTS
# ====================================================================

def check_project_outputs():
    """
    Check generated project files.
    """

    print("\n" + "=" * 78)

    print(
        "PROJECT OUTPUT STATUS"
    )

    print("=" * 78)

    outputs = {

        "Dataset":
            DATA_FILE,

        "Trained Model":
            MODEL_FILE,

        "EDA Report":
            EDA_REPORT,

        "Model Comparison":
            MODEL_COMPARISON,

        "Confusion Matrix":
            CONFUSION_MATRIX,

        "Prediction Results":
            PREDICTION_RESULTS
    }

    for name, file_path in outputs.items():

        if file_path.exists():

            try:

                size = file_path.stat().st_size

                print(
                    f"[✓] {name:<22} "
                    f"Available "
                    f"({size:,} bytes)"
                )

            except OSError:

                print(
                    f"[✓] {name:<22} Available"
                )

        else:

            print(
                f"[ ] {name:<22} "
                f"Not generated yet"
            )

    print("=" * 78)


# ====================================================================
# 10. RUN PYTHON SCRIPT
# ====================================================================

def run_python_script(
    script_path,
    script_name
):
    """
    Execute another Python file using the current Python interpreter.
    """

    print("\n")

    print("=" * 78)

    print(
        f"RUNNING: {script_name}"
    )

    print("=" * 78)

    if not script_path.exists():

        print(
            f"\n[ERROR] File not found:"
        )

        print(
            script_path
        )

        return False

    try:

        result = subprocess.run(

            [
                sys.executable,
                str(script_path)
            ],

            cwd=PROJECT_ROOT,

            check=False
        )

        print("\n" + "-" * 78)

        if result.returncode == 0:

            print(
                f"[SUCCESS] {script_name} "
                f"completed successfully."
            )

            print("-" * 78)

            return True

        print(
            f"[FAILED] {script_name} "
            f"returned exit code "
            f"{result.returncode}."
        )

        print("-" * 78)

        return False

    except KeyboardInterrupt:

        print(
            f"\n[STOPPED] {script_name} "
            f"was interrupted."
        )

        return False

    except Exception as error:

        print(
            f"\n[ERROR] Unable to run "
            f"{script_name}."
        )

        print(
            f"Details: {error}"
        )

        return False


# ====================================================================
# 11. RUN DATA PREPROCESSING
# ====================================================================

def run_preprocessing():
    """
    Run the data preprocessing module.
    """

    print("\n")

    print(
        "Starting Data Preprocessing..."
    )

    return run_python_script(

        PREPROCESSING_FILE,

        "1_data_preprocessing.py"
    )


# ====================================================================
# 12. RUN EDA
# ====================================================================

def run_eda():
    """
    Run Exploratory Data Analysis.
    """

    print("\n")

    print(
        "Starting Exploratory Data Analysis..."
    )

    return run_python_script(

        EDA_FILE,

        "2_eda_analysis.py"
    )


# ====================================================================
# 13. RUN MODEL TRAINING
# ====================================================================

def run_training():
    """
    Run model training and evaluation.
    """

    print("\n")

    print(
        "Starting Machine Learning Model Training..."
    )

    return run_python_script(

        TRAINING_FILE,

        "3_model_training.py"
    )


# ====================================================================
# 14. RUN PREDICTION
# ====================================================================

def run_prediction():
    """
    Run interactive applicant prediction.
    """

    print("\n")

    if not MODEL_FILE.exists():

        print(
            "[ERROR] Trained model not found."
        )

        print(
            "\nPlease train the model first "
            "using option 3."
        )

        return False

    return run_python_script(

        PREDICTION_FILE,

        "4_prediction.py"
    )


# ====================================================================
# 15. RUN COMPLETE WORKFLOW
# ====================================================================

def run_complete_workflow():
    """
    Execute the complete project workflow.

    Workflow:
        Dataset
            ↓
        Preprocessing
            ↓
        EDA
            ↓
        Model Training
            ↓
        Model Evaluation
            ↓
        Prediction
    """

    print("\n")

    print("#" * 78)

    print(
        "          COMPLETE LOAN ML WORKFLOW"
    )

    print("#" * 78)

    start_time = datetime.now()

    print(
        f"\nStarted at: "
        f"{start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # ---------------------------------------------------------------
    # STEP 1
    # ---------------------------------------------------------------

    print("\n")

    print("=" * 78)

    print(
        "STEP 1/4 - DATA PREPROCESSING"
    )

    print("=" * 78)

    preprocessing_success = (
        run_preprocessing()
    )

    if not preprocessing_success:

        print(
            "\nWorkflow stopped because "
            "data preprocessing failed."
        )

        return False

    # ---------------------------------------------------------------
    # STEP 2
    # ---------------------------------------------------------------

    print("\n")

    print("=" * 78)

    print(
        "STEP 2/4 - EXPLORATORY DATA ANALYSIS"
    )

    print("=" * 78)

    eda_success = run_eda()

    if not eda_success:

        print(
            "\nWorkflow stopped because "
            "EDA failed."
        )

        return False

    # ---------------------------------------------------------------
    # STEP 3
    # ---------------------------------------------------------------

    print("\n")

    print("=" * 78)

    print(
        "STEP 3/4 - MODEL TRAINING"
    )

    print("=" * 78)

    training_success = run_training()

    if not training_success:

        print(
            "\nWorkflow stopped because "
            "model training failed."
        )

        return False

    # ---------------------------------------------------------------
    # STEP 4
    # ---------------------------------------------------------------

    print("\n")

    print("=" * 78)

    print(
        "STEP 4/4 - PREDICTION SYSTEM"
    )

    print("=" * 78)

    if MODEL_FILE.exists():

        print(
            "\nModel generated successfully."
        )

        print(
            "The prediction module is now ready."
        )

    else:

        print(
            "\n[WARNING] Model file was not found."
        )

    # ---------------------------------------------------------------
    # END
    # ---------------------------------------------------------------

    end_time = datetime.now()

    duration = end_time - start_time

    print("\n")

    print("#" * 78)

    print(
        "             WORKFLOW COMPLETED"
    )

    print("#" * 78)

    print(
        f"\nStarted:  "
        f"{start_time.strftime('%H:%M:%S')}"
    )

    print(
        f"Finished: "
        f"{end_time.strftime('%H:%M:%S')}"
    )

    print(
        f"Duration: {duration}"
    )

    print(
        "\nGenerated outputs:"
    )

    check_project_outputs()

    return True


# ====================================================================
# 16. DISPLAY MENU
# ====================================================================

def display_menu():
    """
    Display the main application menu.
    """

    print("\n")

    print("=" * 78)

    print(
        "                 MAIN MENU"
    )

    print("=" * 78)

    print(
        "\n1. Run Data Preprocessing"
    )

    print(
        "2. Run EDA Analysis"
    )

    print(
        "3. Train and Evaluate ML Models"
    )

    print(
        "4. Predict New Loan Application"
    )

    print(
        "5. Run Complete Project Workflow"
    )

    print(
        "6. Check Project Outputs"
    )

    print(
        "7. Project Information"
    )

    print(
        "8. Check Project Files"
    )

    print(
        "9. Exit"
    )

    print(
        "\n" + "=" * 78
    )


# ====================================================================
# 17. HANDLE MENU OPTION
# ====================================================================

def handle_menu_option(
    choice
):
    """
    Execute the selected menu option.
    """

    if choice == "1":

        run_preprocessing()

    elif choice == "2":

        run_eda()

    elif choice == "3":

        run_training()

    elif choice == "4":

        run_prediction()

    elif choice == "5":

        run_complete_workflow()

    elif choice == "6":

        check_project_outputs()

    elif choice == "7":

        display_project_information()

    elif choice == "8":

        check_python_files()

    elif choice == "9":

        print("\n")

        print(
            "Thank you for using "
            "Loan Approval Prediction System."
        )

        print(
            "Goodbye!"
        )

        return False

    else:

        print(
            "\nInvalid option."
        )

        print(
            "Please select a number from 1 to 9."
        )

    return True


# ====================================================================
# 18. APPLICATION LOOP
# ====================================================================

def application_loop():
    """
    Run the menu continuously until the user exits.
    """

    while True:

        display_menu()

        try:

            choice = input(
                "Enter your choice (1-9): "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nProgram interrupted by user."
            )

            break

        except EOFError:

            print(
                "\n\nInput closed."
            )

            break

        should_continue = (
            handle_menu_option(
                choice
            )
        )

        if not should_continue:

            break

        # ------------------------------------------------------------
        # Pause after each operation
        # ------------------------------------------------------------

        if choice != "9":

            print("\n")

            input(
                "Press ENTER to return "
                "to the main menu..."
            )


# ====================================================================
# 19. SYSTEM CHECK
# ====================================================================

def run_system_check():
    """
    Perform basic project system checks.
    """

    print("\n")

    print("=" * 78)

    print(
        "PROJECT SYSTEM CHECK"
    )

    print("=" * 78)

    create_project_directories()

    print()

    dataset_status = (
        check_dataset()
    )

    print()

    files_status = (
        check_python_files()
    )

    print()

    if dataset_status:

        print(
            "[✓] Dataset is available."
        )

    else:

        print(
            "[!] Dataset needs attention."
        )

    if files_status:

        print(
            "[✓] Python project files are available."
        )

    else:

        print(
            "[!] One or more Python files are missing."
        )

    print(
        "\nSystem check completed."
    )

    print("=" * 78)


# ====================================================================
# 20. MAIN FUNCTION
# ====================================================================

def main():
    """
    Main entry point of the Loan Approval Prediction project.
    """

    try:

        # ------------------------------------------------------------
        # Display project header
        # ------------------------------------------------------------

        display_project_header()

        # ------------------------------------------------------------
        # Create required directories
        # ------------------------------------------------------------

        create_project_directories()

        # ------------------------------------------------------------
        # Perform system check
        # ------------------------------------------------------------

        run_system_check()

        # ------------------------------------------------------------
        # Start application
        # ------------------------------------------------------------

        application_loop()

    except KeyboardInterrupt:

        print(
            "\n\nApplication stopped by user."
        )

    except Exception as error:

        print(
            "\n" + "=" * 78
        )

        print(
            "UNEXPECTED APPLICATION ERROR"
        )

        print(
            "=" * 78
        )

        print(
            f"\nError: {error}"
        )

        print(
            "\nPlease check the error message "
            "and project files."
        )

        print(
            "=" * 78
        )


# ====================================================================
# 21. PROGRAM ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()