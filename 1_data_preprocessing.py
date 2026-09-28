"""
============================================================
LOAN APPROVAL PREDICTION PROJECT
File: 1_data_preprocessing.py
Purpose:
    - Load raw loan dataset
    - Validate dataset structure
    - Clean missing and duplicate records
    - Handle invalid values
    - Perform feature engineering
    - Prepare features and target
    - Build preprocessing pipeline
    - Save cleaned dataset and preprocessing information

Author: Loan Analytics Project
============================================================
"""

# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = PROJECT_ROOT / "loan_data.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_DATA_FILE = REPORTS_DIR / "cleaned_loan_data.csv"
PREPROCESSING_INFO_FILE = REPORTS_DIR / "preprocessing_summary.txt"


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "Loan_ID",
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
    "Loan_Status",
]


# ============================================================
# 4. DATA PREPROCESSOR CLASS
# ============================================================

class LoanDataPreprocessor:
    """
    Complete data preprocessing class for the loan approval
    prediction project.
    """

    def __init__(self, data_path=DATA_FILE):
        """
        Initialize the preprocessing class.

        Parameters
        ----------
        data_path : Path or str
            Location of the raw CSV dataset.
        """

        self.data_path = Path(data_path)

        self.raw_data = None
        self.cleaned_data = None
        self.feature_data = None
        self.target_data = None

        self.preprocessing_pipeline = None

        self.numeric_columns = []
        self.categorical_columns = []

        self.original_shape = None
        self.cleaned_shape = None

        self.duplicate_count = 0
        self.missing_values_before = 0
        self.missing_values_after = 0

    # ========================================================
    # 5. CREATE PROJECT DIRECTORIES
    # ========================================================

    def create_directories(self):
        """
        Create required project directories if they do not exist.
        """

        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)

        except OSError as error:
            raise RuntimeError(
                f"Unable to create project directories: {error}"
            )

    # ========================================================
    # 6. LOAD DATA
    # ========================================================

    def load_data(self):
        """
        Load the loan dataset from CSV.
        """

        print("\n" + "=" * 70)
        print("STEP 1: LOADING LOAN DATASET")
        print("=" * 70)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.data_path}\n\n"
                "Please make sure loan_data.csv exists in the "
                "project root directory."
            )

        try:
            self.raw_data = pd.read_csv(self.data_path)

            self.original_shape = self.raw_data.shape

            print(f"Dataset loaded successfully.")
            print(f"Rows    : {self.raw_data.shape[0]}")
            print(f"Columns : {self.raw_data.shape[1]}")

        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty.")

        except pd.errors.ParserError as error:
            raise ValueError(
                f"Unable to parse CSV file: {error}"
            )

        except Exception as error:
            raise RuntimeError(
                f"Unexpected error while loading dataset: {error}"
            )

        return self.raw_data

    # ========================================================
    # 7. VALIDATE DATASET
    # ========================================================

    def validate_columns(self):
        """
        Check whether all required columns are present.
        """

        print("\n" + "=" * 70)
        print("STEP 2: VALIDATING DATASET")
        print("=" * 70)

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in self.raw_data.columns
        ]

        if missing_columns:
            raise ValueError(
                "The following required columns are missing:\n"
                + ", ".join(missing_columns)
            )

        print("All required columns are present.")

        return True

    # ========================================================
    # 8. INITIAL DATA INFORMATION
    # ========================================================

    def display_initial_information(self):
        """
        Display basic information about the raw dataset.
        """

        print("\nDataset Information")
        print("-" * 70)

        print(f"Shape: {self.raw_data.shape}")

        print("\nColumn Data Types:")
        print(self.raw_data.dtypes)

        self.missing_values_before = int(
            self.raw_data.isnull().sum().sum()
        )

        print(
            f"\nTotal missing values: "
            f"{self.missing_values_before}"
        )

    # ========================================================
    # 9. STANDARDIZE COLUMN VALUES
    # ========================================================

    def standardize_values(self):
        """
        Standardize categorical values to avoid inconsistent
        representations.
        """

        print("\n" + "=" * 70)
        print("STEP 3: STANDARDIZING DATA VALUES")
        print("=" * 70)

        categorical_columns = [
            "Gender",
            "Married",
            "Education",
            "Self_Employed",
            "Property_Area",
        ]

        for column in categorical_columns:

            if column in self.raw_data.columns:

                self.raw_data[column] = (
                    self.raw_data[column]
                    .astype("string")
                    .str.strip()
                )

        # Standardize common values
        replacements = {
            "Gender": {
                "M": "Male",
                "F": "Female",
                "male": "Male",
                "female": "Female",
            },
            "Married": {
                "yes": "Yes",
                "no": "No",
                "YES": "Yes",
                "NO": "No",
            },
            "Education": {
                "graduate": "Graduate",
                "not graduate": "Not Graduate",
            },
            "Self_Employed": {
                "yes": "Yes",
                "no": "No",
            },
        }

        for column, mapping in replacements.items():

            if column in self.raw_data.columns:
                self.raw_data[column] = (
                    self.raw_data[column]
                    .replace(mapping)
                )

        print("Categorical values standardized.")

    # ========================================================
    # 10. HANDLE DUPLICATES
    # ========================================================

    def remove_duplicates(self):
        """
        Remove duplicate rows from the dataset.
        """

        print("\n" + "=" * 70)
        print("STEP 4: REMOVING DUPLICATE RECORDS")
        print("=" * 70)

        before = len(self.raw_data)

        self.raw_data = self.raw_data.drop_duplicates(
            keep="first"
        )

        after = len(self.raw_data)

        self.duplicate_count = before - after

        print(f"Duplicate records removed: {self.duplicate_count}")
        print(f"Remaining records: {after}")

    # ========================================================
    # 11. CONVERT NUMERIC COLUMNS
    # ========================================================

    def convert_numeric_columns(self):
        """
        Convert numerical columns into appropriate numeric
        data types.
        """

        print("\n" + "=" * 70)
        print("STEP 5: CONVERTING NUMERIC COLUMNS")
        print("=" * 70)

        numeric_columns = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
        ]

        for column in numeric_columns:

            if column in self.raw_data.columns:

                self.raw_data[column] = pd.to_numeric(
                    self.raw_data[column],
                    errors="coerce"
                )

        print("Numeric columns converted successfully.")

    # ========================================================
    # 12. HANDLE INVALID NUMERIC VALUES
    # ========================================================

    def handle_invalid_values(self):
        """
        Convert impossible numeric values to NaN so that they
        can later be handled by the imputation pipeline.
        """

        print("\n" + "=" * 70)
        print("STEP 6: HANDLING INVALID VALUES")
        print("=" * 70)

        numeric_validation_rules = {
            "ApplicantIncome": lambda x: x < 0,
            "CoapplicantIncome": lambda x: x < 0,
            "LoanAmount": lambda x: x <= 0,
            "Loan_Amount_Term": lambda x: x <= 0,
            "Credit_History": lambda x: ~x.isin([0, 1]),
        }

        invalid_count = 0

        for column, rule in numeric_validation_rules.items():

            if column not in self.raw_data.columns:
                continue

            try:

                invalid_mask = rule(self.raw_data[column])

                count = int(invalid_mask.sum())

                if count > 0:
                    self.raw_data.loc[
                        invalid_mask, column
                    ] = np.nan

                    invalid_count += count

                    print(
                        f"{column}: {count} invalid values "
                        f"converted to missing."
                    )

            except Exception as error:

                print(
                    f"Warning: Could not validate {column}: "
                    f"{error}"
                )

        print(
            f"Total invalid values handled: {invalid_count}"
        )

    # ========================================================
    # 13. HANDLE DEPENDENTS
    # ========================================================

    def clean_dependents(self):
        """
        Standardize the Dependents column.

        Converts '3+' to 3 and numeric values to numeric type.
        """

        print("\n" + "=" * 70)
        print("STEP 7: CLEANING DEPENDENTS")
        print("=" * 70)

        if "Dependents" not in self.raw_data.columns:
            return

        self.raw_data["Dependents"] = (
            self.raw_data["Dependents"]
            .astype("string")
            .str.strip()
            .replace({"3+": "3"})
        )

        self.raw_data["Dependents"] = pd.to_numeric(
            self.raw_data["Dependents"],
            errors="coerce"
        )

        print("Dependents column cleaned.")

    # ========================================================
    # 14. FEATURE ENGINEERING
    # ========================================================

    def create_features(self):
        """
        Create additional useful features from existing columns.
        """

        print("\n" + "=" * 70)
        print("STEP 8: FEATURE ENGINEERING")
        print("=" * 70)

        data = self.raw_data

        # ----------------------------------------------------
        # Total Income
        # ----------------------------------------------------

        data["TotalIncome"] = (
            data["ApplicantIncome"].fillna(0)
            + data["CoapplicantIncome"].fillna(0)
        )

        # ----------------------------------------------------
        # Loan to Income Ratio
        # ----------------------------------------------------

        data["LoanToIncomeRatio"] = (
            data["LoanAmount"]
            / data["TotalIncome"].replace(0, np.nan)
        )

        # ----------------------------------------------------
        # Applicant Income Log
        # ----------------------------------------------------

        data["ApplicantIncomeLog"] = np.log1p(
            data["ApplicantIncome"].clip(lower=0)
        )

        # ----------------------------------------------------
        # Loan Amount Log
        # ----------------------------------------------------

        data["LoanAmountLog"] = np.log1p(
            data["LoanAmount"].clip(lower=0)
        )

        # ----------------------------------------------------
        # Coapplicant Income Log
        # ----------------------------------------------------

        data["CoapplicantIncomeLog"] = np.log1p(
            data["CoapplicantIncome"].clip(lower=0)
        )

        # ----------------------------------------------------
        # Total Income Log
        # ----------------------------------------------------

        data["TotalIncomeLog"] = np.log1p(
            data["TotalIncome"].clip(lower=0)
        )

        # ----------------------------------------------------
        # Has Coapplicant
        # ----------------------------------------------------

        data["HasCoapplicant"] = (
            data["CoapplicantIncome"].fillna(0) > 0
        ).astype(int)

        # ----------------------------------------------------
        # High Income Indicator
        # ----------------------------------------------------

        income_median = data["TotalIncome"].median()

        data["HighIncome"] = (
            data["TotalIncome"] >= income_median
        ).astype(int)

        # ----------------------------------------------------
        # Loan Amount Category
        # ----------------------------------------------------

        data["LoanAmountCategory"] = pd.cut(
            data["LoanAmount"],
            bins=[
                -np.inf,
                100,
                200,
                300,
                np.inf
            ],
            labels=[
                "Low",
                "Medium",
                "High",
                "Very High"
            ]
        )

        print("Feature engineering completed.")

        print("\nNew features created:")
        print("1. TotalIncome")
        print("2. LoanToIncomeRatio")
        print("3. ApplicantIncomeLog")
        print("4. LoanAmountLog")
        print("5. CoapplicantIncomeLog")
        print("6. TotalIncomeLog")
        print("7. HasCoapplicant")
        print("8. HighIncome")
        print("9. LoanAmountCategory")

    # ========================================================
    # 15. REMOVE IDENTIFIER COLUMN
    # ========================================================

    def remove_identifier(self):
        """
        Remove Loan_ID because it is an identifier and does not
        provide meaningful predictive information.
        """

        if "Loan_ID" in self.raw_data.columns:

            self.raw_data = self.raw_data.drop(
                columns=["Loan_ID"]
            )

            print("\nLoan_ID removed from modeling dataset.")

    # ========================================================
    # 16. SEPARATE TARGET
    # ========================================================

    def prepare_target(self):
        """
        Separate target variable Loan_Status from features.
        """

        print("\n" + "=" * 70)
        print("STEP 9: PREPARING FEATURES AND TARGET")
        print("=" * 70)

        if "Loan_Status" not in self.raw_data.columns:
            raise ValueError(
                "Loan_Status column not found in dataset."
            )

        self.raw_data["Loan_Status"] = (
            self.raw_data["Loan_Status"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        valid_targets = {"Y", "N"}

        invalid_targets = set(
            self.raw_data["Loan_Status"]
            .dropna()
            .unique()
        ) - valid_targets

        if invalid_targets:

            raise ValueError(
                "Invalid Loan_Status values found: "
                f"{invalid_targets}"
            )

        # Target encoding
        self.raw_data["Loan_Status"] = (
            self.raw_data["Loan_Status"]
            .map({"N": 0, "Y": 1})
        )

        self.target_data = self.raw_data["Loan_Status"].copy()

        self.feature_data = self.raw_data.drop(
            columns=["Loan_Status"]
        )

        print(
            f"Feature shape: {self.feature_data.shape}"
        )

        print(
            f"Target shape: {self.target_data.shape}"
        )

        print("\nTarget distribution:")

        print(
            self.target_data.value_counts(
                dropna=False
            )
        )

    # ========================================================
    # 17. IDENTIFY COLUMN TYPES
    # ========================================================

    def identify_column_types(self):
        """
        Identify numerical and categorical columns.
        """

        self.numeric_columns = (
            self.feature_data
            .select_dtypes(
                include=["int64", "float64", "int32", "float32"]
            )
            .columns
            .tolist()
        )

        self.categorical_columns = (
            self.feature_data
            .select_dtypes(
                include=["object", "string", "category"]
            )
            .columns
            .tolist()
        )

        print("\n" + "=" * 70)
        print("COLUMN TYPE IDENTIFICATION")
        print("=" * 70)

        print("\nNumerical columns:")
        for column in self.numeric_columns:
            print(f"  - {column}")

        print("\nCategorical columns:")
        for column in self.categorical_columns:
            print(f"  - {column}")

    # ========================================================
    # 18. BUILD PREPROCESSING PIPELINE
    # ========================================================

    def build_pipeline(self):
        """
        Build preprocessing pipeline for numerical and
        categorical variables.
        """

        print("\n" + "=" * 70)
        print("STEP 10: BUILDING PREPROCESSING PIPELINE")
        print("=" * 70)

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                ),
            ]
        )

        self.preprocessing_pipeline = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    self.numeric_columns
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_columns
                ),
            ],
            remainder="drop"
        )

        print(
            "Preprocessing pipeline created successfully."
        )

        return self.preprocessing_pipeline

    # ========================================================
    # 19. CLEAN DATASET FOR REPORTING
    # ========================================================

    def create_clean_dataset(self):
        """
        Create a human-readable cleaned dataset for reports.
        """

        self.cleaned_data = self.raw_data.copy()

        self.cleaned_shape = self.cleaned_data.shape

        self.missing_values_after = int(
            self.cleaned_data.isnull().sum().sum()
        )

        return self.cleaned_data

    # ========================================================
    # 20. SAVE CLEANED DATA
    # ========================================================

    def save_cleaned_data(self):
        """
        Save cleaned dataset to reports directory.
        """

        try:

            self.cleaned_data.to_csv(
                PROCESSED_DATA_FILE,
                index=False
            )

            print(
                f"\nCleaned dataset saved to:\n"
                f"{PROCESSED_DATA_FILE}"
            )

        except Exception as error:

            raise RuntimeError(
                f"Unable to save cleaned dataset: {error}"
            )

    # ========================================================
    # 21. SAVE PREPROCESSING SUMMARY
    # ========================================================

    def save_summary(self):
        """
        Save preprocessing summary to a text file.
        """

        try:

            with open(
                PREPROCESSING_INFO_FILE,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    "LOAN APPROVAL PREDICTION\n"
                )

                file.write(
                    "DATA PREPROCESSING SUMMARY\n"
                )

                file.write("=" * 70 + "\n\n")

                file.write(
                    f"Original rows: "
                    f"{self.original_shape[0]}\n"
                )

                file.write(
                    f"Original columns: "
                    f"{self.original_shape[1]}\n"
                )

                file.write(
                    f"Duplicate rows removed: "
                    f"{self.duplicate_count}\n"
                )

                file.write(
                    f"Missing values before cleaning: "
                    f"{self.missing_values_before}\n"
                )

                file.write(
                    f"Missing values remaining: "
                    f"{self.missing_values_after}\n"
                )

                file.write(
                    f"Cleaned rows: "
                    f"{self.cleaned_shape[0]}\n"
                )

                file.write(
                    f"Cleaned columns: "
                    f"{self.cleaned_shape[1]}\n\n"
                )

                file.write(
                    "NUMERICAL FEATURES\n"
                )

                file.write("-" * 70 + "\n")

                for column in self.numeric_columns:
                    file.write(
                        f"- {column}\n"
                    )

                file.write(
                    "\nCATEGORICAL FEATURES\n"
                )

                file.write("-" * 70 + "\n")

                for column in self.categorical_columns:
                    file.write(
                        f"- {column}\n"
                    )

                file.write(
                    "\nFEATURE ENGINEERING\n"
                )

                file.write("-" * 70 + "\n")

                engineered_features = [
                    "TotalIncome",
                    "LoanToIncomeRatio",
                    "ApplicantIncomeLog",
                    "LoanAmountLog",
                    "CoapplicantIncomeLog",
                    "TotalIncomeLog",
                    "HasCoapplicant",
                    "HighIncome",
                    "LoanAmountCategory",
                ]

                for feature in engineered_features:
                    file.write(
                        f"- {feature}\n"
                    )

            print(
                f"Preprocessing summary saved to:\n"
                f"{PREPROCESSING_INFO_FILE}"
            )

        except Exception as error:

            print(
                f"Warning: Could not save summary: {error}"
            )

    # ========================================================
    # 22. COMPLETE PREPROCESSING WORKFLOW
    # ========================================================

    def run(self):
        """
        Execute the complete preprocessing workflow.
        """

        print("\n")
        print("=" * 70)
        print("LOAN APPROVAL DATA PREPROCESSING")
        print("=" * 70)

        self.create_directories()

        self.load_data()

        self.validate_columns()

        self.display_initial_information()

        self.standardize_values()

        self.remove_duplicates()

        self.convert_numeric_columns()

        self.handle_invalid_values()

        self.clean_dependents()

        self.create_features()

        self.remove_identifier()

        self.prepare_target()

        self.identify_column_types()

        self.build_pipeline()

        self.create_clean_dataset()

        self.save_cleaned_data()

        self.save_summary()

        print("\n" + "=" * 70)
        print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 70)

        return (
            self.feature_data,
            self.target_data,
            self.preprocessing_pipeline
        )


# ============================================================
# 23. STANDALONE FUNCTION
# ============================================================

def preprocess_loan_data(data_path=DATA_FILE):
    """
    Convenience function for other project files.

    This function will be imported by:
        3_model_training.py
        5_main.py
    """

    processor = LoanDataPreprocessor(
        data_path=data_path
    )

    return processor.run()


# ============================================================
# 24. DATASET SUMMARY FUNCTION
# ============================================================

def get_dataset_summary(data_path=DATA_FILE):
    """
    Return a quick summary of the original dataset.
    """

    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    data = pd.read_csv(data_path)

    summary = {
        "rows": data.shape[0],
        "columns": data.shape[1],
        "missing_values": int(
            data.isnull().sum().sum()
        ),
        "duplicates": int(
            data.duplicated().sum()
        ),
        "memory_usage_mb": round(
            data.memory_usage(
                deep=True
            ).sum() / (1024 ** 2),
            4
        ),
    }

    return summary


# ============================================================
# 25. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    try:

        features, target, pipeline = (
            preprocess_loan_data()
        )

        print("\nPreprocessing output:")
        print(
            f"Features shape : {features.shape}"
        )

        print(
            f"Target shape   : {target.shape}"
        )

        print(
            "\nPreprocessing pipeline is ready "
            "for model training."
        )

    except FileNotFoundError as error:

        print(
            f"\nERROR: {error}"
        )

    except ValueError as error:

        print(
            f"\nDATA ERROR: {error}"
        )

    except Exception as error:

        print(
            f"\nUNEXPECTED ERROR: {error}"
        )