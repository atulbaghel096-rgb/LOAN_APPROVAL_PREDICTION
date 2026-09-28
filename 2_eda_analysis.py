"""
======================================================================
LOAN APPROVAL PREDICTION PROJECT
File: 2_eda_analysis.py

Purpose:
    - Perform Exploratory Data Analysis (EDA)
    - Analyze applicant demographics
    - Analyze income and loan characteristics
    - Analyze credit history
    - Analyze loan approval patterns
    - Generate statistical summaries
    - Generate professional visualizations
    - Save EDA report
    - Save charts inside reports/eda_charts/

Project Flow:
    loan_data.csv
          |
          v
    1_data_preprocessing.py
          |
          v
    cleaned_loan_data.csv
          |
          v
    2_eda_analysis.py
          |
          v
    EDA Reports + Visualizations

======================================================================
"""

# ====================================================================
# 1. IMPORT LIBRARIES
# ====================================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")


# ====================================================================
# 2. PROJECT PATHS
# ====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

REPORTS_DIR = PROJECT_ROOT / "reports"

CLEANED_DATA_FILE = REPORTS_DIR / "cleaned_loan_data.csv"

EDA_REPORT_FILE = REPORTS_DIR / "eda_report.txt"

EDA_CHARTS_DIR = REPORTS_DIR / "eda_charts"


# ====================================================================
# 3. VISUALIZATION SETTINGS
# ====================================================================

sns.set_theme(
    style="whitegrid",
    context="notebook"
)

plt.rcParams["figure.figsize"] = (10, 6)

plt.rcParams["axes.titlesize"] = 14

plt.rcParams["axes.labelsize"] = 11

plt.rcParams["xtick.labelsize"] = 10

plt.rcParams["ytick.labelsize"] = 10


# ====================================================================
# 4. REQUIRED COLUMNS
# ====================================================================

EXPECTED_COLUMNS = [
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


# ====================================================================
# 5. EDA ANALYZER CLASS
# ====================================================================

class LoanEDAAnalyzer:
    """
    Complete Exploratory Data Analysis class for the
    Loan Approval Prediction project.
    """

    # ----------------------------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------------------------

    def __init__(self, data_path=CLEANED_DATA_FILE):

        self.data_path = Path(data_path)

        self.data = None

        self.report_lines = []

        self.numeric_columns = []

        self.categorical_columns = []

        self.chart_count = 0

    # ----------------------------------------------------------------
    # CREATE DIRECTORIES
    # ----------------------------------------------------------------

    def create_directories(self):
        """
        Create required report and chart directories.
        """

        REPORTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        EDA_CHARTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            "EDA directories created successfully."
        )

    # ----------------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------------

    def load_data(self):
        """
        Load cleaned dataset.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 1: LOADING CLEANED DATA"
        )

        print("=" * 70)

        if not self.data_path.exists():

            raise FileNotFoundError(
                f"\nCleaned dataset not found:\n"
                f"{self.data_path}\n\n"
                f"Please run:\n"
                f"python 1_data_preprocessing.py"
            )

        try:

            self.data = pd.read_csv(
                self.data_path
            )

        except Exception as error:

            raise RuntimeError(
                f"Unable to read cleaned dataset: {error}"
            )

        print(
            f"Dataset loaded successfully."
        )

        print(
            f"Rows    : {self.data.shape[0]}"
        )

        print(
            f"Columns : {self.data.shape[1]}"
        )

        return self.data

    # ----------------------------------------------------------------
    # VALIDATE DATA
    # ----------------------------------------------------------------

    def validate_data(self):
        """
        Validate important columns before performing EDA.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 2: VALIDATING EDA DATA"
        )

        print("=" * 70)

        missing_columns = [
            column
            for column in EXPECTED_COLUMNS
            if column not in self.data.columns
        ]

        if missing_columns:

            print(
                "\nWarning: Some expected columns are missing:"
            )

            for column in missing_columns:
                print(
                    f"  - {column}"
                )

        else:

            print(
                "All expected columns are available."
            )

    # ----------------------------------------------------------------
    # IDENTIFY COLUMN TYPES
    # ----------------------------------------------------------------

    def identify_column_types(self):
        """
        Identify numerical and categorical columns.
        """

        self.numeric_columns = (
            self.data
            .select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        self.categorical_columns = (
            self.data
            .select_dtypes(
                include=[
                    "object",
                    "string",
                    "category"
                ]
            )
            .columns
            .tolist()
        )

        print(
            "\nNumerical columns:"
        )

        for column in self.numeric_columns:

            print(
                f"  - {column}"
            )

        print(
            "\nCategorical columns:"
        )

        for column in self.categorical_columns:

            print(
                f"  - {column}"
            )

    # ----------------------------------------------------------------
    # ADD REPORT LINE
    # ----------------------------------------------------------------

    def add_report(self, text=""):
        """
        Add text to the EDA report.
        """

        self.report_lines.append(
            str(text)
        )

    # ----------------------------------------------------------------
    # DATASET OVERVIEW
    # ----------------------------------------------------------------

    def analyze_dataset_overview(self):
        """
        Generate general dataset information.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 3: DATASET OVERVIEW"
        )

        print("=" * 70)

        rows = self.data.shape[0]

        columns = self.data.shape[1]

        missing_values = int(
            self.data.isnull()
            .sum()
            .sum()
        )

        duplicate_rows = int(
            self.data.duplicated()
            .sum()
        )

        self.add_report(
            "=" * 80
        )

        self.add_report(
            "LOAN APPROVAL PREDICTION - EDA REPORT"
        )

        self.add_report(
            "=" * 80
        )

        self.add_report(
            ""
        )

        self.add_report(
            "1. DATASET OVERVIEW"
        )

        self.add_report(
            "-" * 80
        )

        self.add_report(
            f"Number of rows      : {rows}"
        )

        self.add_report(
            f"Number of columns   : {columns}"
        )

        self.add_report(
            f"Missing values      : {missing_values}"
        )

        self.add_report(
            f"Duplicate rows      : {duplicate_rows}"
        )

        self.add_report(
            ""
        )

        print(
            f"Rows: {rows}"
        )

        print(
            f"Columns: {columns}"
        )

        print(
            f"Missing values: {missing_values}"
        )

        print(
            f"Duplicate rows: {duplicate_rows}"
        )

    # ----------------------------------------------------------------
    # DATA TYPES ANALYSIS
    # ----------------------------------------------------------------

    def analyze_data_types(self):
        """
        Analyze data types.
        """

        print("\nAnalyzing data types...")

        self.add_report(
            "2. DATA TYPES"
        )

        self.add_report(
            "-" * 80
        )

        for column in self.data.columns:

            data_type = str(
                self.data[column].dtype
            )

            self.add_report(
                f"{column:<30} {data_type}"
            )

        self.add_report(
            ""
        )

    # ----------------------------------------------------------------
    # MISSING VALUE ANALYSIS
    # ----------------------------------------------------------------

    def analyze_missing_values(self):
        """
        Analyze missing values by column.
        """

        print(
            "\nAnalyzing missing values..."
        )

        missing = (
            self.data
            .isnull()
            .sum()
        )

        missing = (
            missing[
                missing > 0
            ]
            .sort_values(
                ascending=False
            )
        )

        self.add_report(
            "3. MISSING VALUE ANALYSIS"
        )

        self.add_report(
            "-" * 80
        )

        if missing.empty:

            self.add_report(
                "No missing values found."
            )

            print(
                "No missing values found."
            )

        else:

            for column, count in missing.items():

                percentage = (
                    count
                    / len(self.data)
                ) * 100

                self.add_report(
                    f"{column}: "
                    f"{count} "
                    f"({percentage:.2f}%)"
                )

                print(
                    f"{column}: "
                    f"{count} "
                    f"({percentage:.2f}%)"
                )

        self.add_report(
            ""
        )

    # ----------------------------------------------------------------
    # NUMERICAL STATISTICS
    # ----------------------------------------------------------------

    def analyze_numerical_statistics(self):
        """
        Generate descriptive statistics for numerical columns.
        """

        print(
            "\nCalculating numerical statistics..."
        )

        numeric_data = (
            self.data
            .select_dtypes(
                include=np.number
            )
        )

        statistics = numeric_data.describe().T

        self.add_report(
            "4. NUMERICAL STATISTICS"
        )

        self.add_report(
            "-" * 80
        )

        self.add_report(
            statistics
            .round(2)
            .to_string()
        )

        self.add_report(
            ""
        )

    # ----------------------------------------------------------------
    # CATEGORICAL STATISTICS
    # ----------------------------------------------------------------

    def analyze_categorical_statistics(self):
        """
        Generate frequency statistics for categorical columns.
        """

        print(
            "\nCalculating categorical statistics..."
        )

        self.add_report(
            "5. CATEGORICAL VARIABLE ANALYSIS"
        )

        self.add_report(
            "-" * 80
        )

        for column in self.categorical_columns:

            self.add_report(
                f"\n{column}"
            )

            self.add_report(
                "." * 60
            )

            value_counts = (
                self.data[column]
                .value_counts(
                    dropna=False
                )
            )

            for value, count in value_counts.items():

                percentage = (
                    count
                    / len(self.data)
                ) * 100

                self.add_report(
                    f"{value}: "
                    f"{count} "
                    f"({percentage:.2f}%)"
                )

        self.add_report(
            ""
        )

    # ----------------------------------------------------------------
    # LOAN APPROVAL ANALYSIS
    # ----------------------------------------------------------------

    def analyze_loan_status(self):
        """
        Analyze approved and rejected loan applications.
        """

        print(
            "\nAnalyzing loan approval status..."
        )

        self.add_report(
            "6. LOAN APPROVAL ANALYSIS"
        )

        self.add_report(
            "-" * 80
        )

        if "Loan_Status" not in self.data.columns:

            self.add_report(
                "Loan_Status column not available."
            )

            return

        status_counts = (
            self.data["Loan_Status"]
            .value_counts(
                dropna=False
            )
        )

        total = len(self.data)

        for status, count in status_counts.items():

            percentage = (
                count
                / total
            ) * 100

            if status == 1:

                label = "Approved"

            elif status == 0:

                label = "Rejected"

            else:

                label = str(status)

            self.add_report(
                f"{label}: "
                f"{count} "
                f"({percentage:.2f}%)"
            )

            print(
                f"{label}: "
                f"{count} "
                f"({percentage:.2f}%)"
            )

        self.add_report(
            ""
        )

    # ----------------------------------------------------------------
    # APPROVAL RATE
    # ----------------------------------------------------------------

    def calculate_approval_rate(self):
        """
        Calculate overall loan approval rate.
        """

        if "Loan_Status" not in self.data.columns:

            return None

        approval_rate = (
            self.data["Loan_Status"]
            .mean()
            * 100
        )

        self.add_report(
            f"Overall Loan Approval Rate: "
            f"{approval_rate:.2f}%"
        )

        self.add_report(
            ""
        )

        print(
            f"Overall approval rate: "
            f"{approval_rate:.2f}%"
        )

        return approval_rate

    # ----------------------------------------------------------------
    # GROUP ANALYSIS FUNCTION
    # ----------------------------------------------------------------

    def analyze_approval_by_column(
        self,
        column
    ):
        """
        Analyze loan approval rate by a categorical variable.
        """

        if column not in self.data.columns:

            return

        if "Loan_Status" not in self.data.columns:

            return

        grouped = (
            self.data
            .groupby(column)[
                "Loan_Status"
            ]
            .agg(
                Applications="count",
                Approved="sum",
                Approval_Rate="mean"
            )
        )

        grouped["Approval_Rate"] = (
            grouped["Approval_Rate"]
            * 100
        )

        self.add_report(
            f"Approval Rate by {column}"
        )

        self.add_report(
            "-" * 60
        )

        self.add_report(
            grouped
            .round(2)
            .to_string()
        )

        self.add_report(
            ""
        )

    # ----------------------------------------------------------------
    # DEMOGRAPHIC ANALYSIS
    # ----------------------------------------------------------------

    def analyze_demographics(self):
        """
        Analyze approval by applicant demographics.
        """

        print(
            "\nAnalyzing applicant demographics..."
        )

        self.add_report(
            "7. DEMOGRAPHIC APPROVAL ANALYSIS"
        )

        self.add_report(
            "-" * 80
        )

        demographic_columns = [
            "Gender",
            "Married",
            "Education",
            "Self_Employed",
            "Property_Area",
            "Dependents",
        ]

        for column in demographic_columns:

            self.analyze_approval_by_column(
                column
            )

    # ----------------------------------------------------------------
    # CREDIT HISTORY ANALYSIS
    # ----------------------------------------------------------------

    def analyze_credit_history(self):
        """
        Analyze the relationship between credit history
        and loan approval.
        """

        print(
            "\nAnalyzing credit history..."
        )

        self.add_report(
            "8. CREDIT HISTORY ANALYSIS"
        )

        self.add_report(
            "-" * 80
        )

        if "Credit_History" not in self.data.columns:

            self.add_report(
                "Credit_History not available."
            )

            return

        grouped = (
            self.data
            .groupby("Credit_History")[
                "Loan_Status"
            ]
            .agg(
                Applications="count",
                Approved="sum",
                Approval_Rate="mean"
            )
        )

        grouped["Approval_Rate"] = (
            grouped["Approval_Rate"]
            * 100
        )

        self.add_report(
            grouped
            .round(2)
            .to_string()
        )

        self.add_report(
            ""
        )

    # ----------------------------------------------------------------
    # INCOME ANALYSIS
    # ----------------------------------------------------------------

    def analyze_income(self):
        """
        Analyze applicant and coapplicant income.
        """

        print(
            "\nAnalyzing income patterns..."
        )

        self.add_report(
            "9. INCOME ANALYSIS"
        )

        self.add_report(
            "-" * 80
        )

        income_columns = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "TotalIncome",
        ]

        available_columns = [
            column
            for column in income_columns
            if column in self.data.columns
        ]

        if not available_columns:

            self.add_report(
                "Income columns unavailable."
            )

            return

        income_statistics = (
            self.data[
                available_columns
            ]
            .describe()
            .T
        )

        self.add_report(
            income_statistics
            .round(2)
            .to_string()
        )

        self.add_report(
            ""
        )

    # ----------------------------------------------------------------
    # LOAN AMOUNT ANALYSIS
    # ----------------------------------------------------------------

    def analyze_loan_amount(self):
        """
        Analyze loan amount distribution.
        """

        print(
            "\nAnalyzing loan amounts..."
        )

        self.add_report(
            "10. LOAN AMOUNT ANALYSIS"
        )

        self.add_report(
            "-" * 80
        )

        if "LoanAmount" not in self.data.columns:

            self.add_report(
                "LoanAmount column unavailable."
            )

            return

        loan_statistics = (
            self.data["LoanAmount"]
            .describe()
        )

        for metric, value in loan_statistics.items():

            self.add_report(
                f"{metric}: {value:.2f}"
            )

        self.add_report(
            ""
        )

    # ----------------------------------------------------------------
    # CORRELATION ANALYSIS
    # ----------------------------------------------------------------

    def analyze_correlations(self):
        """
        Analyze numerical correlations.
        """

        print(
            "\nCalculating correlations..."
        )

        self.add_report(
            "11. CORRELATION ANALYSIS"
        )

        self.add_report(
            "-" * 80
        )

        numeric_data = (
            self.data
            .select_dtypes(
                include=np.number
            )
        )

        if numeric_data.empty:

            self.add_report(
                "No numerical variables available."
            )

            return

        correlation_matrix = (
            numeric_data.corr()
        )

        self.add_report(
            correlation_matrix
            .round(3)
            .to_string()
        )

        self.add_report(
            ""
        )

    # =================================================================
    # VISUALIZATION SECTION
    # =================================================================

    # ----------------------------------------------------------------
    # SAVE FIGURE
    # ----------------------------------------------------------------

    def save_figure(
        self,
        filename,
        title=None
    ):
        """
        Save current matplotlib figure.
        """

        file_path = (
            EDA_CHARTS_DIR
            / filename
        )

        if title:

            plt.title(
                title,
                fontsize=14,
                fontweight="bold"
            )

        plt.tight_layout()

        plt.savefig(
            file_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        self.chart_count += 1

        print(
            f"Chart saved: {file_path.name}"
        )

    # ----------------------------------------------------------------
    # CHART 1 - LOAN STATUS
    # ----------------------------------------------------------------

    def plot_loan_status(self):
        """
        Generate loan approval status chart.
        """

        if "Loan_Status" not in self.data.columns:

            return

        plt.figure()

        status_data = (
            self.data["Loan_Status"]
            .map({
                0: "Rejected",
                1: "Approved"
            })
            .value_counts()
        )

        sns.barplot(
            x=status_data.index,
            y=status_data.values
        )

        plt.xlabel(
            "Loan Decision"
        )

        plt.ylabel(
            "Number of Applications"
        )

        self.save_figure(
            "01_loan_status_distribution.png",
            "Loan Approval Distribution"
        )

    # ----------------------------------------------------------------
    # CHART 2 - GENDER
    # ----------------------------------------------------------------

    def plot_gender_distribution(self):
        """
        Plot gender distribution.
        """

        if "Gender" not in self.data.columns:

            return

        plt.figure()

        sns.countplot(
            data=self.data,
            x="Gender"
        )

        plt.xlabel(
            "Gender"
        )

        plt.ylabel(
            "Number of Applicants"
        )

        self.save_figure(
            "02_gender_distribution.png",
            "Applicant Gender Distribution"
        )

    # ----------------------------------------------------------------
    # CHART 3 - EDUCATION
    # ----------------------------------------------------------------

    def plot_education_distribution(self):
        """
        Plot education distribution.
        """

        if "Education" not in self.data.columns:

            return

        plt.figure()

        sns.countplot(
            data=self.data,
            x="Education"
        )

        plt.xlabel(
            "Education"
        )

        plt.ylabel(
            "Number of Applicants"
        )

        plt.xticks(
            rotation=10
        )

        self.save_figure(
            "03_education_distribution.png",
            "Applicant Education Distribution"
        )

    # ----------------------------------------------------------------
    # CHART 4 - PROPERTY AREA
    # ----------------------------------------------------------------

    def plot_property_area(self):
        """
        Plot property area distribution.
        """

        if "Property_Area" not in self.data.columns:

            return

        plt.figure()

        sns.countplot(
            data=self.data,
            x="Property_Area"
        )

        plt.xlabel(
            "Property Area"
        )

        plt.ylabel(
            "Number of Applicants"
        )

        self.save_figure(
            "04_property_area_distribution.png",
            "Property Area Distribution"
        )

    # ----------------------------------------------------------------
    # CHART 5 - APPLICANT INCOME
    # ----------------------------------------------------------------

    def plot_applicant_income(self):
        """
        Plot applicant income distribution.
        """

        if "ApplicantIncome" not in self.data.columns:

            return

        plt.figure()

        sns.histplot(
            data=self.data,
            x="ApplicantIncome",
            kde=True
        )

        plt.xlabel(
            "Applicant Income"
        )

        plt.ylabel(
            "Frequency"
        )

        self.save_figure(
            "05_applicant_income_distribution.png",
            "Applicant Income Distribution"
        )

    # ----------------------------------------------------------------
    # CHART 6 - LOAN AMOUNT
    # ----------------------------------------------------------------

    def plot_loan_amount(self):
        """
        Plot loan amount distribution.
        """

        if "LoanAmount" not in self.data.columns:

            return

        plt.figure()

        sns.histplot(
            data=self.data,
            x="LoanAmount",
            kde=True
        )

        plt.xlabel(
            "Loan Amount"
        )

        plt.ylabel(
            "Frequency"
        )

        self.save_figure(
            "06_loan_amount_distribution.png",
            "Loan Amount Distribution"
        )

    # ----------------------------------------------------------------
    # CHART 7 - CREDIT HISTORY
    # ----------------------------------------------------------------

    def plot_credit_history(self):
        """
        Plot credit history against loan status.
        """

        if (
            "Credit_History"
            not in self.data.columns
            or "Loan_Status"
            not in self.data.columns
        ):

            return

        plt.figure()

        sns.countplot(
            data=self.data,
            x="Credit_History",
            hue="Loan_Status"
        )

        plt.xlabel(
            "Credit History"
        )

        plt.ylabel(
            "Number of Applications"
        )

        plt.legend(
            title="Loan Status",
            labels=[
                "Rejected",
                "Approved"
            ]
        )

        self.save_figure(
            "07_credit_history_vs_approval.png",
            "Credit History vs Loan Approval"
        )

    # ----------------------------------------------------------------
    # CHART 8 - EDUCATION VS APPROVAL
    # ----------------------------------------------------------------

    def plot_education_vs_approval(self):
        """
        Plot education against loan approval.
        """

        if (
            "Education"
            not in self.data.columns
            or "Loan_Status"
            not in self.data.columns
        ):

            return

        plt.figure()

        sns.countplot(
            data=self.data,
            x="Education",
            hue="Loan_Status"
        )

        plt.xlabel(
            "Education"
        )

        plt.ylabel(
            "Number of Applications"
        )

        plt.xticks(
            rotation=10
        )

        plt.legend(
            title="Loan Status",
            labels=[
                "Rejected",
                "Approved"
            ]
        )

        self.save_figure(
            "08_education_vs_approval.png",
            "Education vs Loan Approval"
        )

    # ----------------------------------------------------------------
    # CHART 9 - PROPERTY AREA VS APPROVAL
    # ----------------------------------------------------------------

    def plot_property_vs_approval(self):
        """
        Plot property area against approval.
        """

        if (
            "Property_Area"
            not in self.data.columns
            or "Loan_Status"
            not in self.data.columns
        ):

            return

        plt.figure()

        sns.countplot(
            data=self.data,
            x="Property_Area",
            hue="Loan_Status"
        )

        plt.xlabel(
            "Property Area"
        )

        plt.ylabel(
            "Number of Applications"
        )

        plt.legend(
            title="Loan Status",
            labels=[
                "Rejected",
                "Approved"
            ]
        )

        self.save_figure(
            "09_property_area_vs_approval.png",
            "Property Area vs Loan Approval"
        )

    # ----------------------------------------------------------------
    # CHART 10 - INCOME VS LOAN AMOUNT
    # ----------------------------------------------------------------

    def plot_income_vs_loan_amount(self):
        """
        Plot applicant income against loan amount.
        """

        if (
            "ApplicantIncome"
            not in self.data.columns
            or "LoanAmount"
            not in self.data.columns
        ):

            return

        plt.figure()

        sns.scatterplot(
            data=self.data,
            x="ApplicantIncome",
            y="LoanAmount",
            hue="Loan_Status"
        )

        plt.xlabel(
            "Applicant Income"
        )

        plt.ylabel(
            "Loan Amount"
        )

        plt.legend(
            title="Loan Status",
            labels=[
                "Rejected",
                "Approved"
            ]
        )

        self.save_figure(
            "10_income_vs_loan_amount.png",
            "Applicant Income vs Loan Amount"
        )

    # ----------------------------------------------------------------
    # CHART 11 - TOTAL INCOME VS APPROVAL
    # ----------------------------------------------------------------

    def plot_total_income_vs_approval(self):
        """
        Plot total income against loan status.
        """

        if (
            "TotalIncome"
            not in self.data.columns
            or "Loan_Status"
            not in self.data.columns
        ):

            return

        plt.figure()

        sns.boxplot(
            data=self.data,
            x="Loan_Status",
            y="TotalIncome"
        )

        plt.xlabel(
            "Loan Status"
        )

        plt.ylabel(
            "Total Income"
        )

        plt.xticks(
            [0, 1],
            ["Rejected", "Approved"]
        )

        self.save_figure(
            "11_total_income_vs_approval.png",
            "Total Income vs Loan Approval"
        )

    # ----------------------------------------------------------------
    # CHART 12 - LOAN AMOUNT VS APPROVAL
    # ----------------------------------------------------------------

    def plot_loan_amount_vs_approval(self):
        """
        Plot loan amount against loan status.
        """

        if (
            "LoanAmount"
            not in self.data.columns
            or "Loan_Status"
            not in self.data.columns
        ):

            return

        plt.figure()

        sns.boxplot(
            data=self.data,
            x="Loan_Status",
            y="LoanAmount"
        )

        plt.xlabel(
            "Loan Status"
        )

        plt.ylabel(
            "Loan Amount"
        )

        plt.xticks(
            [0, 1],
            ["Rejected", "Approved"]
        )

        self.save_figure(
            "12_loan_amount_vs_approval.png",
            "Loan Amount vs Loan Approval"
        )

    # ----------------------------------------------------------------
    # CHART 13 - CORRELATION HEATMAP
    # ----------------------------------------------------------------

    def plot_correlation_heatmap(self):
        """
        Generate numerical correlation heatmap.
        """

        numeric_data = (
            self.data
            .select_dtypes(
                include=np.number
            )
        )

        if numeric_data.empty:

            return

        correlation = (
            numeric_data.corr()
        )

        plt.figure(
            figsize=(13, 9)
        )

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cmap="Blues"
        )

        plt.xlabel(
            "Features"
        )

        plt.ylabel(
            "Features"
        )

        self.save_figure(
            "13_correlation_heatmap.png",
            "Numerical Feature Correlation Heatmap"
        )

    # ----------------------------------------------------------------
    # CHART 14 - DEPENDENTS
    # ----------------------------------------------------------------

    def plot_dependents(self):
        """
        Plot number of dependents.
        """

        if "Dependents" not in self.data.columns:

            return

        plt.figure()

        sns.countplot(
            data=self.data,
            x="Dependents"
        )

        plt.xlabel(
            "Number of Dependents"
        )

        plt.ylabel(
            "Number of Applicants"
        )

        self.save_figure(
            "14_dependents_distribution.png",
            "Applicant Dependents Distribution"
        )

    # ----------------------------------------------------------------
    # CHART 15 - SELF EMPLOYED
    # ----------------------------------------------------------------

    def plot_self_employed(self):
        """
        Plot self-employment distribution.
        """

        if "Self_Employed" not in self.data.columns:

            return

        plt.figure()

        sns.countplot(
            data=self.data,
            x="Self_Employed"
        )

        plt.xlabel(
            "Self Employed"
        )

        plt.ylabel(
            "Number of Applicants"
        )

        self.save_figure(
            "15_self_employed_distribution.png",
            "Self Employment Distribution"
        )

    # ----------------------------------------------------------------
    # CHART 16 - MARRIED STATUS
    # ----------------------------------------------------------------

    def plot_marital_status(self):
        """
        Plot marital status distribution.
        """

        if "Married" not in self.data.columns:

            return

        plt.figure()

        sns.countplot(
            data=self.data,
            x="Married"
        )

        plt.xlabel(
            "Marital Status"
        )

        plt.ylabel(
            "Number of Applicants"
        )

        self.save_figure(
            "16_marital_status_distribution.png",
            "Marital Status Distribution"
        )

    # ----------------------------------------------------------------
    # CHART 17 - APPROVAL BY GENDER
    # ----------------------------------------------------------------

    def plot_gender_vs_approval(self):
        """
        Plot gender against loan approval.
        """

        if (
            "Gender" not in self.data.columns
            or "Loan_Status" not in self.data.columns
        ):

            return

        plt.figure()

        sns.countplot(
            data=self.data,
            x="Gender",
            hue="Loan_Status"
        )

        plt.xlabel(
            "Gender"
        )

        plt.ylabel(
            "Number of Applications"
        )

        plt.legend(
            title="Loan Status",
            labels=[
                "Rejected",
                "Approved"
            ]
        )

        self.save_figure(
            "17_gender_vs_approval.png",
            "Gender vs Loan Approval"
        )

    # ----------------------------------------------------------------
    # CHART 18 - MARRIED VS APPROVAL
    # ----------------------------------------------------------------

    def plot_married_vs_approval(self):
        """
        Plot marital status against loan approval.
        """

        if (
            "Married" not in self.data.columns
            or "Loan_Status" not in self.data.columns
        ):

            return

        plt.figure()

        sns.countplot(
            data=self.data,
            x="Married",
            hue="Loan_Status"
        )

        plt.xlabel(
            "Marital Status"
        )

        plt.ylabel(
            "Number of Applications"
        )

        plt.legend(
            title="Loan Status",
            labels=[
                "Rejected",
                "Approved"
            ]
        )

        self.save_figure(
            "18_married_vs_approval.png",
            "Marital Status vs Loan Approval"
        )

    # ----------------------------------------------------------------
    # CHART 19 - LOAN TERM
    # ----------------------------------------------------------------

    def plot_loan_term(self):
        """
        Plot loan term distribution.
        """

        if (
            "Loan_Amount_Term"
            not in self.data.columns
        ):

            return

        plt.figure()

        sns.countplot(
            data=self.data,
            x="Loan_Amount_Term"
        )

        plt.xlabel(
            "Loan Amount Term"
        )

        plt.ylabel(
            "Number of Applications"
        )

        plt.xticks(
            rotation=30
        )

        self.save_figure(
            "19_loan_term_distribution.png",
            "Loan Amount Term Distribution"
        )

    # ----------------------------------------------------------------
    # CHART 20 - APPLICANT INCOME BY STATUS
    # ----------------------------------------------------------------

    def plot_income_by_status(self):
        """
        Compare applicant income for approved and rejected loans.
        """

        if (
            "ApplicantIncome"
            not in self.data.columns
            or "Loan_Status"
            not in self.data.columns
        ):

            return

        plt.figure()

        sns.boxplot(
            data=self.data,
            x="Loan_Status",
            y="ApplicantIncome"
        )

        plt.xlabel(
            "Loan Status"
        )

        plt.ylabel(
            "Applicant Income"
        )

        plt.xticks(
            [0, 1],
            ["Rejected", "Approved"]
        )

        self.save_figure(
            "20_income_by_loan_status.png",
            "Applicant Income by Loan Status"
        )

    # =================================================================
    # BUSINESS INSIGHTS
    # =================================================================

    def generate_business_insights(self):
        """
        Generate simple data-driven observations.
        """

        print(
            "\nGenerating business insights..."
        )

        self.add_report(
            "12. KEY DATA-DRIVEN INSIGHTS"
        )

        self.add_report(
            "-" * 80
        )

        # ------------------------------------------------------------
        # Approval Rate
        # ------------------------------------------------------------

        if "Loan_Status" in self.data.columns:

            approval_rate = (
                self.data["Loan_Status"]
                .mean()
                * 100
            )

            self.add_report(
                f"1. Overall loan approval rate "
                f"is {approval_rate:.2f}%."
            )

        # ------------------------------------------------------------
        # Credit History
        # ------------------------------------------------------------

        if (
            "Credit_History" in self.data.columns
            and "Loan_Status" in self.data.columns
        ):

            credit_analysis = (
                self.data
                .groupby("Credit_History")[
                    "Loan_Status"
                ]
                .mean()
                * 100
            )

            if not credit_analysis.empty:

                highest_credit_group = (
                    credit_analysis.idxmax()
                )

                highest_credit_rate = (
                    credit_analysis.max()
                )

                self.add_report(
                    "2. Credit history group "
                    f"{highest_credit_group} has the "
                    f"highest observed approval rate "
                    f"of {highest_credit_rate:.2f}%."
                )

        # ------------------------------------------------------------
        # Property Area
        # ------------------------------------------------------------

        if (
            "Property_Area" in self.data.columns
            and "Loan_Status" in self.data.columns
        ):

            property_analysis = (
                self.data
                .groupby("Property_Area")[
                    "Loan_Status"
                ]
                .mean()
                * 100
            )

            if not property_analysis.empty:

                area = (
                    property_analysis.idxmax()
                )

                rate = (
                    property_analysis.max()
                )

                self.add_report(
                    "3. The highest observed approval "
                    f"rate by property area is in "
                    f"{area}, at {rate:.2f}%."
                )

        # ------------------------------------------------------------
        # Education
        # ------------------------------------------------------------

        if (
            "Education" in self.data.columns
            and "Loan_Status" in self.data.columns
        ):

            education_analysis = (
                self.data
                .groupby("Education")[
                    "Loan_Status"
                ]
                .mean()
                * 100
            )

            if not education_analysis.empty:

                education = (
                    education_analysis.idxmax()
                )

                rate = (
                    education_analysis.max()
                )

                self.add_report(
                    "4. The highest observed approval "
                    f"rate by education category is "
                    f"{education}, at {rate:.2f}%."
                )

        # ------------------------------------------------------------
        # Loan Amount
        # ------------------------------------------------------------

        if "LoanAmount" in self.data.columns:

            average_loan = (
                self.data["LoanAmount"]
                .mean()
            )

            median_loan = (
                self.data["LoanAmount"]
                .median()
            )

            self.add_report(
                f"5. Average loan amount is "
                f"{average_loan:.2f}, while the "
                f"median loan amount is "
                f"{median_loan:.2f}."
            )

        # ------------------------------------------------------------
        # Income
        # ------------------------------------------------------------

        if "ApplicantIncome" in self.data.columns:

            average_income = (
                self.data["ApplicantIncome"]
                .mean()
            )

            self.add_report(
                f"6. Average applicant income is "
                f"{average_income:.2f}."
            )

        self.add_report(
            ""
        )

    # =================================================================
    # SAVE REPORT
    # =================================================================

    def save_report(self):
        """
        Save all EDA analysis results to text report.
        """

        print(
            "\nSaving EDA report..."
        )

        try:

            with open(
                EDA_REPORT_FILE,
                "w",
                encoding="utf-8"
            ) as file:

                for line in self.report_lines:

                    file.write(
                        line + "\n"
                    )

            print(
                f"EDA report saved:\n"
                f"{EDA_REPORT_FILE}"
            )

        except Exception as error:

            raise RuntimeError(
                f"Unable to save EDA report: {error}"
            )

    # =================================================================
    # RUN ALL VISUALIZATIONS
    # =================================================================

    def generate_all_visualizations(self):
        """
        Generate all EDA charts.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 4: GENERATING VISUALIZATIONS"
        )

        print("=" * 70)

        visualization_functions = [
            self.plot_loan_status,
            self.plot_gender_distribution,
            self.plot_education_distribution,
            self.plot_property_area,
            self.plot_applicant_income,
            self.plot_loan_amount,
            self.plot_credit_history,
            self.plot_education_vs_approval,
            self.plot_property_vs_approval,
            self.plot_income_vs_loan_amount,
            self.plot_total_income_vs_approval,
            self.plot_loan_amount_vs_approval,
            self.plot_correlation_heatmap,
            self.plot_dependents,
            self.plot_self_employed,
            self.plot_marital_status,
            self.plot_gender_vs_approval,
            self.plot_married_vs_approval,
            self.plot_loan_term,
            self.plot_income_by_status,
        ]

        for visualization_function in visualization_functions:

            try:

                visualization_function()

            except Exception as error:

                print(
                    f"Warning: Could not generate "
                    f"{visualization_function.__name__}: "
                    f"{error}"
                )

        print(
            f"\nTotal charts generated: "
            f"{self.chart_count}"
        )

    # =================================================================
    # COMPLETE EDA WORKFLOW
    # =================================================================

    def run(self):
        """
        Execute complete EDA workflow.
        """

        print("\n")

        print("=" * 70)

        print(
            "LOAN APPROVAL PREDICTION - EXPLORATORY DATA ANALYSIS"
        )

        print("=" * 70)

        # ------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------

        self.create_directories()

        # ------------------------------------------------------------
        # Load data
        # ------------------------------------------------------------

        self.load_data()

        # ------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------

        self.validate_data()

        # ------------------------------------------------------------
        # Identify types
        # ------------------------------------------------------------

        self.identify_column_types()

        # ------------------------------------------------------------
        # Analysis
        # ------------------------------------------------------------

        self.analyze_dataset_overview()

        self.analyze_data_types()

        self.analyze_missing_values()

        self.analyze_numerical_statistics()

        self.analyze_categorical_statistics()

        self.analyze_loan_status()

        self.calculate_approval_rate()

        self.analyze_demographics()

        self.analyze_credit_history()

        self.analyze_income()

        self.analyze_loan_amount()

        self.analyze_correlations()

        # ------------------------------------------------------------
        # Business insights
        # ------------------------------------------------------------

        self.generate_business_insights()

        # ------------------------------------------------------------
        # Visualization
        # ------------------------------------------------------------

        self.generate_all_visualizations()

        # ------------------------------------------------------------
        # Save report
        # ------------------------------------------------------------

        self.save_report()

        print("\n" + "=" * 70)

        print(
            "EDA ANALYSIS COMPLETED SUCCESSFULLY"
        )

        print("=" * 70)

        return self.data


# ====================================================================
# 6. STANDALONE FUNCTION
# ====================================================================

def run_eda(
    data_path=CLEANED_DATA_FILE
):
    """
    Convenience function for main project controller.
    """

    analyzer = LoanEDAAnalyzer(
        data_path=data_path
    )

    return analyzer.run()


# ====================================================================
# 7. QUICK DATA SUMMARY
# ====================================================================

def quick_summary(
    data_path=CLEANED_DATA_FILE
):
    """
    Return quick dataset summary.
    """

    data_path = Path(
        data_path
    )

    if not data_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    data = pd.read_csv(
        data_path
    )

    summary = {
        "rows": data.shape[0],

        "columns": data.shape[1],

        "missing_values": int(
            data.isnull()
            .sum()
            .sum()
        ),

        "duplicate_rows": int(
            data.duplicated()
            .sum()
        ),

        "numeric_columns": len(
            data.select_dtypes(
                include=np.number
            ).columns
        ),

        "categorical_columns": len(
            data.select_dtypes(
                include=[
                    "object",
                    "string",
                    "category"
                ]
            ).columns
        ),
    }

    return summary


# ====================================================================
# 8. MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":

    try:

        analyzer = LoanEDAAnalyzer()

        analyzer.run()

        print(
            "\nEDA files have been generated inside:"
        )

        print(
            f"Reports folder: {REPORTS_DIR}"
        )

        print(
            f"Charts folder : {EDA_CHARTS_DIR}"
        )

    except FileNotFoundError as error:

        print(
            f"\nFILE ERROR:\n{error}"
        )

    except ValueError as error:

        print(
            f"\nDATA ERROR:\n{error}"
        )

    except Exception as error:

        print(
            f"\nUNEXPECTED ERROR:\n{error}"
        )