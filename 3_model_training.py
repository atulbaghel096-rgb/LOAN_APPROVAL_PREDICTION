"""
======================================================================
LOAN APPROVAL PREDICTION PROJECT
File: 3_model_training.py

Purpose:
    - Load cleaned loan dataset
    - Prepare features and target
    - Train multiple machine learning models
    - Compare model performance
    - Calculate Accuracy, Precision, Recall and F1-score
    - Calculate ROC-AUC
    - Generate confusion matrix
    - Generate ROC curve
    - Select the best model
    - Save the best model as a PKL file

Models:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest
    4. Gradient Boosting
    5. K-Nearest Neighbors

Output:
    models/best_loan_model.pkl
    reports/model_comparison.csv
    reports/confusion_matrix.png
    reports/roc_curve.png

======================================================================
"""

# ====================================================================
# 1. IMPORT LIBRARIES
# ====================================================================

import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


warnings.filterwarnings("ignore")


# ====================================================================
# 2. PROJECT PATHS
# ====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR = PROJECT_ROOT / "models"

CLEANED_DATA_FILE = (
    REPORTS_DIR /
    "cleaned_loan_data.csv"
)

MODEL_COMPARISON_FILE = (
    REPORTS_DIR /
    "model_comparison.csv"
)

CONFUSION_MATRIX_FILE = (
    REPORTS_DIR /
    "confusion_matrix.png"
)

ROC_CURVE_FILE = (
    REPORTS_DIR /
    "roc_curve.png"
)

BEST_MODEL_FILE = (
    MODELS_DIR /
    "best_loan_model.pkl"
)


# ====================================================================
# 3. MODEL TRAINING CONFIGURATION
# ====================================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ====================================================================
# 4. VISUALIZATION SETTINGS
# ====================================================================

sns.set_theme(
    style="whitegrid",
    context="notebook"
)

plt.rcParams["figure.figsize"] = (
    10,
    6
)


# ====================================================================
# 5. LOAN MODEL TRAINER CLASS
# ====================================================================

class LoanModelTrainer:
    """
    Complete machine learning training pipeline for
    Loan Approval Prediction.
    """

    # ----------------------------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------------------------

    def __init__(
        self,
        data_path=CLEANED_DATA_FILE
    ):

        self.data_path = Path(
            data_path
        )

        self.data = None

        self.X = None

        self.y = None

        self.X_train = None

        self.X_test = None

        self.y_train = None

        self.y_test = None

        self.preprocessor = None

        self.models = {}

        self.trained_models = {}

        self.results = []

        self.best_model_name = None

        self.best_model = None

        self.best_score = -1

    # ----------------------------------------------------------------
    # CREATE DIRECTORIES
    # ----------------------------------------------------------------

    def create_directories(self):
        """
        Create project output directories.
        """

        REPORTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            "Required directories created successfully."
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
                f"Run 1_data_preprocessing.py first."
            )

        try:

            self.data = pd.read_csv(
                self.data_path
            )

        except Exception as error:

            raise RuntimeError(
                f"Unable to load dataset: {error}"
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
    # VALIDATE TARGET
    # ----------------------------------------------------------------

    def validate_target(self):
        """
        Validate the target variable.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 2: VALIDATING TARGET VARIABLE"
        )

        print("=" * 70)

        if "Loan_Status" not in self.data.columns:

            raise ValueError(
                "Loan_Status column was not found "
                "in the cleaned dataset."
            )

        print(
            "Target column: Loan_Status"
        )

        print(
            "\nTarget distribution:"
        )

        print(
            self.data["Loan_Status"]
            .value_counts()
            .sort_index()
        )

    # ----------------------------------------------------------------
    # PREPARE FEATURES
    # ----------------------------------------------------------------

    def prepare_features(self):
        """
        Separate features and target.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 3: PREPARING FEATURES"
        )

        print("=" * 70)

        self.data = self.data.copy()

        # ------------------------------------------------------------
        # Remove accidental index columns
        # ------------------------------------------------------------

        unwanted_columns = [
            "Unnamed: 0",
            "index",
            "Index"
        ]

        columns_to_drop = [
            column
            for column in unwanted_columns
            if column in self.data.columns
        ]

        if columns_to_drop:

            self.data.drop(
                columns=columns_to_drop,
                inplace=True
            )

            print(
                "Removed unnecessary columns:"
            )

            for column in columns_to_drop:

                print(
                    f"  - {column}"
                )

        # ------------------------------------------------------------
        # Separate target
        # ------------------------------------------------------------

        self.y = self.data[
            "Loan_Status"
        ]

        self.X = self.data.drop(
            columns=["Loan_Status"]
        )

        # ------------------------------------------------------------
        # Ensure target is numeric
        # ------------------------------------------------------------

        if self.y.dtype == "object":

            mapping = {
                "Y": 1,
                "N": 0,
                "Yes": 1,
                "No": 0,
                "Approved": 1,
                "Rejected": 0
            }

            self.y = (
                self.y
                .map(mapping)
            )

        self.y = pd.to_numeric(
            self.y,
            errors="coerce"
        )

        # ------------------------------------------------------------
        # Remove rows with invalid target
        # ------------------------------------------------------------

        valid_target = (
            self.y.notna()
        )

        self.X = self.X.loc[
            valid_target
        ].copy()

        self.y = self.y.loc[
            valid_target
        ].astype(int)

        # ------------------------------------------------------------
        # Display feature information
        # ------------------------------------------------------------

        print(
            f"\nNumber of features: "
            f"{self.X.shape[1]}"
        )

        print(
            f"Number of samples: "
            f"{self.X.shape[0]}"
        )

        print(
            "\nFeature columns:"
        )

        for column in self.X.columns:

            print(
                f"  - {column}"
            )

    # ----------------------------------------------------------------
    # IDENTIFY FEATURE TYPES
    # ----------------------------------------------------------------

    def identify_feature_types(self):
        """
        Identify numerical and categorical features.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 4: IDENTIFYING FEATURE TYPES"
        )

        print("=" * 70)

        self.numeric_features = (
            self.X
            .select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        self.categorical_features = (
            self.X
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
            "\nNumerical features:"
        )

        for column in self.numeric_features:

            print(
                f"  - {column}"
            )

        print(
            "\nCategorical features:"
        )

        for column in self.categorical_features:

            print(
                f"  - {column}"
            )

    # ----------------------------------------------------------------
    # TRAIN TEST SPLIT
    # ----------------------------------------------------------------

    def split_data(self):
        """
        Split data into training and testing datasets.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 5: TRAIN / TEST SPLIT"
        )

        print("=" * 70)

        try:

            self.X_train, self.X_test, \
            self.y_train, self.y_test = (
                train_test_split(
                    self.X,
                    self.y,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE,
                    stratify=self.y
                )
            )

        except ValueError:

            print(
                "Warning: Stratified split failed. "
                "Using normal train/test split."
            )

            self.X_train, self.X_test, \
            self.y_train, self.y_test = (
                train_test_split(
                    self.X,
                    self.y,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE
                )
            )

        print(
            f"Training samples: "
            f"{len(self.X_train)}"
        )

        print(
            f"Testing samples: "
            f"{len(self.X_test)}"
        )

    # ----------------------------------------------------------------
    # BUILD PREPROCESSOR
    # ----------------------------------------------------------------

    def build_preprocessor(self):
        """
        Create preprocessing pipeline for numerical
        and categorical variables.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 6: BUILDING PREPROCESSING PIPELINE"
        )

        print("=" * 70)

        # ------------------------------------------------------------
        # Numerical pipeline
        # ------------------------------------------------------------

        numerical_pipeline = Pipeline(
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
                )
            ]
        )

        # ------------------------------------------------------------
        # Categorical pipeline
        # ------------------------------------------------------------

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
                )
            ]
        )

        # ------------------------------------------------------------
        # Combined preprocessor
        # ------------------------------------------------------------

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numerical_pipeline,
                    self.numeric_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_features
                )
            ],
            remainder="drop"
        )

        print(
            "Numerical preprocessing:"
        )

        print(
            "  - Median imputation"
        )

        print(
            "  - StandardScaler"
        )

        print(
            "\nCategorical preprocessing:"
        )

        print(
            "  - Most-frequent imputation"
        )

        print(
            "  - OneHotEncoder"
        )

    # ----------------------------------------------------------------
    # DEFINE MODELS
    # ----------------------------------------------------------------

    def define_models(self):
        """
        Define all machine learning algorithms.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 7: DEFINING MACHINE LEARNING MODELS"
        )

        print("=" * 70)

        self.models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_split=5,
                    random_state=RANDOM_STATE
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ),

            "Gradient Boosting":
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=RANDOM_STATE
                ),

            "K-Nearest Neighbors":
                KNeighborsClassifier(
                    n_neighbors=7
                )
        }

        print(
            "\nModels available:"
        )

        for model_name in self.models:

            print(
                f"  - {model_name}"
            )

    # ----------------------------------------------------------------
    # BUILD MODEL PIPELINE
    # ----------------------------------------------------------------

    def build_model_pipeline(
        self,
        model
    ):
        """
        Combine preprocessing and ML model.
        """

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    self.preprocessor
                ),
                (
                    "classifier",
                    model
                )
            ]
        )

        return pipeline

    # ----------------------------------------------------------------
    # TRAIN SINGLE MODEL
    # ----------------------------------------------------------------

    def train_single_model(
        self,
        model_name,
        model
    ):
        """
        Train one model and calculate evaluation metrics.
        """

        print("\n" + "-" * 70)

        print(
            f"Training: {model_name}"
        )

        print("-" * 70)

        pipeline = (
            self.build_model_pipeline(
                model
            )
        )

        # ------------------------------------------------------------
        # Train
        # ------------------------------------------------------------

        pipeline.fit(
            self.X_train,
            self.y_train
        )

        # ------------------------------------------------------------
        # Predictions
        # ------------------------------------------------------------

        y_pred = (
            pipeline.predict(
                self.X_test
            )
        )

        # ------------------------------------------------------------
        # Probability predictions
        # ------------------------------------------------------------

        y_probability = None

        if hasattr(
            pipeline,
            "predict_proba"
        ):

            try:

                y_probability = (
                    pipeline
                    .predict_proba(
                        self.X_test
                    )[:, 1]
                )

            except Exception:

                y_probability = None

        # ------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------

        accuracy = accuracy_score(
            self.y_test,
            y_pred
        )

        precision = precision_score(
            self.y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            self.y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            self.y_test,
            y_pred,
            zero_division=0
        )

        # ------------------------------------------------------------
        # ROC-AUC
        # ------------------------------------------------------------

        if y_probability is not None:

            try:

                roc_auc = roc_auc_score(
                    self.y_test,
                    y_probability
                )

            except Exception:

                roc_auc = 0.0

        else:

            roc_auc = 0.0

        # ------------------------------------------------------------
        # Classification report
        # ------------------------------------------------------------

        report = classification_report(
            self.y_test,
            y_pred,
            zero_division=0
        )

        print(
            f"\nAccuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )

        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )

        print(
            "\nClassification Report:"
        )

        print(
            report
        )

        # ------------------------------------------------------------
        # Store model
        # ------------------------------------------------------------

        self.trained_models[
            model_name
        ] = {
            "pipeline": pipeline,
            "predictions": y_pred,
            "probabilities": y_probability
        }

        # ------------------------------------------------------------
        # Store metrics
        # ------------------------------------------------------------

        result = {

            "Model": model_name,

            "Accuracy": round(
                accuracy,
                4
            ),

            "Precision": round(
                precision,
                4
            ),

            "Recall": round(
                recall,
                4
            ),

            "F1_Score": round(
                f1,
                4
            ),

            "ROC_AUC": round(
                roc_auc,
                4
            )
        }

        self.results.append(
            result
        )

        return result

    # ----------------------------------------------------------------
    # TRAIN ALL MODELS
    # ----------------------------------------------------------------

    def train_all_models(self):
        """
        Train every configured model.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 8: TRAINING ALL MACHINE LEARNING MODELS"
        )

        print("=" * 70)

        for model_name, model in (
            self.models.items()
        ):

            try:

                self.train_single_model(
                    model_name,
                    model
                )

            except Exception as error:

                print(
                    f"\nERROR while training "
                    f"{model_name}:"
                )

                print(
                    error
                )

    # ----------------------------------------------------------------
    # CREATE COMPARISON DATAFRAME
    # ----------------------------------------------------------------

    def create_comparison_dataframe(self):
        """
        Create model comparison DataFrame.
        """

        if not self.results:

            raise RuntimeError(
                "No model results available."
            )

        comparison = pd.DataFrame(
            self.results
        )

        comparison = comparison.sort_values(
            by="F1_Score",
            ascending=False
        ).reset_index(
            drop=True
        )

        return comparison

    # ----------------------------------------------------------------
    # SAVE MODEL COMPARISON
    # ----------------------------------------------------------------

    def save_model_comparison(self):
        """
        Save model performance comparison to CSV.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 9: SAVING MODEL COMPARISON"
        )

        print("=" * 70)

        comparison = (
            self.create_comparison_dataframe()
        )

        comparison.to_csv(
            MODEL_COMPARISON_FILE,
            index=False
        )

        print(
            f"Model comparison saved:"
        )

        print(
            MODEL_COMPARISON_FILE
        )

        print(
            "\nModel Performance:"
        )

        print(
            comparison.to_string(
                index=False
            )
        )

        return comparison

    # ----------------------------------------------------------------
    # SELECT BEST MODEL
    # ----------------------------------------------------------------

    def select_best_model(self):
        """
        Select best model based on F1-score.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 10: SELECTING BEST MODEL"
        )

        print("=" * 70)

        comparison = (
            self.create_comparison_dataframe()
        )

        best_row = (
            comparison
            .iloc[0]
        )

        self.best_model_name = (
            best_row["Model"]
        )

        self.best_score = (
            best_row["F1_Score"]
        )

        self.best_model = (
            self.trained_models[
                self.best_model_name
            ]["pipeline"]
        )

        print(
            f"\nSelected model: "
            f"{self.best_model_name}"
        )

        print(
            f"F1-score: "
            f"{self.best_score:.4f}"
        )

        return self.best_model

    # ----------------------------------------------------------------
    # SAVE BEST MODEL
    # ----------------------------------------------------------------

    def save_best_model(self):
        """
        Save best trained pipeline using pickle.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 11: SAVING BEST MODEL"
        )

        print("=" * 70)

        if self.best_model is None:

            raise RuntimeError(
                "Best model has not been selected."
            )

        model_package = {

            "model": self.best_model,

            "model_name": self.best_model_name,

            "features": list(
                self.X.columns
            ),

            "target": "Loan_Status",

            "random_state": RANDOM_STATE
        }

        try:

            with open(
                BEST_MODEL_FILE,
                "wb"
            ) as file:

                pickle.dump(
                    model_package,
                    file
                )

        except Exception as error:

            raise RuntimeError(
                f"Unable to save model: {error}"
            )

        print(
            "Best model saved successfully."
        )

        print(
            f"Location:\n"
            f"{BEST_MODEL_FILE}"
        )

    # ----------------------------------------------------------------
    # CONFUSION MATRIX
    # ----------------------------------------------------------------

    def generate_confusion_matrix(self):
        """
        Generate confusion matrix for best model.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 12: GENERATING CONFUSION MATRIX"
        )

        print("=" * 70)

        if self.best_model_name is None:

            raise RuntimeError(
                "Best model not selected."
            )

        predictions = (
            self.trained_models[
                self.best_model_name
            ]["predictions"]
        )

        matrix = confusion_matrix(
            self.y_test,
            predictions
        )

        plt.figure(
            figsize=(8, 6)
        )

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=[
                "Rejected",
                "Approved"
            ],
            yticklabels=[
                "Rejected",
                "Approved"
            ]
        )

        plt.title(
            f"Confusion Matrix - "
            f"{self.best_model_name}",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel(
            "Predicted Label"
        )

        plt.ylabel(
            "Actual Label"
        )

        plt.tight_layout()

        plt.savefig(
            CONFUSION_MATRIX_FILE,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Confusion matrix saved:"
        )

        print(
            CONFUSION_MATRIX_FILE
        )

    # ----------------------------------------------------------------
    # ROC CURVE
    # ----------------------------------------------------------------

    def generate_roc_curve(self):
        """
        Generate ROC curves for all trained models.
        """

        print("\n" + "=" * 70)

        print(
            "STEP 13: GENERATING ROC CURVE"
        )

        print("=" * 70)

        plt.figure(
            figsize=(10, 7)
        )

        plotted = False

        for model_name, model_data in (
            self.trained_models.items()
        ):

            probabilities = (
                model_data["probabilities"]
            )

            if probabilities is None:

                continue

            try:

                fpr, tpr, _ = roc_curve(
                    self.y_test,
                    probabilities
                )

                auc_score = roc_auc_score(
                    self.y_test,
                    probabilities
                )

                plt.plot(
                    fpr,
                    tpr,
                    linewidth=2,
                    label=(
                        f"{model_name} "
                        f"(AUC={auc_score:.3f})"
                    )
                )

                plotted = True

            except Exception as error:

                print(
                    f"Could not generate ROC "
                    f"for {model_name}: {error}"
                )

        # ------------------------------------------------------------
        # Random baseline
        # ------------------------------------------------------------

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random Classifier"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            "ROC Curve Comparison",
            fontsize=14,
            fontweight="bold"
        )

        plt.legend(
            loc="lower right"
        )

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            ROC_CURVE_FILE,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        if plotted:

            print(
                f"ROC curve saved:"
            )

            print(
                ROC_CURVE_FILE
            )

    # ----------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ----------------------------------------------------------------

    def analyze_feature_importance(self):
        """
        Try to extract feature importance from tree-based
        best models.
        """

        if self.best_model is None:

            return

        classifier = (
            self.best_model
            .named_steps
            .get("classifier")
        )

        preprocessor = (
            self.best_model
            .named_steps
            .get("preprocessor")
        )

        if not hasattr(
            classifier,
            "feature_importances_"
        ):

            print(
                "\nFeature importance is not available "
                "for the selected model."
            )

            return

        try:

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

            importance_values = (
                classifier
                .feature_importances_
            )

            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": importance_values
                }
            )

            importance_df = (
                importance_df
                .sort_values(
                    by="Importance",
                    ascending=False
                )
                .head(15)
            )

            feature_file = (
                REPORTS_DIR /
                "feature_importance.csv"
            )

            importance_df.to_csv(
                feature_file,
                index=False
            )

            print(
                "\nTop important features:"
            )

            print(
                importance_df.to_string(
                    index=False
                )
            )

            print(
                f"\nFeature importance saved:"
            )

            print(
                feature_file
            )

            # --------------------------------------------------------
            # Plot feature importance
            # --------------------------------------------------------

            plt.figure(
                figsize=(11, 7)
            )

            sns.barplot(
                data=importance_df,
                x="Importance",
                y="Feature"
            )

            plt.title(
                "Top Feature Importances",
                fontsize=14,
                fontweight="bold"
            )

            plt.xlabel(
                "Importance"
            )

            plt.ylabel(
                "Feature"
            )

            feature_chart = (
                REPORTS_DIR /
                "feature_importance.png"
            )

            plt.tight_layout()

            plt.savefig(
                feature_chart,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print(
                f"Feature importance chart saved:"
            )

            print(
                feature_chart
            )

        except Exception as error:

            print(
                f"Feature importance analysis "
                f"could not be completed: {error}"
            )

    # ----------------------------------------------------------------
    # SAVE TRAINING SUMMARY
    # ----------------------------------------------------------------

    def save_training_summary(self):
        """
        Save a text summary of the model training process.
        """

        summary_file = (
            REPORTS_DIR /
            "training_summary.txt"
        )

        comparison = (
            self.create_comparison_dataframe()
        )

        with open(
            summary_file,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "=" * 80 + "\n"
            )

            file.write(
                "LOAN APPROVAL MODEL TRAINING SUMMARY\n"
            )

            file.write(
                "=" * 80 + "\n\n"
            )

            file.write(
                f"Dataset rows: "
                f"{len(self.data)}\n"
            )

            file.write(
                f"Number of features: "
                f"{self.X.shape[1]}\n"
            )

            file.write(
                f"Training samples: "
                f"{len(self.X_train)}\n"
            )

            file.write(
                f"Testing samples: "
                f"{len(self.X_test)}\n\n"
            )

            file.write(
                "MODEL PERFORMANCE\n"
            )

            file.write(
                "-" * 80 + "\n"
            )

            file.write(
                comparison
                .to_string(
                    index=False
                )
            )

            file.write(
                "\n\n"
            )

            file.write(
                f"Selected Model: "
                f"{self.best_model_name}\n"
            )

            file.write(
                f"Best F1 Score: "
                f"{self.best_score:.4f}\n"
            )

            file.write(
                "\n"
            )

            file.write(
                "The selected model is saved in:\n"
            )

            file.write(
                str(BEST_MODEL_FILE)
            )

        print(
            f"Training summary saved:"
        )

        print(
            summary_file
        )

    # =================================================================
    # COMPLETE TRAINING WORKFLOW
    # =================================================================

    def run(self):
        """
        Execute complete model training workflow.
        """

        print("\n")

        print("=" * 70)

        print(
            "LOAN APPROVAL PREDICTION"
        )

        print(
            "MACHINE LEARNING MODEL TRAINING"
        )

        print("=" * 70)

        # ------------------------------------------------------------
        # Directory setup
        # ------------------------------------------------------------

        self.create_directories()

        # ------------------------------------------------------------
        # Load data
        # ------------------------------------------------------------

        self.load_data()

        # ------------------------------------------------------------
        # Validate target
        # ------------------------------------------------------------

        self.validate_target()

        # ------------------------------------------------------------
        # Prepare features
        # ------------------------------------------------------------

        self.prepare_features()

        # ------------------------------------------------------------
        # Identify feature types
        # ------------------------------------------------------------

        self.identify_feature_types()

        # ------------------------------------------------------------
        # Train/test split
        # ------------------------------------------------------------

        self.split_data()

        # ------------------------------------------------------------
        # Preprocessor
        # ------------------------------------------------------------

        self.build_preprocessor()

        # ------------------------------------------------------------
        # Models
        # ------------------------------------------------------------

        self.define_models()

        # ------------------------------------------------------------
        # Train
        # ------------------------------------------------------------

        self.train_all_models()

        if not self.results:

            raise RuntimeError(
                "No model was successfully trained."
            )

        # ------------------------------------------------------------
        # Comparison
        # ------------------------------------------------------------

        self.save_model_comparison()

        # ------------------------------------------------------------
        # Select best
        # ------------------------------------------------------------

        self.select_best_model()

        # ------------------------------------------------------------
        # Save best model
        # ------------------------------------------------------------

        self.save_best_model()

        # ------------------------------------------------------------
        # Confusion matrix
        # ------------------------------------------------------------

        self.generate_confusion_matrix()

        # ------------------------------------------------------------
        # ROC curve
        # ------------------------------------------------------------

        self.generate_roc_curve()

        # ------------------------------------------------------------
        # Feature importance
        # ------------------------------------------------------------

        self.analyze_feature_importance()

        # ------------------------------------------------------------
        # Training summary
        # ------------------------------------------------------------

        self.save_training_summary()

        # ------------------------------------------------------------
        # Final message
        # ------------------------------------------------------------

        print("\n" + "=" * 70)

        print(
            "MODEL TRAINING COMPLETED SUCCESSFULLY"
        )

        print("=" * 70)

        print(
            f"\nBest Model: "
            f"{self.best_model_name}"
        )

        print(
            f"Best F1 Score: "
            f"{self.best_score:.4f}"
        )

        print(
            f"\nSaved model:"
        )

        print(
            BEST_MODEL_FILE
        )

        return self.best_model


# ====================================================================
# 6. CONVENIENCE FUNCTION
# ====================================================================

def train_models(
    data_path=CLEANED_DATA_FILE
):
    """
    Convenience function to train models.
    """

    trainer = LoanModelTrainer(
        data_path=data_path
    )

    return trainer.run()


# ====================================================================
# 7. LOAD SAVED MODEL
# ====================================================================

def load_saved_model(
    model_path=BEST_MODEL_FILE
):
    """
    Load previously saved model package.
    """

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Saved model not found:\n"
            f"{model_path}"
        )

    with open(
        model_path,
        "rb"
    ) as file:

        model_package = (
            pickle.load(file)
        )

    return model_package


# ====================================================================
# 8. TEST SAVED MODEL
# ====================================================================

def test_saved_model(
    model_path=BEST_MODEL_FILE
):
    """
    Test whether the saved model can be loaded correctly.
    """

    try:

        package = load_saved_model(
            model_path
        )

        print(
            "\nSaved model loaded successfully."
        )

        print(
            f"Model name: "
            f"{package.get('model_name')}"
        )

        print(
            f"Number of features: "
            f"{len(package.get('features', []))}"
        )

        return True

    except Exception as error:

        print(
            f"\nSaved model test failed:"
        )

        print(
            error
        )

        return False


# ====================================================================
# 9. MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":

    try:

        trainer = LoanModelTrainer()

        trainer.run()

        print(
            "\nTesting saved model..."
        )

        test_saved_model()

        print(
            "\nAll model-training outputs "
            "have been generated successfully."
        )

    except FileNotFoundError as error:

        print(
            f"\nFILE ERROR:\n{error}"
        )

    except ValueError as error:

        print(
            f"\nDATA ERROR:\n{error}"
        )

    except ImportError as error:

        print(
            f"\nLIBRARY ERROR:\n{error}"
        )

        print(
            "\nMake sure required libraries "
            "are installed using requirements.txt."
        )

    except Exception as error:

        print(
            f"\nUNEXPECTED ERROR:\n{error}"
        )