"""
======================================================================
LOAN APPROVAL PREDICTION PROJECT
File: 4_prediction.py

Purpose:
    - Load the saved best ML model
    - Accept new applicant information
    - Predict loan approval/rejection
    - Display approval probability
    - Save prediction results
    - Provide interactive prediction functionality

Input:
    Customer/applicant details

Output:
    predictions/prediction_results.csv

======================================================================
"""

# ====================================================================
# 1. IMPORT LIBRARIES
# ====================================================================

import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ====================================================================
# 2. PROJECT PATHS
# ====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = (
    PROJECT_ROOT / "models"
)

PREDICTION_DIR = (
    PROJECT_ROOT / "predictions"
)

MODEL_FILE = (
    MODEL_DIR / "best_loan_model.pkl"
)

PREDICTION_FILE = (
    PREDICTION_DIR /
    "prediction_results.csv"
)


# ====================================================================
# 3. PREDICTION CLASS
# ====================================================================

class LoanPredictor:
    """
    Class responsible for loading the trained model
    and predicting loan approval.
    """

    # ----------------------------------------------------------------
    # INITIALIZATION
    # ----------------------------------------------------------------

    def __init__(
        self,
        model_path=MODEL_FILE
    ):
        """
        Initialize the prediction system.
        """

        self.model_path = Path(
            model_path
        )

        self.model_package = None

        self.model = None

        self.model_name = None

        self.required_features = []

        self.load_model()

        self.create_prediction_directory()

    # ----------------------------------------------------------------
    # CREATE DIRECTORY
    # ----------------------------------------------------------------

    def create_prediction_directory(self):
        """
        Create predictions directory if it does not exist.
        """

        PREDICTION_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    # ----------------------------------------------------------------
    # LOAD MODEL
    # ----------------------------------------------------------------

    def load_model(self):
        """
        Load the trained model from PKL file.
        """

        print("\n" + "=" * 70)

        print(
            "LOADING TRAINED LOAN MODEL"
        )

        print("=" * 70)

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"""
Trained model was not found.

Expected location:
{self.model_path}

Please run:
python 3_model_training.py

before using this prediction file.
"""
            )

        try:

            with open(
                self.model_path,
                "rb"
            ) as file:

                self.model_package = (
                    pickle.load(file)
                )

        except Exception as error:

            raise RuntimeError(
                f"Unable to load trained model: {error}"
            )

        # ------------------------------------------------------------
        # Extract model
        # ------------------------------------------------------------

        if isinstance(
            self.model_package,
            dict
        ):

            self.model = (
                self.model_package.get(
                    "model"
                )
            )

            self.model_name = (
                self.model_package.get(
                    "model_name",
                    "Unknown Model"
                )
            )

            self.required_features = (
                self.model_package.get(
                    "features",
                    []
                )
            )

        else:

            self.model = (
                self.model_package
            )

            self.model_name = (
                "Saved Loan Model"
            )

        if self.model is None:

            raise ValueError(
                "Model object could not be extracted "
                "from the saved PKL file."
            )

        print(
            "\nModel loaded successfully."
        )

        print(
            f"Model: {self.model_name}"
        )

        if self.required_features:

            print(
                "\nRequired features:"
            )

            for feature in (
                self.required_features
            ):

                print(
                    f"  - {feature}"
                )

    # ----------------------------------------------------------------
    # DISPLAY MODEL INFORMATION
    # ----------------------------------------------------------------

    def display_model_information(self):
        """
        Display information about the loaded model.
        """

        print("\n" + "=" * 70)

        print(
            "MODEL INFORMATION"
        )

        print("=" * 70)

        print(
            f"Model Name: {self.model_name}"
        )

        print(
            f"Model File: {self.model_path.name}"
        )

        print(
            f"Number of Features: "
            f"{len(self.required_features)}"
        )

        if self.required_features:

            print(
                "\nFeatures used by the model:"
            )

            for number, feature in enumerate(
                self.required_features,
                start=1
            ):

                print(
                    f"{number}. {feature}"
                )

    # ----------------------------------------------------------------
    # VALIDATE APPLICANT DATA
    # ----------------------------------------------------------------

    def validate_applicant_data(
        self,
        applicant_data
    ):
        """
        Validate applicant information before prediction.
        """

        if not isinstance(
            applicant_data,
            dict
        ):

            raise TypeError(
                "Applicant data must be provided "
                "as a dictionary."
            )

        if not applicant_data:

            raise ValueError(
                "Applicant data cannot be empty."
            )

        # ------------------------------------------------------------
        # Check missing required fields
        # ------------------------------------------------------------

        if self.required_features:

            missing_features = [
                feature
                for feature in self.required_features
                if feature not in applicant_data
            ]

            if missing_features:

                raise ValueError(
                    "Missing required fields: "
                    + ", ".join(
                        missing_features
                    )
                )

        # ------------------------------------------------------------
        # Validate numerical values
        # ------------------------------------------------------------

        numerical_fields = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Dependents"
        ]

        for field in numerical_fields:

            if field not in applicant_data:

                continue

            value = applicant_data[field]

            try:

                numeric_value = float(
                    value
                )

            except (
                TypeError,
                ValueError
            ):

                raise ValueError(
                    f"{field} must contain a numeric value."
                )

            if field in [
                "ApplicantIncome",
                "CoapplicantIncome",
                "LoanAmount"
            ]:

                if numeric_value < 0:

                    raise ValueError(
                        f"{field} cannot be negative."
                    )

            if field == "Credit_History":

                if numeric_value not in [
                    0,
                    1
                ]:

                    raise ValueError(
                        "Credit_History must be "
                        "0 or 1."
                    )

        return True

    # ----------------------------------------------------------------
    # PREPARE DATAFRAME
    # ----------------------------------------------------------------

    def prepare_dataframe(
        self,
        applicant_data
    ):
        """
        Convert applicant dictionary into DataFrame.
        """

        self.validate_applicant_data(
            applicant_data
        )

        data = applicant_data.copy()

        # ------------------------------------------------------------
        # Convert numerical values
        # ------------------------------------------------------------

        numerical_fields = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
            "Dependents"
        ]

        for field in numerical_fields:

            if field in data:

                try:

                    data[field] = float(
                        data[field]
                    )

                except (
                    ValueError,
                    TypeError
                ):

                    pass

        # ------------------------------------------------------------
        # Convert Dependents
        # ------------------------------------------------------------

        if "Dependents" in data:

            if data["Dependents"] == "3+":

                data["Dependents"] = 3

        # ------------------------------------------------------------
        # Create DataFrame
        # ------------------------------------------------------------

        applicant_df = pd.DataFrame(
            [data]
        )

        # ------------------------------------------------------------
        # Match training feature order
        # ------------------------------------------------------------

        if self.required_features:

            for feature in (
                self.required_features
            ):

                if feature not in applicant_df.columns:

                    applicant_df[
                        feature
                    ] = np.nan

            applicant_df = (
                applicant_df[
                    self.required_features
                ]
            )

        return applicant_df

    # ----------------------------------------------------------------
    # PREDICT LOAN
    # ----------------------------------------------------------------

    def predict(
        self,
        applicant_data
    ):
        """
        Predict loan approval for one applicant.
        """

        applicant_df = (
            self.prepare_dataframe(
                applicant_data
            )
        )

        # ------------------------------------------------------------
        # Prediction
        # ------------------------------------------------------------

        try:

            prediction = (
                self.model.predict(
                    applicant_df
                )
            )

        except Exception as error:

            raise RuntimeError(
                f"Prediction failed: {error}"
            )

        prediction_value = int(
            prediction[0]
        )

        # ------------------------------------------------------------
        # Approval status
        # ------------------------------------------------------------

        if prediction_value == 1:

            status = "Approved"

        else:

            status = "Rejected"

        # ------------------------------------------------------------
        # Probability
        # ------------------------------------------------------------

        approval_probability = None

        rejection_probability = None

        if hasattr(
            self.model,
            "predict_proba"
        ):

            try:

                probabilities = (
                    self.model
                    .predict_proba(
                        applicant_df
                    )[0]
                )

                rejection_probability = (
                    float(
                        probabilities[0]
                    ) * 100
                )

                approval_probability = (
                    float(
                        probabilities[1]
                    ) * 100
                )

            except Exception:

                approval_probability = None

                rejection_probability = None

        # ------------------------------------------------------------
        # Prediction result
        # ------------------------------------------------------------

        result = {

            "Prediction_Date": datetime.now()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "Model_Name": self.model_name,

            "Loan_Prediction": status,

            "Prediction_Code": prediction_value,

            "Approval_Probability": (
                round(
                    approval_probability,
                    2
                )
                if approval_probability
                is not None
                else None
            ),

            "Rejection_Probability": (
                round(
                    rejection_probability,
                    2
                )
                if rejection_probability
                is not None
                else None
            )
        }

        # ------------------------------------------------------------
        # Add applicant information
        # ------------------------------------------------------------

        for column in applicant_df.columns:

            value = applicant_df.iloc[
                0
            ][column]

            if pd.isna(value):

                value = ""

            result[column] = value

        return result

    # ----------------------------------------------------------------
    # DISPLAY RESULT
    # ----------------------------------------------------------------

    def display_result(
        self,
        result
    ):
        """
        Display prediction result in a clean format.
        """

        print("\n")

        print("=" * 70)

        print(
            "LOAN PREDICTION RESULT"
        )

        print("=" * 70)

        print(
            f"\nPrediction Date:"
        )

        print(
            result["Prediction_Date"]
        )

        print(
            f"\nModel Used:"
        )

        print(
            result["Model_Name"]
        )

        print(
            "\n" + "-" * 70
        )

        print(
            "LOAN STATUS"
        )

        print(
            "-" * 70
        )

        print(
            f"\nResult: "
            f"{result['Loan_Prediction']}"
        )

        # ------------------------------------------------------------
        # Probability
        # ------------------------------------------------------------

        if (
            result["Approval_Probability"]
            is not None
        ):

            print(
                f"\nApproval Probability: "
                f"{result['Approval_Probability']:.2f}%"
            )

            print(
                f"Rejection Probability: "
                f"{result['Rejection_Probability']:.2f}%"
            )

        print(
            "\n" + "=" * 70
        )

        print(
            "APPLICANT INFORMATION"
        )

        print(
            "=" * 70
        )

        excluded_columns = [
            "Prediction_Date",
            "Model_Name",
            "Loan_Prediction",
            "Prediction_Code",
            "Approval_Probability",
            "Rejection_Probability"
        ]

        for key, value in result.items():

            if key in excluded_columns:

                continue

            print(
                f"{key}: {value}"
            )

        print(
            "=" * 70
        )

    # ----------------------------------------------------------------
    # SAVE RESULT
    # ----------------------------------------------------------------

    def save_prediction(
        self,
        result
    ):
        """
        Save prediction result to CSV.
        """

        PREDICTION_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        result_df = pd.DataFrame(
            [result]
        )

        # ------------------------------------------------------------
        # Append if file exists
        # ------------------------------------------------------------

        if PREDICTION_FILE.exists():

            try:

                old_data = pd.read_csv(
                    PREDICTION_FILE
                )

                result_df = pd.concat(
                    [
                        old_data,
                        result_df
                    ],
                    ignore_index=True
                )

            except Exception as error:

                print(
                    f"Warning: Existing prediction "
                    f"file could not be read: {error}"
                )

        # ------------------------------------------------------------
        # Save
        # ------------------------------------------------------------

        try:

            result_df.to_csv(
                PREDICTION_FILE,
                index=False
            )

        except Exception as error:

            raise RuntimeError(
                f"Unable to save prediction: {error}"
            )

        print(
            f"\nPrediction saved successfully:"
        )

        print(
            PREDICTION_FILE
        )

    # ----------------------------------------------------------------
    # PREDICT AND SAVE
    # ----------------------------------------------------------------

    def predict_and_save(
        self,
        applicant_data
    ):
        """
        Complete prediction workflow.
        """

        result = self.predict(
            applicant_data
        )

        self.display_result(
            result
        )

        self.save_prediction(
            result
        )

        return result

    # ----------------------------------------------------------------
    # GET INTEGER INPUT
    # ----------------------------------------------------------------

    @staticmethod
    def get_integer_input(
        prompt,
        allow_blank=False
    ):
        """
        Safely get integer input from user.
        """

        while True:

            value = input(
                prompt
            ).strip()

            if (
                allow_blank
                and value == ""
            ):

                return None

            try:

                return int(
                    value
                )

            except ValueError:

                print(
                    "Please enter a valid integer."
                )

    # ----------------------------------------------------------------
    # GET FLOAT INPUT
    # ----------------------------------------------------------------

    @staticmethod
    def get_float_input(
        prompt,
        allow_blank=False
    ):
        """
        Safely get float input from user.
        """

        while True:

            value = input(
                prompt
            ).strip()

            if (
                allow_blank
                and value == ""
            ):

                return None

            try:

                return float(
                    value
                )

            except ValueError:

                print(
                    "Please enter a valid number."
                )

    # ----------------------------------------------------------------
    # GET YES / NO INPUT
    # ----------------------------------------------------------------

    @staticmethod
    def get_yes_no_input(
        prompt
    ):
        """
        Get Yes/No input from user.
        """

        while True:

            value = input(
                prompt
            ).strip().lower()

            if value in [
                "yes",
                "y"
            ]:

                return "Yes"

            if value in [
                "no",
                "n"
            ]:

                return "No"

            print(
                "Please enter Yes or No."
            )

    # ----------------------------------------------------------------
    # GET GENDER
    # ----------------------------------------------------------------

    @staticmethod
    def get_gender_input():
        """
        Get applicant gender.
        """

        while True:

            gender = input(
                "Gender (Male/Female): "
            ).strip().title()

            if gender in [
                "Male",
                "Female"
            ]:

                return gender

            print(
                "Please enter Male or Female."
            )

    # ----------------------------------------------------------------
    # GET DEPENDENTS
    # ----------------------------------------------------------------

    @staticmethod
    def get_dependents_input():
        """
        Get number of dependents.
        """

        while True:

            value = input(
                "Dependents (0/1/2/3+): "
            ).strip()

            if value in [
                "0",
                "1",
                "2",
                "3+"
            ]:

                if value == "3+":

                    return 3

                return int(
                    value
                )

            print(
                "Please enter 0, 1, 2 or 3+."
            )

    # ----------------------------------------------------------------
    # GET EDUCATION
    # ----------------------------------------------------------------

    @staticmethod
    def get_education_input():
        """
        Get education status.
        """

        while True:

            education = input(
                "Education (Graduate/Not Graduate): "
            ).strip().title()

            if education in [
                "Graduate",
                "Not Graduate"
            ]:

                return education

            print(
                "Please enter Graduate "
                "or Not Graduate."
            )

    # ----------------------------------------------------------------
    # GET PROPERTY AREA
    # ----------------------------------------------------------------

    @staticmethod
    def get_property_area_input():
        """
        Get property area.
        """

        valid_values = [
            "Urban",
            "Semiurban",
            "Rural"
        ]

        while True:

            area = input(
                "Property Area "
                "(Urban/Semiurban/Rural): "
            ).strip().title()

            # --------------------------------------------------------
            # Handle Semiurban spelling
            # --------------------------------------------------------

            if area.lower() == "semiurban":

                area = "Semiurban"

            if area in valid_values:

                return area

            print(
                "Please enter Urban, "
                "Semiurban or Rural."
            )

    # ----------------------------------------------------------------
    # COLLECT APPLICANT DATA
    # ----------------------------------------------------------------

    def collect_applicant_data(self):
        """
        Collect applicant information interactively.
        """

        print("\n" + "=" * 70)

        print(
            "NEW LOAN APPLICATION"
        )

        print("=" * 70)

        print(
            "\nEnter applicant details below."
        )

        print(
            "Please provide accurate information."
        )

        print()

        # ------------------------------------------------------------
        # Gender
        # ------------------------------------------------------------

        gender = self.get_gender_input()

        # ------------------------------------------------------------
        # Married
        # ------------------------------------------------------------

        married = self.get_yes_no_input(
            "Married (Yes/No): "
        )

        # ------------------------------------------------------------
        # Dependents
        # ------------------------------------------------------------

        dependents = (
            self.get_dependents_input()
        )

        # ------------------------------------------------------------
        # Education
        # ------------------------------------------------------------

        education = (
            self.get_education_input()
        )

        # ------------------------------------------------------------
        # Self Employed
        # ------------------------------------------------------------

        self_employed = (
            self.get_yes_no_input(
                "Self Employed (Yes/No): "
            )
        )

        # ------------------------------------------------------------
        # Applicant Income
        # ------------------------------------------------------------

        applicant_income = (
            self.get_float_input(
                "Applicant Income: "
            )
        )

        # ------------------------------------------------------------
        # Coapplicant Income
        # ------------------------------------------------------------

        coapplicant_income = (
            self.get_float_input(
                "Coapplicant Income: "
            )
        )

        # ------------------------------------------------------------
        # Loan Amount
        # ------------------------------------------------------------

        loan_amount = (
            self.get_float_input(
                "Loan Amount: "
            )
        )

        # ------------------------------------------------------------
        # Loan Term
        # ------------------------------------------------------------

        loan_term = (
            self.get_integer_input(
                "Loan Amount Term "
                "(example 360): "
            )
        )

        # ------------------------------------------------------------
        # Credit History
        # ------------------------------------------------------------

        while True:

            credit_history = input(
                "Credit History "
                "(1 = Good, 0 = Bad): "
            ).strip()

            if credit_history in [
                "0",
                "1"
            ]:

                credit_history = int(
                    credit_history
                )

                break

            print(
                "Please enter only 0 or 1."
            )

        # ------------------------------------------------------------
        # Property Area
        # ------------------------------------------------------------

        property_area = (
            self.get_property_area_input()
        )

        # ------------------------------------------------------------
        # Build dictionary
        # ------------------------------------------------------------

        applicant_data = {

            "Gender": gender,

            "Married": married,

            "Dependents": dependents,

            "Education": education,

            "Self_Employed": self_employed,

            "ApplicantIncome": applicant_income,

            "CoapplicantIncome": coapplicant_income,

            "LoanAmount": loan_amount,

            "Loan_Amount_Term": loan_term,

            "Credit_History": credit_history,

            "Property_Area": property_area
        }

        return applicant_data

    # ----------------------------------------------------------------
    # DEMO PREDICTION
    # ----------------------------------------------------------------

    def demo_prediction(self):
        """
        Run prediction using sample applicant data.
        """

        print("\n" + "=" * 70)

        print(
            "DEMO LOAN APPLICATION"
        )

        print("=" * 70)

        demo_applicant = {

            "Gender": "Male",

            "Married": "Yes",

            "Dependents": 1,

            "Education": "Graduate",

            "Self_Employed": "No",

            "ApplicantIncome": 5000,

            "CoapplicantIncome": 1500,

            "LoanAmount": 150,

            "Loan_Amount_Term": 360,

            "Credit_History": 1,

            "Property_Area": "Urban"
        }

        return self.predict_and_save(
            demo_applicant
        )

    # ----------------------------------------------------------------
    # INTERACTIVE MODE
    # ----------------------------------------------------------------

    def interactive_mode(self):
        """
        Start interactive loan prediction mode.
        """

        print("\n")

        print("=" * 70)

        print(
            "LOAN APPROVAL PREDICTION SYSTEM"
        )

        print("=" * 70)

        print(
            f"\nUsing Model: "
            f"{self.model_name}"
        )

        while True:

            try:

                applicant_data = (
                    self.collect_applicant_data()
                )

                self.predict_and_save(
                    applicant_data
                )

            except (
                ValueError,
                TypeError,
                RuntimeError
            ) as error:

                print(
                    f"\nError: {error}"
                )

            except KeyboardInterrupt:

                print(
                    "\n\nPrediction cancelled by user."
                )

                break

            # --------------------------------------------------------
            # Ask for another prediction
            # --------------------------------------------------------

            while True:

                again = input(
                    "\nDo you want to predict "
                    "another application? (Yes/No): "
                ).strip().lower()

                if again in [
                    "yes",
                    "y"
                ]:

                    break

                if again in [
                    "no",
                    "n"
                ]:

                    print(
                        "\nThank you for using "
                        "Loan Approval Prediction System."
                    )

                    return

                print(
                    "Please enter Yes or No."
                )


# ====================================================================
# 4. SIMPLE PREDICTION FUNCTION
# ====================================================================

def predict_loan(
    applicant_data,
    model_path=MODEL_FILE
):
    """
    Simple function for external files such as 5_main.py.
    """

    predictor = LoanPredictor(
        model_path=model_path
    )

    return predictor.predict_and_save(
        applicant_data
    )


# ====================================================================
# 5. MAIN EXECUTION
# ====================================================================

def main():
    """
    Main function for interactive prediction.
    """

    try:

        predictor = LoanPredictor()

        predictor.display_model_information()

        print("\n" + "=" * 70)

        print(
            "SELECT PREDICTION MODE"
        )

        print("=" * 70)

        print(
            "\n1. Interactive Prediction"
        )

        print(
            "2. Demo Prediction"
        )

        print(
            "3. Exit"
        )

        while True:

            choice = input(
                "\nEnter your choice (1/2/3): "
            ).strip()

            if choice == "1":

                predictor.interactive_mode()

                break

            elif choice == "2":

                predictor.demo_prediction()

                break

            elif choice == "3":

                print(
                    "\nProgram closed."
                )

                break

            else:

                print(
                    "Invalid choice. "
                    "Please select 1, 2 or 3."
                )

    except FileNotFoundError as error:

        print(
            f"\nFILE ERROR:\n{error}"
        )

    except ImportError as error:

        print(
            f"\nLIBRARY ERROR:\n{error}"
        )

    except Exception as error:

        print(
            f"\nUNEXPECTED ERROR:\n{error}"
        )


# ====================================================================
# 6. START PROGRAM
# ====================================================================

if __name__ == "__main__":

    main()