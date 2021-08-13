import numpy as np
import pandas as pd

##################################################### FUNCTIONS #####################################################
def load_data(data_path, variables_path):
    """Function to load the data

    Args:
        data_path (str): Path to data file
        variables_path ([type]): Path to variables' descriptions file

    Returns:
        dataframe: Dataframe of the data with the corresponding column names
    """
    # Loading data
    df = pd.read_csv(data_path)
    column_names = pd.read_csv(variables_path, index_col = 0).to_dict()

    # Renaming columns
    df.columns = column_names["Description"].values()

    # Setting the index
    df = df.set_index("Case_ID registered in Hospital")

    return df

class preprocessor:
    """Class with useful methods to automate the data preprocessing
    """
    @staticmethod
    def dummies(df):
        """It process dummy variables

        Args:
            df (dataframe): Dataframe with dummy variables to be processed.

        Returns:
            dataframe: Dataframe with processed dummy variables
        """
        return pd.get_dummies(df, prefix = ["hospital_type", "hospital_city", "hospital_region", "department", "ward_type", "ward_facility"], columns = ["Unique code for the type of Hospital", "City Code of the Hospital", "Region Code of the Hospital", "Department overlooking the case", "Code for the Ward type", "Code for the Ward Facility"])

    @staticmethod
    def ordinal_mapper(df, to_map:list):
        """It maps ordinal variables replacing old values by the new given ones.

        Args:
            df (dataframe): Dataframe with ordinal variables to be processed.
            to_map (list): List of column names to be mapped.

        Returns:
            dataframe: Dataframe with ordinal variables processed
        """
        #### Mappers
        # Admission Type registered by the Hospital
        admission = {
            "Emergency" : 1,
            "Trauma" : 2,
            "Urgent" : 3
        }

        # Severity of the illness recorded at the time of admission
        severity = {
            "Minor" : 1,
            "Moderate" : 2,
            "Extreme" : 3
        }

        # Age of the patient
        age = {
            "0-10" : 1,
            "11-20" : 2,
            "21-30" : 3,
            "31-40" : 4,
            "41-50" : 5,
            "51-60" : 6,
            "61-70" : 7,
            "71-80" : 8,
            "81-90" : 9,
            "91-100" : 10,
        }

        # Stay Days by the patient
        stay = {
            "0-10" : 1,
            "11-20" : 2,
            "21-30" : 3,
            "31-40" : 4,
            "41-50" : 5,
            "51-60" : 6,
            "61-70" : 7,
            "71-80" : 8,
            "81-90" : 9,
            "91-100" : 10,
            "More than 100 Days" : 11
        }

        #### Filtering the chosen ones
        # All maps
        mapper = {"Admission Type registered by the Hospital" : admission,
                  "Severity of the illness recorded at the time of admission" : severity,
                  "Age of the patient" : age,
                  "Stay Days by the patient" : stay}
        # Just the passed ones through to_map parameter
        mapper = {key : mapper[key] for key in to_map}

        for k, v in mapper.items():
            df[k] = df[k].map(v)

        return df

    @staticmethod
    def target_variable():
        """Target variable dict to match model labels with actual labels

        Returns: Dictionary with pairs model_labels:actual_labels.
        """
        stay = {
            1 : "0-10",
            2 : "11-20",
            3 : "21-30",
            4 : "31-40",
            5 : "41-50",
            6 : "51-60",
            7 : "61-70",
            8 : "71-80",
            9 : "81-90",
            10 : "91-100",
            11 : "More than 100 Days"
        }

        return stay

    @staticmethod
    def remove_outliers(df, variable, threshold, side = "right"):
        """It removes the outliers of a given column in the dataframe.

        Args:
            df (dataframe): Dataframe with outliers to be removed
            variable (str): Column name where the outliers are found
            threshold (int): Cut value to separate "normal" values from outliers
            side (str, optional): If "right", function will remove values >= than the threshold. If "left", it will remove values <= than the threshold. Defaults to "right".

        Returns:
            [type]: [description]
        """
        if side == "right":
            return df[df[variable] <= threshold]
        if side == "left":
            return df[df[variable] >= threshold]
        else:
            return "Please enter a valid option for 'side'"