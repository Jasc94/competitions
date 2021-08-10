import numpy as np
import pandas as pd

##################################################### FUNCTIONS #####################################################
def load_data(data_path, variables_path):
    # Loading data
    df = pd.read_csv(data_path)
    column_names = pd.read_csv(variables_path, index_col = 0).to_dict()

    # Renaming columns
    df.columns = column_names["Description"].values()

    return df

class preprocessor:
    @staticmethod
    def dummies(df):
        return pd.get_dummies(df, prefix = ["hospital_type", "hospital_city", "hospital_region", "department", "ward_type", "ward_facility"], columns = ["Unique code for the type of Hospital", "City Code of the Hospital", "Region Code of the Hospital", "Department overlooking the case", "Code for the Ward type", "Code for the Ward Facility"])

    @staticmethod
    def ordinal_mapper(df, to_map:list):
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