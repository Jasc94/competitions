import numpy as np
import pandas as pd

##################################################### FUNCTIONS #####################################################
def load_data(data_path):
    df = pd.read_csv(data_path, index_col = 0)
    df = df.drop(["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"], axis = 1)
    return df

def split_variables(df):
    categorical_variables = []
    numerical_variables = []

    for column in df.columns:
        if df[column].dtype == "object":
            categorical_variables.append(column)
        else:
            numerical_variables.append(column)
    
    return categorical_variables, numerical_variables

def split_categorical_variables(categorical_variables):
    ordinal_variables = ["LotShape", "LandContour", "LandSlope", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "GarageFinish", "GarageQual", "GarageCond"]
    nominal_variables = [var for var in categorical_variables if var not in ordinal_variables]

    return ordinal_variables, nominal_variables

