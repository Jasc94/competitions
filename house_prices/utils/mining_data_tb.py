import numpy as np
import pandas as pd

##################################################### FUNCTIONS #####################################################
def drops(df):
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

def fill_nans(df, categorical_variables, numerical_variables):
    ### CATEGORICAL VARIABLES
    #Find the modes for all the numerical variables
    categorical_variables_modes = list(df[categorical_variables].mode().values[0])
    # Zip it together with the variable names in a dict
    categorical_variables_replacement = dict(zip(categorical_variables, categorical_variables_modes))

    ### NUMERICAL VARIABLES
    # Replace values using the dict
    df[categorical_variables] = df[categorical_variables].fillna(value = categorical_variables_replacement)

    # Calculate the means for all the numerical variables
    numerical_variables_means = list(df[numerical_variables].mean().values)
    # Zip it together with the variable names in a dict
    numerical_variables_replacement = dict(zip(numerical_variables, numerical_variables_means))

    # Replace values using the dict
    df[numerical_variables] = df[numerical_variables].fillna(value = numerical_variables_replacement)
    
    return df

def dummies(df, nominal_variables):
    df = pd.get_dummies(df, prefix = nominal_variables, columns = nominal_variables)
    return df

def ordinal_variables_transformation(df):
    LotShape_map = {
    "Reg" : 4,
    "IR1" : 3,
    "IR2" : 2,
    "IR3" : 1
    }

    LandContour_map = {
        "Lvl" : 4,
        "Bnk" : 3,
        "HLS" : 2,
        "Low" : 1
    }

    LandSlope_map = {
        "Gtl" : 3,
        "Mod" : 2,
        "Sev" : 1
    }

    ExterQual_map = {
        "Ex" : 5,
        "Gd" : 4,
        "TA" : 3,
        "Fa" : 2,
        "Po" : 1
    }

    ExterCond_map = {
        "Ex" : 5,
        "Gd" : 4,
        "TA" : 3,
        "Fa" : 2,
        "Po" : 1
    }

    BsmtQual_map = {
        "Ex" : 6,
        "Gd" : 5,
        "TA" : 4,
        "Fa" : 3,
        "Po" : 2,
        "NA" : 1
    }

    BsmtCond_map = {
        "Ex" : 6,
        "Gd" : 5,
        "TA" : 4,
        "Fa" : 3,
        "Po" : 2,
        "NA" : 1
    }

    BsmtExposure_map = {
        "Gd" : 5,
        "Av" : 4,
        "Mn" : 3,
        "No" : 2,
        "NA" : 1
    }

    BsmtFinType1_map = {
        "GLQ" : 7,
        "ALQ" : 6,
        "BLQ" : 5,
        "Rec" : 4,
        "LwQ" : 3,
        "Unf" : 2,
        "NA" : 1
    }

    BsmtFinType2_map = {
        "GLQ" : 7,
        "ALQ" : 6,
        "BLQ" : 5,
        "Rec" : 4,
        "LwQ" : 3,
        "Unf" : 2,
        "NA" : 1
    }

    HeatingQC_map = {
        "Ex" : 5,
        "Gd" : 4,
        "TA" : 3,
        "Fa" : 2,
        "Po" : 1
    }

    KitchenQual_map = {
        "Ex" : 5,
        "Gd" : 4,
        "TA" : 3,
        "Fa" : 2,
        "Po" : 1
    }

    Functional_map = {
        "Typ" : 8,
        "Min1" : 7,
        "Min2" : 6,
        "Mod" : 5,
        "Maj1" : 4,
        "Maj2" : 3,
        "Sev" : 2,
        "Sal" : 1
    }

    GarageFinish_map = {
        "Fin" : 4,
        "RFn" : 3,	
        "Unf" : 2,
        "NA" : 1
    }

    GarageFinish_map = {
        "Fin" : 4,
        "RFn" : 3,	
        "Unf" : 2,
        "NA" : 1
    }

    GarageQual_map = {
        "Ex" : 6,
        "Gd" : 5,
        "TA" : 4,
        "Fa" : 3,
        "Po" : 2,
        "NA" : 1
    }

    GarageCond_map = {
        "Ex" : 6,
        "Gd" : 5,
        "TA" : 4,
        "Fa" : 3,
        "Po" : 2,
        "NA" : 1
    }

    df["LotShape"] = df["LotShape"].map(LotShape_map)
    df["LandContour"] = df["LandContour"].map(LandContour_map)
    df["LandSlope"] = df["LandSlope"].map(LandSlope_map)
    df["ExterQual"] = df["ExterQual"].map(ExterQual_map)
    df["ExterCond"] = df["ExterCond"].map(ExterCond_map)
    df["BsmtQual"] = df["BsmtQual"].map(BsmtQual_map)
    df["BsmtCond"] = df["BsmtCond"].map(BsmtCond_map)
    df["BsmtExposure"] = df["BsmtExposure"].map(BsmtExposure_map)
    df["BsmtFinType1"] = df["BsmtFinType1"].map(BsmtFinType1_map)
    df["BsmtFinType2"] = df["BsmtFinType2"].map(BsmtFinType2_map)
    df["HeatingQC"] = df["HeatingQC"].map(HeatingQC_map)
    df["KitchenQual"] = df["KitchenQual"].map(KitchenQual_map)
    df["Functional"] = df["Functional"].map(Functional_map)
    df["GarageFinish"] = df["GarageFinish"].map(GarageFinish_map)
    df["GarageQual"] = df["GarageQual"].map(GarageQual_map)
    df["GarageCond"] = df["GarageCond"].map(GarageCond_map)

    return df

def ready_to_use(df):
    # Step 1
    df = drops(df)
    # Step 2
    categorical_variables, numerical_variables = split_variables(df)
    # Step 3
    ordinal_variables, nominal_variables = split_categorical_variables(categorical_variables)
    # Step 4
    df = fill_nans(df, categorical_variables, numerical_variables)
    # Step 5
    df = dummies(df, nominal_variables)
    # Step 6
    df = ordinal_variables_transformation(df)

    return df
