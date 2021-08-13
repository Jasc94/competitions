import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics

from imblearn.over_sampling import SMOTE

##################################################### FUNCTIONS #####################################################
class modeller:
    """Class with useful methods to train/test machine learning models and save the main metrics.
    """
    #########
    def __init__(self, model):
        
        self.model = model
        
        # Data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features = None
        self.kfold = None

        # Training
        self.train_scores = []
        self.val_scores = []
        self.train_set_structures = []
        self.val_set_structures = []
        self.feature_importances = None

        # Test
        self.train_score = None
        self.test_score = None
        self.train_structure = None
        self.test_structure = None
        self.prediction = None
        self.cm = None        

        # Metrics
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None

    #########
    def load_data(self, X_train, X_test, y_train, y_test, features, kfold):
        """To load the data into the modeller object

        Args:
            X_train (array/dataframe): Train set with the independent variables
            X_test (array/dataframe): Test set with the independent variables
            y_train (array/dataframe): Train set with the target variabl
            y_test (array/dataframe): Test set with the target variable
            features (list): [description]
            kfold ([object): [description]
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.features = features
        self.kfold = kfold

    #########
    def trainer(self, verb = False):
        """It trains the machine learning model using cross validation and stores train/test scores of each round

        Args:
            verb (bool, optional): The larger the value, the more info it will print about the training process. Defaults to False.
        """
        count = 1

        for (train, val) in self.kfold.split(self.X_train, self.y_train):
            # Train-Validation sets
            x_t, y_t = self.X_train[train], self.y_train[train]
            x_v, y_v = self.X_train[val], self.y_train[val]

            # Internal structure
            y_t_unique, y_t_counts = np.unique(y_t, return_counts=True)
            y_v_unique, y_v_counts = np.unique(y_v, return_counts=True)

            self.train_set_structures.append(dict(zip(y_t_unique, y_t_counts / len(y_t))))
            self.val_set_structures.append(dict(zip(y_v_unique, y_v_counts / len(y_v))))

            # Training
            self.model.fit(x_t, y_t)

            # Scores
            train_score = self.model.score(x_t, y_t)
            val_score = self.model.score(x_v, y_v)

            self.train_scores.append(train_score)
            self.val_scores.append(val_score)

            if verb:
                print(f"\n-- Model {count} --")
                print("-" * 25)
                print(">train score:", train_score)
                print(">validation score:", val_score)
                print("-" * 50)
            if verb > 1:
                print("Set structure:")
                print("Train structure:", dict(zip(y_t_unique, y_t_counts / len(y_t))))
                print("Validation structure:", dict(zip(y_v_unique, y_v_counts / len(y_v))))
                print("#" * 75)

            count += 1

        try:
            importances = self.model.feature_importances_
            feature_importances = list(zip(self.features, importances))

            self.feature_importances = pd.DataFrame(feature_importances, columns = ["features", "importance"]).sort_values(by = "importance", ascending = False)
        except:
            pass

    #########
    def tester(self, verb = False):
        """It trains the model in the full training-set and tests it in the test-set. It gets relevant metrics including confusion matrix, recall, etc...

        Args:
            verb (bool, optional): The larger the value, the more info it will print about the training process. Defaults to False.
        """

        # Training model
        self.model.fit(self.X_train, self.y_train)

        # Internal structure
        y_train_unique, y_train_counts = np.unique(self.y_train, return_counts=True)
        y_test_unique, y_test_counts = np.unique(self.y_test, return_counts=True)

        self.train_structure = dict(zip(y_train_unique, y_train_counts / len(self.y_train) * 100))
        self.test_structure = dict(zip(y_test_unique, y_test_counts / len(self.y_test) * 100))

        # Scores
        self.train_score = self.model.score(self.X_train, self.y_train)
        self.test_score = self.model.score(self.X_test, self.y_test)

        # Prediction
        self.prediction = self.model.predict(self.X_test)

        # Confusion matrix
        self.cm = metrics.confusion_matrix(self.y_test, self.prediction)

        # Metrics
        self.precision = metrics.accuracy_score(self.y_test, self.prediction)
        self.precision = metrics.recall_score(self.y_test, self.prediction, average = 'weighted')
        self.recall = metrics.precision_score(self.y_test, self.prediction, average = 'weighted')
        self.f1_score = metrics.f1_score(self.y_test, self.prediction, average = 'weighted')

        if verb:
            print("-- Scores --")
            print(">Train score:", self.train_score)
            print(">Test score:", self.test_score)
            print("-" * 50)

        if verb > 1:
            print("-" * 50)
            print("\n-- Metrics --")
            print("Accuracy:", self.accuracy)
            print("Precision:", self.precision)
            print("Recall:", self.recall)
            print("F1 score:", self.f1_score)

        if verb > 2:
            print("-" * 50)
            print("\n-- Confusion matrix --")
            print(self.cm)
        
        if verb > 3:
            print("-" * 50)
            print("\n-- Data structure --")
            print(f"Train structure (n = {len(self.X_train)}):\n{self.train_structure}")
            print(f"Test structure (n = {len(self.X_test)}):\n{self.test_structure}")
            
            

    #########
    def predictor(self, to_predict):
        """It uses the trained model to predict the label for new values.

        Args:
            to_predict (array): Array with same features used to train the model.

        Returns:
            array: Predicted label
        """
        new_predictions = self.model.predict(to_predict)
        return new_predictions

    #########
    def saver(self, path):
        """It saves the model as .pkl file.

        Args:
            path (str): path to save the model

        Returns:
            str: Succes/error message
        """
        try:
            joblib.dump(self.model, path + ".pkl")
            return "Succesfully saved"

        except:
            return "Something went wrong. Please check all the settings"

#####
class ensembler:
    """Class with useful methods to train/test several machine learning models at once and compare them
    """
    def __init__(self, models):
        # Models
        self.models = models
        self.model_names = [str(model) for model in models]
        self.modellers = [modeller(model) for model in models]

        # Data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features = None
        self.kfold = None

        # Metrics
        self.metrics = None
        self.cms = None
    
    #########
    def load_data(self, X_train, X_test, y_train, y_test, features, kfold):
        """To load the data into the ensembler object

        Args:
            X_train (array/dataframe): Train set with the independent variables
            X_test (array/dataframe): Test set with the independent variables
            y_train (array/dataframe): Train set with the target variabl
            y_test (array/dataframe): Test set with the target variable
            features (list): [description]
            kfold ([object): [description]
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.features = features
        self.kfold = kfold

    #########
    def tester(self, verb = None):
        """It trains the models in the full training-set and tests it in the test-set. It gets relevant metrics including confusion matrix, recall, etc...

        Models comparison is available as the "metrics" attribute after executing this method.
        """
        metric_names = ["Test_score", "Train_score", "Test_score_drop", "Accuracy", "Precision", "Recall", "F1_score"]
        # Scores
        test_scores = []
        train_scores = []
        test_score_drops = []
        # Metrics
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        # List with all the scores and metrics for later use
        metrics_lists = [test_scores, train_scores, test_score_drops, accuracies, precisions, recalls, f1_scores]
        
        # Confusion matrixes
        self.cms = {}

        # Loop through all the models and train/test them and save the relevant metrics
        for modeller in self.modellers:
            # Use the modeller object to train/test
            modeller.load_data(self.X_train, self.X_test, self.y_train, self.y_test, self.features, self.kfold)
            modeller.tester()
            if verb > 1:
                print(f"Model {str(modeller.model)} trained..")

            # Save all relevant model metrics
            # Scores
            test_scores.append(modeller.test_score)
            train_scores.append(modeller.train_score)
            test_score_drops.append((modeller.test_score - modeller.train_score) / modeller.train_score)
            if verb > 1:
                print(f"Model {str(modeller.model)} scores calculated..")

            # Metrics
            accuracies.append(modeller.accuracy)
            precisions.append(modeller.precision)
            recalls.append(modeller.recall)
            f1_scores.append(modeller.f1_score)
            if verb > 1:
                print(f"Model {str(modeller.model)} metrics calculated..")

            # Confusion matrix will be saved in a dict
            self.cms[str(modeller.model)] = modeller.cm
            if verb:
                print(f"Model {str(modeller.model)} is ready")
                print("-" * 50)

        # Create a dataframe with all the metrics (except for confusion matrix)
        self.metrics = pd.DataFrame(metrics_lists, index = metric_names, columns = self.model_names).T
        # Sort the dataframe by Test Score
        self.metrics = self.metrics.sort_values(by = "Test_score", ascending = False)
        if verb:
            print(f"All models trained. Metrics are available")