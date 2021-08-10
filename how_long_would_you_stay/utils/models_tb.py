import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics

from imblearn.over_sampling import SMOTE

##################################################### FUNCTIONS #####################################################
class modeller:
    """[summary]

    Returns:
        [type]: [description]
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
        """[summary]

        Args:
            X_train ([type]): [description]
            X_test ([type]): [description]
            y_train ([type]): [description]
            y_test ([type]): [description]
            features ([type]): [description]
            kfold ([type]): [description]
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.features = features
        self.kfold = kfold

    #########
    def trainer(self, verb = False):
        """[summary]

        Args:
            verb (bool, optional): [description]. Defaults to False.
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
        """[summary]

        Args:
            verb (bool, optional): [description]. Defaults to False.
        """
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
            print("Train structure:", self.train_structure)
            print("Test structure:", self.test_structure)
            
            

    #########
    def predictor(self, to_predict):
        """[summary]

        Args:
            to_predict ([type]): [description]

        Returns:
            [type]: [description]
        """
        new_predictions = self.model.predict(to_predict)
        return new_predictions

    #########
    def saver(self, path):
        """[summary]

        Args:
            path ([type]): [description]

        Returns:
            [type]: [description]
        """
        try:
            joblib.dump(self.model, path + ".pkl")
            return "Succesfully saved"

        except:
            return "Something went wrong. Please check all the settings"


class ensembler:
    """[summary]
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
    
    #########
    def load_data(self, X_train, X_test, y_train, y_test, features, kfold):
        '''
        Load the data for the models
        '''
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.features = features
        self.kfold = kfold

    #########
    def tester(self):
        '''
        It directly trains and tests the given models. It saves the relevant metrics as well
        '''
        metric_names = ["Test_score", "Train_score", "Test_score_drop", "Accuracy", "Precision", "Recall", "F1_score", "Confusion_matrix"]
        test_scores = []
        train_scores = []
        test_score_drops = []
        cms = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        metrics_lists = [test_scores, train_scores, test_score_drops, accuracies, precisions, recalls, f1_scores, cms]

        # Loop through all the models and train/test them and save the relevant metrics
        for model in self.ml_models:
            model.load_data(self.X_train, self.X_test, self.y_train, self.y_test, self.features, self.kfold)
            model.ml_trainer()
            model.ml_tester()
            # Saving model metrics
            test_scores.append(model.test_score)
            train_scores.append(model.train_score)
            test_score_drops.append((model.test_score - model.train_score) / model.train_score)
            accuracies.append(model.accuracy)
            precisions.append(model.precision)
            recalls.append(model.recall)
            f1_scores.append(model.f1_score)
            cms.append(model.cm)

        # Stores all the metrics as a dataframe
        self.metrics = pd.DataFrame(metrics_lists, index = metric_names, columns = self.model_names).T
        self.metrics = self.metrics.sort_values(by = "Test_score", ascending = False)
        #return self.metrics.sort_values(by = "Test_score", ascending = False)