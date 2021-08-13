import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

##################################################### PLOTTERS #####################################################
#####
class plotter:
    """Class with useful plots
    """

    ####
    def __n_rows(self, df, n_columns):
        """Private function to calculate the number of rows based on the dataframe columns and the number of columns the plot should have.

        Args:
            df (dataframe): Dataframe to plot
            n_columns (int): number of columns the final plot should have

        Returns:
            int: number of rows that the plot will have
        """
        # Dataframe columns
        columns = list(df.columns)

        # If the number of columns is even...
        if len(columns) % n_columns == 0:
            axes_rows = len(columns) // n_columns
        # If it is odd...
        else:
            axes_rows = (len(columns) // n_columns) + 1

        return axes_rows
        
    ####
    def categorical(self, df, n_columns, figsize = (12, 12)):
        """Function to create a multiple axes plot based on the dataframe ahd the given structure

        Args:
            df (dataframe): Dataframe to plot
            n_columns (int): number of columns the final plot should have
            figsize (tuple, optional): Matplotlib figure size. Defaults to (12, 12).

        Returns:
            Object: Matplotlib figure with the distribution of each variable as the axes
        """
        # Calculating the number of rows from number of columns and variables to plot
        n_rows_ = self.__n_rows(df, n_columns)

        # Creating the figure and as many axes as needed
        fig, axes = plt.subplots(n_rows_, n_columns, figsize = figsize)
        # To keep the count of the plotted variables
        count = 0

        # Some transformation, because with only one row, the shape is: (2,)
        axes_col = axes.shape[0]
        try:
            axes_row = axes.shape[1]
        except:
            axes_row = 1

        # Loop through rows
        for row in range(axes_col):
            # Loop through columns
            for column in range(axes_row):
                # Data to plot
                x = df.iloc[:, count]
                labels = x.unique()

                # Plot
                sns.countplot(x = x, data = df, ax = axes[row][column])

                # Some extras
                axes[row][column].set(xlabel = "")
                axes[row][column].set_title(df.iloc[:, count].name)
                axes[row][column].set_xticklabels(labels, rotation = 90)

                # To stop plotting
                if (count + 1) < df.shape[1]:
                    count += 1
                else:
                    break

        plt.tight_layout()
        
        return fig

    @staticmethod
    def numerical(df, bins = 20, figsize = (14, 6)):
        """It creates plots with box-plot and dist-plot for each variable in the dataframe

        Args:
            df (dataframe): Dataframe to plot
            bins (int, optional): Number of bins for dist-plot. Defaults to 20.
            figsize (tuple, optional): Matplotlib figure size. Defaults to (14, 6).
        """
        for column in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize)

            sns.boxplot(x = column, data = df, ax = ax1)
            sns.histplot(df[column], ax = ax2, bins = bins)

            plt.show()

    @staticmethod
    def correlation_matrix(correlation, figsize = (14, 14)):
        """It creates a correlation matrix plot

        Args:
            correlation (dataframe): Correlation matrix dataframe
            figsize (tuple, optional): Matplotlib figure size. Defaults to (14, 14).
        """
        # Mask to remove duplicated values (from the diagonal up)
        mask = np.zeros_like(correlation)
        mask[np.triu_indices_from(mask)] = True

        # Plot
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize = figsize)
            ax = sns.heatmap(correlation, annot = True, linewidths = .1, mask=mask, vmax=.3, square=True, cmap = "YlGnBu")

        
    @staticmethod
    def train_val(modeller, figsize = (8, 8)):
        """It plots the train/test score progression of a model during cross validation

        Args:
            modeller (object): Modeller object. It can be found in models_tb.py file.
            figsize (tuple, optional): Matplotlib figure size. Defaults to (8, 8).

        Returns:
            Object: Matplotlib figure with line-plot of train/test scores
        """
        fig = plt.figure(figsize = figsize)
        
        plt.plot(modeller.train_scores, label = "Train scores")
        plt.plot(modeller.val_scores, label = "Test scores")

        plt.legend()
        plt.xticks(range(5))

        return fig

    @staticmethod
    def confusion_matrix(modeller, labels, figsize = (12, 12)):
        """It plots a heatmap representing the machine learning model confusion matrix.

        Args:
            modeller (object): Modeller object. It can be found in models_tb.py file.
            labels (list): List of labels for the x-axis
            figsize (tuple, optional): Matplotlib figure size. Defaults to (12, 12).

        Returns:
            Object: Matplotlib figure. Heatmap representing the machine learning model confusion matrix.
        """
        col_sum = modeller.cm.sum(axis = 0, keepdims = True)
        col_rel = modeller.cm / col_sum

        # To plot
        fig = plt.figure(figsize = figsize)
        sns.heatmap(col_rel, annot = True, linewidths = .1, square=True, cmap = "YlGnBu", cbar = False)

        plt.xticks(np.arange(11) + .5, list(labels.values()), rotation = 90)

        plt.xlabel("Predicted label")
        plt.ylabel("Actual label")

        return fig