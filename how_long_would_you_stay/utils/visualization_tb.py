import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

##################################################### PLOTTERS #####################################################
#####
class plotter:
    '''
    Class to perform some exploratory analysis on the data
    '''

    ####
    def __n_rows(self, df, n_columns):
        '''
        It calculates the number of rows (for the axes) depending on the number of variables to plot and the columns we want for the figure.
        args:
        n_columns: number of columns
        '''
        columns = list(df.columns)

        if len(columns) % n_columns == 0:
            axes_rows = len(columns) // n_columns
        else:
            axes_rows = (len(columns) // n_columns) + 1

        return axes_rows
        
    ####
    def categorical(self, df, n_columns, figsize = (12, 12)):
        '''
        It creates a plot with multiple rows and columns. It returns a figure.
        n_columns: number of columns for the row
        kind: ("strip", "dist", "box")
        figsize: size of the figure
        '''
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
        for column in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize)

            sns.boxplot(x = column, data = df, ax = ax1)
            sns.histplot(df[column], ax = ax2, bins = bins)

            plt.show()

    @staticmethod
    def correlation_matrix(correlation, figsize = (14, 14)):
        # Mask to remove duplicated values (from the diagonal up)
        mask = np.zeros_like(correlation)
        mask[np.triu_indices_from(mask)] = True

        # Plot
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize = figsize)
            ax = sns.heatmap(correlation, annot = True, linewidths = .1, mask=mask, vmax=.3, square=True, cmap = "YlGnBu")

        
    @staticmethod
    def train_val(modeller, figsize = (8, 8)):
        fig = plt.figure(figsize = figsize)
        
        plt.plot(modeller.train_scores, label = "Train scores")
        plt.plot(modeller.val_scores, label = "Test scores")

        plt.legend()
        plt.xticks(range(5))

        return fig

    @staticmethod
    def confusion_matrix(modeller, labels, figsize = (12, 12)):
        col_sum = modeller.cm.sum(axis = 0, keepdims = True)
        col_rel = modeller.cm / col_sum

        # To plot
        fig = plt.figure(figsize = figsize)
        sns.heatmap(col_rel, annot = True, linewidths = .1, square=True, cmap = "YlGnBu")

        plt.xticks(np.arange(11) + .5, list(labels.values()), rotation = 90)

        return fig