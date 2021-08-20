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
    def multi_axes(self, df, target_variable = "SalePrice", n_columns = 3, plot_type = "scatter", figsize = (12, 12)):
        """Function to create a multiple axes plot based on the dataframe ahd the given structure

        Args:
            df (dataframe): Dataframe to plot
            target_variable (str): Variable to plot in the y axis
            n_columns (int): number of columns the final plot should have
            plot_type (str): Scatter for scatter plot, violin for violinplot. Defaults to scatter.
            figsize (tuple, optional): Matplotlib figure size. Defaults to (12, 12).

        Returns:
            Object: Matplotlib figure with the scatterplot of each variable against the target variable as the axes
        """
        # Calculating the number of rows from number of columns and variables to plot
        n_rows_ = self.__n_rows(df, n_columns)

        # Creating the figure and as many axes as needed
        fig, axes = plt.subplots(n_rows_, n_columns, figsize = figsize, sharey = True)
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
                if plot_type == "scatter":
                    sns.scatterplot(x = x, y = target_variable, data = df, ax = axes[row][column])
                if plot_type == "violin":
                    sns.violinplot(x = x, y = target_variable, data = df, ax = axes[row][column])

                # Some extras
                axes[row][column].set_xticklabels(labels, rotation = 90)

                # To stop plotting
                if (count + 1) < df.shape[1]:
                    count += 1
                else:
                    break

        plt.tight_layout()
        
        return fig