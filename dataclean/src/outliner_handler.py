import pandas as pd
import numpy as np


class OutlierHandler:
    """
    A class for detecting and handling outliers in numerical data.

    Methods:
        detect_outliers(data, columns, method='iqr', threshold=1.5):
            Detects outliers using IQR or Z-score method.
        remove_outliers(data, columns, method='iqr', threshold=1.5):
            Removes rows with outliers in specified columns.
        cap_outliers(data, columns, method='iqr', threshold=1.5):
            Caps extreme values to handle outliers.

    Example usage:
        data = pd.DataFrame({
            'A': [10, 12, 14, 15, 100, 15, 12, 14, 15, 100],
            'B': [1, 2, 2, 2, 200, 2, 1, 2, 2, 200]
        })

        handler = OutlierHandler()

        # Detect outliers
        outliers = handler.detect_outliers(data, columns=['A', 'B'])
        print("Outliers detected:", outliers)

        # Remove outliers
        cleaned_data = handler.remove_outliers(data, columns=['A', 'B'])
        print("Data after removing outliers:\n", cleaned_data)

        # Cap outliers
        capped_data = handler.cap_outliers(data, columns=['A', 'B'])
        print("Data after capping outliers:\n", capped_data)

    """

    def detect_outliers(self, data: pd.DataFrame, columns: list, method: str = 'iqr', threshold: float = 1.5) -> dict:
        """
        Detects outliers in specified columns using IQR or Z-score method.

        Parameters:
            data (pd.DataFrame): The input data.
            columns (list): List of column names to check for outliers.
            method (str): The method to use for outlier detection ('iqr' or 'zscore').
            threshold (float): The threshold for defining outliers. Default is 1.5 for IQR or 3 for Z-score.

        Returns:
            dict: Dictionary with column names as keys and lists of outlier indices as values.
        """
        outliers = {}
        for col in columns:
            if method == 'iqr':
                Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (threshold * IQR)
                upper_bound = Q3 + (threshold * IQR)
                outliers[col] = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index.tolist()
            elif method == 'zscore':
                mean, std = data[col].mean(), data[col].std()
                outliers[col] = data[((data[col] - mean).abs() / std) > threshold].index.tolist()
            else:
                raise ValueError("Invalid method. Choose 'iqr' or 'zscore'.")
        return outliers

    def remove_outliers(self, data: pd.DataFrame, columns: list, method: str = 'iqr',
                        threshold: float = 1.5) -> pd.DataFrame:
        """
        Removes rows with outliers in specified columns.

        Parameters:
            data (pd.DataFrame): The input data.
            columns (list): List of column names to check for outliers.
            method (str): The method to use for outlier detection ('iqr' or 'zscore').
            threshold (float): The threshold for defining outliers.

        Returns:
            pd.DataFrame: DataFrame with outliers removed.
        """
        outliers = self.detect_outliers(data, columns, method, threshold)
        indices_to_remove = list(set(sum(outliers.values(), [])))
        return data.drop(indices_to_remove)

    def cap_outliers(self, data: pd.DataFrame, columns: list, method: str = 'iqr',
                     threshold: float = 1.5) -> pd.DataFrame:
        """
        Caps outliers in specified columns within a defined range.

        Parameters:
            data (pd.DataFrame): The input data.
            columns (list): List of column names to cap outliers.
            method (str): The method to use for outlier detection ('iqr' or 'zscore').
            threshold (float): The threshold for defining outliers.

        Returns:
            pd.DataFrame: DataFrame with outliers capped within bounds.
        """
        for col in columns:
            if method == 'iqr':
                Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (threshold * IQR)
                upper_bound = Q3 + (threshold * IQR)
                data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
                data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
            elif method == 'zscore':
                mean, std = data[col].mean(), data[col].std()
                lower_bound, upper_bound = mean - threshold * std, mean + threshold * std
                data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
                data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
            else:
                raise ValueError("Invalid method. Choose 'iqr' or 'zscore'.")
        return data
