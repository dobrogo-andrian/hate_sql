import numpy as np
import pandas as pd


class TransformationNormalizationHandler:
    """
    A class for transforming and normalizing data features with added functionality for handling skewed distributions and custom transformations.

    Methods:
        log_transform(data, columns, handle_negatives=False):
            Applies logarithmic transformation, with automatic handling of non-positive values if specified.
        normalize(data, columns, target_range=(0, 1)):
            Normalizes columns to a specified range.
        boxcox_transform(data, columns, lambda_range=(-2, 2)):
            Applies an adaptive Box-Cox transformation by tuning lambda to stabilize variance.
        sqrt_transform(data, columns, handle_negatives=False):
            Applies square root transformation with options for negative handling.
    """

    def log_transform(self, data: pd.DataFrame, columns: list, handle_negatives: bool = False) -> pd.DataFrame:
        """
        Applies logarithmic transformation with handling for negative values.

        Parameters:
            data (pd.DataFrame): The input data.
            columns (list): List of columns to apply log transformation.
            handle_negatives (bool): If True, handles non-positive values by offsetting them automatically.

        Returns:
            pd.DataFrame: DataFrame with log-transformed features.
        """
        for col in columns:
            if (data[col] <= 0).any() and handle_negatives:
                offset = abs(data[col].min()) + 1
                data[col] = np.log(data[col] + offset)
            elif (data[col] <= 0).any():
                raise ValueError(
                    f"Column '{col}' contains non-positive values. Use handle_negatives=True to offset values.")
            else:
                data[col] = np.log(data[col])
        return data

    def normalize(self, data: pd.DataFrame, columns: list, target_range: tuple = (0, 1)) -> pd.DataFrame:
        """
        Normalizes columns to a specified range.

        Parameters:
            data (pd.DataFrame): The input data.
            columns (list): List of columns to normalize.
            target_range (tuple): The range to scale values into, e.g., (0, 1).

        Returns:
            pd.DataFrame: DataFrame with normalized features.
        """
        min_target, max_target = target_range
        for col in columns:
            col_min = data[col].min()
            col_max = data[col].max()
            # Normalization formula
            data[col] = ((data[col] - col_min) / (col_max - col_min)) * (max_target - min_target) + min_target
        return data

    def boxcox_transform(self, data: pd.DataFrame, columns: list, lambda_range: tuple = (-2, 2)) -> pd.DataFrame:
        """
        Applies an adaptive Box-Cox transformation by tuning lambda within a specified range to best stabilize variance.

        Parameters:
            data (pd.DataFrame): The input data.
            columns (list): List of columns to apply Box-Cox transformation.
            lambda_range (tuple): Range of lambda values to explore for Box-Cox transformation.

        Returns:
            pd.DataFrame: DataFrame with Box-Cox-transformed features.
        """
        for col in columns:
            if (data[col] <= 0).any():
                raise ValueError(
                    f"Column '{col}' contains non-positive values, which Box-Cox transformation cannot handle.")
            best_lambda = None
            best_variance = float('inf')
            for lam in np.linspace(lambda_range[0], lambda_range[1], 100):
                transformed = (data[col] ** lam - 1) / lam if lam != 0 else np.log(data[col])
                variance = np.var(transformed)
                if variance < best_variance:
                    best_lambda = lam
                    best_variance = variance
            # Apply transformation using the best lambda found
            data[col] = (data[col] ** best_lambda - 1) / best_lambda if best_lambda != 0 else np.log(data[col])
        return data

    def sqrt_transform(self, data: pd.DataFrame, columns: list, handle_negatives: bool = False) -> pd.DataFrame:
        """
        Applies square root transformation with optional handling for negative values.

        Parameters:
            data (pd.DataFrame): The input data.
            columns (list): List of columns to apply square root transformation.
            handle_negatives (bool): If True, offsets negative values to enable transformation.

        Returns:
            pd.DataFrame: DataFrame with square-root transformed features.
        """
        for col in columns:
            if (data[col] < 0).any() and handle_negatives:
                offset = abs(data[col].min()) + 1
                data[col] = np.sqrt(data[col] + offset)
            elif (data[col] < 0).any():
                raise ValueError(
                    f"Column '{col}' contains negative values. Use handle_negatives=True to offset values.")
            else:
                data[col] = np.sqrt(data[col])
        return data
