import pandas as pd
from itertools import combinations


class FeatureEngineering:
    """
    A class for generating additional features to enhance model performance through feature engineering.

    Methods:
        generate_interaction_terms(data, columns, degree=2, include_polynomial=False):
            Generates interaction terms for specified features up to a specified degree.
        binning(data, column, bins, labels=None, strategy='quantile'):
            Bins continuous variables into categories with multiple strategy options.
        create_polynomial_features(data, columns, degree=2):
            Creates polynomial features for specified columns.
        expand_categorical(data, column, threshold=0.05):
            Expands a categorical column into dummy variables based on a frequency threshold.
        group_and_aggregate(data, column, target, agg_func='mean'):
            Groups data by a specified column and creates an aggregated feature based on the target column.
        generate_lagged_features(data, column, lags):
            Generates lagged features for time series or ordered data.
        rolling_statistics(data, column, window, stats=['mean', 'std']):
            Computes rolling statistics (e.g., mean, std) for a specified column.
        detect_high_correlation(data, threshold=0.9):
            Identifies pairs of features with correlation above the specified threshold.
    """

    def generate_interaction_terms(self, data: pd.DataFrame, columns: list, degree: int = 2,
                                   include_polynomial: bool = False) -> pd.DataFrame:
        for i in range(2, degree + 1):
            for combo in combinations(columns, i):
                col_name = f"{'_x_'.join(combo)}"
                data[col_name] = data[list(combo)].prod(axis=1)

        if include_polynomial:
            for col in columns:
                for i in range(2, degree + 1):
                    col_name = f"{col}_pow_{i}"
                    data[col_name] = data[col] ** i

        return data

    def binning(self, data: pd.DataFrame, column: str, bins: int, labels=None,
                strategy: str = 'quantile') -> pd.DataFrame:
        if strategy == 'quantile':
            data[f'{column}_binned'] = pd.qcut(data[column], bins, labels=labels)
        elif strategy == 'uniform':
            data[f'{column}_binned'] = pd.cut(data[column], bins, labels=labels)
        elif strategy == 'custom':
            if not isinstance(bins, list):
                raise ValueError("For 'custom' strategy, 'bins' should be a list of bin edges.")
            data[f'{column}_binned'] = pd.cut(data[column], bins=bins, labels=labels)
        else:
            raise ValueError("Strategy should be one of 'quantile', 'uniform', or 'custom'.")

        return data

    def create_polynomial_features(self, data: pd.DataFrame, columns: list, degree: int = 2) -> pd.DataFrame:
        for col in columns:
            for i in range(2, degree + 1):
                data[f"{col}_pow_{i}"] = data[col] ** i
        return data

    def expand_categorical(self, data: pd.DataFrame, column: str, threshold: float = 0.05) -> pd.DataFrame:
        freq = data[column].value_counts(normalize=True)
        significant_values = freq[freq > threshold].index
        for value in significant_values:
            data[f"{column}_{value}"] = (data[column] == value).astype(int)

        data[f"{column}_other"] = (~data[column].isin(significant_values)).astype(int)
        return data

    def group_and_aggregate(self, data: pd.DataFrame, column: str, target: str, agg_func: str = 'mean') -> pd.DataFrame:
        agg_name = f"{column}_{agg_func}_{target}"
        grouped = data.groupby(column)[target].transform(agg_func)
        data[agg_name] = grouped
        return data

    def generate_lagged_features(self, data: pd.DataFrame, column: str, lags: int = 1) -> pd.DataFrame:
        """
        Generates lagged features for time series or ordered data.

        Parameters:
            data (pd.DataFrame): The input data.
            column (str): Column to create lagged features for.
            lags (int): Number of lagged features to create.

        Returns:
            pd.DataFrame: DataFrame with added lagged features.
        """
        for lag in range(1, lags + 1):
            data[f"{column}_lag_{lag}"] = data[column].shift(lag)
        return data

    def rolling_statistics(self, data: pd.DataFrame, column: str, window: int,
                           stats: list = ['mean', 'std']) -> pd.DataFrame:
        """
        Computes rolling statistics (e.g., mean, std) for a specified column.

        Parameters:
            data (pd.DataFrame): The input data.
            column (str): Column to compute rolling statistics for.
            window (int): The rolling window size.
            stats (list): List of statistics to compute (e.g., 'mean', 'std', 'min', 'max').

        Returns:
            pd.DataFrame: DataFrame with added rolling statistics features.
        """
        for stat in stats:
            if stat == 'mean':
                data[f"{column}_rolling_mean_{window}"] = data[column].rolling(window=window).mean()
            elif stat == 'std':
                data[f"{column}_rolling_std_{window}"] = data[column].rolling(window=window).std()
            elif stat == 'min':
                data[f"{column}_rolling_min_{window}"] = data[column].rolling(window=window).min()
            elif stat == 'max':
                data[f"{column}_rolling_max_{window}"] = data[column].rolling(window=window).max()
            else:
                raise ValueError("Supported statistics: 'mean', 'std', 'min', 'max'")
        return data

