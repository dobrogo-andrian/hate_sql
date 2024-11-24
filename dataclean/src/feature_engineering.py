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

    def generate_interaction_terms(self, data: pd.DataFrame, columns: list, degree: int = 2, include_polynomial: bool = False) -> pd.DataFrame:
        """
    Generates interaction terms and polynomial features for the given DataFrame.

    This function creates interaction terms for the specified columns of the input DataFrame
    by multiplying combinations of the columns up to the specified degree. Additionally, if requested, 
    it can generate polynomial features for each column up to the specified degree.

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame on which interaction terms and polynomial features will be generated.
    
    columns : list
        A list of column names to create interaction terms and polynomial features from.
    
    degree : int, optional, default: 2
        The maximum degree of interaction terms and polynomial features. 
        Interaction terms will be generated for combinations up to this degree.
    
    include_polynomial : bool, optional, default: False
        If True, polynomial features (i.e., power terms) for each column will be generated. 
        If False, no polynomial features will be created.

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with added interaction terms and, optionally, polynomial features.

        """
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

    def binning(self, data: pd.DataFrame, column: str, bins: int, labels=None, strategy: str = 'quantile') -> pd.DataFrame:
        """
    Bins a given column in the DataFrame into discrete intervals based on the specified strategy.

    This function allows you to bin a continuous column into a specified number of intervals (bins) 
    using different binning strategies such as quantiles, uniform intervals, or custom bin edges.

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame on which the binning will be performed.
    
    column : str
        The name of the column to be binned.
    
    bins : int or list
        The number of bins to divide the data into if using the 'quantile' or 'uniform' strategy.
        If using the 'custom' strategy, this should be a list of bin edges.
    
    labels : list, optional, default: None
        Labels for the bins. If None, integer labels are used. The length of the labels list 
        should match the number of bins.
    
    strategy : str, optional, default: 'quantile'
        The binning strategy to be used. Options are:
            - 'quantile': Bins the data into quantiles, ensuring that each bin contains 
              approximately the same number of observations.
            - 'uniform': Bins the data into equal-width intervals.
            - 'custom': Bins the data based on custom bin edges specified in the 'bins' parameter.

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with an additional column containing the binned values.
        The new column is named '{column}_binned'.

    Raises:
    -------
    ValueError:
        If an invalid strategy is provided or if 'bins' is not a list when using the 'custom' strategy.
    """
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
        """
    Generates polynomial features for specified columns in the DataFrame.

    This function creates new columns in the DataFrame, each representing the powers of the specified columns
    up to the given degree. Polynomial features are added as new columns with names indicating the original 
    column name followed by the power (e.g., 'col_pow_2', 'col_pow_3').

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame where the polynomial features will be added.
    
    columns : list
        A list of column names for which the polynomial features will be generated.
    
    degree : int, optional, default: 2
        The maximum degree of the polynomial features. The function will generate features from degree 2 up to 
        the specified degree for each column.

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with additional columns representing polynomial features for the specified columns.
    """
        for col in columns:
            for i in range(2, degree + 1):
                data[f"{col}_pow_{i}"] = data[col] ** i
        return data

    def expand_categorical(self, data: pd.DataFrame, column: str, threshold: float = 0.05) -> pd.DataFrame:
        """
    Expands a categorical column into multiple binary columns based on value frequency.

    This function converts a categorical column into multiple binary columns, where each new column
    represents a specific value in the original column. If the frequency of a value is greater than the 
    specified threshold, a binary column is created for that value. All other values are grouped into 
    a column named '{column}_other'.

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame containing the categorical column to be expanded.
    
    column : str
        The name of the categorical column to be expanded.
    
    threshold : float, optional, default: 0.05
        The frequency threshold for determining which values to expand into separate binary columns.
        Only values with a frequency greater than this threshold will have their own binary column.
        All other values will be combined into the '{column}_other' column.

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with additional binary columns representing each significant category 
        in the original column. An additional column for the 'other' category is also added.
    """
        freq = data[column].value_counts(normalize=True)
        significant_values = freq[freq > threshold].index
        for value in significant_values:
            data[f"{column}_{value}"] = (data[column] == value).astype(int)

        data[f"{column}_other"] = (~data[column].isin(significant_values)).astype(int)
        return data

    def group_and_aggregate(self, data: pd.DataFrame, column: str, target: str, agg_func: str = 'mean') -> pd.DataFrame:
        """
    Groups the data by a specified column and performs an aggregation on the target column.

    This function groups the DataFrame by the specified column, then aggregates the target column 
    using the specified aggregation function (e.g., 'mean', 'sum'). The result is added to the DataFrame
    as a new column with a name indicating the column, aggregation function, and target.

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame to perform the grouping and aggregation on.
    
    column : str
        The column by which the data will be grouped.
    
    target : str
        The column to be aggregated after grouping by the specified column.
    
    agg_func : str, optional, default: 'mean'
        The aggregation function to apply to the target column. Common options include:
        - 'mean' : Mean of the target column for each group.
        - 'sum' : Sum of the target column for each group.
        - Other aggregation functions supported by pandas, such as 'median', 'std', etc.

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with a new column containing the aggregated values for each group.
    """
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

