import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


class ScalingEncodingHandler:
    """
    A class for scaling numerical features and encoding categorical variables with added functionality.

    Methods:
        scale_features(data, columns=None, method='standard', fillna_strategy=None):
            Scales specified or automatically detected numerical columns.
        encode_categorical(data, columns=None, method='onehot', drop_first=False, fillna_strategy=None):
            Encodes specified or automatically detected categorical columns.
    """

    def scale_features(self, data: pd.DataFrame, columns: list = None, method: str = 'standard',
                       fillna_strategy=None) -> pd.DataFrame:
        """
        Scales specified or automatically detected numerical columns using the chosen method.

        Parameters:
            data (pd.DataFrame): The input data.
            columns (list): List of numerical columns to scale. If None, automatically detects numerical columns.
            method (str): Scaling method, 'standard' for StandardScaler or 'minmax' for MinMaxScaler.
            fillna_strategy (str or float): Strategy for handling NaN values ('mean', 'median', 'zero', or a float to fill).

        Returns:
            pd.DataFrame: DataFrame with scaled features.
        """
        # Automatically detect numerical columns if none are specified
        if columns is None:
            columns = data.select_dtypes(include=['number']).columns.tolist()

        # Handle NaN values
        if fillna_strategy is not None:
            if fillna_strategy == 'mean':
                data[columns] = data[columns].fillna(data[columns].mean())
            elif fillna_strategy == 'median':
                data[columns] = data[columns].fillna(data[columns].median())
            elif fillna_strategy == 'zero':
                data[columns] = data[columns].fillna(0)
            elif isinstance(fillna_strategy, (int, float)):
                data[columns] = data[columns].fillna(fillna_strategy)
            else:
                raise ValueError("Invalid fillna_strategy. Choose 'mean', 'median', 'zero', or a numeric value.")

        # Choose scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid method. Choose 'standard' or 'minmax'.")

        # Apply scaling
        data[columns] = scaler.fit_transform(data[columns])
        return data

    def encode_categorical(self, data: pd.DataFrame, columns: list = None, method: str = 'onehot',
                           drop_first: bool = False, fillna_strategy: str = None) -> pd.DataFrame:
        """
        Encodes specified or automatically detected categorical columns using chosen encoding.

        Parameters:
            data (pd.DataFrame): The input data.
            columns (list): List of categorical columns to encode. If None, automatically detects object-type columns.
            method (str): Encoding method, 'onehot' for OneHotEncoding or 'label' for LabelEncoding.
            drop_first (bool): If True, drops the first level of encoded columns (useful for linear models).
            fillna_strategy (str): Strategy for handling NaN values in categorical columns ('mode' or a specific category name).

        Returns:
            pd.DataFrame: DataFrame with encoded categorical variables.
        """
        # Automatically detect categorical columns if none are specified
        if columns is None:
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Handle NaN values
        if fillna_strategy is not None:
            if fillna_strategy == 'mode':
                data[columns] = data[columns].fillna(data[columns].mode().iloc[0])
            elif isinstance(fillna_strategy, str):
                data[columns] = data[columns].fillna(fillna_strategy)
            else:
                raise ValueError("Invalid fillna_strategy. Choose 'mode' or provide a string category to fill.")

        # Apply encoding
        if method == 'onehot':
            data = pd.get_dummies(data, columns=columns, drop_first=drop_first)
        elif method == 'label':
            for col in columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
        else:
            raise ValueError("Invalid method. Choose 'onehot' or 'label'.")

        return data
