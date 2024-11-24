# dataclean/data_validation.py

import pandas as pd
import numpy as np
from scipy.stats import normaltest


class DataValidator:
    """
    Example usage:
        import pandas as pd
        from dataclean.data_validation import DataValidator

        # Load a sample dataset
        data = pd.read_csv('sample_data.csv')

        # Initialize the DataValidator
        validator = DataValidator(data)

        # Basic info about the dataset
        validator.get_basic_info()

        # Get summary statistics for numerical columns
        print(validator.summary_statistics())

        # Check for missing values
        print(validator.missing_value_report())

        # Detect duplicate rows
        print(validator.duplicate_rows())

        # Detect outliers in a specific column
        print(validator.outlier_detection('age'))

        # Check if a column has the expected data type
        validator.data_type_check('age', np.number)

        # describes all necessary info about dataset
        describe_data(data)
    """

    def __init__(self, data: pd.DataFrame):
        """Initialize with a pandas DataFrame."""
        self.data = data

    def describe_data(self):
        """Provides a comprehensive summary of the dataset, combining key insights from other validation methods."""

        # Dataset shape and column data types
        print("Basic Dataset Information:")
        print("--------------------------")
        self.get_basic_info()

        # Summary statistics
        print("\nSummary Statistics:")
        print("-------------------")
        print(self.data.describe())

        # Missing value report
        print("\nMissing Value Report:")
        print("---------------------")
        missing_report = self.missing_value_report()
        if missing_report.empty:
            print("No missing values.")
        else:
            print(missing_report)

        # Duplicate rows
        print("\nDuplicate Rows Report:")
        print("----------------------")
        self.duplicate_rows()

        # Unique values report
        print("\nUnique Values Report:")
        print("---------------------")
        self.unique_values_report()

        # High correlation check
        print("\nHigh Correlation Pairs (Threshold > 0.8):")
        print("-----------------------------------------")
        self.correlation_check(threshold=0.8)


        print("\nOutlier Summary for Numerical Columns:")
        print("-------------------------------------")
        for column in self.data.select_dtypes(include=[np.number]).columns:
            self.outlier_detection(column)

        # Distribution check for numerical columns
        print("\nDistribution Check for Numerical Columns:")
        print("----------------------------------------")
        for column in self.data.select_dtypes(include=[np.number]).columns:
            self.check_distribution(column)

        print(
            "\nData validation completed. Review the above details for any necessary cleaning or preprocessing steps.")

    def get_basic_info(self):
        """Prints basic information about the dataset, such as shape, columns, and data types."""
        print("Dataset Shape:", self.data.shape)
        print("\nData Types:")
        print(self.data.dtypes)
        print("\nFirst few rows of data:")
        print(self.data.head())

    def summary_statistics(self):
        """Returns summary statistics for numerical columns."""
        return self.data.describe()

    def missing_value_report(self):
        """Returns a report of missing values in each column."""
        missing_values = self.data.isnull().sum()
        total_values = len(self.data)
        missing_report = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': (missing_values / total_values) * 100
        })
        return missing_report[missing_report['Missing Values'] > 0]

    def duplicate_rows(self):
        """Identifies and returns duplicate rows in the dataset."""
        duplicates = self.data[self.data.duplicated()]
        if duplicates.empty:
            print("No duplicate rows found.")
        else:
            print(f"Found {len(duplicates)} duplicate rows.")
            return duplicates

    def outlier_detection(self, column):
        """Detects outliers in a specified numerical column using the IQR method."""
        q1 = self.data[column].quantile(0.25)
        q3 = self.data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]

        print(f"Number of outliers in '{column}': {len(outliers)}")
        return outliers

    def data_type_check(self, column, expected_type):
        """Checks if the specified column matches the expected data type."""
        if not np.issubdtype(self.data[column].dtype, expected_type):
            print(f"Column '{column}' has incorrect type. Expected {expected_type}, got {self.data[column].dtype}")
        else:
            print(f"Column '{column}' matches the expected type {expected_type}.")


    def validate_schema(self, expected_schema):
        """
        Validates if the dataset matches the expected schema.
        `expected_schema` is a dictionary where keys are column names and values are data types.
        """
        issues = {}
        for column, dtype in expected_schema.items():
            if column not in self.data.columns:
                issues[column] = "Column missing"
            elif not np.issubdtype(self.data[column].dtype, dtype):
                issues[column] = f"Incorrect type. Expected {dtype}, got {self.data[column].dtype}"
        if not issues:
            print("Schema validation passed.")
        else:
            print("Schema validation issues:", issues)
        return issues

    def validate_range(self, column, min_value, max_value):
        """
        Checks if values in a specified column fall within a given range.
        """
        out_of_range = self.data[(self.data[column] < min_value) | (self.data[column] > max_value)]
        if out_of_range.empty:
            print(f"All values in '{column}' are within the specified range.")
        else:
            print(f"{len(out_of_range)} values in '{column}' are out of the range [{min_value}, {max_value}].")
        return out_of_range

    def validate_categories(self, column, expected_categories):
        """
        Checks if a categorical column contains only the expected categories.
        """
        unique_values = set(self.data[column].unique())
        unexpected = unique_values - set(expected_categories)
        if not unexpected:
            print(f"All values in '{column}' are within the expected categories.")
        else:
            print(f"Unexpected categories in '{column}': {unexpected}")
        return unexpected

    def correlation_check(self, threshold=0.8):
        """
        Identifies pairs of features with high correlation.
        Only considers columns that can be converted to floats.
        """
        # Select columns that can be converted to numeric
        numeric_data = self.data.select_dtypes(include=[float, int])

        # Compute the correlation matrix for numeric columns
        corr_matrix = numeric_data.corr()

        # Find pairs with high correlation above the threshold
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

        if high_corr_pairs:
            for col1, col2, corr in high_corr_pairs:
                print(f"{col1} - {col2}: {corr:.2f}")
        else:
            print("No highly correlated pairs found.")
        return high_corr_pairs

    def check_distribution(self, column, alpha=0.05):
        """
        Tests if a numerical column follows a normal distribution.
        Uses the D'Agostino and Pearson's test.
        """
        k2, p_value = normaltest(self.data[column].dropna())
        if p_value < alpha:
            print(f"Column '{column}' does not follow a normal distribution (p-value: {p_value}).")
        else:
            print(f"Column '{column}' follows a normal distribution (p-value: {p_value}).")
        return p_value

    def unique_values_report(self):
        """
        Returns the number of unique values for each column in the dataset.
        """
        unique_counts = self.data.nunique()
        print("Unique values per column:")
        print(unique_counts)
        return unique_counts

    def check_cardinality(self, threshold=50):
        """
        Identifies columns with high cardinality (number of unique values greater than a specified threshold).
        """
        high_cardinality_columns = [col for col in self.data.columns if self.data[col].nunique() > threshold]
        print("Columns with high cardinality:", high_cardinality_columns)
        return high_cardinality_columns
