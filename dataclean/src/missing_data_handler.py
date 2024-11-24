# dataclean/missing_data_handler.py

import pandas as pd
import numpy as np


class MissingDataHandler:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def detect_missing(self):
        """
        Provides a detailed report on missing values by column and row, including
        percentages, total count, and summary statistics on missing data spread.
        """
        missing_by_column = self.data.isnull().sum()
        total_rows = len(self.data)
        percent_missing = (missing_by_column / total_rows) * 100
        missing_report = pd.DataFrame({
            "Missing Count": missing_by_column,
            "Percent Missing": percent_missing,
            "Data Type": self.data.dtypes
        })

        print("Detailed Missing Data Report by Column:")
        print(missing_report.sort_values(by="Percent Missing", ascending=False))

        return missing_report

    def identify_missing_patterns(self):
        """
        Identifies common patterns or clusters of missing data.
        Returns a DataFrame showing groups of rows with similar missing patterns.
        """
        # Check which columns each row has missing values in
        pattern_matrix = self.data.isnull().astype(int)
        unique_patterns = pattern_matrix.drop_duplicates()

        pattern_summary = pattern_matrix.groupby(list(pattern_matrix.columns)).size().reset_index(name="Count")
        pattern_summary['Pattern'] = pattern_summary.apply(
            lambda row: ', '.join([col for col, val in row.items() if val == 1]), axis=1)

        print("Identified Missing Data Patterns:")
        print(pattern_summary[['Pattern', 'Count']])

        return pattern_summary

    def fill_missing(self, strategy="mean", columns=None, value=None):
        """
        Fills missing values with more context: prints before/after statistics and
        supports conditional fills based on correlations with other columns.
        """
        if columns is None:
            columns = self.data.columns  # Default to all columns if none specified

        for col in columns:
            num_missing_before = self.data[col].isnull().sum()
            if num_missing_before == 0:
                continue

            if strategy == "mean":
                fill_value = self.data[col].mean()
            elif strategy == "median":
                fill_value = self.data[col].median()
            elif strategy == "mode":
                fill_value = self.data[col].mode()[0]
            elif strategy == "constant" and value is not None:
                fill_value = value
            else:
                raise ValueError("Invalid strategy or missing 'value' for 'constant' strategy.")

            # Fill missing values
            self.data[col] = self.data[col].fillna(fill_value)  # Avoid inplace=True
            num_missing_after = self.data[col].isnull().sum()

            print(
                f"Filled '{col}' with {strategy} ({fill_value}). Missing values before: {num_missing_before}, after: {num_missing_after}")

    def fill_based_on_correlation(self, target_column):
        """
        Fills missing values in the specified column by predicting them based on
        the most correlated feature.
        """
        # Compute correlation matrix
        correlations = self.data.corr()
        most_correlated_col = correlations[target_column].dropna().sort_values(ascending=False).index[1]

        if self.data[target_column].isnull().sum() > 0:
            fill_values = self.data.groupby(most_correlated_col)[target_column].transform(lambda x: x.fillna(x.mean()))
            self.data[target_column].fillna(fill_values, inplace=True)
            print(f"Filled missing values in '{target_column}' based on '{most_correlated_col}' correlation.")

    def remove_missing(self, axis=0, threshold=0.5):
        """
        Drops rows or columns with missing values exceeding the threshold,
        with an option to save them separately for further analysis.
        """
        initial_shape = self.data.shape
        missing_proportion = self.data.isnull().mean(axis=axis)

        if axis == 0:  # Drop rows
            rows_to_drop = missing_proportion[missing_proportion > threshold].index
            removed_data = self.data.loc[rows_to_drop]
            self.data.drop(rows_to_drop, inplace=True)
            print(f"Dropped {len(rows_to_drop)} rows exceeding {threshold * 100}% missing threshold.")
        elif axis == 1:  # Drop columns
            cols_to_drop = missing_proportion[missing_proportion > threshold].index
            removed_data = self.data[cols_to_drop]
            self.data.drop(cols_to_drop, axis=1, inplace=True)
            print(f"Dropped {len(cols_to_drop)} columns exceeding {threshold * 100}% missing threshold.")

        new_shape = self.data.shape
        print(f"Data shape changed from {initial_shape} to {new_shape}.")

        return removed_data  # Return dropped rows or columns for user review

    def summarize_missing_impact(self):
        """
        Provides an impact analysis on missing data, indicating which features
        have the strongest correlations with target variables, and recommends
        potential filling methods.
        """
        missing_cols = [col for col in self.data.columns if self.data[col].isnull().any()]
        impact_summary = {}

        for col in missing_cols:
            non_missing_corr = self.data.dropna().corr()[col].dropna().abs().sort_values(ascending=False)
            top_corr_col = non_missing_corr.index[0] if not non_missing_corr.empty else None
            recommended_fill = "mean" if self.data[col].dtype in [np.float64, np.int64] else "mode"

            impact_summary[col] = {
                "Most Correlated Feature": top_corr_col,
                "Recommended Fill Method": recommended_fill
            }

        impact_df = pd.DataFrame.from_dict(impact_summary, orient="index")
        print("Missing Data Impact Analysis and Fill Recommendations:")
        print(impact_df)

        return impact_df
