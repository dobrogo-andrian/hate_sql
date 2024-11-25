import pandas as pd
import numpy as np
from dataclean.src.data_valydator import DataValidator
from dataclean.src.feature_engineering import FeatureEngineering
from dataclean.src.missing_data_handler import MissingDataHandler
from dataclean.src.outliner_handler import OutlierHandler
from dataclean.src.scaling_encoding_handler import ScalingEncodingHandler
from dataclean.src.transformation_normalization_handler import TransformationNormalizationHandler

class DataClean:

    def __init__(self, df: pd.DataFrame):
        self.df = df 

    def data_validator(self):
        dv = DataValidator(self.df)
        dv.describe_data()
        dv.get_basic_info()
        dv.summary_statistics()
        dv.missing_value_report()
        dv.duplicate_rows()
        dv.correlation_check()
        dv.unique_values_report()
        dv.check_cardinality()

    def missing_data_handler(self):
        ms = MissingDataHandler(self.df)
        ms.detect_missing()
        ms.identify_missing_patterns()
        ms.remove_missing()
        ms.summarize_missing_impact()
        

