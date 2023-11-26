from sklearn.impute import SimpleImputer #handling missing values
from sklearn.preprocessing import StandardScaler # handling feature scaling
from sklearn.preprocessing import OrdinalEncoder # ordinal Encoding
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys, os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object
## Data transformation config
@dataclass
class DataTransformationConfig:
    preprocessor_ob_file = os.path.join("artifacts", "preprocessor.pkl")
    


## Data ingestion class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info("Data transformation initiated")
            categorical_columns = ['cut', 'color', 'clarity']
            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
            
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info("Pipeline initiated")
            num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
                    ]
                )

            cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinalencoder", OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ("scaler", StandardScaler())
                    ]
                )

            preprocessor = ColumnTransformer(
            [
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
                    ]
                )
            
            return preprocessor
                   
        except Exception as e:
            logging.info("Error in Data tranasformation")
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            
            logging.info("Read train and test data completed")
            logging.info(f"train dataframe head : \n {train_df.head().to_string()}")
            logging.info(f"test dataframe head : \n {test_df.head().to_string()}")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformation_object()
            
            target_column_name = "price"
            drop_columns = [target_column_name, "id"]
            
            ## features into independent and dependent features
            
            input_feature_train_df = train_df.drop(labels=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(labels=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            ## apply transformation
            
            input_features_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_features_test_array = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.c_[input_features_train_array, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_array, np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file,
                obj=preprocessing_obj
            )
            logging.info("Proprocessor pickle is created and saved")
            
            return (
                train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file
            )
        
        except Exception as e:
            logging.info("Error in initiate data transformation obj")
            raise CustomException(e, sys)
            
            
            
        


