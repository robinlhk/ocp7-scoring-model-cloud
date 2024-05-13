"""
This is a boilerplate pipeline 'feature_processing'
generated using Kedro 0.19.5
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

def get_clean_features(df):
    # Replace infinity values with nans
    df = df.replace([np.inf, -np.inf], np.nan)
    # Remove columns with more than 80% missing values
    df = df.dropna(thresh=0.8 * df.shape[0], axis=1)
    # Remove columns with only one unique value
    df = df.loc[:, df.apply(pd.Series.nunique) != 1]

    return df


def process_features_for_ml(df, imputation_strategy='median'):
    # Drop the target from the training data
    if 'TARGET' in df.columns:
        train = df.drop(columns=['TARGET'])
    train = df.drop(columns=['SK_ID_CURR'])

    # Extract feature names
    feature_names = list(train.columns)
    # categorial_features = train.select_dtypes(include = ['object']).columns
    # numerical_features = train.select_dtypes(exclude = ['object']).columns

    # Median imputation of missing values
    imputer = SimpleImputer(strategy=imputation_strategy)
    train = imputer.fit_transform(train)

    # Scale numerical features to 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train)

    print('Training Features shape: ', train.shape)
    df1 = pd.DataFrame(train, columns=feature_names)
    df1['SK_ID_CURR'] = df['SK_ID_CURR']
    if 'TARGET' in df.columns:
        df1['TARGET'] = df['TARGET']
    return df1