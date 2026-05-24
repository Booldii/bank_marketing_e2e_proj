import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import OrdinalEncoder as ce_encoder

NUMERIC_COLS = [
    'age', 'campaign', 'previous', 'cons.price.idx',
    'cons.conf.idx', 'euribor3m'
]
OHE_COLS = [
    'job', 'marital', 'housing', 'loan',
    'contact', 'day_of_week', 'poutcome'
]
WINSORIZE_COLS = ['age', 'campaign']

class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, limits=(0.001, 0.001)):
        self.limits = limits
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.lower_bounds_ = X_df.quantile(self.limits[0]).values
        self.upper_bounds_ = X_df.quantile(1 - self.limits[1]).values
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for i in range(X_copy.shape[1]):
            X_copy[:, i] = np.clip(X_copy[:, i],self.lower_bounds_[i], self.upper_bounds_[i])
        return X_copy

    def get_feature_names_out(self, input_features=None):
        return input_features


EDU_MAPPING= {
        'illiterate': 0,
        'basic.4y': 0,
        'basic.6y': 0,
        'basic.9y': 0,
        'high.school': 1,
        'professional.course': 2,
        'university.degree': 3,
        np.nan: np.nan
}

MONTH_CATS = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun',
    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
]

def get_preprocessor():
    """"""

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    winsorize_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('winsorizer', WinsorizerTransformer(limits=(0.001, 0.001))),
        ('scaler', StandardScaler())
    ])

    ohe_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    edu_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent').set_output(transform='pandas')),
        ('encoder', ce_encoder(mapping=[{'col': 'education', 'mapping': EDU_MAPPING}], handle_unknown='ignore'))
    ])

    month_pipeline = Pipeline(steps=[
        ('encoder', OrdinalEncoder(categories=[MONTH_CATS], handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])

    num_no_winsorize = [col for col in NUMERIC_COLS if col not in WINSORIZE_COLS]

    ct = ColumnTransformer(
        transformers=[
            ('num_std', numeric_pipeline, num_no_winsorize),
            ('num_win', winsorize_pipeline, WINSORIZE_COLS),
            ('cat_ohe', ohe_pipeline, OHE_COLS),
            ('cat_edu', edu_pipeline, ['education']),
            ('cat_month', month_pipeline, ['month'])
        ],
        remainder='passthrough'
    )
    return ct