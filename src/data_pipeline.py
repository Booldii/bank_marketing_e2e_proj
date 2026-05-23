import pandas as pd
import numpy as np
from scipy.stats import mstats

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer


NUMERIC_COLS = [
    'age', 'campaign', 'previous', 'cons.price.idx',
    'cons.conf.idx', 'euribor3m'
]
OHE_COLS = [
    'job', 'marital', 'housing', 'loan',
    'contact', 'day_of_week', 'poutcome'
]

WINSORIZE_COLS = ['age', 'campaign']

EDUCATION_MAP = {
    'illiterate': 0, 'basic.4y': 0, 'basic.6y': 0, 'basic.9y': 0,
    'high.school': 1, 'professional.course': 2, 'university.degree': 3,
    np.nan: np.nan
}

MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    np.nan: np.nan
}

def apply_winsorization(X, limits=(0.001, 0.001)):
    X_copy = np.copy(X)
    for i in range(X_copy.shape[1]):
        X_copy[:, i] = mstats.winsorize(X_copy[:, i], limits=limits)
    return X_copy

def map_ordinal_features(X, mapping_dict):
    return pd.DataFrame(X).replace(mapping_dict).values.astype(float)

numeric_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ]
)

winsorize_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('winsorizer', FunctionTransformer(apply_winsorization, kw_args={'limits': (0.001, 0.001)}))
])

ohe_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

edu_pipeline = Pipeline(steps=[
    ('mapper', FunctionTransformer(map_ordinal_features, kw_args={'mapping_dict': EDUCATION_MAP})),
    ('imputer', SimpleImputer(strategy='median'))
])

month_pipeline = Pipeline(steps=[
    ('mapper', FunctionTransformer(map_ordinal_features, kw_args={'mapping_dict': MONTH_MAP})),
    ('imputer', SimpleImputer(strategy='most_frequent'))
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

if __name__ == "__main__":
    df = pd.read_csv(r'../data/raw/bank-additional-full.csv', sep = ';')
    df['y'] = np.where(df['y'] == 'yes', 1, 0)

    df = df.replace('unknown', np.nan)

    X_cols = NUMERIC_COLS + OHE_COLS + ['education', 'month']
    X = df[X_cols]
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_processed = ct.fit_transform(X_train)
    X_test_processed = ct.transform(X_test)

    print(f"Kształt X_train przed: {X_train.shape}")
    print(f"Kształt X_train po transformacjach: {X_train_processed.shape}")
