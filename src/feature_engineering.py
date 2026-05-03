from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "scope1_plus_scope2_location_mt"

NUMERIC_FEATURES = [
    "reporting_year",
    "scope1_mt_co2e",
    "scope2_location_mt_co2e",
    "emissions_intensity_mt_per_musd",
    "revenue_usd_millions",
    "net_zero_target_year",
]

CATEGORICAL_FEATURES = [
    "country",
    "region",
    "sector",
    "industry",
    "third_party_verified",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_features_and_target(df):
    """Split a modeling dataframe into features and target."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y


def build_preprocessor():
    """Build preprocessing steps for numeric and categorical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor
