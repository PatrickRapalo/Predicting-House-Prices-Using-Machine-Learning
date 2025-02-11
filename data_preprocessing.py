
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def load_data(filepath="ames_house_data.csv"):
    """Load dataset from CSV."""
    return pd.read_csv(filepath)


def preprocess_data(df):
    """Preprocess dataset: handle missing values, encode categorical data, scale numerical data."""
    
    # Select features and target variable
    X = df.drop(columns=["SalePrice"])  # Features
    y = df["SalePrice"]  # Target

    # Identify numerical and categorical features
    num_features = X.select_dtypes(include=['int64', 'float64']).columns
    cat_features = X.select_dtypes(include=['object']).columns

    # Define transformers
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    # Fit-transform the data
    X_preprocessed = preprocessor.fit_transform(X)

    # Save the preprocessor for future use
    joblib.dump(preprocessor, "preprocessor.pkl")

    return X_preprocessed, y

if __name__ == "__main__":
    df = load_data()
    X, y = preprocess_data(df)




