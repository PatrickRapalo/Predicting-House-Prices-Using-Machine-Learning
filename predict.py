import joblib
import pandas as pd
import data_preprocessing as dp

def predict(new_data):
    """Make price predictions using the trained model."""
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")

    # Convert new data into DataFrame (if necessary)
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])

    # Preprocess new data
    new_data_processed = preprocessor.transform(new_data)

    # Make prediction
    prediction = model.predict(new_data_processed)
    return prediction[0]

if __name__ == "__main__":
    # Example house data
    new_house = {
        "OverallQual": 7,
        "GrLivArea": 2000,
        "GarageCars": 2,
        "TotalBsmtSF": 1500,
        "YearBuilt": 2005,
        "LotArea": 8000
    }
    
    price = predict(new_house)
    print(f"Predicted House Price: ${price:.2f}")
