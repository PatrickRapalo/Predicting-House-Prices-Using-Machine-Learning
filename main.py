import train_model
import predict

if __name__ == "__main__":
    print("Starting house price prediction pipeline...")

    # Train model
    train_model.train_model()

    # Example prediction
    new_house = {
        "OverallQual": 7,
        "GrLivArea": 2000,
        "GarageCars": 2,
        "TotalBsmtSF": 1500,
        "YearBuilt": 2005,
        "LotArea": 8000
    }
    
    price = predict.predict(new_house)
    print(f"Predicted House Price: ${price:.2f}")
