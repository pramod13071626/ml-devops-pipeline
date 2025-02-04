import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(data_path, model_path):
    # Load the cleaned data
    data = pd.read_csv(data_path)
    
    # Simple example: Assuming last column is the target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_model('data/clean_data.csv', 'models/model.pkl')
