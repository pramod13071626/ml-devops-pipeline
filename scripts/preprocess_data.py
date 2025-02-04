import pandas as pd

def preprocess_data(input_path, output_path):
    # Read the raw data
    data = pd.read_csv(input_path)
    
    # Basic preprocessing: Fill missing values
    data.fillna(method='ffill', inplace=True)
    
    # Save the cleaned data
    data.to_csv(output_path, index=False)
    print(f"Data preprocessed and saved to {output_path}")

if __name__ == "__main__":
    preprocess_data('data/raw_data.csv', 'data/clean_data.csv')
