from flask import Flask, render_template, request
import pandas as pd
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the dataset and machine learning model
data = pd.read_csv('final_dataset.csv')
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))

@app.route('/')
def index():
    """
    Render the home page with dropdown options for bedrooms, bathrooms, sizes, and zip codes.
    These options are derived from the dataset to ensure that users select valid inputs.
    """
    # Extract unique values for each category and sort them
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    # Render the 'index.html' template with these options
    return render_template('index.html', 
                           bedrooms=bedrooms, 
                           bathrooms=bathrooms, 
                           sizes=sizes, 
                           zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle the prediction logic when the user submits the form.
    It takes the form data, preprocesses it, checks for any unknown categories, and
    finally uses the pre-trained model to predict the price based on the inputs.
    """
    # Capture the user input from the form
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    # Create a DataFrame from the user input
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                               columns=['beds', 'baths', 'size', 'zip_code'])

    # Debugging: Print the raw input data
    print("Input Data:")
    print(input_data)

    # Convert 'baths' column to numeric, setting errors='coerce' to handle invalid data
    input_data['baths'] = pd.to_numeric(input_data['baths'], errors='coerce')

    # Convert all input data to the appropriate numeric types
    input_data = input_data.astype({
        'beds': int, 
        'baths': float, 
        'size': float, 
        'zip_code': int
    })

    # Handle any unknown categories in the input data
    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            print(f"Unknown categories in {column}: {unknown_categories}")
            # Replace unknown categories with the most common value in the original dataset
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    # Debugging: Print the processed input data
    print("Processed Input Data:")
    print(input_data)

    # Use the model to predict the price based on the processed input data
    prediction = model.predict(input_data)[0]

    # Return the prediction as a string (this will likely be displayed on the webpage)
    return str(prediction)

if __name__ == "__main__":
    # Run the Flask app in debug mode on port 5000
    app.run(debug=True, port=5000)
