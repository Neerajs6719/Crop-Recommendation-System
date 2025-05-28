from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load("random_forest_crop_model.pkl")

# Create a label encoder (we'll need this to decode predictions)
label_encoder = LabelEncoder()
crop_labels = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 
               'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 
               'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 
               'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
label_encoder.fit(crop_labels)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Create DataFrame for prediction
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]
        
        return render_template('prediction_result.html', 
                             prediction=predicted_crop,
                             N=N, P=P, K=K, 
                             temperature=temperature, 
                             humidity=humidity, 
                             ph=ph, 
                             rainfall=rainfall)
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)