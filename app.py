from flask import Flask, render_template, request, redirect, url_for
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import joblib

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

try:
    model = load_model('Model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
try:
    le = joblib.load('label_encoder.pkl')
    print("LabelEncoder loaded successfully!")
except Exception as e:
    print(f"Error loading LabelEncoder: {e}")


# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        return redirect(url_for('index'))
    return render_template('login.html')

# Route for the abstract page
@app.route('/charts')
def abstract():
    return render_template('charts.html')

# Route for the performance page
@app.route('/performance')
def performance():
    return render_template('performance.html')

# Route for the prediction page
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        result = "Prediction result here"  # Replace with actual prediction result
        return render_template('prediction.html', result=result)
    return render_template('prediction.html')

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=2.5, offset=0.6)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/result', methods=['GET', 'POST'])
def result():
    print("hi")
    if request.method == 'POST':
        
        if 'file' not in request.files:
            print("No file in the request.")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            print("No file selected.")
            return redirect(request.url)

        if file:
            # Ensure the file is saved correctly
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

            # Replace backslashes with forward slashes for compatibility
            file_path = file_path.replace("\\", "/")
            print(f"File path: {file_path}")  # Debugging line

            try:
                # Save the uploaded file temporarily
                file.save(file_path)  
                print(f"File saved to: {file_path}")  # Debugging line

                # Extract features and make predictions
                def prediction_(path_sound):
                    # Extract features from the audio file
                    data_sound = extract_features(path_sound)
                    if data_sound is None:
                        return "Error in feature extraction"
                    
                    # Prepare data for model prediction
                    X = np.array(data_sound).reshape(1, 40, 1)  # Adjust the shape for the model
                    pred_ = model.predict(X)
                    pred_ = np.argmax(pred_, axis=1)

                    # Make the prediction (inverse transform)
                    pred_class = le.inverse_transform(pred_)
                    return pred_class[0]

                # Make the prediction
                predicted_class = prediction_(file_path)
                print("Predicted class is:", predicted_class)
                print("type:",type(predicted_class))
                
                # Optionally, delete the file after prediction
                os.remove(file_path)

                # Pass the prediction result to the result page
                return render_template('result.html', prediction=predicted_class)

            except Exception as e:
                print(f"Error during file handling or prediction: {e}")
                return render_template('result.html', prediction_text="Error during prediction.")

    return render_template('result.html', prediction="No file uploaded.")


@app.route('/logout')
def logout():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5002)
