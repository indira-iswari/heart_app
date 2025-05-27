import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the dataset
try:
    df = pd.read_csv('dataset.csv')  # Your dataset
except FileNotFoundError:
    print('Error: Dataset file not found.')
    exit(1)

# Prepare features and target
try:
    X = df.drop('target', axis=1)
    y = df['target']
except KeyError:
    print('Error: Target column not found in the dataset.')
    exit(1)

# Train model
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
except Exception as e:
    print(f'Error during model training: {e}')
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []
        for col in X.columns:
            value = request.form.get(col)
            if value is None:
                return render_template('index.html', prediction_text=f'Missing value for {col}.')
            features.append(float(value))

        prediction = model.predict([features])
        result = 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'
        return render_template('index.html', prediction_text=result)
    except ValueError:
        return render_template('index.html', prediction_text='Invalid input. Please enter numeric values.')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
