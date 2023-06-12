from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
sc = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        test_train_split = float(request.form['test_train_split'])
        algorithm = request.form['algorithm']
        interface_variable = request.form['interface_variable']
    
        dataset = pd.read_csv(file)
        if 'target' not in dataset.columns:
            return "Error: 'target' column not found in the dataset"
        
        label_encoder = LabelEncoder()
        dataset['target'] = label_encoder.fit_transform(dataset['target'])
        X = dataset.drop('target', axis=1)
        y = dataset['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_train_split)
        
        if algorithm == 'Random Forest':
            model = RandomForestClassifier()
        elif algorithm == 'LR':
            model = LogisticRegression(solver='liblinear', multi_class='ovr')
        elif algorithm == 'LDA':
            model = LinearDiscriminantAnalysis()
        elif algorithm == 'KNN':
            model = KNeighborsClassifier()
        elif algorithm == 'CART':
            model = DecisionTreeClassifier()
        elif algorithm == 'NB':
            model = GaussianNB()
        elif algorithm == 'SVM':
            model = SVC(gamma='auto')

        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        model.fit(X_train, y_train)
        accuracy = (accuracy_score(y_test, model.predict(X_test))*100)
        print("accuracy score is ",accuracy)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        validation_score = 1-np.mean(cv_scores)
        print("validation score is ",validation_score)

        with open('model.pkl', 'wb') as file:
            pickle.dump((model, label_encoder, sc), file)
    
        return "Model trained successfully!"
    
    except Exception as e:
        return "Error: " + str(e)
        
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form['input_data']
    pickle_file = request.files['pickle_file']
    
    try:
        with open('model.pkl', 'rb') as file:
            model, label_encoder, sc = pickle.load(file)
    except ValueError:
        return "Error: Invalid pickle file"
    input_data = np.array([float(value) for value in input_data.split(',')])
    input_data = input_data.reshape(1, -1) 
    try:
        input_data = sc.transform(input_data)
    except ValueError:
        return "Error: Invalid input data"
    
    prediction = model.predict(input_data)
    prediction = label_encoder.inverse_transform(prediction)[0]
    return f"Prediction: {prediction}"

if __name__ == '__main__':
    app.run(debug=True)
