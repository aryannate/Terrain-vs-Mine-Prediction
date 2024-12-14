# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset from a CSV file into a Pandas DataFrame
sonar_data = pd.read_csv('/content/sonar data.csv', header=None)

# Displaying the first few rows of the dataset
print(sonar_data.head())

# Checking the shape of the dataset (number of rows and columns)
print("Dataset shape:", sonar_data.shape)

# Generating descriptive statistics for the dataset
print("Dataset description:\n", sonar_data.describe())

# Checking the distribution of the target labels (column 60)
print("Value counts for target labels:\n", sonar_data[60].value_counts())

# Calculating the mean feature values grouped by the target label
print("Mean values grouped by target label:\n", sonar_data.groupby(60).mean())

# Splitting the dataset into features (X) and target label (Y)
X = sonar_data.drop(columns=60, axis=1)  # Features (all columns except the 60th)
Y = sonar_data[60]  # Target labels (60th column)

# Displaying the features and labels
print("Features:\n", X)
print("Labels:\n", Y)

# Splitting the data into training and testing sets (90% train, 10% test)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)

# Displaying the shapes of the training and testing datasets
print("Feature set shapes: Total =", X.shape, ", Training =", X_train.shape, ", Testing =", X_test.shape)

# Displaying training data (features and labels)
print("Training features:\n", X_train)
print("Training labels:\n", Y_train)

# Initializing the Logistic Regression model
model = LogisticRegression()

# Training the model with the training data
model.fit(X_train, Y_train)

# Predicting on the training data
X_train_prediction = model.predict(X_train)

# Calculating accuracy on the training data
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data:', training_data_accuracy)

# Predicting on the test data
X_test_prediction = model.predict(X_test)

# Calculating accuracy on the test data
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on testing data:', testing_data_accuracy)

# ----- Making a Prediction System -----

# Input data for prediction (example instance)
input_data = (
    0.0228, 0.0106, 0.0130, 0.0842, 0.1117, 0.1506, 0.1776, 0.0997, 0.1428, 0.2227,
    0.2621, 0.3109, 0.2859, 0.3316, 0.3755, 0.4499, 0.4765, 0.6254, 0.7304, 0.8702,
    0.9349, 0.9614, 0.9126, 0.9443, 1.0000, 0.9455, 0.8815, 0.7520, 0.7068, 0.5986,
    0.3857, 0.2510, 0.2162, 0.0968, 0.1323, 0.1344, 0.2250, 0.3244, 0.3939, 0.3806,
    0.3258, 0.3654, 0.2983, 0.1779, 0.1535, 0.1199, 0.0959, 0.0765, 0.0649, 0.0313,
    0.0185, 0.0098, 0.0178, 0.0077, 0.0074, 0.0095, 0.0055, 0.0045, 0.0063, 0.0039
)

# Converting input data to a NumPy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the array for prediction (1 instance with 60 features)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Making a prediction
prediction = model.predict(input_data_reshaped)
print("Prediction result:", prediction)

# Interpreting the prediction result
if prediction[0] == 'R':
    print('The object is a Rock')
else:
    print('The object is a Mine')
