# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/Adithya-Siddam/nn-classification/assets/93427248/fb4cae88-8460-4b7c-a9d9-f1fd25cd2a74)

## DESIGN STEPS

### STEP 1:
Loading the dataset.

### STEP 2:
Checking the null values and converting the string datatype into integer or float datatype using label encoder

### STEP 3:
Split the dataset into training and testing

### STEP 4:
Create MinMaxScaler objects,fit the model and transform the dat

### STEP 5:
Build the Neural Network Model and compile the mode

### STEP 6:
Train the model with the training data

### STEP 7:
Plot the training loss and validation loss.Predicting the model through classification report,confusion matrix.Predict the new sample data.

## PROGRAM

### Name: S Adithya Chowdary.
### Register Number: 212221230100.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt
crux = pd.read_csv('customers.csv')
crux.columns
crux.dtypes
crux.shape
crux.isnull().sum()
crux_clear = crux.dropna(axis=0)
crux_clear.isnull().sum()
crux_clear.shape
crux_clear.dtypes
crux_clear['Gender'].unique()
crux_clear['Ever_Married'].unique()
crux_clear['Graduated'].unique()
crux_clear['Profession'].unique()
crux_clear['Spending_Score'].unique()
crux_clear['Work_Experience'].unique()
crux_clear['Segmentation'].unique()
crux_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
crux_enc = OrdinalEncoder(categories=crux_list)
crux_1 = crux_clear.copy()
crux_1.head()
crux_1[['Gender',
             'Ever_Married',
              'Graduated','Profession',
              'Spending_Score']] = crux_enc.fit_transform(crux_1[['Gender',
                                                                 'Ever_Married',
                                                                 'Graduated','Profession',
                                                                 'Spending_Score']])
crux_1.dtypes
crux_1 = crux_1.drop('ID',axis=1)
crux_1 = crux_1.drop('Var_1',axis=1)
crux_1.dtypes
le = LabelEncoder()
crux_1['Segmentation'] = le.fit_transform(crux_1['Segmentation'])
crux_1.dtypes

# Calculate the correlation matrix
corr = crux_1.corr()

# Plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)
sns.pairplot(crux_1)
crux_1.describe()

crux_1['Segmentation'].unique()
X=crux_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values
y1 = crux_1[['Segmentation']].values
crux_hot_enc = OneHotEncoder()
crux_hot_enc.fit(y1)
y1.shape
y = crux_hot_enc.transform(y1).toarray()
y.shape
y1[0]
y[0]
X.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)
X_train[0]
X_train.shape
scaler_crux = MinMaxScaler()
scaler_crux.fit(X_train[:,2].reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)
X_train_scaled[:,2] = scaler_crux.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_crux.transform(X_test[:,2].reshape(-1,1)).reshape(-1)
crux_model = Sequential([
    Dense(units = 4, input_shape=[8]),
    Dense(units = 8),
    Dense(units = 4, activation = 'softmax')
])
crux_model.compile(optimizer='adam',
                 loss= 'categorical_crossentropy',
                 metrics=['accuracy'])
crux_model.fit(x=X_train_scaled,y=y_train,
             epochs= 2000,
             batch_size= 50,
             validation_data=(X_test_scaled,y_test),
             )
metrics = pd.DataFrame(crux_model.history.history)
 metrics = pd.DataFrame(crux_model.history.history)
 metrics.tail()
 metrics.plot()
x_test_predictions = np.argmax(crux_model.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
x_single_prediction = np.argmax(crux_model.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))
```

## Dataset Information
## Costumers.csv:


![image](https://github.com/Adithya-Siddam/nn-classification/assets/93427248/c719e12d-c325-4688-8b1e-bf7c09cde0a9)

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/Adithya-Siddam/nn-classification/assets/93427248/664cf42a-e41a-496b-8ca1-185123f40758)

### Classification Report

![image](https://github.com/Adithya-Siddam/nn-classification/assets/93427248/b4835cd3-2a76-453c-9871-548fbef75146)

### Confusion Matrix

![image](https://github.com/Adithya-Siddam/nn-classification/assets/93427248/62c6b90d-a4bd-4eea-81f8-3096efd194f4)


### New Sample Data Prediction

![image](https://github.com/Adithya-Siddam/nn-classification/assets/93427248/6e16c545-caba-4c95-9863-2070daa9dc9b)

## RESULT
Thus,a neural network classification model for the given dataset is developed.

