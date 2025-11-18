#!/usr/bin/env python
# coding: utf-8

# In[7]:


# libraries
import streamlit as st 

# common libraries

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import random
import pickle
import datetime


# In[9]:


# fixing seed
np.random.seed(42)
random.seed(42)


# In[4]:


# preprocessing libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# In[7]:


# metrics
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


# In[12]:


# tensor flow related
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy


# In[9]:


# Loading all model and pickle files
with open('label_encoder_gender.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)

with open('onehot_encoder_geography.pkl', 'rb') as file:
    geo_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

model = load_model('model.keras')


# In[ ]:


st.title("Churn Prediction")

geography = st.selectbox('Geography', geo_encoder.categories_[0])
gender = st.selectbox('Gender', gender_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
number_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# In[ ]:


# data transformation
input_data = pd.DataFrame({'CreditScore':credit_score,
			   'Geography':geography,
                           'Gender':gender,
                           'Age':age,
			   'Tenure':tenure,
                           'Balance':balance,
                           'NumOfProducts':number_of_products,
                           'HasCrCard':has_cr_card,
                           'IsActiveMember':is_active_member,
			   'EstimatedSalary':estimated_salary
                          }, 
index = [0]
)

geo_encoded = geo_encoder.transform(input_data[['Geography']])
encoded_df = pd.DataFrame(geo_encoded.toarray(), columns = geo_encoder.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.drop('Geography', axis = 1), encoded_df], axis = 1)
input_data['Gender'] = gender_encoder.transform(input_data['Gender'])
final_data = scaler.transform(input_data)


# In[ ]:


#prediction
prediction = model.predict(final_data)
prob = prediction[0][0]
st.write(f"probability: {prob:.2f}")

if prob>0.5:
    result = 'churn'
else:
    result = 'Not churn'
    
st.write(result)

