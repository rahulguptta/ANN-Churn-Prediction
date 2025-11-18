#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# fixing seed
np.random.seed(42)
random.seed(42)


# In[3]:


# preprocessing libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# In[4]:


# metrics
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


# In[5]:


# tensor flow related
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy


# In[14]:


# tensorboard and callbacks
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

