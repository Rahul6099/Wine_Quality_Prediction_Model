#!/usr/bin/env python
# coding: utf-8

# # IMPORTING THE ESSENTIAL LIBRARIES

# In[14]:


# Importing essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Choose your machine learning algorithm(s)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Model evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# For deep learning (if needed)
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# Ignore warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility (optional)
np.random.seed(0)



# In[ ]:


# pip install numpy pandas matplotlib seaborn scikit-learn tensorflow


# In[12]:


# Importing essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning libraries for regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Data visualization settings (optional)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

# Ignore warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility (optional)
np.random.seed(0)


# # SELECTING THE DATA PATH
# 

# In[10]:


file_path = r'C:\Users\engin\Downloads\wineQT.csv'



# # DEFINING THE DATA

# In[11]:


# Read the CSV file into a DataFrame
wineQT = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to verify the import
wineQT.head()



# ## Exploratry Data Analysis
# 

# In[15]:


wineQT.shape


# In[16]:


wineQT.info()


# In[17]:


wineQT.isnull().sum()


# In[18]:


wineQT.describe()


# In[21]:


wineQT['quality'].value_counts()


# In[35]:


sns.countplot(data=wineQT, x='quality')
plt.show()


# In[42]:


wineQT.hist(bins=100, figsize=(25,25))
plt.show()


# In[46]:


plt.figure(figsize=(10, 7))
sns.heatmap(wineQT.corr(), annot=True)
plt.title('Correlation Between the Columns')
plt.show()


# In[50]:


wineQT.corr()['quality'].sort_values()


# In[52]:


sns.barplot(x=wineQT['quality'], y=wineQT['alcohol'])
plt.title('Bar Plot of Quality vs Alcohol')
plt.show()


# # Data Processing
# 

# In[54]:


wineQT['quality'] = wineQT.apply(lambda x: 1 if x['quality'] >= 7 else 0, axis=1)


# In[55]:


wineQT['quality'].value_counts()


# In[56]:


x = wineQT.drop('quality', axis=1)
y = wineQT['quality']


# In[57]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3,random_state= 42)


# In[58]:


print ("x_train", x_train.shape)
print ("y_train", y_train.shape)
print ("x_test", x_test.shape)
print ("y_test", x_test.shape)


# # Model Training

# ### logistic regression model

# In[61]:


logreg = LogisticRegression()  # Note the capital 'L' in LogisticRegression
logreg.fit(x_train, y_train)  # Corrected variable names 'x_train' and 'y_train'
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(y_test, logreg_pred)  # Reversed the order of arguments

print("Test accuracy is: {:.2f}%".format(logreg_acc * 100))  # Fixed the format string


# In[63]:


print(classification_report(y_test, logreg_pred))


# In[66]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.style.use('classic')
cm = confusion_matrix(y_test, logreg_pred, labels=logreg.classes_)  # Fixed 'labels' and 'logreg.classes_'
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)  # Fixed 'display_labels'

print("TN", cm[0, 0])  # Fixed indices
print("FN", cm[1, 0])  # Fixed indices
print("TP", cm[1, 1])  # Fixed indices
print("FP", cm[0, 1])  # Fixed indices



# In[67]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.style.use('classic')
cm = confusion_matrix(y_test, logreg_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)

# Plot the confusion matrix as a heatmap
disp.plot(cmap='Blues', values_format='d')

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# # Decision Tree Model

# In[68]:


# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=0)


# In[69]:


# Train the model on the training data
dt_classifier.fit(x_train, y_train)


# In[70]:


# Make predictions on the test data
y_pred = dt_classifier.predict(x_test)


# In[71]:


# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[72]:


# Generate and print the classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)


# In[73]:


# Generate and print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# In[84]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Make predictions on the test data
dt_pred = dt_classifier.predict(x_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, dt_pred)

# Set the figure size for the confusion matrix plot
plt.figure(figsize=(8, 6))

# Create a ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_classifier.classes_)

# Plot the confusion matrix as a heatmap
disp.plot(cmap='Blues', values_format='d')

# Set the title and axis labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.show()


# # LINEAR REGRESSION MODEL
# 

# In[89]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[90]:


# Load your dataset (replace 'your_data.csv' with your dataset file)
# Read the CSV file into a DataFrame
wineQT = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to verify the import
wineQT.head()


# In[95]:


# Select your features (X) and target variable (y)
X = wineQT[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
y = wineQT['quality']


# In[96]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[97]:


# Create a Linear Regression model
model = LinearRegression()


# In[98]:


# Train the model on the training data
model.fit(X_train, y_train)


# In[99]:


# Make predictions on the test data
y_pred = model.predict(X_test)


# In[100]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)



# In[101]:


# Display regression results
print("Mean Squared Error:", mse)
print("R-squared (R2) Score:", r2)


# In[102]:


# Plot actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values in Linear Regression")
plt.show()


# # Interpreting the results of the linear regression analysis:
# 
# The Mean Squared Error (MSE) is approximately 0.380, indicating the model's predictions are off by about 0.38 units squared, on average. Lower MSE values are generally preferred, but the assessment of whether this is good or bad depends on the specific context of the data and the problem.
# 
# The R-squared (R2) score is about 0.317, implying that roughly 31.7% of the variance in the target variable is explained by the model. An R2 score closer to 1 suggests a better model fit, while an R2 of 0 indicates that the model doesn't explain any of the variance.
# 
# In summary, the model seems to have some predictive power, but it only explains a portion of the variance in the target variable. There may be other unaccounted factors contributing to the variance. Further feature engineering, different model selection, or additional data collection could potentially enhance the model's performance.

# In[ ]:




