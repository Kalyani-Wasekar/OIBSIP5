##########  SALES PREDICTION  ##########

print('SALES PREDICTION')
#import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Load the data
data = pd.read_csv('task5salespredictiondata.csv')

#print the dataset
print(data)

#Split the data into features (X) and target variable (y)
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create a linear regression model
model = LinearRegression()

#Train the model
model.fit(X_train, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test)

#Print the actual vs. predicted values
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(result_df)

#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

#Visualize the predictions
plt.scatter(X_test['TV'], y_test, color='red', label='Actual')
plt.scatter(X_test['TV'], y_pred, color='blue', label='Predicted')
plt.xlabel('TV Advertising')
plt.ylabel('Sales')
plt.legend()
plt.show()
