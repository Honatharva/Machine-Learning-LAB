import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    slope = model.coef_[0]
    intercept = model.intercept_
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return slope, intercept, rmse, y_pred

def plot_regression(df, X_col, y_col, y_pred):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_col, y=y_col, data=df, label='Actual')
    plt.plot(df[X_col], y_pred, color='red', label='Regression Line')
    plt.title('Linear Regression: Salary vs Years of Experience')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main Execution ---

df_sal = load_data('C:/Files/Python/Salary_Data.csv')
X = df_sal[['YearsExperience']]
y = df_sal['Salary']

model = train_model(X, y)
slope, intercept, rmse, y_pred = evaluate_model(model, X, y)

print(f"Slope (Coefficient): {slope}")
print(f"Intercept: {intercept}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

plot_regression(df_sal, 'YearsExperience', 'Salary', y_pred)
