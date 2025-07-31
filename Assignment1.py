import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

# Load data
data = pd.read_csv("used_cars_data.csv")
data = data.drop(['S.No.'], axis=1)

# Car age and validity
current_year = date.today().year
data['Car_Age'] = current_year - data['Year']
data['Validity'] = 15 - data['Car_Age']

# Extract brand and model
data['Brand'] = data['Name'].str.split().str.get(0)
data['Model_1'] = data['Name'].str.split().str.get(1)
data['Model_2'] = data['Name'].str.split().str.get(2)
data['Model_1'] = data['Model_1'].fillna('')
data['Model_2'] = data['Model_2'].fillna('')
data['Model'] = data['Model_1'] + data['Model_2']
data = data.drop(['Model_1', 'Model_2'], axis=1)

# Standardize brand names
data["Brand"].replace({"ISUZU": "Isuzu", "Mini": "Mini Cooper", "Land": "Land Rover"}, inplace=True)

# Print brand names
print(data['Brand'].unique())

# Identify categorical and numerical columns
cat_cols = data.select_dtypes(include=['object']).columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()

print("\nCategorical Variables:")
print(cat_cols)
print("\nNumerical Variables:")
print(num_cols)

# Bar chart - Transmission Types
transmission_counts = data['Transmission'].value_counts()
plt.figure(figsize=(6, 4))
transmission_counts.plot(kind='bar', color='orange')
plt.title('Distribution of Transmission Types')
plt.xlabel('Transmission Type')
plt.ylabel('Number of Cars')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Pie chart - Transmission Types
plt.figure(figsize=(6, 6))
transmission_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Transmission Type Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()

# Histogram - Car Prices
plt.figure(figsize=(8, 5))
data['Price'].dropna().plot(kind='hist', bins=30, color='purple', edgecolor='black')
plt.title('Histogram of Car Prices')
plt.xlabel('Price (in Lakhs)')
plt.ylabel('Number of Cars')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Histogram - Mileage by Fuel Type
fuel_types = data['Fuel_Type'].unique()
plt.figure(figsize=(10, 6))
for fuel in fuel_types:
    subset = data[data['Fuel_Type'] == fuel]
    plt.hist(subset['Mileage'].dropna(), bins=25, alpha=0.6, label=fuel)

plt.title('Mileage Distribution by Fuel Type')
plt.xlabel('Mileage (kmpl or km/kg)')
plt.ylabel('Number of Cars')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Histogram - Fuel Type Distribution
plt.figure(figsize=(6, 4))
data['Fuel_Type'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Fuel Type Distribution')
plt.xlabel('Fuel Type')
plt.ylabel('Number of Cars')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Histogram - Mileage Overall
plt.figure(figsize=(6, 4))
plt.hist(data['Mileage'].dropna(), bins=15, color='lightgreen', edgecolor='black')
plt.title('Distribution of Car Mileage')
plt.xlabel('Mileage (kmpl or km/kg)')
plt.ylabel('Number of Cars')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Prepare for scatter plot
data['Mileage'] = data['Mileage'].astype(str).str.extract(r'([\d\.]+)').astype(float)
data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
data = data.dropna(subset=['Mileage', 'Price'])

# Scatter Plot: Mileage vs Price
plt.figure(figsize=(8, 5))
plt.scatter(data['Mileage'], data['Price'], alpha=0.6, color='green', edgecolors='black')
plt.title('Scatter Plot: Mileage vs Price')
plt.xlabel('Mileage (kmpl or km/kg)')
plt.ylabel('Price (in Lakhs)')
plt.grid(True)
plt.tight_layout()
plt.show()
