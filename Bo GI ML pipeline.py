# ML pipeline for predicting the impact of Catalan GIs in Catalonia's economy, predicting future GIs, and mapping GIs for tourists (aka Visitors) & local producers (aka Hosts)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Catalunya GI dataset
data = pd.read_csv('Cleaned_Catalunya_GI_Data.csv')

# Load economic datasets from Idescat
gdp_per_capita = pd.read_csv("2025-3-15 GDP and GDP per inhabitant. Counties and Aran.csv")
gva_basic_prices = pd.read_csv("2025-3-18 GVA at basic prices. By branches of activity. At current prices.csv")
gva_by_sector = pd.read_csv("2025-3-18 GVA. By sectors. Counties.csv")

# Standardize region names
gi_data["Region"] = gi_data["Region"].str.strip().str.lower()
gdp_per_capita["Region"] = gdp_per_capita["Region"].str.strip().str.lower()
gva_basic_prices["Region"] = gva_basic_prices["Region"].str.strip().str.lower()
gva_by_sector["Region"] = gva_by_sector["Region"].str.strip().str.lower()

# Convert Year columns to numeric
gi_data["Year of Registration"] = pd.to_numeric(gi_data["Year of Registration"], errors="coerce")
gdp_per_capita["Year"] = pd.to_numeric(gdp_per_capita["Year"], errors="coerce")
gva_basic_prices["Year"] = pd.to_numeric(gva_basic_prices["Year"], errors="coerce")
gva_by_sector["Year"] = pd.to_numeric(gva_by_sector["Year"], errors="coerce")

# Merge datasets
merged_data = pd.merge(gi_data, gdp_per_capita, left_on=["Region", "Year of Registration"], right_on=["Region", "Year"], how="left")
merged_data = pd.merge(merged_data, gva_basic_prices, left_on=["Region", "Year of Registration"], right_on=["Region", "Year"], how="left")
merged_data = pd.merge(merged_data, gva_by_sector, left_on=["Region", "Year of Registration"], right_on=["Region", "Year"], how="left")

# Drop duplicate Year columns
merged_data.drop(columns=["Year_x", "Year_y"], inplace=True, errors='ignore')

# Save the merged dataset
merged_data.to_csv("Merged_Catalonia_GI_Economic_Data.csv", index=False)


# Split the data into features and target variable
X = data.drop('impact_on_economy', axis=1)
y = data['impact_on_economy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)