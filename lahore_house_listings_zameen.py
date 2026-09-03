
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("lahore_house_listings_zameen.csv")

# 2. Drop useless raw columns immediately
df = df.drop(columns=['Title', 'Date Posted', 'Link'], errors='ignore')

# -------------------------
# Convert Area to Marla
# -------------------------
def convert_area(area):
    try:
        if pd.isna(area) or not isinstance(area, str):
            return np.nan
        value, unit = area.strip().split()
        value = float(value)

        if unit.lower() == 'kanal':
            return value * 20
        elif unit.lower() == 'marla':
            return value
        return np.nan
    except (ValueError, IndexError):
        return np.nan

df['Area'] = df['Area'].apply(convert_area)

# -------------------------
# Convert Price to PKR
# -------------------------
def convert_pkr(price):
    try:
        if pd.isna(price) or not isinstance(price, str):
            return np.nan
        parts = price.strip().split()

        if len(parts) == 1:
            return np.nan

        value = float(parts[0])
        unit = parts[1].lower()

        if unit == 'crore':
            return value * 10_000_000
        elif unit == 'lakh':
            return value * 100_000
        elif unit == 'arab':
            return value * 1_000_000_000

        return np.nan
    except (ValueError, AttributeError, IndexError):
        return np.nan

df['Price'] = df['Price'].apply(convert_pkr)

# -------------------------
# Convert Bedrooms & Bathrooms
# -------------------------
df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')
df['Bathrooms'] = pd.to_numeric(df['Bathrooms'], errors='coerce')

# Drop rows missing crucial numbers before filtering outliers
df = df.dropna(subset=['Price', 'Area'])

# -------------------------------------------------------------
# 🌟 FIXED: Outlier Filters Moved Here (After Data Conversion)
# -------------------------------------------------------------
# Keep prices between 25 Lakh and 15 Crore
df = df[(df['Price'] >= 2_500_000) & (df['Price'] <= 150_000_000)]

# Keep normal house sizes between 2 Marlas and 40 Marlas
df = df[(df['Area'] >= 2) & (df['Area'] <= 40)]

# -------------------------
# Fill Remaining Missing Features
# -------------------------
df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].median())
df['Bathrooms'] = df['Bathrooms'].fillna(df['Bathrooms'].median())

if 'Built Year' in df.columns:
    df['Built Year'] = df['Built Year'].fillna(df['Built Year'].median())

# -------------------------
# One-Hot Encoding
# -------------------------
df = pd.get_dummies(df, columns=['Location', 'Type', 'Purpose'], dtype=int)

# -------------------------
# Machine Learning: Random Forest
# -------------------------
# Apply Log Transform to handle wide price distribution safely
y = np.log1p(df["Price"])
X = df.drop(columns=["Price"])

# Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
# leaves = [1, 2, 4, 8, 16]


model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

model.fit(X_train, y_train)

import joblib

model_data = {
    "model":model,
    "columns": X_train.columns.tolist()
}

joblib.dump(model_data, "lahore_house_price_model.pkl")

print("Model and columns saved successfully!")
# prediction_log = model.predict(X_test)

# prediction = np.expm1(prediction_log)
# actual_price = np.expm1(y_test)

    # mae = mean_absolute_error(actual_price, prediction)
    # rmse = mean_squared_error(actual_price, prediction) ** 0.5
    # r2 = r2_score(actual_price, prediction)

    # print(f"\n--- min_samples_leaf = {leaf} ---")
    # print(f"MAE: {mae:,.0f} PKR")
    # print(f"RMSE: {rmse:,.0f} PKR")
    # print(f"R2 Score: {r2:.3f}")

