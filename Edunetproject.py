import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV # Changed from GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint # For sampling integer parameters
from scipy.stats import uniform # For sampling float parameters

crop_data_df = pd.read_csv('clean_crop_data.csv')

print("Dataset loaded from 'clean_crop_data.csv'.")
print("Head of the dataset:")
print(crop_data_df.head())
print("\nDataset Info:")
crop_data_df.info()
print("\nColumns in your CSV file:")
print(crop_data_df.columns.tolist())

crop_data_df = pd.get_dummies(crop_data_df, columns=['Item'], prefix='Item')
print("\nDataFrame after One-Hot Encoding:")
print(crop_data_df.head())
print("\nUpdated Columns after encoding:")
print(crop_data_df.columns.tolist())

numerical_features = ['average_rain_fall_mm_per_year', 'avg_temp', 'pesticides_tonnes']
item_features = [col for col in crop_data_df.columns if col.startswith('Item_')]

input_features = numerical_features + item_features
target_variable = 'hg/ha_yield'

for col in input_features + [target_variable]:
    if col not in crop_data_df.columns:
        print(f"Error: Column '{col}' not found in the dataset after encoding.")
        print("Available columns are: ", crop_data_df.columns.tolist())
        exit()

for col in input_features + [target_variable]:
    crop_data_df[col] = crop_data_df[col].fillna(crop_data_df[col].mean())

X = crop_data_df[input_features]
y = crop_data_df[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data size: {X_train.shape[0]} samples")
print(f"Testing data size: {X_test.shape[0]} samples")

# --- Hyperparameter Tuning with RandomizedSearchCV ---
print("\nStarting Hyperparameter Tuning for Random Forest Regressor using RandomizedSearchCV (faster!)...")
# Define the parameter distributions to sample from
param_distributions = {
    'n_estimators': randint(50, 300), # Random integer between 50 and 300
    'max_features': ['sqrt', 'log2'],
    'max_depth': randint(10, 30), # Random integer between 10 and 30
    'min_samples_leaf': randint(1, 5),
    'min_samples_split': randint(2, 10)
}

# Initialize the RandomizedSearchCV object
# n_iter: Number of parameter settings that are sampled.
#         A smaller number means faster execution but potentially less optimal results.
#         You can increase this if you have more time.
random_search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=42),
                                   param_distributions=param_distributions,
                                   n_iter=50, # Number of random combinations to try (you can adjust this)
                                   cv=5,
                                   scoring='r2',
                                   n_jobs=-1,
                                   random_state=42, # For reproducibility of random sampling
                                   verbose=1)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, y_train)

# Get the best model found by RandomizedSearchCV
model = random_search.best_estimator_

print("\nHyperparameter tuning complete!")
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best R-squared score from cross-validation: {random_search.best_score_:.2f}")

print("\nTraining the Random Forest Regressor model with best parameters...")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Evaluation Metrics (on Test Set with Tuned Model) ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

print("\n--- Model Performance Summary ---")
if r2 > 0.90:
    print("The R-squared score is exceptionally high, indicating excellent predictive performance. The model captures most of the variance in crop yields.")
elif r2 > 0.75:
    print("The R-squared score is high, indicating strong predictive performance.")
elif r2 > 0.5:
    print("The R-squared score is moderate, suggesting reasonable performance.")
else:
    print("The R-squared score is low, suggesting limited predictive power. Consider adding more relevant features (like soil quality), or expanding the hyperparameter search space.")

print(f"\nNote on MSE/RMSE: The absolute value of MSE ({mse:.2f}) and RMSE ({rmse:.2f}) depends on the scale of your target variable ('{target_variable}'). A high R-squared (like {r2:.2f}) indicates the model is very good at predicting the relative changes, even if the absolute error (RMSE) seems large in isolation.")

print("\n--- Feature Importances (Random Forest) ---")
feature_importances = pd.Series(model.feature_importances_, index=input_features).sort_values(ascending=False)
print(feature_importances)

print("\n--- Graphs and Visualization ---")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel(f"Actual {target_variable}")
plt.ylabel(f"Predicted {target_variable}")
plt.title("Actual vs. Predicted Crop Yields (Tuned Random Forest)")
plt.grid(True)

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel("Prediction Errors (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors (Residuals) - Tuned Random Forest")
plt.grid(True)

plt.figure(figsize=(10, 8))
sns.heatmap(crop_data_df[numerical_features + [target_variable]].corr(), annot=True, cmap='viridis', fmt=".2f")
plt.title("Correlation Matrix of Numerical Features and Crop Yield")

plt.show()

print("\n--- Predict Crop Yield ---")
print("Please provide the following details for prediction:")

available_crop_types = crop_data_df.columns[crop_data_df.columns.str.startswith('Item_')].str.replace('Item_', '').tolist()
print(f"Available Crop Types: {', '.join(available_crop_types)}")

new_rainfall = float(input("   Enter Average Rainfall (mm per year): "))
new_temperature = float(input("   Enter Average Temperature (°C): "))
new_pesticides = float(input("   Enter Pesticides Usage (tonnes): "))
new_crop_type = input(f"   Enter Crop Type ({'/'.join(available_crop_types)}): ")

new_data_for_prediction = pd.DataFrame([[new_rainfall, new_temperature, new_pesticides]], columns=numerical_features)

for crop_type_col in item_features:
    new_data_for_prediction[crop_type_col] = 0

if f'Item_{new_crop_type}' in new_data_for_prediction.columns:
    new_data_for_prediction[f'Item_{new_crop_type}'] = 1
else:
    print(f"Warning: '{new_crop_type}' is not a recognized crop type from the training data. Prediction might be less accurate.")

new_data_for_prediction = new_data_for_prediction[input_features]

predicted_yield = model.predict(new_data_for_prediction)[0]
print(f"\nPredicted Crop Yield: {predicted_yield:.2f} {target_variable}")

