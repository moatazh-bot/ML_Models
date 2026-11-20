# ==========================================
# 1. IMPORT NECESSARY LIBRARIES
# ==========================================
import numpy as np                  # For numerical operations and generating data
import pandas as pd                 # For data manipulation (DataFrames)
import matplotlib.pyplot as plt     # For plotting graphs
from sklearn.model_selection import train_test_split # To split data into training and testing sets
from sklearn.linear_model import LinearRegression    # The actual Linear Regression algorithm
from sklearn.metrics import mean_squared_error, r2_score # To evaluate the model's performance

# ==========================================
# 2. CREATE / LOAD THE DATASET
# ==========================================
# For this example, we will create a synthetic dataset.
# Imagine we are predicting "House Price" based on "Square Footage".

# Generate random square footage between 1000 and 4000 sq ft
np.random.seed(42) # Set seed for reproducible results
square_footage = np.random.randint(1000, 4000, 100)

# Create house prices based on a formula: Price = (150 * sq_ft) + 50000 + some random noise
# 150 is the true weight (slope), 50000 is the true bias (intercept)
noise = np.random.randn(100) * 20000 # Add some random noise to make it realistic
house_prices = (150 * square_footage) + 50000 + noise

# Combine them into a Pandas DataFrame (like an Excel sheet)
data = pd.DataFrame({
    'SquareFootage': square_footage,
    'Price': house_prices
})

# ==========================================
# 3. PREPARE DATA FOR THE MODEL
# ==========================================
# In Machine Learning, 'X' represents the input features, and 'y' represents the target output.
X = data[['SquareFootage']]  # X must be a 2D array (DataFrame), so we use double brackets
y = data['Price']            # y is a 1D array (Series)

# Split the data: 80% for training the model, 20% for testing its accuracy
# random_state=42 ensures we get the same split every time we run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. BUILD AND TRAIN THE MODEL
# ==========================================
# Initialize the Linear Regression model
model = LinearRegression()

# Train (or "fit") the model using the training data
# The model will learn the relationship (slope and intercept) between X_train and y_train
model.fit(X_train, y_train)

# Extract the learned parameters
slope = model.coef_[0]       # The coefficient (weight) learned by the model
intercept = model.intercept_ # The intercept (bias) learned by the model

print(f"--- Model Parameters ---")
print(f"Learned Slope (Coefficient): {slope:.2f}")
print(f"Learned Intercept: {intercept:.2f}")
print(f"Equation: Price = {slope:.2f} * SquareFootage + {intercept:.2f}\n")

# ==========================================
# 5. MAKE PREDICTIONS
# ==========================================
# Use the trained model to predict prices for the unseen test data
y_pred = model.predict(X_test)

# ==========================================
# 6. EVALUATE THE MODEL
# ==========================================
# We compare the actual prices (y_test) with the predicted prices (y_pred)

# Mean Squared Error (MSE): Average of the squared differences between actual and predicted
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error (RMSE): Square root of MSE (in the same units as the target)
rmse = np.sqrt(mse)

# R-squared (R2): Represents the proportion of variance explained by the model (0 to 1, closer to 1 is better)
r2 = r2_score(y_test, y_pred)

print(f"--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.4f}\n")

# ==========================================
# 7. VISUALIZE THE RESULTS
# ==========================================
# Create a figure with 2 plots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Training Data and the Regression Line
axes[0].scatter(X_train, y_train, color='blue', label='Training Data', alpha=0.6)
# Plot the regression line using the learned slope and intercept
axes[0].plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Regression Line')
axes[0].set_title('Training Data & Regression Line')
axes[0].set_xlabel('Square Footage')
axes[0].set_ylabel('House Price')
axes[0].legend()

# Plot 2: Actual vs Predicted Prices (for the Test set)
axes[1].scatter(y_test, y_pred, color='green', alpha=0.6)
# Plot a perfect prediction line (y = x) for reference
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
axes[1].set_title('Actual vs Predicted Prices (Test Set)')
axes[1].set_xlabel('Actual Prices')
axes[1].set_ylabel('Predicted Prices')
axes[1].legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()