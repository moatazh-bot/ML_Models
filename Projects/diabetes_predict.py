import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import load_diabetes
from  sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


diabetes=load_diabetes(as_frame=True)
X=diabetes.data
y=diabetes.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)

mse=mean_squared_error(y_test,y_pred)
rmse=root_mean_squared_error(y_test,y_pred) #rmse = np.sqrt(mse) # Using np.sqrt for maximum compatibility
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("--- Model Evaluation ---")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}\n")

coefficients=pd.DataFrame({
    'Features':X.columns,
    'Coefficient':model.coef_
})


print("--- Feature Coefficients (Sorted by Impact) ---")
print(coefficients.sort_values(by='Coefficient', ascending=False))

# 7. Visual Proof: Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal', edgecolors='w', s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Disease Progression")
plt.ylabel("Predicted Disease Progression")
plt.title("Actual vs Predicted Disease Progression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()