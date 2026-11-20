import numpy as np 
import pandas as pd 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#loading and Preparing Data
data = load_diabetes(as_frame=True)
X,y=data.data,data.target
print(X.head())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.transform(X_test)

#Training and comparing the models 

ols=LinearRegression().fit(X_train_s,y_train)
ridge=Ridge(alpha=10).fit(X_train_s,y_train)

for name,model in [("OLS",ols),("Ridge(alpha=10)",ridge)]:
    pred=model.predict(X_test_s)
    print(f"{name:18s} RMSE={np.sqrt(mean_squared_error(y_test,pred)):.2f}"
          f"R2={r2_score(y_test,pred):.3f}")

coefs=pd.DataFrame({
    "OLS":ols.coef_,
    "Ridge":ridge.coef_
},index=X.columns)
print(coefs.round(2))

#Chosing alpha with cross_validation 

alphas = np.logspace(-3,3,50)
ridge_cv=RidgeCV(alphas=alphas,cv=5).fit(X_train_s,y_train)

print("Best alpha :" ,ridge_cv.alpha_)
pred=ridge_cv.predict(X_test_s)
print("Test RMSE:",np.sqrt(mean_squared_error(y_test,pred)))
print("Test R2:  ",r2_score(y_test,pred))

#Data visualization 

import matplotlib.pyplot as plt

coef_path = [Ridge(alpha=a).fit(X_train_s, y_train).coef_ for a in alphas]

plt.figure(figsize=(8, 5))
plt.plot(alphas, coef_path)
plt.xscale("log")
plt.xlabel("alpha (regularization strength)")
plt.ylabel("coefficient value")
plt.title("Ridge coefficient paths")
plt.legend(X.columns, loc="upper right", fontsize=8)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

y_pred = ridge_cv.predict(X_test_s)
residuals = y_test - y_pred

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 1. Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred, alpha=0.7, edgecolor="k")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", label="Perfect prediction")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title(f"Actual vs Predicted (R² = {r2_score(y_test, y_pred):.3f})")
ax.legend()

# 2. Residuals vs Predicted
ax = axes[0, 1]
ax.scatter(y_pred, residuals, alpha=0.7, edgecolor="k")
ax.axhline(0, color="r", linestyle="--")
ax.set_xlabel("Predicted")
ax.set_ylabel("Residual (Actual - Predicted)")
ax.set_title("Residuals vs Predicted")

# 3. Distribution of residuals
ax = axes[1, 0]
ax.hist(residuals, bins=20, edgecolor="k", alpha=0.8)
ax.axvline(0, color="r", linestyle="--")
ax.set_xlabel("Residual")
ax.set_ylabel("Count")
ax.set_title("Residual distribution")

# 4. Test samples: actual vs predicted, sorted by actual value
ax = axes[1, 1]
order = np.argsort(y_test.values)
ax.plot(y_test.values[order], "o-", label="Actual", markersize=4)
ax.plot(y_pred[order], "s-", label="Predicted", markersize=4, alpha=0.7)
ax.set_xlabel("Test sample (sorted by actual value)")
ax.set_ylabel("Disease progression")
ax.set_title("Actual vs Predicted per sample")
ax.legend()

plt.tight_layout()
plt.show()