import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset directly from seaborn
df = sns.load_dataset('mpg')

# print("Original Data Shape:", df.shape)
# print("\nFirst 5 rows:")
# print(df.head())
# print("\nData Types:")
# print(df.dtypes)

df_clean=df.dropna().copy()
df_clean=df_clean.drop('name',axis=1)
df_clean['origin']=df_clean['origin'].map({1:'usa',2:'japan',3:'europe'})
df_clean=pd.get_dummies(df_clean,columns=['origin'],drop_first=True)

print("\nCleaned Data Shape:", df_clean.shape)
print(df_clean.head())

plt.figure(figsize=(12, 4))

# Plot 1: Weight vs MPG
plt.subplot(1, 2, 1)
sns.scatterplot(x='weight', y='mpg', data=df_clean, alpha=0.6)
plt.title("Weight vs. MPG")

# Plot 2: Horsepower vs MPG
plt.subplot(1, 2, 2)
sns.scatterplot(x='horsepower', y='mpg', data=df_clean, alpha=0.6)
plt.title("Horsepower vs. MPG")

plt.tight_layout()
plt.show()

X=df_clean.drop('mpg',axis=1)
y=df_clean['mpg']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model=LinearRegression()
model.fit(X_train_scaled,y_train)

y_pred=model.predict(X_test_scaled)

mse=mean_squared_error(y_test,y_pred)
rmse=root_mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("--- Model Evaluation ---")
print(f"MAE:  {mae:.2f} MPG")
print(f"RMSE: {rmse:.2f} MPG")
print(f"R²:   {r2:.4f}")

coefficients=pd.DataFrame({
    'Features':X.columns,
    'Coefficient':model.coef_
})

print("\nFeature Coefficients (Impact on MPG):")
print(coefficients.sort_values(by='Coefficient', ascending=False))


