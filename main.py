import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# Load dataset
df = pd.read_csv("train.csv")

# Check dataset shape
print(df.shape)

# Check basic info
print(df.info())

# Check missing values
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])

# Drop columns with too many missing values
df = df.drop(["PoolQC", "MiscFeature", "Alley", "Fence"], axis=1)

# Check new shape
print(df.shape)

# -----------------------------
# Fill numeric missing values
# -----------------------------

numeric_missing = df.select_dtypes(include=['int64', 'float64']).isnull().sum()

print("\nNumeric columns with missing values:")
print(numeric_missing[numeric_missing > 0])

# Fill numeric columns with mean
for column in df.select_dtypes(include=['int64', 'float64']).columns:
    df[column] = df[column].fillna(df[column].mean())

# -----------------------------
# Fill categorical missing values
# -----------------------------

categorical_missing = df.select_dtypes(include=['object']).isnull().sum()

print("\nCategorical columns with missing values:")
print(categorical_missing[categorical_missing > 0])

# Fill categorical columns with mode
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Final check
print("\nRemaining missing values:")
print(df.isnull().sum().sort_values(ascending=False).head())

y=df["SalePrice"]
x=df.drop(["SalePrice"],axis=1)
X=pd.get_dummies(x)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)

model = LinearRegression()

model.fit(X_train,y_train)
predictions= model.predict(X_test)

error= mean_absolute_error(y_test,predictions)
print(error)
Model= DecisionTreeRegressor(max_depth=10)
Model.fit(X_train,y_train)
Predictions= Model.predict(X_test)

Error= mean_absolute_error(y_test,Predictions)
print(Error)

Mod= RandomForestRegressor(n_estimators=200,random_state=42)
Mod.fit(X_train,y_train)
pred=Mod.predict(X_test)

Err=mean_absolute_error(y_test,pred)
print(Err)