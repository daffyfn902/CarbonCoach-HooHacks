# Imports 

import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

# load data here

df = pd.read_csv("carbon_emissions.csv")

# define x and y

# X is features
X = df.drop(columns = ["CarbonEmission"])

# Y is target
y = df["CarbonEmission"]


# preprocessing

categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

# scales numerical values, transforms categorical values

preprocessor = ColumnTransformer([
    ("number", StandardScaler(), numeric_cols),
    ("category", OneHotEncoder(handle_unknown="ignore"), categorical_cols)])



# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train random forest (all features)
rf_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42))])

rf_model.fit(X_train, y_train)

# Model Evaluation 
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print("Pre optimization stats:")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")




# Extract feature importance 
cat_features = rf_model.named_steps["preprocess"] \
    .named_transformers_["category"] \
    .get_feature_names_out(categorical_cols)

all_features = list(numeric_cols) + list(cat_features)

# get importance values
importances = rf_model.named_steps["model"].feature_importances_

# put into dataframe
feature_df = pd.DataFrame({
    "feature": all_features,
    "importance": importances})

# sort and print top 10
feature_df = feature_df.sort_values("importance", ascending=False)

print(feature_df.head(18))

# Choose final features( based on lasso / feature importance)
# Drop irrelevant features
selected_features = [
    "Vehicle Monthly Distance Km",
    "Frequency of Traveling by Air",
    "Vehicle Type",
    "How Many New Clothes Monthly",
    "Waste Bag Weekly Count",
    "Heating Energy Source"]

X_reduced = df[selected_features]


# new reduced column types
categorical_cols_r = X_reduced.select_dtypes(include=["object"]).columns.tolist()
numeric_cols_r     = X_reduced.select_dtypes(include=["int64", "float64"]).columns.tolist()


# new reduced column preprocessor
preprocessor_reduced = ColumnTransformer([
    ("number",   StandardScaler(), numeric_cols_r),
    ("category", OneHotEncoder(handle_unknown="ignore"), categorical_cols_r),])


# new reduced pipeline
rf_model_reduced = Pipeline([
    ("preprocess", preprocessor_reduced),
    ("model", RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42 ))])

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42)

rf_model_reduced.fit(X_train, y_train)
y_pred = rf_model_reduced.predict(X_test)

print("Post optimization stats:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:",  r2_score(y_test, y_pred))

# shows that r squared value is consistent
scores = cross_val_score(rf_model_reduced, X_reduced, y, cv=5, scoring='r2')
print("5-Fold CV R²:", scores.mean())


# Save the reduced model
joblib.dump(rf_model_reduced, "carbon_model.pkl")
# Start flask










