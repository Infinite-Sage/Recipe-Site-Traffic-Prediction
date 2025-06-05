# 🥗 Recipe Site Traffic Prediction – Data Science Project

#### This project predicts recipe popularity on a website. The goal: help the product team decide which recipes to feature on the homepage by identifying those most likely to drive high traffic.

---

#### 🔥 Project Overview

This project uses nutritional features and categorical information from a recipe dataset to build a binary classification model predicting "High Traffic" vs "Low Traffic" recipes.

Key Deliverables:

    Data validation and cleaning

    Exploratory Data Analysis (EDA) with insightful plots

    Feature engineering and preprocessing

    Model development: Logistic Regression, Random Forest, XGBoost

    Model evaluation using precision and accuracy (goal: precision ≥ 0.80)

    Business recommendations

---

## 📊 Notebooks & Files

- notebook.ipynb: Main code and analysis

- cleaned_recipe_site_traffic.csv: Cleaned dataset

- recipe_site_traffic_2212.csv: Raw dataset

---

## Demo: Predictive Function

A simple interactive function to predict traffic based on a Recipe ID:

```python
try:
    # Check if X_test exists
    if 'X_test' not in globals():
        raise NameError("X_test is not defined. Please run the data split and define X_test.")
    
    # Prompt the user for a Recipe ID
    recipe_id_input = input(f"Enter Recipe ID (0 to {len(X_test)-1}): ").strip()
    recipe_id = int(recipe_id_input)
    
    # Validate Recipe ID range
    if recipe_id < 0 or recipe_id >= len(X_test):
        raise IndexError("Recipe ID out of range")
    
    # Select the row from the test set
    X_query = X_test.iloc[[recipe_id]]
    
    # Apply the preprocessing pipeline
    X_query_transformed = fitted_models['Logistic Regression'].named_steps['preprocess'].transform(X_query)
    
    # Predict using Logistic Regression
    y_pred_query = fitted_models['Logistic Regression'].named_steps['clf'].predict(X_query_transformed)
    
    if y_pred_query[0] == 1:
        print("🚀 This recipe will likely produce HIGH TRAFFIC!")
    else:
        print("🔍 This recipe will likely produce NORMAL TRAFFIC!")

except ValueError:
    print("⚠️ Invalid input. Please enter a valid integer for the Recipe ID.")
except IndexError as e:
    print(f"⚠️ {e}. Please enter a Recipe ID between 0 and {len(X_test)-1}.")
except NameError as e:
    print(f"⚠️ {e} Please make sure you've run the data split and model training steps.")
```
Final Summary

    Logistic Regression achieved 0.90 precision, meeting the 0.80 business goal

    Random Forest and XGBoost also performed well

    Business recommendations provided for deployment and monitoring

Business Recommendations

    Deploy Logistic Regression for homepage recipe selection to avoid wasting slots on low-traffic recipes

    Use Random Forest as a secondary model to find even more high-traffic recipes

    Prioritize high-protein and higher-calorie recipes in categories like Vegetables, Potatoes, and Pork

🔧 Requirements

    Python 3.x

    pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost

📝 License

MIT License
