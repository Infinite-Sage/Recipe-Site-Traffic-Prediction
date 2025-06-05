# 🥗 Recipe Site Traffic Prediction – Data Science Project

#### This project predicts which recipes are most likely to drive high website traffic when featured on the homepage, helping the product team select recipes that boost customer engagement and subscriptions.

---

#### 🔥 Project Overview

This project uses nutritional and category data from recipes to predict high-traffic vs. low-traffic recipes. I cleaned and analyzed the data, built machine learning models, and provided actionable recommendations. My work also reflects key transferable skills—like data cleaning, variance analysis, and reporting.

**Key Deliverables:**
- Data validation and cleaning
- Exploratory Data Analysis (EDA) to uncover trends
- Feature engineering and preprocessing
- Model development: Logistic Regression, Random Forest, XGBoost
- Model evaluation using precision and accuracy (goal: precision ≥ 0.80)
- Business recommendations

---

## 📊 Notebooks & Files

- `notebook.ipynb`: Contains all the code, analysis, and visualizations
- `cleaned_recipe_site_traffic.csv`: Cleaned dataset
- `recipe_site_traffic_2212.csv`: Raw dataset

---
## Data Cleaning & Validation

    - Identified and handled missing values using a grouped median approach.

    - Removed rows with missing nutritional data, ensuring data quality.

    - Standardized categories and serving sizes.

    - Converted target variable to a clean binary (high/low traffic) to simplify prediction.

---
## 🔍 Exploratory Analysis
    - Analyzed distributions of calories, protein, and other nutritional features to spot trends.
    - Found that high-protein recipes often drive more traffic.
    - Visualized recipes by category and serving size to understand popularity.
---
## Model Development
I built three machine learning models:

    Logistic Regression: Easy to interpret and excellent for precision.

    Random Forest: Great for handling unusual or “extreme” recipes (outliers).

    XGBoost: Powerful model for capturing complex patterns.

These models predict whether a recipe is likely to be high-traffic or not.

---
## Results

Key Terms Explained:

- Precision: How often the model correctly identifies recipes that actually bring in high traffic. Think of it as “hitting the right target” — it’s about avoiding false positives.

- Accuracy: Overall, how many recipes the model correctly classifies as either high-traffic or low-traffic. It’s a measure of “total correctness” but can be influenced by class balance.

- Logistic Regression achieved the highest precision (0.90), ensuring fewer false positives—important for confidently choosing homepage recipes.

- Random Forest and XGBoost balanced precision and accuracy, useful for broader scenarios.

- Key features driving traffic predictions included protein content, calories, and recipe categories — insights that can inform content and marketing strategies.

---
## Business Recommendations
- Use Logistic Regression as the primary model to select homepage recipes with high confidence.

- Prioritize high-protein, higher-calorie recipes in categories like Vegetables, Potatoes, and Pork, which consistently attracted higher traffic.

- Monitor model performance regularly, adjusting thresholds as needed to align with marketing goals.

## 🔧 Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost



## Interactive Demo: Predictive Function

This interactive function allows you to enter a recipe ID and see whether it’s predicted to drive high traffic:

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

    1. Logistic Regression achieved 0.90 precision, meeting the 0.80 business goal and proving it’s a reliable first pick for high-traffic recipe selection.

    2. Random Forest and XGBoost also performed well with solid trade-offs in accuracy and precision.

    3. Prioritize high-protein and higher-calorie recipes in categories like Vegetables, Potatoes, and Pork.

Business Recommendations

    Deploy Logistic Regression as the primary model — it’s a precision monster that minimizes false positives, saving homepage space for recipes that actually get clicks.

    Use Random Forest as a secondary model when broader coverage is needed.

    Prioritize high-protein, higher-calorie recipes, especially in Vegetables, Potatoes, and Pork — these consistently attract more traffic.
    
    Monitor model performance weekly—adjust thresholds as marketing goals evolve.
    
    Next Level: Incorporate user engagement metrics (like clicks/time on page) to enhance predictions.


