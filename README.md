# 🥗 Recipe Site Traffic Prediction – Data Science Project

#### 🔥 Project Overview

#### Goal: This project aims to predict which recipes are most likely to drive high traffic on the homepage so marketing and content teams can feature the best dishes—boosting clicks, engagement, and subscriptions. It uses machine learning models trained on nutritional and categorical data from a single recipe website.

---

#### Project Summary

    - Business Goal: Prioritize homepage recipes with high traffic potential (precision ≥ 0.80).

    - Approach: Use historical recipe data to build and evaluate models that predict high-traffic recipes, using key nutrition and category data.

**Key Deliverables:**
- Data validation and cleaning
- Exploratory Data Analysis (EDA) to uncover trends
- Feature engineering and preprocessing
- Model development: Logistic Regression, Random Forest, XGBoost
- Model evaluation using precision and accuracy (goal: precision ≥ 0.80)
- Business recommendations

#### Business Impact & Value
Why It Matters: 
- Predicting high-traffic recipes means less guesswork for content planners and marketers. Featuring recipes that actually drive clicks makes homepage real estate work harder—boosting engagement and potential subscription growth.

Estimated ROI:
- With the current dataset, we expect this model could boost homepage click-through rates by 10–15%, translating into higher user engagement and potential subscription growth.

How It Helps:
- Marketing Teams: Prioritize recipes that have historically performed well.
- Product Managers: Align homepage curation with user preferences observed in this dataset—protein-rich recipes in categories like Vegetables, Potatoes, and Pork consistently get more attention.
- Content Strategists: Focus on recipes that engage users based on this dataset’s trends.

#### Limitations & Future Work

- Sample Size: ~895 recipes limit generalizability—this model captures historical patterns specific to this dataset and site.
- Category & Nutrition Bias: In this dataset, high-protein recipes in certain categories (like Vegetables, Potatoes, and Pork) performed best historically. This may not hold true for all audiences or websites. A/B testing and continuous validation are recommended.

---

## 📊 Notebooks & Files

- `notebook.ipynb`: Contains all the code, analysis, and visualizations
- `cleaned_recipe_site_traffic.csv`: Cleaned dataset
- `recipe_site_traffic_2212.csv`: Raw dataset

---
## Data Cleaning & Validation

    - Identified and handled missing values using a grouped median imputation to fill partial gaps.

    - Removed rows with all nutrition data missing (52 recipes dropped) (e.g. calories, protein), ensuring better data quality.

    - Standardized category labels and serving sizes by merging similar categories (e.g. 'Chicken' and 'Chicken Breast').

    - Converted target variable to a clean binary (high/low traffic) to simplify prediction.

---
## 🔍 Exploratory Data Analysis (EDA)
    - Analyzed nutrition distributions — e.g. protein, calories, carbs, sugar to spot trends and their impact on traffic.
    - Found calories are right-skewed—log transformation recommended.
    - Observed that higher-protein recipes tended to get more clicks in this dataset, though this might not apply universally.
    - Visualized how certain categories like Vegetables, Potatoes, and Pork were associated with higher traffic in this case study.
    
---
## Model Development & Evaluation
    - Evaluated models using Precision (key business priority) and Accuracy.
Models Built:

    Logistic Regression: Easy to interpret, high precision (0.90)—excellent at minimizing false positives.

    Random Forest: Good at capturing complex patterns and handling outliers.

    XGBoost: Powerful for capturing non-linear relationships in the data.

These models predict whether a recipe is likely to be high-traffic or not based on this dataset.

---

Key Terms Explained:

- Precision: How often the model correctly identifies recipes that actually bring in high traffic—our key business metric. Think of it as “hitting the right target” — it’s about avoiding false positives.

- Accuracy: Overall correctness, accounting for both high and low traffic classifications. Can be influenced by class imbalance.

#### Highlights
- Logistic Regression achieved the highest precision (0.90)—best for reducing false positives in this case.

- Random Forest and XGBoost provided balanced precision and accuracy—good secondary options.

- Top features driving predictions included protein, calories, and certain categories—insights specific to this dataset.

---
## Business Recommendations
- Primary Model: Use Logistic Regression as the main model—it consistently achieved high precision in this case study.

- Secondary Model: Random Forest can be useful if broader coverage is needed.

- Focus Areas (for this dataset): Recipes with higher protein and calories, especially in categories like Vegetables, Potatoes, and Pork, showed higher traffic—but this pattern might not generalize. Validate with A/B testing.

- Monitoring: Weekly model monitoring—aim for precision ≥ 0.80 to meet marketing goals.

- Next Steps: Integrate user engagement metrics (like clicks or time spent on page) for deeper insights.

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

    1. Logistic Regression = top precision in this dataset—reliable first pick for homepage recipe selection.

    2. Random Forest & XGBoost = strong alternatives with balanced trade-offs.

    3. Key drivers in this dataset included protein, calories, and categories like Vegetables, Potatoes, and Pork—recommend validating these features in your own data.

    4. Weekly monitoring—keep precision ≥ 0.80.

    5. Future enhancements: Consider integrating user engagement metrics to improve predictions.

#### Disclaimer:

Important: This model and its recommendations are based on a single dataset from one recipe website. Results may vary depending on audience, region, and business goals. We recommend applying A/B testing, user research, or further model tuning before using these insights as universal truths for all food manufacturing or recipe marketing scenarios.
