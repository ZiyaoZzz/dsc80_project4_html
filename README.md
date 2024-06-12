# Recipe Nutrition VS. Preparation Time & Steps

**Name(s)**: Kate Zhou, Ziyao Zhou

**Website Link**: https://ziyaozzz.github.io/dsc80_project4_html/

## Overview

This data science project for DSC80 at UCSD investigates the correlation between a recipe's rating and the percentage of total calories in the recipe that come from sugar.

## Introduction

**Questions:**

1. **What is the relationship between food preparation steps and nutrition?**
    - According to the U.S. Bureau of Labor Statistics, among people 15 and older in the United States, 57.2 percent spent an average of 53 minutes preparing food and drink on an average day in 2022. As college students, we often prefer to cook simple dishes and may overlook the quality and nutrition of these dishes. We wonder if food cooked in less time and fewer steps has poorer nutrition compared to food cooked in longer time and more steps.

### Dataset Overview

Our project investigates whether the number of food preparation steps affects the nutritional quality of food. We use two main datasets:

1. **Recipe Dataset**: Contains 83,782 rows (unique recipes) and 10 columns with the following information:

| Column | Description |
|:---:|:---:|
| `name` | Recipe name |
| `id` | Recipe ID |
| `minutes` | Minutes to prepare recipe |
| `contributor_id` | User ID who submitted this recipe |
| `submitted` | Date recipe was submitted |
| `tags` | Food.com tags for recipe |
| `nutrition` | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value” |
| `n_steps` | Number of steps in recipe |
| `steps` | Text for recipe steps, in order |
| `description` | User-provided description |
| `ingredients` | Text for recipe ingredients |
| `n_ingredients` | Number of ingredients in recipe |

2. **Interactions Dataset**: Contains 731,927 rows, each representing a user review of a specific recipe. The columns include:

| Column | Description |
|:---:|:---:|
| `user_id` | User ID |
| `recipe_id` | Recipe ID |
| `date` | Date of interaction |
| `rating` | Rating given |
| `review` | Review text |

Given the datasets, we are investigating whether the number of food preparation steps affects the nutritional quality of the food. To facilitate the investigation of our question, we separated the values in the `nutrition` columns into the corresponding columns, `calories (#)`, `total fat (PDV)`, `sugar (PDV)`, etc. PDV, or percent daily value, shows how much a nutrient in a serving of food contributes to a total daily diet. Moreover, we calculated the number of preparation steps for each recipe and stored this information in a new column, `prep_steps`. Recipes were then categorized based on the average number of steps, where those with `prep_steps` higher than the average were considered more complex.

The most relevant columns to answer our question are `calories (#)`, `total fat (PDV)`, `sugar (PDV)`, `prep_steps`, and `nutrition_score`, which is a composite score reflecting the overall nutritional quality of a recipe.

By seeking an answer to our question, we hope to gain insight into whether simpler recipes, in terms of preparation steps and time, tend to be less nutritious compared to more complex recipes. This could help individuals, especially college students, make better-informed choices about their cooking habits. Additionally, this information could lead to future work on promoting awareness about the nutritional impact of cooking methods and preparation steps.

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

**Step 1: Merge Datasets**

To combine unique recipes with their reviews and ratings, and to analyze how frequently the recipes are used, we performed a left merge of the recipes and interactions datasets on `id` and `recipe_id`.

**Step 2: Drop Irrelevant Columns**

We removed columns that are not relevant to our analysis to simplify the dataset.

**Step 3: Check Data Types**

We examined the data types of all columns to identify any necessary type conversions.

**Step 4: Handle Missing Ratings**

Ratings of 0 were replaced with `np.nan` to indicate missing data.

**Step 5: Split Nutrition Column**

The nutrition column, which contains a list of nutritional values, was split into individual columns for better analysis.

**Step 6: Identify High Saturated Fat Foods**

We created a new column to indicate whether a recipe is high in saturated fat, based on the guideline that an average person should consume no more than 30g of saturated fat per day.

| name                                 |   minutes | tags                                                                                                                                                                                                                        |   n_steps | steps                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | ingredients                                                                                                                                                                    |   n_ingredients |   rating | review                                                                                                                                                                                                                                                                                                                                           |   calories |   total_fat |   sugar |   sodium |   protein |   saturated_fat |   carbohydrates | high_saturated   | rating_missingness   |
|:-------------------------------------|----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------:|---------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------:|------------:|--------:|---------:|----------:|----------------:|----------------:|:-----------------|:---------------------|
| 1 brownies in the world    best ever |        40 | ['60-minutes-or-less', 'time-to-make', ...] |        10 | ['heat the oven to 350f and arrange the rack in the middle', ...'] | ['bittersweet chocolate', 'unsalted butter', ...] |               9 |        4 | These were pretty good, but took forever to bake...  |      138.4 |          10 |      50 |        3 |         3 |              19 |               6 | False            | False                |
| 1 in canada chocolate chip cookies   |        45 | ['60-minutes-or-less', 'time-to-make', ...]                                                               |        12 | ['pre-heat oven the 350 degrees f', ...] | ['white sugar', 'brown sugar', ...]                    |              11 |        5 | Originally I was gonna cut the recipe in half ... |      595.1 |          46 |     211 |       22 |        13 |              51 |              26 | True             | False                |
| 412 broccoli casserole               |        40 | ['60-minutes-or-less', 'time-to-make', ...]                                                                        |         6 | ['preheat oven to 350 degrees', ...] | ['frozen broccoli cuts', 'cream of chicken soup', ...]          |               9 |        5 | This was one of the best broccoli casseroles that I have ever made... |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 | True             | False                |
|                                      |           ||           |||                 |          | The photos you took (shapeweaver) inspired me to make this recipe and it... |            |             |         |          |           |                 |                 |                  |                      |
|                                      |           ||           |||                 |          | Thanks so much for sharing your recipe shapeweaver. It was wonderful!  Going into... |            |             |         |          |           |                 |                 |                  |                      |
| 412 broccoli casserole               |        40 | ['60-minutes-or-less', 'time-to-make', ...]                                                                        |         6 | ['preheat oven to 350 degrees', ...] | ['frozen broccoli cuts', 'cream of chicken soup', ...]          |               9 |        5 | I made this for my son's first birthday party this weekend. Our guests INHALED it... |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 | True             | False                |
| 412 broccoli casserole               |        40 | ['60-minutes-or-less', 'time-to-make', ...]                                                                        |         6 | ['preheat oven to 350 degrees', ...] | ['frozen broccoli cuts', 'cream of chicken soup', ...]          |               9 |        5 | Loved this.  Be sure to completely thaw the broccoli.  I didn... |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 | True             | False                |

### Exploratory Data Analysis

**Univariate Analysis**

We examined the distribution of the number of steps in recipes. The distribution is right-skewed, indicating that most recipes have a low number of steps, with fewer recipes requiring a high number of steps.

<iframe
  src="pictures/univariate_analysis1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


<iframe
  src="pictures/univariate_analysis1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The shapes of the distributions for recipes with and without high saturated fat content are similar, making it challenging to draw conclusions about the impact of saturated fat on preparation complexity from this analysis alone.

**Bivariate Analysis**

We analyzed the relationship between the number of steps in a recipe and its saturated fat content using various statistical measures (mean, min, max, median).

<iframe
  src="pictures/bivariate_analysis2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


**Interesting Aggregates**

We further investigated the relationship between the number of steps in a recipe and its nutritional content using a pivot table. The initial analysis suggests that more complex recipes (with more steps) may offer more balanced and richer nutritional profiles, including both less desirable elements like saturated fat and beneficial ones like protein.


|   n_steps |   calories |   carbohydrates |   protein |   saturated_fat |   sodium |   sugar |   total_fat |
|----------:|-----------:|----------------:|----------:|----------------:|---------:|--------:|------------:|
|         1 |    263.95  |         9.47153 |   11.7536 |         19.6263 |  41.7618 | 64.0372 |     19.7101 |
|         2 |    299.458 |        11.0513  |   15.4452 |         20.5875 |  35.5582 | 67.5614 |     20.516  |
|         3 |    286.54  |        10.0532  |   17.1299 |         23.5745 |  24.558  | 60.2696 |     20.4407 |
|         4 |    324.994 |        10.4715  |   22.9811 |         28.7361 |  24.5348 | 60.8759 |     24.568  |
|         5 |    341.972 |        10.5541  |   27.4319 |         31.3291 |  28.5839 | 54.3202 |     26.1504 |

# Assessment of Missingness
Based on our research dataframe, the only two columns that have missing value are rating and review, which have some missing values (15036 missing in he rating column and 58 missing in the review column).

## NMAR Analysis
We believe that the absence of ratings in the 'review' column is likely Not Missing at Random (NMAR). This inference arises from the notion that individuals tend to write review to recipes based on the strength of their emotional response. If someone feels indifferent towards a recipe, they may be less inclined to provide a rating, resulting in missing values. Conversely, individuals who have strong positive or negative sentiments towards a recipe are more motivated to assign a rating, reflecting their satisfaction or dissatisfaction. This behavior suggests that the missingness in the 'rating' column is tied to the level of emotional engagement individuals have with the recipes. Understanding this pattern of missingness is crucial for accurately interpreting user feedback and conducting sentiment analysis within the dataset.

## Missingness Dependency
We proceeded to investigate the missingness of 'rating' in the merged DataFrame by examining its dependency on two key factors: 'saturated fat (PDV)', representing the proportion of sugar out of the total calories, and 'n_steps', denoting the number of steps in the recipe. 
The null hypothesis: The missingness of ratings is independent of the saturated fat (PDV) in the recipe.
The alternative hypothesis: The missingness of ratings is dependent by the saturated fat (PDV) in the recipe. 
To test this, we calculated the absolute difference in mean proportion of saturated fat between the group with missing ratings and the group without missing ratings. Our significance level was set at 0.05 to determine the statistical significance of the findings.

<iframe
  src="pictures/scatter_plot.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

<iframe
  src="pictures/permutation_test.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

By the p-value of 0.0 < 0.05. We reject the null hypothesis, we do not have enough evidence to support that the rating and saturated_fat are independent to each other.

## Step 4: Hypothesis Testing

**Questions:**
1. the relationship between rare receipe items and their preparation time and steps. 

Null Hypothesis: rare occured recipe item take same steps to prepare as the often occured recipe items.

Alternative Hypothesis: rare occured recipe items take more steps to prepare.

Test Statistic: difference in group means or medians

2. the relationship between rare receipe items and their preparation time and steps. -Try this first

Null Hypothesis: all recipes are equally healthy(same in Saturated Fat).

Alternative Hypothesis: recipes with '3-steps-or-less' tag are less healthy (higher in Saturated Fat) than other recipes.

Test Statistic: the difference in means or medians of recipes with '3-steps-or-less' tag and recipes without this tag.

<iframe
  src="pictures/hypothesis_test1.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


3. relationship between calories and preparation steps. 

Null Hypothesis: all recipes provide equal amounts of calories 

Alternative Hypothesis: recipes with a '3-steps-or-less' tag provide less amounts of calories than other recipes. 

<iframe
  src="pictures/hypothesis_test2.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The observed difference is -65.736 and the p-value is 0. 

The null hypothesis is not supported. 
We fail to reject the alternative hypothesis.

## Step 5: Framing a Prediction Problem

### Prediction Problem: 
- Predicting the number of steps needed to cook a recipe.
- **Type**: Regression

### Approach

We aim to predict the number of steps required to cook a recipe using a regression model. Our baseline model will be **Linear Regression** to capture the linear relationship between the input features and the target variable.

### Features Used

- **Calories**: Represents the nutritional content of the recipe.
- **Minutes**: Indicates the preparation time of the recipe.

### Model Choice

We opt for **Linear Regression** as our baseline model. This model will help us estimate the coefficients of 'calories' and 'minutes', providing insights into their respective impacts on the number of steps.

### Evaluation Metric

To assess the model's predictive accuracy, we will use the **Root Mean Squared Error (RMSE)** metric. RMSE quantifies the average deviation between the predicted and actual number of steps, offering a comprehensive measure of prediction error.

### Data Preparation

- Split the dataset into training and test sets.
- Standardize 'calories' and 'minutes' using StandardScaler to ensure they are on a comparable scale.

### Model Evaluation

We will evaluate the model's performance on the test set using RMSE.

### Further Exploration

To enhance our model and mitigate potential issues like multicollinearity, we will:
- Investigate additional feature combinations (e.g., 'protein', 'fat', 'carbohydrates').
- Create new features such as 'calories per minute' or 'calories per step'.
- Consider advanced techniques such as **Lasso Regression**, **Principal Component Analysis (PCA)**, and **Cross-Validation** to avoid overfitting and improve model robustness.

## Step 6: Baseline Model

### Statistical Analysis of the Baseline Model

1. **Root Mean Squared Error (RMSE)**:
   - **Training RMSE**: 6.38
   - **Testing RMSE**: 6.32

#### Interpretation

- **RMSE**: This metric represents the average deviation between the predicted number of steps and the actual number of steps. Lower values indicate better predictive accuracy.
  - **Training RMSE (6.38)**: This value indicates that, on average, the predicted number of steps deviates from the actual number by approximately 6.38 steps in the training set.
  - **Testing RMSE (6.32)**: This value indicates that, on average, the predicted number of steps deviates from the actual number by approximately 6.32 steps in the test set.

2. **Model Performance**:
   - The close values of training RMSE and testing RMSE suggest that the model has generalized well to the test set and is not overfitting. This indicates a good fit of the model to the data given the features used (`calories` and `minutes`).

3. **Feature Importance**:
   - While Linear Regression provides coefficients that can indicate the importance of each feature, the actual coefficients were not included in the provided code. However, we can infer that both `calories` and `minutes` play a significant role in predicting the number of steps.

4. **Model Simplicity**:
   - Using only two features (`calories` and `minutes`) makes the model simple and interpretable. However, it might not capture the full complexity of the recipe steps, indicating room for improvement by including additional features or engineering new ones.

### Conclusion

The baseline Linear Regression model shows promising results with an RMSE of 6.38 on the training set and 6.32 on the testing set. These results suggest that the model is performing reasonably well, but there is potential for improvement through additional features, advanced modeling techniques, and hyperparameter tuning.

## Step 7: Final Model

### Description

#### Prediction Problem
The task is to predict the number of steps required to cook a recipe, formulated as a regression problem.

#### Baseline Model
Initially, a baseline model was developed using `LinearRegression`, focusing on the linear relationship between input features (`calories`, `minutes`) and the target variable (`n_steps`). The baseline model provided insights into the impacts of these features on the number of steps, using RMSE as the evaluation metric. The baseline model achieved a training RMSE of 6.38 and a testing RMSE of 6.32.

#### Final Model
To improve upon the baseline model, we engineered new features and explored a more sophisticated model. Specifically, we implemented a `RandomForestRegressor` model. The final model pipeline included feature scaling and hyperparameter tuning using `GridSearchCV`.

### Steps and Implementation

1. **Feature Selection**:
    - Selected features: `calories`, `minutes`, `carbohydrates`, and `protein`.
    - Target variable: `n_steps`.

2. **Train-Test Split**:
    - Split the dataset into training (80%) and testing (20%) sets using `train_test_split`.

3. **Feature Scaling**:
    - Standardized the numeric features using `StandardScaler`.

4. **Pipeline Construction**:
    - Created a pipeline combining feature scaling and the `RandomForestRegressor`.

5. **Hyperparameter Tuning**:
    - Defined the hyperparameters for tuning:
        - `n_estimators`: Number of trees in the forest.
        - `max_depth`: Maximum depth of the tree.
        - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
    - Used `GridSearchCV` to search for the best combination of hyperparameters with a 5-fold cross-validation, optimizing for negative mean squared error.

6. **Model Training and Evaluation**:
    - Fitted the `GridSearchCV` to the training data to find the best estimator.
    - Made predictions on both the training and testing sets.
    - Evaluated the model using RMSE.

### Results

- **Training RMSE**: Achieved an RMSE of 1.1137512591694114.
- **Testing RMSE**: Achieved an RMSE of 3.016534652563711.
- **Best Hyperparameters**:
    - `n_estimators`: 200
    - `max_depth`: None
    - `min_samples_leaf`: 1

The final model significantly improved the testing RMSE compared to the baseline model, demonstrating that the `RandomForestRegressor` with hyperparameter tuning and feature engineering provided a better fit for the prediction task.

### Conclusion

By incorporating feature scaling and leveraging the ensemble power of `RandomForestRegressor` with optimal hyperparameters, the final model effectively reduced the prediction error. The model is able to predict the number of steps very accurately using other features. 

## Step 8: Fairness Analysis

### Fairness Analysis of the Final Model

In this analysis, we will evaluate whether our final model, `best_model`, performs worse for recipes with low calories compared to recipes with high calories. Specifically, we will assess the model's fairness based on the RMSE metric.

#### Groups Definition

- **Group X**: Recipes with calories below the median (`low-calories`).
- **Group Y**: Recipes with calories above the median (`high-calories`).

#### Evaluation Metric

- **RMSE** (Root Mean Squared Error): Measures the average magnitude of the prediction error, providing a comprehensive measure of prediction accuracy.

#### Hypotheses

- **Null Hypothesis (H0)**: The model is fair. The RMSE for low-calories and high-calories recipes are roughly the same, and any differences are due to random chance.
- **Alternative Hypothesis (H1)**: The model is unfair. The RMSE for low-calories recipes is higher than the RMSE for high-calories recipes.

#### Permutation Test Implementation

1. **Calculate the median of the `calories` column**.
2. **Split the test set into two groups based on the median value**:
    - Group X: Recipes with `calories` below the median.
    - Group Y: Recipes with `calories` above the median.
3. **Compute the RMSE for each group using the `best_model`**.
4. **Perform the permutation test** to assess the significance of the observed difference in RMSE between the two groups.


### Reasons for Calorie-Based Fairness Analysis

#### Understanding Fairness in Machine Learning

Fairness in machine learning ensures that predictive models do not disproportionately favor or disadvantage certain groups. By evaluating model performance across different groups, we aim to identify and mitigate potential biases, making the model more reliable and equitable for all users.

#### Why Calorie-Based Groups?
- **Health Conscious Users**: Recipes with lower calories are often preferred by health-conscious individuals, those managing weight, or individuals with specific dietary requirements. It's important that these users receive accurate predictions to plan their meals effectively.
- **Nutritional Accuracy**: High-calorie recipes may contain more ingredients or complex preparation steps. Ensuring accurate predictions for these recipes helps users maintain balanced diets and avoid unintended health consequences.

<iframe
  src="pictures/histogram.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Results and Conclusion

#### Results

- **RMSE for low-calories group**: 2.66
- **RMSE for high-calories group**: 3.33
- **Observed RMSE difference**: 0.67
- **Permutation test p-value**: 0.0

#### Conclusion

The fairness analysis of our final model, `best_model`, was conducted to assess whether the model performs differently for recipes with low calories compared to those with high calories. The RMSE for the low-calories group was found to be 2.66, while the RMSE for the high-calories group was 3.33. This indicates that the model is more accurate for recipes with lower calories.

The permutation test yielded a p-value of 0.0, which is well below the significance level of 0.05. This result allows us to reject the null hypothesis that the model's performance is the same for both groups. Instead, we accept the alternative hypothesis that the model's performance differs significantly between the low-calories and high-calories groups.

Given this significant difference, we conclude that our model demonstrates a performance bias, favoring recipes with lower calories. This insight is crucial for further refining the model to ensure fair and equitable performance across different groups.