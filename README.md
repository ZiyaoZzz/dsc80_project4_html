# Recipe saturated fat study
Name(s): Kate Zhou, Ziyao Zhou
<<<<<<< HEAD

Website Link: [title](https://ziyaozzz.github.io/dsc80_project4_html/)

# Overview
This data science project for DSC80 at UCSD investigates the correlation between a recipe's rating and the percentage of total calories in the recipe that come from sugar.

# Step 1: Introduction
Questions:

The relationship between food preparation steps and nutrition.
According to U.S. Bureau of Labor Statistics, among people 15 and older in the United States, 57.2 percent spent averaged 53 minutes preparing food and drink on an average day in 2022. As college students, we often prefer to cook simple dish and ignore the quality and nutrition of the dish. We wonder if food cooked in less time and steps have poorer nutrition than the food cooked in longer time and more steps.


Here is a brief introduction to the recipe dataset, contains 83782 rows (unique recipes), with 10 columns recording the following information:

| Column | Description |
|:---:|:---:|
| 'name' | Recipe name |
| 'id' | Recipe ID |
| 'minutes' | Minutes to prepare recipe |
| 'contributor_id' | User ID who submitted this recipe |
| 'submitted' | Date recipe was submitted |
| 'tags' | Food.com tags for recipe |
| 'nutrition' | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value” |
| 'n_steps' | Number of steps in recipe |
| 'steps' | Text for recipe steps, in order |
| 'description' | User-provided description |
| 'ingredients' | Text for recipe ingredients |
| 'n_ingredients' | Number of ingredients in recipe |

Here is a brief introduction to the interactions dataset, contains 731927 rows and each row contains a review from the user on a specific recipe. The columns it includes are:

| Column | Description |
|:---:|:---:|
| 'user_id' | User ID |
| 'recipe_id' | Recipe ID |
| 'date' | Date of interaction |
| 'rating' | Rating given |
| 'review' | Review text |

Given the datasets, we are investigating whether the number of food preparation steps affects the nutritional quality of the food. To facilitate the investigation of our question, we separated the values in the 'nutrition' columns into the corresponding columns, 'calories (#)', 'total fat (PDV)', 'sugar (PDV)', etc. PDV, or percent daily value, shows how much a nutrient in a serving of food contributes to a total daily diet. Moreover, we calculated the number of preparation steps for each recipe and stored this information in a new column, 'prep_steps'. Recipes were then categorized based on the average number of steps, where those with 'prep_steps' higher than the average were considered more complex.

The most relevant columns to answer our question are 'calories (#)', 'total fat (PDV)', 'sugar (PDV)', 'prep_steps', and 'nutrition_score', which is a composite score reflecting the overall nutritional quality of a recipe.

By seeking an answer to our question, we hope to gain insight into whether simpler recipes, in terms of preparation steps and time, tend to be less nutritious compared to more complex recipes. This could help individuals, especially college students, make better-informed choices about their cooking habits. Additionally, this information could lead to future work on promoting awareness about the nutritional impact of cooking methods and preparation steps.

# Data Cleaning and Exploratory Data Analysis

## Univariate Analysis
For this analysis, we examined the distribution of the number of steps in a recipe. As the plot below shows, the distribution is skewed to the right, indicating that most of the recipes on food.com have a low number of steps. There is also a decreasing trend, indicating that as the number of steps in recipes increases, there are fewer of those recipes on food.com.



## Bivariate Analysis
Additionally, we analyzed the relationship between the number of steps in a recipe and its saturated fat content, distinguishing between different sources. The line plot below shows the trends for this relationship (min, max, median, mean), providing insight into how the complexity of a recipe (as measured by the number of steps) correlates with its healthiness (as measured by saturated fat content).



## Interesting Aggregates
For this section, we investigated the relationship between the number of steps in a recipe and its saturated fat content using pivot table. 
The initial analysis suggests that dishes prepared with more steps tend to have higher nutritional content, both in terms of potentially less desirable elements like saturated fat and sugar, as well as beneficial ones like protein. This might indicate that more complex recipes, despite taking more time and effort, could offer more balanced and richer nutritional profiles.

Further statistical tests will be needed to confirm the significance of these observations and ensure that the trends observed are not due to random variation in the sample data.




# Step 3: Assessment of Missingness
Based on our research dataframe, the only two columns that have missing value are rating and review, which have some missing values (15036 missing in he rating column and 58 missing in the review column).

## NMAR Analysis
We believe that the absence of ratings in the 'review' column is likely Not Missing at Random (NMAR). This inference arises from the notion that individuals tend to write review to recipes based on the strength of their emotional response. If someone feels indifferent towards a recipe, they may be less inclined to provide a rating, resulting in missing values. Conversely, individuals who have strong positive or negative sentiments towards a recipe are more motivated to assign a rating, reflecting their satisfaction or dissatisfaction. This behavior suggests that the missingness in the 'rating' column is tied to the level of emotional engagement individuals have with the recipes. Understanding this pattern of missingness is crucial for accurately interpreting user feedback and conducting sentiment analysis within the dataset.

## Missingness Dependency
We proceeded to investigate the missingness of 'rating' in the merged DataFrame by examining its dependency on two key factors: 'saturated fat (PDV)', representing the proportion of sugar out of the total calories, and 'n_steps', denoting the number of steps in the recipe. 
The null hypothesis: The missingness of ratings is independent of the saturated fat (PDV) in the recipe.
The alternative hypothesis: The missingness of ratings is dependent by the saturated fat (PDV) in the recipe. 
To test this, we calculated the absolute difference in mean proportion of saturated fat between the group with missing ratings and the group without missing ratings. Our significance level was set at 0.05 to determine the statistical significance of the findings.

# Hypothesis Testing

The relationship between rare receipe items and their preparation time and steps.

**Null Hypothesis**: all recipes are equally healthy(same in Saturated Fat).

**Alternative Hypothesis**: recipes with '3-steps-or-less' tag are less healthy (higher in Saturated Fat) than other recipes.

**Test Statistic**: the difference in means or medians of recipes with '3-steps-or-less' tag and recipes without this tag.
=======
Website Link: https://ziyaozzz.github.io/dsc80_project4_html/
>>>>>>> d9b066a33be5ed9e48c23b17bd2e39cf3c4ff55b
