## Google Data Scientist Mock Interview
  The goal of the data challenge is to assess the candidate's skills in data manipulation, feature processing, machine learning modeling, and extracting business insights. The candidates will be asked a series of questions related to these aspects with PySpark coding skills, utilizing the provided dataset.
  
## Business Problem:

As a data scientist at Google, your mission is to leverage customer data and develop a robust predictive model to identify and anticipate customer churn. Customer churn, a critical concern for Google, refers to the occurrence when customers discontinue their engagement and stop transacting with the company. Your objective is to analyze the provided dataset, which encompasses a comprehensive range of customer behavior and transactional features. By accurately identifying potential churners, Google can proactively implement targeted strategies to retain these customers, enhance customer satisfaction, and drive long-term business growth.

## Questions Roadmap
    - Data Manipulation
        - Analyze Gender distribution among churned customers
        - Calculate customers' loyalty group
        - Analyze male age group income
        - Calculate churn per month
    - Machine Learning
        - Data Preprocessing, Modeling


## Data Set Information:

The company has provided you with a dataset containing information about its customers. The dataset includes the following columns:

|Column	| Description |
|--------------|-------------|
|customer_id	| the unique identifier for each customer. |
|age	| the age of each customer.|
|gender	| the gender of each customer.|
|income	| annual income of each customer.|
|loyalty	| the number of years each customer has been with the company|
|churn	| indicating whether the customer churned (1) or not (0).|
|creation_date |	date when the user signed up|

## Solution Overview

### Data Manipulation
#### Q1_Answer:
It filters the data DataFrame to include only the churned customers. This is done by applying a filter condition where the churn column is equal to 1. After filtering, the churned customers are grouped by their gender using the groupBy() method. This groups the data based on the values in the "gender" column. The count() method is applied to the grouped data, which calculates the number of male and female customers in each group.

In summary, this code aims to analyze the gender distribution among churned customers by filtering the data, grouping it by gender, and counting the number of male and female customers in each category. The output provides insights into how churn is distributed among different genders.

#### Q2_Answer:
The code filters the data DataFrame to include only the churned customers. This is done by applying a filter condition where the churn column is equal to 1. The churned customers are then divided into loyalty groups based on their loyalty score. The withColumn() method is used to create a new column called "loyalty_group". The when function is applied to specify the conditions for each loyalty group. Customers with a loyalty score less than 4 are classified as "Low", customers with a loyalty score between 4 and 7 (inclusive) are classified as "Medium", and customers with a loyalty score greater than or equal to 7 are classified as "High". The code then groups the churned customers by their loyalty group using the groupBy() method. The count() method is applied to the grouped data, which calculates the number of people in each loyalty group.

In summary, this code aims to analyze the distribution of churned customers based on their loyalty levels. It filters the data to include only churned customers, categorizes them into low, medium, and high loyalty groups, and calculates the count of people in each group. The output provides insights into the distribution of churned customers across different loyalty levels.

#### Q3_Answer:
It generates a new column called "age_group" in the DataFrame data. The withColumn() method is used to create the column, and the when function is applied to define the conditions for each age group. Customers with an age less than 30 are categorized as "<30", customers with an age between 30 and 50 (inclusive) are categorized as "30-50", and all other customers are categorized as "50+". The DataFrame data_with_age_group is then filtered to include only male customers. This is done by applying a filter condition where the "gender" column is equal to "Male". The male customers are grouped by both "churn" and "age_group" using the groupBy() method. The agg() function is used to apply an aggregation operation on the grouped data. In this case, the avg() function is applied to calculate the average income for each group. The resulting column is aliased as "avg_income" using the alias() method.

In summary, this code aims to analyze the relationship between churn, age groups, and average income among male customers. It generates the age groups based on the customers' age, filters the data to include only male customers, calculates the average income for each age group among males, and displays the results. The output provides insights into how churn, age groups, and income are related among male customers.

#### Q4_Answer:
The "creation_time" column in the DataFrame "data" is converted from a string type to a date type using the "to_date()" function. The window is partitioned by the year and month of the "creation_time" column and ordered by the "creation_time". The "sum()" function is applied as a window function using the "withColumn()" method to calculate the churn count per month. The result is stored in a new column called "churn_count". The "select()" method is used to select the "year", "month", and "churn_count" columns from the DataFrame "data_with_churn_count". The "distinct()" method is called on the DataFrame "churn_count_per_month" to remove any duplicate rows. The churn count per month is sorted in ascending order of the year using the "orderBy()" method on the DataFrame "churn_count_per_month".

In summary, this code aims to analyze the churn count per month based on the "creation_time" column. It converts the date column to a date type, calculates the churn count per month using a window function, selects the relevant columns, sorts the result by year, and displays the churn count per month. The output provides insights into the churn patterns over time.

### Machine Learning
#### Data Preprocessing
This code snippet applies one-hot encoding to the "gender" column in the DataFrame "data". It uses the "StringIndexer" class to assign numerical indices to each unique value in the column. The transformed DataFrame includes a new column called "gender_index" that contains the encoded indices.

In summary, this code snippet performs one-hot encoding on the "gender" column in the DataFrame "data". It uses the "StringIndexer" class to encode the categorical labels into numerical indices and adds a new column with the encoded indices.

#### Modeling
It creates a feature vector by selecting the relevant columns "age", "gender_index", "income", and "loyalty". These columns are combined into a single vector column named "features" using the VectorAssembler class. The data is split into training and test sets using the randomSplit() method, with 80% of the data assigned to the training set and 20% to the test set. The seed parameter is set to ensure reproducibility. A Gradient Boosted Tree classifier (GBTClassifier) is instantiated, with the input features column set to "features" and the label column set to "churn". A pipeline is created using the Pipeline class, which defines the sequence of transformations and the model to be trained. In this case, the pipeline consists of the VectorAssembler and GBTClassifier stages. The feature importances are extracted from the trained model's last stage (GBTClassifier) using the featureImportances attribute. The feature importance scores are printed, showing the importance of each feature in predicting churn. Predictions are made on the test data and stored in the "predictions" DataFrame. The evaluator is used to calculate various metrics to evaluate the model's performance. The accuracy of the model is calculated using the evaluate() method of the evaluator. Other metrics such as weighted precision, weighted recall, and weighted F1-score are calculated with appropriate metric names.

In summary, this code snippet builds a pipeline to train a Gradient Boosted Tree classifier for predicting customer churn. It combines relevant features, trains the model, evaluates its performance using various metrics, and prints the results.

