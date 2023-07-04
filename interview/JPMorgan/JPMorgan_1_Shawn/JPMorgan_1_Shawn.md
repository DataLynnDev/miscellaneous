## JPMorgan&Chase Data Scientist Mock Interview

 The goal of the data challenge is to assess the candidate's skills in data manipulation, feature processing, machine learning modeling, and extracting business insights. The candidates will be asked a series of questions related to these aspects with PySpark coding skills, utilizing the provided dataset.
 
## **Business Problem Statement**

As a financial institution, J.P. Morgan has a vast amount of transaction data. A significant issue we face is the detection and prevention of fraudulent transactions. Your task as a data scientist is to build a model that can predict potential fraudulent transactions using machine learning.

This problem requires a deep understanding of financial transactions, solid skills in data preprocessing and merging, as well as a strong grasp of advanced machine learning algorithms.

## Questions Roadmap
- Data Manipulation
	- Unusual times transactions
	- Patterns of unusual times transactions
	- Merchants connected to these account
- Feature Engineering
	- Engineering new features
- Modeling
	- Handling imbalanced dataset
	- Choose from different models
	- Time-series cross-validation
- Business Insights

## Dataset Description

***Table 1: Transactions***

| Column Name | Description |
| ------------- | ------------- |
| Transaction_ID | Unique identifier for each transaction |
| Account_ID | Unique identifier for each account involved in the transaction |
| Transaction_Date | The date when the transaction occurred |
| Transaction_Amount | The amount of the transaction in USD |
| Merchant_ID | Unique identifier for the merchant where the transaction took place |
| Merchant_Category | The category of the merchant (e.g., supermarket, electronics, online shopping, etc.) |
| Transaction_Type | The type of the transaction (e.g., POS, ATM, Online, etc.) |
| Is_Fraud | Whether the transaction was marked as fraudulent (1: Yes, 0: No) |

***Table 2: Accounts***

| Column Name | Description |
| ------------- | ------------- |
| Account_ID | Unique identifier for each account |
| Account_Creation_Date | The date when the account was created |
| Account_Type | The type of the account (e.g., Savings, Current, Credit Card, etc.) |
| Account_Balance | The current balance in the account |
| Account_Owner_Age | The age of the account owner |
| Account_Owner_Gender | The gender of the account owner |
| Account_Owner_City | The city where the account owner resides |

## **Questions Overview**

### **Data Manipulation**

#### Q1. 
Some transactions occur at unusual times of the day. Can you identify accounts that frequently transact outside standard business hours (9AM-5PM)? (Python)

First, it merges two DataFrames, transactions df and accounts df, based on the 'Account ID' column using an inner join, resulting in the merged df DataFrame.

Then, it creates a new column 'Transaction Time' in the merged df DataFrame by converting the 'Transaction Date' column of the transactions df DataFrame to a datetime object and extracting the time component.

Next, it sets the business hours start and end times as time objects.

After that, it filters the merged df DataFrame to find unusual transactions occurring outside the business hours, storing them in the unusual transactions df DataFrame.

Finally, it retrieves the unique account IDs from the unusual transactions df DataFrame and assigns them to the unusual account ids variable.

#### Q2.
Using the accounts identified in Q1, can you find out if there's a trend or pattern in the transaction amounts during these unusual hours? (Python)

The code snippet imports the `matplotlib.pyplot` library for plotting. It creates a figure with a size of 10x5. It then groups the `unusual transactions df` DataFrame by the hour of the `Transaction Date`, converts it to a datetime format, and calculates the mean of the `Transaction Amount`. The resulting mean values are plotted as a bar graph using the `plot()` function. The plot is given a title, x-axis label, and y-axis label. Finally, the plot is displayed using the `show()` function. Overall, this code generates a bar graph showing the average transaction amount by hour.

#### Q3.
Within the identified accounts from Q1, are there any that are linked with merchants who conduct most of their transactions outside business hours? (Python)

The code calculates the ratio of unusual transactions for each unique merchant ID by dividing the count of unusual transactions by the count of total transactions. It then identifies the merchants with a ratio greater than 0.5, indicating a higher proportion of unusual transactions. The resulting list, "unusual merchants," contains the unique merchant IDs that meet this criterion.

  "Now, let's move on to feature engineering. Based on the identified accounts and their transaction behavior, we want to create new features."

### **Feature Engineering**
#### Q4.
Given the identified accounts and their transaction behavior, can you engineer a feature that quantifies the irregularity of their transactions? (Python)

The given code performs calculations related to unusual transactions and their correlation with fraud. The first line adds a new column called "Is Unusual" to a DataFrame called "merged df." This column is populated based on a lambda function that checks if each transaction time falls outside the defined business hours, assigning a value of 1 for unusual transactions and 0 otherwise. The next line calculates the correlation between the "Is Unusual" and "Is Fraud" columns in the "merged df" DataFrame. Finally, the last line adds another column called "Transaction Irregularity" to the "merged df" DataFrame. This column contains the average ratio of unusual transactions for each account, calculated using the "mean" function grouped by the "Account ID" column.
#### Q5.
Can you create a feature that captures the frequency of transactions with merchants who conduct most of their business outside normal hours? (Python)

The code snippet creates a new column called 'Merchant Frequency NonBusiness' in a DataFrame called 'merged df'. It uses the 'apply' function to iterate over each row of the DataFrame. For each row, a lambda function is applied, which checks if the value in the 'Merchant ID' column exists in a list called 'unusual merchants'. If the condition is true, it assigns a value of 1 to the 'Merchant Frequency NonBusiness' column; otherwise, it assigns 0. In summary, this code marks the frequency of non-business merchants in the dataset by adding a binary indicator column.

#### Q6.
Based on the accounts' transaction amounts, can you engineer a feature that represents the variability of these amounts, particularly for transactions happening outside business hours? (Python)


The code calculates the standard deviation of the 'Transaction Amount' column for each unique 'Account ID' in the 'merged df' dataframe. The result is then assigned to a new column called 'Transaction Amount Variability' in the same dataframe.

### **Modeling**
#### Q7.
What approach would you use to handle the class imbalance typically seen in fraud detection problems like this one? Explain why you would choose this approach over others. (Python)

**Answer:**

 Handling class imbalance can be approached in several ways. Some popular techniques include:
 1. Over-sampling the minority class
 2. Under-sampling the majority class
 3. Using a combination of over- and under-sampling
 4. Using anomaly detection techniques

 For this particular problem, I would opt for a combination of over-sampling the minority class (fraud transactions)
 and under-sampling the majority class (non-fraud transactions). The reason for choosing this approach is to help
 the model learn and differentiate the characteristics of both classes without causing a significant bias towards
 the majority class. Also, this approach does not lead to loss of data as in the case of only under-sampling.
 We can use SMOTE (Synthetic Minority Over-sampling Technique) to handle the class imbalance.

 
This code snippet performs various data preprocessing tasks. It converts date columns to datetime format, extracts features such as year, month, day, and day of the week from the transaction and account creation dates. It also encodes categorical variables using one-hot encoding and handles missing values. Finally, it applies the SMOTE algorithm to oversample the minority class for balancing the dataset.

#### Q8.
Would you choose a linear model like logistic regression or a tree-based model like random forest for this problem? What factors would influence your decision? How would you test and compare different models? (Python)


**Answer:**

Both logistic regression and random forest can be suitable for this problem, depending on the nature of our data.

Logistic regression assumes that the predictors (X) are independent of each other and have a linear relationship with the log odds of the outcome variable.

On the other hand, random forest is a powerful non-linear model which can capture complex relationships and interactions between variables.
For this specific problem, since we have engineered complex features, a random forest might be better equipped to model these features.
However, it is always good to test both types of models and compare their performance. Cross-validation would be a good technique to use for this comparison.

The provided code snippet uses scikit-learn library to perform classification tasks. It imports three modules: LogisticRegression, RandomForestClassifier, and cross val score from the model selection module.

The code begins by standardizing the input data using StandardScaler. Then, it creates instances of LogisticRegression and RandomForestClassifier models.

Next, cross val score is used to perform cross-validation on the models. The input data X over and labels y over are passed along with the number of folds (cv=3).

Finally, the average scores of logistic regression and random forest models are printed using the mean() function on the respective cross val score results.

#### Q9.
How would you validate your model considering the time-series nature of the data? What specific techniques would you use to avoid leakage and overfitting? (Python)

**Answer：**

Given the time-series nature of the data, it's important to use techniques that are sensitive to this structure.
Simple cross-validation can result in leakage and overfitting because it does not respect the temporal order of observations.
Instead, we can use time-series cross-validation techniques such as forward chaining where we have a rolling basis for the training and test set.
For instance, fold 1 : training [1], test [2]
             fold 2 : training [1 2], test [3]
             fold 3 : training [1 2 3], test [4] and so on.
This way, we respect the temporal order in the data.

The code snippet imports the TimeSeriesSplit class from the sklearn.model selection module and initializes it with a number of splits set to 5. The data X and y are then converted to NumPy arrays. The code then enters a loop where the TimeSeriesSplit is used to generate train-test splits for the data. Inside the loop, the X and y arrays are split into X train, X test, y train, and y test based on the current split indexes. A RandomForestClassifier is created and trained on the training data using the fit() method. Finally, the accuracy score of the trained model is calculated using the score() method on the test data.


### **Business Insights**
#### Q10.
Which features ended up being the most important in your model for predicting fraud? How would you explain the impact of these features to non-technical stakeholders?
#### Q11.
Fraud patterns can change rapidly. How often would you retrain your model? What process would you set up to ensure that it adapts to new fraud patterns without manual intervention?
