# **Values to Users**
    - New insightful features for better modeling
    - Dimension reduction comparison between LDA and PCA
    - Detailed explanation for LDA
    - Stacking

From this data challenge, you will know some great features for mobile price prediction, how to use LDA to perform dimension reduction and why LDA outperform than PCA in this particular scenario(Classification Problem).

# **Data Set Information:**

The dataset in question is an extensive compilation of mobile phone specifications, capturing a wide array of features that define the capabilities and functions of each device. These features encompass a broad range of mobile phone properties, from hardware specifications such as RAM and Internal Memory, to various other characteristics like screen size, battery life, camera quality, and more. The primary objective of this dataset is to facilitate an accurate prediction of mobile phone price ranges based on these feature sets. The dataset has been assembled through meticulous data collection efforts aimed at recording sales figures and corresponding features of mobile phones from a multitude of companies.

# **Business Problem:**

Bob has embarked on an entrepreneurial journey by launching his own mobile company. His ambition is to pose a significant challenge to market heavyweights such as Apple and Samsung. However, Bob confronts a critical challenge: he lacks a reliable mechanism to estimate the pricing of the mobile phones his company is producing. In this fiercely competitive market, pricing cannot be based on assumptions or gut feelings; it needs to be backed by data-driven insights and an understanding of what features drive the value of a phone.

Hence, Bob has undertaken a strategic initiative: he has collected sales data of various mobile phones from diverse companies, hoping to uncover correlations between the features of a mobile phone (like RAM, Internal Memory, etc.) and its selling price. Unfortunately, Bob doesn't possess the requisite expertise in Machine Learning to derive meaningful insights from this data. That's where we come in. Bob seeks our assistance to analyze this data and create a model that can predict a price range for a given set of mobile features. Notably, the task at hand doesn't necessitate predicting the precise price, but rather a price bracket that signifies the relative costliness of the device. This crucial information will equip Bob's company with the ability to strategically price their devices in a way that balances competitive positioning and profitability.

**Model Type:**

Classification

**Feature Category:**

binary, continuous, discrete


**Indenpendent Feature X:**
1. battery_power (Continuous): The total energy that a battery can store, measured in mAh.

2. blue (Binary): Whether the phone has Bluetooth support (1) or not (0).

3. clock_speed (Continuous): Speed at which microprocessor executes instructions.

4. dual_sim (Binary): Whether the phone supports dual SIM cards (1) or not (0).

5. fc (Discrete): The megapixel count of the front camera.

6. four_g (Binary): Whether the phone supports 4G (1) or not (0).

7. int_memory (Discrete): Internal memory of the phone in gigabytes.

8. m_dep (Continuous): Mobile depth in cm.

9. mobile_wt (Discrete): Weight of the mobile phone.

10. n_cores (Discrete): Number of processor cores.

11. pc (Discrete): Megapixel count of the primary camera.

12. px_height (Discrete): Pixel resolution height.

13. px_width (Discrete): Pixel resolution width.

14. ram (Discrete): Random Access Memory in megabytes.

15. sc_h (Continuous): Screen height in cm.

16. sc_w (Continuous): Screen width in cm.

17. talk_time (Discrete): Longest time that a single battery charge will last.

18. three_g (Binary): Whether the phone supports 3G (1) or not (0).

19. touch_screen (Binary): Whether the phone has a touch screen (1) or not (0).

20. wifi (Binary): Whether the phone has wifi (1) or not (0).

**Dependent Feature Y:**
price_range (Categorical): It is an ordinal categorical variable with four classes representing the price range: 0 (low cost), 1 (medium cost), 2 (high cost), 3 (very high cost).

# **Solution Overview**

## 1. Load Data
 Load and showcase Data

## 2. Data Cleaning
 handling missing values and checking data type
 Check data balance




## 3. Benchmark Solution
 Since the data are clean and of numeric type, Logistic regression training is directly used as the baseline, and standardization should be used to unify the scale before the linear model is used

## 4. Feature Engineering
  Using existing data to create many new features have many benefits: Enhanced Model Performance,Better Representation of Data,Incorporation of Domain Knowledge,Reducing Feature Dimensionality,Improved Interpretability,Addressing Non-Linearity,Temporal and Spatial Insights

## 5. Feature Selection with PCA
 If there is too much features to train,feature selection and data dimensionality reduction is necessary. Data dimensionality reduction refers to the process of transforming data from a high-dimensional space to a lower-dimensional space. The goal of this process is to reduce the complexity of the dataset while retaining as much relevant information as possible.
 
 Here, the most commonly used PCA is firstly used as the dimensionality reduction method.

 PCA, or Principal Component Analysis, is a statistical method used for dimensionality reduction in datasets. The primary goal of PCA is to identify patterns in data and express the data in a way that highlights their similarities and differences.

## 6. Feature Selection with LDA
 As a result, PCA not only did not have a positive impact on metric, it actually reduced metric. There may be several reasons for this:

 1. Loss of Information: When you reduce the dimensionality of your data using PCA, you're essentially projecting the data onto a lower-dimensional subspace. While PCA tries to retain the directions with the highest variance, some information is inevitably lost during this process. If the discarded dimensions contain important signals for prediction, the model performance may suffer.

 2. Changed Feature Space: The transformed feature space after applying PCA might not be as suitable for the model as the original features. Certain algorithms, especially non-linear ones, might perform better with the original features rather than the principal components.

 3. Over-Compression: If you aggressively reduce the dimensionality (e.g., keeping too few principal components), you might lose too much information. This over-compression can lead to poor performance because the model doesn't have enough information to make accurate predictions.

 4. Unsuitable for Certain Data Types: PCA makes assumptions about the data, such as linearity. If the data doesn't meet these assumptions (e.g., if intrinsic manifolds in the data are non-linear), PCA might not be the best choice for dimensionality reduction.

 5. Loss of Interpretability: The principal components are linear combinations of the original features. The model becomes less interpretable as these new features don't have a direct and clear relationship with the original features. This might not necessarily affect the metrics, but it is a consideration for practical applications where interpretability is important.

 LDA, or Linear Discriminant Analysis, is both a dimensionality reduction technique and a classification method. It is primarily used in the context of supervised learning, where data points are labeled with certain classes.
 LDA is particularly beneficial for classification tasks because it reduces dimensions in a way that preserves and emphasizes the differences between classes, which is crucial for classification.So LDA is worth a try.

## 7.1 Model Enhancement - Gradient Boost
 Tree models tend to perform better on complex data sets because they handle nonlinear relationships better and are more robust to linear models.

 Gradient Boosting is a machine learning algorithm used for regression and classification problems. It works by sequentially adding predictors to an ensemble, each correcting its predecessor. The method leverages the concept of gradient descent to minimize errors, hence "boosting" the model's accuracy over several iterations of training.

 When PCA is used for dimensionality reduction and the metric does not increase but decreases, it is evident that PCA is indeed not suitable for classification.


## 7.2 Model Enhancement - Random Forest
Random Forest is a versatile machine learning method capable of performing both regression and classification tasks. It utilizes a multitude of decision trees operating as an ensemble; each tree is trained on a random subset of data and features. Final predictions are made by averaging outputs (regression) or majority voting (classification).

RandomForest improves the metric of the model by another 6%.


## 7.3 Model Enhancement - Stacking

 Stacking, short for stacked generalization, is an ensemble learning technique that combines multiple machine learning models in order to achieve better predictive performance compared to using a single model alone.

