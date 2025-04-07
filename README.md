# Breast Cancer Dataset Analysis with PCA and Logistic Regression

## Overview

This project demonstrates the application of Principal Component Analysis (PCA) for dimensionality reduction on the Breast Cancer dataset from `sklearn.datasets`. It also includes an optional step of implementing Logistic Regression for classification on the PCA-transformed data. This analysis aims to identify essential underlying features and build a predictive model.

## Project Structure

The Jupyter Notebook (`your_notebook_name.ipynb` - replace with the actual name of your notebook) contains the following steps:

1.  **Load the Dataset:** Loads the Breast Cancer dataset using `sklearn.datasets`.
2.  **Exploratory Data Analysis (EDA):** Provides initial insights into the dataset, including feature names, shape, summary statistics, missing values, and the distribution of the target variable.
3.  **Standardization:** Scales the features using `StandardScaler` to have zero mean and unit variance, which is crucial for PCA.
4.  **PCA Implementation (Explained Variance):** Applies PCA to compute all principal components and analyzes the explained variance ratio of each component. Visualizations are provided to help determine the optimal number of components.
5.  **Dimensionality Reduction:** Reduces the dataset to 2 principal components using PCA. The transformed data is then visualized in a 2D scatter plot, colored by the target variable.
6.  **Bonus: Logistic Regression with PCA-transformed Data:**
    * Splits the 2-component PCA data into training and testing sets.
    * Trains a Logistic Regression model on the PCA-transformed training data.
    * Evaluates the model's performance using accuracy, classification report, and confusion matrix.
    * Visualizes the decision boundary of the Logistic Regression model on the 2D PCA space.

## Libraries Used

* `pandas`: For data manipulation and analysis.
* `sklearn.datasets`: To load the Breast Cancer dataset.
* `sklearn.preprocessing`: For data standardization (`StandardScaler`).
* `sklearn.decomposition`: For Principal Component Analysis (`PCA`).
* `matplotlib.pyplot`: For creating static, interactive, and animated visualizations.
* `seaborn`: For making statistical graphics more visually appealing and informative.
* `sklearn.model_selection`: For splitting the data into training and testing sets (`train_test_split`).
* `sklearn.linear_model`: For implementing Logistic Regression (`LogisticRegression`).
* `sklearn.metrics`: For evaluating the classification model (`accuracy_score`, `classification_report`, `confusion_matrix`).
* `numpy`: For numerical operations, especially when creating the meshgrid for the decision boundary plot.
* `warnings`: To handle and potentially suppress warning messages.

## How to Run the Code

1.  **Prerequisites:** Ensure you have Python 3 installed along with the necessary libraries. You can install the required libraries using pip:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn numpy
    ```
2.  **Jupyter Notebook:** The code is designed to be run in a Jupyter Notebook environment. If you don't have Jupyter Notebook installed, you can install it using:
    ```bash
    pip install notebook
    ```
3.  **Execution:**
    * Save the provided Python code as a `.ipynb` file (e.g., `pca_analysis.ipynb`).
    * Open the Jupyter Notebook by running `jupyter notebook` in your terminal or command prompt.
    * Navigate to the saved `.ipynb` file and open it.
    * Execute each cell of the notebook sequentially to run the analysis and see the output and visualizations.

##Explanation of the Code
Step 1: Import Libraries: Import necessary libraries:

- pandas for data manipulation.
- load_breast_cancer from sklearn.datasets to load the dataset.
- StandardScaler from sklearn.preprocessing for standardization.
- PCA from sklearn.decomposition for Principal Component Analysis.
- matplotlib.pyplot and seaborn for visualization.
- train_test_split from sklearn.model_selection for splitting data.
- LogisticRegression from sklearn.linear_model for the classification model.
- accuracy_score, classification_report, and confusion_matrix from sklearn.metrics for model evaluation.
- numpy for numerical operations (used in the bonus section for plotting the decision boundary).

Step 2: Load the Dataset:

- Loads the Breast Cancer dataset using load_breast_cancer().
- Creates a Pandas DataFrame X for the features and a Pandas Series y for the target variable.
- Prints basic information about the dataset.

Step 3: Exploratory Data Analysis (EDA):

- Prints the first 5 rows of the feature data.
- Provides descriptive statistics of the features using describe().
- Checks for missing values using isnull().sum().
- Displays the distribution of the target variable (malignant vs. benign) using value_counts() and a countplot.

Step 4: Standardization:

- Initializes a StandardScaler.
- Fits the scaler to the feature data X and then transforms it to obtain the standardized data X_scaled.
- Creates a DataFrame X_scaled_df from the scaled data.
- Prints the first 5 rows of the scaled data and verifies that the mean is close to 0 and the standard deviation is close to 1 for the first feature.

Step 5: PCA Implementation (Calculating Explained Variance):

- Initializes a PCA object without specifying the number of components (to calculate all principal components).
- Fits PCA to the scaled data X_scaled.
- explained_variance_ratio_ attribute gives the proportion of variance explained by each principal component.
- cumulative_explained_variance_ calculates the cumulative sum of the explained variance ratios.
- Prints the explained variance ratio for each component and the cumulative explained variance.
- Visualizes the explained variance ratio and cumulative explained variance using line plots to help determine the optimal number of components.

Step 6: Dimensionality Reduction (Reducing to 2 Principal Components):

- Initializes a PCA object with n_components=2 to reduce the dimensionality to two principal components.
- Fits PCA to the scaled data X_scaled and transforms it to obtain the reduced data X_pca_2.
- Creates a Pandas DataFrame df_pca from the reduced data with columns 'PC1' and 'PC2'.
- Adds the original target variable y to the DataFrame for visualization.
- Prints the first 5 rows of the reduced DataFrame.
- Visualizes the reduced data using a scatter plot, with different colors for malignant and benign tumors.

Bonus: Logistic Regression with PCA-transformed Data:

- Splits the PCA-transformed data (df_pca[['PC1', 'PC2']]) and the target variable (df_pca['target']) into training and testing sets.
- Initializes and trains a LogisticRegression model using the PCA-transformed training data.
- Makes predictions on the PCA-transformed test data.
- Evaluates the performance of the Logistic Regression model using accuracy score, classification report, and confusion matrix.   
- Visualizes the decision boundary of the Logistic Regression model on the 2D PCA-transformed data. This helps to understand how the model separates the two classes in     the reduced feature space.

## Understanding the Output

1. EDA:** The output will show the initial characteristics of the dataset, helping you understand the features and the distribution of the target variable.
2. Standardization:** You'll see the first few rows of the standardized data, confirming that the features have been scaled.
3. PCA Implementation (Explained Variance):** The explained variance ratio for each principal component and the cumulative explained variance will be printed. The accompanying plots will visually illustrate how much variance is captured by each component and the cumulative variance as the number of components increases.
4. Dimensionality Reduction:** The first few rows of the reduced 2-component dataset will be displayed, followed by a scatter plot visualizing the data in the 2D PCA space, with points colored according to their class (malignant or benign).
5. Bonus: Logistic Regression:**
    * The accuracy of the Logistic Regression model on the PCA-transformed test data will be printed.
    * The classification report will provide precision, recall, F1-score, and support for each class.
    * The confusion matrix will show the counts of true positives, true negatives, false positives, and false negatives.
    * The decision boundary plot will visualize how the Logistic Regression model separates the two classes in the 2D PCA space.

## Notes on the Bonus Code (Logistic Regression)

The bonus section includes a fix to address a `UserWarning` related to feature names during prediction for the decision boundary plot. The code now explicitly creates a Pandas DataFrame with the correct column names ('PC1', 'PC2') from the meshgrid data before making predictions with the trained Logistic Regression model. This ensures that the prediction input has the expected feature names, resolving the warning and making the code more robust.

This analysis provides a foundation for understanding the most important underlying features in the Breast Cancer dataset and demonstrates how PCA can be used for dimensionality reduction before applying a classification algorithm like Logistic Regression. The 2 principal components capture a significant portion of the dataset's variance and can be used as a lower-dimensional representation for further modeling or analysis.

Stellamaris Okeh
Stellamarisijeoma0@gmail.com
