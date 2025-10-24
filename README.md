# Heart Disease Analysis and Prediction

## Description:
This repository contains a data analysis and machine learning project aimed at predicting the presence of heart disease in patients. The analysis uses a publicly available dataset from Kaggle to explore various factors that influence heart disease risk. A logistic regression model is built to predict the likelihood of heart disease based on key health attributes such as age, cholesterol levels, blood pressure, and more.

## Key Features:
- **Exploratory Data Analysis (EDA)** to understand key factors affecting heart disease.
- **Data Preprocessing**: Handling missing values, encoding categorical variables.
- **Data Visualizations**: Visualize the distribution of key variables and their relationships using Seaborn and Matplotlib.
- **Logistic Regression Model**: Predict the likelihood of heart disease using machine learning.
- **Model Evaluation**: Evaluate the model's accuracy and performance.

## Dataset:
The dataset used in this project is the **Heart Disease UCI dataset** from Kaggle. It contains medical data about patients and whether they have heart disease (target variable: `target`).

### Columns:
- **age**: Age of the patient
- **sex**: Gender (0 = female, 1 = male)
- **cp**: Chest pain type (4 types)
- **trestbps**: Resting blood pressure
- **chol**: Serum cholesterol
- **fbs**: Fasting blood sugar (> 120 mg/dl = 1, else = 0)
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: Depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels colored by fluoroscopy
- **thal**: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversable defect)
- **target**: Whether the patient has heart disease (1 = yes, 0 = no)

## Technologies Used:
- **Python** (for data manipulation and modeling)
- **Pandas** (for data manipulation)
- **NumPy** (for numerical operations)
- **Seaborn** & **Matplotlib** (for data visualization)
- **scikit-learn** (for machine learning)

## Objectives:
- Explore and visualize patterns in healthcare data that relate to heart disease.
- Build a **predictive model** to assess heart disease risk based on health data.
- Evaluate the model's performance and accuracy.

## Steps:
1. **Data Exploration and Preprocessing**:
   - Load and clean the dataset.
   - Handle missing values, convert categorical variables, and perform other preprocessing steps.
   
2. **Exploratory Data Analysis (EDA)**:
   - Perform univariate, bivariate, and multivariate analysis.
   - Visualize data using histograms, scatter plots, and correlation heatmaps.
   
3. **Model Building**:
   - Build a logistic regression model using scikit-learn.
   - Split the data into training and test sets (80% training, 20% testing).
   - Train the model and evaluate its performance.

4. **Model Evaluation**:
   - Print the accuracy score of the model.
   - Optionally, analyze confusion matrices and ROC curves to assess model performance.

## Results:
The model achieved an accuracy of **82.44%** on the test data, making it a useful tool for predicting heart disease based on key health metrics.

## Challenges Faced:
- **Handling Missing Values**: Some columns had missing data which had to be handled either by imputation or removal.
- **Model Convergence Warning**: The logistic regression model initially had convergence issues, but it was resolved by increasing the number of iterations.
- **Data Imbalance**: The dataset has a slight imbalance between the two target classes (0 and 1), which can affect model performance.

## Future Work:
- **Hyperparameter Tuning**: Use grid search or randomized search to fine-tune model hyperparameters.
- **Alternative Models**: Test other machine learning models (e.g., Random Forest, SVM) to improve prediction accuracy.
- **Deploying the Model**: Create an interactive web app to allow users to input their health data and get predictions on heart disease risk.

## License:
This project is open-source and available under the [MIT License](LICENSE).

