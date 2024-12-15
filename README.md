# ProyectoMachineLearning
# **Airbnb Price Prediction using Machine Learning**  
## **Predicción de Precios de Airbnb usando Machine Learning**

---

## **Project Overview / Descripción del Proyecto**  
This project aims to predict Airbnb listing prices using machine learning models. It includes data cleaning, feature engineering, model training, hyperparameter tuning, and evaluation.  
Este proyecto tiene como objetivo predecir los precios de los alojamientos de Airbnb utilizando modelos de machine learning. Incluye limpieza de datos, ingeniería de características, entrenamiento de modelos, ajuste de hiperparámetros y evaluación.

---

## **Table of Contents / Tabla de Contenidos**  
1. [Objective / Objetivo](#objective--objetivo)  
2. [Dataset](#dataset)  
3. [Workflow / Flujo de Trabajo](#workflow--flujo-de-trabajo)  
4. [Technologies Used / Tecnologías Utilizadas](#technologies-used--tecnologías-utilizadas)  
5. [Models and Results / Modelos y Resultados](#models-and-results--modelos-y-resultados)  
6. [Conclusion / Conclusión](#conclusion--conclusión)  
7. [Future Work / Trabajo Futuro](#future-work--trabajo-futuro)  
8. [Setup Instructions / Instrucciones de Instalación](#setup-instructions--instrucciones-de-instalación)  

---

## **Objective / Objetivo**  
**English**: Predict Airbnb listing prices using machine learning regression models.  
**Español**: Predecir los precios de alojamientos de Airbnb utilizando modelos de regresión.

---

## **Dataset**  
- **Source**: `airbnb-listings-extract.csv`  
- **Description**:  
  - **Numerical Features**: Bedrooms, Minimum Nights, Availability, Reviews, etc.  
  - **Categorical Features**: Room Type, Neighbourhood Group.  
  - **Target Variable**: `Price`  

---

## **Workflow / Flujo de Trabajo**  

### **1. Data Cleaning / Limpieza de Datos**  
- Imputation of missing values:  
   - Numerical: Median  
   - Categorical: Mode  
- Outlier treatment using IQR (Interquartile Range).  

### **2. Exploratory Data Analysis (EDA) / Análisis Exploratorio**  
- Correlation analysis with the target variable.  
- Visualization of distributions, outliers, and feature relationships.  

### **3. Feature Engineering / Ingeniería de Características**  
- Pipelines:  
   - Numerical: Median Imputation + Standard Scaling.  
   - Categorical: Mode Imputation + OneHotEncoding.  
- Feature selection using **LassoCV** and elimination of highly correlated variables.  

### **4. Modeling / Modelado**  
- Models implemented:  
   - Linear Regression  
   - Decision Tree  
   - Random Forest (GridSearchCV for hyperparameter tuning)  
   - XGBoost (RandomizedSearchCV for optimization)  

### **5. Evaluation / Evaluación**  
- Metrics:  
   - **RMSE**: Root Mean Squared Error  
   - **R²**: Coefficient of Determination  
- Visualizations:  
   - Predicted vs Actual values (Scatterplot).  
   - Residual distribution (Histogram).  

---

## **Technologies Used / Tecnologías Utilizadas**  
- Python  
- Pandas, Numpy  
- Scikit-learn  
- XGBoost  
- Seaborn, Matplotlib  
- GridSearchCV, RandomizedSearchCV  

---

## **Models and Results / Modelos y Resultados**

| **Model / Modelo**    | **RMSE (Test)** | **R² (Test)** | **Comments / Comentarios**              |
|-----------------------|-----------------|--------------|----------------------------------------|
| Linear Regression     | 25.50           | 0.47         | Baseline model, acceptable performance.|
| Decision Tree         | 33.88           | -0.04        | Overfitting, poor generalization.      |
| Random Forest         | 23.46           | 0.50         | Best results, robust and reliable.     |
| XGBoost               | 23.51           | 0.50         | Similar performance to Random Forest.  |

---

## **Conclusion / Conclusión**  
- The **Random Forest** and **XGBoost** models provided the best performance with an RMSE of ~23.5 and an R² of 0.50.  
- Los modelos **Random Forest** y **XGBoost** ofrecieron el mejor rendimiento con un RMSE de ~23.5 y un R² de 0.50.  

---

## **Future Work / Trabajo Futuro**  
1. Introduce new features, such as distances to points of interest or additional amenities.  
2. Explore advanced algorithms like **LightGBM** or **CatBoost**.  
3. Analyze learning curves to assess the sufficiency of data.  
4. Improve feature engineering for better capturing location and review effects.  

---

## **Author / Autor
Paola León
GitHub: @paucimi

## **License / Licencia
This project is licensed under the MIT License - see the LICENSE file for details.
Este proyecto está licenciado bajo la MIT License - consulte el archivo LICENSE para más detalles.
