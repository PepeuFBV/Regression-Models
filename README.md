# Predicting the Municipal Human Development Index (MHDI) Using Machine Learning Models: A Comparative Analysis

This repository contains the implementation of a comparative analysis of machine learning models for predicting the Municipal Human Development Index (MHDI, represented by IDHM in this dataset) in Brazil. The main focus is on evaluating and comparing the performance of different regression models, such as Linear Regression, KNN, Decision Trees, Random Forest and Extra Trees, in predicting the MHDI of Brazilian municipalities. The dataset used in this study is the Atlas of Human Development in Brazil, which contains information about the MHDI and other socio-economic indicators of Brazilian municipalities. Finding out the best approach for predicting the MHDI can help governments and policymakers to make better decisions and allocate resources more efficiently was the main motivation behind this study.

## Key Features

- **Dataset**: The dataset used contains socioeconomic variables in Brazilian municipalities, there are 5573 entries and 81 columns (80 features and 1 target variable).

- **Data Preprocessing**: Techniques such as normalization, outlier removal, and dimensionality reduction (PCA) are applied to improve the model performance.

- **Model Evaluation**: The performance of 5 different regression models is evaluated using the R2 score as the evaluation metric, with Extra Trees Regressor achieving the best performance.

## Project Structure

- `data/`: Contains the dataset used in this study for model training and testing.

- `images/`: Contains images of important plots generated during the analysis.

- `main.py`: Main script that loads the dataset, preprocesses the data and generates the preprocessed dataset.

- `model_`: Any model that is used will start with this prefix.

## Models

The following models are used in this study:

- Linear Regression

  - Simple and interpretable model that assumes a linear relationship between the input features and the target variable.
  - Best R2 score: 0.7468

- KNN

  - Distance-based algorithm that predicts the target variable based on the similarity of the input features to the training data.
  - Best R2 score: 0.3919

- Decision Trees

  - Non-linear model that predicts the target variable by recursively splitting the data based on the input features.
  - Best R2 score: 0.7874

- Random Forest

  - Ensemble model that combines multiple decision trees to improve the prediction performance.
  - Best R2 score: 0.5430

- Extra Trees
  - Similar to Random Forest, but with a different splitting strategy that can lead to better performance.
  - Best R2 score: 0.8816

## Getting Started

Since the code is run on `ipynb` files, you can run the code on your local machine by following these steps:

1. Clone the repository:

```bash
git clone https://github.com/PepeuFBV/Regression-Models.git
cd Regression-Models
```

2. Install the required libraries:

In the beggining of each notebook

3. Run the `main.ipynb` notebook to generate the preprocessed dataset:

```bash
jupyter notebook main.ipynb
```

4. Choose any of the models and run the respective notebook to train and evaluate the model.

## Results

The best performing model in our analysis was the Extra Trees Regressor, with an R2 score of 0.8816. This model outperformed the other models in terms of prediction accuracy and generalization to unseen data. The results of the model evaluation are summarized below:

| Model             | R2 Score |
| ----------------- | -------- |
| Linear Regression | 0.7468   |
| KNN               | 0.3919   |
| Decision Trees    | 0.7874   |
| Random Forest     | 0.5430   |
| Extra Trees       | 0.8816   |

## Conclusion

Our findings indicate that the Extra Trees Regressor is the best model due to its capability to handle non-linear relationships and high-dimensional data.

## Future Work

- Hyperparameter tuning: Increase the range and variety of hyperparameters to find the best combination for each model.

- Feature engineering: Create new features that can improve the prediction performance of the models.

- Experiment with alternative data preprocessing techniques: Test different normalization methods, outlier removal strategies, and dimensionality reduction techniques to improve the model performance.

- Better understanding of the data: Explore the relationships between the input features and the target variable to gain insights that can help improve the model performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
