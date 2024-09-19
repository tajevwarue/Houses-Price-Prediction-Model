# House Price Prediction Model

This repository contains the code and data for predicting house prices using the `kc_house` dataset. The model is built using a stacking ensemble of five different regression models.

## Dataset

The dataset contains the following columns:

| Column Name        | Data Type   |
|--------------------|-------------|
| id                 | int64       |
| date               | object      |
| price              | float64     |
| bedrooms           | int64       |
| bathrooms          | float64     |
| sqft_living        | int64       |
| sqft_lot           | int64       |
| floors             | float64     |
| waterfront         | int64       |
| view               | int64       |
| condition          | int64       |
| grade              | int64       |
| sqft_above         | int64       |
| sqft_basement      | int64       |
| yr_built           | int64       |
| yr_renovated       | int64       |
| zipcode            | int64       |
| lat                | float64     |
| long               | float64     |
| sqft_living15      | int64       |
| sqft_lot15         | int64       |

## Data Preprocessing

- **Missing Values**: Missing values in `sqft_above` were filled using the median, as its distribution was skewed to the right.
- **Encoding**: `zipcode` was encoded using binary encoding.
- **Feature Engineering**: Distance from the center of the city was calculated using the `geopy` library.

## Model

The final model is a stacking ensemble of the following regressors:

1. **Linear Regression**
2. **Random Forest Regressor**
   - `max_depth=29`
   - `max_features=None`
   - `n_estimators=281`
3. **K-Nearest Neighbors Regressor**
   - `n_neighbors=4`
   - `p=1`
   - `weights='distance'`
4. **CatBoost Regressor**
5. **Neural Networks (MLPRegressor)**
   - `max_iter=1000`
   - `random_state=101`

The final estimator in the stacking ensemble is an **Extra Trees Regressor** with the following hyperparameters:
- `max_depth=27`
- `max_features=None`
- `min_samples_split=5`
- `n_estimators=628`

Hyperparameter tuning was performed using **Optuna**.

## Model Performance

The model's performance on the test set is as follows:

- **R² Score**: 0.8721
- **Mean Absolute Error (MAE)**: 69,022.46
- **Mean Squared Error (MSE)**: 14,764,289,913.48
- **Root Mean Squared Error (RMSE)**: 121,508.39

## Usage

To run the model, follow these steps:

1. Clone the repository.
2. Install the required dependencies.
3. Run the preprocessing script to prepare the data.
4. Train the model using the training script.
5. Evaluate the model using the evaluation script.

## Dependencies

- pandas
- numpy
- scikit-learn
- catboost
- xgboost
- pickle
- geopy
- optuna

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to Mr Sunday Kingsley and Caleb Somanya for their guidance.