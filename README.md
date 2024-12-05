# Dynamic Pricing Model for E-Commerce Websites

## Overview

The **Dynamic Pricing Model for E-Commerce Websites** is a machine learning-based solution that predicts the optimal sale price of products based on key features such as regular price, stock levels, and product categories. By leveraging historical pricing and sales data, this model helps e-commerce businesses optimize pricing strategies in real-time, ensuring competitive pricing while maximizing profitability.

The model is built using **XGBoost** for regression, providing high accuracy and efficiency in predicting pricing. It can be easily extended to incorporate additional features such as demand fluctuations, competitor pricing, and seasonal trends.

## Features

- Predicts the optimal sale price for e-commerce products.
- Uses key product features such as regular price, stock quantity, and category.
- Built with **XGBoost**, an efficient and scalable machine learning algorithm.
- Achieves high predictive accuracy (R² > 0.99) based on historical data.

## Dataset

The dataset used for this model is the **Divi Engine WooCommerce Sample Products Dataset**, which can be downloaded from [Divi Engine](https://diviengine.com/woocommerce-sample-products-csv-import-file-freebie/). This dataset contains information about product features, pricing, and stock levels that are essential for training the dynamic pricing model.

## Getting Started

To run the project locally, follow the instructions below to set up the environment and use the system.

### Prerequisites

- **Python**: 3.7+
- **Required Libraries**:
  - `pandas`
  - `numpy`
  - `xgboost`
  - `scikit-learn`
  - `joblib`

These dependencies can be installed via **pip** or **conda**.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmdmohamedd/ecommerce-dynamic-pricing.git
   cd ecommerce-dynamic-pricing
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset or use your own dataset. Ensure it contains relevant columns such as:
   - `Regular price`
   - `Sale price`
   - `Stock`
   - `Categories`
   
   The provided dataset can be loaded as shown in the `dynamic_pricing_model.ipynb` notebook.

### Running the Model

1. Open the `dynamic_pricing_model.ipynb` notebook in Jupyter Notebook or JupyterLab.

2. Execute the notebook cells in sequence:
   - **Data Preprocessing**: Cleans and prepares the dataset.
   - **Model Training**: Trains the dynamic pricing model using the `XGBoost` regressor.
   - **Prediction**: Uses the trained model to predict the dynamic price for new products.

### Example Prediction

You can modify the following code in the notebook to predict the sale price for a new product:

```python
new_product = pd.DataFrame({
    'Regular price': [25],  # Example regular price
    'Categories': [2],      # Example category code (ensure this matches your data)
    'Stock': [100]          # Example stock quantity
})

# Predict the sale price for the new product
predicted_price = model.predict(new_product)
print(f"Predicted Dynamic Price: {predicted_price[0]}")
```

### Model Evaluation

The model performance can be evaluated using metrics such as **Mean Squared Error (MSE)** and **R-squared (R²)**. These metrics provide insights into the accuracy of price predictions and the model’s ability to generalize to new data.

```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
```

## Model Deployment

Once the model is trained and evaluated, it can be saved and deployed using the following:

```python
import joblib

# Save the trained model
joblib.dump(model, 'dynamic_pricing_model.pkl')
```

The saved model can be loaded and used for predictions in production environments.

## Contributing

Contributions to this project are welcome! If you have any ideas for improving the model, feature suggestions, or bug fixes, feel free to open an issue or submit a pull request.

### Steps to contribute:

1. Fork this repository.
2. Create a new branch for your feature (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new pull request.

---

### Additional Notes

- **Scalability**: The model can be further scaled by integrating additional data sources, such as customer demand, competitor pricing, and product seasonality.
- **Customization**: You can customize the feature set and model parameters to suit specific business needs or datasets.

Feel free to reach out if you have any questions or need assistance with the implementation.

---
