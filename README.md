
# 🏠 House Price Prediction — PyTorch (Jupyter)

*Optional Flask API for deployment*

### Overview

This project trains a **PyTorch** neural network to predict house prices from tabular features. It runs entirely in **Jupyter Notebook** and includes preprocessing, training/evaluation, visualizations, feature importance, and model saving. A minimal **Flask** API is provided as an optional deployment step.

### Key Features

* **Data pipeline**: scaling with `StandardScaler`, train/test split
* **Model**: 2 hidden layers (ReLU, Dropout), **Adam** optimizer
* **Metrics & plots**: **MAE, RMSE, R²**, loss curves, actual vs. predicted scatter
* **Interpretability**: feature importance from first-layer weights
* **Reusability**: saves `house_price_model.pth`, `scaler_X.pkl`, `scaler_y.pkl`
* **Deployment (optional)**: `house_price_api.py` with `/predict` endpoint

### Tech Stack

**PyTorch**, **Scikit-learn**, **Pandas**, **NumPy**, **Matplotlib**, **Joblib**
*(Optional)* **Flask** for serving predictions

### Notebook Flow

1. **Data acquisition**

   * Tries `sklearn.datasets.load_boston()`; if unavailable, falls back to a **synthetic** dataset with similar schema.
2. **Preprocessing**

   * Feature/target scaling (`StandardScaler`), train/test split
3. **Model training**

   * MLP (ReLU + Dropout), **Adam**, MSE loss; tracks train/test loss
4. **Evaluation**

   * Reports **MAE/RMSE/R²**; plots predictions vs actuals
5. **Artifacts & inference**

   * Saves model/scalers; `predict_house_price(features)` helper
6. **(Optional)** Flask API

   * Loads saved artifacts and exposes `/predict` (JSON)

### Getting Started

```bash
# 1) Create env (optional)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt
```

**`requirements.txt`**

```txt
torch
pandas
numpy
scikit-learn
matplotlib
joblib
flask  # optional (only if you use the API)
```

### Run (Jupyter workflow)

1. Open the notebook and run all cells.
2. After training, artifacts are saved:

   * `house_price_model.pth`
   * `scaler_X.pkl`, `scaler_y.pkl`
3. Use `predict_house_price([...])` with a 13-feature list to get a price prediction.

### Optional: Run the Flask API

```bash
python house_price_api.py
# POST JSON: {"features": [13 feature values in the same order as training]}
# Example:
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\": [0.02731,0.0,7.07,0,0.469,6.421,78.9,4.9671,2,242,17.8,396.90,9.14]}"
```

### Notes

* If `load_boston()` is unavailable (deprecated in some versions), the notebook **automatically switches to a synthetic dataset** with similar feature names.
* Replace metrics in your README with your actual results after training.

### Project Structure

```
.
├─ notebook.ipynb
├─ house_price_api.py              # optional
├─ house_price_model.pth           # saved after training
├─ scaler_X.pkl / scaler_y.pkl     # saved after training
├─ requirements.txt
└─ README.md
```

### Future Work

* Hyperparameter tuning (hidden sizes, dropout, learning rate)
* Better interpretability (permutation importance, SHAP)
* Real dataset integration + validation scheme (K-Fold)
* Containerize API with Docker (optional)

