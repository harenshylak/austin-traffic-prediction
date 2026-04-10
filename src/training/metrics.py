"""
metrics.py
Evaluation metrics — all computed on de-normalized predictions (inverse StandardScaler).
Reported at each horizon (H=3 / 6 / 12 → 15 / 30 / 60 min):

  MAE  = mean(|y_pred - y_true|)               # primary metric, mph
  RMSE = sqrt(mean((y_pred - y_true)^2))        # penalises large errors
  MAPE = 100 * mean(|y_pred - y_true| / y_true) # percentage error
"""

# TODO: implement
