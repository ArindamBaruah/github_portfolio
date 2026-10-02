"""
GBM model factory.
Tries real XGBoost / LightGBM first (best choice, use when you have internet access
to `pip install xgboost lightgbm`). Falls back to sklearn's HistGradientBoosting
(a close algorithmic cousin of LightGBM) when neither is importable -- which is the
case in this sandbox (no network access to install packages).

Whichever backend is used, a small hyperparameter grid is searched using a
time-respecting split (last 20% of the training fold held out as validation --
never a random/shuffled split, to stay consistent with the purged walk-forward protocol).
"""
import numpy as np
import warnings

try:
    import xgboost as xgb
    BACKEND = 'xgboost'
except ImportError:
    try:
        import lightgbm as lgb
        BACKEND = 'lightgbm'
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        BACKEND = 'sklearn_hgb'

_WARNED = False


def backend_name():
    global _WARNED
    if BACKEND == 'sklearn_hgb' and not _WARNED:
        warnings.warn(
            "xgboost/lightgbm not installed (no internet access in this sandbox) -- "
            "falling back to sklearn HistGradientBoosting, LightGBM's closest algorithmic "
            "cousin in scikit-learn. Install xgboost/lightgbm and this module will use them "
            "automatically, no code changes needed.", RuntimeWarning)
        _WARNED = True
    return BACKEND


PARAM_GRID = [
    {'max_depth': 3, 'lr': 0.05, 'n_estimators': 150},
    {'max_depth': 4, 'lr': 0.05, 'n_estimators': 150},
    {'max_depth': 4, 'lr': 0.03, 'n_estimators': 250},
    {'max_depth': 6, 'lr': 0.05, 'n_estimators': 100},
]


def _make_model(task, params, backend):
    if backend == 'xgboost':
        common = dict(max_depth=params['max_depth'], learning_rate=params['lr'],
                      n_estimators=params['n_estimators'], subsample=0.8, colsample_bytree=0.8,
                      random_state=42, n_jobs=-1)
        return xgb.XGBClassifier(**common, eval_metric='logloss') if task == 'clf' else xgb.XGBRegressor(**common)
    elif backend == 'lightgbm':
        common = dict(max_depth=params['max_depth'], learning_rate=params['lr'],
                      n_estimators=params['n_estimators'], subsample=0.8, colsample_bytree=0.8,
                      random_state=42, n_jobs=-1, verbose=-1)
        return lgb.LGBMClassifier(**common) if task == 'clf' else lgb.LGBMRegressor(**common)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        common = dict(max_depth=params['max_depth'], learning_rate=params['lr'],
                      max_iter=params['n_estimators'], random_state=42)
        return HistGradientBoostingClassifier(**common) if task == 'clf' else HistGradientBoostingRegressor(**common)


def fit_best_gbm(X_train, y_train, task='clf'):
    """Small time-respecting hyperparameter search + final fit on the full training fold."""
    backend = backend_name()
    n = len(X_train)
    cut = int(n * 0.8)
    Xtr, Xval = X_train[:cut], X_train[cut:]
    ytr, yval = y_train[:cut], y_train[cut:]

    best_score, best_params = -np.inf if task == 'clf' else np.inf, PARAM_GRID[0]
    for params in PARAM_GRID:
        m = _make_model(task, params, backend)
        m.fit(Xtr, ytr)
        if task == 'clf':
            from sklearn.metrics import roc_auc_score
            proba = m.predict_proba(Xval)[:, 1]
            score = roc_auc_score(yval, proba) if len(np.unique(yval)) > 1 else 0.5
            if score > best_score:
                best_score, best_params = score, params
        else:
            from sklearn.metrics import mean_squared_error
            pred = m.predict(Xval)
            score = mean_squared_error(yval, pred)
            if score < best_score:
                best_score, best_params = score, params

    final_model = _make_model(task, best_params, backend)
    final_model.fit(X_train, y_train)
    return final_model, best_params, backend
