"""STEP 2-5: EDA, feature engineering, model training, evaluation."""
import pandas as pd, numpy as np, joblib, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

df = pd.read_csv("data/food_waste.csv", parse_dates=["date"]).sort_values(["store_id", "category", "date"])
print(df.describe().T[["mean", "min", "max"]].round(2)); print("Missing:", df.isna().sum().sum())

# ---- EDA plots
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
df.groupby("category").waste_kg.mean().sort_values().plot.barh(ax=ax[0, 0], title="Avg daily waste by category")
df.groupby("dow").waste_kg.mean().plot.bar(ax=ax[0, 1], title="Avg waste by weekday (0=Mon)")
df.groupby(df.date.dt.to_period("M")).waste_kg.sum().plot(ax=ax[1, 0], title="Total monthly waste (kg)")
df.groupby("store_type").waste_kg.mean().plot.bar(ax=ax[1, 1], title="Avg waste by store type")
plt.tight_layout(); plt.savefig("reports/eda.png", dpi=110); plt.close()

# ---- Feature engineering (lags use ONLY past data -> no leakage)
g = df.groupby(["store_id", "category"]).waste_kg
df["lag_1"] = g.shift(1); df["lag_7"] = g.shift(7)
df["roll_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
df["roll_28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
df["doy"] = df.date.dt.dayofyear
df = df.dropna().reset_index(drop=True)
cat_cols = ["store_id", "store_type", "category"]
X = pd.get_dummies(df.drop(columns=["date", "waste_kg", "lat", "lon"]), columns=cat_cols, dtype=int)
y = df.waste_kg

# ---- Time-based split (train on past, test on future)
cut = pd.Timestamp("2024-09-01")
tr, te = df.date < cut, df.date >= cut
Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

def report(name, p):
    return dict(model=name, MAE=mean_absolute_error(yte, p),
                RMSE=mean_squared_error(yte, p)**.5, R2=r2_score(yte, p))
results = [report("Baseline (7-day avg)", Xte.roll_7)]
models = {"RandomForest": RandomForestRegressor(200, min_samples_leaf=5, n_jobs=-1, random_state=1),
          "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=.06, random_state=1)}
for n_, m in models.items():
    m.fit(Xtr, ytr); results.append(report(n_, m.predict(Xte)))
res = pd.DataFrame(results).round(3); print("\n", res.to_string(index=False))
res.to_csv("reports/model_comparison.csv", index=False)

best = models["HistGradientBoosting"]
joblib.dump({"model": best, "columns": list(X.columns)}, "models/waste_model.joblib")

# ---- Explainability + prediction plot
pi = permutation_importance(best, Xte.sample(5000, random_state=1), yte.loc[Xte.sample(5000, random_state=1).index],
                            n_repeats=3, random_state=1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values().tail(12)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
imp.plot.barh(ax=ax[0], title="Top drivers of food waste (permutation importance)")
ax[1].scatter(yte.sample(1500, random_state=1), best.predict(Xte.loc[yte.sample(1500, random_state=1).index]), s=6, alpha=.5)
ax[1].plot([0, yte.max()], [0, yte.max()], "r--"); ax[1].set(title="Actual vs Predicted waste (kg)", xlabel="Actual", ylabel="Predicted")
plt.tight_layout(); plt.savefig("reports/model_eval.png", dpi=110)

# save the latest feature rows so Step 3 can predict "tomorrow"
df.to_pickle("data/featured.pkl")
