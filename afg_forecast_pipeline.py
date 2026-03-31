"""
AI Finance Guard — Forecast + Backtest + Service Simulation Pipeline
=====================================================================
Wejscie : SKU_FeatureOnly *.csv  +  SKU_FeatureTrain *.csv
Wyjscie : Forecast_Demand.csv, AFG_ML_EVAL_BACKTEST.csv, Plan_Production.csv

Modele:
  - lightgbm : LightGBM quantile regression (zaawansowany)
  - arima    : ARIMA + residual quantile (baseline sezonowy)

Uzycie:
  python afg_forecast_pipeline.py                        # domyslnie lightgbm
  python afg_forecast_pipeline.py --model arima
  python afg_forecast_pipeline.py --model lightgbm,arima # porownanie
"""

import os, sys, warnings, pathlib, argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# 0.  KONFIGURACJA
# ──────────────────────────────────────────────
HORIZONS = list(range(1, 7))
QUANTILES = {"P50": 0.5, "P90": 0.9}
TARGETS = ["KPI_OrdersIn_Qty", "KPI_SalesQty"]
AVAILABLE_MODELS = ["lightgbm", "arima"]

ID_COLS = [
    "ProductID", "RokMiesiac", "EOM", "Year", "Month", "Quarter",
    "HierarchyLevel", "ProductLine", "Family",
]
TRAIN_ONLY_COLS = [
    "KPI_LostSales_Qty_LT", "KPI_FillRate_Qty_LT", "KPI_LostSales_Value_LT",
]


# ══════════════════════════════════════════════
# MODEL: LightGBM
# ══════════════════════════════════════════════
class LightGBMModel:
    name = "lightgbm"

    def __init__(self):
        import lightgbm as lgb
        self._lgb = lgb
        self._model = None

    def fit(self, X, y, quantile=0.5):
        params = dict(
            n_estimators=150, learning_rate=0.08, max_depth=5,
            num_leaves=24, min_child_samples=5, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            verbose=-1, n_jobs=-1, random_state=42,
            objective="quantile", alpha=quantile,
        )
        self._model = self._lgb.LGBMRegressor(**params)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return np.maximum(self._model.predict(X), 0)


# ══════════════════════════════════════════════
# MODEL: ARIMA (per-SKU univariate baseline)
# ══════════════════════════════════════════════
class ARIMAModel:
    """
    ARIMA per SKU — szybki baseline sezonowy.
    Kwantyle przez residual bootstrap (std * z_score).
    """
    name = "arima"

    def __init__(self):
        from statsmodels.tsa.arima.model import ARIMA
        self._cls = ARIMA
        self._fits = {}       # pid -> fitted model
        self._resid_std = {}  # pid -> residual std
        self._fallback = {}   # pid -> last known value
        self._quantile = 0.5

    def fit(self, X, y, quantile=0.5):
        self._quantile = quantile
        if "_arima_pid" not in X.columns:
            return self

        df = pd.DataFrame({
            "pid": X["_arima_pid"].values,
            "eom": X["_arima_eom"].values,
            "y": y.values,
        })

        for pid, g in df.groupby("pid"):
            g = g.sort_values("eom")
            vals = g["y"].values.astype(float)
            self._fallback[pid] = vals[-1] if len(vals) > 0 else 0

            if len(vals) < 6:
                self._fits[pid] = None
                self._resid_std[pid] = max(np.std(vals), 1) if len(vals) > 1 else 1
                continue

            try:
                m = self._cls(vals, order=(1, 1, 1))
                best_fit = m.fit(method_kwargs={"warn_convergence": False})
            except Exception:
                try:
                    m = self._cls(vals, order=(1, 0, 1))
                    best_fit = m.fit(method_kwargs={"warn_convergence": False})
                except Exception:
                    best_fit = None

            self._fits[pid] = best_fit
            if best_fit is not None:
                resid = best_fit.resid
                self._resid_std[pid] = max(np.nanstd(resid), 1)
            else:
                self._resid_std[pid] = max(np.std(vals), 1)

        return self

    def predict(self, X):
        if "_arima_pid" not in X.columns or "_arima_h" not in X.columns:
            return np.zeros(len(X))

        from scipy.stats import norm
        z = norm.ppf(self._quantile) if self._quantile != 0.5 else 0.0

        preds = np.zeros(len(X))
        for i in range(len(X)):
            pid = X.iloc[i]["_arima_pid"]
            h = int(X.iloc[i]["_arima_h"])
            fit = self._fits.get(pid)

            if fit is not None:
                try:
                    fc = fit.forecast(steps=h)
                    point = fc.iloc[-1] if hasattr(fc, 'iloc') else fc[-1]
                    std = self._resid_std.get(pid, 1) * np.sqrt(h)
                    preds[i] = max(0, point + z * std)
                except Exception:
                    preds[i] = max(0, self._fallback.get(pid, 0) + z * self._resid_std.get(pid, 1))
            else:
                preds[i] = max(0, self._fallback.get(pid, 0) + z * self._resid_std.get(pid, 1))

        return preds


MODEL_REGISTRY = {
    "lightgbm": LightGBMModel,
    "arima": ARIMAModel,
}


# ──────────────────────────────────────────────
# 1.  WCZYTANIE DANYCH
# ──────────────────────────────────────────────
INPUT_DIR = pathlib.Path(".") / "dane_input"
OUTPUT_DIR = pathlib.Path(".") / "dane_output"


def find_csv(pattern):
    matches = sorted(INPUT_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Nie znaleziono pliku: {INPUT_DIR / pattern}")
    return str(matches[-1])


def load_data():
    fo_path = find_csv("SKU_FeatureOnly*.csv")
    tr_path = find_csv("SKU_FeatureTrain*.csv")
    print(f"[1] Wczytywanie danych...")
    print(f"    FeatureOnly : {fo_path}")
    print(f"    Train       : {tr_path}")

    fo = pd.read_csv(fo_path, encoding="utf-8-sig", decimal=",")
    tr = pd.read_csv(tr_path, encoding="utf-8-sig", decimal=",")

    fo["ProductID"] = fo["ProductID"].astype(str)
    tr["ProductID"] = tr["ProductID"].astype(str)
    fo["EOM"] = pd.to_datetime(fo["EOM"])
    tr["EOM"] = pd.to_datetime(tr["EOM"])

    print(f"    FeatureOnly  : {fo.shape[0]} wierszy, {fo.shape[1]} kolumn")
    print(f"    Train        : {tr.shape[0]} wierszy, {tr.shape[1]} kolumn")
    return fo, tr


# ──────────────────────────────────────────────
# 2.  WALIDACJA
# ──────────────────────────────────────────────
def validate(fo, tr):
    print("[2] Walidacja danych...")
    assert fo.duplicated(subset=["ProductID", "EOM"]).sum() == 0, "Duplikaty w FeatureOnly!"
    assert tr.duplicated(subset=["ProductID", "EOM"]).sum() == 0, "Duplikaty w Train!"
    assert fo.shape[0] == tr.shape[0], f"Rozna liczba wierszy!"
    assert fo["ProductID"].notna().all() and fo["EOM"].notna().all()
    for t in TARGETS:
        assert t in fo.columns, f"Brak targetu {t}!"
    for c in TRAIN_ONLY_COLS:
        assert c in tr.columns, f"Brak {c} w Train!"
    print("    OK.")


# ──────────────────────────────────────────────
# 3.  ETYKIETY
# ──────────────────────────────────────────────
def prepare_labels(df):
    print("[3] Etykiety (shift t+h)...")
    df = df.sort_values(["ProductID", "EOM"]).reset_index(drop=True)
    for target in TARGETS:
        short = "orders" if "OrdersIn" in target else "sales"
        for h in HORIZONS:
            df[f"y_{short}_h{h}"] = df.groupby("ProductID")[target].shift(-h)
    n_valid = df[[c for c in df.columns if c.startswith("y_")]].notna().all(axis=1).sum()
    print(f"    12 kolumn, {n_valid}/{len(df)} kompletnych.")
    return df


# ──────────────────────────────────────────────
# 4.  FEATURES
# ──────────────────────────────────────────────
def get_feature_cols(df):
    exclude = set(ID_COLS + TARGETS + TRAIN_ONLY_COLS)
    exclude |= {c for c in df.columns if c.startswith("y_") or c.startswith("_arima_")}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "int64")]


# ──────────────────────────────────────────────
# 5.  TRENING
# ──────────────────────────────────────────────
def _prepare_X(df, feature_cols, model_name, h=None):
    X = df[feature_cols].copy()
    if model_name == "arima":
        X["_arima_pid"] = df["ProductID"].values
        X["_arima_eom"] = df["EOM"].values
        if h is not None:
            X["_arima_h"] = h
    return X


def train_models(df_train, feature_cols, model_name, verbose=True):
    if verbose:
        print(f"[5] Trening: {model_name.upper()}...")
    models = {}
    i = 0
    for target in TARGETS:
        short = "orders" if "OrdersIn" in target else "sales"
        for h in HORIZONS:
            y_col = f"y_{short}_h{h}"
            mask = df_train[y_col].notna()
            X = _prepare_X(df_train[mask], feature_cols, model_name, h)
            y = df_train.loc[mask, y_col]
            if len(y) < 6:
                continue
            for q_name, alpha in QUANTILES.items():
                m = MODEL_REGISTRY[model_name]()
                m.fit(X, y, quantile=alpha)
                models[(short, h, q_name)] = m
                i += 1
    if verbose:
        print(f"    Wytrenowano {i} modeli.")
    return models


# ──────────────────────────────────────────────
# 6.  ROLLING BACKTEST
# ──────────────────────────────────────────────
def rolling_backtest(df, feature_cols, model_name, min_train_months=24,
                     cutoff_step=3):
    print(f"[6] Rolling backtest ({model_name}, step={cutoff_step})...")
    all_eoms = sorted(df["EOM"].unique())

    if len(all_eoms) < min_train_months + max(HORIZONS):
        min_train_months = max(12, len(all_eoms) - max(HORIZONS))

    candidates = all_eoms[min_train_months: -max(HORIZONS)]
    if not candidates:
        candidates = all_eoms[min_train_months - 6: -max(HORIZONS)]
    if cutoff_step > 1:
        candidates = candidates[::cutoff_step]

    print(f"    {len(candidates)} cutoffow "
          f"({pd.Timestamp(candidates[0]).strftime('%Y-%m')} - "
          f"{pd.Timestamp(candidates[-1]).strftime('%Y-%m')})")

    results = []
    for ci, cutoff in enumerate(candidates):
        df_tr = df[df["EOM"] <= cutoff]
        models = train_models(df_tr, feature_cols, model_name, verbose=False)

        for target in TARGETS:
            short = "orders" if "OrdersIn" in target else "sales"
            for h in HORIZONS:
                y_col = f"y_{short}_h{h}"
                pred_mask = (df["EOM"] == cutoff) & df[y_col].notna()
                if pred_mask.sum() == 0:
                    continue
                X_pred = _prepare_X(df[pred_mask], feature_cols, model_name, h)
                actual = df.loc[pred_mask, y_col].values
                pids = df.loc[pred_mask, "ProductID"].values
                forecast_eom = cutoff + pd.DateOffset(months=h)

                for q_name in QUANTILES:
                    key = (short, h, q_name)
                    if key not in models:
                        continue
                    preds = models[key].predict(X_pred)
                    for j in range(len(actual)):
                        results.append({
                            "Run_Cutoff_EOM": cutoff,
                            "Forecast_EOM": forecast_eom,
                            "Horizon_M": h, "ProductID": pids[j],
                            "Target": short, "Quantile": q_name,
                            "Actual": actual[j], "Predicted": preds[j],
                        })

        if (ci + 1) % 3 == 0 or ci == len(candidates) - 1:
            print(f"    Cutoff {ci+1}/{len(candidates)}")

    bt = pd.DataFrame(results)
    print(f"    {len(bt)} rekordow.")
    return bt


# ──────────────────────────────────────────────
# 7.  METRYKI
# ──────────────────────────────────────────────
def compute_metrics(bt, model_name=None):
    print("[7] Metryki...")
    records = []
    for (target, horizon, q_name), g in bt.groupby(["Target", "Horizon_M", "Quantile"]):
        actual, pred = g["Actual"].values, g["Predicted"].values
        mae = np.mean(np.abs(actual - pred))
        wape = np.sum(np.abs(actual - pred)) / max(np.sum(np.abs(actual)), 1)
        bias = np.mean(pred - actual)
        tau = 0.5 if q_name == "P50" else 0.9
        errors = actual - pred
        pinball = np.mean(np.where(errors >= 0, tau * errors, (tau - 1) * errors))
        coverage = np.mean(actual <= pred)
        rec = {"Target": target, "Horizon_M": horizon, "Quantile": q_name,
               "MAE": round(mae, 2), "WAPE": round(wape, 4),
               "Bias": round(bias, 2), "Pinball": round(pinball, 4),
               "Coverage": round(coverage, 4), "N": len(g)}
        if model_name:
            rec["Model"] = model_name
        records.append(rec)
    metrics = pd.DataFrame(records)
    print(metrics.to_string(index=False))
    return metrics


# ──────────────────────────────────────────────
# 8.  PROGNOZA FINALNA
# ──────────────────────────────────────────────
def final_forecast(df, feature_cols, model_name):
    print(f"[8] Prognoza finalna ({model_name})...")
    models = train_models(df, feature_cols, model_name)
    last_eom = df["EOM"].max()
    last_rows = df[df["EOM"] == last_eom].copy()
    print(f"    EOM: {last_eom.strftime('%Y-%m-%d')}, SKU: {len(last_rows)}")

    rows = []
    for target in TARGETS:
        short = "orders" if "OrdersIn" in target else "sales"
        for h in HORIZONS:
            forecast_eom = last_eom + pd.DateOffset(months=h)
            for q_name in QUANTILES:
                key = (short, h, q_name)
                if key not in models:
                    continue
                X = _prepare_X(last_rows, feature_cols, model_name, h)
                preds = models[key].predict(X)
                for i, (_, row) in enumerate(last_rows.iterrows()):
                    rows.append({
                        "ProductID": row["ProductID"],
                        "Forecast_EOM": forecast_eom,
                        "Horizon_M": h, "Target": short,
                        "Quantile": q_name,
                        "Predicted": round(preds[i], 2),
                        "Run_Cutoff_EOM": last_eom,
                    })
    fc = pd.DataFrame(rows)
    print(f"    {len(fc)} prognoz.")
    return fc


# ──────────────────────────────────────────────
# 9.  PIVOT
# ──────────────────────────────────────────────
def build_forecast_demand(fc):
    print("[9] Forecast_Demand...")
    rows = []
    for (pid, feom, hm, cutoff), g in fc.groupby(
            ["ProductID", "Forecast_EOM", "Horizon_M", "Run_Cutoff_EOM"]):
        row = {"ProductID": pid, "Forecast_EOM": feom,
               "Horizon_M": hm, "Run_Cutoff_EOM": cutoff}
        for _, r in g.iterrows():
            row[f"Pred_{r['Target'].capitalize()}_{r['Quantile']}"] = r["Predicted"]
        rows.append(row)
    out = pd.DataFrame(rows)
    desired = ["ProductID", "Forecast_EOM", "Horizon_M",
               "Pred_Orders_P50", "Pred_Orders_P90",
               "Pred_Sales_P50", "Pred_Sales_P90", "Run_Cutoff_EOM"]
    existing = [c for c in desired if c in out.columns]
    rest = [c for c in out.columns if c not in desired]
    out = out[existing + rest]
    print(f"    {len(out)} wierszy.")
    return out


# ──────────────────────────────────────────────
# 10. SYMULACJA SERWISU
# ──────────────────────────────────────────────
def service_simulation(fc_demand, df):
    print("[10] Symulacja serwisu...")
    last_eom = df["EOM"].max()
    last = df[df["EOM"] == last_eom][
        ["ProductID", "OPS_FG_Qty_EOM", "LT_LeadTimeMonths_Used", "LT_PW_Qty"]
    ].copy()
    last.columns = ["ProductID", "FG_Start", "LT_raw", "AvgProd"]
    last["FG_Start"] = last["FG_Start"].fillna(0)
    last["LT"] = last["LT_raw"].fillna(2).astype(int)

    baseline = df.groupby("ProductID")["LT_PW_Qty"].mean().reset_index()
    baseline.columns = ["ProductID", "ProdBaseline"]
    last = last.merge(baseline, on="ProductID", how="left")
    last["ProdBaseline"] = last["ProdBaseline"].fillna(0)

    cols = ["ProductID", "Horizon_M", "Pred_Orders_P90"]
    if "Forecast_EOM" in fc_demand.columns:
        cols.append("Forecast_EOM")
    dem = fc_demand[fc_demand["Pred_Orders_P90"].notna()][cols].copy()
    dem = dem.rename(columns={"Pred_Orders_P90": "Demand"})

    plan_rows = []
    for pid in last["ProductID"].unique():
        info = last[last["ProductID"] == pid].iloc[0]
        fg, lt, bl = float(info["FG_Start"]), int(info["LT"]), float(info["ProdBaseline"])
        sku_dem = dem[dem["ProductID"] == pid].sort_values("Horizon_M")
        total_lost = total_demand = 0.0

        for _, d in sku_dem.iterrows():
            h, demand_m = int(d["Horizon_M"]), float(d["Demand"])
            rec = max(0, demand_m - fg * 0.3) if h > lt else demand_m * 0.8
            arrival = rec if h > lt else 0
            avail = fg + arrival
            served = min(demand_m, avail)
            lost = max(0, demand_m - avail)
            total_lost += lost
            total_demand += demand_m
            fg = max(0, avail - served)

            ratio = rec / bl if bl > 0 else (1 if rec > 0 else 0)
            action = "STOP" if ratio < 0.2 else ("SLOW" if ratio < 0.8 else "SPEED")

            plan_rows.append({
                "ProductID": pid,
                "Forecast_EOM": d.get("Forecast_EOM", last_eom + pd.DateOffset(months=h)),
                "Horizon_M": h, "Demand_P90": round(demand_m, 2),
                "RecommendedProductionQty": round(rec, 0), "ActionLabel": action,
                "LT_LeadTimeMonths_Used": lt,
                "Availability": round(avail, 2), "Served": round(served, 2),
                "LostSales_M": round(lost, 2),
            })

        fr6m = 1 - total_lost / max(total_demand, 1)
        for r in plan_rows:
            if r["ProductID"] == pid:
                r["LostSales_6M"] = round(total_lost, 2)
                r["FillRate_6M"] = round(fr6m, 4)

    plan = pd.DataFrame(plan_rows)
    summary = plan.groupby("ProductID").agg(
        LostSales_6M=("LostSales_6M", "first"), FillRate_6M=("FillRate_6M", "first"),
    ).reset_index()
    print(f"    {len(plan)} wierszy.")
    print(summary.to_string(index=False))
    print(f"    Sredni FillRate: {summary['FillRate_6M'].mean():.4f}")
    return plan


# ──────────────────────────────────────────────
# 11. BACKTEST CSV
# ──────────────────────────────────────────────
def build_backtest_csv(bt):
    print("[11] Backtest CSV...")
    w = bt.pivot_table(
        index=["Run_Cutoff_EOM", "Forecast_EOM", "Horizon_M", "ProductID", "Target"],
        columns="Quantile", values=["Predicted", "Actual"], aggfunc="first",
    ).reset_index()
    cols = []
    for c in w.columns:
        if isinstance(c, tuple) and c[1]:
            cols.append(f"{c[0]}_{c[1]}")
        elif isinstance(c, tuple):
            cols.append(c[0])
        else:
            cols.append(c)
    w.columns = cols
    if "Actual_P50" in w.columns:
        w["Actual"] = w["Actual_P50"]
        w = w.drop(columns=["Actual_P50", "Actual_P90"], errors="ignore")
    if "Predicted_P50" in w.columns:
        w["AE"] = np.abs(w["Actual"] - w["Predicted_P50"])
    if "Predicted_P90" in w.columns:
        w["Covered_P90"] = (w["Actual"] <= w["Predicted_P90"]).astype(int)
    print(f"    {len(w)} wierszy.")
    return w


# ──────────────────────────────────────────────
# 12. EKSPORT
# ──────────────────────────────────────────────
def export_results(fc_demand, bt_csv, plan, metrics, model_name):
    OUTPUT_DIR.mkdir(exist_ok=True)
    s = f"_{model_name}"
    print(f"[12] Eksport ({model_name}) -> dane_output/...")
    fc_demand.to_csv(OUTPUT_DIR / f"Forecast_Demand{s}.csv", index=False, encoding="utf-8-sig")
    bt_csv.to_csv(OUTPUT_DIR / f"AFG_ML_EVAL_BACKTEST{s}.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(OUTPUT_DIR / f"Plan_Production{s}.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / f"AFG_ML_METRICS{s}.csv", index=False, encoding="utf-8-sig")
    # Kompatybilnosc — pliki bez sufiksu
    fc_demand.to_csv(OUTPUT_DIR / "Forecast_Demand.csv", index=False, encoding="utf-8-sig")
    bt_csv.to_csv(OUTPUT_DIR / "AFG_ML_EVAL_BACKTEST.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(OUTPUT_DIR / "Plan_Production.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / "AFG_ML_METRICS.csv", index=False, encoding="utf-8-sig")
    print(f"    4 pliki zapisane.")


# ──────────────────────────────────────────────
# 13. POROWNANIE
# ──────────────────────────────────────────────
def compare_models(all_metrics):
    OUTPUT_DIR.mkdir(exist_ok=True)
    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "AFG_ML_MODEL_COMPARISON.csv", index=False, encoding="utf-8-sig")
    print(f"\n[13] Porownanie -> dane_output/AFG_ML_MODEL_COMPARISON.csv")
    s = combined[combined["Quantile"] == "P50"].groupby("Model")["WAPE"].mean()
    print("    Sredni WAPE P50:")
    for m, w in s.items():
        print(f"      {m:12s}: {w:.2%}")
    print(f"    Najlepszy: {s.idxmin()} ({s.min():.2%})")
    return combined


# ──────────────────────────────────────────────
# RUN SINGLE MODEL
# ──────────────────────────────────────────────
def run_single(model_name, df, feature_cols):
    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name.upper()}")
    print(f"{'='*60}")

    bt = rolling_backtest(df, feature_cols, model_name)
    metrics = compute_metrics(bt, model_name)
    fc = final_forecast(df, feature_cols, model_name)
    fc_demand = build_forecast_demand(fc)

    if "Forecast_EOM" not in fc_demand.columns:
        eom_map = fc[["ProductID", "Horizon_M", "Forecast_EOM"]].drop_duplicates()
        fc_demand = fc_demand.merge(eom_map, on=["ProductID", "Horizon_M"], how="left")

    plan = service_simulation(fc_demand, df)
    bt_csv = build_backtest_csv(bt)
    export_results(fc_demand, bt_csv, plan, metrics, model_name)
    return metrics


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI Finance Guard ML Pipeline")
    parser.add_argument("--model", type=str, default="lightgbm",
                        help="Model(e): lightgbm, arima, lub lightgbm,arima")
    args = parser.parse_args()

    model_names = [m.strip().lower() for m in args.model.split(",")]
    for m in model_names:
        if m not in AVAILABLE_MODELS:
            print(f"BLAD: '{m}' nie jest dostepny. Uzyj: {AVAILABLE_MODELS}")
            sys.exit(1)

    print("=" * 60)
    print("  AI Finance Guard — ML Forecast Pipeline")
    print(f"  Modele: {', '.join(model_names)}")
    print("=" * 60)

    fo, tr = load_data()
    validate(fo, tr)
    df = prepare_labels(fo)
    feature_cols = get_feature_cols(df)
    print(f"[4] Features: {len(feature_cols)} kolumn.")

    all_metrics = []
    for model_name in model_names:
        m = run_single(model_name, df, feature_cols)
        all_metrics.append(m)

    if len(model_names) > 1:
        compare_models(all_metrics)

    print("\n" + "=" * 60)
    print("  PIPELINE ZAKONCZONY POMYSLNIE")
    print("=" * 60)


if __name__ == "__main__":
    main()
