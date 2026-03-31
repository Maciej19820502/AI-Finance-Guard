"""
AI Finance Guard — Testy walidacyjne pipeline'u
=================================================
Uruchom po afg_forecast_pipeline.py.
Weryfikuje: schematy, sanity checks, porównanie z targetami Train,
stabilność backtestową, ręczne przeliczenie serwisu.
"""

import sys, os, pathlib, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Konfiguracja
# ──────────────────────────────────────────────
HERE = pathlib.Path(".")
INPUT_DIR = HERE / "dane_input"
OUTPUT_DIR = HERE / "dane_output"
PASS = 0
FAIL = 0
WARN = 0


def ok(name, detail=""):
    global PASS
    PASS += 1
    print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))


def fail(name, detail=""):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def warn(name, detail=""):
    global WARN
    WARN += 1
    print(f"  [WARN] {name}" + (f"  ({detail})" if detail else ""))


def check(condition, name, detail=""):
    if condition:
        ok(name, detail)
    else:
        fail(name, detail)


# ──────────────────────────────────────────────
# Wczytanie plików
# ──────────────────────────────────────────────
print("=" * 65)
print("  AI Finance Guard — Test Suite")
print("=" * 65)
print()

# Sprawdzenie istnienia plików wyjściowych
REQUIRED_FILES = [
    "Forecast_Demand.csv",
    "AFG_ML_EVAL_BACKTEST.csv",
    "Plan_Production.csv",
    "AFG_ML_METRICS.csv",
]
print("[T0] Sprawdzenie istnienia plików wyjściowych")
all_exist = True
for f in REQUIRED_FILES:
    exists = (OUTPUT_DIR / f).exists()
    check(exists, f"Plik dane_output/{f} istnieje")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n  BRAK PLIKÓW WYJŚCIOWYCH — uruchom najpierw afg_forecast_pipeline.py")
    sys.exit(1)

fc = pd.read_csv(OUTPUT_DIR / "Forecast_Demand.csv")
bt = pd.read_csv(OUTPUT_DIR / "AFG_ML_EVAL_BACKTEST.csv")
pp = pd.read_csv(OUTPUT_DIR / "Plan_Production.csv")
mt = pd.read_csv(OUTPUT_DIR / "AFG_ML_METRICS.csv")
tr = pd.read_csv(
    sorted(INPUT_DIR.glob("SKU_FeatureTrain*.csv"))[-1],
    encoding="utf-8-sig", decimal=","
)
fo = pd.read_csv(
    sorted(INPUT_DIR.glob("SKU_FeatureOnly*.csv"))[-1],
    encoding="utf-8-sig", decimal=","
)

print()

# ──────────────────────────────────────────────
# T1. SCHEMATY WYJŚCIOWE (vs Runbook)
# ──────────────────────────────────────────────
print("[T1] Walidacja schematów wyjściowych (vs Runbook)")

# Forecast_Demand.csv
fc_required = [
    "ProductID", "Forecast_EOM", "Horizon_M",
    "Pred_Orders_P50", "Pred_Orders_P90",
    "Pred_Sales_P50", "Pred_Sales_P90",
    "Run_Cutoff_EOM",
]
for col in fc_required:
    check(col in fc.columns, f"Forecast_Demand zawiera '{col}'")

# Plan_Production.csv
pp_required = [
    "ProductID", "Forecast_EOM", "RecommendedProductionQty",
    "ActionLabel", "LT_LeadTimeMonths_Used",
    "LostSales_6M", "FillRate_6M",
]
for col in pp_required:
    check(col in pp.columns, f"Plan_Production zawiera '{col}'")

# AFG_ML_EVAL_BACKTEST.csv
bt_required = [
    "Run_Cutoff_EOM", "Forecast_EOM", "Horizon_M", "ProductID", "Target",
]
for col in bt_required:
    check(col in bt.columns, f"AFG_ML_EVAL_BACKTEST zawiera '{col}'")

# Metryki
mt_required = ["Target", "Horizon_M", "Quantile", "MAE", "WAPE", "Bias", "Pinball", "Coverage"]
for col in mt_required:
    check(col in mt.columns, f"AFG_ML_METRICS zawiera '{col}'")

print()

# ──────────────────────────────────────────────
# T2. SANITY CHECKS
# ──────────────────────────────────────────────
print("[T2] Sanity checks danych")

# Prognozy >= 0
for col in ["Pred_Orders_P50", "Pred_Orders_P90", "Pred_Sales_P50", "Pred_Sales_P90"]:
    vals = fc[col].dropna()
    check((vals >= 0).all(), f"Forecast: {col} >= 0", f"min={vals.min():.2f}")

# P90 >= P50 (kwantyl wyższy powinien być >= niższy w większości przypadków)
orders_p90_ge_p50 = (fc["Pred_Orders_P90"] >= fc["Pred_Orders_P50"]).mean()
if orders_p90_ge_p50 >= 0.7:
    ok(f"Forecast: Orders P90 >= P50 w {orders_p90_ge_p50:.0%} przypadków")
else:
    warn(f"Forecast: Orders P90 >= P50 tylko w {orders_p90_ge_p50:.0%} przypadków")

sales_p90_ge_p50 = (fc["Pred_Sales_P90"] >= fc["Pred_Sales_P50"]).mean()
if sales_p90_ge_p50 >= 0.7:
    ok(f"Forecast: Sales P90 >= P50 w {sales_p90_ge_p50:.0%} przypadków")
else:
    warn(f"Forecast: Sales P90 >= P50 tylko w {sales_p90_ge_p50:.0%} przypadków")

# FillRate w [0, 1]
fr = pp["FillRate_6M"]
check((fr >= 0).all() and (fr <= 1).all(), "Plan: FillRate_6M w [0, 1]",
      f"zakres [{fr.min():.4f}, {fr.max():.4f}]")

# LostSales >= 0
check((pp["LostSales_6M"] >= 0).all(), "Plan: LostSales_6M >= 0")

# RecommendedProductionQty >= 0
check((pp["RecommendedProductionQty"] >= 0).all(), "Plan: RecommendedProductionQty >= 0")

# ActionLabel tylko dozwolone wartości
allowed_actions = {"STOP", "SLOW", "SPEED"}
actual_actions = set(pp["ActionLabel"].unique())
check(actual_actions.issubset(allowed_actions),
      "Plan: ActionLabel in {STOP, SLOW, SPEED}",
      f"znalezione: {actual_actions}")

# Każdy SKU ma 6 horyzontów
sku_horizon_counts = fc.groupby("ProductID")["Horizon_M"].nunique()
check((sku_horizon_counts == 6).all(),
      "Forecast: każdy SKU ma 6 horyzontów",
      f"SKU z != 6: {(sku_horizon_counts != 6).sum()}")

pp_horizon_counts = pp.groupby("ProductID")["Horizon_M"].nunique()
check((pp_horizon_counts == 6).all(),
      "Plan: każdy SKU ma 6 horyzontów",
      f"SKU z != 6: {(pp_horizon_counts != 6).sum()}")

# Liczba SKU spójna
n_sku_fo = fo["ProductID"].nunique()
n_sku_fc = fc["ProductID"].nunique()
n_sku_pp = pp["ProductID"].nunique()
check(n_sku_fc == n_sku_fo, f"Forecast obejmuje wszystkie SKU",
      f"input={n_sku_fo}, forecast={n_sku_fc}")
check(n_sku_pp == n_sku_fo, f"Plan obejmuje wszystkie SKU",
      f"input={n_sku_fo}, plan={n_sku_pp}")

# Brak NaN w kluczowych kolumnach
check(fc["ProductID"].notna().all(), "Forecast: brak NaN w ProductID")
check(fc["Forecast_EOM"].notna().all(), "Forecast: brak NaN w Forecast_EOM")
check(pp["Forecast_EOM"].notna().all(), "Plan: brak NaN w Forecast_EOM")
check(pp["ActionLabel"].notna().all(), "Plan: brak NaN w ActionLabel")

print()

# ──────────────────────────────────────────────
# T3. COVERAGE P90 (~90%) — cel z Runbooka
# ──────────────────────────────────────────────
print("[T3] Coverage P90 w backteście (cel Runbooka: ~90%)")

coverage_orders = mt[(mt["Target"] == "orders") & (mt["Quantile"] == "P90")]["Coverage"]
coverage_sales = mt[(mt["Target"] == "sales") & (mt["Quantile"] == "P90")]["Coverage"]

avg_cov_orders = coverage_orders.mean()
avg_cov_sales = coverage_sales.mean()

if 0.80 <= avg_cov_orders <= 0.97:
    ok(f"Orders P90 coverage: {avg_cov_orders:.2%}", "akceptowalny zakres 80-97%")
elif avg_cov_orders >= 0.70:
    warn(f"Orders P90 coverage: {avg_cov_orders:.2%}", "poniżej celu 90%, ale >70%")
else:
    fail(f"Orders P90 coverage: {avg_cov_orders:.2%}", "znacząco poniżej celu")

if 0.80 <= avg_cov_sales <= 0.97:
    ok(f"Sales P90 coverage: {avg_cov_sales:.2%}", "akceptowalny zakres 80-97%")
elif avg_cov_sales >= 0.70:
    warn(f"Sales P90 coverage: {avg_cov_sales:.2%}", "poniżej celu 90%, ale >70%")
else:
    fail(f"Sales P90 coverage: {avg_cov_sales:.2%}", "znacząco poniżej celu")

# Coverage per horyzont
print("    Coverage P90 per horyzont (orders):")
for _, row in mt[(mt["Target"] == "orders") & (mt["Quantile"] == "P90")].iterrows():
    h = int(row["Horizon_M"])
    c = row["Coverage"]
    flag = "OK" if 0.80 <= c <= 0.97 else ("!" if c < 0.80 else "high")
    print(f"      h={h}: {c:.2%} [{flag}]")

print("    Coverage P90 per horyzont (sales):")
for _, row in mt[(mt["Target"] == "sales") & (mt["Quantile"] == "P90")].iterrows():
    h = int(row["Horizon_M"])
    c = row["Coverage"]
    flag = "OK" if 0.80 <= c <= 0.97 else ("!" if c < 0.80 else "high")
    print(f"      h={h}: {c:.2%} [{flag}]")

print()

# ──────────────────────────────────────────────
# T4. WAPE — rozsądne wartości
# ──────────────────────────────────────────────
print("[T4] Jakość prognoz P50 (WAPE)")

wape_orders = mt[(mt["Target"] == "orders") & (mt["Quantile"] == "P50")]["WAPE"]
wape_sales = mt[(mt["Target"] == "sales") & (mt["Quantile"] == "P50")]["WAPE"]

avg_wape_o = wape_orders.mean()
avg_wape_s = wape_sales.mean()

check(avg_wape_o < 0.30, f"Orders WAPE < 30%", f"średnie WAPE={avg_wape_o:.2%}")
check(avg_wape_s < 0.30, f"Sales WAPE < 30%", f"średnie WAPE={avg_wape_s:.2%}")

# WAPE rośnie z horyzontem (oczekiwane)
wape_h1 = mt[(mt["Target"] == "orders") & (mt["Quantile"] == "P50") & (mt["Horizon_M"] == 1)]["WAPE"].iloc[0]
wape_h6 = mt[(mt["Target"] == "orders") & (mt["Quantile"] == "P50") & (mt["Horizon_M"] == 6)]["WAPE"].iloc[0]
if wape_h6 >= wape_h1:
    ok("Orders WAPE rośnie z horyzontem (oczekiwane)", f"h1={wape_h1:.2%} → h6={wape_h6:.2%}")
else:
    warn("Orders WAPE nie rośnie z horyzontem", f"h1={wape_h1:.2%} → h6={wape_h6:.2%}")

print()

# ──────────────────────────────────────────────
# T5. BIAS — sprawdzenie systematycznego błędu
# ──────────────────────────────────────────────
print("[T5] Bias P50 (systematyczny błąd)")

for target in ["orders", "sales"]:
    biases = mt[(mt["Target"] == target) & (mt["Quantile"] == "P50")]["Bias"]
    avg_bias = biases.mean()
    # P90 powinien mieć pozytywny bias (zawyżać)
    biases_p90 = mt[(mt["Target"] == target) & (mt["Quantile"] == "P90")]["Bias"]
    avg_bias_p90 = biases_p90.mean()

    if abs(avg_bias) < 10:
        ok(f"{target} P50 bias akceptowalny", f"średni bias={avg_bias:.2f}")
    else:
        warn(f"{target} P50 bias znaczący", f"średni bias={avg_bias:.2f}")

    check(avg_bias_p90 > 0,
          f"{target} P90 bias pozytywny (zawyża — to dobrze dla bufora serwisu)",
          f"średni bias={avg_bias_p90:.2f}")

print()

# ──────────────────────────────────────────────
# T6. PORÓWNANIE Z TARGETAMI WALIDACYJNYMI Z TRAIN
# ──────────────────────────────────────────────
print("[T6] Porównanie FillRate/LostSales z targetami Train")

tr["ProductID"] = tr["ProductID"].astype(str)
pp["ProductID"] = pp["ProductID"].astype(str)
fc["ProductID"] = fc["ProductID"].astype(str)
bt["ProductID"] = bt["ProductID"].astype(str)
fo["ProductID"] = fo["ProductID"].astype(str)

# Historyczne FillRate per SKU z Train
hist_fr = tr.groupby("ProductID")["KPI_FillRate_Qty_LT"].mean().reset_index()
hist_fr.columns = ["ProductID", "HistFillRate"]

# Pipeline FillRate
plan_fr = pp.groupby("ProductID")["FillRate_6M"].first().reset_index()
plan_fr.columns = ["ProductID", "PlanFillRate"]

comp = hist_fr.merge(plan_fr, on="ProductID", how="inner")

print("    Porównanie FillRate (historyczny vs plan):")
print("    " + "-" * 55)
print(f"    {'SKU':>10s}  {'Hist(Train)':>12s}  {'Plan(ML)':>10s}  {'Delta':>8s}")
print("    " + "-" * 55)
for _, r in comp.iterrows():
    delta = r["PlanFillRate"] - r["HistFillRate"]
    flag = " <-- spadek!" if delta < -0.15 else ""
    print(f"    {r['ProductID']:>10s}  {r['HistFillRate']:>12.2%}  {r['PlanFillRate']:>10.2%}  {delta:>+8.2%}{flag}")

avg_hist = comp["HistFillRate"].mean()
avg_plan = comp["PlanFillRate"].mean()
print("    " + "-" * 55)
print(f"    {'ŚREDNIA':>10s}  {avg_hist:>12.2%}  {avg_plan:>10.2%}  {avg_plan - avg_hist:>+8.2%}")

# Korelacja — czy ranking SKU jest spójny
if len(comp) >= 5:
    corr = comp["HistFillRate"].corr(comp["PlanFillRate"])
    check(corr > 0.3, f"Korelacja rankingu FillRate (hist vs plan)",
          f"r={corr:.3f}")
else:
    warn("Za mało SKU do korelacji rankingu")

# Historyczne LostSales per SKU z Train
hist_ls = tr.groupby("ProductID")["KPI_LostSales_Qty_LT"].mean().reset_index()
hist_ls.columns = ["ProductID", "HistLS"]
plan_ls = pp.groupby("ProductID")["LostSales_6M"].first().reset_index()
plan_ls.columns = ["ProductID", "PlanLS"]
comp_ls = hist_ls.merge(plan_ls, on="ProductID", how="inner")

print()
print("    Porównanie LostSales (średnia hist. miesięczna vs suma 6M planu):")
print("    " + "-" * 55)
print(f"    {'SKU':>10s}  {'Hist/mies':>12s}  {'Plan 6M':>10s}")
print("    " + "-" * 55)
for _, r in comp_ls.iterrows():
    print(f"    {r['ProductID']:>10s}  {r['HistLS']:>12.1f}  {r['PlanLS']:>10.1f}")

print()

# ──────────────────────────────────────────────
# T7. STABILNOŚĆ BACKTESTOWA (dryf metryk w czasie)
# ──────────────────────────────────────────────
print("[T7] Stabilność backtestowa (dryf metryk w czasie)")

bt["Run_Cutoff_EOM"] = pd.to_datetime(bt["Run_Cutoff_EOM"])

# WAPE per cutoff dla orders P50
orders_p50_bt = bt[(bt["Target"] == "orders")].copy()
if "Predicted_P50" in orders_p50_bt.columns and "Actual" in orders_p50_bt.columns:
    orders_p50_bt["AE"] = np.abs(orders_p50_bt["Actual"] - orders_p50_bt["Predicted_P50"])
    wape_by_cutoff = orders_p50_bt.groupby("Run_Cutoff_EOM").apply(
        lambda g: g["AE"].sum() / max(g["Actual"].abs().sum(), 1),
        include_groups=False
    )

    if len(wape_by_cutoff) >= 5:
        first_half = wape_by_cutoff.iloc[:len(wape_by_cutoff)//2].mean()
        second_half = wape_by_cutoff.iloc[len(wape_by_cutoff)//2:].mean()
        drift = second_half - first_half

        if abs(drift) < 0.10:
            ok(f"Brak znaczącego dryfu WAPE", f"1. połowa={first_half:.2%}, 2. połowa={second_half:.2%}, drift={drift:+.2%}")
        else:
            warn(f"Dryf WAPE między połowami backtestów", f"1.={first_half:.2%}, 2.={second_half:.2%}, drift={drift:+.2%}")

        # Std deviation WAPE per cutoff
        wape_std = wape_by_cutoff.std()
        if wape_std < 0.15:
            ok(f"WAPE stabilne w czasie", f"std={wape_std:.4f}")
        else:
            warn(f"WAPE niestabilne", f"std={wape_std:.4f}")
    else:
        warn("Za mało cutoffów do analizy dryfu")
else:
    warn("Brak kolumn Predicted_P50/Actual w backteście do analizy dryfu")

print()

# ──────────────────────────────────────────────
# T8. RĘCZNA WERYFIKACJA SERWISU (1 SKU)
# ──────────────────────────────────────────────
print("[T8] Ręczna weryfikacja obliczeń serwisowych (1 SKU)")

# Wybierz SKU z danymi serwisowymi
test_sku = pp["ProductID"].iloc[0]
sku_plan = pp[pp["ProductID"] == test_sku].sort_values("Horizon_M")

# Odtwórz FG start
fo["ProductID"] = fo["ProductID"].astype(str)
last_eom = pd.to_datetime(fo["EOM"]).max()
sku_fo = fo[(fo["ProductID"].astype(str) == str(test_sku))]
last_fg_row = sku_fo[pd.to_datetime(sku_fo["EOM"]) == last_eom]

if len(last_fg_row) > 0:
    fg_start = float(last_fg_row["OPS_FG_Qty_EOM"].iloc[0]) if pd.notna(last_fg_row["OPS_FG_Qty_EOM"].iloc[0]) else 0.0
    lt = int(last_fg_row["LT_LeadTimeMonths_Used"].iloc[0])
    print(f"    SKU: {test_sku}, FG_start={fg_start}, LT={lt} mies.")
    print(f"    {'h':>3s}  {'Demand':>8s}  {'RecProd':>8s}  {'Avail':>8s}  {'Served':>8s}  {'Lost':>8s}  {'Action':>6s}")
    print("    " + "-" * 60)

    recomputed_lost = 0
    recomputed_demand = 0
    fg = fg_start

    for _, r in sku_plan.iterrows():
        h = int(r["Horizon_M"])
        demand = r["Demand_P90"]
        rec_prod = r["RecommendedProductionQty"]

        # Replikacja logiki pipeline
        prod_arrival = rec_prod if h > lt else 0
        availability = fg + prod_arrival
        served = min(demand, availability)
        lost = max(0, demand - availability)
        fg = max(0, availability - served)

        recomputed_lost += lost
        recomputed_demand += demand

        # Porównanie
        match_lost = abs(lost - r["LostSales_M"]) < 0.1
        flag = "" if match_lost else " <-- MISMATCH!"
        print(f"    {h:>3d}  {demand:>8.1f}  {rec_prod:>8.0f}  {availability:>8.1f}  {served:>8.1f}  {lost:>8.1f}  {r['ActionLabel']:>6s}{flag}")

    recomputed_fr = 1 - recomputed_lost / max(recomputed_demand, 1)
    plan_fr_val = sku_plan["FillRate_6M"].iloc[0]
    plan_ls_val = sku_plan["LostSales_6M"].iloc[0]

    print("    " + "-" * 60)
    print(f"    Recomputed: LostSales_6M={recomputed_lost:.2f}  FillRate_6M={recomputed_fr:.4f}")
    print(f"    Pipeline:   LostSales_6M={plan_ls_val:.2f}  FillRate_6M={plan_fr_val:.4f}")

    check(abs(recomputed_lost - plan_ls_val) < 0.5,
          "LostSales_6M zgodny z ręcznym przeliczeniem",
          f"delta={abs(recomputed_lost - plan_ls_val):.2f}")
    check(abs(recomputed_fr - plan_fr_val) < 0.01,
          "FillRate_6M zgodny z ręcznym przeliczeniem",
          f"delta={abs(recomputed_fr - plan_fr_val):.4f}")
else:
    warn(f"Brak danych FG dla SKU {test_sku} — pominięto weryfikację ręczną")

print()

# ──────────────────────────────────────────────
# T9. POWTARZALNOŚĆ (deterministyczność)
# ──────────────────────────────────────────────
print("[T9] Powtarzalność (random_state)")
# Sprawdzamy czy pipeline ma zdefiniowany random_state
pipeline_path = HERE / "afg_forecast_pipeline.py"
if pipeline_path.exists():
    code = pipeline_path.read_text(encoding="utf-8")
    has_seed = "random_state" in code
    check(has_seed, "Pipeline ma zdefiniowany random_state")
    if has_seed:
        ok("Wyniki powinny być deterministyczne przy ponownym uruchomieniu")
else:
    warn("Nie znaleziono afg_forecast_pipeline.py")

print()

# ──────────────────────────────────────────────
# T10. KOMPLETNOŚĆ HORYZONTU BACKTESTU
# ──────────────────────────────────────────────
print("[T10] Kompletność rolling backtestu")

n_cutoffs = bt["Run_Cutoff_EOM"].nunique()
n_targets = bt["Target"].nunique()
n_horizons = bt["Horizon_M"].nunique()

check(n_cutoffs >= 10, f"Wystarczająca liczba cutoffów", f"{n_cutoffs}")
check(n_targets == 2, f"Oba targety w backteście", f"orders + sales")
check(n_horizons == 6, f"Wszystkie 6 horyzontów w backteście")

# Nie powinno być przyszłych dat jako actual
bt["Forecast_EOM"] = pd.to_datetime(bt["Forecast_EOM"])
max_forecast = bt["Forecast_EOM"].max()
max_data = last_eom
check(max_forecast <= max_data + pd.DateOffset(months=1),
      "Backtest nie wykracza poza dostępne dane",
      f"max_forecast={max_forecast.strftime('%Y-%m')}, max_data={max_data.strftime('%Y-%m')}")

print()

# ──────────────────────────────────────────────
# PODSUMOWANIE
# ──────────────────────────────────────────────
print("=" * 65)
total = PASS + FAIL + WARN
print(f"  WYNIK: {PASS} PASS / {FAIL} FAIL / {WARN} WARN  (łącznie {total} testów)")
if FAIL == 0:
    print("  STATUS: WSZYSTKIE TESTY PRZESZŁY POMYŚLNIE")
else:
    print(f"  STATUS: {FAIL} TESTÓW NIE PRZESZŁO — sprawdź szczegóły powyżej")
print("=" * 65)
