# Prompt do Claude Code — budowa aplikacji AI Finance Guard

Ponizszy prompt nalezy wkleic do Claude Code po uruchomieniu go w katalogu projektu.
Mozna go wkleic w calosci lub etapami (kazdy etap osobno).

---

## ETAP 1: Analiza danych i pipeline ML

```
W katalogu dane_input/ znajduja sie pliki CSV:
- SKU_FeatureOnly_*.csv — dane z features (63 kolumny)
- SKU_FeatureTrain_*.csv — to samo + 3 kolumny walidacyjne serwisu

Granulacja: 1 wiersz = ProductID x EOM (miesiac). Format: decimal=",", encoding UTF-8-sig.

Targety prognozy: KPI_OrdersIn_Qty (popyt) i KPI_SalesQty (sprzedaz).
Horyzont: 6 miesiecy (h=1..6).
Prognozy: P50 (dokladnosc) + P90 (bufor serwisowy).

Stworz plik afg_forecast_pipeline.py ktory:

1. Wczytuje dane z dane_input/, waliduje (unikalnosc kluczy, kompletnosc targetow)
2. Buduje etykiety przez shift: y(t+h) = target(t+h) dla h=1..6
3. Implementuje 2 modele do wyboru:
   - lightgbm: LightGBM quantile regression (P50 alpha=0.5, P90 alpha=0.9)
   - arima: ARIMA(1,1,1) per SKU z kwantylami przez residual std * z_score
4. Rolling backtest (cutoff co 3 miesiace, min 24 mies. treningowych)
5. Oblicza metryki: MAE, WAPE, Bias, Pinball, Coverage
6. Generuje prognozę finalną na 6M od ostatniego EOM
7. Symuluje serwis: Availability = FG_Begin + Prod_Arrival(po lead time),
   LostSales = max(0, Demand_P90 - Availability), FillRate = 1 - LostSales/Demand
8. Generuje plan produkcji z etykietami STOP/SLOW/SPEED
9. Eksportuje wyniki do dane_output/:
   - Forecast_Demand.csv (P50/P90 per SKU x horyzont)
   - AFG_ML_EVAL_BACKTEST.csv (backtest z metrykami)
   - Plan_Production.csv (rekomendacje + FillRate + LostSales)
   - AFG_ML_METRICS.csv (zagregowane metryki)
   - AFG_ML_MODEL_COMPARISON.csv (jesli uruchomiono 2+ modeli)

CLI: python afg_forecast_pipeline.py --model lightgbm
     python afg_forecast_pipeline.py --model lightgbm,arima

Dzialaj autonomicznie, uruchom pipeline i pokaz wyniki.
```

---

## ETAP 2: Testy walidacyjne

```
Stworz plik test_pipeline.py ktory waliduje wyniki w dane_output/:

1. Istnienie 4 plikow wyjsciowych
2. Schematy kolumn vs Runbook (Forecast_Demand, Plan_Production, Backtest, Metrics)
3. Sanity checks: prognozy >= 0, P90 >= P50, FillRate w [0,1], ActionLabel in {STOP,SLOW,SPEED},
   kazdy SKU ma 6 horyzontow, kompletnosc SKU
4. Coverage P90 ~90% (cel Runbooka)
5. WAPE P50 < 30%, rosnie z horyzontem
6. Bias P50 maly, P90 pozytywny
7. Porownanie FillRate/LostSales z targetami walidacyjnymi z Train
   (KPI_LostSales_Qty_LT, KPI_FillRate_Qty_LT)
8. Stabilnosc backtestowa (brak dryfu WAPE)
9. Reczna weryfikacja obliczen serwisowych dla 1 SKU
10. Determinizm (random_state), kompletnosc backtestu

Dane wejsciowe czyta z dane_input/, wynikowe z dane_output/.
Uruchom testy i pokaz wyniki.
```

---

## ETAP 3: Dashboard webowy

```
Stworz plik app.py — dashboard Streamlit (streamlit run app.py):

Sidebar:
- Upload plikow CSV (FeatureOnly + FeatureTrain) -> zapisuje do dane_input/
- Multiselect modeli (lightgbm, arima)
- Przyciski: Uruchom Pipeline, Uruchom Testy
- Checkboxy wlaczania/wylaczania modeli w dashboardzie
- Selectbox aktywnego modelu do podgladu

7 zakladek:

1. Executive: KPI karty (Avg FillRate, Total LostSales, SKU w planie, SKU zagrozonych),
   wykresy slupkowe FillRate i LostSales per SKU, tabela Top Risk

2. Prognozy: wybor SKU, 2 wykresy (OrdersIn + Sales) z historia + P50 + P90 + pasmo
   niepewnosci, tabela prognoz

3. Plan produkcji: wybor SKU, KPI (FillRate, LostSales, Lead Time),
   wykres Demand vs Availability vs Production vs Served,
   tabela z kolorowymi etykietami STOP/SLOW/SPEED,
   przeglad wszystkich SKU

4. Backtest: radio orders/sales, scatter Actual vs Predicted,
   histogram bledow, WAPE per cutoff

5. Metryki: 2 kolumny — P50 (WAPE + MAE per horyzont) i P90 (Coverage + Pinball),
   pelna tabela

6. Porownanie modeli (jesli 2+ wlaczonych): WAPE i Coverage per model per horyzont,
   tabela podsumowujaca

7. Logi: pipeline + testy

Na dole kazdej zakladki dodaj rozwijana sekcje LEGENDA (st.expander)
z wyjasnieniem wszystkich wielkosci prostym jezykiem.

Pliki wejsciowe z dane_input/, wynikowe z dane_output/.
Uzyj plotly do wykresow. Dzialaj autonomicznie.
```

---

## ETAP 4: Dane testowe

```
Z pelnego zbioru danych (11 SKU) wygeneruj pliki testowe w dane_input/:
- 1 SKU (np. 568859): SKU_FeatureOnly_1SKU_test.csv + SKU_FeatureTrain_1SKU_test.csv
- 3 SKU (rozne profile — krotki LT, dlugi LT, sredni): SKU_FeatureOnly_3SKU_test.csv + SKU_FeatureTrain_3SKU_test.csv

Nazwy z prefixem TEST_ lub sufiksem _test zeby pipeline ich nie zlapał automatycznie.
```

---

## Uwagi dla studentow

- Kazdy etap mozna wkleic osobno — Claude Code zapamietuje kontekst rozmowy
- Jesli cos nie dziala, powiedz "napraw blad" i wklej komunikat
- Mozecie modyfikowac prompty — np. zmienic liczbe horyzontow, dodac nowy model
- Po kazdym etapie sprawdzcie czy pliki sie wygenerowaly i czy pipeline przechodzi
- Dashboard mozna uruchomic komenda: streamlit run app.py
