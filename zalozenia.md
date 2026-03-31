# Zalozenia aplikacji — AI Finance Guard

## Cel

Aplikacja ML prognozujaca przyszle zapotrzebowanie na produkty (SKU) w horyzoncie 6 miesiecy i generujaca rekomendacje produkcyjne (STOP / SLOW / SPEED), aby minimalizowac utracona sprzedaz przy kontroli zapasow.

## Architektura

```
dane_input/CSV  -->  Pipeline ML  -->  dane_output/CSV  -->  Dashboard / Power BI
```

### 3 moduly:

1. **Forecast Demand** — prognoza OrdersIn_Qty i SalesQty per SKU na 6M, w wariantach P50 (dokladnosc) i P90 (bufor serwisowy)
2. **Rolling Backtest** — walidacja historyczna prognoz (cutoff co 3 miesiace, metryki MAE/WAPE/Bias/Pinball/Coverage)
3. **Service Simulation + Production Plan** — symulacja dostepnosci z lead time i generowanie planu produkcji z etykietami akcji

## Modele ML

### Decyzja: LightGBM + ARIMA (bez CatBoost)

Poczatkowo planowano LightGBM + CatBoost. **Zrezygnowano z CatBoost** z powodow:
- Problemy z kodowaniem UTF-8 w nazwach kolumn (polskie znaki)
- Porownywalna szybkosc do LightGBM, ale bez przewagi jakosciowej
- Dodatkowa zaleznosc bez istotnej wartosci

**Finalny zestaw:**

| Model | Typ | Rola | Opis |
|---|---|---|---|
| **LightGBM** | Gradient boosting (tablicowy) | Zaawansowany | Uzywa pelnego feature store (rolling, YoY, backlog, lead time, sezonowosc). Quantile regression P50+P90. |
| **ARIMA** | Szereg czasowy (univariate) | Baseline | Klasyczny model sezonowy. Sluzy jako punkt odniesienia do oceny czy LightGBM faktycznie jest lepszy. |

Modele mozna uruchamiac **pojedynczo lub razem** — przy dwoch modelach generowane jest automatyczne porownanie.

## Dane wejsciowe

Dwa pliki CSV w `dane_input/`:

### SKU_FeatureOnly_*.csv (63 kolumny)
- **Klucze:** ProductID, RokMiesiac, EOM
- **Targety prognozy:** KPI_OrdersIn_Qty, KPI_SalesQty
- **Features:** rolling 3M/6M, lag-12, YoY, Book-to-Bill, backlog momentum, FG coverage, lead time, sygnaly kosztowe
- **Format:** decimal=`,` (europejski), encoding UTF-8-sig

### SKU_FeatureTrain_*.csv (66 kolumn)
- Wszystko z FeatureOnly + 3 kolumny walidacyjne serwisu:
  - KPI_LostSales_Qty_LT
  - KPI_FillRate_Qty_LT
  - KPI_LostSales_Value_LT

### Granulacja
- 1 wiersz = 1 SKU x 1 miesiac (EOM)
- Sortowanie: ProductID rosnaco, EOM rosnaco

## Pliki wyjsciowe (dane_output/)

| Plik | Zawiera | Dla kogo |
|---|---|---|
| `Forecast_Demand.csv` | Prognozy P50/P90 per SKU x horyzont | Power BI: strona Forecast |
| `Plan_Production.csv` | Rekomendacje STOP/SLOW/SPEED + FillRate + LostSales | Power BI: strona Drivers/Executive |
| `AFG_ML_EVAL_BACKTEST.csv` | Historyczna jakosc prognoz | Power BI: diagnostyka |
| `AFG_ML_METRICS.csv` | Zagregowane metryki (MAE, WAPE, Coverage) | Power BI: KPI jakosci |
| `AFG_ML_MODEL_COMPARISON.csv` | Porownanie modeli (jesli uruchomiono 2+) | Power BI: wybor modelu |

Pliki sa tez generowane z sufiksem modelu (np. `_lightgbm`, `_arima`).

## Dashboard webowy (Streamlit)

7 zakladek:
1. **Executive** — KPI karty, wykresy FillRate/LostSales per SKU, Top Risk
2. **Prognozy** — historia + P50/P90 per SKU z pasmem niepewnosci
3. **Plan produkcji** — Demand vs Availability vs Production, etykiety STOP/SLOW/SPEED
4. **Backtest** — Actual vs Predicted scatter, rozklad bledow, WAPE w czasie
5. **Metryki** — WAPE/MAE per horyzont (P50), Coverage/Pinball (P90)
6. **Porownanie modeli** — WAPE i Coverage per model obok siebie
7. **Logi** — pelen log pipeline i testow

Kazda zakladka ma rozwijana **LEGENDE** z wyjasnieniem wszystkich wielkosci.

Sidebar: upload CSV, wybor modeli (multiselect), przyciski uruchomienia, checkboxy wlaczania/wylaczania modeli.

## Kluczowe pojecia biznesowe

| Pojecie | Znaczenie |
|---|---|
| **OrdersIn** | Naplyw zamowien od klientow (popyt) |
| **Sales** | Zrealizowana sprzedaz (z faktur) |
| **FillRate** | % popytu obsluzonego (1.0 = 100%) |
| **LostSales** | Popyt nieobsluzony z powodu braku towaru |
| **Lead Time** | Czas cyklu produkcyjnego (z CYKL_PROD_DNI) |
| **P50** | Prognoza srodkowa (50% szans powyzej/ponizej) |
| **P90** | Prognoza pesymistyczna (90% szans ze rzeczywistosc nie przekroczy) |
| **STOP/SLOW/SPEED** | Rekomendacja produkcyjna vs historyczna srednia |

## Ograniczenia i zalozenia

- Granulacja miesieczna (brak danych dziennych)
- TERMIN w zamowieniach jest niewiarygodny — nie bazujemy na due-date
- Lead time z CYKL_PROD_DNI (mediana per SKU)
- SKU z lead time > 6 miesiecy maja z natury niski FillRate w horyzoncie 6M
- Backtest co 3 miesiace (kompromis miedzy dokladnoscia a czasem obliczen)
