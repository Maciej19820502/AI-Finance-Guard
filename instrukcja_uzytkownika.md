# Instrukcja uzytkownika — AI Finance Guard

## Szybki start (5 minut)

### 1. Uruchom dashboard
```bash
cd sciezka/do/NOWE
streamlit run app.py
```
Otworz przegladarke: **http://localhost:8501**

### 2. Wczytaj dane
W panelu bocznym (sidebar):
- **FeatureOnly CSV** — uploaduj plik `SKU_FeatureOnly_*.csv`
- **FeatureTrain CSV** — uploaduj plik `SKU_FeatureTrain_*.csv`

Pliki trafia do `dane_input/`.

### 3. Wybierz model i uruchom
- Wybierz model(e): **lightgbm** (rekomendowany) i/lub **arima** (baseline)
- Kliknij **Uruchom Pipeline**
- Poczekaj na zakonczenie (1 SKU = sekundy, 11 SKU = kilka minut)

### 4. Przegladaj wyniki
Wyniki pojawia sie w zakladkach dashboardu. Mozesz tez kliknac **Uruchom Testy** zeby zwalidowac jakosc.

---

## Co daje kazda zakladka

### Executive — "jak jest ogolnie?"
- **Avg FillRate** — ogolny poziom obslugi (cel: >70%)
- **Total LostSales** — ile tracimy lacznie
- **Czerwone SKU** — ktore produkty wymagaja natychmiastowej reakcji
- **Akcja:** patrzysz na Top Risk i decydujesz gdzie interweniowac

### Prognozy — "ile bedziemy potrzebowac?"
- Wybierz SKU z listy
- **P50** (zielona) = najbardziej prawdopodobna wartosc
- **P90** (czerwona) = scenariusz pesymistyczny (bufor bezpieczenstwa)
- **Pasmo** = niepewnosc — im szersze, tym mniej pewna prognoza
- **Akcja:** uzyj P50 do planowania budzetu, P90 do planowania zapasow

### Plan produkcji — "co produkowac?"
- Wybierz SKU
- **SPEED** (zielona) = produkuj normalnie lub wiecej
- **SLOW** (pomaranczowa) = zmniejsz produkcje
- **STOP** (czerwona) = wstrzymaj produkcje
- **Lead Time** = wazne! Jesli LT > 6 mies., produkcja uruchomiona dzis nie zdazy w horyzoncie prognozy
- **Akcja:** przekaz rekomendacje do planowania produkcji

### Backtest — "czy prognoza jest wiarygodna?"
- Punkty blisko czerwonej linii = dobra prognoza
- Rozklad bledow centrowany na 0 = brak systematycznego bledu
- Stabilny WAPE w czasie = model konsekwentny
- **Akcja:** jesli WAPE > 30% lub duzy dryf — model wymaga poprawy

### Metryki — "jak dokladna jest prognoza?"
- **WAPE < 20%** = dobra prognoza
- **Coverage P90 ~90%** = bufor dobrze skalibrowany
- Jesli Coverage = 100% to P90 jest za wysoka (marnujemy zapas)
- **Akcja:** porownaj metryki miedzy horyzontami (h1 vs h6)

### Porownanie modeli — "ktory model lepszy?"
- Widoczna gdy wlaczono 2+ modeli
- **Nizszy WAPE = lepszy model**
- **Coverage blizej 90% = lepiej skalibrowany**
- **Akcja:** wybierz model z najnizszym WAPE do produkcji

---

## Comiesięczny cykl pracy

```
1. Eksportuj nowe dane z BI -> pliki SKU_Feature*.csv
2. Wrzuc do dane_input/ (lub uploaduj przez dashboard)
3. Kliknij "Uruchom Pipeline"
4. Kliknij "Uruchom Testy" (opcjonalnie)
5. Przejrzyj zakladke Executive -> zidentyfikuj ryzyka
6. Przejrzyj Plan produkcji -> przekaz rekomendacje STOP/SLOW/SPEED
7. Pliki z dane_output/ zaimportuj do Power BI
```

## Praca z CLI (zaawansowane)

```bash
# Sam LightGBM
python afg_forecast_pipeline.py --model lightgbm

# Sam ARIMA (baseline)
python afg_forecast_pipeline.py --model arima

# Oba modele + porownanie
python afg_forecast_pipeline.py --model lightgbm,arima

# Testy walidacyjne
python test_pipeline.py
```

---

## Interpretacja kluczowych wartosci

| Wartosc | Co znaczy | Dobra | Zla |
|---|---|---|---|
| FillRate 6M | % popytu obsluzonego | > 70% | < 50% |
| LostSales 6M | Szt. nieobsluzone | Blisko 0 | Rosnaca |
| WAPE P50 | Blad prognozy % | < 15% | > 30% |
| Coverage P90 | Trafnosc bufora | 85-95% | < 80% lub 100% |
| Lead Time | Cykl produkcji (mies.) | 1-3 | > 6 (nie zdazymy) |
| SPEED | Produkuj normalnie | Popyt pokryty | — |
| SLOW | Zmniejsz produkcje | Zapas rosnie | — |
| STOP | Wstrzymaj produkcje | Brak popytu | — |

## FAQ

**Dlaczego SKU 472404 ma FillRate 12%?**
Bo ma Lead Time 8 miesiecy. W horyzoncie 6M nic co dzis uruchomimy nie zdazy dotrzec. To nie blad modelu — to realna informacja ze trzeba planowac wczesniej.

**Dlaczego ARIMA jest gorszy od LightGBM?**
ARIMA uzywa tylko historii danego SKU (1 zmienna). LightGBM uzywa 52 features (rolling, YoY, backlog, lead time). Wiecej informacji = lepsza prognoza.

**Czy moge dodac nowy model?**
Tak — w pliku `afg_forecast_pipeline.py` dodajcie nowa klase w MODEL_REGISTRY (np. Prophet, XGBoost).

**Co jesli upload nie dziala?**
Skopiujcie pliki CSV recznie do folderu `dane_input/` i kliknijcie "Uruchom Pipeline".
