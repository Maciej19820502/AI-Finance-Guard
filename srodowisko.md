# Srodowisko pracy — jak zaczac z Claude Code

## Co to jest Claude Code?

Claude Code to narzedzie AI, ktore pisze kod za Was. Mowicie mu co chcecie zrobic, a on tworzy pliki, uruchamia programy i naprawia bledy. Dziala w terminalu (konsoli).

## Wymagania

1. **Windows 10/11** (lub Mac/Linux)
2. **Python 3.10+** zainstalowany — sprawdzcie w terminalu:
   ```
   python --version
   ```
3. **Node.js 18+** — potrzebny do Claude Code:
   ```
   node --version
   ```
4. **Konto Anthropic** z kluczem API lub subskrypcja Claude Pro/Max

## Instalacja Claude Code

### Krok 1: Zainstaluj Claude Code
```bash
npm install -g @anthropic-ai/claude-code
```

### Krok 2: Uruchom w katalogu projektu
```bash
cd sciezka/do/waszego/folderu
claude
```

### Krok 3: Zaloguj sie
Przy pierwszym uruchomieniu Claude Code poprosi o klucz API lub logowanie. Postepujcie wg instrukcji na ekranie.

## Jak rozmawiac z Claude Code

Claude Code dziala jak czat w terminalu. Piszecie polecenia po polsku lub angielsku.

### Przyklady polecen:
```
> Przeanalizuj pliki CSV w tym folderze
> Napisz skrypt Python ktory wczyta dane i wytrenuje model LightGBM
> Uruchom testy i napraw bledy
> Dodaj wykres do dashboardu
```

### Wazne zasady:
- **Badz konkretny** — "napisz forecast pipeline z LightGBM na danych SKU" jest lepsze niz "zrob cos z danymi"
- **Podawaj kontekst** — "w pliku SKU_FeatureOnly.csv sa dane miesięczne per SKU, kolumna KPI_OrdersIn_Qty to target"
- **Mozesz wklejac tresc plikow** — np. specyfikacje, opisy kolumn
- **Jesli cos nie dziala** — powiedz "napraw blad" i wklej komunikat bledu

### Przydatne komendy w Claude Code:
- `/help` — pomoc
- `/clear` — wyczysc kontekst rozmowy
- `! polecenie` — uruchom polecenie shell bezposrednio (np. `! python skrypt.py`)

## Instalacja bibliotek Python

Claude Code zainstaluje je sam gdy beda potrzebne, ale mozecie tez recznie:
```bash
pip install pandas numpy lightgbm statsmodels streamlit plotly scikit-learn
```

## Struktura katalogow

Zalecana struktura projektu:
```
NOWE/
├── dane_input/          # tu wrzucacie pliki CSV z danymi
├── dane_output/         # tu trafiaja wyniki (prognozy, plan, metryki)
├── afg_forecast_pipeline.py   # pipeline ML (Claude Code wygeneruje)
├── app.py                     # dashboard webowy
└── test_pipeline.py           # testy walidacyjne
```

## Pierwsze uruchomienie

1. Wrzuccie pliki CSV do folderu `dane_input/`
2. Uruchomcie Claude Code: `claude`
3. Wklejcie prompt z pliku `prompt_claude_code.md`
4. Obserwujcie jak Claude Code tworzy aplikacje
5. Po zakonczeniu: `streamlit run app.py` — otworzy dashboard

## Typowe problemy

| Problem | Rozwiazanie |
|---|---|
| `python: command not found` | Zainstaluj Python lub uzyj `python3` |
| `pip install` nie dziala | Uzyj `python -m pip install` |
| Streamlit nie startuje | Sprawdz czy port 8501 jest wolny |
| Claude Code nie widzi plikow | Upewnij sie ze jestes w dobrym katalogu (`cd`) |
| Blad encoding (polskie znaki) | Claude Code ogarnie to sam — nie zmieniajcie kodowania plikow |
