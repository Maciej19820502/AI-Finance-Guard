"""
AI Finance Guard — Web Dashboard (Streamlit)
=============================================
Uruchom:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import subprocess, sys

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
HERE = Path(__file__).parent
INPUT_DIR = HERE / "dane_input"
OUTPUT_DIR = HERE / "dane_output"
AVAILABLE_MODELS = ["lightgbm", "arima"]

st.set_page_config(
    page_title="AI Finance Guard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def find_latest(pattern, directory=None):
    d = directory or INPUT_DIR
    matches = sorted(d.glob(pattern))
    return matches[-1] if matches else None


def load_csv(name, directory=None):
    d = directory or OUTPUT_DIR
    p = d / name
    return pd.read_csv(p) if p.exists() else None


def load_input(pattern):
    p = find_latest(pattern, INPUT_DIR)
    return pd.read_csv(p, encoding="utf-8-sig", decimal=",") if p else None


def get_available_model_results():
    """Znajdz modele, dla ktorych istnieja wyniki w dane_output/."""
    found = {}
    for m in AVAILABLE_MODELS:
        if (OUTPUT_DIR / f"AFG_ML_METRICS_{m}.csv").exists():
            found[m] = True
    if (OUTPUT_DIR / "AFG_ML_METRICS.csv").exists() and not found:
        found["lightgbm"] = True
    return list(found.keys())


def load_model_results(model_name):
    """Wczytaj wyniki dla danego modelu z dane_output/."""
    suffix = f"_{model_name}"
    fc = load_csv(f"Forecast_Demand{suffix}.csv")
    pp = load_csv(f"Plan_Production{suffix}.csv")
    bt = load_csv(f"AFG_ML_EVAL_BACKTEST{suffix}.csv")
    mt = load_csv(f"AFG_ML_METRICS{suffix}.csv")
    if fc is None:
        fc = load_csv("Forecast_Demand.csv")
    if pp is None:
        pp = load_csv("Plan_Production.csv")
    if bt is None:
        bt = load_csv("AFG_ML_EVAL_BACKTEST.csv")
    if mt is None:
        mt = load_csv("AFG_ML_METRICS.csv")
    return fc, pp, bt, mt


def run_pipeline(models_str):
    result = subprocess.run(
        [sys.executable, str(HERE / "afg_forecast_pipeline.py"),
         "--model", models_str],
        capture_output=True, text=True, cwd=str(HERE), timeout=900,
        encoding="utf-8", errors="replace",
    )
    return result.stdout + result.stderr, result.returncode


def run_tests():
    result = subprocess.run(
        [sys.executable, str(HERE / "test_pipeline.py")],
        capture_output=True, text=True, cwd=str(HERE), timeout=120,
        encoding="utf-8", errors="replace",
    )
    return result.stdout + result.stderr, result.returncode


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("AI Finance Guard")
    st.caption("Prognoza popytu i plan produkcji (6M)")
    st.divider()

    # Upload
    st.subheader("1. Dane wejsciowe")
    uploaded_fo = st.file_uploader("FeatureOnly CSV", type="csv", key="fo")
    uploaded_tr = st.file_uploader("FeatureTrain CSV", type="csv", key="tr")

    if uploaded_fo and uploaded_tr:
        INPUT_DIR.mkdir(exist_ok=True)
        (INPUT_DIR / "SKU_FeatureOnly_upload.csv").write_bytes(uploaded_fo.getvalue())
        (INPUT_DIR / "SKU_FeatureTrain_upload.csv").write_bytes(uploaded_tr.getvalue())
        st.success("Pliki zapisane do dane_input/.")

    st.divider()

    # Model selection
    st.subheader("2. Wybor modeli")
    sel_models = st.multiselect(
        "Modele do uruchomienia",
        AVAILABLE_MODELS,
        default=["lightgbm"],
        help="Wybierz 1+ modeli. Przy wielu modelach generowane jest porownanie."
    )

    col1, col2 = st.columns(2)
    with col1:
        btn_pipeline = st.button("Uruchom Pipeline", type="primary", use_container_width=True)
    with col2:
        btn_tests = st.button("Uruchom Testy", use_container_width=True)

    if btn_pipeline:
        if not sel_models:
            st.error("Wybierz przynajmniej 1 model.")
        else:
            models_str = ",".join(sel_models)
            with st.spinner(f"Pipeline: {models_str}..."):
                logs, code = run_pipeline(models_str)
            if code == 0:
                st.success("Pipeline OK")
            else:
                st.error(f"Pipeline FAIL (kod {code})")
            st.session_state["pipeline_logs"] = logs
            st.rerun()

    if btn_tests:
        with st.spinner("Testy..."):
            logs, code = run_tests()
        st.session_state["test_logs"] = logs
        if code == 0:
            st.success("Testy OK")
        else:
            st.error(f"Testy FAIL (kod {code})")
        st.rerun()

    st.divider()

    # Aktywne wyniki — checkboxy do włączania/wyłączania modeli w dashboardzie
    st.subheader("3. Wyniki w dashboardzie")
    all_available = get_available_model_results()
    if all_available:
        enabled_models = []
        for m in all_available:
            on = st.checkbox(m, value=True, key=f"enable_{m}")
            if on:
                enabled_models.append(m)
        # Zapisz do session_state
        st.session_state["enabled_models"] = enabled_models
    else:
        st.text("Brak wynikow")
        st.session_state["enabled_models"] = []


# ──────────────────────────────────────────────
# AKTYWNY MODEL DO PODGLADU
# ──────────────────────────────────────────────
available = st.session_state.get("enabled_models", get_available_model_results())

if not available:
    st.warning("Brak wynikow — uruchom pipeline lub wlacz model w panelu bocznym.")
    if "pipeline_logs" in st.session_state:
        st.code(st.session_state["pipeline_logs"], language="text")
    st.stop()

# Wybór modelu do podglądu
active_model = st.selectbox(
    "Aktywny model (podglad wynikow)",
    available,
    key="active_model",
)

fc, pp, bt, mt = load_model_results(active_model)
tr = load_input("SKU_FeatureTrain*.csv")

if fc is None or pp is None:
    st.error(f"Brak plikow wynikowych dla modelu {active_model}.")
    st.stop()

# Stringify
fc["ProductID"] = fc["ProductID"].astype(str)
pp["ProductID"] = pp["ProductID"].astype(str)
if bt is not None:
    bt["ProductID"] = bt["ProductID"].astype(str)
all_skus = sorted(fc["ProductID"].unique())


# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tabs = ["Executive", "Prognozy", "Plan produkcji", "Backtest", "Metryki"]
if len(available) > 1 or (OUTPUT_DIR / "AFG_ML_MODEL_COMPARISON.csv").exists():
    tabs.append("Porownanie modeli")
tabs.append("Logi")

tab_objects = st.tabs(tabs)
tab_idx = {name: i for i, name in enumerate(tabs)}


# ──────────────────────────────────────────────
# TAB: EXECUTIVE
# ──────────────────────────────────────────────
with tab_objects[tab_idx["Executive"]]:
    st.header(f"Executive Summary — {active_model.upper()}")

    plan_summary = pp.groupby("ProductID").agg(
        FillRate_6M=("FillRate_6M", "first"),
        LostSales_6M=("LostSales_6M", "first"),
    ).reset_index()

    avg_fr = plan_summary["FillRate_6M"].mean()
    total_ls = plan_summary["LostSales_6M"].sum()
    n_sku = len(plan_summary)
    n_risk = (plan_summary["FillRate_6M"] < 0.7).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg FillRate 6M", f"{avg_fr:.1%}")
    c2.metric("Total LostSales 6M", f"{total_ls:,.0f} szt.")
    c3.metric("SKU w planie", n_sku)
    c4.metric("SKU zagrozonych (<70%)", n_risk,
              delta=f"-{n_risk}" if n_risk > 0 else "0", delta_color="inverse")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("FillRate 6M per SKU")
        colors = ["#e74c3c" if fr < 0.5 else "#f39c12" if fr < 0.7 else "#2ecc71"
                  for fr in plan_summary["FillRate_6M"]]
        fig = go.Figure(go.Bar(
            x=plan_summary["ProductID"], y=plan_summary["FillRate_6M"],
            marker_color=colors,
            text=[f"{v:.0%}" for v in plan_summary["FillRate_6M"]],
            textposition="outside",
        ))
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange",
                      annotation_text="Target 70%")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.1],
                          height=400, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("LostSales 6M per SKU")
        fig = go.Figure(go.Bar(
            x=plan_summary["ProductID"], y=plan_summary["LostSales_6M"],
            marker_color="#e74c3c",
            text=[f"{v:,.0f}" for v in plan_summary["LostSales_6M"]],
            textposition="outside",
        ))
        fig.update_layout(height=400, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top SKU Risk")
    risk = plan_summary.sort_values("FillRate_6M").head(5).copy()
    risk["FillRate_6M"] = risk["FillRate_6M"].apply(lambda x: f"{x:.1%}")
    risk["LostSales_6M"] = risk["LostSales_6M"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(risk, use_container_width=True, hide_index=True)

    # LEGENDA
    st.divider()
    with st.expander("LEGENDA", expanded=False):
        st.markdown("""
**Avg FillRate 6M** — jaki % popytu jestesmy w stanie obsluzyc w ciagu 6 miesiecy.
100% = zero brakow, 70% = 30% popytu nieobsluzone. Im wyzej tym lepiej.

**Total LostSales 6M** — ile sztuk lacznie nie sprzedamy z powodu braku towaru
w ciagu 6 miesiecy. Im nizej tym lepiej.

**SKU w planie** — liczba produktow objetych prognoza.

**SKU zagrozonych (<70%)** — ile produktow ma FillRate ponizej 70%.
To czerwone flagi wymagajace natychmiastowej reakcji.

**Wykres FillRate per SKU:**
- Zielony slupek = OK (>=70%)
- Pomaranczowy = uwaga (50-70%)
- Czerwony = krytyczny (<50%)
- Linia przerywana = target 70%

**Wykres LostSales per SKU** — im wyzszy slupek, tym wiecej tracimy
na danym SKU. To priorytet do dzialania.

**Top SKU Risk** — 5 najgorszych produktow wg FillRate.
Od tych zaczynamy interwencje.
""")


# ──────────────────────────────────────────────
# TAB: PROGNOZY
# ──────────────────────────────────────────────
with tab_objects[tab_idx["Prognozy"]]:
    st.header(f"Prognozy — {active_model.upper()}")
    selected_sku = st.selectbox("SKU", all_skus, key="fc_sku")
    sku_fc = fc[fc["ProductID"] == selected_sku].sort_values("Horizon_M")

    hist_data = None
    if tr is not None:
        tr["ProductID"] = tr["ProductID"].astype(str)
        h = tr[tr["ProductID"] == selected_sku].copy()
        if len(h) > 0:
            h["EOM"] = pd.to_datetime(h["EOM"])
            hist_data = h.sort_values("EOM").tail(18)

    col_o, col_s = st.columns(2)
    fc_eom = pd.to_datetime(sku_fc["Forecast_EOM"])

    for col, label, p50c, p90c, hist_col in [
        (col_o, "OrdersIn (popyt)", "Pred_Orders_P50", "Pred_Orders_P90", "KPI_OrdersIn_Qty"),
        (col_s, "Sales (sprzedaz)", "Pred_Sales_P50", "Pred_Sales_P90", "KPI_SalesQty"),
    ]:
        with col:
            st.subheader(label)
            fig = go.Figure()
            if hist_data is not None:
                fig.add_trace(go.Scatter(
                    x=hist_data["EOM"], y=hist_data[hist_col],
                    mode="lines+markers", name="Historia",
                    line=dict(color="#3498db"),
                ))
            fig.add_trace(go.Scatter(
                x=fc_eom, y=sku_fc[p50c],
                mode="lines+markers", name="P50",
                line=dict(color="#2ecc71", dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=fc_eom, y=sku_fc[p90c],
                mode="lines+markers", name="P90",
                line=dict(color="#e74c3c", dash="dot"),
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_eom, fc_eom[::-1]]),
                y=pd.concat([sku_fc[p90c], sku_fc[p50c][::-1]]),
                fill="toself", fillcolor="rgba(231,76,60,0.1)",
                line=dict(width=0), showlegend=False,
            ))
            fig.update_layout(height=400, margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tabela prognoz")
    disp = sku_fc[["ProductID", "Horizon_M", "Forecast_EOM",
                   "Pred_Orders_P50", "Pred_Orders_P90",
                   "Pred_Sales_P50", "Pred_Sales_P90"]].copy()
    disp.columns = ["ProductID", "Horyzont", "Miesiac", "Orders P50", "Orders P90", "Sales P50", "Sales P90"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # LEGENDA
    st.divider()
    with st.expander("LEGENDA", expanded=False):
        st.markdown("""
**OrdersIn (popyt)** — naplyw zamowien od klientow (ile rynek chce kupic).
To glowny sygnal popytu w systemie.

**Sales (sprzedaz)** — faktycznie zrealizowana sprzedaz (z faktur).
Moze byc nizsza niz popyt jesli brakuje towaru.

**Historia** (niebieska linia) — dane rzeczywiste z przeszlosci.

**P50** (zielona przerywana) — prognoza "srodkowa". Z 50% prawdopodobienstwem
rzeczywistosc bedzie wyzsza lub nizsza. Uzywamy do planowania budzetu i baseline produkcji.

**P90** (czerwona kropkowana) — prognoza "pesymistyczna". Z 90% prawdopodobienstwem
rzeczywistosc NIE przekroczy tej wartosci. Uzywamy do planowania buforow bezpieczenstwa
i zapasow.

**Pasmo czerwone** (P50-P90) — zakres niepewnosci prognozy. Im szersze pasmo,
tym mniej pewna prognoza. Normalne jest ze pasmo rosnie z horyzontem (dalej = trudniej).

**Horyzont** — za ile miesiecy (1 = przyszly miesiac, 6 = za pol roku).

**Tabela prognoz** — wartosci liczbowe P50/P90 dla kazdego miesiaca w horyzoncie 6M.
""")


# ──────────────────────────────────────────────
# TAB: PLAN PRODUKCJI
# ──────────────────────────────────────────────
with tab_objects[tab_idx["Plan produkcji"]]:
    st.header(f"Plan produkcji — {active_model.upper()}")
    selected_sku_p = st.selectbox("SKU", all_skus, key="pp_sku")
    sku_pp = pp[pp["ProductID"] == selected_sku_p].sort_values("Horizon_M")

    fr_val = sku_pp["FillRate_6M"].iloc[0]
    ls_val = sku_pp["LostSales_6M"].iloc[0]
    lt_val = sku_pp["LT_LeadTimeMonths_Used"].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("FillRate 6M", f"{fr_val:.1%}")
    c2.metric("LostSales 6M", f"{ls_val:,.0f} szt.")
    c3.metric("Lead Time", f"{lt_val} mies.")

    st.divider()
    fc_eom_p = pd.to_datetime(sku_pp["Forecast_EOM"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=fc_eom_p, y=sku_pp["RecommendedProductionQty"],
                         name="Rec. Production", marker_color="#3498db", opacity=0.7))
    fig.add_trace(go.Scatter(x=fc_eom_p, y=sku_pp["Demand_P90"],
                             mode="lines+markers", name="Demand P90",
                             line=dict(color="#e74c3c", width=3)))
    fig.add_trace(go.Scatter(x=fc_eom_p, y=sku_pp["Availability"],
                             mode="lines+markers", name="Availability",
                             line=dict(color="#2ecc71", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=fc_eom_p, y=sku_pp["Served"],
                             mode="lines+markers", name="Served",
                             line=dict(color="#f39c12", width=2)))
    fig.update_layout(barmode="overlay", height=450, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rekomendacje per miesiac")
    disp_pp = sku_pp[["ProductID", "Horizon_M", "Forecast_EOM", "Demand_P90",
                      "RecommendedProductionQty", "ActionLabel",
                      "Availability", "Served", "LostSales_M"]].copy()
    disp_pp.columns = ["ProductID", "Horyzont", "Miesiac", "Demand P90", "Rec. Produkcja",
                       "Akcja", "Dostepnosc", "Obsluzono", "Lost Sales"]

    def color_action(val):
        colors = {"STOP": "#e74c3c", "SLOW": "#f39c12", "SPEED": "#2ecc71"}
        c = colors.get(val, "")
        return f"background-color: {c}; color: white" if c else ""

    st.dataframe(disp_pp.style.map(color_action, subset=["Akcja"]),
                 use_container_width=True, hide_index=True)

    # LEGENDA
    st.divider()
    with st.expander("LEGENDA", expanded=False):
        st.markdown("""
**FillRate 6M** — jaki % popytu obsluzymy dla tego SKU w horyzoncie 6 miesiecy.
100% = pelna dostepnosc, <50% = powazny problem.

**LostSales 6M** — ile sztuk nie sprzedamy z powodu braku towaru (suma 6 miesiecy).

**Lead Time** — ile miesiecy trwa produkcja tego SKU (z pola CYKL_PROD_DNI).
Kluczowe: jesli LT = 8 miesiecy, to w horyzoncie 6M nic co dzis uruchomimy
nie zdazy dotrzec — stad niski FillRate dla takich SKU.

**Wykres:**
- Niebieski slupek (**Rec. Production**) — ile sztuk rekomendujemy wyprodukowac.
- Czerwona linia (**Demand P90**) — ile rynek moze chciec (wariant pesymistyczny).
- Zielona przerywana (**Availability**) — ile faktycznie bedzie dostepne
  (zapas FG + produkcja ktora dotarla po lead time).
- Pomaranczowa (**Served**) — ile faktycznie obsluzymy (minimum z popytu i dostepnosci).

**Etykiety akcji:**
- **SPEED** (zielona) — przyspiesz/utrzymaj produkcje (>= 80% historycznej sredniej).
- **SLOW** (pomaranczowa) — spowolnij produkcje (20-80% sredniej).
- **STOP** (czerwona) — wstrzymaj produkcje (< 20% sredniej).

**Demand P90** — prognozowany popyt w wariancie pesymistycznym (z prognozy P90).
Planujemy produkcje pod ten scenariusz, zeby minimalizowac ryzyko brakow.

**Dostepnosc** — stan FG na poczatku miesiaca + produkcja ktora dotarla.
Jesli dostepnosc < demand, powstaje utracona sprzedaz (Lost Sales).
""")

    st.divider()
    st.subheader("Przeglad wszystkich SKU")
    overview = pp.groupby("ProductID").agg(
        Lead_Time=("LT_LeadTimeMonths_Used", "first"),
        Avg_Demand_P90=("Demand_P90", "mean"),
        Avg_RecProd=("RecommendedProductionQty", "mean"),
        FillRate_6M=("FillRate_6M", "first"),
        LostSales_6M=("LostSales_6M", "first"),
    ).reset_index().sort_values("FillRate_6M")
    overview["Avg_Demand_P90"] = overview["Avg_Demand_P90"].round(1)
    overview["Avg_RecProd"] = overview["Avg_RecProd"].round(1)
    overview["FillRate_6M"] = overview["FillRate_6M"].apply(lambda x: f"{x:.1%}")
    st.dataframe(overview, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# TAB: BACKTEST
# ──────────────────────────────────────────────
with tab_objects[tab_idx["Backtest"]]:
    st.header(f"Rolling Backtest — {active_model.upper()}")
    if bt is None:
        st.warning("Brak danych backtestowych.")
    else:
        bt["Run_Cutoff_EOM"] = pd.to_datetime(bt["Run_Cutoff_EOM"])
        bt["Forecast_EOM"] = pd.to_datetime(bt["Forecast_EOM"])
        target_choice = st.radio("Target", ["orders", "sales"], horizontal=True, key="bt_target")
        bt_t = bt[bt["Target"] == target_choice]

        if "Predicted_P50" in bt_t.columns and "Actual" in bt_t.columns:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Actual vs Predicted (P50)")
                fig = px.scatter(bt_t, x="Actual", y="Predicted_P50",
                                 color="Horizon_M", color_continuous_scale="Viridis", opacity=0.6)
                mx = max(bt_t["Actual"].max(), bt_t["Predicted_P50"].max())
                fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines",
                                         line=dict(color="red", dash="dash"), showlegend=False))
                fig.update_layout(height=450, margin=dict(t=30))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader("Rozklad bledow P50")
                errors = bt_t["Predicted_P50"] - bt_t["Actual"]
                fig = px.histogram(errors, nbins=50,
                                   labels={"value": "Error (Pred - Actual)"})
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                fig.update_layout(height=450, margin=dict(t=30), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("WAPE per cutoff")
            bt_ae = bt_t.copy()
            bt_ae["AE"] = np.abs(bt_ae["Actual"] - bt_ae["Predicted_P50"])
            wape_t = bt_ae.groupby("Run_Cutoff_EOM").apply(
                lambda g: g["AE"].sum() / max(g["Actual"].abs().sum(), 1),
                include_groups=False,
            ).reset_index()
            wape_t.columns = ["Cutoff", "WAPE"]
            fig = px.line(wape_t, x="Cutoff", y="WAPE", markers=True)
            fig.update_layout(yaxis_tickformat=".1%", height=350, margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Surowe dane"):
            st.dataframe(bt_t.head(200), use_container_width=True, hide_index=True)

    # LEGENDA
    st.divider()
    with st.expander("LEGENDA", expanded=False):
        st.markdown("""
**Backtest (rolling backtest)** — test historyczny: model uczy sie na danych
do pewnego momentu (cutoff), prognozuje do przodu, i porownujemy z tym
co sie naprawde stalo. Powtarzamy dla wielu cutoffow.

**Actual vs Predicted (P50):**
- Kazdy punkt = 1 prognoza dla 1 SKU w 1 miesiacu.
- Czerwona linia = idealna prognoza (predicted = actual).
- Im blizej punkty leza linii, tym lepszy model.
- Kolor = horyzont (jasniejszy = dalszy, zwykle mniej dokladny).

**Rozklad bledow P50:**
- Histogram bledow (prognoza minus rzeczywistosc).
- Centrowany na 0 = brak systematycznego bledu.
- Przesuniety w lewo = model zaniza (underprediction).
- Przesuniety w prawo = model zawyza (overprediction).

**WAPE per cutoff:**
- Jak zmienia sie jakosc prognozy w czasie.
- Stabilna linia = model konsekwentny.
- Skoki = dane sie gwaltownie zmienily (sezon, kryzys) lub model jest niestabilny.

**Target: orders vs sales:**
- orders = prognoza naplywu zamowien (OrdersIn).
- sales = prognoza sprzedazy zrealizowanej (SalesQty).
""")


# ──────────────────────────────────────────────
# TAB: METRYKI
# ──────────────────────────────────────────────
with tab_objects[tab_idx["Metryki"]]:
    st.header(f"Metryki — {active_model.upper()}")
    if mt is None:
        st.warning("Brak metryk.")
    else:
        c50, c90 = st.columns(2)
        with c50:
            st.subheader("P50 (dokladnosc)")
            mt_p50 = mt[mt["Quantile"] == "P50"]
            fig = px.bar(mt_p50, x="Horizon_M", y="WAPE", color="Target",
                         barmode="group", text_auto=".1%")
            fig.update_layout(yaxis_tickformat=".0%", height=350, margin=dict(t=30),
                              xaxis_title="Horyzont")
            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(mt_p50, x="Horizon_M", y="MAE", color="Target",
                         barmode="group", text_auto=".1f")
            fig.update_layout(height=350, margin=dict(t=30), xaxis_title="Horyzont")
            st.plotly_chart(fig, use_container_width=True)

        with c90:
            st.subheader("P90 (bufor serwisowy)")
            mt_p90 = mt[mt["Quantile"] == "P90"]
            fig = px.bar(mt_p90, x="Horizon_M", y="Coverage", color="Target",
                         barmode="group", text_auto=".1%")
            fig.add_hline(y=0.9, line_dash="dash", line_color="red",
                          annotation_text="Target 90%")
            fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0.7, 1.0],
                              height=350, margin=dict(t=30), xaxis_title="Horyzont")
            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(mt_p90, x="Horizon_M", y="Pinball", color="Target",
                         barmode="group", text_auto=".2f")
            fig.update_layout(height=350, margin=dict(t=30), xaxis_title="Horyzont")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        mt_display = mt.copy()
        if "Model" not in mt_display.columns:
            mt_display.insert(0, "Model", active_model)
        st.dataframe(mt_display, use_container_width=True, hide_index=True)

    # LEGENDA
    st.divider()
    with st.expander("LEGENDA", expanded=False):
        st.markdown("""
**P50 — metryki dokladnosci (lewa kolumna):**

- **WAPE** (Weighted Absolute Percentage Error) — procentowy blad prognozy wazony wolumenem.
  <10% = bardzo dobra, 10-20% = dobra, 20-30% = akceptowalna, >30% = slaba.
  Normalnie rosnie z horyzontem (dalej = trudniej prognozowac).

- **MAE** (Mean Absolute Error) — sredni blad w sztukach.
  Np. MAE=5 znaczy: srednio mylimy sie o 5 szt. w kazdym miesiacu.

**P90 — metryki bufora serwisowego (prawa kolumna):**

- **Coverage** — w ilu % przypadkow rzeczywistosc zmieszcila sie ponizej prognozy P90.
  Cel = ~90% (strzalka na wykresie).
  Jesli 100% = prognoza za wysoka (marnujemy zapas na zbyt duzy bufor).
  Jesli <80% = bufor za maly (zbyt czesto braknie towaru).

- **Pinball** (Pinball Loss) — techniczny wskaznik kalibracji kwantylu.
  Im nizszy tym lepsza kalibracja P90. Na co dzien nie trzeba sie nim przejmowac,
  ale jesli jest duzy — P90 jest zle skalibrowane.

**Horyzont** — os X na wykresach. h=1 to prognoza na nastepny miesiac,
h=6 to prognoza na za pol roku.

**Target: orders vs sales** — kolory na wykresach rozrozniaja
prognozę popytu (orders) od prognozy sprzedazy (sales).
""")


# ──────────────────────────────────────────────
# TAB: POROWNANIE MODELI
# ──────────────────────────────────────────────
if "Porownanie modeli" in tab_idx:
    with tab_objects[tab_idx["Porownanie modeli"]]:
        st.header("Porownanie modeli")

        comp_path = OUTPUT_DIR / "AFG_ML_MODEL_COMPARISON.csv"
        if comp_path.exists():
            comp = pd.read_csv(comp_path)
        else:
            parts = []
            for m in available:
                mp = OUTPUT_DIR / f"AFG_ML_METRICS_{m}.csv"
                if mp.exists():
                    d = pd.read_csv(mp)
                    if "Model" not in d.columns:
                        d["Model"] = m
                    parts.append(d)
            comp = pd.concat(parts, ignore_index=True) if parts else None

        # Filtruj tylko wlaczone modele
        if comp is not None and "Model" in comp.columns:
            comp = comp[comp["Model"].isin(available)]

        if comp is not None and "Model" in comp.columns and comp["Model"].nunique() > 1:
            models_in_comp = sorted(comp["Model"].unique())
            st.info(f"Modele: {', '.join(models_in_comp)}")

            # WAPE P50 porownanie
            st.subheader("WAPE P50 per model i horyzont")
            comp_p50 = comp[comp["Quantile"] == "P50"]

            for target in ["orders", "sales"]:
                st.markdown(f"**{target.upper()}**")
                tdata = comp_p50[comp_p50["Target"] == target]
                fig = px.bar(tdata, x="Horizon_M", y="WAPE", color="Model",
                             barmode="group", text_auto=".1%")
                fig.update_layout(yaxis_tickformat=".0%", height=350,
                                  margin=dict(t=30), xaxis_title="Horyzont")
                st.plotly_chart(fig, use_container_width=True)

            # Coverage P90
            st.subheader("Coverage P90 per model")
            comp_p90 = comp[comp["Quantile"] == "P90"]
            for target in ["orders", "sales"]:
                st.markdown(f"**{target.upper()}**")
                tdata = comp_p90[comp_p90["Target"] == target]
                fig = px.bar(tdata, x="Horizon_M", y="Coverage", color="Model",
                             barmode="group", text_auto=".1%")
                fig.add_hline(y=0.9, line_dash="dash", line_color="red")
                fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0.6, 1.0],
                                  height=350, margin=dict(t=30), xaxis_title="Horyzont")
                st.plotly_chart(fig, use_container_width=True)

            # Tabela zbiorcza
            st.subheader("Podsumowanie (sredni WAPE P50)")
            summary = comp_p50.groupby("Model")["WAPE"].mean().reset_index()
            summary.columns = ["Model", "Sredni WAPE P50"]
            summary["Sredni WAPE P50"] = summary["Sredni WAPE P50"].apply(lambda x: f"{x:.2%}")
            summary = summary.sort_values("Sredni WAPE P50")
            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Pelna tabela porownawcza")
            # Upewnij sie ze Model jest pierwsza kolumna
            comp_display = comp.copy()
            if "Model" in comp_display.columns:
                cols = ["Model"] + [c for c in comp_display.columns if c != "Model"]
                comp_display = comp_display[cols]
            st.dataframe(comp_display, use_container_width=True, hide_index=True)

            # LEGENDA
            st.divider()
            with st.expander("LEGENDA", expanded=False):
                st.markdown("""
**Porownanie modeli** — zestawienie wynikow backtestowych roznych algorytmow ML
uruchomionych na tych samych danych. Pozwala wybrac najlepszy model.

**Dostepne modele:**
- **LightGBM** — zaawansowany model gradient boosting. Uzywa pelnego zestawu
  features (rolling, YoY, backlog, lead time, sezonowosc). Zwykle najdokladniejszy.
- **ARIMA** — klasyczny model szeregu czasowego (baseline). Uzywa tylko historii
  danego SKU, bez dodatkowych features. Sluzy jako punkt odniesienia.

**WAPE P50 per model i horyzont:**
- Slupki obok siebie. Nizszy slupek = lepszy model dla danego horyzontu.
- LightGBM powinien wygrywac (ma features), ARIMA to baseline do porownania.

**Coverage P90 per model:**
- Czerwona linia = target 90%.
- Model blizej 90% jest lepiej skalibrowany.
- ARIMA czesto ma ~100% (za szerokie przedzialy — "bezpiecznie ale niedokladnie").

**Podsumowanie (sredni WAPE P50):**
- Jedna liczba per model. Im nizsza, tym lepszy model ogolnie.
- To glowne kryterium wyboru modelu do produkcji.
""")

        elif comp is not None:
            st.info("Uruchom pipeline z wieloma modelami, aby zobaczyc porownanie.")
            st.dataframe(comp, use_container_width=True, hide_index=True)
        else:
            st.info("Brak danych do porownania. Uruchom pipeline z 2+ modelami.")


# ──────────────────────────────────────────────
# TAB: LOGI
# ──────────────────────────────────────────────
with tab_objects[tab_idx["Logi"]]:
    st.header("Logi uruchomien")
    if "pipeline_logs" in st.session_state:
        st.subheader("Pipeline")
        st.code(st.session_state["pipeline_logs"], language="text")
    else:
        st.info("Brak logow pipeline.")

    if "test_logs" in st.session_state:
        st.subheader("Testy")
        st.code(st.session_state["test_logs"], language="text")
    else:
        st.info("Brak logow testow.")

    # LEGENDA
    st.divider()
    with st.expander("LEGENDA", expanded=False):
        st.markdown("""
**Logi Pipeline** — pelny log ostatniego uruchomienia pipeline ML:
- Jakie pliki wczytano i ile wierszy/kolumn maja.
- Ile modeli wytrenowano.
- Wyniki backtestow (metryki per horyzont).
- Prognozy finalne i plan produkcji per SKU.
- Przydatne do diagnostyki bledow — jesli pipeline FAIL, szukaj tu szczegolow.

**Logi Testow** — wynik 70 automatycznych testow walidacyjnych:
- **PASS** = test przeszedl pomyslnie.
- **FAIL** = test nie przeszedl — wymaga interwencji.
- **WARN** = ostrzezenie — wartosc w granicach, ale blisko limitu.

Testy weryfikuja: schematy plikow (vs Runbook), poprawnosc danych
(brak NaN, zakresy wartosci), jakosc prognoz (WAPE, Coverage),
zgodnosc z targetami historycznymi (FillRate/LostSales z Train),
stabilnosc backtestowa (brak dryfu) i powtarzalnosc (random_state).
""")
