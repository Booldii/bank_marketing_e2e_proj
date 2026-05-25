import streamlit as st
import requests
import pandas as pd
from datetime import date
import plotly.graph_objects as go


API_URL = "http://localhost:8000"

st.set_page_config(page_title="CRM - Bank ML", layout="wide")

st.sidebar.header("Ustawienia Systemu")
st.sidebar.markdown("Wybierz datę, dla której chcesz przeprowadzić symulację połączeń.")

sim_date = st.sidebar.date_input(
    "Dzisiejsza data (symulacja):",
    min_value=date(2011, 1, 1),
    max_value=date(2012, 12, 31),
    value=date(2012, 5, 15)
)

sim_year = sim_date.year
sim_month = sim_date.strftime("%b").lower()

days_map = {0: 'mon', 1: 'tue', 2: 'wed', 3: 'thu', 4: 'fri', 5: 'sat', 6: 'sun'}
sim_day = days_map[sim_date.weekday()]

st.sidebar.info(f"Parametry przekazywane do modelu:\n* **Rok:** {sim_year}\n* **Miesiąc:** {sim_month.capitalize()}")

st.title("Panel Telemarketera (ML Scoring)")

try:
    response = requests.get(f"{API_URL}/clients")
    if response.status_code == 200:
        clients_data = response.json()["clients"]
    else:
        clients_data = []
        st.error("Błąd pobierania bazy klientów z serwera.")
except requests.exceptions.ConnectionError:
    clients_data = []
    st.error("Brak połączenia z API. Upewnij się, że backend (FastAPI) jest uruchomiony na porcie 8000.")

if not clients_data:
    st.warning("Brak klientów o statusie 'Do obdzwonienia' w bazie.")
else:
    client_options = {
        c["client_id"]: f"Klient #{c['client_id']} | Wiek: {c['age']} | Zawód: {c['job']} | Edukacja: {c['education']}"
        for c in clients_data
    }

    selected_id = st.selectbox(
        "Wybierz klienta do analizy:",
        options=list(client_options.keys()),
        format_func=lambda x: client_options[x]
    )

    selected_client = next(c for c in clients_data if c["client_id"] == selected_id)

    if st.button("🔍 Oblicz szansę na sukces i pobierz dane rynkowe", type="primary"):
        with st.spinner("Odpytywanie modelu ML oraz pobieranie danych makroekonomicznych z Eurostatu..."):

            payload = selected_client.copy()
            payload.pop("client_id", None)
            payload.pop("contact_status", None)

            payload["year"] = sim_year
            payload["month"] = sim_month
            payload["day_of_week"] = sim_day

            pred_response = requests.post(f"{API_URL}/predict", json=payload)

            if pred_response.status_code == 200:
                result = pred_response.json()
                prob = result["probability"]
                macro = result["macro_used"]
                shap_exp = result["shap_explanation"]

                st.markdown("---")

                col1, col2, col3 = st.columns([1, 2, 1.5])

                with col1:
                    st.subheader("Scoring")
                    if prob > 0.4:
                        st.success(f"Szansa: {prob:.1%}")
                    elif prob > 0.2:
                        st.warning(f"Szansa: {prob:.1%}")
                    else:
                        st.error(f"Szansa: {prob:.1%}")

                with col2:
                    st.subheader("Analiza decyzji modelu (SHAP)")

                    all_impacts = shap_exp["positive"] + shap_exp["negative"]
                    if all_impacts:
                        labels = [f"{i['feature']} ({i['value']})" for i in all_impacts]
                        values = [i['impact'] for i in all_impacts]
                        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

                        fig = go.Figure(go.Bar(
                            x=values,
                            y=labels,
                            orientation='h',
                            marker_color=colors,
                            text=[f"{v:+.3f}" for v in values],
                            textposition='auto',
                        ))

                        fig.update_layout(
                            height=300,
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Siła wpływu na decyzję",
                            yaxis=dict(autorange="reversed")
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Model nie znalazł dominujących czynników dla tego klienta.")

                with col3:
                    st.subheader("Sytuacja Rynkowa (Eurostat)")
                    st.info(f"""
                    **Wskaźniki dla: {sim_month.capitalize()} {sim_year}**
                    * **Euribor 3M:** {macro['euribor3m']}%
                    * **Inflacja (CPI):** {macro['cons.price.idx']} pkt
                    * **Ufność (CCI):** {macro['cons.conf.idx']} pkt
                    """)

                st.markdown("---")

                st.write("### Zarejestruj wynik połączenia:")
                f_col1, f_col2, _ = st.columns([1, 1, 4])

                with f_col1:
                    if st.button("✅ Sukces (Założono lokatę)", use_container_width=True):
                        requests.post(f"{API_URL}/feedback", json={"client_id": selected_id, "result": "Sukces"})
                        st.success("Zapisano! Odświeżam bazę...")
                        st.rerun()

                with f_col2:
                    if st.button("❌ Odmowa", use_container_width=True):
                        requests.post(f"{API_URL}/feedback", json={"client_id": selected_id, "result": "Odmowa"})
                        st.warning("Zapisano. Odświeżam bazę...")
                        st.rerun()
            else:
                st.error(f"Błąd predykcji. Szczegóły: {pred_response.text}")
