import streamlit as st
import requests
import pandas as pd
from datetime import date
import plotly.graph_objects as go
import os
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

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

st.sidebar.markdown("---")
st.sidebar.subheader("Obecna Sytuacja Rynkowa")

try:
    macro_res = requests.get(f"{API_URL}/macro", params={"year": sim_year, "month": sim_month}, timeout=3)
    if macro_res.status_code == 200:
        macro_data = macro_res.json()["macro"]

        # Wyświetlamy estetyczne kafelki metryk w panelu bocznym
        st.sidebar.metric(label="Euribor 3M", value=f"{macro_data['euribor3m']}%")
        st.sidebar.metric(label="Wskaźnik cen konsumpcyjnych (CPI)", value=f"{macro_data['cons.price.idx']} pkt")
        st.sidebar.metric(label="Wskaźnik ufności konsumenckiej (CCI)", value=f"{macro_data['cons.conf.idx']} pkt")
    else:
        st.sidebar.error("Brak danych makro z Eurostatu.")
except requests.exceptions.RequestException:
    st.sidebar.warning("Brak połączenia z API dla danych makro.")

st.title("Panel Telemarketera")

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
    st.warning("Brak klientów o statusie 'to_call' w bazie.")
else:
    client_options = {
        c["client_id"]: f"Klient #{c['client_id']} | Wiek: {c['age']} | Zawód: {c['job']} | Edukacja: {c['education']}"
        for c in clients_data
    }

    selected_id = st.selectbox(
        "Wybierz klienta do obdzwonienia:",
        options=list(client_options.keys()),
        format_func=lambda x: client_options[x]
    )

    selected_client = next(c for c in clients_data if c["client_id"] == selected_id)

    st.markdown("---")

    c_info, c_action = st.columns([2, 1])

    with c_info:
        st.subheader("Kartoteka Klienta")
        st.write(
            f"**Wiek:** {selected_client['age']} | **Praca:** {selected_client['job'].capitalize()} | **Edukacja:** {selected_client['education']}")
        st.write(f"**Historia kredytowa (housing/loan):** {selected_client['housing']} / {selected_client['loan']}")
        st.write(f"**Poprzednia kampania:** {selected_client['poutcome']}")

    with c_action:
        st.subheader("Wynik Rozmowy")
        if st.button("✅ Sukces (Lokata)", width='stretch'):
            requests.post(f"{API_URL}/feedback", json={"client_id": selected_id, "result": "Success"})
            st.success("Zapisano! Odświeżam...")
            time.sleep(0.5)
            st.rerun()

        if st.button("❌ Odmowa", width='stretch'):
            requests.post(f"{API_URL}/feedback", json={"client_id": selected_id, "result": "Failure"})
            st.warning("Odmowa. Odświeżam...")
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")

    if st.button("🔍 Uruchom Asystenta ML (Scoring & SHAP)"):
        with st.spinner("Odpytywanie Eurostatu i kalkulacja modelu..."):

            payload = selected_client.copy()
            payload.pop("client_id", None)
            payload.pop("contact_status", None)
            payload["year"] = sim_year
            payload["month"] = sim_month
            payload["day_of_week"] = sim_day

            payload["macro_data"] = macro_data

            pred_response = requests.post(f"{API_URL}/predict", json=payload)

            if pred_response.status_code == 200:
                result = pred_response.json()
                prob = result["probability"]
                macro = result["macro_used"]
                shap_exp = result["shap_explanation"]

                ml_col1, ml_col2 = st.columns([1, 2])

                with ml_col1:
                    st.metric("Szansa na Sukces", f"{prob:.1%}")

                with ml_col2:
                    all_impacts = shap_exp["positive"] + shap_exp["negative"]
                    if all_impacts:
                        labels = [f"{i['feature']} ({i['value']})" for i in all_impacts]
                        values = [i['impact'] for i in all_impacts]
                        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

                        fig = go.Figure(go.Bar(
                            x=values, y=labels, orientation='h',
                            marker_color=colors, text=[f"{v:+.3f}" for v in values], textposition='auto'
                        ))
                        fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig, width='stretch')
            else:
                st.error("Błąd usługi ML.")
