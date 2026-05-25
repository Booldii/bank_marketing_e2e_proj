# CRM Bank ML & Analytics

Projekt akademicki realizujący skonteneryzowany system CRM dla banku z modułem predictive analytics. Krótko mówiąc: łączymy FastAPI, Streamlita i Postgresa w Dockerze, żeby przewidywać zachowania klientów w czasie rzeczywistym za pomocą modeli ML.
Całość jest w pełni skonteneryzowana i gotowa do wdrożenia w środowisku chmurowym.

##  Stack Technologiczny

* **Frontend:** Streamlit (Panel analityczny i interfejs doradcy)
* **Backend:** FastAPI (Szybkie API RESTful)
* **Baza Danych:** PostgreSQL
* **Machine Learning:** Scikit-Learn, Pandas (Modele decyzyjne m.in. Random Forest)
* **DevOps / Wdrożenie:** Docker, Docker Compose, Linux (GCP Compute Engine)

## Architektura Systemu

Aplikacja składa się z trzech niezależnych kontenerów połączonych wewnętrzną siecią Dockera:
1.  **Frontend (Port 8599):** Interfejs użytkownika komunikujący się z API.
2.  **Backend (Port 8000):** Serwer API obsługujący logikę biznesową i predykcje z modelu ML.
3.  **Baza Danych (Wewnętrzny 5432 / Zewnętrzny 5433):** Magazyn danych klientów. Port bazy jest celowo odizolowany i wystawiony na zewnątrz wyłącznie do celów deweloperskich.