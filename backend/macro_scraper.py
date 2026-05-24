import requests
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class EurostatScraper:
    def __init__(self):
        self.month_map = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        self.base_url = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"

    def _fetch_eurostat_value(self, dataset_code: str, params: dict, fallback_val: float) -> float:
        """
        Uniwersalna metoda do odpytywania API Eurostatu.
        Zwraca wartość float lub fallback_val w przypadku błędu.
        """
        url = f"{self.base_url}{dataset_code}"
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            values_dict = data.get('value', {})
            if values_dict:
                first_value = list(values_dict.values())[0]
                return float(first_value)
            else:
                logger.warning(f"Brak danych w odpowiedzi dla {dataset_code}.")
                return fallback_val

        except Exception as e:
            logger.error(f"Błąd pobierania {dataset_code}: {e}")
            return fallback_val

    def get_all_macro(self, year: int, month_str: str) -> dict:
        """
        Pobiera komplet 3 zmiennych makroekonomicznych dla podanej daty.
        """
        month_num = self.month_map.get(month_str.lower(), '01')

        time_period = f"{year}-{month_num}"

        logger.info(f"--- Rozpoczynam pobieranie danych makro dla: {month_str} {year} ---")

        euribor_params = {
            "format": "JSON",
            "GEO": "EA",
            "INT_RT": "IRT_M3",
            "TIME": time_period
        }
        euribor_val = self._fetch_eurostat_value("irt_st_m", euribor_params, fallback_val=1.0)
        logger.info(f"Euribor 3M: {euribor_val}")

        # 2. CPI
        cpi_params = {
            "format": "JSON",
            "geo": "PT",
            "coicop": "CP00",
            "unit": "I15",
            "time": time_period
        }
        cpi_val = self._fetch_eurostat_value("prc_hicp_midx", cpi_params, fallback_val=93.2)
        logger.info(f"CPI: {cpi_val}")

        # 3. CCI
        cci_params = {
            "format": "JSON",
            "geo": "PT",
            "indic": "BS-CSMCI",
            "s_adj": "SA",
            "time": time_period
        }
        cci_val = self._fetch_eurostat_value("ei_bsco_m", cci_params, fallback_val=-40.0)
        logger.info(f"CCI: {cci_val}")

        return {
            'euribor3m': euribor_val,
            'cons.price.idx': cpi_val,
            'cons.conf.idx': cci_val
        }