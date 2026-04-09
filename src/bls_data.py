import requests
import pandas as pd
import joblib
import os

def fetch_bls_turnover():
    """
    Fetches real US quit rates from Bureau of Labor Statistics.
    JOLTS survey — updated monthly, free, no API key needed for basic use.
    Falls back to hardcoded 2024 data if API is unavailable.
    """

    # Fallback data in case BLS API is down or rate limited
    fallback_data = {
        'All Industries':           2.2,
        'Manufacturing':            1.7,
        'Retail Trade':             3.1,
        'Professional Services':    2.4,
        'Healthcare':               2.0,
        'Finance & Insurance':      1.6,
    }

    try:
        url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        payload = {
            "seriesid": [
                "JTS000000000000000QUR",
                "JTS320000000000000QUR",
                "JTS510000000000000QUR",
                "JTS540099000000000QUR",
            ],
            "startyear": "2023",
            "endyear":   "2024",
        }

        response = requests.post(url, json=payload, timeout=10)
        results  = response.json().get('Results', {}).get('series', [])

        labels = {
            "JTS000000000000000QUR": "All Industries",
            "JTS320000000000000QUR": "Manufacturing",
            "JTS510000000000000QUR": "Retail Trade",
            "JTS540099000000000QUR": "Professional Services",
        }

        records = []
        for series in results:
            label = labels.get(series['seriesID'], series['seriesID'])
            for row in series['data']:
                if row['period'] != 'M13':
                    records.append({
                        'Industry':  label,
                        'Year':      int(row['year']),
                        'Period':    row['period'],
                        'QuitRate':  float(row['value'])
                    })

        if not records:
            raise ValueError("Empty response from BLS API")

        df = pd.DataFrame(records)

        # Get latest annual average per industry
        latest = df.groupby('Industry')['QuitRate'].mean().round(2).to_dict()

        os.makedirs('models', exist_ok=True)
        joblib.dump(latest, 'models/bls_data.pkl')
        print("BLS data fetched successfully from API")
        return latest

    except Exception as e:
        print(f"BLS API unavailable ({e}), using fallback data")
        joblib.dump(fallback_data, 'models/bls_data.pkl')
        return fallback_data

def load_bls_data():
    if os.path.exists('models/bls_data.pkl'):
        return joblib.load('models/bls_data.pkl')
    return fetch_bls_turnover()

if __name__ == '__main__':
    data = fetch_bls_turnover()
    print("\nNational Quit Rates:")
    for industry, rate in data.items():
        print(f"  {industry}: {rate}%")