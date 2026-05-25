import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def generate_mock_clients(n_clients=100):
    np.random.seed(42)

    jobs = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
            'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed']
    marital = ['divorced', 'married', 'single']
    education = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
                 'professional.course', 'university.degree']
    contact_types = ['cellular', 'telephone']
    poutcome = ['failure', 'nonexistent', 'success']
    yes_no = ['no', 'yes']

    data = {
        'client_id': range(1, n_clients + 1),
        'age': np.random.randint(18, 85, n_clients),
        'job': np.random.choice(jobs, n_clients),
        'marital': np.random.choice(marital, n_clients, p=[0.1, 0.6, 0.3]),
        'education': np.random.choice(education, n_clients),
        'housing': np.random.choice(yes_no, n_clients, p=[0.45, 0.55]),
        'loan': np.random.choice(yes_no, n_clients, p=[0.8, 0.2]),
        'contact': np.random.choice(contact_types, n_clients, p=[0.8, 0.2]),
        'campaign': np.random.randint(0, 4, n_clients),
        'previous': np.random.choice([0, 1, 2, 3], n_clients, p=[0.8, 0.1, 0.05, 0.05]),
        'poutcome': np.random.choice(poutcome, n_clients, p=[0.1, 0.8, 0.1]),

        'contact_status': ['to_call'] * n_clients
    }
    df = pd.DataFrame(data)
    df["poutcome"] = np.where(df["previous"] == 0, "nonexistent", df["poutcome"])
    df["previous"] = np.where(df["poutcome"] == "nonexistent", 0, df["previous"])
    return df

def save_to_postgres(df):
    db_url = "postgresql://bank_admin:supersecret@db:5432/crm_database"
    try:
        engine = create_engine(db_url)
        df.to_sql('clients', engine, if_exists='replace', index=False)
        print("Saved client date in PostgreSQL.")
    except Exception as e:
        print(f"Save error: {e}")

if __name__ == "__main__":
    clients_df = generate_mock_clients(100)
    save_to_postgres(clients_df)