import pandas as pd
import numpy as np
import sqlite3

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
        'campaign': np.random.randint(1, 5, n_clients),
        'previous': np.random.choice([0, 1, 2, 3], n_clients, p=[0.8, 0.1, 0.05, 0.05]),
        'poutcome': np.random.choice(poutcome, n_clients, p=[0.1, 0.8, 0.1]),

        'contact_status': ['to_call'] * n_clients
    }
    df = pd.DataFrame(data)
    return df

def save_to_sqlite(df, db_path='clients.db'):
    conn = sqlite3.connect(db_path)
    df.to_sql('clients', conn, if_exists='replace', index=False)
    conn.close()

if __name__ == "__main__":
    db_filename = 'clients.db'
    clients_df = generate_mock_clients(100)
    save_to_sqlite(clients_df, db_filename)