import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def process_data(file_input):
    """Carga y procesa el RFM desde el archivo local o cargado."""
    # Soporta tanto el path (string) como el objeto de Streamlit
    df = pd.read_excel(file_input)

    # Limpieza rápida de columnas (eliminar espacios)
    df.columns = [col.strip() for col in df.columns]

    # Ingeniería de variables básica
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']

    # Cálculo de métricas RFM
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSum': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSum': 'Monetary'})

    # Tratamiento de Outliers (Percentil 99) para que el gráfico 3D sea legible
    rfm = rfm[rfm['Monetary'] < rfm['Monetary'].quantile(0.99)]
    
    return rfm

def apply_segmentation(rfm, n_clusters=4):
    """Aplica K-Means con estabilización de clústeres."""
    # Log transform para normalizar distribuciones retail
    rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_log)

    # Entrenamiento del modelo
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    rfm['Cluster_Raw'] = model.fit_predict(scaled)

    # ESTABILIZACIÓN: Ordenar clústeres por 'Monetary' (Gasto)
    # Así, el Clúster 3 siempre será el de mayor gasto (VIP)
    order = rfm.groupby('Cluster_Raw')['Monetary'].mean().sort_values().index
    mapping = {val: i for i, val in enumerate(order)}
    rfm['Cluster'] = rfm['Cluster_Raw'].map(mapping)
    
    return rfm

def get_explanation_tree(rfm):
    """Entrena un árbol de decisión y mide su precisión (Accuracy)"""
    X = rfm[['Recency', 'Frequency', 'Monetary']]
    y = rfm['Cluster']
    
    #  split 70% entrenamiento / 30% prueba 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    #  Entren con el 70%
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    
    # Verifi con el 30% restante
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return tree, X.columns, accuracy


