# dags/kmeans_dag.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sklearn.datasets import load_boston
import os

# Ensure output folder exists
OUTPUT_FOLDER = '/opt/airflow/dags/output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1
}

with DAG(
    'kmeans_dag',
    default_args=default_args,
    description='K-Means Clustering on Boston Housing Dataset',
    schedule_interval=None,  # manual trigger
    start_date=datetime(2025, 10, 6),
    catchup=False
) as dag:

    # ----------------- FUNCTIONS -----------------
    def load_dataset():
        boston = load_boston()
        df = pd.DataFrame(boston.data, columns=boston.feature_names)
        df['PRICE'] = boston.target
        return df

    def compute_elbow(df, max_k=10):
        X = df.drop(columns=['PRICE'])
        inertia = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
        
        plt.figure()
        plt.plot(range(1, max_k + 1), inertia, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.savefig(os.path.join(OUTPUT_FOLDER, 'elbow_plot.png'))
        plt.close()

    def run_kmeans(df, k=3):
        X = df.drop(columns=['PRICE'])
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['cluster'] = kmeans.fit_predict(X)
        df.to_csv(os.path.join(OUTPUT_FOLDER, f'kmeans_{k}_clusters.csv'), index=False)

    # ----------------- TASKS -----------------
    def task_load_dataset(**kwargs):
        df = load_dataset()
        kwargs['ti'].xcom_push(key='dataset', value=df.to_dict())

    def task_elbow(**kwargs):
        ti = kwargs['ti']
        df_dict = ti.xcom_pull(key='dataset', task_ids='load_dataset_task')
        df = pd.DataFrame(df_dict)
        compute_elbow(df, max_k=10)

    def task_run_kmeans(**kwargs):
        ti = kwargs['ti']
        df_dict = ti.xcom_pull(key='dataset', task_ids='load_dataset_task')
        df = pd.DataFrame(df_dict)
        run_kmeans(df, k=3)

    # ----------------- OPERATORS -----------------
    load_dataset_task = PythonOperator(
        task_id='load_dataset_task',
        python_callable=task_load_dataset
    )

    run_elbow_task = PythonOperator(
        task_id='run_elbow_task',
        python_callable=task_elbow
    )

    run_kmeans_task = PythonOperator(
        task_id='run_kmeans_task',
        python_callable=task_run_kmeans
    )

    # ----------------- DEPENDENCIES -----------------
    load_dataset_task >> run_elbow_task >> run_kmeans_task
