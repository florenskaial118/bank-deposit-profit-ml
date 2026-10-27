import io
import zipfile

import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from sqlalchemy import create_engine

from jobs.config import DATA_URL, DWH
from jobs.train import train


def extract():
    response = requests.get(DATA_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    frame = pd.read_csv(archive.open("bank-additional/bank-additional-full.csv"), sep=";")
    frame.insert(0, "source_row", range(1, len(frame) + 1))
    frame.to_sql("raw_contacts", create_engine(DWH), if_exists="replace", index=False)


def build_layers():
    return [
        "DROP TABLE IF EXISTS mart_campaign_features;",
        "DROP TABLE IF EXISTS ods_contacts;",

        # переименовываем колонки
        """CREATE TABLE ods_contacts AS
        SELECT
            source_row,
            age,
            job,
            marital,
            education,
            "default" AS credit_default,
            housing,
            loan,
            contact,
            month,
            day_of_week,
            duration,
            campaign,
            pdays,
            previous,
            poutcome,
            "emp.var.rate"   AS emp_var_rate,
            "cons.price.idx" AS cons_price_idx,
            "cons.conf.idx"  AS cons_conf_idx,
            euribor3m,
            "nr.employed"    AS nr_employed,
            y
        FROM raw_contacts;""",

        # только нужные колонки + pdays > never_contacted
        """CREATE TABLE mart_campaign_features AS
        SELECT
            source_row,
            age,
            job,
            marital,
            education,
            credit_default,
            housing,
            loan,
            contact,
            month,
            day_of_week,
            campaign,
            CASE WHEN pdays = 999 THEN 1 ELSE 0 END AS never_contacted,
            previous,
            poutcome,
            cons_price_idx,
            cons_conf_idx,
            nr_employed,
            CASE
                WHEN lower(CAST(y AS TEXT)) IN ('yes', '1', 'true') THEN 1
                ELSE 0
            END AS subscribed
        FROM ods_contacts
        ORDER BY source_row;""",
    ]


with DAG(
    dag_id="bank_term_deposit",
    schedule=None,
    catchup=False,
    tags=["ml", "bank"],
) as dag:
    extract_task = PythonOperator(task_id="extract", python_callable=extract)
    build_mart = PostgresOperator(
        task_id="build_mart",
        postgres_conn_id="dwh_postgres",
        sql=build_layers(),
        autocommit=True,
        split_statements=False,
    )
    train_task = PythonOperator(task_id="train_model", python_callable=train)

    extract_task >> build_mart >> train_task