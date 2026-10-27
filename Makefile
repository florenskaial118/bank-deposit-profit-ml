.PHONY: airflow down logs

airflow:
	docker compose --profile airflow up -d --build postgres-dwh postgres-airflow airflow-init airflow-webserver airflow-scheduler
	@printf "Airflow UI: http://localhost:8088\n"
	@printf "Login: admin / admin\n"
	@printf "Postgres: localhost:5432, database dwh, user admin, password admin\n"

down:
	docker compose --profile airflow down

logs:
	docker compose --profile airflow logs -f airflow-webserver airflow-scheduler postgres-dwh
