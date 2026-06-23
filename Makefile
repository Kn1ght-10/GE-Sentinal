.PHONY: install demo demo-fast collect api dashboard test lint docker-build

install:
	pip install -e ".[serve,dev]"

demo:
	python -m ge_sentinel.cli demo

demo-fast:
	python -m ge_sentinel.cli demo --fast

collect:
	python -m ge_sentinel.cli collect --loop 12

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

test:
	pytest -q

lint:
	ruff check .

docker-build:
	docker compose build
