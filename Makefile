.PHONY: setup lint test data train evaluate predict app clean

setup:        ## install dependencies + package (editable)
	pip install -r requirements.txt && pip install -e .

lint:
	flake8 src tests scripts app --max-line-length=100 --extend-ignore=E203,W503

test:
	pytest -q

data:         ## print dataset download instructions
	python scripts/download_data.py

train:
	python scripts/train.py --config config/config.yaml

evaluate:
	python scripts/evaluate.py --checkpoint models/best_model.pt --split testing

predict:      ## make predict AUDIO=path/to/file.wav
	python scripts/predict.py --checkpoint models/best_model.pt --audio $(AUDIO)

app:
	streamlit run app/streamlit_app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + ; rm -rf .pytest_cache *.egg-info
