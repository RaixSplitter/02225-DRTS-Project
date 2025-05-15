
CONFIG := experiment.yaml

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .


# Run the app
run:
	python simsystem/main.py --config-name $(CONFIG)