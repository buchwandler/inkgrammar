.PHONY: install lint test validate render release clean

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	python -m pytest

validate:
	inkgrammar model validate data/models/demo-looped.yaml --strict

render:
	inkgrammar model render data/models/demo-looped.yaml "minimum" --output work/minimum.png

release:
	inkgrammar build-release data/models/demo-looped.yaml --version 0.2.0 --output dist

clean:
	rm -rf build dist work .pytest_cache .ruff_cache *.egg-info inkgrammar/_version.py
