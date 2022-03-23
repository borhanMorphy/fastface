.PHONY: install format

install:
	pip install -e ".[dev]"

format:
	isort fastface
	black fastface

test-format:
	black --check fastface
