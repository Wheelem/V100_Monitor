.PHONY: test lint type-check docs

test:
	pytest tests/

lint:
	flake8 GUI/

type-check:
	mypy GUI/

docs:
	@echo "Documentation is not yet set up"