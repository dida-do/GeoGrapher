.PHONY: help env env-remove install install-dev format lint test \
	docs-sphinx

PROJECTNAME=geographer

help:
	@echo "Available commands:"
	@echo "env              create the venv '$(PROJECTNAME)-env'."
	@echo "env-remove       remove '$(PROJECTNAME)-env' venv."
	@echo "install          install package in editable mode."
	@echo "format           format code."
	@echo "lint             run linters."
	@echo "test             run unit tests."
	@echo "docs-sphinx      build sphinx documentation."

env:
	python -m venv $(PROJECTNAME)-env && \
		$(PROJECTNAME)-env/bin/pip install --upgrade pip

env-remove:
	rm -rf $(PROJECTNAME)-env

install:
	pip install --upgrade pip wheel pip-tools &&\
	python -m pip install -e .

install-dev:
	pip install --upgrade pip wheel pip-tools &&\
	python -m pip install -e ".[dev]"

format:
	yapf -i --recursive $(PROJECTNAME)
	isort -rc --atomic $(PROJECTNAME)
	docformatter -i -r $(PROJECTNAME)

lint:
	yapf --diff --recursive $(PROJECTNAME)
	pylint -v $(PROJECTNAME) tests
	mypy $(PROJECTNAME) tests

test:
	pytest -v

docs-sphinx:
	cd docs && make html
