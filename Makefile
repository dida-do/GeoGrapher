.PHONY: help env env-update env-remove init install format lint test \
	docs-sphinx requirements

PROJECTNAME=rs_tools

help:
	@echo "Available commands:"
	@echo "env              create the venv '$(PROJECTNAME)-env'."
	@echo "env-update       update '$(PROJECTNAME)-env' venv."
	@echo "env-remove       remove '$(PROJECTNAME)-env' venv."
	@echo "install          install package in editable mode."
	@echo "format           format code."
	@echo "lint             run linters."
	@echo "test             run unit tests."
	@echo "docs-sphinx      build sphinx documentation."

env:
	python -m venv $(PROJECTNAME)-env && \
		$(PROJECTNAME)-env/bin/pip install --upgrade pip

env-update:
	pip install --upgrade -r requirements.txt

env-remove:
	rm -rf aubesa-env

install:
	pip install --upgrade pip wheel pip-tools &&\
	pip-sync requirements.txt
	pip install -e .

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
