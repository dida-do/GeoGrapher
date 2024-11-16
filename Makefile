## Summary of available make targets:
##
## make help         -- Display this message
## make -B venv      -- (Re)install a development virtual environment
## make format       -- Run code formatter
## make lint         -- Run linter and type checker
## make test         -- Run tests
## make docs         -- Run documentation
##
## This Makefile needs to be run inside a virtual environment

ifndef VIRTUAL_ENV
$(error "This Makefile needs to be run inside a virtual environment")
endif

.PHONY: help venv format lint test docs

PROJECTNAME=geographer

help:
	@sed -rn 's/^## ?//;T;p' $(MAKEFILE_LIST)

venv: $(VIRTUAL_ENV)/timestamp

$(VIRTUAL_ENV)/timestamp: pyproject.toml
	pip install --upgrade pip
	pip install -e ".[dev,docs]"
ifneq ($(wildcard requirements/extra.txt),)
	pip install -r requirements/extra.txt
endif
	touch $(VIRTUAL_ENV)/timestamp

format: venv
	ruff check $(PROJECTNAME) tests --fix

lint: venv
	ruff check $(PROJECTNAME) tests

test: venv
	pytest -v -m "not slow"

test-slow: venv
	pytest -v -m "slow"

docs: venv
	cd docs && sphinx-apidoc -o source/ ../$(PROJECTNAME) && make clean html
