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

$(VIRTUAL_ENV)/timestamp: pyproject.toml setup.cfg
	pip install --upgrade pip
	pip install -e ".[dev,docs]"
ifneq ($(wildcard requirements/extra.txt),)
	pip install -r requirements/extra.txt
endif
	touch $(VIRTUAL_ENV)/timestamp

format: venv
	isort $(PROJECTNAME)
	docformatter -i -r $(PROJECTNAME)
	black $(PROJECTNAME)

lint: venv
	black --check $(PROJECTNAME)
	pylint -v $(PROJECTNAME) tests
	mypy $(PROJECTNAME) tests

test: venv
	pytest -v

docs: venv
	cd docs && make html
