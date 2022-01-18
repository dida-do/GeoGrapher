.PHONY: env env-update env-remove lint install

# put the name of your project here (has to be the name of the package dir)
PROJECTNAME=rs_tools

help:
	@echo "Available commands:"
	@echo "env              create the conda environment '$(PROJECTNAME)-env'."
	@echo "env-update       update '$(PROJECTNAME)-env'."
	@echo "env-remove       remove '$(PROJECTNAME)-env'."
	@echo "install	        install package in editable mode."
	@echo "format           format code."
	@echo "lint             run linters."
	@echo "test             run unit tests."

env:
	conda env create --file environment.yml

env-update:
	conda env update --name $(PROJECTNAME)-env --file environment.yml

env-remove:
	conda remove --name $(PROJECTNAME)-env --all

install:
ifeq (${CONDA_DEFAULT_ENV}, $(PROJECTNAME)-env)
	pip install -e .
else
	@echo "Activate conda env first with 'conda activate $(PROJECTNAME)-env'"
endif


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
