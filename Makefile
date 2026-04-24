PYTHON ?= python
UV ?= uv

.PHONY: setup test smoke lint process-fixture ci-local

setup:
	$(UV) pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m amatra.cli smoke --mock

lint:
	PYTHONPATH=src:app/src $(PYTHON) -m compileall src/amatra app/src/amatra_app tests
	PYTHONPATH=src:app/src $(PYTHON) -m amatra.tools.check_dependencies

process-fixture:
	$(PYTHON) -m amatra.cli process-fixture --fixture multiple-bubbles

ci-local:
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) smoke
