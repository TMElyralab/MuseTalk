all:

upload:
	rm -r dist
	python setup.py sdist
	twine upload dist/*

test:
	# For some reason the import changes for Python 3 caused the Python 2 test
	# loader to give up without loading any tests. So we discover them ourselves.
	# python -m unittest
	find chumpy -name 'test_*.py' | sed -e 's/\.py$$//' -e 's/\//./' | xargs python -m unittest

coverage: clean qcov
qcov: all
	env LD_PRELOAD=$(PRELOADED) coverage run --source=. -m unittest discover -s .
	coverage html
	coverage report -m
