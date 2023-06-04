# NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
default:
	c++ -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup $$(python3 -m pybind11 --includes) src/simple_ml_ext.cpp -o src/simple_ml_ext.so
test:
	# python3 -m pytest -k "add"
	# python3 -m pytest -k "parse_mnist"
	# python3 -m pytest -k "softmax_loss"
	python3 -m pytest -k "softmax_regression_epoch and not cpp"