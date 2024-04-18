PYTHON := $(shell which python3)
NETRON := netron

MODELS_DIR:=./models
SRC_DIR:=./src

KERAS_MODEL=$(MODELS_DIR)/model.keras
LITE_MODEL=$(MODELS_DIR)/lite/int_model.tflite
PY_EXE:=$(SRC_DIR)/cnn.py

## Train the model and save it as a model.keras format
train:
	$(PYTHON) $(PY_EXE) --train

evaluate:
	$(PYTHON) $(PY_EXE) --eval

lite:
	$(PYTHON) $(PY_EXE) --lite	

load_train:
	$(PYTHON) $(PY_EXE) --load $(KERAS_MODEL) --train

load_evaluate:
	$(PYTHON) $(PY_EXE) --load $(KERAS_MODEL) --eval

load_lite:
	$(PYTHON) $(PY_EXE) --load $(KERAS_MODEL) --lite int


netron:
	$(NETRON) --port 8081 $(LITE_MODEL)

## Clean up generated files
clean:
	@echo "Cleaning up nothing for now"
