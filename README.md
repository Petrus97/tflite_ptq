# Experimenting using MNIST and TFLite
[TODO] Add references

## Visualise TFLite model as JSON
```
flatc --raw-binary -t schema.fbs -- models/lite/int_model.tflite
```

### Run the models
1. Install `pyenv`
2. `pyenv install 3.11`
3. Install `poetry`
4. Go into the workspace and `pyenv local 3.11` (.python_version already exists, so it shoudl be automatic)
5. `poetry init`
6. `poetry env use python`
7. `poetry run python --version` (should be 3.11.x)
8. `poetry update` (should read the pyproject.toml)
9. `poetry shell` (create a shell using the virtual env created by poetry)
10. Run a model `python3 src/mlp.py -t -e --lite int`

## References
https://discuss.tensorflow.org/t/error-when-using-tflite-interpreter-in-flask/4961/21
https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/examples/digit_classifier/ml/mnist_tflite.ipynb#scrollTo=xPtbtEJ2uacB
https://github.com/sanchit88/tf_model_quant/tree/master
https://blog.tensorflow.org/2019/06/tensorflow-integer-quantization.html
https://www.tensorflow.org/lite/performance/quantization_spec
https://www.tensorflow.org/lite/performance/post_training_quantization