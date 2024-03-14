# Experimenting using MNIST and TFLite
[TODO] Add references

## Visualise TFLite model as JSON
```
flatc --raw-binary -t schema.fbs -- models/lite/int_model.tflite
```

## References
https://discuss.tensorflow.org/t/error-when-using-tflite-interpreter-in-flask/4961/21
https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/examples/digit_classifier/ml/mnist_tflite.ipynb#scrollTo=xPtbtEJ2uacB
https://github.com/sanchit88/tf_model_quant/tree/master
https://blog.tensorflow.org/2019/06/tensorflow-integer-quantization.html
https://www.tensorflow.org/lite/performance/quantization_spec
https://www.tensorflow.org/lite/performance/post_training_quantization