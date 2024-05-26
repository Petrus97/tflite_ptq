import pathlib
from prettytable import PrettyTable
import numpy as np
from keras.models import Sequential
from dataset import *
from logger import logging


table = PrettyTable()
table.field_names = ["Predicted", "Original", "Confidence", "Match"]


class TFLiteUtils:
    LITE_MODEL_DIR = MODEL_DIR + "/lite"

    def __init__(self, model: Sequential, mode: str, model_name: str = ""):
        self.model = model
        self.mode = mode
        self.model_name = model_name
        self.converter = tf.lite.TFLiteConverter.from_keras_model(model)
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.__create_lite_model_dir__()

    def __create_lite_model_dir__(self):
        lite_models_dir = pathlib.Path(self.LITE_MODEL_DIR)
        lite_models_dir.mkdir(exist_ok=True, parents=True)

    def set_dataset(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        logging.debug(self.x_train.shape)
        logging.debug(self.y_train.shape)
        logging.debug(self.x_test.shape)
        logging.debug(self.y_test.shape)

    def convert_to_tflite(self):
        print("Converting to TFLite...")
        print(self.model.summary())
        match self.mode:
            case "dyn":
                self.convert_dyn_range()
            case "float16":
                self.convert_float16_quant()
            case "int":
                self.convert_int_quant()

    def convert_dyn_range(self):
        # Dynamic range quantization
        tflite_dyn_model = self.converter.convert()
        with open(
            f"{self.LITE_MODEL_DIR}/dyn_{self.model_name}_model.tflite", "wb"
        ) as f:
            f.write(tflite_dyn_model)
        interpreter = tf.lite.Interpreter(model_content=tflite_dyn_model)
        self.evaluate_tflite(interpreter)

    def convert_float16_quant(self):
        # Float16 quantization
        self.converter.target_spec.supported_types = [tf.float16]
        tflite_f16_model = self.converter.convert()
        with open(
            f"{self.LITE_MODEL_DIR}/f16_{self.model_name}_model.tflite", "wb"
        ) as f:
            f.write(tflite_f16_model)
        interpreter = tf.lite.Interpreter(model_content=tflite_f16_model)
        self.evaluate_tflite(interpreter)

    def convert_int_quant(self):
        # Full integer quantization
        def representative_dataset():
            images = self.x_train  # tf.cast(mnist_train[0], tf.float32) / 255.0
            logging.debug(images[0].shape)
            # images = mnist_train[0]
            mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
            for input_value in mnist_ds.take(100):
                logging.debug(input_value.shape)
                yield [input_value]

        self.converter.representative_dataset = representative_dataset
        self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        self.converter.inference_input_type = tf.uint8  # or tf.uint8
        self.converter.inference_output_type = tf.int8  # or tf.uint8
        tflite_int_model = self.converter.convert()
        with open(f"{self.LITE_MODEL_DIR}/int_{self.model_name}_model.tflite", "wb") as f:
            f.write(tflite_int_model)
        interpreter = tf.lite.Interpreter(model_content=tflite_int_model, experimental_preserve_all_tensors=True)
        self.evaluate_tflite(interpreter)
        self.check_tensors(interpreter, 8)

    def evaluate_tflite(self, interpreter: tf.lite.Interpreter):
        interpreter.allocate_tensors()
        correct = 0
        incorrect_idxs = list()
        for i in range(self.x_test.shape[0]): # 10000 images
            input_image = mnist_test[0][i] # FIXME, there shuld be a better way
            logging.debug(input_image.shape)
            input_image = np.reshape(input_image, self.x_test[0].shape)
            logging.debug(input_image.shape)
            input_image = np.expand_dims(input_image, axis=0)
            out = self.predict_lite(interpreter, input_image)
            digit = np.argmax(out[0])
            actual_digit = np.argmax(self.y_test[i])
            confidence = out[0][digit]
            if i < 10:
                table.add_row(
                    [digit, actual_digit, confidence, "✅" if digit == actual_digit else "❌"]
                )
            if digit == actual_digit:
                correct += 1
            else:
                incorrect_idxs.append(i)
        print(table)
        logging.warn("Lite accuracy: {}{}".format((correct / 10000) * 100, "%"))

    def predict_lite(self, interpreter: tf.lite.Interpreter, data_in: np.ndarray) -> np.ndarray:
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], data_in)
        interpreter.invoke()
        out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        return out
    
    def check_tensors(self, interpreter: tf.lite.Interpreter, idx: int):
        input_image = mnist_test[0][idx]
        input_image = np.reshape(input_image, self.x_test[0].shape)
        input_image = np.expand_dims(input_image, axis=0)
        # input_image.tofile("same_img.data")
        # np.save("image_test",input_image)
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_image)
        # print(interpreter.get_tensor(interpreter.get_input_details()[0]["index"]))
        interpreter.invoke()
        logging.debug(interpreter.get_tensor_details())
        for t_detail in interpreter.get_tensor_details():
            logging.info(t_detail)
            tensor = t_detail["index"]
            logging.debug("{} - {}".format(t_detail["name"], t_detail["index"]))
            self.save_tensor(interpreter, tensor, t_detail["name"])
            # try:
            #     array = interpreter.get_tensor(tensor)
            #     np.save(t_detail["name"].replace("/", "_"), array)
            #     print(array)
            # except ValueError:
            #     print("Tensor data is null")
        out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        logging.warn("output tensor: {}".format(out))
        logging.warn("predicted {} - real {}".format(np.argmax(out), np.argmax(y_test[idx])))

    def save_tensor(self, interpreter: tf.lite.Interpreter, tensor_idx: int, name: str):
        ARRAY_DIR = "numpy_arrays/"
        try:
            array = interpreter.get_tensor(tensor_idx)
            np.save(ARRAY_DIR + name.replace("/", "_"), array)
            if ("quantize" in name) or ("default" in name): # skip these tensors
                return
            elif ("conv" in name) or ("max" in name):
                self.__print_np__(array)
            elif name == "": # skip these tensors
                return
            else:
                print(array)
        except ValueError:
            logging.error("Tensor data is null")
    
    def __print_np__(self, array: np.ndarray):
        threshold = np.get_printoptions()["threshold"]
        linewidth = np.get_printoptions()["linewidth"]
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        if(len(array.shape) == 4 and array.shape[0] == 1):
            for i in range(array.shape[3]):
                print(array[0,:,:,i])
        else:
            print(array)
        np.set_printoptions(threshold=threshold, linewidth=linewidth)
            


# def check(interpreter: tf.lite.Interpreter):
#     # check the output tensor for the first image
#     input_image = mnist_test[0][0]
#     # print(input_image)
#     input_image = np.expand_dims(input_image, axis=0)
#     interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_image)
#     # print(interpreter.get_tensor(interpreter.get_input_details()[0]["index"]))
#     interpreter.invoke()
#     for t_detail in interpreter.get_tensor_details():
#         tensor = t_detail["index"]
#         print("{}".format(t_detail["name"]))
#         print(interpreter.get_tensor(tensor))
#     out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
#     print(out)


# def predict_lite(interpreter: tf.lite.Interpreter, data_in) -> np.ndarray:
#     interpreter.set_tensor(interpreter.get_input_details()[0]["index"], data_in)
#     interpreter.invoke()
#     out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
#     return out


# def evaluate_tflite(interpreter: tf.lite.Interpreter):
#     interpreter.allocate_tensors()  # Needed before execution!
#     for i in range(20):
#         input_image = mnist_test[0][i]
#         # input_image = x_test[i]
#         logging.debug(input_image.shape)
#         input_image = np.expand_dims(input_image, axis=0)  # To have (1,28,28,1) tensor
#         out = predict_lite(interpreter, input_image)
#         digit = np.argmax(out[0])
#         actual_digit = np.argmax(y_test[i])
#         confidence = out[0][digit]
#         table.add_row(
#             [digit, actual_digit, confidence, "✅" if digit == actual_digit else "❌"]
#         )
#         # print(
#         #     f"Predicted Digit: {digit} - Real Digit: {actual_digit}\nConfidence: {out[0][digit]}"
#         # )
#         # if digit != actual_digit:
#         #     print("Confidence vector:", out[0])
#     print(table)
#     correct = 0
#     incorrect_idxs = list()
#     for i in range(10000):
#         input_image = mnist_test[0][i]
#         # input_image = x_test[i]
#         input_image = np.expand_dims(input_image, axis=0)  # To have (1,28,28,1) tensor
#         out = predict_lite(interpreter, input_image)
#         digit = np.argmax(out[0])
#         actual_digit = np.argmax(y_test[i])
#         confidence = out[0][digit]
#         if digit == actual_digit:
#             correct += 1
#         else:
#             incorrect_idxs.append(i)
#         # table.add_row([digit, actual_digit, confidence, "✅" if digit == actual_digit else "❌"])
#     print("Lite accuracy:", (correct / 10000) * 100, "%")
#     print(incorrect_idxs)
#     # check the output tensor for the first image
#     input_image = mnist_test[0][8]
#     # input_image = x_test[10]
#     # print(input_image)
#     input_image = np.expand_dims(input_image, axis=0)
#     # input_image.tofile("same_img.data")
#     # np.save("image_test",input_image)
#     interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_image)
#     # print(interpreter.get_tensor(interpreter.get_input_details()[0]["index"]))
#     interpreter.invoke()
#     logging.debug(interpreter.get_tensor_details())
#     for t_detail in interpreter.get_tensor_details():
#         logging.info(t_detail)
#         tensor = t_detail["index"]
#         logging.debug("{} - {}".format(t_detail["name"], t_detail["index"]))
#         try:
#             array = interpreter.get_tensor(tensor)
#             np.save(t_detail["name"].replace("/", "_"), array)
#             print(array)
#         except ValueError:
#             print("Tensor data is null")
#     out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
#     print(out)
#     print(np.argmax(out), np.argmax(y_test[8]))


# def convert_to_tflite(model: Sequential, mode: str, model_name: str = ""):
#     print("Converting to TFLite...")
#     print(model.summary())
#     LITE_MODEL_DIR = MODEL_DIR + "/lite"
#     lite_models_dir = pathlib.Path(LITE_MODEL_DIR)
#     lite_models_dir.mkdir(exist_ok=True, parents=True)
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     match mode:
#         case "dyn":
#             # Dynamic range quantization
#             tflite_dyn_model = converter.convert()
#             with open(f"{LITE_MODEL_DIR}/dyn_{model_name}_model.tflite", "wb") as f:
#                 f.write(tflite_dyn_model)
#             interpreter = tf.lite.Interpreter(model_content=tflite_dyn_model)
#             evaluate_tflite(interpreter)
#         case "float16":
#             # Float16 quantization
#             converter.target_spec.supported_types = [tf.float16]
#             tflite_f16_model = converter.convert()
#             with open(f"{LITE_MODEL_DIR}/f16_{model_name}_model.tflite", "wb") as f:
#                 f.write(tflite_f16_model)
#             interpreter = tf.lite.Interpreter(model_content=tflite_f16_model)
#             evaluate_tflite(interpreter)
#         case "int":
#             # Full integer quantization
#             def representative_dataset():
#                 images = x_train  # tf.cast(mnist_train[0], tf.float32) / 255.0
#                 logging.warn(images[0].shape)
#                 # images = mnist_train[0]
#                 mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
#                 for input_value in mnist_ds.take(100):
#                     input_value = np.expand_dims(input_value, axis=0)
#                     logging.warn(input_value.shape)
#                     yield [input_value]

#             converter.representative_dataset = representative_dataset
#             converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#             converter.inference_input_type = tf.uint8  # or tf.uint8
#             converter.inference_output_type = tf.int8  # or tf.uint8
#             tflite_int_model = converter.convert()
#             with open(f"{LITE_MODEL_DIR}/int_{model_name}_model.tflite", "wb") as f:
#                 f.write(tflite_int_model)
#             interpreter = tf.lite.Interpreter(model_content=tflite_int_model)
#             evaluate_tflite(interpreter)
