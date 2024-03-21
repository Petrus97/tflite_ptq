import os
import tflite
import numpy as np
import json


operator_names = {attr_value: attr_name for attr_name, attr_value in vars(tflite.BuiltinOperator).items() if not callable(attr_value) and not attr_name.startswith("__")}
type_names = {attr_value: attr_name for attr_name, attr_value in vars(tflite.TensorType).items() if not callable(attr_value) and not attr_name.startswith("__")}

def get_operator_name(index):
    return operator_names.get(index, "Unknown")

def get_type_name(index):
    return type_names.get(index, "Unknown")

def get_tensor(model: tflite.Model, tensor: tflite.Tensor) -> np.ndarray:
    data = model.Buffers(tensor.Buffer()).DataAsNumpy()
    data_type: np.number = np.uint8
    match tensor.Type():
        case tflite.TensorType.FLOAT32:
            data_type = np.float32
        case tflite.TensorType.FLOAT16:
            data_type = np.float16
        case tflite.TensorType.INT32:
            data_type = np.int32
        case tflite.TensorType.UINT8:
            data_type = np.uint8
        case tflite.TensorType.INT64:
            data_type = np.int64
        case tflite.TensorType.STRING:
            data_type = np.str_
        case tflite.TensorType.BOOL:
            data_type = np.bool_
        case tflite.TensorType.INT16:
            data_type = np.int16
        case tflite.TensorType.COMPLEX64:
            data_type = np.complex64
        case tflite.TensorType.INT8:
            data_type = np.int8
        case tflite.TensorType.FLOAT64:
            data_type = np.float64
        case tflite.TensorType.COMPLEX128:
            data_type = np.complex128
        case tflite.TensorType.UINT64:
            data_type = np.uint64
        case tflite.TensorType.RESOURCE:
            data_type = np.object_
        case tflite.TensorType.VARIANT:
            data_type = np.object_
        case tflite.TensorType.UINT32:
            data_type = np.uint32
        case tflite.TensorType.UINT16:
            data_type = np.uint16
    # if tensor.ShapeLength() == 1:
    #     print(tensor.Shape(0))
    #     shape = [tensor.Shape(0), 1]
    #     # print(model.Buffers(tensor.Buffer()).DataAsNumpy().astype(np.int32)) # .reshape(shape).astype(data_type))
    #     print(data.view(dtype=data_type).reshape(tensor.ShapeAsNumpy()))
    # else:
        # print(model.Buffers(tensor.Buffer()).DataAsNumpy().reshape(tensor.ShapeAsNumpy()).astype(data_type))
    data = data.view(dtype=data_type).reshape(tensor.ShapeAsNumpy())
    print(data)
    return data



def import_model(model_name: str = "int_model.tflite"):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tflm_dir = os.path.abspath(cur_dir + '/models/lite/')
    tflm_name = model_name
    path = os.path.join(tflm_dir, tflm_name)
    with open(path, 'rb') as f:
        buf = f.read()
        model = tflite.Model.GetRootAs(buf, 0)
        print(model)
        # My TFLite model has only one subgraph
        subgraph = model.Subgraphs(0)
        # Tensors metadata of subgraph
        tensors = [subgraph.Tensors(i) for i in range(subgraph.TensorsLength())]
        # Save the opcodes
        opcodes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]
        print([get_operator_name(op.BuiltinCode()) for op in opcodes])
        # Here the order of the operations executed in the graph
        print("Subgraph operations")
        layer = 0
        json_ser = {}
        json_ser["layers"] = []
        print(json_ser)
        for i in range(subgraph.OperatorsLength()):
            layer_json = {}
            op = subgraph.Operators(i)
            print(op.OpcodeIndex(), get_operator_name(opcodes[op.OpcodeIndex()].BuiltinCode()))
            input_tensors = [tensors[op.Inputs(i)] for i in range(op.InputsLength())]
            output_tensors = [tensors[op.Outputs(i)] for i in range(op.OutputsLength())]
            # print(output_tensors)
            if get_operator_name(opcodes[op.OpcodeIndex()].BuiltinCode()) == 'FULLY_CONNECTED':
                input = input_tensors[0]
                input_scale = input.Quantization().Scale(0)
                input_zp = input.Quantization().ZeroPoint(0)
                print(input.ShapeAsNumpy().transpose())
                print(f"Input layer {layer}. Scale:", input_scale, "Zero Point:",input_zp)
                weights = input_tensors[1]
                weights_scale = weights.Quantization().Scale(0)
                weights_zp = weights.Quantization().ZeroPoint(0)
                print(f"Weights layer {layer}. Scale:", weights_scale, "Zero Point:",weights_zp)
                weights_array = get_tensor(model, weights)
                bias = input_tensors[2]
                bias_scale = bias.Quantization().Scale(0)
                bias_zp = bias.Quantization().ZeroPoint(0)
                print(f"Bias layer {layer}. Scale:", bias_scale, "Zero Point:",bias_zp)
                bias_array = get_tensor(model, bias)
                output = output_tensors[0]
                output_scale = output.Quantization().Scale(0)
                output_zp = output.Quantization().ZeroPoint(0)
                # Save scaling factors
                layer_json["s_input"] = input_scale
                layer_json["s_weight"] = weights_scale
                layer_json["s_bias"] = bias_scale
                layer_json["s_output"] = output_scale
                # Save zero points
                layer_json["z_input"] = input_zp
                layer_json["z_weight"] = weights_zp
                layer_json["z_bias"] = bias_zp
                layer_json["z_output"] = output_zp
                # Weights array
                layer_json["weights"] = {}
                layer_json["weights"]["dtype"] = str(weights_array.dtype)
                layer_json["weights"]["shape"] = weights_array.shape
                layer_json["weights"]["data"] = weights_array.tolist()
                # Bias array
                ## Precompute par qb - (Zin @ qw)
                z_in_arr = (np.ones(shape=input.ShapeAsNumpy(), dtype=np.int32) * input_zp).transpose()
                print(z_in_arr.dtype)
                z_in_dot_qw = weights_array @ z_in_arr
                q_bias = bias_array.reshape(z_in_dot_qw.shape)  - z_in_dot_qw # - (weights_array @ (np.ones(shape=input.ShapeAsNumpy()) * input_zp).transpose())
                layer_json["bias"] = {}
                layer_json["bias"]["dtype"] = str(q_bias.dtype)
                layer_json["bias"]["shape"] = q_bias.shape
                layer_json["bias"]["data"] = q_bias.tolist()
                # print(weights_array.dtype, weights_array.shape)
                # print(layer_json)
                json_ser["layers"].insert(layer, layer_json)
                layer += 1
        with open("extracted.json", "w") as f:
            f.write(json.dumps(json_ser))





if __name__ == '__main__':
    import_model(model_name="int_model_tmp.tflite")
    # import_model()