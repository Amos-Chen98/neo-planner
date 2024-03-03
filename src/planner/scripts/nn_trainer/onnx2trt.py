'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2024-03-03 17:00:39
'''
import os
import torch
import onnx
import tensorrt as trt

IMG_WIDTH = 480
IMG_HEIGHT = 360
VECTOR_SIZE = 24
INPUT_SIZE = IMG_WIDTH * IMG_HEIGHT + VECTOR_SIZE
EXPLICIT_BATCH = 1

current_path = os.path.dirname(os.path.abspath(__file__))[:-19]  # -8 removes '/scripts', -11 removes '/nn_trainer'
onnx_save_path = '/saved_net/planner_net.onnx'
trt_save_path = '/saved_net/planner_net.trt'


def convert_onnx_to_trt(onnx_model_path, trt_model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    print("TensorRT version: ", trt.__version__)

    # Load onnx model
    onnx_model = onnx.load(onnx_model_path)
    print("Onnx model loaded!")

    # Create builder and network
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(EXPLICIT_BATCH)

    # Create ONNX parser
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_model.SerializeToString()):
        print("ERROR: Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit(0)

    # Set builder config and profile
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, INPUT_SIZE), (1, INPUT_SIZE), (1, INPUT_SIZE))
    config.add_optimization_profile(profile)

    # build and save serialized network
    model_in_memory = builder.build_serialized_network(network, config)
    with open(trt_model_path, "wb") as f:
        f.write(model_in_memory)
    print("TensorRT model saved!")


if __name__ == "__main__":

    onnx_model_path = current_path + onnx_save_path
    trt_model_path = current_path + trt_save_path

    convert_onnx_to_trt(onnx_model_path, trt_model_path)
