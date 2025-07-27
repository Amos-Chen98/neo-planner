import numpy as np
import tensorrt as trt
import time
import pycuda.driver as cuda
import pycuda.autoinit
import threading

# Set paths
trt_model_path = "/home/chen/ros_ws/drone_ws/src/planner/saved_net/planner_net.trt"

IMG_WIDTH = 480
IMG_HEIGHT = 360
VECTOR_SIZE = 24
INPUT_LEN = IMG_WIDTH*IMG_HEIGHT+VECTOR_SIZE
# Set input shape
INPUT_SHAPE = (1, INPUT_LEN)

# view the current cuda device
print("Current cuda device: ", cuda.Device(0).name())

time_start = time.time()
# Load TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(trt_model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
time_end = time.time()
print('TensorRT engine loaded, time cost: ', time_end-time_start)


# Create TensorRT context
with engine.create_execution_context() as context:
    print("Current thread: ", threading.currentThread())    # Allocate device memory for input and output
    input_index = engine.get_binding_index("input")
    output_index = engine.get_binding_index("output")
    input_shape = engine.get_binding_shape(input_index)
    output_shape = engine.get_binding_shape(output_index)
    input_dtype = trt.nptype(engine.get_binding_dtype(input_index))
    output_dtype = trt.nptype(engine.get_binding_dtype(output_index))
    input_size = np.product(input_shape) * np.dtype(input_dtype).itemsize
    output_size = np.product(output_shape) * np.dtype(output_dtype).itemsize
    d_input = cuda.mem_alloc(int(input_size))
    d_output = cuda.mem_alloc(int(output_size))

    # Create numpy array input
    input_data = np.random.rand(*INPUT_SHAPE).astype(np.float16)

    # show current context
    print("Current context: ", context)
    # Copy input to device memory
    cuda.memcpy_htod(d_input, input_data)

    time_start = time.time()

    # Execute inference
    context.execute_v2(bindings=[int(d_input), int(d_output)])
    time_end = time.time()
    print('Inference time cost: ', time_end-time_start)

    # Copy output from device memory
    output_data = np.empty(output_shape, dtype=output_dtype)
    
    cuda.memcpy_dtoh(output_data, d_output)    

    # Print output
    print(output_data)

    
print("Current context2: ", context)
# Create numpy array input
input_data = np.random.rand(*INPUT_SHAPE).astype(np.float16)

# show current context
# Copy input to device memory
cuda.memcpy_htod(d_input, input_data)

# print current thread
print("Current thread: ", threading.currentThread())