add_cuda_exec(mat_builder_test mat_builder_test.cu)

add_cuda_exec(buffer_test buffer_test.cu)

add_cuda_exec(graph_test graph_test.cu)

add_cuda_exec(thrust_test thrust_test.cu)

add_executable(loader_test loader_test.cpp)

add_cuda_exec(spmv_test spmv_test.cu)

add_executable(module_test module_test.cpp)

add_executable(json_model_test json_model_test.cpp)

add_executable(refl_test refl_test.cpp)

add_executable(onnx_test onnx_test.cpp)
target_link_libraries(onnx_test onnx)

add_executable(onnx_model_test onnx_model_test.cpp)
target_link_libraries(onnx_model_test onnx)
