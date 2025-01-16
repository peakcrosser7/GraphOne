add_library(lib3_onnx SHARED onnx.pb.cc)
target_link_libraries(lib3_onnx ${Protobuf_LIBRARIES})