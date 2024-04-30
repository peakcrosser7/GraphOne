add_library(onnx SHARED onnx.pb.cc)
target_link_libraries(onnx ${Protobuf_LIBRARIES})