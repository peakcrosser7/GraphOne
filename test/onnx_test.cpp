#include "onnx/onnx.pb.h"
#include <fstream>
#include <iostream>

void print_dim(const ::onnx::TensorShapeProto_Dimension &dim) {
    switch (dim.value_case()) {
    case onnx::TensorShapeProto_Dimension::ValueCase::kDimParam:
        std::cout << "(param)" << dim.dim_param();
        break;
    case onnx::TensorShapeProto_Dimension::ValueCase::kDimValue:
        std::cout << "(value)" << dim.dim_value();
        break;
    default:
        assert(false && "should never happen");
    }
}

void print_io_info(
    const ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto> &info) {
    for (auto input_data : info) {
        auto shape = input_data.type().tensor_type().shape();
        std::cout << "  " << input_data.name() << ":";
        std::cout << "[";
        if (shape.dim_size() != 0) {
            int size = shape.dim_size();
            for (int i = 0; i < size - 1; ++i) {
                print_dim(shape.dim(i));
                std::cout << ",";
            }
            print_dim(shape.dim(size - 1));
        }
        std::cout << "]\n";
    }
}

void print_initializer_info(
    const ::google::protobuf::RepeatedPtrField<::onnx::TensorProto> &info) {
    for (auto input_data : info) {
        auto data_type = input_data.data_type();
        std::cout << "  data_type: "
                  << onnx::TensorProto::DataType_Name(data_type) << std::endl;
        auto dims = input_data.dims();

        auto raw_data = input_data.raw_data();     // weight
        float *data_r = (float *)raw_data.c_str(); // raw_data 读取
        int k = raw_data.size() / 4;               // float 是4个字节

        // auto tile = input_data.xb_number(0);
        // float *y = reinterpret_cast<char*>(&raw_data)(4);
        // std::cout << raw_data.size() << std::endl;
        // auto shape = input_data.type().tensor_type().shape();
        std::cout << "  name:" << input_data.name() << "\n";
        // << "tile: " << tile << ":";
        std::cout << "  shapes: ";
        for (auto dim : dims)
            std::cout << dim << " ";
        std::cout << std::endl;

        std::cout << "  weights:\n"
                  << "[";
        for (int i = 0; i < k; ++i) {
            std::cout << data_r[i] << " "; // print weight
        }

        std::cout << "]\n";
    }
}

void print_node_info(
    const ::google::protobuf::RepeatedPtrField<::onnx::NodeProto> &info) {
    for (auto input_data : info) {
        auto op_type = input_data.op_type();
        // AttributeProto
        auto shape = input_data.attribute();
        std::cout << op_type << " " << input_data.name() << ":";
        std::cout << std::endl << "  Inputs:";
        for (auto inp : input_data.input())
            std::cout << inp << " ";
        std::cout << std::endl << "  Outputs:";
        for (auto outp : input_data.output())
            std::cout << outp << " ";

        std::cout << std::endl << "  Attr:\n  [";
        // Print Attribute
        for (auto y : shape) {
            std::cout << y.name() << ": ";
            for (auto t : y.ints())
                std::cout << t << " ";
        }
        std::cout << "]\n";
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <onnx_file>\n";
        return -1;
    }

    const char *ONNX_FILE = argv[1];
    // 消息解析
    //  onnx::ModelProto out_msg;
    // {
    // 	std::fstream input(ONNX_FILE, std::ios::in | std::ios::binary);
    // 	if (!out_msg.ParseFromIstream(&input)) {
    // 	  std::cerr << "failed to parse" << std::endl;
    // 	  return -1;
    // 	}
    // 	std::cout << out_msg.graph().node_size() << std::endl;
    // }
    onnx::ModelProto model;
    // std::ifstream input(ONNX_FILE, std::ios::ate | std::ios::binary);
    // // get current position in file
    // std::streamsize size = input.tellg();
    // // move to start of file
    // input.seekg(0, std::ios::beg);
    // // read raw data
    // std::vector<char> buffer(size);
    // input.read(buffer.data(), size);
    // model.ParseFromArray(buffer.data(), size); // parse protobuf

    std::fstream input(ONNX_FILE, std::ios::in | std::ios::binary);
    if (!model.ParseFromIstream(&input)) {
        std::cerr << "failed to parse" << std::endl;
        return -1;
    }

    auto graph = model.graph();
    std::cout << "graph node size:" << model.graph().node_size() << std::endl;
    std::cout << "initializer_size:" << graph.initializer_size() << std::endl;
    std::cout << "\ngraph inputs:\n";
    print_io_info(graph.input());
    std::cout << "\ngraph outputs:\n";
    print_io_info(graph.output());
    std::cout << "\ngraph initializer:\n";
    print_initializer_info(graph.initializer());
    std::cout << "\ngraph node:\n";
    print_node_info(graph.node());

    return 0;
}