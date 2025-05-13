# GraphOne+: A unified heterogeneous platform for graph analysis problems

GraphOne+, a unified heterogeneous platform that supports graph analysis problems including graph computation, graph learning, and graph mining. This platform aims to provide users with a simple interface to implement graph algorithms and efficiently perform computations on heterogeneous devices. Currently, it supports the computation of graph computation problems on GPU devices.

## Setup
### Requirements
* Operating System: Linux (Ubuntu 18.04 and above)
* Hardware: NVIDIA GPU (preferably with a Compute Capability of 7.0 or higher)
* Software:
    * GCC (version 9.4.0 and above)
    * CUDA (version 12.4 and above)
    * CMake (version 3.8 and above)
    * LibTorch (version 2.1.0 and above)

### Build
Execute the following instructions in the **root directory** of the project. The generated executable files will be located in the `build/apps` directory.
```shell
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=<your_libtorch_path>
make -j
```

### Datasets
Support for matrix market format graph datasets.  
You can download graph datasets from the website https://ogb.stanford.edu/.

## How to Use
### Run the demo
You can directly run executable programs located in the `build/apps` directory after building the project.
```
./apps/sssp -i ../datasets/ak2010/ak2010.adj --src 0
```

## Support or Contact
GraphOne is developed at SCTS&CGCL Lab (http://grid.hust.edu.cn/) by Haoyan Huang, Yu Huang, and Long Zheng. For any questions, please contact Haoyan Huang (vastrockh@hust.edu.cn), Yu Huang
(yuh@hust.edu.cn) and Long Zheng (longzh@hust.edu.cn).
