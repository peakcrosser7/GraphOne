# GraphOne: A unified heterogeneous platform for graph analysis problems

GraphOne, a unified heterogeneous platform that supports graph analysis problems including graph computation, graph learning, and graph mining. This platform aims to provide users with a simple interface to implement graph algorithms and efficiently perform computations on heterogeneous devices. Currently, it supports the computation of graph computation problems on GPU devices.

## Setup
### Requirements
* Operating System: Linux (Ubuntu 18.04 and above)
* Hardware: NVIDIA GPU (preferably with a Compute Capability of 7.0 or higher)
* Software:
    * GCC (version 9.4.0 and above)
    * CUDA (version 11.8 and above)
    * CMake (version 3.12 and above)

### Build
Execute the following instructions in the **root directory** of the project. The generated executable files will be located in the `build/apps` directory.
```shell
mkdir build && cd build
cmake .. 
make
```

### Datasets
Support for adjacency list format graph datasets with configurable data delimiter and comment lines.  
You can download graph datasets from the website https://ogb.stanford.edu/.

## How to Use
### Run the demo
You can directly run executable programs located in the `build/apps` directory after building the project.
```
./apps/sssp_adv -i ../datasets/ak2010/ak2010.adj --src 1
```

### Implement Your Algorithms
You can refer to the source code files in the 'apps' directory to implement your graph algorithm program.  
There are two key points to consider.
First, inherit the `ComponentX` class to implement your algorithm logic, including the required status members and initialization functions.
```cpp
template <typename graph_t>
struct SSSPComp : ComponentX<graph_t, sssp_hstatus_t, sssp_dstatus_t> {
    using comp_t = ComponentX<graph_t, sssp_hstatus_t, sssp_dstatus_t>;
    using comp_t::ComponentX;

    void Init() override {
        auto& src_vid = this->h_status.src_vid;
        auto& dists = this->h_status.dists;

        archi::fill<arch>(dists.begin(), dists.end(), kMaxDist);
        dists.set(src_vid, 0);
    }

    void BeforeEngine() override {
        ++this->d_status.iter;
    }
};
```
Second, choose an appropriate algorithm execution engine and implement corresponding funtors.
```cpp
struct SSSPFunctor : BlasFunctor<vid_t, dist_t, sssp_dstatus_t, dist_t, dist_t> {

    __ONE_ARCH_INL__
    static dist_t default_info() {
        return kMaxDist;
    }

    __ONE_ARCH_INL__
    static dist_t default_result() {
        return kMaxDist;
    }

    __ONE_DEV_INL__
    static dist_t construct_each(const vid_t& vid, const sssp_dstatus_t& d_status) {
        return d_status.dists[vid];
    }

    __ONE_DEV_INL__
    static dist_t gather_combine(const dist_t& weight, const dist_t& info) {
        return (info == kMaxDist) ? info : weight + info;
    }

    __ONE_DEV_INL__
    static dist_t gather_reduce(const dist_t& lhs, const dist_t& rhs) {
        return std::min(lhs, rhs);
    }

    __ONE_DEV_INL__
    static bool apply_each(const vid_t& vid, const dist_t& res, sssp_dstatus_t& d_status) {
        if (res < d_status.dists[vid]) {
            d_status.dists[vid] = res;
            return true;
        }
        return false;
    }
};
```
Finally, call the `Run()` function to execute the program.
```cpp
DenseVec<arch, dist_t> dists(g.num_vertices());
DenseVec<arch, vid_t> visited(g.num_vertices());

sssp_hstatus_t h_status{src, dists};
sssp_dstatus_t d_status{0, dists.data()};
DblBufFrontier<arch, vid_t> frontier(g.num_vertices(), {src});

SSSPComp<graph_t> comp(g, h_status, d_status);

Run<SSSPFunctor>(comp, frontier);
```

## Support or Contact
GraphOne is developed at SCTS&CGCL Lab (http://grid.hust.edu.cn/) by Haoyan Huang, Yu Huang, and Long Zheng. For any questions, please contact Haoyan Huang (vastrockh@hust.edu.cn), Yu Huang
(yuh@hust.edu.cn) and Long Zheng (longzh@hust.edu.cn).
