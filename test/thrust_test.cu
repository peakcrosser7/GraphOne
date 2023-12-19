#include "test.hpp"

#include "GraphGenlX/graph_genlx.h"
#include "GraphGenlX/archi/thrust/thrust.h"

using namespace std;
using namespace graph_genlx;

int main() {
    constexpr arch_t arch = arch_t::cuda;
    int sz = 10;
    Buffer<arch, int> output(sz);

    auto bypass = [] __GENLX_DEV__ (const int& i) {
        return -i;
    };

    archi::transform<arch>(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(sz),
        output.data(),
        bypass
    );

    cout << output.ToString() << endl;

    return 0;
}