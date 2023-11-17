#include "GraphGenlX/buffer.h"

#include "test.hpp"

using namespace std;
using namespace graph_genlx;

template<typename T>
__global__ void print(T* arr, int size) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        for (int i = 0; i < size; ++i) {
            printf("%d - %d\n", i, arr[i]);
        }
    }
}

int main() {
    int cnt = 10;
    Buffer<int, arch_t::cpu> buf(cnt);

    for (int i = 0; i < cnt; ++i) {
        buf[i] = i;
    }

    for (int i = 0; i < cnt; ++i) {
        cout << buf[i] << endl;
    }
    sort(buf.data(), buf.data() + buf.size(), greater<int>());

    Buffer<int, arch_t::cuda> gbuf;
    gbuf = buf;


    print<<<1,1>>>(gbuf.data(), gbuf.size());
    
    return 0;
}