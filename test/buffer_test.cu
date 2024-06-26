#include "GraphOne/base/buffer.h"

#include "test.hpp"

using namespace std;
using namespace graph_one;

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
    Buffer<arch_t::cpu, int> buf(cnt);

    for (int i = 0; i < cnt; ++i) {
        buf[i] = i;
    }

    for (int i = 0; i < cnt; ++i) {
        cout << buf[i] << endl;
    }
    sort(buf.data(), buf.data() + buf.size(), greater<int>());

    Buffer<arch_t::cuda, int> gbuf;
    gbuf = buf;

    print<<<1,1>>>(gbuf.data(), gbuf.size());
    cudaDeviceSynchronize();

    Buffer<arch_t::cuda, int> ept_buf;
    cout << "ept_buf:" << ept_buf.ToString() << endl;

    constexpr int N_EPT = 3;
    Buffer<arch_t::cuda, int> ept_bufs[N_EPT];
    for (int i = 0; i < N_EPT; ++i) {
        cout << "ept_buf[" << i << "]:" << ept_buf.ToString() << endl;
    }
    
    return 0;
}