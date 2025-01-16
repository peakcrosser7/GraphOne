#include <random>

#include "GraphOne/mat/dense.h"
#include "GraphOne/archi/blas/gemm/gemm.h"


using namespace graph_one;
using namespace std;

constexpr arch_t arch = arch_t::cuda;
using value_t = float;

int main() {
    int m = 64, n = 80, k = 40;
    DenseMat<arch_t::cpu, value_t> matA(m, k);
    DenseMat<arch_t::cpu, value_t> matB(k, n);

    random_device rd;
    std::mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < matA.capacity(); ++i) {
        matA.data()[i] = dis(gen);
    }
    
    for (int i = 0; i < matB.capacity(); ++i) {
        matB.data()[i] = dis(gen);
    }

    DenseMat<arch, value_t> matA_d = matA;
    DenseMat<arch, value_t> matB_d = matB;
    DenseMat<arch, value_t> matC_d(m, n);

    auto gemm = blas::MakeGemm<arch, blas::GemmCutlass,
                               true, true, true>(
        m, n, k,
        matA_d.data(), matB_d.data(), matC_d.data());
    
    gemm();

    printx("matC: ", matC_d);


    return 0;
}