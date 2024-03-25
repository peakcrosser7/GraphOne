#include "GraphOne/loader/loader.h"
#include "GraphOne/mat/convert.h"

#include "test.hpp"

using namespace std;
using namespace graph_one;

int main() {
    auto csr = Loader<>()
                   .LoadEdgesFromTxt<int>("../datasets/sample/sample.adj")
                   .ToCsr<arch_t::cuda>();
    cout << csr.ToString() << endl;

    auto csc = mat::ToCsc(csr);
    cout << csc.ToString() << endl;

    return 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
}