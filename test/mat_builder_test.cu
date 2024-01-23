#include "GraphGenlX/loader/loader.h"
#include "GraphGenlX/mat/convert.h"

#include "test.hpp"

using namespace std;
using namespace graph_genlx;

int main() {
    auto csr = Loader<>()
                   .LoadEdgesFromTxt<int>("../datasets/sample/sample.adj")
                   .ToCsr<arch_t::cuda>();
    cout << csr.ToString() << endl;

    auto csc = mat::ToCsc(csr);
    cout << csc.ToString() << endl;

    return 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
}