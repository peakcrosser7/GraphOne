#include "GraphGenlX/loader/loader.h"
#include "GraphGenlX/mat/convert.h"

#include "test.hpp"

using namespace std;
using namespace graph_genlx;

int main() {
    auto csr = Loader<>()
                   .LoadCsrFromTxt<arch_t::cuda, int>("../data/sample/sample.adj");
    cout << csr.ToString() << endl;

    auto csc = mat::ToCsc(csr);
    cout << csc.ToString() << endl;

    return 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
}