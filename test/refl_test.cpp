// Look at this gist for implementation details.
#include <string>
#include <vector>
#include <iosfwd>

#include "GraphOne/utils/refl.hpp"

using namespace graph_one;

// Let's define a type data having three parameters in its ctor.
// And then do reflection over it.
struct inner {};

class data {
public:
    data(int t0, std::string t1, std::vector<int> t2)
    : t0_{std::move(t0)}, t1_{std::move(t1)}, t2_{std::move(t2)}
    {}


private:
    int t0_;
    std::string t1_;
    std::vector<int> t2_;
};

int main(int argc, char *argv[]) {
    // Check the number of ctor parameters was detected correctly.
    constexpr static auto data_ctor_nparams = refl::fields_number_ctor<data>(0);
    static_assert(data_ctor_nparams != 0);
    static_assert(data_ctor_nparams != 1);
    static_assert(data_ctor_nparams != 2);
    static_assert(data_ctor_nparams == 3);

    // Check the types of ctor parameters were detected correctly.
    using data_ctor_type = refl::as_tuple<data>;
    static_assert(std::is_same_v<data_ctor_type,
                                std::tuple<int, std::string, std::vector<int>>>);

    return 0;
}
