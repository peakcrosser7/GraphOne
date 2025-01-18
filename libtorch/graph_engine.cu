#include <vector>

#include "cuda.h"

#include <torch/torch.h>

struct GraphX {
    torch::Tensor adj;
    torch::Tensor adj_t;

    std::vector<torch::Tensor> props;

};


namespace op {
    
struct Mult {

    __device__ __host__ constexpr static
    T identity() {
        return T(1);
    }

    template <typename T>
    __device__ __host__ constexpr static 
    T opeartor() (const T& lhs, const T& rhs) {
        return lhs * rhs;
    }
};

struct Add {
    __device__ __host__ constexpr static
    T identity() {
        return T(0);
    }

    template <typename T>
    __device__ __host__ static T opeartor() (const T& lhs, const T& rhs) {
        return lhs + rhs;
    }
};

struct DummyApply {
    template <typename T>
    __device__ __host__ static T opeartor() (const T& x) {
        return x;
    }
};

} // namespace op 


struct FowardOps {
    // use adj_t 
    bool apply_dst = true;

}

// ApplyVertex
template <typename combine_func_t, typename reduce_func_t, typename apply_func_t = DummyApply>
torch::Tensor GraphFroward(
    GraphX& g,
    torch::Tensor vertex_feat, torch::Tensor edge_feat,
    combine_func_t combine_func, 
    reduce_func_t reduce_func, apply_func_t apply_func = DummyApply{}) {

    torch::Tensor output;
    if constexpr (std::is_same_v<combine_func_t, op::Mult> 
        && std::is_same_v<reduce_func_t, op::Add>) {

    } else {

    }

}


template <typename combine_func_t, typename reduce_func_t, typename apply_func_t>
struct GraphFunctor {
    combine_func_t combine_func;
    reduce_func_t reduce_func;
    apply_func_t apply_func;
};

template <typename combine_func_t, typename reduce_func_t, typename apply_func_t = DummyApply>
GraphFunctor<combine_func_t, reduce_func_t, apply_func_t>
make_functor(combine_func_t combine_func, 
    reduce_func_t reduce_func, apply_func_t apply_func = DummyApply{}) {
    return GraphFunctor<combine_func_t, reduce_func_t, apply_func_t>(combine_func, reduce_func, apply_func);
}


// Apply Edge
template <typename combine_func_t, typename reduce_func_t, typename apply_func_t = DummyApply>
torch::Tensor GraphFroward(
    GraphX& g,
    torch::Tensor vertex_feat1, troch::Tensor vertex_feat2,
    torch::Tensor edge_feat,
    combine_func_t combine_func, 
    reduce_func_t reduce_func, apply_func_t apply_func = DummyApply{}) {

    torch::Tensor output;
    if constexpr (std::is_same_v<combine_func_t, op::Mult> 
        && std::is_same_v<reduce_func_t, op::Add>) {

    } else {

    }

}




int main() {

    auto std_functor = make_functor(op::Mult{}, op::Add{});


    GraphX g;
}
