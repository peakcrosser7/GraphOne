#pragma once

#include <cassert>

#include <torch/torch.h>

namespace graph_one {
    
struct BaseApplier {
    virtual torch::Tensor operator() (torch::Tensor x) const = 0;
};


struct DummyApplier : BaseApplier {
    torch::Tensor operator() (torch::Tensor x) const override {
        return x;
    }
};

class MaskApplier : public BaseApplier {
public:
    MaskApplier(torch::Tensor mask, torch::Scalar val = 0.) : mask_(mask), val_(val) {}

    torch::Tensor operator() (torch::Tensor x) const override {
        assert(x.layout() == mask_.layout());
        assert(x.sizes() == mask_.sizes());
        
        return torch::where(mask_.to(torch::kBool), x, val_);
    }

private:
    torch::Tensor mask_;
    torch::Scalar val_;

};

} // namespace graph_one
