#pragma once

#include "GraphOne/frontier/base.h"


namespace graph_one {

class GnnFrontier : public BaseFrontier<DENSE_BASED, false> {
public:

    /// @brief converge according to the frontier
    /// @return default return false when not according to the frontier
    bool IsConvergent() override {
        return false;
    }
};
    
} // namespace graph_one
