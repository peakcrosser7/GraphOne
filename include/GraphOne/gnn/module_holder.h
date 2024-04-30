#pragma once

#include <memory>
#include <type_traits>

#include "GraphOne/utils/log.hpp"

namespace graph_one::gnn {

struct ModuleHolderIndicator {};

// A collection of templates that answer the question whether a type `T` is a
// `ModuleHolder`, and if so whether its contained type is of type `C`. This is
// tricky because it is hard to short circuit in template metaprogramming. A
// naive and incorrect solution to this problem would be something like
// `disable_if<is_module_holder<T>::value && typename T::ContainedType == C>`.
// This would disable all types that are not `ModuleHolder`s, because even
// though the `is_module_holder<T>::value` may be `false` for such types the
// `T::ContainedType` access would be ill-formed and thus fail the whole
// expression by the rules of SFINAE. Instead we have to use template
// specialization to statically branch on the first condition
// (`is_module_holder<T>`) and are only then allowed to query
// `T::ContainedType` in the branch for which the condition was true.

// A type trait that is true for types that are `ModuleHolder`s.
template <typename T>
using is_module_holder = std::is_base_of<ModuleHolderIndicator, std::decay_t<T>>;

// Base template.
template <bool is_module_holder_value, typename T, typename C>
struct is_module_holder_of_impl;

// False branch. `T` is not a `ModuleHolder` and thus not a `ModuleHolder` with
// contained type `C`.
template <typename T, typename C>
struct is_module_holder_of_impl<false, T, C> : std::false_type {};

// True branch. `T` is a `ModuleHolder` and thus we can legit access its
// `contained_type` and compare it against `C`.
template <typename T, typename C>
struct is_module_holder_of_impl<true, T, C>
    : std::is_same<typename T::contained_type, C> {};

// check `T` is a `ModuleHolder` and its contained module type is `C`
// Helper template.
template <typename T, typename C>
struct is_module_holder_of : is_module_holder_of_impl<
                                 is_module_holder<T>::value,
                                 std::decay_t<T>,
                                 std::decay_t<C>> {};

template <typename contained_t>
class ModuleHolder : ModuleHolderIndicator {
public:
    using contained_type = contained_t;

    /// Default constructs the contained module if if has a default constructor,
    /// else produces a static error.
    ModuleHolder() : impl_(default_construct()) {
        static_assert(
            std::is_default_constructible_v<contained_t>,
            "You are trying to default construct a module which has "
            "no default constructor. Use = nullptr to give it the empty state "
            "(e.g. `Linear linear = nullptr;` instead of `Linear linear;`).");
    }

    /// Constructs the `ModuleHolder` with an empty contained value. Access to
    /// the underlying module is not permitted and will throw an exception, until
    /// a value is assigned.
    ModuleHolder(std::nullptr_t) : impl_(nullptr) {}

    /// Constructs the `ModuleHolder` with a contained module, forwarding all
    /// arguments to its constructor.
    template <
        typename holder_t, typename... args_t,
        typename = std::enable_if_t<
            !(is_module_holder_of<holder_t, contained_type>::value &&
              (sizeof...(args_t) == 0))>
    >
    explicit ModuleHolder(holder_t &&holder, args_t &&...args)
        : impl_(new contained_t(std::forward<holder_t>(holder),
                              std::forward<args_t>(args)...)) {}

    ModuleHolder(std::shared_ptr<contained_t> module)
        : impl_(std::move(module)) {}

    /// Forwards to the contained module.
    contained_t *operator->() { 
        return get(); 
    }

    /// Forwards to the contained module.
    const contained_t *operator->() const { 
        return get(); 
    }

    template<typename... args_t>
    auto operator()(args_t&&... args) {
        return impl_->forward(std::forward<args_t>(args)...);
    }

    const std::shared_ptr<contained_t>& ptr() const {
        return impl_;
    } 

    /// Returns true if the `ModuleHolder` does not contain a module.
    bool is_empty() const noexcept {
        return impl_ == nullptr;
    }

    /// Returns a pointer to the underlying module.
    contained_t* get() {
        if(is_empty()) {
            LOG_ERROR("Accessing empty ModuleHolder");
        }
        return impl_.get();
    }

    /// Returns a const pointer to the underlying module.
    const contained_t* get() const {
        if(is_empty()) {
            LOG_ERROR("Accessing empty ModuleHolder");
        }
        return impl_.get();
    }

private:
    std::shared_ptr<contained_t> default_construct() {
        if constexpr (std::is_default_constructible_v<contained_t>) {
            return std::make_shared<contained_t>();
        } else {
            return nullptr;
        }   
    }

protected:
    std::shared_ptr<contained_t> impl_;

};

/// Defines a class `ModuleName` which inherits from `ModuleHolder` to provide a
/// wrapper over a `std::shared_ptr<ModuleImpl>`.
/// `Impl` is a type alias for `ModuleImpl` which provides a way to call static
/// method of `ModuleImpl`.
#define GRAPH_MODULE_IMPL(ModuleName, ModuleImpl)                        \
    class ModuleName : public graph_one::gnn::ModuleHolder<ModuleImpl> { \
    public:                                                              \
        using graph_one::gnn::ModuleHolder<ModuleImpl>::ModuleHolder;    \
        using impl_type = ModuleImpl;                                    \
    }

#define GRAPH_MODULE(ModuleName) GRAPH_MODULE_IMPL(ModuleName, ModuleName##Impl)

    
} // namespace graph_one