/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cute/config.hpp>
#include <cute/int_tuple.hpp>
#include <cute/stride.hpp>
#include <cute/underscore.hpp>
#include <cute/numeric/arithmetic_tuple.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/integral_ratio.hpp>
#include <cute/numeric/numeric_types.hpp>  // cute::sizeof_bits

namespace cute
{

// Aliases

template <class... Shapes>
using Shape = cute::tuple<Shapes...>;

template <class... Strides>
using Stride = cute::tuple<Strides...>;

template <class... Strides>
using Step = cute::tuple<Strides...>;

template <class... Coords>
using Coord = cute::tuple<Coords...>;

template <class... Layouts>
using Tile = cute::tuple<Layouts...>;

template <class... Ts>
CUTE_HOST_DEVICE constexpr
Shape<Ts...>
make_shape(Ts const&... t) {
  return {t...};
}
template <class... Ts>
CUTE_HOST_DEVICE constexpr
Stride<Ts...>
make_stride(Ts const&... t) {
  return {t...};
}
template <class... Ts>
CUTE_HOST_DEVICE constexpr
Step<Ts...>
make_step(Ts const&... t) {
  return {t...};
}
template <class... Ts>
CUTE_HOST_DEVICE constexpr
Coord<Ts...>
make_coord(Ts const&... t) {
  return {t...};
}
template <class... Ts>
CUTE_HOST_DEVICE constexpr
Tile<Ts...>
make_tile(Ts const&... t)
{
  return {t...};
}

//
// Layout
//

template <class Shape, class Stride = LayoutLeft::Apply<Shape> >
struct Layout
    : private cute::tuple<Shape, Stride>   // EBO for static layouts
{
  // Expensive in compilation time...
  //static_assert(is_congruent<Shape, Stride>::value, "Shape and Stride must be congruent");

  // NOTE: This defaults static Shapes/Strides correctly, but not dynamic
  CUTE_HOST_DEVICE constexpr
  Layout(Shape  const& shape  = {}, Stride const& stride = {})
      : cute::tuple<Shape, Stride>(shape, stride)
  {}

  //
  // Accessors
  //

  static constexpr int rank  = rank_v<Shape>;

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  layout() {
    return *this;
  }

  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  layout() const {
    return *this;
  }

  template <int... I>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  shape() {
    return get<0,I...>(static_cast<cute::tuple<Shape, Stride>&>(*this));
  }

  template <int... I>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  shape() const {
    return get<0,I...>(static_cast<cute::tuple<Shape, Stride> const&>(*this));
  }

  template <int... I>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  stride() {
    return get<1,I...>(static_cast<cute::tuple<Shape, Stride>&>(*this));
  }

  template <int... I>
  CUTE_HOST_DEVICE constexpr
  decltype(auto)
  stride() const {
    return get<1,I...>(static_cast<cute::tuple<Shape, Stride> const&>(*this));
  }

  //
  // Mappings
  //

  // Map a logical coordinate to a linear index (Coord has no Underscore slice operators)
  // OR
  // Slice the layout and return the sublayout (Coord has an Underscore slice op)
  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  auto
  operator()(Coord const& coord) const {
    if constexpr (has_underscore<Coord>::value) {
      return slice(coord, *this);
    } else {
      return crd2idx(coord, shape(), stride());
    }

    CUTE_GCC_UNREACHABLE;
  }

  // Convenience function for multi-dimensional coordinates
  template <class Coord0, class Coord1, class... Coords>
  CUTE_HOST_DEVICE constexpr
  auto
  operator()(Coord0 const& c0, Coord1 const& c1, Coords const&... cs) const {
    return operator()(make_coord(c0,c1,cs...));
  }

  //
  // Compose
  //

  template <class OtherLayout>
  CUTE_HOST_DEVICE constexpr
  auto
  compose(OtherLayout const& other) const {
    return composition(*this, other);
  }

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr
  auto
  compose(Layouts const&... layouts) const {
    return composition(*this, make_tile(layouts...));
  }

  template <class OtherShape>
  CUTE_HOST_DEVICE constexpr
  auto
  with_shape(OtherShape const& shape) const {
    return composition(*this, make_layout(shape));
  }

  template <class... Shapes>
  CUTE_HOST_DEVICE constexpr
  auto
  with_shape(Shapes const&... shapes) const {
    return composition(*this, make_layout(make_shape(shapes...)));
  }

  //
  // Tile
  //

  template <class OtherLayout>
  CUTE_HOST_DEVICE constexpr
  auto
  tile(OtherLayout const& other) const {
    return tiled_divide(*this, other);
  }

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr
  auto
  tile(Layouts const&... layouts) const {
    return tiled_divide(*this, make_tile(layouts...));
  }

  //
  // Utility
  //

  //
  // Index to Coordinate
  //

  // NOTE: Only valid for compact layouts

  // Return the (hierarchical) ND logical coordinate corresponding to the linear index
  // @post crd2idx(@a result, shape(), stride()) == idx
  // @post congruent(@a result, shape())
  template <class IInt,
            __CUTE_REQUIRES(is_integral<IInt>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_hier_coord(IInt const& idx) const {
    return cute::idx2crd(idx, shape(), stride());
  }

  // Return the (flat) ND logical coordinate corresponding to the linear index
  // @post crd2idx(@a result, shape(), stride()) == idx
  // @post rank(@a result) == rank(shape()) && depth(@a result) == 1
  template <class IInt,
            __CUTE_REQUIRES(is_integral<IInt>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_flat_coord(IInt const& idx) const {
    return cute::crd2crd(this->get_hier_coord(idx), shape(), repeat<rank>(Int<1>{}));
  }

  // Return the generalized column-major 1D logical coordinate corresponding to the linear index
  // @post crd2idx(@a result, shape(), stride()) == idx
  // @post is_integral<decltype(@a result)>::value
  template <class IInt,
            __CUTE_REQUIRES(is_integral<IInt>::value)>
  CUTE_HOST_DEVICE constexpr
  auto
  get_1d_coord(IInt const& idx) const {
    return cute::crd2idx(this->get_hier_coord(idx), shape());
  }

  //
  // Coordinate to Coordinate
  //

#if 0
  // Return the (hierarchical) ND logical coordinate corresponding to the linear index
  // @post congruent(@a result, shape())
  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  auto
  crd_2_hier_coord(Coord const& crd) const {
    return cute::crd2crd(crd, shape(), shape());
  }

  // Return the (flat) ND logical coordinate corresponding to the linear index
  // @post rank(@a result) == rank(shape()) && depth(@a result) == 1
  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  auto
  crd_2_flat_coord(Coord const& crd) const {
    return cute::crd2crd(crd, shape(), product_each(shape()));
  }

  // Return the generalized column-major 1D logical coordinate corresponding to the linear index
  // @post is_integral<decltype(@a result)>::value
  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  auto
  crd_2_1d_coord(Coord const& crd) const {
    //return cute::crd2crd(crd, shape(), product(shape()));
    return cute::crd2idx(crd, shape());
  }
#endif
};

// Equality, return a static or dynamic boolean
template <class ShapeA, class StrideA,
          class ShapeB, class StrideB>
CUTE_HOST_DEVICE constexpr
auto
operator==(Layout<ShapeA,StrideA> const& layoutA, Layout<ShapeB,StrideB> const& layoutB)
{
  return layoutA.shape() == layoutB.shape() && layoutA.stride() == layoutB.stride();
}

template <class Layout>
struct is_layout : false_type {};
template <class Shape, class Stride>
struct is_layout<Layout<Shape,Stride>> : true_type {};

//
// Layout construction
//

template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
make_layout(Shape const& shape, Stride const& stride)
{
  static_assert(is_tuple<Shape >::value || is_integral<Shape >::value);
  static_assert(is_tuple<Stride>::value || is_integral<Stride>::value);
  return Layout<Shape,Stride>(shape, stride);
}

template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
make_layout(Shape const& shape)
{
  static_assert(is_tuple<Shape >::value || is_integral<Shape >::value);
  return make_layout(shape, compact_major<LayoutLeft>(shape));
}

//
// Convenience tags for common layouts
//

template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
make_layout(Shape const& shape, LayoutLeft)
{
  return make_layout(shape, compact_major<LayoutLeft>(shape));
}

template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
make_layout(Shape const& shape, LayoutRight)
{
  return make_layout(shape, compact_major<LayoutRight>(shape));
}

//
// Construct a layout from multiple layouts by concatenation
//

// One argument overload
template <class Shape0, class Stride0>
CUTE_HOST_DEVICE constexpr
auto
make_layout(Layout<Shape0,Stride0> const& layout0)
{
  return make_layout(make_shape (layout0.shape() ),
                     make_stride(layout0.stride()));
}

// Two argument overload
template <class Shape0, class Stride0,
          class Shape1, class Stride1>
CUTE_HOST_DEVICE constexpr
auto
make_layout(Layout<Shape0,Stride0> const& layout0,
            Layout<Shape1,Stride1> const& layout1)
{
  return make_layout(make_shape (layout0.shape() , layout1.shape() ),
                     make_stride(layout0.stride(), layout1.stride()));
}

// Var argument overload
template <class Shape0, class Stride0,
          class Shape1, class Stride1,
          class... Shapes, class... Strides>
CUTE_HOST_DEVICE constexpr
auto
make_layout(Layout<Shape0,Stride0> const& layout0,
            Layout<Shape1,Stride1> const& layout1,
            Layout<Shapes,Strides> const&... layouts)
{
  return make_layout(make_shape (layout0.shape() , layout1.shape() , layouts.shape()... ),
                     make_stride(layout0.stride(), layout1.stride(), layouts.stride()...));
}

//
// Advanced Layout constructions
//

// Make a compact layout with shape @a shape and strides following the order induced by @a order.
// Dynamic values in @a order are ignored, considered large, and considered ordered from left to right.
// Example:
//   make_ordered_layout(Shape<_2,_2,_2,_2>{}, Step<_0,_2,_3,_1>{})
//     ->  (_2,_2,_2,_2):(_1,_4,_8,_2)
//   make_ordered_layout(make_shape(2,3,4,5), make_step(Int<2>{}, 67, 42, Int<50>{}))
//     -> (2,3,4,5):(_1,10,30,2)
template <class Shape, class Order>
CUTE_HOST_DEVICE constexpr
auto
make_ordered_layout(Shape const& shape, Order const& order)
{
  return make_layout(shape, compact_order(shape, order));
}

// Make a compact layout with the same shape as @a layout
//   and strides following the order induced by @a layout.stride().
// Static-0 strides in the input @a layout are preserved in the output.
// Example:
//   make_layout_like(Layout<Shape<_2,_2,_2,_2>, Stride<_0,_2,_4,_1>>{})
//     ->  (_2,_2,_2,_2):(_0,_2,_4,_1)
//   make_layout_like(make_layout(make_shape(2,3,4,5), make_stride(Int<0>{},42,Int<1>{},Int<0>{})))
//     -> (2,3,4,5):(_0,4,_1,_0)
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
make_layout_like(Layout<Shape,Stride> const& layout)
{
  return make_layout(layout.shape(),
                     compact_order(filter_zeros(layout.stride(), layout.shape()), layout.stride()));
}

// Make a compact layout with the same shape as @a layout
//   and strides following the order induced by @a layout.stride(),
//   except mode-0 is always stride-1 and generated column-major.
// The 0th mode is commonly used for MMA_Atoms or Copy_Atoms so this
//   generates the 0th mode with LayoutLeft (preserving stride-0s) regardless of the reference layout
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
make_fragment_like(Layout<Shape,Stride> const& layout)
{
  constexpr int R = Layout<Shape,Stride>::rank;
  if constexpr (R > 1 && is_static<Shape>::value) {
    return tiled_product(make_layout(get<0>(layout.shape()),
                                     compact_major<LayoutLeft>(filter_zeros(get<0>(layout.stride()), get<0>(layout.shape())))),
                         make_ordered_layout(take<1,R>(layout.shape()), take<1,R>(layout.stride())));
  } else {
    return make_layout(layout.shape());
  }

  CUTE_GCC_UNREACHABLE;
}

template <class Shape,
          __CUTE_REQUIRES(is_tuple<Shape>::value || is_integral<Shape>::value)>
CUTE_HOST_DEVICE constexpr
auto
make_fragment_like(Shape const& shape)
{
  return make_layout(shape);
}

//
// Make an identity layout that maps a coordinate to itself
//

template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
make_identity_layout(Shape const& shape)
{
  return make_layout(shape, make_basis_like(shape));
}

//
// Operations to manipulate Layouts like a tuple of pairs
//

// Return the Is...th sublayout.
// For Is... = <I0,I1,...,IN>, equivalent to get<IN>(...get<I1>(get<I0>(layout)))
template <size_t... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
get(Layout<Shape,Stride> const& layout)
{
  return make_layout(get<Is...>(layout.shape()),
                     get<Is...>(layout.stride()));
}

// Return a new layout with only the modes in the range [B,E)
template <int B, int E, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
take(Layout<Shape,Stride> const& layout)
{
  static_assert(B < E, "take: empty range error");
  static_assert(0 <= B && E <= Layout<Shape,Stride>::rank, "take: range out of bounds");
  return make_layout(take<B,E>(layout.shape()),
                     take<B,E>(layout.stride()));
}

// Return a new layout with only the modes Is... = <I0,I1,...,IN>
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
select(Layout<Shape,Stride> const& layout)
{
  return make_layout(select<Is...>(layout.shape()),
                     select<Is...>(layout.stride()));
}

// Return a layout with depth at most 1
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
flatten(Layout<Shape,Stride> const& layout)
{
  return make_layout(flatten(layout.shape()),
                     flatten(layout.stride()));
}

// Return a layout whose profile is congruent to TargetProfile
// @pre Input layout is flat, flatten(@a layout) == @a layout
// @pre Input layout can be folded to profile, rank(@a layout) == rank(flatten(@a target_profile))
// @post congruent(@a result, @a target_profile)
template <class Shape, class Stride, class TargetProfile>
CUTE_HOST_DEVICE constexpr
auto
unflatten(Layout<Shape,Stride> const& layout, TargetProfile const& target_profile)
{
  return make_layout(unflatten(layout.shape(),  target_profile),
                     unflatten(layout.stride(), target_profile));
}

//
// Utilities
//

// Return the sublayout of mode I...
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
decltype(auto)
layout(Layout<Shape,Stride> const& layout)
{
  if constexpr (sizeof...(Is) == 0) {
    return layout;
  } else {
    return get<Is...>(layout);
  }

  CUTE_GCC_UNREACHABLE;
}

// Return the shape of a mode
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
decltype(auto)
shape(Layout<Shape,Stride>& layout)
{
  return layout.template shape<Is...>();
}

template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
decltype(auto)
shape(Layout<Shape,Stride> const& layout)
{
  return layout.template shape<Is...>();
}

// Return the stride of a mode
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
decltype(auto)
stride(Layout<Shape,Stride>& layout)
{
  return layout.template stride<Is...>();
}

template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
decltype(auto)
stride(Layout<Shape,Stride> const& layout)
{
  return layout.template stride<Is...>();
}

// Return the number of elements in a mode
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
size(Layout<Shape,Stride> const& layout)
{
  return size(shape<Is...>(layout));
}

// Return the number of modes
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
rank(Layout<Shape,Stride> const& layout)
{
  return rank(shape<Is...>(layout));
}

// Return the depth of the layout
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
depth(Layout<Shape,Stride> const& layout)
{
  return depth(shape<Is...>(layout));
}

// Return the codomain shape of a mode
// @post size(coshape(@a a)) == cosize(@a a)
// @return C Coordinate with smallest elements such that
//           @a elem_less(sub_layout(c), C) for all c < size(@a sub_layout)
//           where sub_layout = get<Is...>(layout).
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
coshape(Layout<Shape,Stride> const& layout)
{
  // Protect against negative strides
  auto abs_sub_layout = make_layout(shape<Is...>(layout),
                                    transform_leaf(stride<Is...>(layout), abs_fn{}));
  auto co_coord = as_arithmetic_tuple(abs_sub_layout(size(abs_sub_layout) - Int<1>{}));
  return co_coord + repeat_like(co_coord, Int<1>{});
}

// Return the codomain size of a mode
// @return M smallest integer such that
//           @a sub_layout(c) < M for all c < size(@a sub_layout)
//           where sub_layout = get<Is...>(layout).
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
cosize(Layout<Shape,Stride> const& layout)
{
  return size(coshape<Is...>(layout));
}

template <class Layout>
using cosize_t = decltype(cosize(declval<Layout>()));

template <class Layout>
static constexpr auto cosize_v = cosize_t<Layout>::value;

// With crd2idx(coord, shape), makes sense to have crd2idx(coord, Layout) as well
template <class Coord, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
crd2idx(Coord const& c, Layout<Shape,Stride> const& layout)
{
  return crd2idx(c, layout.shape(), layout.stride());
}

//
// Slice and Dice a layout
//

template <class Coord, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
slice(Coord const& c, Layout<Shape,Stride> const& layout)
{
  return make_layout(slice(c, layout.shape()),
                     slice(c, layout.stride()));
}

template <class Coord, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
slice_and_offset(Coord const& c, Layout<Shape,Stride> const& layout)
{
  return cute::make_tuple(slice(c, layout), crd2idx(c, layout));
}

template <class Coord, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
dice(Coord const& c, Layout<Shape,Stride> const& layout)
{
  return make_layout(dice(c, layout.shape()),
                     dice(c, layout.stride()));
}

// Compute a pointer offset and (potentially modified) layout from a coordinate
// This exists so it can be overloaded for ComposedLayout
template <class Coord, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
domain_offset(Coord const& coord, Layout<Shape,Stride> const& layout)
{
  return cute::make_tuple(layout, layout(coord));
}

//
// Transform the modes of a layout
//

namespace detail {

template <class Tuple, class F, int... I>
CUTE_HOST_DEVICE constexpr
auto
transform_layout(Tuple const& t, F&& f, seq<I...>)
{
  return make_layout(f(get<I>(t))...);
}

template <class Tuple0, class Tuple1, class F, int... I, int... I0, int... I1>
CUTE_HOST_DEVICE constexpr
auto
transform_layout(Tuple0 const& t0, Tuple1 const& t1, F&& f, seq<I...>, seq<I0...>, seq<I1...>)
{
  return make_layout(f(get<I>(t0),get<I>(t1))..., get<I0>(t0)..., get<I1>(t1)...);
}

} // end namespace detail

template <class Tuple, class F>
CUTE_HOST_DEVICE constexpr
auto
transform_layout(Tuple const& t, F&& f)
{
  return detail::transform_layout(t, f, make_seq<decltype(rank(t))::value>{});
}

template <class Tuple0, class Tuple1, class F>
CUTE_HOST_DEVICE constexpr
auto
transform_layout(Tuple0 const& t0, Tuple1 const& t1, F&& f)
{
  constexpr int R0 = decltype(rank(t0))::value;
  constexpr int R1 = decltype(rank(t1))::value;
  constexpr int R  = (R0 < R1) ? R0 : R1;
  return detail::transform_layout(t0, t1, f, make_seq<R>{}, make_range<R,R0>{}, make_range<R,R1>{});
}

//
// Coalesce and Filter
//

namespace detail {

// Look at each element and the front of the stack (in order of priority)
// front(NewLayout)  get<I>(Layout)
//      s0:d0           _1:d1     =>  continue
//      _1:d0           s1:d1     =>  replace_front    s1:d1
//      s0:s1*d1        s1:d1     =>  replace_front s0*s1:d1
//      s0:d0           s1:d1     =>  prepend          s1:d1
//
// @pre OldShape and OldStride are flat
template <int I, class OldShape, class OldStride, class NewShape, class NewStride>
CUTE_HOST_DEVICE constexpr
auto
bw_coalesce(OldShape const& old_shape, OldStride const& old_stride,
            NewShape const& new_shape, NewStride const& new_stride)
{
  if constexpr (I == -1) {
    // Base case, we're done
    if constexpr (is_constant<1, NewShape>::value) {
      return Layout<_1,_0>{};
    } else {
      return Layout<NewShape,NewStride>{new_shape,new_stride};
    }
  } else if constexpr (is_constant<1, decltype(get<I>(old_shape))>::value) {
    // shape<I>(layout) == _1, skip it and continue
    return bw_coalesce<I-1>(old_shape, old_stride, new_shape, new_stride);
  } else if constexpr (is_constant<1, NewShape>::value) {
    // Replace our shape-1 with anything (Can only happen on input new_shape/new_stride)
    return bw_coalesce<I-1>(old_shape, old_stride, get<I>(old_shape), get<I>(old_stride));
  } else if constexpr (is_static<decltype(get<0>(new_shape))>::value &&
                       is_constant<true, decltype(get<I>(old_shape) * get<I>(old_stride) == get<0>(new_stride))>::value) {
    // Merge modes because the shapes and strides match
    return bw_coalesce<I-1>(old_shape, old_stride,
                            replace_front(new_shape,  get<I>(old_shape) * get<0>(new_shape)),
                            replace_front(new_stride, get<I>(old_stride)));
  } else {
    // Can't replace or merge, so prepend a new mode
    return bw_coalesce<I-1>(old_shape, old_stride,
                            prepend(new_shape,  get<I>(old_shape)),
                            prepend(new_stride, get<I>(old_stride)));
  }

  CUTE_GCC_UNREACHABLE;
}

// cute::coalesce promises to not change the Layout as a function from integers to codomain.
// It accomplishes this inside of the Layout's domain, but not always outside of the domain.
//   Example: (_4,_1):(_1,_0) coalesces to _4:_1.
// detail::coalesce_x preserves the Layout function inside its domain and outside.
//
// @post depth(@a result) <= 1
// @post for all i, 0 <= i, @a layout(i) == @a result(i)
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
coalesce_x(Layout<Shape,Stride> const& layout)
{
  auto flat_shape  = flatten(layout.shape());
  auto flat_stride = flatten(layout.stride());

  constexpr int R = decltype(rank(flat_shape))::value;
  if constexpr (is_constant<1, decltype(get<R-1>(flat_shape))>::value) {
    return detail::bw_coalesce<R-2>(flat_shape, flat_stride,             Int<2>{}, get<R-1>(flat_stride));
  } else {
    return detail::bw_coalesce<R-2>(flat_shape, flat_stride, get<R-1>(flat_shape), get<R-1>(flat_stride));
  }
}

// Apply coalesce_x at the terminals of trg_profile
template <class Shape, class Stride, class IntTuple>
CUTE_HOST_DEVICE constexpr
auto
coalesce_x(Layout<Shape,Stride> const& layout, IntTuple const& trg_profile)
{
  if constexpr (is_tuple<IntTuple>::value) {
    static_assert(tuple_size<IntTuple>::value <= Layout<Shape,Stride>::rank);
    return cute::transform_layout(layout, trg_profile, [](auto const& l, auto const& t) { return coalesce_x(l,t); });
  } else {
    return coalesce_x(layout);
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

// "Simplify" the layout by combining modes that are possible to combine
// Does not respect the shape of the layout, but does preserve total size
// @post size(@a result) == size(@a layout)
// @post depth(@a result) <= 1
// @post for all i, 0 <= i < size(@a layout), @a layout(i) == @a result(i)
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
coalesce(Layout<Shape,Stride> const& layout)
{
  auto flat_shape  = flatten(layout.shape());
  auto flat_stride = flatten(layout.stride());

  constexpr int R = decltype(rank(flat_shape))::value;
  return detail::bw_coalesce<R-2>(flat_shape, flat_stride, get<R-1>(flat_shape), get<R-1>(flat_stride));
}

// Apply coalesce at the terminals of trg_profile
template <class Shape, class Stride, class IntTuple>
CUTE_HOST_DEVICE constexpr
auto
coalesce(Layout<Shape,Stride> const& layout, IntTuple const& trg_profile)
{
  if constexpr (is_tuple<IntTuple>::value) {
    static_assert(tuple_size<IntTuple>::value <= Layout<Shape,Stride>::rank);
    return transform_layout(layout, trg_profile, [](auto const& l, auto const& t) { return coalesce(l,t); });
  } else {
    return coalesce(layout);
  }

  CUTE_GCC_UNREACHABLE;
}

// Combine static and dynamic modes of a shape.
// @post size(@a result) == size(@a shape)
// @post depth(@a result) <= 1
template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
coalesce(Shape const& shape)
{
  static_assert(is_integral<Shape>::value || is_tuple<Shape>::value);

  return cute::fold_first(flatten(shape), [](auto const& init, auto const& a) {
    if constexpr (is_static<decltype(back(init))>::value == is_static<decltype(a)>::value) {
      return replace_back(init, back(init) * a);  // Both static or both dynamic, coalesce and replace
    } else {
      return append(init, a);                     // Can't coalesce, so append
    }
  });
}

// Replace the modes in layout that have a 0-stride with a 1-size
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
filter_zeros(Layout<Shape,Stride> const& layout)
{
  return make_layout(filter_zeros(layout.stride(), layout.shape()), layout.stride());
}

// Replace the modes in layout that correspond to a 0 at the terminals of trg_profile with a 1-size
template <class Shape, class Stride, class IntTuple>
CUTE_HOST_DEVICE constexpr
auto
filter_zeros(Layout<Shape,Stride> const& layout, IntTuple const& trg_profile)
{
  return make_layout(filter_zeros(trg_profile, layout.shape()), layout.stride());
}

// Remove all of the 0-strides and 1-sizes
// Return 1-shape if empty
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
filter(Layout<Shape,Stride> const& layout)
{
  return coalesce(filter_zeros(layout));
}

// Apply filter at the terminals of trg_profile
template <class Shape, class Stride, class IntTuple>
CUTE_HOST_DEVICE constexpr
auto
filter(Layout<Shape,Stride> const& layout, IntTuple const& trg_profile)
{
  if constexpr (is_tuple<IntTuple>::value) {
    static_assert(tuple_size<IntTuple>::value <= Layout<Shape,Stride>::rank);
    return transform_layout(layout, trg_profile, [](auto const& l, auto const& t) { return filter(l,t); });
  } else {
    return filter(layout);
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Append, Prepend, Replace
//

template <int N, class ShapeA, class StrideA, class ShapeX = _1, class StrideX = _0>
CUTE_HOST_DEVICE constexpr
auto
append(Layout<ShapeA,StrideA> const& layout,
       Layout<ShapeX,StrideX> const& x = {})
{
  return make_layout(append<N>(layout.shape(),  x.shape()),
                     append<N>(layout.stride(), x.stride()));
}

template <class ShapeA, class StrideA, class ShapeX = _1, class StrideX = _0>
CUTE_HOST_DEVICE constexpr
auto
append(Layout<ShapeA,StrideA> const& layout,
       Layout<ShapeX,StrideX> const& x = {})
{
  return make_layout(append(layout.shape(),  x.shape()),
                     append(layout.stride(), x.stride()));
}

template <int N, class ShapeA, class StrideA, class ShapeX = _1, class StrideX = _0>
CUTE_HOST_DEVICE constexpr
auto
prepend(Layout<ShapeA,StrideA> const& layout,
        Layout<ShapeX,StrideX> const& x = {})
{
  return make_layout(prepend<N>(layout.shape(),  x.shape()),
                     prepend<N>(layout.stride(), x.stride()));
}

template <class ShapeA, class StrideA, class ShapeX = _1, class StrideX = _0>
CUTE_HOST_DEVICE constexpr
auto
prepend(Layout<ShapeA,StrideA> const& layout,
        Layout<ShapeX,StrideX> const& x = {})
{
  return make_layout(prepend(layout.shape(),  x.shape()),
                     prepend(layout.stride(), x.stride()));
}

template <int N, class ShapeA, class StrideA, class ShapeX, class StrideX>
CUTE_HOST_DEVICE constexpr
auto
replace(Layout<ShapeA,StrideA> const& layout,
        Layout<ShapeX,StrideX> const& x)
{
  return make_layout(replace<N>(layout.shape(),  x.shape()),
                     replace<N>(layout.stride(), x.stride()));
}

template <int B, int E, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
group(Layout<Shape,Stride> const& layout)
{
  return make_layout(group<B,E>(layout.shape()),
                     group<B,E>(layout.stride()));
}

//
// Composition of two layouts: lhs o rhs
// @post compatible(rhs, result)
// @post result(c) = lhs(rhs(c))
//         for all c in the domain of rhs
//

namespace detail {

template <class LShape, class LStride,
          class RShape, class RStride>
CUTE_HOST_DEVICE constexpr
auto
composition_impl(LShape const& lhs_shape, LStride const& lhs_stride,
                 RShape const& rhs_shape, RStride const& rhs_stride)
{
  if constexpr (is_tuple<RShape>::value) {
    // Apply the right-distributivity of Layout composition
    return transform_layout(rhs_shape, rhs_stride, [&](auto const& s, auto const& d) {
      return composition_impl(lhs_shape, lhs_stride, s, d);
    });
  } else
  if constexpr (is_scaled_basis<RStride>::value) {
    // Special case for a ScaledBasis stride
    return composition_impl(basis_get(rhs_stride, lhs_shape), basis_get(rhs_stride, lhs_stride),
                            rhs_shape, basis_value(rhs_stride));
  } else
  if constexpr (is_constant<0, RStride>::value) {
    // Special case shortcut for any static stride-0
    return Layout<RShape, RStride>{rhs_shape, rhs_stride};
  } else
  if constexpr (is_integral<decltype(lhs_shape)>::value) {
    // Special case shortcut for any integral LShape
    return Layout{rhs_shape, rhs_stride * lhs_stride};
  } else
  if constexpr (is_constant<1, RStride>::value) {
    // Special case shortcut for any static stride-1
    constexpr int R  = rank_v<LShape>;
    auto result_shape_0  = take<0,R-1>(lhs_shape);

    // Mod out the rhs_shape from the lhs_shape
    auto const [result_shape_1, rest_shape]  = fold(result_shape_0, cute::make_tuple(cute::make_tuple(), rhs_shape),
      [] (auto const& init, auto const& si) {
        return cute::make_tuple(append(get<0>(init), shape_min(abs(si), get<1>(init))), shape_div(get<1>(init), abs(si)));
      });

    // Jump into coalesce and append (rest_shape, get<R-1>(lhs_stride))
    return detail::bw_coalesce<R-2>(result_shape_1, lhs_stride, rest_shape, get<R-1>(lhs_stride));
  } else {
    // General case: integral RShape and RStride, tuple LShape and LStride
    constexpr int R  = rank_v<LShape>;
    auto result_shape_0  = take<0,R-1>(lhs_shape);
    auto result_stride_0 = take<0,R-1>(lhs_stride);

    // Divide out the rhs_stride from the lhs_shape
    auto const [result_shape_1, rest_stride] = fold(result_shape_0, cute::make_tuple(cute::make_tuple(), rhs_stride),
      [] (auto const& init, auto const& di) {
        return cute::make_tuple(append(get<0>(init), shape_div(di, get<1>(init))), shape_div(get<1>(init), di));
      });

    // Apply any lhs_shape changes to the stride
    auto result_stride_1 = elem_scale(result_stride_0, shape_div(result_shape_0, result_shape_1));

    // Mod out the rhs_shape from the lhs_shape
    auto const [result_shape_2, rest_shape] = fold(result_shape_1, cute::make_tuple(cute::make_tuple(), rhs_shape),
      [] (auto const& init, auto const& si) {
        return cute::make_tuple(append(get<0>(init), shape_min(abs(si), get<1>(init))), shape_div(get<1>(init), abs(si)));
      });

    // Jump into coalesce and append (rest_shape, rest_stride * get<R-1>(lhs_stride))
    return detail::bw_coalesce<R-2>(result_shape_2, result_stride_1, rest_shape, rest_stride * get<R-1>(lhs_stride));
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

template <class LShape, class LStride,
          class RShape, class RStride>
CUTE_HOST_DEVICE constexpr
auto
composition(Layout<LShape,LStride> const& lhs,
            Layout<RShape,RStride> const& rhs)
{
  auto coprofile = repeat_like(decltype(coshape(rhs)){}, Int<0>{});
  auto flat_lhs = detail::coalesce_x(lhs, coprofile);
  return detail::composition_impl(flat_lhs.shape(), flat_lhs.stride(), rhs.shape(), rhs.stride());
}

template <class LShape, class LStride, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
composition(Layout<LShape,LStride> const& lhs,
            Tiler                  const& rhs)
{
  if constexpr (is_tuple<Tiler>::value) {
    static_assert(tuple_size<Tiler>::value <= Layout<LShape,LStride>::rank);
    // Drop any modes of lhs that aren't hit by rhs
    return detail::transform_layout(lhs, rhs, [](auto const& l, auto const& r) { return composition(l,r); }, make_seq<tuple_size<Tiler>::value>{}, seq<>{}, seq<>{});
  } else if constexpr (is_underscore<Tiler>::value) {
    return lhs;
  } else if constexpr (is_integral<Tiler>::value) {
    auto flat_lhs = detail::coalesce_x(lhs);
    return detail::composition_impl(flat_lhs.shape(), flat_lhs.stride(), rhs, Int<1>{});
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Complement
//
// Build the complement of a layout.
// @post size(@a result) >= @a cosize_hi / size(filter(@a layout)));
// @post For all i in [1,size(@a result)),
//           @a result(i) < @a result(i-1)
//           For all j in [0, size(@a layout)),
//               @a result(i) != @a layout(j)
//

namespace detail {

// @pre @a layout has been filtered (flattened and no stride-0 or size-1 modes).
template <class Shape, class Stride, class CoTarget>
CUTE_HOST_DEVICE constexpr
auto
complement(Shape const& shape, Stride const& stride, CoTarget const& cotarget)
{
  if constexpr (is_constant<0, Stride>::value) {
    // Special case for irreducible rank-1 stride-0 layout
    return make_layout(coalesce(cotarget));
  } else {
    // General case
    constexpr int R = rank_v<Shape>;
    static_assert(R == 1 || is_static<Stride>::value,
                  "Dynamic-stride complement only for rank-1 layouts");

    // Should just be a sort and a fold...
    // Then we could even handle dynamic strides (but they would destroy all static strides)
    auto [shape_, stride_, result_shape_, result_stride] =
      fold(make_seq<R-1>{},
           cute::make_tuple(shape, stride, cute::make_tuple(), cute::make_tuple(Int<1>{})),
           [](auto const& init, auto i)
           {
              auto [shape, stride, result_shape, result_stride] = init;
              auto min_stride = cute::min(stride);
              auto min_idx    = cute::find(stride, min_stride);
              auto new_shape  = min_stride / get<i>(result_stride);
              auto new_stride = min_stride * get<min_idx>(shape);
              static_assert(not is_constant<0, decltype(new_shape)>::value, "Non-injective Layout detected in complement.");

              return cute::make_tuple(remove<min_idx>(shape),              // Remove the min_idx from shape
                                      remove<min_idx>(stride),             // Remove the min_idx from stride
                                      append(result_shape , new_shape ),   // new shape  = min_stride / last_stride
                                      append(result_stride, new_stride));  // new stride = min_stride * curr_shape
            });

    // Append the last shape mode
    auto new_shape    = get<0>(stride_) / get<R-1>(result_stride);         // new shape  = min_stride / last_stride
    static_assert(not is_constant<0, decltype(new_shape)>::value, "Non-injective Layout detected in complement.");
    auto result_shape = append(result_shape_, new_shape);

    // Compute the rest_shape and rest_stride
    auto new_stride  = get<0>(stride_) * get<0>(shape_);                   // new stride = min_stride * curr_shape
    auto rest_shape  = coalesce(ceil_div(cotarget, new_stride));
    auto rest_stride = compact_major<LayoutLeft>(rest_shape, new_stride);

    // Coalesce and append (rest_shape, rest_stride)
    return coalesce(make_layout(make_shape (result_shape , rest_shape ),
                                make_stride(result_stride, rest_stride)));
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

template <class Shape, class Stride, class CoTarget>
CUTE_HOST_DEVICE constexpr
auto
complement(Layout<Shape,Stride> const& layout, CoTarget const& cotarget)
{
  auto filter_layout = filter(layout);
  return detail::complement(filter_layout.shape(), filter_layout.stride(), shape(cotarget));
}

template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
complement(Layout<Shape,Stride> const& layout)
{
  auto filter_layout = filter(layout);
  return detail::complement(filter_layout.shape(), filter_layout.stride(), cosize(filter_layout));
}

//
// Right-Inverse and Left-Inverse
//

namespace detail {

template <int NextStride, class Shape, class Stride, int... Is>
CUTE_HOST_DEVICE constexpr
auto
inverse_seq(Shape const& shape, Stride const& stride, seq<Is...>)
{
  auto next_I = cute::find_if(stride, [](auto a) { return is_constant<NextStride, decltype(a)>{}; });

  if constexpr (next_I == decltype(rank(stride))::value) {
    // If not found, return current seq
    return seq<Is...>{};
  } else {
    // auto next_stride = get<next_I>(shape) * get<next_I>(stride);
    // NOTE: Needed for g++-7
    using next_stride = decltype(get<next_I>(shape) * get<next_I>(stride));

    if constexpr (is_static<next_stride>::value && !is_constant<NextStride, next_stride>::value) {
      // If next_stride is static and unique, then continue
      return inverse_seq<next_stride::value>(shape, stride, seq<Is..., next_I>{});
    } else {
      // Else return current seq + next_I
      return seq<Is..., next_I>{};
    }
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

//
// Build the right-inverse of a layout
// @pre is_static<Layout>
// @result A layout @a result such that
//    @a layout(@a result(i)) == i for all i < size(@a result)
// @result A layout @a result such that
//    composition(@a layout, @a result) is identical to make_layout(shape(result))
//

template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
right_inverse(Layout<Shape,Stride> const& layout)
{
  auto flat_layout = coalesce(layout);
  auto astride = transform_leaf(flat_layout.stride(), abs_fn{});

  // Find Int<1>{}, the starting stride, and follow the strides to gen inverse_seq
  [[maybe_unused]] auto iseq = detail::inverse_seq<1>(flat_layout.shape(), astride, seq<>{});

  if constexpr (iseq.size() == 0) {
    return Layout<_1,_0>{};     // Empty case, nothing found
  } else {
    // Generate the corresponding new strides and construct
    auto rstride = compact_major<LayoutLeft>(flat_layout.shape());
    return make_layout(unwrap(transform(iseq, [&](auto i) { return shape<i>(flat_layout); })),
                       unwrap(transform(iseq, [&](auto i) { return signum(stride<i>(flat_layout)) * get<i>(rstride); })));
  }

  CUTE_GCC_UNREACHABLE;
}

CUTE_HOST_DEVICE constexpr
auto
right_inverse(Underscore const& _)
{
  return _;
}

//
// Build the left-inverse of a layout
// @pre is_static<Layout>
// @pre @a layout is an injective function
// @result A layout @a result such that
//    @a result(@a layout(i)) == i for all i < size(@a layout)
// @result A layout @a result such that
//    composition(@a result, @a layout) is identical to make_layout(shape(layout))
//

template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
left_inverse(Layout<Shape,Stride> const& layout)
{
  return right_inverse(make_layout(layout, complement(layout)));
}

CUTE_HOST_DEVICE constexpr
auto
left_inverse(Underscore const& _)
{
  return _;
}

//
// Max Common Layout
//

/* Return a layout that points to the maximum number of contiguous elements
 * that logically correspond in the layouts of @a a and @a b.
 *
 * @returns Layout R
 * @post For all 0 <= i < size(R), a(R(i)) == i and b(R(i)) == i
 */
template <class ShapeA, class StrideA,
          class ShapeB, class StrideB>
CUTE_HOST_DEVICE constexpr
auto
max_common_layout(Layout<ShapeA,StrideA> const& a,
                  Layout<ShapeB,StrideB> const& b)
{
  Layout inv_b  = right_inverse(b);
  Layout common = coalesce(composition(a, inv_b));

  // Keep only the static identity component of the common layout
  if constexpr (is_static<decltype(shape<0>(common))>::value &&
                is_constant<1, decltype(stride<0>(common))>::value) {
    // Truncate to the size of the contiguous vector (static stride-1 mode)
    return composition(inv_b, layout<0>(common));
  } else {
    return Layout<_1,_0>{};
  }
}

/* Return Int<N> such that N is the maximum number of contiguous elements
 * that logically correspond in the layouts of @a a and @a b.
 *
 * @returns Int<N> with N >= 1
 * @post For all 0 <= n < N, a(b.get_1d_coord(n)) == n
 *       (NOTE: Problems with negative strides/coords in this post-condition)
 */
template <class ShapeA, class StrideA,
          class ShapeB, class StrideB>
CUTE_HOST_DEVICE constexpr
auto
max_common_vector(Layout<ShapeA,StrideA> const& a,
                  Layout<ShapeB,StrideB> const& b)
{
  Layout common = coalesce(composition(a, right_inverse(b)));

  // Keep only the static identity component of the common layout
  if constexpr (is_static<decltype(shape<0>(common))>::value &&
                is_constant<1, decltype(stride<0>(common))>::value) {
    // Truncate to the size of the contiguous vector (static stride-1 mode)
    return shape<0>(common);
  } else {
    return Int<1>{};
  }

  CUTE_GCC_UNREACHABLE;
}

/* Return a layout that distributes ShapeB over ShapeA.
 *
 * @returns Layout result
 * @post evenly_divides(@a b, size(@a result))
 * @post evenly_divides(@a a, @a result)
 * @post For all i,j in [0,size(@a result)) with i < j, @a result(i) < @a result(j). Surjective and Ordered.
 * @post composition(make_layout(shape(@a a)), @a result) is admissible
 * \code
 *   // Note that 6 does not divide this shape
 *   Layout layoutA = Layout<Shape<Int<15>,Int<14>>>{};
 *
 *   // Want to tile any 6 elements and don't care where they come from
 *   Layout dist = domain_distribute(layoutA, Int<6>{});   // (_3,_2):(_1,_15)
 *
 *   // Not guaranteed to find all 6 though...
 *   CUTE_STATIC_ASSERT_V(Int<6>{} == size(dist));
 *
 *   Layout result = zipped_divide(layoutA, dist);         // (_6,Rest)
 * \endcode
 */
template <class ShapeA, class ShapeB>
CUTE_HOST_DEVICE constexpr
auto
domain_distribute(ShapeA const& a, ShapeB const& b)
{
  static_assert(is_integral<ShapeB>::value);
  static_assert(is_static<ShapeB>::value);

  auto flat_shape_a = flatten(shape(a));

  static_assert(is_static<decltype(flat_shape_a)>::value);

  // Compute the shape of the result
  auto [result_shape, b_rest] = cute::fold(flat_shape_a, cute::make_tuple(cute::tuple<>{}, size(b)), [](auto init, auto a_) {
    auto [result, b_] = init;
    auto gcd_ = gcd(a_, b_);
    return cute::make_tuple(append(result, gcd_), b_ / gcd_);
  });

  // Compute the stride of the result
  auto result_stride = compact_major<LayoutLeft>(flat_shape_a);

  return coalesce(make_layout(result_shape, result_stride));
}

//
// Kernel (Nullspace) of a Layout
//

namespace detail {

template <int NextI, class Stride, int... Is>
CUTE_HOST_DEVICE constexpr
auto
nullspace_seq(Stride const& stride, seq<Is...>)
{
  if constexpr (NextI == rank_v<Stride>) {
    return seq<Is...>{};
  } else
  if constexpr (is_constant<0, decltype(get<NextI>(stride))>::value) {
    return detail::nullspace_seq<NextI+1>(stride, seq<Is..., NextI>{});
  } else {
    return detail::nullspace_seq<NextI+1>(stride, seq<Is...>{});
  }

  CUTE_GCC_UNREACHABLE;
}

} // end namespace detail

//
// Build the nullspace of a layout
// @result A layout @a result such that
//    size(@a result) == size(@a layout) / size(filter(@a layout))
//    @a layout(@a result(i)) == 0 for all i < size(@a result)
//

template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
nullspace(Layout<Shape,Stride> const& layout)
{
  auto flat_layout = flatten(layout);

  auto iseq = detail::nullspace_seq<0>(flat_layout.stride(), seq<>{});

  if constexpr (iseq.size() == 0) {
    return Layout<_1,_0>{};     // Empty case, nothing found
  } else {
    // Generate the corresponding new strides and construct
    auto rstride = compact_major<LayoutLeft>(flat_layout.shape());
    return make_layout(unwrap(transform(iseq, [&](auto i) { return shape<i>(flat_layout); })),
                       unwrap(transform(iseq, [&](auto i) { return get<i>(rstride); })));
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Zip
//

template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
zip(Layout<Shape,Stride> const& layout)
{
  return make_layout(zip(layout.shape()),
                     zip(layout.stride()));
}

template <class TShape, class TStride,
          class UShape, class UStride>
CUTE_HOST_DEVICE constexpr
auto
zip(Layout<TShape,TStride> const& layoutA,
    Layout<UShape,UStride> const& layoutB)
{
  return make_layout(zip(layoutA.shape(),  layoutB.shape()),
                     zip(layoutA.stride(), layoutB.stride()));
}

//
// Tile unzip
//   Logical product and logical divide (on layouts) produce rank-2 results by design.
//   Follow the profile of @a tile and zip the rank-2 modes located at the terminals into
//   their own mode.
//

template <class LShape, class LStride, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
tile_unzip(Layout<LShape,LStride> const& layout,
           Tiler                  const& tiler)
{
  return make_layout(zip2_by(layout.shape(),  tiler),
                     zip2_by(layout.stride(), tiler));
}

//
// Logical divide
//

template <class LShape, class LStride,
          class TShape, class TStride>
CUTE_HOST_DEVICE constexpr
auto
logical_divide(Layout<LShape,LStride> const& layout,
               Layout<TShape,TStride> const& tiler)
{
  return composition(layout, make_layout(tiler, complement(tiler, shape(layout))));
}

template <class LShape, class LStride, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
logical_divide(Layout<LShape,LStride> const& layout,
               Tiler                  const& tiler)
{
  if constexpr (is_tuple<Tiler>::value) {
    static_assert(tuple_size<Tiler>::value <= Layout<LShape,LStride>::rank, "logical_divide: Too many modes in tiler.");
    return transform_layout(layout, tiler, [](auto const& l, auto const& t) { return logical_divide(l,t); });
  } else if constexpr (is_underscore<Tiler>::value) {
    return layout;
  } else if constexpr (is_integral<Tiler>::value) {
    return logical_divide(layout, make_layout(tiler));
  }

  CUTE_GCC_UNREACHABLE;
}

// Generalization of ceil_div for Layout lhs
//   is effectively the "rest mode" of logical_divide.
// Occurs in the calculation of gridDim, for example, for generalized tilers
// Example:
//   dim3 gridDim(size(ceil_div(problem_shape_M, cta_tiler_M)),
//                size(ceil_div(problem_shape_N, cta_tiler_N)));
// This does not consider compositional acceptance, so it may be the case that
//   ceil_div produces a result while logical_divide (and friends) do not.
template <class Target, class TShape, class TStride>
CUTE_HOST_DEVICE constexpr
auto
ceil_div(Target                 const& target,
         Layout<TShape,TStride> const& tiler)
{
  return shape(complement(tiler, shape(target)));
}

//
// Convenience operator
//   that produces layouts like ((BLK_A,BLK_B,...),(a,b,...,x,y))
//   by gathering the tile modes and residuals into a rank-2 result.
//

template <class LShape, class LStride,
          class Tiler>
CUTE_HOST_DEVICE constexpr
auto
zipped_divide(Layout<LShape,LStride> const& layout,
              Tiler                  const& tiler)
{
  return tile_unzip(logical_divide(layout, tiler), tiler);
}

// Same as zipped_divide, but unpacks the second mode: ((BLK_A,BLK_B,...),a,b,...,x,y)
template <class LShape, class LStride,
          class Tiler>
CUTE_HOST_DEVICE constexpr
auto
tiled_divide(Layout<LShape,LStride> const& layout,
             Tiler                  const& tiler)
{
  auto result = zipped_divide(layout, tiler);

  auto R1 = rank<1>(result);
  return result(_, repeat<R1>(_));
}

// Same as zipped_divide, but unpacks both modes: (BLK_A,BLK_B,...,a,b,...,x,y)
template <class LShape, class LStride,
          class Tiler>
CUTE_HOST_DEVICE constexpr
auto
flat_divide(Layout<LShape,LStride> const& layout,
            Tiler                  const& tiler)
{
  auto result = zipped_divide(layout, tiler);

  auto R0 = rank<0>(result);
  auto R1 = rank<1>(result);
  return result(repeat<R0>(_), repeat<R1>(_));
}

//
// Logical product
//

template <class LShape, class LStride,
          class TShape, class TStride>
CUTE_HOST_DEVICE constexpr
auto
logical_product(Layout<LShape,LStride> const& block,
                Layout<TShape,TStride> const& tiler)
{
  return make_layout(block, composition(complement(block, size(block)*cosize(tiler)), tiler));
}

template <class LShape, class LStride, class Tiler>
CUTE_HOST_DEVICE constexpr
auto
logical_product(Layout<LShape,LStride> const& block,
                Tiler                  const& tiler)
{
  if constexpr (is_tuple<Tiler>::value) {
    static_assert(tuple_size<Tiler>::value <= Layout<LShape,LStride>::rank, "logical_product: Too many modes in tiler.");
    return transform_layout(block, tiler, [](auto const& l, auto const& t) { return logical_product(l,t); });
  } else if constexpr (is_underscore<Tiler>::value) {
    return block;
  } else if constexpr (is_integral<Tiler>::value) {
    return logical_product(block, make_layout(tiler));
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Convenience operator
//   that produces layouts like ((BLK_A,BLK_B,...),(a,b,...,x,y))
//   by gathering the block modes and products into a rank-2 result.
//

template <class LShape, class LStride,
          class Tiler>
CUTE_HOST_DEVICE constexpr
auto
zipped_product(Layout<LShape,LStride> const& block,
               Tiler                  const& tiler)
{
  return tile_unzip(logical_product(block, tiler), tiler);
}

// Same as zipped_product, but unpacks the second mode: ((BLK_A,BLK_B,...),a,b,...,x,y)
template <class LShape, class LStride,
          class Tiler>
CUTE_HOST_DEVICE constexpr
auto
tiled_product(Layout<LShape,LStride> const& block,
              Tiler                  const& tiler)
{
  auto result = zipped_product(block, tiler);

  auto R1 = rank<1>(result);
  return result(_, repeat<R1>(_));
}

// Same as zipped_product, but unpacks both modes: (BLK_A,BLK_B,...,a,b,...,x,y)
template <class LShape, class LStride,
          class Tiler>
CUTE_HOST_DEVICE constexpr
auto
flat_product(Layout<LShape,LStride> const& block,
             Tiler                  const& tiler)
{
  auto result = zipped_product(block, tiler);

  auto R0 = rank<0>(result);
  auto R1 = rank<1>(result);
  return result(repeat<R0>(_), repeat<R1>(_));
}

//
// Rank-sensitive products
//

// blocked_product -- Reproduce a block over a tiler.
// Think of every element of "tiler" as a "block"
//   and return the layout of the resulting structure.
// @post rank(@a result) == cute::max(rank(@a block), rank(@a tiler))
template <class TShape, class TStride,
          class UShape, class UStride>
CUTE_HOST_DEVICE constexpr
auto
blocked_product(Layout<TShape,TStride> const& block,
                Layout<UShape,UStride> const& tiler)
{
  constexpr int R = cute::max(rank_v<TShape>, rank_v<UShape>);

  auto result = logical_product(append<R>(block), append<R>(tiler));

  return coalesce(zip(get<0>(result), get<1>(result)), tuple_repeat<R>(Int<1>{}));
}

// raked_product -- Reproduce a block over a tiler with block-interleaving.
// Think of every element of "tiler" as a "block", interleave those blocks,
//   and return the layout of the resulting structure.
// @post rank(@a result) == cute::max(rank(@a block), rank(@a tiler))
template <class TShape, class TStride,
          class UShape, class UStride>
CUTE_HOST_DEVICE constexpr
auto
raked_product(Layout<TShape,TStride> const& block,
              Layout<UShape,UStride> const& tiler)
{
  constexpr int R = cute::max(rank_v<TShape>, rank_v<UShape>);

  auto result = logical_product(append<R>(block), append<R>(tiler));

  return coalesce(zip(get<1>(result), get<0>(result)), tuple_repeat<R>(Int<1>{}));
}

// tile_to_shape -- Perform a product of a layout so that the result matches a target shape.
// This is similar to blocked_product, but specifies the result shape instead of the
//   product shape, which is more convenient in certain circumstances.
// @param block The layout to repeat
// @param trg_shape The target shape of the result
// @param ord_shape The order of the modes of @a trg_shape to tile @a layout with.
//                  Defaults to GenColMajor, so @a layout will repeat
//                    across the first mode first, the second mode second, etc
//                  E.g. Step<_2,_1,_3> will cause @a layout to repeat
//                    across the second mode first, the first mode second, and the third mode last.
// @pre rank(@a block) <= rank(@a trg_shape)
// @post compatible(@a trg_shape, shape(@a result))
template <class Shape, class Stride,
          class TrgShape, class ModeOrder = LayoutLeft>
CUTE_HOST_DEVICE constexpr
auto
tile_to_shape(Layout<Shape,Stride> const& block,
              TrgShape             const& trg_shape,
              ModeOrder            const& ord_shape = {})
{
  CUTE_STATIC_ASSERT_V(rank(block) <= rank(trg_shape), "Rank of layout must be <= rank of target shape.");
  constexpr int R = rank_v<TrgShape>;

  auto padded_block = append<R>(block);

  auto block_shape  = product_each(shape(padded_block));
  auto target_shape = product_each(shape(trg_shape));

  // Assert proper division
  if constexpr (is_static<decltype(target_shape)>::value) {
    CUTE_STATIC_ASSERT_V(evenly_divides(target_shape, block_shape),
                         "tile_to_shape: block shape does not divide the target shape.");
  }

  auto product_shape = ceil_div(target_shape, block_shape);

  return coalesce(blocked_product(padded_block, make_ordered_layout(product_shape, ord_shape)), product_shape);
}

//
// Upcast
//   For stride-1 mode, divide size by N. Divide all other strides by N.
//

template <int N, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
upcast(Shape const& shape, Stride const& stride)
{
  if constexpr (is_tuple<Shape>::value) {                  // tuple stride
    return transform_layout(shape, stride, [](auto const& s, auto const& d) { return upcast<N>(s,d); });
  } else if constexpr (is_constant<0, Stride>::value) {    // static-0 stride
    return Layout<Shape,Stride>{shape,stride};
  } else if constexpr (is_static<Stride>::value) {         // static stride
    return make_layout(shape_div(shape,  shape_div(Int<N>{}, abs(stride))),
                       shape_div(stride, Int<N>{}));
  } else {                                                 // dynamic stride
    // assume dynamic strides are larger than N and divisible
    // assert(stride % N == 0);
    return make_layout(shape, safe_div(stride, Int<N>{}));
  }

  CUTE_GCC_UNREACHABLE;
}

template <int N, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
upcast(Layout<Shape,Stride> const& layout)
{
  return upcast<N>(layout.shape(), layout.stride());
}

//
// Downcast
//   For stride-1 mode, multiply size by N. Multiply all other strides by N.
//

template <int N, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
downcast(Shape const& shape, Stride const& stride)
{
  if constexpr (is_tuple<Shape>::value) {
    return transform_layout(shape, stride, [](auto const& s, auto const& d) { return downcast<N>(s,d); });
  } else if constexpr (is_constant<1, Stride>::value || is_constant<-1, Stride>::value) {
    return make_layout(shape * Int<N>{}, stride);
  } else {
    return make_layout(shape, stride * Int<N>{});
  }

  CUTE_GCC_UNREACHABLE;
}

template <int N, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
downcast(Layout<Shape,Stride> const& layout)
{
  CUTE_STATIC_ASSERT(has_int1<Stride>::value, "Downcast requires adjacent elements");
  return downcast<N>(layout.shape(), layout.stride());
}

//
// Recast
//

template <class OldType, class NewType,
          class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
recast_layout(Layout<Shape,Stride> const& layout)
{
  using scale = decltype(trait_ratio(sizeof_bits<NewType>{}, sizeof_bits<OldType>{}));
  if constexpr (scale::num == 1 && scale::den == 1) {
    return layout;
  }
  else if constexpr (scale::num == 1) {
    return downcast<scale::den>(layout);
  }
  else if constexpr (scale::den == 1) {
    return upcast<scale::num>(layout);
  }
  else {
    static_assert(dependent_false<scale>, "Recast not supported.");
  }

  CUTE_GCC_UNREACHABLE;
}

// Determine the maximum alignment of a Layout.
// The maximum alignment is the largest N for which upcast<N>(layout) will compile.
//   upcast<N>(layout) compiles when the static shapes and strides pass divisibility checks.
//   Therefore, upcast<M>(layout) will also compile for all divisors M of N.
// Note that this only considers the static shapes and strides of the Layout
//   in symmetry with upcast<N> only checking against static shapes and strides and assuming all
//   dynamic shapes and strides are large and multiples of N.
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
max_alignment(Layout<Shape,Stride> const& layout)
{
  auto flat_layout   = coalesce(layout);
  auto static_shape  = transform( shape(flat_layout), [](auto s){ return conditional_return<is_static<decltype(s)>::value>(s, Int<1>{}); });
  auto static_stride = transform(stride(flat_layout), [](auto d){ return conditional_return<is_static<decltype(d)>::value>(d, Int<0>{}); });
  auto filter_layout = make_layout(static_shape, static_stride);
  auto permuted = logical_divide(filter_layout, right_inverse(filter_layout));
  return gcd(size<0>(permuted), stride<1>(permuted));
}

//
// Display utilities
//

template <class Shape, class Stride>
CUTE_HOST_DEVICE void print(Layout<Shape,Stride> const& layout)
{
  print(layout.shape()); print(":"); print(layout.stride());
}

#if !defined(__CUDACC_RTC__)
template <class Shape, class Stride>
CUTE_HOST std::ostream& operator<<(std::ostream& os, Layout<Shape,Stride> const& layout)
{
  return os << shape(layout) << ":" << stride(layout);
}
#endif

// Generic 2D Layout to console table
template <class Layout>
CUTE_HOST_DEVICE
void
print_layout(Layout const& layout)  // (m,n) -> idx
{
  CUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});

  int idx_width = num_digits(cosize(layout)) + 2;
  const char* delim = "+-----------------------";

  print(layout); print("\n");

  // Column indices
  print("    ");
  for (int n = 0; n < size<1>(layout); ++n) { printf("  %*d ", idx_width-2, n); }
  printf("\n");

  // Print out A m-by-n
  for (int m = 0; m < size<0>(layout); ++m) {
    // Header
    print("    ");
    for (int n = 0; n < size<1>(layout); ++n) { printf("%.*s", idx_width+1, delim); }
    printf("+\n");
    // Values
    printf("%2d  ", m);  // Row indices
    for (int n = 0; n < size<1>(layout); ++n) { printf("| %*d ", idx_width-2, int(layout(m,n))); }
    printf("|\n");
  }
  // Footer
  print("    ");
  for (int n = 0; n < size<1>(layout); ++n) { printf("%.*s", idx_width+1, delim); }
  printf("+\n");
}

// Generic ThrVal 2D Layout to console table
template <class Layout, class ThrID>
CUTE_HOST_DEVICE
void
print_layout(Layout const& layout, ThrID const& thrid)  // (m,n) -> (tid,vid)  and  tid -> thr_idx
{
  CUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});

  print(layout); print("\n");
  print(thrid);  print("\n");

  // Print out m-by-n
  for (int m = 0; m < size<0>(layout); ++m) {
    // Header
    for (int n = 0; n < size<1>(layout); ++n) printf("+------");
    printf("+\n");
    // Values
    for (int n = 0; n < size<1>(layout); ++n) printf("|%03d-%02d", int(thrid(layout(m,n) % size(thrid))), int(layout(m,n) / size(thrid)));
    printf("|\n");
  }
  // Footer
  for (int n = 0; n < size<1>(layout); ++n) printf("+------");
  printf("+\n");
}

struct TikzColor_White {
  CUTE_HOST_DEVICE char const*
  operator()(int idx) const {
    return "white";
  }
};

struct TikzColor_BWx8 {
  CUTE_HOST_DEVICE char const*
  operator()(int idx) const {
    static char const* color_map[8] = {"black!00", "black!40", "black!20", "black!60",
                                       "black!10", "black!50", "black!30", "black!70"};
    return color_map[idx % 8];
  }
};

struct TikzColor_TV {
  CUTE_HOST_DEVICE char const*
  operator()(int tid, int vid) const {
    static char const* color_map[8] = {"{rgb,255:red,175;green,175;blue,255}",
                                       "{rgb,255:red,175;green,255;blue,175}",
                                       "{rgb,255:red,255;green,255;blue,175}",
                                       "{rgb,255:red,255;green,175;blue,175}",
                                       "{rgb,255:red,210;green,210;blue,255}",
                                       "{rgb,255:red,210;green,255;blue,210}",
                                       "{rgb,255:red,255;green,255;blue,210}",
                                       "{rgb,255:red,255;green,210;blue,210}"};
    return color_map[tid % 8];
  }
};

// Generic 2D Layout to LaTeX printer
template <class LayoutA, class TikzColorFn = TikzColor_BWx8>
CUTE_HOST_DEVICE
void
print_latex(LayoutA const& layout_a,   // (m,n) -> idx
            TikzColorFn color = {})    // lambda(idx) -> tikz color string
{
  CUTE_STATIC_ASSERT_V(rank(layout_a) <= Int<2>{});
  auto layout = append<2>(layout_a, Layout<_1,_0>{});

  // Commented print(layout)
  printf("%% Layout: "); print(layout); printf("\n");
  // Header
  printf("\\documentclass[convert]{standalone}\n"
         "\\usepackage{tikz}\n\n"
         "\\begin{document}\n"
         "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]\n\n");

  // Layout
  for (int i = 0; i < size<0>(layout); ++i) {
    for (int j = 0; j < size<1>(layout); ++j) {
      int idx = layout(i,j);
      printf("\\node[fill=%s] at (%d,%d) {%d};\n",
             color(idx), i, j, idx);
    }
  }
  // Grid
  printf("\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (%d,%d);\n\n",
         int(size<0>(layout)), int(size<1>(layout)));
  // Labels
  for (int i =  0, j = -1; i < size<0>(layout); ++i) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
  }
  for (int i = -1, j =  0; j < size<1>(layout); ++j) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
  }

  // Footer
  printf("\\end{tikzpicture}\n"
         "\\end{document}\n");
}

// Generic ThrVal 2D Layout to LaTeX TikZ
template <class Layout, class ThrID, class TikzColorFn = TikzColor_TV>
CUTE_HOST_DEVICE
void
print_latex(Layout const& layout,    // (m,n) -> (tid,vid)
            ThrID  const& thr,       // tid -> thr_idx
            TikzColorFn color = {})  // lambda(thr_idx,val_idx) -> tikz color string
{
  CUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});

  // Commented prints
  printf("%% Layout: "); print(layout); printf("\n");
  printf("%% ThrID : "); print(thr);  printf("\n");
  // Header
  printf("\\documentclass[convert]{standalone}\n"
         "\\usepackage{tikz}\n\n"
         "\\begin{document}\n"
         "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]\n\n");

  // Layout
  for (int i = 0; i < size<0>(layout); ++i) {
    for (int j = 0; j < size<1>(layout); ++j) {
      int thrid   = layout(i,j) % size(thr);
      int val_idx = layout(i,j) / size(thr);
      int thr_idx = thr(thrid);

      printf("\\node[fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color(thr_idx, val_idx),
             i, j,
             thr_idx, val_idx);
    }
  }
  // Grid
  printf("\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (%d,%d);\n\n",
         int(size<0>(layout)), int(size<1>(layout)));
  // Labels
  for (int i = 0, j = -1; i < size<0>(layout); ++i) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
  }
  for (int j = 0, i = -1; j < size<1>(layout); ++j) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
  }

  // Footer
  printf("\\end{tikzpicture}\n"
         "\\end{document}\n");
}

} // end namespace cute