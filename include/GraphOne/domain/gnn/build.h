#pragma once

#include <type_traits>
#include <tuple>
#include <string>
#include <fstream>

#include "nlohmann/json.hpp"

#include "GraphOne/domain/gnn/module_holder.h"
#include "GraphOne/utils/log.hpp"

namespace graph_one::gnn {

// // 解包tuple元素类型到一个可调用对象的助手函数
// template<typename Tuple, std::size_t... Is>
// auto JsonToTuple(const nlohmann::json& j, const char* const json_map[], std::index_sequence<Is...>) {
//     return std::make_tuple(j[json_map[Is]].get<typename std::tuple_element<Is, Tuple>::type>()...);
// }

// // 解包tuple元素类型的公共接口
// template<typename Tuple, int map_len>
// auto JsonToTuple(const nlohmann::json& j, const char* const json_map[]) {
//     constexpr int tuple_sz = std::tuple_size<Tuple>::value;
//     static_assert(tuple_sz == map_len);
//     if (j.size() < map_len) {
//         LOG_ERROR("json object's size is smaller than the length of `json_map`, "
//             "need '>=", map_len, "', got ", j.size(), "'");
//     }
//     for (int i = 0; i < map_len; ++i) {
//         if (!j.contains(json_map[i])) {
//             LOG_ERROR("json object does not match the `json_map`, ",
//                 "as the key '", json_map[i], "' is missing");
//         }
//     }
//     return JsonToTuple<Tuple>(j, json_map, std::make_index_sequence<tuple_sz>{});
// }

template <typename module_holer_t, typename tuple_t, std::size_t... index_t>
module_holer_t build_module_from_tuple_(const tuple_t& t, std::index_sequence<index_t...>) {
    return module_holer_t(std::get<index_t>(t)...);
}

template <typename module_holer_t, typename... args_t>
module_holer_t build_module(const std::tuple<args_t...>& t) {
    return build_module_from_tuple_<module_holer_t>(t, std::make_index_sequence<std::tuple_size_v<std::tuple<args_t...>>>{});
}

template <typename module_t, typename = void>
struct has_ctor_args : std::false_type {};

template <typename module_t>
struct has_ctor_args<module_t, std::void_t<decltype(module_t::ctor_args)>> : std::true_type {};

template <typename T>
struct is_char_ptr_array : std::false_type {};

template <std::size_t N>
struct is_char_ptr_array<const char* const[N]> : std::true_type {};

template <typename module_holer_t, std::size_t... index_t>
module_holer_t build_module_from_json_(const nlohmann::json& j, const char* const json_map[], std::index_sequence<index_t...>) {
    return module_holer_t(j[json_map[index_t]]...);
}

/// build module from json object
/// MUST HAVE a static constexpr feild named `ctor_args` 
/// in module implementation, which is matched with the paramter 
/// names of the module's ctor
template <typename module_holer_t>
typename std::enable_if_t<
              is_module_holder<module_holer_t>::value &&
              has_ctor_args<typename module_holer_t::contained_type>::value &&
              is_char_ptr_array<decltype(module_holer_t::contained_type::ctor_args)>::value, 
              module_holer_t> 
build_module(const nlohmann::json& j) {
    using contained_t = typename module_holer_t::contained_type;

    // Check the types of ctor parameters were detected correctly.
    // using ctor_param_t = refl::as_tuple<contained_t>;
    constexpr auto& json_map = contained_t::ctor_args;
    constexpr int map_len = sizeof(json_map) / sizeof(char*);
    // auto t = JsonToTuple<ctor_param_t, sizeof(json_map)/sizeof(char*)>(j, json_map);

    if (j.size() < map_len) {
        LOG_ERROR("json object's size is smaller than the length of `ctor_args`, "
            "need '>=", map_len, "', got ", j.size(), "'");
    }
    for (int i = 0; i < map_len; ++i) {
        if (!j.contains(json_map[i])) {
            LOG_ERROR("json object does not match the `json_map`, ",
                "as the key '", json_map[i], "' is missing");
        }
    }

    return build_module_from_json_<module_holer_t>(j, json_map, 
        std::make_index_sequence<map_len>{});
}

template <typename module_holer_t>
typename std::enable_if_t<
              is_module_holder<module_holer_t>::value &&
              has_ctor_args<typename module_holer_t::contained_type>::value &&
              is_char_ptr_array<decltype(module_holer_t::contained_type::ctor_args)>::value, 
              module_holer_t> 
build_module(const std::string& model_path) {
    nlohmann::json data = nlohmann::json::parse(std::ifstream(model_path));
    return build_module<module_holer_t>(data);
}

} // namespace graph_one::gnn