// grizzlars_shim_internal.h — private helpers shared across the
// grizzlars_shim_*.cpp translation units. Not part of the public API (not
// touched by litgen), just internal plumbing split out of grizzlars_shim.cpp
// for readability.
#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace grizzlars_detail
{

inline const std::string & require_type(
    const std::unordered_map<std::string, std::string> & col_types,
    const std::string & name,
    const std::string & expected)
{
    auto it = col_types.find(name);
    if (it == col_types.end())
        throw std::runtime_error("no such column: " + name);
    if (it->second != expected)
        throw std::runtime_error(
            "column '" + name + "' is of type " + it->second + ", not " + expected);
    return it->second;
}

// double/int64 are the two numeric column types; returns which one.
inline const std::string & require_numeric(
    const std::unordered_map<std::string, std::string> & col_types,
    const std::string & name)
{
    auto it = col_types.find(name);
    if (it == col_types.end())
        throw std::runtime_error("no such column: " + name);
    if (it->second != "double" && it->second != "int64")
        throw std::runtime_error("column '" + name + "' is not numeric (" + it->second + ")");
    return it->second;
}

inline bool apply_op(const std::string & op, double a, double b)
{
    if (op == ">")  return a > b;
    if (op == ">=") return a >= b;
    if (op == "<")  return a < b;
    if (op == "<=") return a <= b;
    if (op == "==") return a == b;
    if (op == "!=") return a != b;
    throw std::runtime_error("unknown comparison operator: " + op);
}

} // namespace grizzlars_detail
