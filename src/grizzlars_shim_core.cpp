// grizzlars_shim_core.cpp — construction/loading + basic accessors.
// Part of the GrizzlarFrame implementation, split across grizzlars_shim_*.cpp
// files by concern for readability (see grizzlars_shim.h for the full method
// list/documentation).
#include "grizzlars_shim.h"
#include "grizzlars_shim_internal.h"

using hmdf::nan_policy;
using grizzlars_detail::require_type;

void GrizzlarFrame::load_index(std::vector<uint64_t> indices)
{
    // uint64_t -> unsigned long is a genuine narrowing/widening conversion
    // (they differ in width under LLP64), not just a copy, so this pass
    // can't be avoided by moving.
    std::vector<unsigned long> idx(indices.begin(), indices.end());
    df_.load_index(std::move(idx));
}

void GrizzlarFrame::load_column_double(const std::string & name, std::vector<double> values)
{
    df_.load_column<double>(name.c_str(), std::move(values), nan_policy::pad_with_nans);
    if (col_types_.find(name) == col_types_.end())
        col_order_.push_back(name);
    col_types_[name] = "double";
}

void GrizzlarFrame::load_column_int64(const std::string & name, std::vector<int64_t> values)
{
    df_.load_column<int64_t>(name.c_str(), std::move(values), nan_policy::pad_with_nans);
    if (col_types_.find(name) == col_types_.end())
        col_order_.push_back(name);
    col_types_[name] = "int64";
}

void GrizzlarFrame::load_column_bool(const std::string & name, std::vector<uint8_t> values)
{
    // Stored as uint8_t, not bool: hmdf's generic sort/permutation code does
    // an internal std::swap() on column elements, and std::vector<bool>'s
    // bit-packed proxy reference doesn't satisfy std::swap() under MSVC's
    // STL. uint8_t avoids the footgun entirely while keeping the Python-
    // facing dtype label "bool" (see get_column_bool / col_types_).
    df_.load_column<uint8_t>(name.c_str(), std::move(values), nan_policy::pad_with_nans);
    if (col_types_.find(name) == col_types_.end())
        col_order_.push_back(name);
    col_types_[name] = "bool";
}

void GrizzlarFrame::load_column_string(const std::string & name, std::vector<std::string> values)
{
    df_.load_column<std::string>(name.c_str(), std::move(values), nan_policy::pad_with_nans);
    if (col_types_.find(name) == col_types_.end())
        col_order_.push_back(name);
    col_types_[name] = "string";
}

GrizzlarFrame GrizzlarFrame::deep_copy() const
{
    GrizzlarFrame out;
    out.df_ = df_;
    out.col_order_ = col_order_;
    out.col_types_ = col_types_;
    return out;
}

std::vector<uint64_t> GrizzlarFrame::get_index() const
{
    const auto & idx = df_.get_index();
    return std::vector<uint64_t>(idx.begin(), idx.end());
}

std::vector<std::string> GrizzlarFrame::columns() const
{
    return col_order_;
}

std::pair<size_t, size_t> GrizzlarFrame::shape() const
{
    return { df_.get_index().size(), col_order_.size() };
}

bool GrizzlarFrame::has_column(const std::string & name) const
{
    return col_types_.find(name) != col_types_.end();
}

std::string GrizzlarFrame::col_type(const std::string & name) const
{
    auto it = col_types_.find(name);
    if (it == col_types_.end())
        throw std::runtime_error("no such column: " + name);
    return it->second;
}

void GrizzlarFrame::drop_column(const std::string & name)
{
    const std::string type = col_type(name);
    if (type == "double")
        df_.remove_column<double>(name.c_str());
    else if (type == "int64")
        df_.remove_column<int64_t>(name.c_str());
    else if (type == "bool")
        df_.remove_column<uint8_t>(name.c_str());
    else
        df_.remove_column<std::string>(name.c_str());
    col_types_.erase(name);
    col_order_.erase(std::remove(col_order_.begin(), col_order_.end(), name), col_order_.end());
}

std::vector<double> GrizzlarFrame::get_column_double(const std::string & name) const
{
    require_type(col_types_, name, "double");
    const auto & col = df_.get_column<double>(name.c_str());
    return std::vector<double>(col.begin(), col.end());
}

std::vector<int64_t> GrizzlarFrame::get_column_int64(const std::string & name) const
{
    require_type(col_types_, name, "int64");
    const auto & col = df_.get_column<int64_t>(name.c_str());
    return std::vector<int64_t>(col.begin(), col.end());
}

std::vector<uint8_t> GrizzlarFrame::get_column_bool(const std::string & name) const
{
    require_type(col_types_, name, "bool");
    const auto & col = df_.get_column<uint8_t>(name.c_str());
    return std::vector<uint8_t>(col.begin(), col.end());
}

std::vector<std::string> GrizzlarFrame::get_column_string(const std::string & name) const
{
    require_type(col_types_, name, "string");
    const auto & col = df_.get_column<std::string>(name.c_str());
    return std::vector<std::string>(col.begin(), col.end());
}

