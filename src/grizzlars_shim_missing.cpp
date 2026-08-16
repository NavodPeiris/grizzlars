// grizzlars_shim_missing.cpp — fillna/ffill/bfill, drop_na, drop_duplicates.
#include "grizzlars_shim.h"
#include "grizzlars_shim_internal.h"

#include <cmath>
#include <limits>
#include <unordered_map>

using grizzlars_detail::require_type;

void GrizzlarFrame::fillna_double(const std::string & col, double value)
{
    require_type(col_types_, col, "double");
    auto & c = df_.get_column<double>(col.c_str());
    for (auto & v : c)
        if (std::isnan(v))
            v = value;
}

void GrizzlarFrame::fillna_string(const std::string & col, const std::string & value)
{
    require_type(col_types_, col, "string");
    auto & c = df_.get_column<std::string>(col.c_str());
    for (auto & v : c)
        if (v.empty())
            v = value;
}

void GrizzlarFrame::ffill_col(const std::string & col)
{
    const std::string & type = col_type(col);
    if (type == "double")
    {
        auto & c = df_.get_column<double>(col.c_str());
        double last = std::numeric_limits<double>::quiet_NaN();
        for (auto & v : c) { if (std::isnan(v)) v = last; else last = v; }
    }
    else if (type == "string")
    {
        auto & c = df_.get_column<std::string>(col.c_str());
        std::string last;
        for (auto & v : c) { if (v.empty()) v = last; else last = v; }
    }
}

void GrizzlarFrame::bfill_col(const std::string & col)
{
    const std::string & type = col_type(col);
    if (type == "double")
    {
        auto & c = df_.get_column<double>(col.c_str());
        double next = std::numeric_limits<double>::quiet_NaN();
        for (auto it = c.rbegin(); it != c.rend(); ++it) { if (std::isnan(*it)) *it = next; else next = *it; }
    }
    else if (type == "string")
    {
        auto & c = df_.get_column<std::string>(col.c_str());
        std::string next;
        for (auto it = c.rbegin(); it != c.rend(); ++it) { if (it->empty()) *it = next; else next = *it; }
    }
}

GrizzlarFrame GrizzlarFrame::drop_na(const std::string & col) const
{
    const std::string & type = col_type(col);
    const size_t n = shape().first;
    std::vector<long> positions;
    positions.reserve(n);
    if (type == "double")
    {
        const auto & c = df_.get_column<double>(col.c_str());
        for (size_t i = 0; i < n; ++i)
            if (!std::isnan(c[i]))
                positions.push_back(static_cast<long>(i));
    }
    else if (type == "string")
    {
        const auto & c = df_.get_column<std::string>(col.c_str());
        for (size_t i = 0; i < n; ++i)
            if (!c[i].empty())
                positions.push_back(static_cast<long>(i));
    }
    else
    {
        for (size_t i = 0; i < n; ++i)
            positions.push_back(static_cast<long>(i));
    }
    return from_positions(positions);
}

GrizzlarFrame GrizzlarFrame::drop_duplicates(const std::string & col) const
{
    const std::string & type = col_type(col);
    const size_t n = shape().first;
    std::vector<long> positions;
    positions.reserve(n);
    std::unordered_map<std::string, bool> seen;
    for (size_t i = 0; i < n; ++i)
    {
        std::string key;
        if (type == "double")      key = std::to_string(df_.get_column<double>(col.c_str())[i]);
        else if (type == "int64")  key = std::to_string(df_.get_column<int64_t>(col.c_str())[i]);
        else if (type == "bool")   key = std::to_string(df_.get_column<uint8_t>(col.c_str())[i]);
        else                       key = df_.get_column<std::string>(col.c_str())[i];
        if (seen.emplace(key, true).second)
            positions.push_back(static_cast<long>(i));
    }
    return from_positions(positions);
}
