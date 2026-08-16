// grizzlars_shim_cleaning.cpp — data cleaning: clip/round/abs/rename/astype,
// replace_col, isin_col, duplicated_rows, diff_col.
#include "grizzlars_shim.h"
#include "grizzlars_shim_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
#include <unordered_map>

using grizzlars_detail::require_type;
using grizzlars_detail::require_numeric;

// ── data cleaning ───────────────────────────────────────────────────────────

void GrizzlarFrame::clip_col(const std::string & col, double lower, double upper)
{
    const std::string & type = require_numeric(col_types_, col);
    if (type == "double")
    {
        auto & c = df_.get_column<double>(col.c_str());
        for (auto & v : c) if (!std::isnan(v)) v = std::min(std::max(v, lower), upper);
    }
    else
    {
        auto & c = df_.get_column<int64_t>(col.c_str());
        for (auto & v : c)
            v = static_cast<int64_t>(std::min(std::max(static_cast<double>(v), lower), upper));
    }
}

void GrizzlarFrame::round_col(const std::string & col, int decimals)
{
    require_type(col_types_, col, "double");
    auto & c = df_.get_column<double>(col.c_str());
    const double factor = std::pow(10.0, decimals);
    for (auto & v : c)
        if (!std::isnan(v))
            v = std::round(v * factor) / factor;
}

void GrizzlarFrame::abs_col(const std::string & col)
{
    const std::string & type = require_numeric(col_types_, col);
    if (type == "double")
    {
        auto & c = df_.get_column<double>(col.c_str());
        for (auto & v : c) v = std::fabs(v);
    }
    else
    {
        auto & c = df_.get_column<int64_t>(col.c_str());
        for (auto & v : c) v = std::llabs(v);
    }
}

void GrizzlarFrame::rename_col(const std::string & old_name, const std::string & new_name)
{
    if (col_types_.find(new_name) != col_types_.end())
        throw std::runtime_error("column already exists: " + new_name);
    const std::string type = col_type(old_name);
    if (type == "double")
        df_.rename_column(old_name.c_str(), new_name.c_str());
    else if (type == "int64")
        df_.rename_column(old_name.c_str(), new_name.c_str());
    else if (type == "bool")
        df_.rename_column(old_name.c_str(), new_name.c_str());
    else
        df_.rename_column(old_name.c_str(), new_name.c_str());
    col_types_[new_name] = type;
    col_types_.erase(old_name);
    for (auto & n : col_order_)
        if (n == old_name) { n = new_name; break; }
}

void GrizzlarFrame::astype_col(const std::string & col, const std::string & target_type)
{
    const std::string from = col_type(col);
    if (from == target_type)
        return;
    const size_t n = shape().first;

    // Materialize the source column as plain values first (as strings, the
    // common denominator), then reload under the requested type.
    std::vector<std::string> as_str(n);
    if (from == "double")
    {
        const auto & c = df_.get_column<double>(col.c_str());
        for (size_t i = 0; i < n; ++i) as_str[i] = std::isnan(c[i]) ? "" : std::to_string(c[i]);
    }
    else if (from == "int64")
    {
        const auto & c = df_.get_column<int64_t>(col.c_str());
        for (size_t i = 0; i < n; ++i) as_str[i] = std::to_string(c[i]);
    }
    else if (from == "bool")
    {
        const auto & c = df_.get_column<uint8_t>(col.c_str());
        for (size_t i = 0; i < n; ++i) as_str[i] = std::to_string(c[i]);
    }
    else
    {
        const auto & c = df_.get_column<std::string>(col.c_str());
        as_str = c;
    }

    if (from == "double")      df_.remove_column<double>(col.c_str());
    else if (from == "int64")  df_.remove_column<int64_t>(col.c_str());
    else if (from == "bool")   df_.remove_column<uint8_t>(col.c_str());
    else                       df_.remove_column<std::string>(col.c_str());

    if (target_type == "double")
    {
        std::vector<double> vals(n);
        for (size_t i = 0; i < n; ++i)
            vals[i] = as_str[i].empty() ? std::numeric_limits<double>::quiet_NaN() : std::strtod(as_str[i].c_str(), nullptr);
        df_.load_column<double>(col.c_str(), std::move(vals));
    }
    else if (target_type == "int64")
    {
        std::vector<int64_t> vals(n);
        for (size_t i = 0; i < n; ++i)
            vals[i] = as_str[i].empty() ? 0 : std::strtoll(as_str[i].c_str(), nullptr, 10);
        df_.load_column<int64_t>(col.c_str(), std::move(vals));
    }
    else if (target_type == "bool")
    {
        std::vector<uint8_t> vals(n);
        for (size_t i = 0; i < n; ++i)
            vals[i] = (as_str[i] == "1" || as_str[i] == "true" || as_str[i] == "True") ? 1 : 0;
        df_.load_column<uint8_t>(col.c_str(), std::move(vals));
    }
    else
    {
        df_.load_column<std::string>(col.c_str(), std::move(as_str));
    }
    col_types_[col] = target_type;
}

void GrizzlarFrame::replace_col_double(const std::string & col, const std::vector<double> & from, const std::vector<double> & to)
{
    const std::string & type = require_numeric(col_types_, col);
    std::unordered_map<double, double> mapping;
    for (size_t i = 0; i < from.size() && i < to.size(); ++i) mapping[from[i]] = to[i];
    if (type == "double")
    {
        auto & c = df_.get_column<double>(col.c_str());
        for (auto & v : c) { auto it = mapping.find(v); if (it != mapping.end()) v = it->second; }
    }
    else
    {
        auto & c = df_.get_column<int64_t>(col.c_str());
        for (auto & v : c)
        {
            auto it = mapping.find(static_cast<double>(v));
            if (it != mapping.end()) v = static_cast<int64_t>(it->second);
        }
    }
}

void GrizzlarFrame::replace_col_string(const std::string & col, const std::vector<std::string> & from, const std::vector<std::string> & to)
{
    require_type(col_types_, col, "string");
    std::unordered_map<std::string, std::string> mapping;
    for (size_t i = 0; i < from.size() && i < to.size(); ++i) mapping[from[i]] = to[i];
    auto & c = df_.get_column<std::string>(col.c_str());
    for (auto & v : c) { auto it = mapping.find(v); if (it != mapping.end()) v = it->second; }
}

std::vector<uint8_t> GrizzlarFrame::isin_col_double(const std::string & col, const std::vector<double> & values) const
{
    const std::vector<double> data = get_column_double_or_cast(col);
    const std::set<double> s(values.begin(), values.end());
    std::vector<uint8_t> out(data.size());
    for (size_t i = 0; i < data.size(); ++i) out[i] = s.count(data[i]) ? 1 : 0;
    return out;
}

std::vector<uint8_t> GrizzlarFrame::isin_col_string(const std::string & col, const std::vector<std::string> & values) const
{
    require_type(col_types_, col, "string");
    const auto & data = df_.get_column<std::string>(col.c_str());
    const std::set<std::string> s(values.begin(), values.end());
    std::vector<uint8_t> out(data.size());
    for (size_t i = 0; i < data.size(); ++i) out[i] = s.count(data[i]) ? 1 : 0;
    return out;
}

std::vector<uint8_t> GrizzlarFrame::duplicated_rows(const std::vector<std::string> & cols, const std::string & keep) const
{
    const size_t n = shape().first;
    std::vector<std::string> keys(n);
    for (const auto & col : cols)
    {
        const std::string & type = col_type(col);
        for (size_t i = 0; i < n; ++i)
        {
            keys[i] += '\x1f';
            if (type == "double")      keys[i] += std::to_string(df_.get_column<double>(col.c_str())[i]);
            else if (type == "int64")  keys[i] += std::to_string(df_.get_column<int64_t>(col.c_str())[i]);
            else if (type == "bool")   keys[i] += std::to_string(df_.get_column<uint8_t>(col.c_str())[i]);
            else                       keys[i] += df_.get_column<std::string>(col.c_str())[i];
        }
    }

    std::vector<uint8_t> out(n, 0);
    if (keep == "false")
    {
        std::unordered_map<std::string, size_t> counts;
        for (const auto & k : keys) ++counts[k];
        for (size_t i = 0; i < n; ++i) out[i] = counts[keys[i]] > 1 ? 1 : 0;
    }
    else if (keep == "last")
    {
        std::unordered_map<std::string, size_t> last_pos;
        for (size_t i = 0; i < n; ++i) last_pos[keys[i]] = i;
        for (size_t i = 0; i < n; ++i) out[i] = (last_pos[keys[i]] != i) ? 1 : 0;
    }
    else
    {
        std::unordered_map<std::string, bool> seen;
        for (size_t i = 0; i < n; ++i)
        {
            if (seen.find(keys[i]) != seen.end()) out[i] = 1;
            else seen[keys[i]] = true;
        }
    }
    return out;
}

std::vector<double> GrizzlarFrame::diff_col(const std::string & col, int64_t periods) const
{
    const std::vector<double> data = get_column_double_or_cast(col);
    const int64_t len = static_cast<int64_t>(data.size());
    std::vector<double> out(data.size(), std::numeric_limits<double>::quiet_NaN());
    for (int64_t i = periods; i < len; ++i)
        out[static_cast<size_t>(i)] = data[static_cast<size_t>(i)] - data[static_cast<size_t>(i - periods)];
    return out;
}

