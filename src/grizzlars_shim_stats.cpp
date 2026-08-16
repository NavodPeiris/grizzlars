// grizzlars_shim_stats.cpp — scalar statistics: mean/std/sum/min/max,
// quantile/corr/cov/skew/kurt/mode, nunique/n_missing/count, unique_*,
// value_counts_*, describe. Real hmdf visitors do the actual computation
// (MeanVisitor, StdVisitor, ExtremumVisitor, CorrVisitor, CovVisitor,
// SkewVisitor, KurtosisVisitor, ModeVisitor) except quantile(), which
// deliberately bypasses hmdf's QuantileVisitor — see the comment there.
#include "grizzlars_shim.h"
#include "grizzlars_shim_internal.h"

#include <DataFrame/DataFrameStatsVisitors.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>

using ulong = unsigned long;
using grizzlars_detail::require_type;
using grizzlars_detail::require_numeric;

// ── scalar statistics ─────────────────────────────────────────────────────

std::vector<double> GrizzlarFrame::get_column_double_or_cast(const std::string & name) const
{
    const std::string & type = require_numeric(col_types_, name);
    if (type == "double")
    {
        const auto & c = df_.get_column<double>(name.c_str());
        return std::vector<double>(c.begin(), c.end());
    }
    const auto & c = df_.get_column<int64_t>(name.c_str());
    return std::vector<double>(c.begin(), c.end());
}

double GrizzlarFrame::mean(const std::string & col) const
{
    const auto & type = require_numeric(col_types_, col);
    if (type == "double")
    {
        hmdf::MeanVisitor<double, ulong> v;
        df_.single_act_visit<double>(col.c_str(), v);
        return v.get_result();
    }
    hmdf::MeanVisitor<int64_t, ulong> v;
    df_.single_act_visit<int64_t>(col.c_str(), v);
    return static_cast<double>(v.get_result());
}

double GrizzlarFrame::std_dev(const std::string & col) const
{
    const auto & type = require_numeric(col_types_, col);
    if (type == "double")
    {
        hmdf::StdVisitor<double, ulong> v;
        df_.single_act_visit<double>(col.c_str(), v);
        return v.get_result();
    }
    hmdf::StdVisitor<int64_t, ulong> v;
    df_.single_act_visit<int64_t>(col.c_str(), v);
    return static_cast<double>(v.get_result());
}

double GrizzlarFrame::sum(const std::string & col) const
{
    const auto & type = require_numeric(col_types_, col);
    if (type == "double")
    {
        hmdf::SumVisitor<double, ulong> v;
        df_.single_act_visit<double>(col.c_str(), v);
        return v.get_result();
    }
    hmdf::SumVisitor<int64_t, ulong> v;
    df_.single_act_visit<int64_t>(col.c_str(), v);
    return static_cast<double>(v.get_result());
}

double GrizzlarFrame::col_min(const std::string & col) const
{
    const auto & type = require_numeric(col_types_, col);
    if (type == "double")
    {
        hmdf::MinVisitor<double, ulong> v;
        df_.single_act_visit<double>(col.c_str(), v);
        return v.get_result();
    }
    hmdf::MinVisitor<int64_t, ulong> v;
    df_.single_act_visit<int64_t>(col.c_str(), v);
    return static_cast<double>(v.get_result());
}

double GrizzlarFrame::col_max(const std::string & col) const
{
    const auto & type = require_numeric(col_types_, col);
    if (type == "double")
    {
        hmdf::MaxVisitor<double, ulong> v;
        df_.single_act_visit<double>(col.c_str(), v);
        return v.get_result();
    }
    hmdf::MaxVisitor<int64_t, ulong> v;
    df_.single_act_visit<int64_t>(col.c_str(), v);
    return static_cast<double>(v.get_result());
}

double GrizzlarFrame::quantile(const std::string & col, double q) const
{
    // Deliberately NOT hmdf's QuantileVisitor/KthValueVisitor: its
    // find_kth_element_ always pivots on the last element with no
    // randomization, so already-sorted or reverse-sorted columns (an
    // auto-increment id, a pre-sorted index, ...) hit the textbook
    // quickselect O(n^2) worst case — confirmed directly (5+ seconds on a
    // 100k-row sorted int64 column vs 2ms on unsorted data of the same
    // size). std::nth_element has a guaranteed worst-case bound, so the
    // same "read from the real hmdf column" data is used here, just
    // selected with a safe algorithm instead of hmdf's.
    std::vector<double> data = get_column_double_or_cast(col);
    data.erase(std::remove_if(data.begin(), data.end(),
        [](double v) { return std::isnan(v); }), data.end());
    const size_t n = data.size();
    if (n == 0)
        return std::numeric_limits<double>::quiet_NaN();

    const double pos = q * static_cast<double>(n - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));

    std::nth_element(data.begin(), data.begin() + static_cast<long>(lo), data.end());
    const double lo_val = data[lo];
    if (hi == lo)
        return lo_val;
    std::nth_element(data.begin(), data.begin() + static_cast<long>(hi), data.end());
    const double hi_val = data[hi];
    return lo_val + (pos - static_cast<double>(lo)) * (hi_val - lo_val);
}

double GrizzlarFrame::corr(const std::string & col1, const std::string & col2) const
{
    const auto & t1 = require_numeric(col_types_, col1);
    const auto & t2 = require_numeric(col_types_, col2);
    if (t1 == "double" && t2 == "double")
    {
        hmdf::CorrVisitor<double, ulong> v;
        df_.single_act_visit<double, double>(col1.c_str(), col2.c_str(), v);
        return v.get_result();
    }
    if (t1 == "int64" && t2 == "int64")
    {
        hmdf::CorrVisitor<int64_t, ulong> v;
        df_.single_act_visit<int64_t, int64_t>(col1.c_str(), col2.c_str(), v);
        return static_cast<double>(v.get_result());
    }
    // Mixed double/int64 pair: hmdf's CorrVisitor requires both columns to
    // share one C++ type, so fall back to a manual pass over already-typed
    // column vectors (still real hmdf-managed storage, just no matching
    // single visitor instantiation exists for a mixed-type pair).
    const std::vector<double> a = get_column_double_or_cast(col1);
    const std::vector<double> b = get_column_double_or_cast(col2);
    const size_t n = std::min(a.size(), b.size());
    double mean_a = 0.0, mean_b = 0.0;
    for (size_t i = 0; i < n; ++i) { mean_a += a[i]; mean_b += b[i]; }
    mean_a /= static_cast<double>(n);
    mean_b /= static_cast<double>(n);
    double cov_ab = 0.0, var_a = 0.0, var_b = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double da = a[i] - mean_a, db = b[i] - mean_b;
        cov_ab += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    return cov_ab / std::sqrt(var_a * var_b);
}

double GrizzlarFrame::cov(const std::string & col1, const std::string & col2) const
{
    const auto & t1 = require_numeric(col_types_, col1);
    const auto & t2 = require_numeric(col_types_, col2);
    if (t1 == "double" && t2 == "double")
    {
        hmdf::CovVisitor<double, ulong> v;
        df_.single_act_visit<double, double>(col1.c_str(), col2.c_str(), v);
        return v.get_result();
    }
    if (t1 == "int64" && t2 == "int64")
    {
        hmdf::CovVisitor<int64_t, ulong> v;
        df_.single_act_visit<int64_t, int64_t>(col1.c_str(), col2.c_str(), v);
        return static_cast<double>(v.get_result());
    }
    const std::vector<double> a = get_column_double_or_cast(col1);
    const std::vector<double> b = get_column_double_or_cast(col2);
    const size_t n = std::min(a.size(), b.size());
    double mean_a = 0.0, mean_b = 0.0;
    for (size_t i = 0; i < n; ++i) { mean_a += a[i]; mean_b += b[i]; }
    mean_a /= static_cast<double>(n);
    mean_b /= static_cast<double>(n);
    double cov_ab = 0.0;
    for (size_t i = 0; i < n; ++i)
        cov_ab += (a[i] - mean_a) * (b[i] - mean_b);
    return n > 1 ? cov_ab / static_cast<double>(n - 1) : 0.0;
}

double GrizzlarFrame::skew_col(const std::string & col) const
{
    const auto & type = require_numeric(col_types_, col);
    if (type == "double")
    {
        hmdf::SkewVisitor<double, ulong> v(true);
        df_.single_act_visit<double>(col.c_str(), v);
        return v.get_result();
    }
    hmdf::SkewVisitor<int64_t, ulong> v(true);
    df_.single_act_visit<int64_t>(col.c_str(), v);
    return static_cast<double>(v.get_result());
}

double GrizzlarFrame::kurt_col(const std::string & col) const
{
    const auto & type = require_numeric(col_types_, col);
    if (type == "double")
    {
        hmdf::KurtosisVisitor<double, ulong> v(true);
        df_.single_act_visit<double>(col.c_str(), v);
        return v.get_result();
    }
    hmdf::KurtosisVisitor<int64_t, ulong> v(true);
    df_.single_act_visit<int64_t>(col.c_str(), v);
    return static_cast<double>(v.get_result());
}

namespace
{
constexpr size_t MODE_TOP_N = 16;
}

std::vector<double> GrizzlarFrame::mode_col_double(const std::string & col) const
{
    require_type(col_types_, col, "double");
    hmdf::ModeVisitor<MODE_TOP_N, double, ulong> v;
    df_.single_act_visit<double>(col.c_str(), v);
    const auto & result = v.get_result();
    std::vector<double> out;
    if (result[0].value == nullptr)
        return out;
    const size_t top_count = result[0].repeat_count();
    for (const auto & item : result)
    {
        if (item.value == nullptr || item.repeat_count() != top_count)
            break;
        out.push_back(item.get_value());
    }
    return out;
}

std::vector<int64_t> GrizzlarFrame::mode_col_int64(const std::string & col) const
{
    require_type(col_types_, col, "int64");
    hmdf::ModeVisitor<MODE_TOP_N, int64_t, ulong> v;
    df_.single_act_visit<int64_t>(col.c_str(), v);
    const auto & result = v.get_result();
    std::vector<int64_t> out;
    if (result[0].value == nullptr)
        return out;
    const size_t top_count = result[0].repeat_count();
    for (const auto & item : result)
    {
        if (item.value == nullptr || item.repeat_count() != top_count)
            break;
        out.push_back(item.get_value());
    }
    return out;
}

std::vector<std::string> GrizzlarFrame::mode_col_string(const std::string & col) const
{
    require_type(col_types_, col, "string");
    hmdf::ModeVisitor<MODE_TOP_N, std::string, ulong> v;
    df_.single_act_visit<std::string>(col.c_str(), v);
    const auto & result = v.get_result();
    std::vector<std::string> out;
    if (result[0].value == nullptr)
        return out;
    const size_t top_count = result[0].repeat_count();
    for (const auto & item : result)
    {
        if (item.value == nullptr || item.repeat_count() != top_count)
            break;
        out.push_back(item.get_value());
    }
    return out;
}

size_t GrizzlarFrame::nunique(const std::string & col) const
{
    const std::string & type = col_type(col);
    if (type == "double")
    {
        const auto & c = df_.get_column<double>(col.c_str());
        return std::set<double>(c.begin(), c.end()).size();
    }
    if (type == "int64")
    {
        const auto & c = df_.get_column<int64_t>(col.c_str());
        return std::set<int64_t>(c.begin(), c.end()).size();
    }
    if (type == "bool")
    {
        const auto & c = df_.get_column<uint8_t>(col.c_str());
        return std::set<uint8_t>(c.begin(), c.end()).size();
    }
    const auto & c = df_.get_column<std::string>(col.c_str());
    return std::set<std::string>(c.begin(), c.end()).size();
}

size_t GrizzlarFrame::n_missing(const std::string & col) const
{
    const std::string & type = col_type(col);
    if (type == "double")
    {
        const auto & c = df_.get_column<double>(col.c_str());
        return static_cast<size_t>(std::count_if(c.begin(), c.end(),
            [](double v) { return std::isnan(v); }));
    }
    if (type == "string")
    {
        const auto & c = df_.get_column<std::string>(col.c_str());
        return static_cast<size_t>(std::count(c.begin(), c.end(), std::string{}));
    }
    return 0;
}

size_t GrizzlarFrame::count(const std::string & col) const
{
    return shape().first - n_missing(col);
}

std::vector<double> GrizzlarFrame::unique_double(const std::string & col) const
{
    require_type(col_types_, col, "double");
    const auto & c = df_.get_column<double>(col.c_str());
    std::set<double> s(c.begin(), c.end());
    return std::vector<double>(s.begin(), s.end());
}

std::vector<int64_t> GrizzlarFrame::unique_int64(const std::string & col) const
{
    require_type(col_types_, col, "int64");
    const auto & c = df_.get_column<int64_t>(col.c_str());
    std::set<int64_t> s(c.begin(), c.end());
    return std::vector<int64_t>(s.begin(), s.end());
}

std::vector<std::string> GrizzlarFrame::unique_string(const std::string & col) const
{
    require_type(col_types_, col, "string");
    const auto & c = df_.get_column<std::string>(col.c_str());
    std::set<std::string> s(c.begin(), c.end());
    return std::vector<std::string>(s.begin(), s.end());
}

namespace
{
template <typename T>
std::vector<std::pair<T, size_t>> count_values(const std::vector<T> & data)
{
    std::unordered_map<T, size_t> counts;
    std::vector<T> order;
    for (const auto & v : data)
    {
        auto [it, inserted] = counts.emplace(v, 0);
        if (inserted) order.push_back(v);
        ++it->second;
    }
    std::vector<std::pair<T, size_t>> out;
    out.reserve(order.size());
    for (const auto & v : order) out.emplace_back(v, counts.at(v));
    std::stable_sort(out.begin(), out.end(),
        [](const auto & a, const auto & b) { return a.second > b.second; });
    return out;
}
} // namespace

GrizzlarFrame GrizzlarFrame::value_counts_double(const std::string & col) const
{
    require_type(col_types_, col, "double");
    const auto & c = df_.get_column<double>(col.c_str());
    const auto counted = count_values(std::vector<double>(c.begin(), c.end()));
    GrizzlarFrame out;
    std::vector<unsigned long> idx(counted.size());
    std::vector<double> values(counted.size());
    std::vector<int64_t> counts(counted.size());
    for (size_t i = 0; i < counted.size(); ++i) { idx[i] = static_cast<unsigned long>(i); values[i] = counted[i].first; counts[i] = static_cast<int64_t>(counted[i].second); }
    out.df_.load_index(std::move(idx));
    out.df_.load_column<double>("value", std::move(values));
    out.df_.load_column<int64_t>("count", std::move(counts));
    out.col_order_ = { "value", "count" };
    out.col_types_ = { {"value", "double"}, {"count", "int64"} };
    return out;
}

GrizzlarFrame GrizzlarFrame::value_counts_int64(const std::string & col) const
{
    require_type(col_types_, col, "int64");
    const auto & c = df_.get_column<int64_t>(col.c_str());
    const auto counted = count_values(std::vector<int64_t>(c.begin(), c.end()));
    GrizzlarFrame out;
    std::vector<unsigned long> idx(counted.size());
    std::vector<int64_t> values(counted.size());
    std::vector<int64_t> counts(counted.size());
    for (size_t i = 0; i < counted.size(); ++i) { idx[i] = static_cast<unsigned long>(i); values[i] = counted[i].first; counts[i] = static_cast<int64_t>(counted[i].second); }
    out.df_.load_index(std::move(idx));
    out.df_.load_column<int64_t>("value", std::move(values));
    out.df_.load_column<int64_t>("count", std::move(counts));
    out.col_order_ = { "value", "count" };
    out.col_types_ = { {"value", "int64"}, {"count", "int64"} };
    return out;
}

GrizzlarFrame GrizzlarFrame::value_counts_string(const std::string & col) const
{
    require_type(col_types_, col, "string");
    const auto & c = df_.get_column<std::string>(col.c_str());
    const auto counted = count_values(std::vector<std::string>(c.begin(), c.end()));
    GrizzlarFrame out;
    std::vector<unsigned long> idx(counted.size());
    std::vector<std::string> values(counted.size());
    std::vector<int64_t> counts(counted.size());
    for (size_t i = 0; i < counted.size(); ++i) { idx[i] = static_cast<unsigned long>(i); values[i] = counted[i].first; counts[i] = static_cast<int64_t>(counted[i].second); }
    out.df_.load_index(std::move(idx));
    out.df_.load_column<std::string>("value", std::move(values));
    out.df_.load_column<int64_t>("count", std::move(counts));
    out.col_order_ = { "value", "count" };
    out.col_types_ = { {"value", "string"}, {"count", "int64"} };
    return out;
}

std::map<std::string, std::map<std::string, double>> GrizzlarFrame::describe() const
{
    std::map<std::string, std::map<std::string, double>> out;
    for (const auto & name : col_order_)
    {
        const std::string & type = col_types_.at(name);
        if (type != "double" && type != "int64")
            continue;
        std::map<std::string, double> stats;
        stats["count"] = static_cast<double>(count(name));
        stats["mean"] = mean(name);
        stats["std"] = std_dev(name);
        stats["min"] = col_min(name);
        stats["25%"] = quantile(name, 0.25);
        stats["50%"] = quantile(name, 0.5);
        stats["75%"] = quantile(name, 0.75);
        stats["max"] = col_max(name);
        out[name] = std::move(stats);
    }
    return out;
}

