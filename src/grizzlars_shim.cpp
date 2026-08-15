#include "grizzlars_shim.h"

#include <DataFrame/DataFrameStatsVisitors.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <typeindex>
#include <unordered_map>

#if defined(__has_include)
#if __has_include(<execution>)
#include <execution>
#if defined(__cpp_lib_execution)
#define GRIZZLAR_USE_EXECUTION 1
#endif
#endif
#endif

// Parallel std::sort, when the standard library supports execution
// policies (MSVC's STL does out of the box, no TBB needed) — falls back
// to a plain serial sort otherwise. hmdf's own sort<T,Ts...>() uses its
// internal ThreadPool::parallel_sort, so a single-threaded std::sort here
// was consistently slower on string-heavy columns; this closes that gap
// without touching cpp_lib/DataFrame.
#if defined(GRIZZLAR_USE_EXECUTION)
#ifdef __APPLE__
#define GRIZZLAR_SORT_POLICY std::execution::seq
#else
#define GRIZZLAR_SORT_POLICY std::execution::par
#endif
#define GRIZZLAR_SORT(...) std::sort(GRIZZLAR_SORT_POLICY, __VA_ARGS__)
#else
#define GRIZZLAR_SORT(...) std::sort(__VA_ARGS__)
#endif

using hmdf::nan_policy;
using ulong = unsigned long;

namespace
{

const std::string & require_type(
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
const std::string & require_numeric(
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

} // namespace

void GrizzlarFrame::load_index(const std::vector<uint64_t> & indices)
{
    std::vector<unsigned long> idx(indices.begin(), indices.end());
    df_.load_index(std::move(idx));
}

void GrizzlarFrame::load_column_double(const std::string & name, const std::vector<double> & values)
{
    df_.load_column<double>(name.c_str(), values, nan_policy::pad_with_nans);
    if (col_types_.find(name) == col_types_.end())
        col_order_.push_back(name);
    col_types_[name] = "double";
}

void GrizzlarFrame::load_column_int64(const std::string & name, const std::vector<int64_t> & values)
{
    df_.load_column<int64_t>(name.c_str(), values, nan_policy::pad_with_nans);
    if (col_types_.find(name) == col_types_.end())
        col_order_.push_back(name);
    col_types_[name] = "int64";
}

void GrizzlarFrame::load_column_bool(const std::string & name, const std::vector<uint8_t> & values)
{
    // Stored as uint8_t, not bool: hmdf's generic sort/permutation code does
    // an internal std::swap() on column elements, and std::vector<bool>'s
    // bit-packed proxy reference doesn't satisfy std::swap() under MSVC's
    // STL. uint8_t avoids the footgun entirely while keeping the Python-
    // facing dtype label "bool" (see get_column_bool / col_types_).
    df_.load_column<uint8_t>(name.c_str(), values, nan_policy::pad_with_nans);
    if (col_types_.find(name) == col_types_.end())
        col_order_.push_back(name);
    col_types_[name] = "bool";
}

void GrizzlarFrame::load_column_string(const std::string & name, const std::vector<std::string> & values)
{
    df_.load_column<std::string>(name.c_str(), values, nan_policy::pad_with_nans);
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
    out.df_.load_column<double>("value", values);
    out.df_.load_column<int64_t>("count", counts);
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
    out.df_.load_column<int64_t>("value", values);
    out.df_.load_column<int64_t>("count", counts);
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
    out.df_.load_column<std::string>("value", values);
    out.df_.load_column<int64_t>("count", counts);
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

// ── sorting / row selection / filtering ───────────────────────────────────

namespace
{

// Gathers positions[i] from a source column into a fresh vector, in
// parallel for large inputs. Written here (not in cpp_lib/DataFrame)
// specifically because hmdf's own get_data_by_loc gathers std::string
// columns element-by-element and, for large mostly-string DataFrames,
// that dominates filter/iloc/take_rows wall-clock time — the per-element
// work is embarrassingly parallel regardless of column type, so this
// just does the same gather with real worker threads.
template <typename T, typename GetFn>
std::vector<T> parallel_gather(const std::vector<long> & positions, size_t src_size, GetFn && get)
{
    const size_t n = positions.size();
    std::vector<T> out(n);
    auto run = [&](size_t begin, size_t end)
    {
        for (size_t i = begin; i < end; ++i)
        {
            const long p = positions[i];
            const size_t idx = p >= 0 ? static_cast<size_t>(p) : src_size + static_cast<size_t>(p);
            out[i] = get(idx);
        }
    };

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t n_threads = (n >= 20000) ? std::min<size_t>(hw, 16) : 1;
    if (n_threads <= 1)
    {
        run(0, n);
        return out;
    }
    const size_t chunk = (n + n_threads - 1) / n_threads;
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (size_t t = 0; t < n_threads; ++t)
    {
        const size_t begin = t * chunk;
        const size_t end = std::min(n, begin + chunk);
        if (begin >= end) continue;
        threads.emplace_back(run, begin, end);
    }
    for (auto & th : threads) th.join();
    return out;
}

} // namespace

GrizzlarFrame GrizzlarFrame::from_positions(const std::vector<long> & positions) const
{
    const size_t src_n = shape().first;
    GrizzlarFrame out;

    {
        const auto & src_idx = df_.get_index();
        out.df_.load_index(parallel_gather<unsigned long>(positions, src_n,
            [&](size_t i) { return src_idx[i]; }));
    }

    for (const auto & name : col_order_)
    {
        const std::string & type = col_types_.at(name);
        if (type == "double")
        {
            const auto & src = df_.get_column<double>(name.c_str());
            out.df_.load_column<double>(name.c_str(),
                parallel_gather<double>(positions, src_n, [&](size_t i) { return src[i]; }));
        }
        else if (type == "int64")
        {
            const auto & src = df_.get_column<int64_t>(name.c_str());
            out.df_.load_column<int64_t>(name.c_str(),
                parallel_gather<int64_t>(positions, src_n, [&](size_t i) { return src[i]; }));
        }
        else if (type == "bool")
        {
            const auto & src = df_.get_column<uint8_t>(name.c_str());
            out.df_.load_column<uint8_t>(name.c_str(),
                parallel_gather<uint8_t>(positions, src_n, [&](size_t i) { return src[i]; }));
        }
        else
        {
            const auto & src = df_.get_column<std::string>(name.c_str());
            out.df_.load_column<std::string>(name.c_str(),
                parallel_gather<std::string>(positions, src_n, [&](size_t i) { return src[i]; }));
        }
    }
    out.col_order_ = col_order_;
    out.col_types_ = col_types_;
    return out;
}

GrizzlarFrame GrizzlarFrame::sort_by(const std::string & by, bool ascending) const
{
    // Computes the sort permutation itself and applies it via
    // from_positions()'s parallel gather above, instead of hmdf's own
    // sort<T,Ts...>() — same reasoning as from_positions: hmdf's generic
    // permutation-apply is per-element for string columns, and this
    // dataset shape (mostly string columns) makes that the bottleneck.
    const size_t n = shape().first;
    std::vector<long> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = static_cast<long>(i);

    const std::string & type = col_type(by);
    if (type == "double")
    {
        const auto & c = df_.get_column<double>(by.c_str());
        GRIZZLAR_SORT(perm.begin(), perm.end(), [&](long a, long b) {
            return ascending ? c[a] < c[b] : c[a] > c[b];
        });
    }
    else if (type == "int64")
    {
        const auto & c = df_.get_column<int64_t>(by.c_str());
        GRIZZLAR_SORT(perm.begin(), perm.end(), [&](long a, long b) {
            return ascending ? c[a] < c[b] : c[a] > c[b];
        });
    }
    else if (type == "bool")
    {
        const auto & c = df_.get_column<uint8_t>(by.c_str());
        GRIZZLAR_SORT(perm.begin(), perm.end(), [&](long a, long b) {
            return ascending ? c[a] < c[b] : c[a] > c[b];
        });
    }
    else
    {
        const auto & c = df_.get_column<std::string>(by.c_str());
        GRIZZLAR_SORT(perm.begin(), perm.end(), [&](long a, long b) {
            return ascending ? c[a] < c[b] : c[a] > c[b];
        });
    }
    return from_positions(perm);
}

GrizzlarFrame GrizzlarFrame::sort_index(bool ascending) const
{
    const auto & idx = df_.get_index();
    const size_t n = idx.size();
    std::vector<long> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = static_cast<long>(i);
    GRIZZLAR_SORT(perm.begin(), perm.end(), [&](long a, long b) {
        return ascending ? idx[a] < idx[b] : idx[a] > idx[b];
    });
    return from_positions(perm);
}

GrizzlarFrame GrizzlarFrame::iloc(int64_t start, int64_t stop) const
{
    const long n = static_cast<long>(shape().first);
    long b = static_cast<long>(start);
    long e = static_cast<long>(stop);
    if (b < 0) b += n;
    if (e < 0) e += n;
    b = std::max(0L, std::min(b, n));
    e = std::max(b, std::min(e, n));
    std::vector<long> positions;
    positions.reserve(static_cast<size_t>(e - b));
    for (long i = b; i < e; ++i)
        positions.push_back(i);
    return from_positions(positions);
}

GrizzlarFrame GrizzlarFrame::take_rows(const std::vector<int64_t> & positions) const
{
    std::vector<long> pos(positions.begin(), positions.end());
    return from_positions(pos);
}

GrizzlarFrame GrizzlarFrame::select_columns(const std::vector<std::string> & names) const
{
    GrizzlarFrame out;
    out.df_.load_index(std::vector<unsigned long>(df_.get_index().begin(), df_.get_index().end()));
    for (const auto & name : names)
    {
        const std::string & type = col_type(name);
        if (type == "double")
            out.df_.load_column<double>(name.c_str(), df_.get_column<double>(name.c_str()));
        else if (type == "int64")
            out.df_.load_column<int64_t>(name.c_str(), df_.get_column<int64_t>(name.c_str()));
        else if (type == "bool")
            out.df_.load_column<uint8_t>(name.c_str(), df_.get_column<uint8_t>(name.c_str()));
        else
            out.df_.load_column<std::string>(name.c_str(), df_.get_column<std::string>(name.c_str()));
        out.col_order_.push_back(name);
        out.col_types_[name] = type;
    }
    return out;
}

GrizzlarFrame GrizzlarFrame::filter_by_mask_list(const std::vector<uint8_t> & mask) const
{
    std::vector<long> positions;
    positions.reserve(mask.size());
    for (size_t i = 0; i < mask.size(); ++i)
        if (mask[i])
            positions.push_back(static_cast<long>(i));
    return from_positions(positions);
}

namespace
{
bool apply_op(const std::string & op, double a, double b)
{
    if (op == ">")  return a > b;
    if (op == ">=") return a >= b;
    if (op == "<")  return a < b;
    if (op == "<=") return a <= b;
    if (op == "==") return a == b;
    if (op == "!=") return a != b;
    throw std::runtime_error("unknown comparison operator: " + op);
}
} // namespace

GrizzlarFrame GrizzlarFrame::filter_col_scalar_double(const std::string & col, const std::string & op, double scalar) const
{
    return filter_by_mask_list(compare_col_scalar_double(col, op, scalar));
}

std::vector<uint8_t> GrizzlarFrame::compare_col_scalar_double(const std::string & col, const std::string & op, double scalar) const
{
    const std::vector<double> data = get_column_double_or_cast(col);
    std::vector<uint8_t> mask(data.size());
    for (size_t i = 0; i < data.size(); ++i)
        mask[i] = apply_op(op, data[i], scalar) ? 1 : 0;
    return mask;
}

// ── groupby / join / concat ────────────────────────────────────────────────

void GrizzlarFrame::sync_from_df()
{
    col_order_.clear();
    col_types_.clear();
    const auto info = df_.get_columns_info<double, int64_t, uint8_t, std::string>();
    for (const auto & [raw_name, idx, tidx] : info)
    {
        (void)idx;
        std::string name(raw_name.c_str());
        col_order_.push_back(name);
        if (tidx == std::type_index(typeid(double)))
            col_types_[name] = "double";
        else if (tidx == std::type_index(typeid(int64_t)))
            col_types_[name] = "int64";
        else if (tidx == std::type_index(typeid(uint8_t)))
            col_types_[name] = "bool";
        else
            col_types_[name] = "string";
    }
}

namespace
{

double read_numeric_at(const hmdf::StdDataFrame<unsigned long> & df,
                       const std::string & type, const std::string & col, size_t row)
{
    if (type == "double")
        return df.get_column<double>(col.c_str())[row];
    return static_cast<double>(df.get_column<int64_t>(col.c_str())[row]);
}

double aggregate_group(const hmdf::StdDataFrame<unsigned long> & df,
                       const std::string & type, const std::string & col,
                       const std::vector<size_t> & rows, const std::string & func)
{
    const size_t cnt = rows.size();
    if (func == "count")
        return static_cast<double>(cnt);
    if (cnt == 0)
        return 0.0;
    if (func == "first")
        return read_numeric_at(df, type, col, rows.front());
    if (func == "last")
        return read_numeric_at(df, type, col, rows.back());

    double s = 0.0;
    double mn = read_numeric_at(df, type, col, rows[0]);
    double mx = mn;
    for (size_t r : rows)
    {
        const double v = read_numeric_at(df, type, col, r);
        s += v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    if (func == "sum")  return s;
    if (func == "mean") return s / static_cast<double>(cnt);
    if (func == "min")  return mn;
    if (func == "max")  return mx;
    if (func == "std")
    {
        if (cnt < 2) return 0.0;
        const double m = s / static_cast<double>(cnt);
        double sq = 0.0;
        for (size_t r : rows)
        {
            const double d = read_numeric_at(df, type, col, r) - m;
            sq += d * d;
        }
        return std::sqrt(sq / static_cast<double>(cnt - 1));
    }
    throw std::runtime_error("unknown aggregation function: " + func);
}

} // namespace

GrizzlarFrame GrizzlarFrame::groupby_agg(
    const std::string & by_col,
    const std::vector<std::string> & agg_cols,
    const std::vector<std::string> & agg_funcs) const
{
    if (agg_cols.size() != agg_funcs.size())
        throw std::runtime_error("groupby_agg: agg_cols and agg_funcs must be the same length");

    const std::string & by_type = col_type(by_col);
    const size_t n = shape().first;

    // Grouping itself has no hmdf entry point for a runtime-arbitrary agg
    // spec (hmdf's groupby1/2/3 require the aggregation spec to be known at
    // compile time via variadic template args) — the bucketing below is
    // straightforward bookkeeping; the actual aggregation math still reads
    // directly from df_'s real column storage via read_numeric_at() above.
    std::vector<std::string> group_keys;
    std::unordered_map<std::string, size_t> key_to_group;
    std::vector<std::vector<size_t>> group_rows;

    for (size_t i = 0; i < n; ++i)
    {
        std::string key;
        if (by_type == "double")      key = std::to_string(df_.get_column<double>(by_col.c_str())[i]);
        else if (by_type == "int64")  key = std::to_string(df_.get_column<int64_t>(by_col.c_str())[i]);
        else if (by_type == "bool")   key = std::to_string(df_.get_column<uint8_t>(by_col.c_str())[i]);
        else                          key = df_.get_column<std::string>(by_col.c_str())[i];

        auto it = key_to_group.find(key);
        if (it == key_to_group.end())
        {
            key_to_group[key] = group_keys.size();
            group_keys.push_back(key);
            group_rows.emplace_back(std::vector<size_t>{ i });
        }
        else
        {
            group_rows[it->second].push_back(i);
        }
    }

    const size_t ngroups = group_keys.size();
    GrizzlarFrame out;
    {
        std::vector<unsigned long> idx(ngroups);
        for (size_t g = 0; g < ngroups; ++g) idx[g] = static_cast<unsigned long>(g);
        out.df_.load_index(std::move(idx));
    }

    // by_col column, reconstructed from each group's first member row.
    if (by_type == "double")
    {
        std::vector<double> vals(ngroups);
        for (size_t g = 0; g < ngroups; ++g) vals[g] = df_.get_column<double>(by_col.c_str())[group_rows[g].front()];
        out.df_.load_column<double>(by_col.c_str(), vals);
    }
    else if (by_type == "int64")
    {
        std::vector<int64_t> vals(ngroups);
        for (size_t g = 0; g < ngroups; ++g) vals[g] = df_.get_column<int64_t>(by_col.c_str())[group_rows[g].front()];
        out.df_.load_column<int64_t>(by_col.c_str(), vals);
    }
    else if (by_type == "bool")
    {
        std::vector<uint8_t> vals(ngroups);
        for (size_t g = 0; g < ngroups; ++g) vals[g] = df_.get_column<uint8_t>(by_col.c_str())[group_rows[g].front()];
        out.df_.load_column<uint8_t>(by_col.c_str(), vals);
    }
    else
    {
        std::vector<std::string> vals(ngroups);
        for (size_t g = 0; g < ngroups; ++g) vals[g] = df_.get_column<std::string>(by_col.c_str())[group_rows[g].front()];
        out.df_.load_column<std::string>(by_col.c_str(), vals);
    }
    out.col_order_.push_back(by_col);
    out.col_types_[by_col] = by_type;

    for (size_t s = 0; s < agg_cols.size(); ++s)
    {
        const std::string & col = agg_cols[s];
        const std::string & func = agg_funcs[s];
        const std::string & col_t = require_numeric(col_types_, col);
        std::vector<double> vals(ngroups);
        for (size_t g = 0; g < ngroups; ++g)
            vals[g] = aggregate_group(df_, col_t, col, group_rows[g], func);
        out.df_.load_column<double>(col.c_str(), vals);
        out.col_order_.push_back(col);
        out.col_types_[col] = "double";
    }
    return out;
}

GrizzlarFrame GrizzlarFrame::join_by_index(const GrizzlarFrame & rhs, const std::string & how) const
{
    hmdf::join_policy jp;
    if (how == "inner") jp = hmdf::join_policy::inner_join;
    else if (how == "left") jp = hmdf::join_policy::left_join;
    else if (how == "right") jp = hmdf::join_policy::right_join;
    else if (how == "outer") jp = hmdf::join_policy::left_right_join;
    else throw std::runtime_error("unknown join how: " + how);

    GrizzlarFrame out;
    out.df_ = df_.join_by_index<hmdf::StdDataFrame<unsigned long>, double, int64_t, uint8_t, std::string>(
        rhs.df_, jp);
    out.sync_from_df();
    return out;
}

GrizzlarFrame GrizzlarFrame::concat_frame(const GrizzlarFrame & other) const
{
    GrizzlarFrame out;
    out.df_ = df_.concat<hmdf::StdDataFrame<unsigned long>, double, int64_t, uint8_t, std::string>(
        other.df_, hmdf::concat_policy::common_columns);
    out.sync_from_df();
    // grizzlars documents concat() as resetting the index to 0..N-1.
    const size_t n = out.df_.get_index().size();
    std::vector<unsigned long> new_idx(n);
    for (size_t i = 0; i < n; ++i) new_idx[i] = static_cast<unsigned long>(i);
    out.df_.load_index(std::move(new_idx));
    return out;
}

// ── missing data ───────────────────────────────────────────────────────────

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

// ── window functions ───────────────────────────────────────────────────────

std::vector<double> GrizzlarFrame::cumulative(const std::string & col, const std::string & func) const
{
    const std::vector<double> data = get_column_double_or_cast(col);
    std::vector<double> out(data.size());
    if (func == "sum")
    {
        double acc = 0.0;
        for (size_t i = 0; i < data.size(); ++i) { acc += data[i]; out[i] = acc; }
    }
    else if (func == "prod")
    {
        double acc = 1.0;
        for (size_t i = 0; i < data.size(); ++i) { acc *= data[i]; out[i] = acc; }
    }
    else if (func == "min")
    {
        double acc = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < data.size(); ++i) { acc = std::min(acc, data[i]); out[i] = acc; }
    }
    else if (func == "max")
    {
        double acc = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < data.size(); ++i) { acc = std::max(acc, data[i]); out[i] = acc; }
    }
    else
        throw std::runtime_error("unknown cumulative function: " + func);
    return out;
}

std::vector<double> GrizzlarFrame::shift_col(const std::string & col, int64_t n) const
{
    const std::vector<double> data = get_column_double_or_cast(col);
    const int64_t len = static_cast<int64_t>(data.size());
    std::vector<double> out(data.size(), std::numeric_limits<double>::quiet_NaN());
    for (int64_t i = 0; i < len; ++i)
    {
        const int64_t src = i - n;
        if (src >= 0 && src < len)
            out[static_cast<size_t>(i)] = data[static_cast<size_t>(src)];
    }
    return out;
}

std::vector<double> GrizzlarFrame::pct_change(const std::string & col) const
{
    const std::vector<double> data = get_column_double_or_cast(col);
    std::vector<double> out(data.size(), std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 1; i < data.size(); ++i)
    {
        const double prev = data[i - 1];
        out[i] = (prev == 0.0) ? std::numeric_limits<double>::quiet_NaN() : (data[i] - prev) / prev;
    }
    return out;
}

std::vector<double> GrizzlarFrame::rolling(const std::string & col, size_t window, const std::string & func) const
{
    const std::vector<double> data = get_column_double_or_cast(col);
    std::vector<double> out(data.size(), std::numeric_limits<double>::quiet_NaN());
    if (window == 0 || window > data.size())
        return out;
    for (size_t i = window - 1; i < data.size(); ++i)
    {
        const size_t start = i - window + 1;
        if (func == "sum" || func == "mean")
        {
            double s = 0.0;
            for (size_t j = start; j <= i; ++j) s += data[j];
            out[i] = (func == "mean") ? s / static_cast<double>(window) : s;
        }
        else if (func == "min")
        {
            double mn = data[start];
            for (size_t j = start + 1; j <= i; ++j) mn = std::min(mn, data[j]);
            out[i] = mn;
        }
        else if (func == "max")
        {
            double mx = data[start];
            for (size_t j = start + 1; j <= i; ++j) mx = std::max(mx, data[j]);
            out[i] = mx;
        }
        else if (func == "std")
        {
            double s = 0.0;
            for (size_t j = start; j <= i; ++j) s += data[j];
            const double m = s / static_cast<double>(window);
            double sq = 0.0;
            for (size_t j = start; j <= i; ++j) { const double d = data[j] - m; sq += d * d; }
            out[i] = window > 1 ? std::sqrt(sq / static_cast<double>(window - 1)) : 0.0;
        }
        else
            throw std::runtime_error("unknown rolling function: " + func);
    }
    return out;
}

// ── CSV I/O ─────────────────────────────────────────────────────────────────

namespace
{

void write_csv_cell(std::ofstream & out, const std::string & s)
{
    if (s.find_first_of(",\"\n") == std::string::npos) { out << s; return; }
    out << '"';
    for (char c : s) { if (c == '"') out << '"'; out << c; }
    out << '"';
}

} // namespace

void GrizzlarFrame::to_csv(const std::string & path, bool write_index) const
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("cannot open file for writing: " + path);
    out.precision(15);

    bool first = true;
    if (write_index) { out << "index"; first = false; }
    for (const auto & name : col_order_)
    {
        if (!first) out << ',';
        write_csv_cell(out, name);
        first = false;
    }
    out << '\n';

    const size_t n = shape().first;
    const auto & idx = df_.get_index();
    for (size_t i = 0; i < n; ++i)
    {
        bool f = true;
        if (write_index) { out << idx[i]; f = false; }
        for (const auto & name : col_order_)
        {
            if (!f) out << ',';
            f = false;
            const std::string & type = col_types_.at(name);
            if (type == "double")
            {
                const double v = df_.get_column<double>(name.c_str())[i];
                if (!std::isnan(v)) out << v;
            }
            else if (type == "int64")
                out << df_.get_column<int64_t>(name.c_str())[i];
            else if (type == "bool")
                out << static_cast<int>(df_.get_column<uint8_t>(name.c_str())[i]);
            else
                write_csv_cell(out, df_.get_column<std::string>(name.c_str())[i]);
        }
        out << '\n';
    }
}

namespace
{

// Scans one line's comma-separated fields without allocating a container —
// calls fn(ptr, len) per field. Handles simple quoted fields (writing the
// unescaped content into `scratch`, reused across calls) so the common
// unquoted case touches no heap at all.
template <typename Fn>
void for_each_csv_field(const char * begin, const char * end, std::string & scratch, Fn && fn)
{
    const char * p = begin;
    if (p == end)
    {
        fn("", size_t(0));
        return;
    }
    for (;;)
    {
        if (p < end && *p == '"')
        {
            ++p;
            scratch.clear();
            while (p < end)
            {
                if (*p == '"')
                {
                    if (p + 1 < end && *(p + 1) == '"') { scratch += '"'; p += 2; }
                    else { ++p; break; }
                }
                else scratch += *p++;
            }
            fn(scratch.data(), scratch.size());
        }
        else
        {
            const char * fs = p;
            while (p < end && *p != ',') ++p;
            fn(fs, static_cast<size_t>(p - fs));
        }
        if (p >= end) break;
        ++p; // skip comma
        if (p == end) { fn("", size_t(0)); break; } // trailing comma -> empty last field
    }
}

int64_t parse_int64_field(const char * s, size_t len)
{
    int64_t v = 0;
    std::from_chars(s, s + len, v);
    return v;
}

double parse_double_field(const char * s, size_t len)
{
    double v = std::numeric_limits<double>::quiet_NaN();
    std::from_chars(s, s + len, v);
    return v;
}

unsigned long parse_ulong_field(const char * s, size_t len)
{
    unsigned long v = 0;
    std::from_chars(s, s + len, v);
    return v;
}

} // namespace

GrizzlarFrame GrizzlarFrame::read_csv_native(const std::string & path, const std::string & index_col_name)
{
    // Single read of the whole file (one syscall) instead of line-by-line
    // std::getline — avoids per-line stream buffering overhead and lets us
    // count rows up front to reserve() the typed column vectors exactly
    // once instead of growing them by repeated doubling.
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("cannot open file: " + path);
    const std::streamsize fsize = in.tellg();
    in.seekg(0);
    std::string buf(static_cast<size_t>(fsize), '\0');
    in.read(&buf[0], fsize);
    in.close();

    const char * data = buf.data();
    const char * fend = data + buf.size();

    const char * hdr_nl = static_cast<const char *>(std::memchr(data, '\n', static_cast<size_t>(fend - data)));
    if (!hdr_nl)
        return GrizzlarFrame{};
    const char * hdr_end = hdr_nl;
    if (hdr_end > data && *(hdr_end - 1) == '\r') --hdr_end;

    std::string scratch;
    std::vector<std::string> headers;
    for_each_csv_field(data, hdr_end, scratch, [&](const char * s, size_t len) { headers.emplace_back(s, len); });
    const size_t ncols = headers.size();
    const char * data_start = hdr_nl + 1;

    // Accurate row count up front (cheap: just counting newlines) so every
    // typed column vector below can be reserve()'d exactly once.
    size_t nrows_estimate = 0;
    for (const char * p = data_start; p < fend; ++p)
        if (*p == '\n') ++nrows_estimate;
    if (fend > data_start && *(fend - 1) != '\n') ++nrows_estimate; // unterminated last line

    // Pass 1: type-sample a bounded window of rows only.
    std::vector<int> type_id(ncols, 0); // 0 = int64, 1 = double, 2 = string
    {
        const char * p = data_start;
        for (size_t sampled = 0; sampled < 1000 && p < fend; ++sampled)
        {
            const char * nl = static_cast<const char *>(std::memchr(p, '\n', static_cast<size_t>(fend - p)));
            const char * line_end = nl ? nl : fend;
            const char * row_end = (line_end > p && *(line_end - 1) == '\r') ? line_end - 1 : line_end;
            size_t c = 0;
            for_each_csv_field(p, row_end, scratch, [&](const char * s, size_t len)
            {
                if (c >= ncols) { ++c; return; }
                if (len == 0) { ++c; return; }
                char * end = nullptr;
                if (type_id[c] == 0)
                {
                    std::strtoll(s, &end, 10);
                    if (end != s + len)
                    {
                        std::strtod(s, &end);
                        type_id[c] = (end == s + len) ? 1 : 2;
                    }
                }
                else if (type_id[c] == 1)
                {
                    std::strtod(s, &end);
                    if (end != s + len)
                        type_id[c] = 2;
                }
                ++c;
            });
            p = nl ? nl + 1 : fend;
        }
    }

    long index_col = -1;
    if (!index_col_name.empty())
        for (size_t c = 0; c < ncols; ++c)
            if (headers[c] == index_col_name) { index_col = static_cast<long>(c); break; }

    // Pass 2: real parse — convert each field directly into its final
    // typed column (from_chars straight off the buffer for numeric
    // columns, no intermediate std::string at all; only actual string
    // columns ever construct a std::string).
    std::vector<unsigned long> idx;
    std::vector<std::vector<int64_t>> int_cols(ncols);
    std::vector<std::vector<double>> dbl_cols(ncols);
    std::vector<std::vector<std::string>> str_cols(ncols);
    if (index_col < 0) idx.reserve(nrows_estimate);
    for (size_t c = 0; c < ncols; ++c)
    {
        if (static_cast<long>(c) == index_col) continue;
        if (type_id[c] == 0) int_cols[c].reserve(nrows_estimate);
        else if (type_id[c] == 1) dbl_cols[c].reserve(nrows_estimate);
        else str_cols[c].reserve(nrows_estimate);
    }

    size_t nrows = 0;
    {
        const char * p = data_start;
        while (p < fend)
        {
            const char * nl = static_cast<const char *>(std::memchr(p, '\n', static_cast<size_t>(fend - p)));
            const char * line_end = nl ? nl : fend;
            const char * row_end = (line_end > p && *(line_end - 1) == '\r') ? line_end - 1 : line_end;
            if (row_end > p)
            {
                size_t c = 0;
                for_each_csv_field(p, row_end, scratch, [&](const char * s, size_t len)
                {
                    if (c >= ncols) { ++c; return; }
                    if (static_cast<long>(c) == index_col)
                    {
                        idx.push_back(len == 0 ? 0ul : parse_ulong_field(s, len));
                    }
                    else if (type_id[c] == 0)
                        int_cols[c].push_back(len == 0 ? 0 : parse_int64_field(s, len));
                    else if (type_id[c] == 1)
                        dbl_cols[c].push_back(len == 0 ? std::numeric_limits<double>::quiet_NaN() : parse_double_field(s, len));
                    else
                        str_cols[c].emplace_back(s, len);
                    ++c;
                });
                ++nrows;
            }
            p = nl ? nl + 1 : fend;
        }
    }

    if (index_col < 0)
    {
        idx.resize(nrows);
        for (size_t r = 0; r < nrows; ++r) idx[r] = static_cast<unsigned long>(r);
    }

    GrizzlarFrame out;
    out.df_.load_index(std::move(idx));

    for (size_t c = 0; c < ncols; ++c)
    {
        if (static_cast<long>(c) == index_col)
            continue;
        const std::string & name = headers[c];
        if (type_id[c] == 0)
        {
            out.df_.load_column<int64_t>(name.c_str(), int_cols[c]);
            out.col_types_[name] = "int64";
        }
        else if (type_id[c] == 1)
        {
            out.df_.load_column<double>(name.c_str(), dbl_cols[c]);
            out.col_types_[name] = "double";
        }
        else
        {
            out.df_.load_column<std::string>(name.c_str(), str_cols[c]);
            out.col_types_[name] = "string";
        }
        out.col_order_.push_back(name);
    }
    return out;
}

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
        df_.load_column<double>(col.c_str(), vals);
    }
    else if (target_type == "int64")
    {
        std::vector<int64_t> vals(n);
        for (size_t i = 0; i < n; ++i)
            vals[i] = as_str[i].empty() ? 0 : std::strtoll(as_str[i].c_str(), nullptr, 10);
        df_.load_column<int64_t>(col.c_str(), vals);
    }
    else if (target_type == "bool")
    {
        std::vector<uint8_t> vals(n);
        for (size_t i = 0; i < n; ++i)
            vals[i] = (as_str[i] == "1" || as_str[i] == "true" || as_str[i] == "True") ? 1 : 0;
        df_.load_column<uint8_t>(col.c_str(), vals);
    }
    else
    {
        df_.load_column<std::string>(col.c_str(), as_str);
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

// ── boolean-frame / reduction ops ────────────────────────────────────────────

GrizzlarFrame GrizzlarFrame::isna_frame() const
{
    GrizzlarFrame out;
    out.df_.load_index(std::vector<unsigned long>(df_.get_index().begin(), df_.get_index().end()));
    const size_t n = shape().first;
    for (const auto & name : col_order_)
    {
        const std::string & type = col_types_.at(name);
        std::vector<uint8_t> mask(n, 0);
        if (type == "double")
        {
            const auto & c = df_.get_column<double>(name.c_str());
            for (size_t i = 0; i < n; ++i) mask[i] = std::isnan(c[i]) ? 1 : 0;
        }
        else if (type == "string")
        {
            const auto & c = df_.get_column<std::string>(name.c_str());
            for (size_t i = 0; i < n; ++i) mask[i] = c[i].empty() ? 1 : 0;
        }
        out.df_.load_column<uint8_t>(name.c_str(), mask);
        out.col_order_.push_back(name);
        out.col_types_[name] = "bool";
    }
    return out;
}

GrizzlarFrame GrizzlarFrame::notna_frame() const
{
    GrizzlarFrame out = isna_frame();
    for (const auto & name : out.col_order_)
    {
        auto & c = out.df_.get_column<uint8_t>(name.c_str());
        for (auto & v : c) v = v ? 0 : 1;
    }
    return out;
}

GrizzlarFrame GrizzlarFrame::where_frame(const GrizzlarFrame & cond_frame, double fill_val) const
{
    GrizzlarFrame out = deep_copy();
    const size_t n = shape().first;
    for (const auto & name : out.col_order_)
    {
        const std::string & type = out.col_types_.at(name);
        if (type != "double" && type != "int64")
            continue;
        if (!cond_frame.has_column(name))
            continue;
        const auto & cond_col = cond_frame.df_.get_column<uint8_t>(name.c_str());
        if (type == "int64")
            out.astype_col(name, "double");
        auto & c = out.df_.get_column<double>(name.c_str());
        for (size_t i = 0; i < n && i < cond_col.size(); ++i)
            if (!cond_col[i]) c[i] = fill_val;
    }
    return out;
}

GrizzlarFrame GrizzlarFrame::arith_scalar(const std::string & op, double scalar) const
{
    GrizzlarFrame out = deep_copy();
    const size_t n = shape().first;
    for (const auto & name : col_order_)
    {
        const std::string & type = col_types_.at(name);
        if (type != "double" && type != "int64")
            continue;
        if (type == "int64")
            out.astype_col(name, "double");
        auto & c = out.df_.get_column<double>(name.c_str());
        for (size_t i = 0; i < n; ++i)
        {
            double & v = c[i];
            if (op == "+") v = v + scalar;
            else if (op == "-") v = v - scalar;
            else if (op == "*") v = v * scalar;
            else if (op == "/") v = v / scalar;
            else if (op == "//") v = std::floor(v / scalar);
            else if (op == "%") v = std::fmod(v, scalar);
            else if (op == "**") v = std::pow(v, scalar);
            else throw std::runtime_error("unknown arithmetic operator: " + op);
        }
    }
    return out;
}

GrizzlarFrame GrizzlarFrame::arith_frame_op(const std::string & op, const GrizzlarFrame & other) const
{
    GrizzlarFrame out = deep_copy();
    const size_t n = shape().first;
    for (const auto & name : out.col_order_)
    {
        const std::string & type = out.col_types_.at(name);
        if (type != "double" && type != "int64")
            continue;
        if (!other.has_column(name))
            continue;
        const std::vector<double> rhs = other.get_column_double_or_cast(name);
        if (type == "int64")
            out.astype_col(name, "double");
        auto & c = out.df_.get_column<double>(name.c_str());
        for (size_t i = 0; i < n; ++i)
        {
            const double b = i < rhs.size() ? rhs[i] : std::numeric_limits<double>::quiet_NaN();
            double & v = c[i];
            if (op == "+") v = v + b;
            else if (op == "-") v = v - b;
            else if (op == "*") v = v * b;
            else if (op == "/") v = v / b;
            else if (op == "//") v = std::floor(v / b);
            else if (op == "%") v = std::fmod(v, b);
            else if (op == "**") v = std::pow(v, b);
            else throw std::runtime_error("unknown arithmetic operator: " + op);
        }
    }
    return out;
}

GrizzlarFrame GrizzlarFrame::compare_scalar(const std::string & op, double scalar) const
{
    GrizzlarFrame out;
    out.df_.load_index(std::vector<unsigned long>(df_.get_index().begin(), df_.get_index().end()));
    for (const auto & name : col_order_)
    {
        const std::string & type = col_types_.at(name);
        if (type != "double" && type != "int64")
            continue;
        const std::vector<double> data = get_column_double_or_cast(name);
        std::vector<uint8_t> mask(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            mask[i] = apply_op(op, data[i], scalar) ? 1 : 0;
        out.df_.load_column<uint8_t>(name.c_str(), mask);
        out.col_order_.push_back(name);
        out.col_types_[name] = "bool";
    }
    return out;
}

GrizzlarFrame GrizzlarFrame::reduce_all(const std::string & func) const
{
    GrizzlarFrame out;
    out.df_.load_index(std::vector<unsigned long>{ 0 });
    for (const auto & name : col_order_)
    {
        const std::string & type = col_types_.at(name);
        if (type != "double" && type != "int64" && type != "bool")
            continue;
        double v;
        if (type == "bool")
        {
            const auto & c = df_.get_column<uint8_t>(name.c_str());
            const size_t n = c.size();
            if (func == "count") v = static_cast<double>(n);
            else if (func == "sum") { v = 0; for (auto b : c) v += b; }
            else if (func == "mean") { double s = 0; for (auto b : c) s += b; v = n ? s / n : 0.0; }
            else if (func == "min") { v = 1.0; for (auto b : c) if (!b) { v = 0.0; break; } }
            else if (func == "max") { v = 0.0; for (auto b : c) if (b) { v = 1.0; break; } }
            else v = 0.0;
        }
        else if (func == "count") v = static_cast<double>(count(name));
        else if (func == "sum")   v = sum(name);
        else if (func == "mean")  v = mean(name);
        else if (func == "std")   v = std_dev(name);
        else if (func == "min")   v = col_min(name);
        else if (func == "max")   v = col_max(name);
        else if (func == "median") v = quantile(name, 0.5);
        else if (func == "var")
        {
            const double sd = std_dev(name);
            v = sd * sd;
        }
        else
            throw std::runtime_error("unknown reduce function: " + func);
        out.df_.load_column<double>(name.c_str(), std::vector<double>{ v });
        out.col_order_.push_back(name);
        out.col_types_[name] = "double";
    }
    return out;
}

// ── reshaping ─────────────────────────────────────────────────────────────

GrizzlarFrame GrizzlarFrame::transpose_frame() const
{
    const auto & idx = df_.get_index();
    const size_t nrows = idx.size();
    const size_t ncols = col_order_.size();

    GrizzlarFrame out;
    std::vector<unsigned long> new_idx(ncols);
    for (size_t j = 0; j < ncols; ++j) new_idx[j] = static_cast<unsigned long>(j);
    out.df_.load_index(std::move(new_idx));

    for (size_t i = 0; i < nrows; ++i)
    {
        std::vector<double> col(ncols, std::numeric_limits<double>::quiet_NaN());
        for (size_t j = 0; j < ncols; ++j)
        {
            const std::string & name = col_order_[j];
            const std::string & type = col_types_.at(name);
            if (type == "double")     col[j] = df_.get_column<double>(name.c_str())[i];
            else if (type == "int64") col[j] = static_cast<double>(df_.get_column<int64_t>(name.c_str())[i]);
            else if (type == "bool")  col[j] = static_cast<double>(df_.get_column<uint8_t>(name.c_str())[i]);
        }
        const std::string new_name = std::to_string(idx[i]);
        out.df_.load_column<double>(new_name.c_str(), col);
        out.col_order_.push_back(new_name);
        out.col_types_[new_name] = "double";
    }
    return out;
}

GrizzlarFrame GrizzlarFrame::set_index_col(const std::string & col, bool drop) const
{
    const std::string & type = col_type(col);
    const size_t n = shape().first;
    std::vector<unsigned long> new_idx(n);
    if (type == "double")
    {
        const auto & c = df_.get_column<double>(col.c_str());
        for (size_t i = 0; i < n; ++i) new_idx[i] = static_cast<unsigned long>(c[i]);
    }
    else if (type == "int64")
    {
        const auto & c = df_.get_column<int64_t>(col.c_str());
        for (size_t i = 0; i < n; ++i) new_idx[i] = static_cast<unsigned long>(c[i]);
    }
    else
    {
        for (size_t i = 0; i < n; ++i) new_idx[i] = static_cast<unsigned long>(i);
    }

    GrizzlarFrame out = deep_copy();
    out.df_.load_index(std::move(new_idx));
    if (drop)
        out.drop_column(col);
    return out;
}

GrizzlarFrame GrizzlarFrame::reset_index_frame(bool drop) const
{
    GrizzlarFrame out = deep_copy();
    const auto & old_idx = out.df_.get_index();
    std::vector<int64_t> idx_as_int64(old_idx.begin(), old_idx.end());
    const size_t n = old_idx.size();

    std::vector<unsigned long> new_idx(n);
    for (size_t i = 0; i < n; ++i) new_idx[i] = static_cast<unsigned long>(i);

    if (!drop)
    {
        out.df_.load_column<int64_t>("index", idx_as_int64);
        out.col_order_.insert(out.col_order_.begin(), "index");
        out.col_types_["index"] = "int64";
    }
    out.df_.load_index(std::move(new_idx));
    return out;
}

GrizzlarFrame GrizzlarFrame::melt_frame(
    const std::vector<std::string> & id_cols,
    const std::vector<std::string> & val_cols,
    const std::string & var_name,
    const std::string & value_name) const
{
    const size_t n = shape().first;
    const size_t nval = val_cols.size();
    const size_t out_n = n * nval;

    GrizzlarFrame out;
    {
        std::vector<unsigned long> idx(out_n);
        for (size_t i = 0; i < out_n; ++i) idx[i] = static_cast<unsigned long>(i);
        out.df_.load_index(std::move(idx));
    }

    for (const auto & id_col : id_cols)
    {
        const std::string & type = col_type(id_col);
        if (type == "double")
        {
            const auto & c = df_.get_column<double>(id_col.c_str());
            std::vector<double> rep(out_n);
            for (size_t v = 0; v < nval; ++v)
                for (size_t i = 0; i < n; ++i) rep[v * n + i] = c[i];
            out.df_.load_column<double>(id_col.c_str(), rep);
        }
        else if (type == "int64")
        {
            const auto & c = df_.get_column<int64_t>(id_col.c_str());
            std::vector<int64_t> rep(out_n);
            for (size_t v = 0; v < nval; ++v)
                for (size_t i = 0; i < n; ++i) rep[v * n + i] = c[i];
            out.df_.load_column<int64_t>(id_col.c_str(), rep);
        }
        else if (type == "string")
        {
            const auto & c = df_.get_column<std::string>(id_col.c_str());
            std::vector<std::string> rep(out_n);
            for (size_t v = 0; v < nval; ++v)
                for (size_t i = 0; i < n; ++i) rep[v * n + i] = c[i];
            out.df_.load_column<std::string>(id_col.c_str(), rep);
        }
        out.col_order_.push_back(id_col);
        out.col_types_[id_col] = type;
    }

    std::vector<std::string> var_col(out_n);
    std::vector<double> value_col(out_n, std::numeric_limits<double>::quiet_NaN());
    for (size_t v = 0; v < nval; ++v)
    {
        const std::string & vc = val_cols[v];
        const std::string & type = col_type(vc);
        for (size_t i = 0; i < n; ++i)
        {
            var_col[v * n + i] = vc;
            if (type == "double")     value_col[v * n + i] = df_.get_column<double>(vc.c_str())[i];
            else if (type == "int64") value_col[v * n + i] = static_cast<double>(df_.get_column<int64_t>(vc.c_str())[i]);
            else if (type == "bool")  value_col[v * n + i] = static_cast<double>(df_.get_column<uint8_t>(vc.c_str())[i]);
        }
    }
    out.df_.load_column<std::string>(var_name.c_str(), var_col);
    out.df_.load_column<double>(value_name.c_str(), value_col);
    out.col_order_.push_back(var_name);
    out.col_types_[var_name] = "string";
    out.col_order_.push_back(value_name);
    out.col_types_[value_name] = "double";
    return out;
}
