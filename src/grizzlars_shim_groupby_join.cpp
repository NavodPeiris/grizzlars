// grizzlars_shim_groupby_join.cpp — groupby_agg, join_by_index, concat_frame,
// and sync_from_df (column bookkeeping rediscovery used after both).
#include "grizzlars_shim.h"
#include "grizzlars_shim_internal.h"

#include <algorithm>
#include <cmath>
#include <typeindex>
#include <unordered_map>

using grizzlars_detail::require_numeric;

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

// `at(row)` is a pre-bound accessor into an already-fetched column
// reference — aggregate_group() must NOT call get_column<T>() itself,
// since that's invoked once per row *within* each group otherwise (a real
// measured cost: get_column<T> re-does a name lookup, and possibly a
// lock, on every call — for a 49k-row/11-group groupby that's tens of
// thousands of redundant lookups instead of one per column).
template <typename Accessor>
double aggregate_group(Accessor && at, const std::vector<size_t> & rows, const std::string & func)
{
    const size_t cnt = rows.size();
    if (func == "count")
        return static_cast<double>(cnt);
    if (cnt == 0)
        return 0.0;
    if (func == "first")
        return at(rows.front());
    if (func == "last")
        return at(rows.back());

    double s = 0.0;
    double mn = at(rows[0]);
    double mx = mn;
    for (size_t r : rows)
    {
        const double v = at(r);
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
            const double d = at(r) - m;
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
    // directly from df_'s real column storage.
    //
    // The by_col column reference is fetched once *before* the loop (not
    // once per row inside it, as an earlier version did) — get_column<T>
    // does a name lookup on every call, and doing that 49k+ times measured
    // as a real chunk of groupby's wall-clock time on medium-size data.
    std::vector<std::string> group_keys;
    std::unordered_map<std::string, size_t> key_to_group;
    key_to_group.reserve(n / 4 + 8);
    std::vector<std::vector<size_t>> group_rows;

    auto bucket = [&](const std::string & key, size_t i)
    {
        auto it = key_to_group.find(key);
        if (it == key_to_group.end())
        {
            key_to_group.emplace(key, group_keys.size());
            group_keys.push_back(key);
            group_rows.emplace_back(std::vector<size_t>{ i });
        }
        else
        {
            group_rows[it->second].push_back(i);
        }
    };

    if (by_type == "double")
    {
        const auto & c = df_.get_column<double>(by_col.c_str());
        for (size_t i = 0; i < n; ++i) bucket(std::to_string(c[i]), i);
    }
    else if (by_type == "int64")
    {
        const auto & c = df_.get_column<int64_t>(by_col.c_str());
        for (size_t i = 0; i < n; ++i) bucket(std::to_string(c[i]), i);
    }
    else if (by_type == "bool")
    {
        const auto & c = df_.get_column<uint8_t>(by_col.c_str());
        for (size_t i = 0; i < n; ++i) bucket(std::to_string(c[i]), i);
    }
    else
    {
        const auto & c = df_.get_column<std::string>(by_col.c_str());
        for (size_t i = 0; i < n; ++i) bucket(c[i], i);
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
        out.df_.load_column<double>(by_col.c_str(), std::move(vals));
    }
    else if (by_type == "int64")
    {
        std::vector<int64_t> vals(ngroups);
        for (size_t g = 0; g < ngroups; ++g) vals[g] = df_.get_column<int64_t>(by_col.c_str())[group_rows[g].front()];
        out.df_.load_column<int64_t>(by_col.c_str(), std::move(vals));
    }
    else if (by_type == "bool")
    {
        std::vector<uint8_t> vals(ngroups);
        for (size_t g = 0; g < ngroups; ++g) vals[g] = df_.get_column<uint8_t>(by_col.c_str())[group_rows[g].front()];
        out.df_.load_column<uint8_t>(by_col.c_str(), std::move(vals));
    }
    else
    {
        std::vector<std::string> vals(ngroups);
        for (size_t g = 0; g < ngroups; ++g) vals[g] = df_.get_column<std::string>(by_col.c_str())[group_rows[g].front()];
        out.df_.load_column<std::string>(by_col.c_str(), std::move(vals));
    }
    out.col_order_.push_back(by_col);
    out.col_types_[by_col] = by_type;

    for (size_t s = 0; s < agg_cols.size(); ++s)
    {
        const std::string & col = agg_cols[s];
        const std::string & func = agg_funcs[s];
        const std::string & col_t = require_numeric(col_types_, col);
        std::vector<double> vals(ngroups);
        if (col_t == "double")
        {
            const auto & c = df_.get_column<double>(col.c_str());
            auto at = [&](size_t r) { return c[r]; };
            for (size_t g = 0; g < ngroups; ++g)
                vals[g] = aggregate_group(at, group_rows[g], func);
        }
        else
        {
            const auto & c = df_.get_column<int64_t>(col.c_str());
            auto at = [&](size_t r) { return static_cast<double>(c[r]); };
            for (size_t g = 0; g < ngroups; ++g)
                vals[g] = aggregate_group(at, group_rows[g], func);
        }
        out.df_.load_column<double>(col.c_str(), std::move(vals));
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
