// grizzlars_shim_filter_sort.cpp — sorting (sort_by/sort_index), row
// selection (iloc/take_rows/select_columns/from_positions), and filtering
// (filter_by_mask_list/filter_col_scalar_double/compare_col_scalar_double).
#include "grizzlars_shim.h"
#include "grizzlars_shim_internal.h"

#include <DataFrame/Utils/Threads/ThreadGranularity.h>

#include <algorithm>

using grizzlars_detail::apply_op;

namespace
{

// Gathers positions[i] from a source column into a fresh vector, in
// parallel for large inputs. Written here (not in cpp_lib/DataFrame)
// specifically because hmdf's own get_data_by_loc gathers std::string
// columns element-by-element and, for large mostly-string DataFrames,
// that dominates filter/iloc/take_rows wall-clock time — the per-element
// work is embarrassingly parallel regardless of column type.
//
// Dispatches onto hmdf's own already-running ThreadGranularity::thr_pool_
// (started once at module import via set_optimum_thread_level()) via
// parallel_loop(), rather than spawning fresh std::thread objects per
// call — an earlier version did the latter, which meant up to 16 new OS
// threads created *per column* (up to ~190 thread creations for one
// 12-column filter). A/B measured directly: single-threaded gather took
// 827ms on a 2M-row mostly-string filter; spawning fresh threads got that
// to ~360ms; reusing the existing pool via parallel_loop avoids paying
// thread-creation cost on every single call.
template <typename T, typename GetFn>
std::vector<T> parallel_gather(const std::vector<long> & positions, size_t src_size, GetFn && get)
{
    const size_t n = positions.size();
    std::vector<T> out(n);
    auto routine = [&](size_t begin, size_t end) -> void
    {
        for (size_t i = begin; i < end; ++i)
        {
            const long p = positions[i];
            const size_t idx = p >= 0 ? static_cast<size_t>(p) : src_size + static_cast<size_t>(p);
            out[i] = get(idx);
        }
    };

    if (n < 20000)
    {
        routine(0, n);
        return out;
    }
    auto futures = hmdf::ThreadGranularity::thr_pool_.parallel_loop<T>(size_t(0), n, routine);
    for (auto & fut : futures) fut.get();
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
        hmdf::ThreadGranularity::thr_pool_.parallel_sort(perm.begin(), perm.end(), [&](long a, long b) {
            return ascending ? c[a] < c[b] : c[a] > c[b];
        });
    }
    else if (type == "int64")
    {
        const auto & c = df_.get_column<int64_t>(by.c_str());
        hmdf::ThreadGranularity::thr_pool_.parallel_sort(perm.begin(), perm.end(), [&](long a, long b) {
            return ascending ? c[a] < c[b] : c[a] > c[b];
        });
    }
    else if (type == "bool")
    {
        const auto & c = df_.get_column<uint8_t>(by.c_str());
        hmdf::ThreadGranularity::thr_pool_.parallel_sort(perm.begin(), perm.end(), [&](long a, long b) {
            return ascending ? c[a] < c[b] : c[a] > c[b];
        });
    }
    else
    {
        const auto & c = df_.get_column<std::string>(by.c_str());
        hmdf::ThreadGranularity::thr_pool_.parallel_sort(perm.begin(), perm.end(), [&](long a, long b) {
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
    hmdf::ThreadGranularity::thr_pool_.parallel_sort(perm.begin(), perm.end(), [&](long a, long b) {
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
