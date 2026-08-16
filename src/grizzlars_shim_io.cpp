// grizzlars_shim_io.cpp — CSV I/O: to_csv (straightforward sequential
// writer) and read_csv_native (two-pass type-sampling reader, parallel
// chunked parse for large files via hmdf's already-running thread pool).
#include "grizzlars_shim.h"

#include <DataFrame/Utils/Threads/ThreadGranularity.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <stdexcept>

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
    //
    // For large files, the data region is split at newline boundaries into
    // one chunk per pool thread, each parsed independently (no shared
    // mutable state during parsing — every chunk only touches its own
    // local vectors) and dispatched onto hmdf's already-running
    // ThreadGranularity::thr_pool_. Results are merged in order afterward,
    // moving (not copying) string cells into the final vectors. Small
    // files stay on the single-chunk/single-threaded path unchanged —
    // this must not regress already-fast small/numeric-heavy loads (the
    // per-file stock-data benchmark loads are a few thousand rows each,
    // comfortably under the threshold).
    struct ChunkResult
    {
        size_t nrows{ 0 };
        std::vector<unsigned long> idx;
        std::vector<std::vector<int64_t>> int_cols;
        std::vector<std::vector<double>> dbl_cols;
        std::vector<std::vector<std::string>> str_cols;
        explicit ChunkResult(size_t nc) : int_cols(nc), dbl_cols(nc), str_cols(nc) {}
    };

    auto parse_chunk = [&](const char * cs, const char * ce) -> ChunkResult
    {
        ChunkResult r(ncols);
        std::string local_scratch;
        const char * p = cs;
        while (p < ce)
        {
            const char * nl = static_cast<const char *>(std::memchr(p, '\n', static_cast<size_t>(ce - p)));
            const char * line_end = nl ? nl : ce;
            const char * row_end = (line_end > p && *(line_end - 1) == '\r') ? line_end - 1 : line_end;
            if (row_end > p)
            {
                size_t c = 0;
                for_each_csv_field(p, row_end, local_scratch, [&](const char * s, size_t len)
                {
                    if (c >= ncols) { ++c; return; }
                    if (static_cast<long>(c) == index_col)
                        r.idx.push_back(len == 0 ? 0ul : parse_ulong_field(s, len));
                    else if (type_id[c] == 0)
                        r.int_cols[c].push_back(len == 0 ? 0 : parse_int64_field(s, len));
                    else if (type_id[c] == 1)
                        r.dbl_cols[c].push_back(len == 0 ? std::numeric_limits<double>::quiet_NaN() : parse_double_field(s, len));
                    else
                        r.str_cols[c].emplace_back(s, len);
                    ++c;
                });
                ++r.nrows;
            }
            p = nl ? nl + 1 : ce;
        }
        return r;
    };

    constexpr size_t PARALLEL_PARSE_THRESHOLD = 50000;
    std::vector<ChunkResult> chunks;
    if (nrows_estimate >= PARALLEL_PARSE_THRESHOLD)
    {
        const size_t nthreads = std::min<size_t>(
            std::max<size_t>(1, static_cast<size_t>(hmdf::ThreadGranularity::get_thread_level())), 16);
        std::vector<const char *> chunk_starts(nthreads + 1);
        chunk_starts[0] = data_start;
        const size_t data_len = static_cast<size_t>(fend - data_start);
        const size_t approx_chunk = (data_len + nthreads - 1) / nthreads;
        for (size_t t = 1; t < nthreads; ++t)
        {
            const char * split = data_start + t * approx_chunk;
            if (split >= fend) { chunk_starts[t] = fend; continue; }
            const char * nl = static_cast<const char *>(std::memchr(split, '\n', static_cast<size_t>(fend - split)));
            chunk_starts[t] = nl ? nl + 1 : fend;
        }
        chunk_starts[nthreads] = fend;

        std::vector<std::future<ChunkResult>> futures;
        futures.reserve(nthreads);
        for (size_t t = 0; t < nthreads; ++t)
        {
            const char * cs = chunk_starts[t];
            const char * ce = chunk_starts[t + 1];
            if (cs >= ce) continue;
            futures.push_back(hmdf::ThreadGranularity::thr_pool_.dispatch(false, parse_chunk, cs, ce));
        }
        chunks.reserve(futures.size());
        for (auto & fut : futures) chunks.push_back(fut.get());
    }
    else
    {
        chunks.push_back(parse_chunk(data_start, fend));
    }

    size_t nrows = 0;
    for (const auto & ch : chunks) nrows += ch.nrows;

    std::vector<unsigned long> idx;
    std::vector<std::vector<int64_t>> int_cols(ncols);
    std::vector<std::vector<double>> dbl_cols(ncols);
    std::vector<std::vector<std::string>> str_cols(ncols);
    idx.reserve(nrows);
    for (size_t c = 0; c < ncols; ++c)
    {
        if (static_cast<long>(c) == index_col) continue;
        if (type_id[c] == 0) int_cols[c].reserve(nrows);
        else if (type_id[c] == 1) dbl_cols[c].reserve(nrows);
        else str_cols[c].reserve(nrows);
    }

    for (auto & ch : chunks)
    {
        if (index_col >= 0)
            idx.insert(idx.end(), ch.idx.begin(), ch.idx.end());
        for (size_t c = 0; c < ncols; ++c)
        {
            if (static_cast<long>(c) == index_col) continue;
            if (type_id[c] == 0)
                int_cols[c].insert(int_cols[c].end(), ch.int_cols[c].begin(), ch.int_cols[c].end());
            else if (type_id[c] == 1)
                dbl_cols[c].insert(dbl_cols[c].end(), ch.dbl_cols[c].begin(), ch.dbl_cols[c].end());
            else
                str_cols[c].insert(str_cols[c].end(),
                    std::make_move_iterator(ch.str_cols[c].begin()),
                    std::make_move_iterator(ch.str_cols[c].end()));
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
            out.df_.load_column<int64_t>(name.c_str(), std::move(int_cols[c]));
            out.col_types_[name] = "int64";
        }
        else if (type_id[c] == 1)
        {
            out.df_.load_column<double>(name.c_str(), std::move(dbl_cols[c]));
            out.col_types_[name] = "double";
        }
        else
        {
            out.df_.load_column<std::string>(name.c_str(), std::move(str_cols[c]));
            out.col_types_[name] = "string";
        }
        out.col_order_.push_back(name);
    }
    return out;
}

