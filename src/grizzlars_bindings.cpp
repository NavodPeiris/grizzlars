// grizzlars_bindings.cpp — pybind11 bindings for the hmdf C++ DataFrame library.
//
// Exposes a GrizzlarFrame class that wraps StdDataFrame<unsigned long>.
// Supported column types: double (float64), int64, bool, str.
// Index type: unsigned 64-bit integer (auto-assigned 0..N-1 if omitted).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <DataFrame/DataFrame.h>
#include <DataFrame/DataFrameStatsVisitors.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <execution>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace py = pybind11;
using namespace hmdf;

using ulong = unsigned long;
using GDF   = StdDataFrame<ulong>;

// ─── type detection ──────────────────────────────────────────────────────────

static std::string detect_type(py::object data) {
    if (py::isinstance<py::array>(data)) {
        char kind = py::cast<py::array>(data).dtype().kind();
        if (kind == 'f') return "double";
        if (kind == 'i' || kind == 'u') return "int64";
        if (kind == 'b') return "bool";
        return "string";
    }
    if (py::isinstance<py::list>(data)) {
        py::list lst = py::cast<py::list>(data);
        if (lst.empty()) return "double";
        py::object first = lst[0];
        if (py::isinstance<py::bool_>(first)) return "bool";
        if (py::isinstance<py::float_>(first)) return "double";
        if (py::isinstance<py::int_>(first))   return "int64";
        if (py::isinstance<py::str>(first))    return "string";
    }
    return "double";
}

// ─── conversion helpers ──────────────────────────────────────────────────────

template<typename T>
static std::vector<T> to_vec(py::object obj) {
    if (py::isinstance<py::array>(obj)) {
        auto arr = py::cast<
            py::array_t<T, py::array::c_style | py::array::forcecast>>(obj);
        auto buf = arr.request();
        auto* ptr = static_cast<T*>(buf.ptr);
        return std::vector<T>(ptr, ptr + buf.size);
    }
    std::vector<T> result;
    for (auto item : py::cast<py::list>(obj))
        result.push_back(py::cast<T>(item));
    return result;
}

static std::vector<std::string> to_str_vec(py::object obj) {
    std::vector<std::string> result;
    if (py::isinstance<py::array>(obj)) {
        for (auto item : py::cast<py::list>(py::cast<py::array>(obj).attr("tolist")()))
            result.push_back(py::cast<std::string>(item));
        return result;
    }
    for (auto item : py::cast<py::list>(obj))
        result.push_back(py::cast<std::string>(item));
    return result;
}

// ─── CSV native-reader helpers ────────────────────────────────────────────────

// Strip hmdf ":count:<type>" suffix from a column name, e.g.
//   "FORD_Close:12265:<double>"  →  "FORD_Close"
// Standard headers (no '>') are returned unchanged.
static std::string strip_hmdf_annotation(const std::string &s) {
    if (s.size() < 5 || s.back() != '>') return s;
    auto lt = s.rfind('<');
    if (lt == std::string::npos || lt < 2 || s[lt - 1] != ':') return s;
    size_t ed = lt - 2;
    if (!std::isdigit((unsigned char)s[ed])) return s;
    size_t sd = ed;
    while (sd > 0 && std::isdigit((unsigned char)s[sd - 1])) --sd;
    if (sd == 0 || s[sd - 1] != ':') return s;
    return s.substr(0, sd - 1);
}

// RFC-4180-compliant CSV row parser.
// Fast path: unquoted fields are assigned directly from a pointer range
// (one emplace_back + one memcpy) instead of N char-by-char appends.
// Slow path: quoted fields handle escaped double-quotes correctly.
static void parse_csv_row_fast(const char *p, size_t len,
                                std::vector<std::string> &fields) {
    fields.clear();
    const char *end = p + len;
    if (end > p && *(end - 1) == '\r') --end;
    if (p == end) return;

    for (;;) {
        if (p >= end) {
            // trailing comma → empty last field
            fields.emplace_back();
            break;
        }
        if (*p == '"') {
            // Quoted field: slow path handles embedded commas and "" escapes
            ++p;
            std::string f;
            while (p < end) {
                char c = *p++;
                if (c == '"') {
                    if (p < end && *p == '"') { f += '"'; ++p; }
                    else break;
                } else {
                    f += c;
                }
            }
            fields.push_back(std::move(f));
        } else {
            // Unquoted field: scan to next comma, assign whole range at once
            const char *fs = p;
            while (p < end && *p != ',') ++p;
            fields.emplace_back(fs, static_cast<size_t>(p - fs));
        }
        if (p >= end) break;
        ++p;  // skip comma separator
    }
}

// ─── memory-mapped file helpers ───────────────────────────────────────────────

struct MmapView {
    const char *data{nullptr};
    size_t      size{0};
    void *      handle{nullptr};  // platform opaque handle
};

// Returns a read-only memory-mapped view of the file.
// Falls back gracefully if mmap fails (view.data == nullptr).
static MmapView mmap_open(const std::string &path) {
    MmapView v;
#ifdef _WIN32
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                nullptr, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
                                nullptr);
    if (hFile == INVALID_HANDLE_VALUE) return v;
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(hFile, &sz)) { CloseHandle(hFile); return v; }
    v.size = static_cast<size_t>(sz.QuadPart);
    if (v.size == 0) { CloseHandle(hFile); return v; }
    HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(hFile);
    if (!hMap) return v;
    v.data   = static_cast<const char *>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
    v.handle = hMap;
    if (!v.data) { CloseHandle(hMap); v.handle = nullptr; }
#else
    // POSIX mmap
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return v;
    struct stat st{};
    if (::fstat(fd, &st) != 0) { ::close(fd); return v; }
    v.size = static_cast<size_t>(st.st_size);
    if (v.size == 0) { ::close(fd); return v; }
    void *ptr = ::mmap(nullptr, v.size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (ptr == MAP_FAILED) { v.size = 0; return v; }
    v.data   = static_cast<const char *>(ptr);
    v.handle = ptr;
#endif
    return v;
}

static void mmap_close(MmapView &v) {
    if (!v.data) return;
#ifdef _WIN32
    UnmapViewOfFile(v.data);
    if (v.handle) CloseHandle(v.handle);
#else
    ::munmap(v.handle, v.size);
#endif
    v = MmapView{};
}

static bool csv_try_int64(const char *s, size_t len, int64_t &out) {
    if (!len) return false;
    char *end; errno = 0;
    long long v = std::strtoll(s, &end, 10);
    if ((size_t)(end - s) != len || errno) return false;
    out = static_cast<int64_t>(v); return true;
}
static bool csv_try_double(const char *s, size_t len, double &out) {
    if (!len) return false;
    char *end;
    out = std::strtod(s, &end);
    return (size_t)(end - s) == len;
}

// ─── GrizzlarFrame ────────────────────────────────────────────────────────────

class GrizzlarFrame {
public:
    GDF df_;
    std::unordered_map<std::string, std::string> col_types_;
    std::vector<std::string> col_order_;

    // ── private helpers ──────────────────────────────────────────────────────

    // Build a new GrizzlarFrame containing only the rows at positions `locs`.
    // Uses resize+pointer-write (avoids push_back overhead) and prefetches
    // the next gather address to hide random-access latency.
    GrizzlarFrame extract_rows(const std::vector<size_t> &locs) const {
        GrizzlarFrame out;
        const size_t n = locs.size();
        const auto &src_idx = df_.get_index();

        // ── index ──────────────────────────────────────────────────────────────
        std::vector<ulong> new_idx(n);
        {
            ulong *dst = new_idx.data();
            const ulong *src = src_idx.data();
            for (size_t j = 0; j < n; ++j) {
                if (j + HMDF_PF_DIST < n) HMDF_PREFETCH_R(src + locs[j + HMDF_PF_DIST]);
                dst[j] = src[locs[j]];
            }
        }
        out.df_.load_index(std::move(new_idx));

        // ── data columns ───────────────────────────────────────────────────────
        for (const auto &name : col_order_) {
            out.col_order_.push_back(name);
            const std::string &type = col_types_.at(name);
            out.col_types_[name] = type;

            if (type == "double") {
                const auto &v = df_.get_column<double>(name.c_str());
                std::vector<double> nv(n);
                double       *dst = nv.data();
                const double *src = v.data();
                for (size_t j = 0; j < n; ++j) {
                    if (j + HMDF_PF_DIST < n) HMDF_PREFETCH_R(src + locs[j + HMDF_PF_DIST]);
                    dst[j] = src[locs[j]];
                }
                out.df_.load_column<double>(name.c_str(), std::move(nv));
            } else if (type == "int64") {
                const auto &v = df_.get_column<int64_t>(name.c_str());
                std::vector<int64_t> nv(n);
                int64_t       *dst = nv.data();
                const int64_t *src = v.data();
                for (size_t j = 0; j < n; ++j) {
                    if (j + HMDF_PF_DIST < n) HMDF_PREFETCH_R(src + locs[j + HMDF_PF_DIST]);
                    dst[j] = src[locs[j]];
                }
                out.df_.load_column<int64_t>(name.c_str(), std::move(nv));
            } else if (type == "bool") {
                const auto &v = df_.get_column<bool>(name.c_str());
                std::vector<bool> nv(n);
                for (size_t j = 0; j < n; ++j) nv[j] = v[locs[j]];
                out.df_.load_column<bool>(name.c_str(), std::move(nv));
            } else {
                const auto &v = df_.get_column<std::string>(name.c_str());
                std::vector<std::string> nv(n);
                for (size_t j = 0; j < n; ++j) nv[j] = v[locs[j]];
                out.df_.load_column<std::string>(name.c_str(), std::move(nv));
            }
        }
        return out;
    }

    // Parallel scatter: apply a row permutation/index list to all columns
    // simultaneously.  Inspired by vaex/polars: each column is independent,
    // so N columns can be gathered on N threads with no synchronisation.
    GrizzlarFrame extract_rows_parallel(const std::vector<size_t> &locs) const {
        const size_t n_out = locs.size();
        const size_t ncols = col_order_.size();

        GrizzlarFrame out;
        out.col_order_ = col_order_;
        out.col_types_ = col_types_;

        struct ColOut {
            std::vector<int64_t>     ints;
            std::vector<double>      dbls;
            std::vector<bool>        bools;
            std::vector<std::string> strs;
        };
        std::vector<ulong>  new_idx(n_out);
        std::vector<ColOut> col_outs(ncols);
        for (size_t ci = 0; ci < ncols; ++ci) {
            const std::string &type = col_types_.at(col_order_[ci]);
            if      (type == "double") col_outs[ci].dbls.resize(n_out);
            else if (type == "int64")  col_outs[ci].ints.resize(n_out);
            else if (type == "bool")   col_outs[ci].bools.resize(n_out, false);
            else                       col_outs[ci].strs.resize(n_out);
        }

        // gather_unit: 0=index, 1..ncols = each column
        auto gather_unit = [&](size_t unit) {
            if (unit == 0) {
                const ulong *src = df_.get_index().data();
                for (size_t j = 0; j < n_out; ++j) {
                    if (j + HMDF_PF_DIST < n_out) HMDF_PREFETCH_R(src + locs[j + HMDF_PF_DIST]);
                    new_idx[j] = src[locs[j]];
                }
            } else {
                const size_t ci    = unit - 1;
                const auto  &cname = col_order_[ci];
                const auto  &type  = col_types_.at(cname);
                if (type == "double") {
                    const double *src = df_.get_column<double>(cname.c_str()).data();
                    double       *dst = col_outs[ci].dbls.data();
                    for (size_t j = 0; j < n_out; ++j) {
                        if (j + HMDF_PF_DIST < n_out) HMDF_PREFETCH_R(src + locs[j + HMDF_PF_DIST]);
                        dst[j] = src[locs[j]];
                    }
                } else if (type == "int64") {
                    const int64_t *src = df_.get_column<int64_t>(cname.c_str()).data();
                    int64_t       *dst = col_outs[ci].ints.data();
                    for (size_t j = 0; j < n_out; ++j) {
                        if (j + HMDF_PF_DIST < n_out) HMDF_PREFETCH_R(src + locs[j + HMDF_PF_DIST]);
                        dst[j] = src[locs[j]];
                    }
                } else if (type == "bool") {
                    const auto &sv = df_.get_column<bool>(cname.c_str());
                    for (size_t j = 0; j < n_out; ++j)
                        col_outs[ci].bools[j] = sv[locs[j]];
                } else {
                    const auto &sv = df_.get_column<std::string>(cname.c_str());
                    auto       &dv = col_outs[ci].strs;
                    for (size_t j = 0; j < n_out; ++j)
                        dv[j] = sv[locs[j]];
                }
            }
        };

        const size_t total_units = ncols + 1;
        const size_t nthreads = (n_out >= 50000 && ncols >= 2)
            ? std::min(total_units, (size_t)std::thread::hardware_concurrency())
            : 1;

        if (nthreads <= 1) {
            for (size_t u = 0; u < total_units; ++u) gather_unit(u);
        } else {
            const size_t upt = (total_units + nthreads - 1) / nthreads;
            std::vector<std::future<void>> futs;
            for (size_t t = 0; t < nthreads; ++t) {
                size_t us = t * upt, ue = std::min(us + upt, total_units);
                if (us >= total_units) break;
                futs.push_back(std::async(std::launch::async,
                    [us, ue, &gather_unit]() {
                        for (size_t u = us; u < ue; ++u) gather_unit(u);
                    }));
            }
            for (auto &f : futs) f.wait();
        }

        out.df_.load_index(std::move(new_idx));
        for (size_t ci = 0; ci < ncols; ++ci) {
            const auto &cname = col_order_[ci];
            const auto &type  = col_types_.at(cname);
            if      (type == "double")  out.df_.load_column<double>(cname.c_str(),      std::move(col_outs[ci].dbls));
            else if (type == "int64")   out.df_.load_column<int64_t>(cname.c_str(),     std::move(col_outs[ci].ints));
            else if (type == "bool")    out.df_.load_column<bool>(cname.c_str(),        std::move(col_outs[ci].bools));
            else                        out.df_.load_column<std::string>(cname.c_str(), std::move(col_outs[ci].strs));
        }
        return out;
    }

    // Wrap an hmdf GDF result (from join/concat) back into a GrizzlarFrame,
    // rediscovering column names and types via get_columns_info.
    static GrizzlarFrame from_gdf(GDF &&gdf) {
        GrizzlarFrame out;
        out.df_ = std::move(gdf);
        auto info =
            out.df_.get_columns_info<double, int64_t, bool, std::string>();
        for (const auto &[raw_name, idx, tidx] : info) {
            std::string name(raw_name.c_str());
            out.col_order_.push_back(name);
            if      (tidx == std::type_index(typeid(double)))      out.col_types_[name] = "double";
            else if (tidx == std::type_index(typeid(int64_t)))     out.col_types_[name] = "int64";
            else if (tidx == std::type_index(typeid(bool)))        out.col_types_[name] = "bool";
            else                                                    out.col_types_[name] = "string";
        }
        return out;
    }

    // Native C++ CSV reader — multi-threaded, reads entire file at once.
    //
    // Algorithm:
    //   1. Read whole file into memory (one syscall).
    //   2. Parse header + type-sample first 1000 rows.
    //   3. Split data region into N chunks at newline boundaries.
    //   4. Parse each chunk in a separate std::async thread.
    //   5. Merge per-chunk column vectors and assemble GrizzlarFrame.
    static GrizzlarFrame read_csv_native(const std::string &path,
                                          const std::string &index_col_name) {
        // ── Step 1: map whole file (zero-copy on OS page cache) ───────────────
        MmapView mmap = mmap_open(path);
        // Fallback to file.read() if mmap failed
        std::string fbuf_fallback;
        const char *data;
        size_t fsz;
        if (mmap.data) {
            data = mmap.data;
            fsz  = mmap.size;
        } else {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file) throw std::runtime_error("Cannot open CSV: " + path);
            const std::streamsize fsz2 = file.tellg();
            file.seekg(0);
            fbuf_fallback.assign(static_cast<size_t>(fsz2) + 1, '\0');
            file.read(&fbuf_fallback[0], fsz2);
            data = fbuf_fallback.data();
            fsz  = static_cast<size_t>(fsz2);
        }
        struct MmapGuard { MmapView &v; ~MmapGuard() { mmap_close(v); } } _guard{mmap};

        const char *fend = data + fsz;

        // ── Step 2: header ─────────────────────────────────────────────────────
        const char *hdr_nl = static_cast<const char *>(std::memchr(data, '\n', fend - data));
        if (!hdr_nl) return GrizzlarFrame{};

        std::vector<std::string> row_buf;
        row_buf.reserve(32);
        parse_csv_row_fast(data, static_cast<size_t>(hdr_nl - data), row_buf);
        const size_t ncols = row_buf.size();
        std::vector<std::string> headers(ncols);
        for (size_t c = 0; c < ncols; ++c)
            headers[c] = strip_hmdf_annotation(row_buf[c]);

        const char *data_start = hdr_nl + 1;

        // ── type sampling (first 1000 data rows from buffer) ───────────────────
        // 0 = int64   1 = double   2 = string
        std::vector<int> type_id(ncols, 0);
        {
            const char *sp = data_start;
            for (int samp = 0; samp < 1000 && sp < fend; ++samp) {
                const char *nl = static_cast<const char *>(
                    std::memchr(sp, '\n', fend - sp));
                if (!nl) nl = fend;
                parse_csv_row_fast(sp, static_cast<size_t>(nl - sp), row_buf);
                for (size_t c = 0; c < ncols && c < row_buf.size(); ++c) {
                    if (type_id[c] == 2) continue;
                    const std::string &v = row_buf[c];
                    if (v.empty()) continue;
                    if (type_id[c] == 0) {
                        int64_t x;
                        if (!csv_try_int64(v.c_str(), v.size(), x)) {
                            double d;
                            type_id[c] = csv_try_double(v.c_str(), v.size(), d) ? 1 : 2;
                        }
                    } else {
                        double d;
                        if (!csv_try_double(v.c_str(), v.size(), d)) type_id[c] = 2;
                    }
                }
                sp = nl + 1;
            }
        }

        // ── find index column ──────────────────────────────────────────────────
        size_t idx_col = ncols;
        for (size_t c = 0; c < ncols; ++c) {
            if (!index_col_name.empty() && headers[c] == index_col_name) {
                idx_col = c; break;
            }
        }

        // ── Step 3: chunk boundaries at newlines ───────────────────────────────
        const size_t nthreads = static_cast<size_t>(
            std::max(1u, std::thread::hardware_concurrency()));
        const size_t data_len = static_cast<size_t>(fend - data_start);
        const size_t chunk_size = (data_len + nthreads - 1) / nthreads;

        std::vector<const char *> chunk_starts(nthreads + 1);
        chunk_starts[0] = data_start;
        for (size_t t = 1; t < nthreads; ++t) {
            const char *split = data_start + t * chunk_size;
            if (split >= fend) { chunk_starts[t] = fend; continue; }
            const char *nl = static_cast<const char *>(
                std::memchr(split, '\n', fend - split));
            chunk_starts[t] = nl ? nl + 1 : fend;
        }
        chunk_starts[nthreads] = fend;

        // ── Step 4: parallel parse ─────────────────────────────────────────────
        struct ColBuf {
            int                      type_id{0};
            std::vector<int64_t>     ints;
            std::vector<double>      dbls;
            std::vector<std::string> strs;
        };
        struct ChunkResult {
            std::vector<ColBuf> cols;
            size_t nrows{0};
            explicit ChunkResult(size_t nc) : cols(nc) {}
            ChunkResult() = default;
        };

        std::vector<std::future<ChunkResult>> futures;
        futures.reserve(nthreads);
        for (size_t t = 0; t < nthreads; ++t) {
            const char *cs = chunk_starts[t];
            const char *ce = chunk_starts[t + 1];
            futures.push_back(std::async(std::launch::async,
                [cs, ce, ncols, &type_id]() -> ChunkResult {
                    ChunkResult r(ncols);
                    for (size_t c = 0; c < ncols; ++c)
                        r.cols[c].type_id = type_id[c];
                    // Pre-reserve: eliminates O(log N) resize passes for 250 K+ rows
                    // Each resize of a string vector moves all existing strings.
                    const size_t est = static_cast<size_t>(ce > cs ? ce - cs : 0) / 30 + 256;
                    for (size_t c = 0; c < ncols; ++c) {
                        if      (type_id[c] == 0) r.cols[c].ints.reserve(est);
                        else if (type_id[c] == 1) r.cols[c].dbls.reserve(est);
                        else                       r.cols[c].strs.reserve(est);
                    }
                    std::vector<std::string> row;
                    row.reserve(32);
                    const char *p = cs;
                    while (p < ce) {
                        const char *nl = static_cast<const char *>(
                            std::memchr(p, '\n', ce - p));
                        if (!nl) nl = ce;
                        size_t len = static_cast<size_t>(nl - p);
                        if (len > 0) {
                            parse_csv_row_fast(p, len, row);
                            ++r.nrows;
                            for (size_t c = 0; c < ncols && c < row.size(); ++c) {
                                switch (r.cols[c].type_id) {
                                    case 0: { int64_t x=0; csv_try_int64(row[c].c_str(),row[c].size(),x); r.cols[c].ints.push_back(x); break; }
                                    case 1: { double  x=0; csv_try_double(row[c].c_str(),row[c].size(),x); r.cols[c].dbls.push_back(x); break; }
                                    // Move from row buffer — parse_csv_row_fast clears row on
                                    // the next call, so the moved-from state is safe to destroy.
                                    case 2: r.cols[c].strs.push_back(std::move(row[c])); break;
                                }
                            }
                        }
                        p = nl + 1;
                    }
                    return r;
                }
            ));
        }

        // Collect results
        std::vector<ChunkResult> chunks;
        chunks.reserve(nthreads);
        size_t total_rows = 0;
        for (auto &f : futures) {
            chunks.push_back(f.get());
            total_rows += chunks.back().nrows;
        }

        // ── Step 5: merge per-chunk columns ───────────────────────────────────
        struct MergedCol {
            int                      type_id{0};
            std::vector<int64_t>     ints;
            std::vector<double>      dbls;
            std::vector<std::string> strs;
        };
        std::vector<MergedCol> merged(ncols);
        for (size_t c = 0; c < ncols; ++c) {
            merged[c].type_id = type_id[c];
            switch (type_id[c]) {
                case 0:
                    merged[c].ints.reserve(total_rows);
                    for (auto &ch : chunks)
                        merged[c].ints.insert(merged[c].ints.end(),
                            ch.cols[c].ints.begin(), ch.cols[c].ints.end());
                    break;
                case 1:
                    merged[c].dbls.reserve(total_rows);
                    for (auto &ch : chunks)
                        merged[c].dbls.insert(merged[c].dbls.end(),
                            ch.cols[c].dbls.begin(), ch.cols[c].dbls.end());
                    break;
                case 2:
                    merged[c].strs.reserve(total_rows);
                    for (auto &ch : chunks)
                        merged[c].strs.insert(merged[c].strs.end(),
                            std::make_move_iterator(ch.cols[c].strs.begin()),
                            std::make_move_iterator(ch.cols[c].strs.end()));
                    break;
            }
        }

        // ── assemble GrizzlarFrame ─────────────────────────────────────────────
        GrizzlarFrame out;
        std::vector<ulong> idx_vec;
        if (idx_col < ncols && merged[idx_col].type_id == 0) {
            idx_vec.reserve(total_rows);
            for (auto x : merged[idx_col].ints)
                idx_vec.push_back(static_cast<ulong>(x));
        } else {
            idx_vec.resize(total_rows);
            std::iota(idx_vec.begin(), idx_vec.end(), 0);
        }
        out.df_.load_index(std::move(idx_vec));

        for (size_t c = 0; c < ncols; ++c) {
            if (c == idx_col) continue;
            const std::string &nm = headers[c];
            out.col_order_.push_back(nm);
            switch (merged[c].type_id) {
                case 0:
                    out.col_types_[nm] = "int64";
                    out.df_.load_column<int64_t>(nm.c_str(), std::move(merged[c].ints));
                    break;
                case 1:
                    out.col_types_[nm] = "double";
                    out.df_.load_column<double>(nm.c_str(), std::move(merged[c].dbls));
                    break;
                case 2:
                    out.col_types_[nm] = "string";
                    out.df_.load_column<std::string>(nm.c_str(), std::move(merged[c].strs));
                    break;
            }
        }
        return out;
    }

    // Per-group aggregation for groupby_agg.
    // Computes directly over the source column (no intermediate vals vector).
    double aggregate_group(const std::string &col,
                           const std::vector<size_t> &indices,
                           const std::string &func) const {
        const size_t cnt = indices.size();
        if (func == "count") return static_cast<double>(cnt);
        if (cnt == 0) return 0.0;

        const std::string &type = col_types_.at(col);
        if (type != "double" && type != "int64")
            throw std::runtime_error("Cannot aggregate non-numeric column: " + col);

        auto get = [&](size_t i) -> double {
            if (type == "double") return df_.get_column<double>(col.c_str())[indices[i]];
            return static_cast<double>(df_.get_column<int64_t>(col.c_str())[indices[i]]);
        };

        if (func == "first") return get(0);
        if (func == "last")  return get(cnt - 1);

        double s = 0;
        double mn = get(0), mx = get(0);
        for (size_t i = 0; i < cnt; ++i) {
            double v = get(i);
            s += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        if (func == "sum")  return s;
        if (func == "mean") return s / static_cast<double>(cnt);
        if (func == "min")  return mn;
        if (func == "max")  return mx;
        if (func == "std") {
            double mean_v = s / static_cast<double>(cnt);
            double sq = 0;
            for (size_t i = 0; i < cnt; ++i) {
                double d = get(i) - mean_v;
                sq += d * d;
            }
            return cnt > 1 ? std::sqrt(sq / static_cast<double>(cnt - 1)) : 0.0;
        }
        throw std::runtime_error("Unknown aggregation function: " + func);
    }

    // Single-pass groupby — inspired by polars/vaex hash aggregation.
    // Maintains running statistics per group (no per-group index storage).
    // For N rows and G groups: O(N) time, O(G) memory (vs O(N) for index vectors).
    template<typename K>
    GrizzlarFrame do_groupby(const std::string &by_col,
                              const std::vector<K> &key_vec,
                              const std::vector<std::pair<std::string,std::string>> &specs) const {
        const size_t nspecs = specs.size();
        const size_t N = key_vec.size();

        // Pre-fetch column data pointers for direct access (avoids repeated map lookups)
        struct ColPtr {
            const double  *dbl{nullptr};
            const int64_t *i64{nullptr};
        };
        std::vector<ColPtr> col_ptrs(nspecs);
        for (size_t s = 0; s < nspecs; ++s) {
            const auto &col = specs[s].first;
            const std::string &type = col_types_.at(col);
            if (type == "double")
                col_ptrs[s].dbl = df_.get_column<double>(col.c_str()).data();
            else if (type == "int64")
                col_ptrs[s].i64 = df_.get_column<int64_t>(col.c_str()).data();
            else
                throw std::runtime_error("Cannot aggregate non-numeric column: " + col);
        }

        // Running-stats accumulator per group per spec — no index vectors stored
        struct RunState {
            double   sum{0}, min_v{1e300}, max_v{-1e300}, sum_sq{0};
            double   first_v{0}, last_v{0};
            int64_t  count{0};
            bool     initialized{false};
        };

        std::unordered_map<K, std::vector<RunState>> groups;
        groups.reserve(N / 8 + 16);

        for (size_t i = 0; i < N; ++i) {
            auto &states = groups[key_vec[i]];
            if (states.empty()) states.resize(nspecs);
            for (size_t s = 0; s < nspecs; ++s) {
                const double v = col_ptrs[s].dbl ? col_ptrs[s].dbl[i]
                                                 : static_cast<double>(col_ptrs[s].i64[i]);
                RunState &st = states[s];
                if (!st.initialized) {
                    st.first_v = v; st.min_v = v; st.max_v = v;
                    st.initialized = true;
                }
                st.count++;
                st.sum    += v;
                st.sum_sq += v * v;
                st.last_v  = v;
                if (v < st.min_v) st.min_v = v;
                if (v > st.max_v) st.max_v = v;
            }
        }

        // Deterministic output order
        std::vector<K> sorted_keys;
        sorted_keys.reserve(groups.size());
        for (auto &[k, _] : groups) sorted_keys.push_back(k);
        std::sort(sorted_keys.begin(), sorted_keys.end());

        const size_t ng = sorted_keys.size();
        std::vector<K> result_keys;
        result_keys.reserve(ng);
        std::vector<std::vector<double>> agg_results(nspecs);
        for (auto &v : agg_results) v.reserve(ng);

        for (const K &key : sorted_keys) {
            result_keys.push_back(key);
            const auto &states = groups[key];
            for (size_t s = 0; s < nspecs; ++s) {
                const RunState &st   = states[s];
                const auto     &func = specs[s].second;
                double res = 0;
                if      (func == "count") res = static_cast<double>(st.count);
                else if (func == "sum")   res = st.sum;
                else if (func == "mean")  res = st.sum / static_cast<double>(st.count);
                else if (func == "min")   res = st.min_v;
                else if (func == "max")   res = st.max_v;
                else if (func == "first") res = st.first_v;
                else if (func == "last")  res = st.last_v;
                else if (func == "std") {
                    // Welford/two-pass equivalent via E[X²] - E[X]²
                    double m  = st.sum / static_cast<double>(st.count);
                    double var = st.sum_sq / static_cast<double>(st.count) - m * m;
                    if (st.count > 1)
                        var = var * static_cast<double>(st.count) /
                              static_cast<double>(st.count - 1);
                    res = st.count > 0 ? std::sqrt(std::max(0.0, var)) : 0.0;
                } else {
                    throw std::runtime_error("Unknown aggregation function: " + func);
                }
                agg_results[s].push_back(res);
            }
        }

        GrizzlarFrame out;
        std::vector<ulong> new_idx(ng);
        std::iota(new_idx.begin(), new_idx.end(), 0);
        out.df_.load_index(std::move(new_idx));

        out.col_order_.push_back(by_col);
        if constexpr (std::is_same_v<K, std::string_view>) {
            // string_view keys — convert to std::string for hmdf storage
            out.col_types_[by_col] = "string";
            std::vector<std::string> str_keys;
            str_keys.reserve(result_keys.size());
            for (auto sv : result_keys) str_keys.emplace_back(sv);
            out.df_.load_column<std::string>(by_col.c_str(), std::move(str_keys));
        } else {
            out.col_types_[by_col] = col_types_.at(by_col);
            out.df_.load_column<K>(by_col.c_str(), std::move(result_keys));
        }

        for (size_t s = 0; s < nspecs; ++s) {
            const auto &col = specs[s].first;
            out.col_order_.push_back(col);
            out.col_types_[col] = "double";
            out.df_.load_column<double>(col.c_str(), std::move(agg_results[s]));
        }
        return out;
    }

    // ── require helpers ──────────────────────────────────────────────────────

    std::string require_numeric(const std::string &col) const {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        if (it->second != "double" && it->second != "int64")
            throw std::runtime_error(
                "Column '" + col + "' is not numeric (type: " + it->second + ")");
        return it->second;
    }

    // ── loading ──────────────────────────────────────────────────────────────

    void load_index(py::object indices) {
        auto vec = to_vec<ulong>(indices);
        df_.load_index(std::move(vec));
    }

    void load_column(const std::string &name, py::object data) {
        std::string type = detect_type(data);
        if (col_types_.find(name) == col_types_.end())
            col_order_.push_back(name);
        col_types_[name] = type;

        if (type == "double")
            df_.load_column<double>(name.c_str(), to_vec<double>(data));
        else if (type == "int64")
            df_.load_column<int64_t>(name.c_str(), to_vec<int64_t>(data));
        else if (type == "bool")
            df_.load_column<bool>(name.c_str(), to_vec<bool>(data));
        else
            df_.load_column<std::string>(name.c_str(), to_str_vec(data));
    }

    // ── accessors ────────────────────────────────────────────────────────────

    py::array_t<ulong> get_index() const {
        const auto &vec = df_.get_index();
        py::array_t<ulong> result(static_cast<py::ssize_t>(vec.size()));
        auto buf = result.request();
        std::copy(vec.begin(), vec.end(), static_cast<ulong*>(buf.ptr));
        return result;
    }

    py::object get_column(const std::string &name) const {
        auto it = col_types_.find(name);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + name);
        const std::string &type = it->second;
        if (type == "double") {
            const auto &vec = df_.get_column<double>(name.c_str());
            py::array_t<double> r(static_cast<py::ssize_t>(vec.size()));
            std::copy(vec.begin(), vec.end(), static_cast<double*>(r.request().ptr));
            return r;
        }
        if (type == "int64") {
            const auto &vec = df_.get_column<int64_t>(name.c_str());
            py::array_t<int64_t> r(static_cast<py::ssize_t>(vec.size()));
            std::copy(vec.begin(), vec.end(), static_cast<int64_t*>(r.request().ptr));
            return r;
        }
        if (type == "bool") {
            const auto &vec = df_.get_column<bool>(name.c_str());
            py::list lst;
            for (bool v : vec) lst.append(py::bool_(v));
            return lst;
        }
        const auto &vec = df_.get_column<std::string>(name.c_str());
        py::list lst;
        for (const auto &s : vec) lst.append(py::str(s));
        return lst;
    }

    std::vector<std::string> columns() const { return col_order_; }

    py::tuple shape() const {
        auto [r, c] = df_.shape();
        return py::make_tuple(r, c);
    }

    bool has_column(const std::string &name) const {
        return col_types_.find(name) != col_types_.end();
    }

    std::string col_type(const std::string &name) const {
        auto it = col_types_.find(name);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + name);
        return it->second;
    }

    // ── statistics ───────────────────────────────────────────────────────────

    double mean(const std::string &col) {
        auto type = require_numeric(col);
        if (type == "double") { MeanVisitor<double,ulong> v; df_.single_act_visit<double>(col.c_str(),v); return v.get_result(); }
        MeanVisitor<int64_t,ulong> v; df_.single_act_visit<int64_t>(col.c_str(),v); return static_cast<double>(v.get_result());
    }
    double std_dev(const std::string &col) {
        auto type = require_numeric(col);
        if (type == "double") { StdVisitor<double,ulong> v; df_.single_act_visit<double>(col.c_str(),v); return v.get_result(); }
        StdVisitor<int64_t,ulong> v; df_.single_act_visit<int64_t>(col.c_str(),v); return static_cast<double>(v.get_result());
    }
    double sum(const std::string &col) {
        auto type = require_numeric(col);
        if (type == "double") { SumVisitor<double,ulong> v; df_.single_act_visit<double>(col.c_str(),v); return v.get_result(); }
        SumVisitor<int64_t,ulong> v; df_.single_act_visit<int64_t>(col.c_str(),v); return static_cast<double>(v.get_result());
    }
    double col_min(const std::string &col) const {
        auto type = require_numeric(col);
        if (type == "double") { const auto &v=df_.get_column<double>(col.c_str()); if(v.empty()) throw std::runtime_error("Empty"); return *std::min_element(v.begin(),v.end()); }
        const auto &v=df_.get_column<int64_t>(col.c_str()); if(v.empty()) throw std::runtime_error("Empty"); return static_cast<double>(*std::min_element(v.begin(),v.end()));
    }
    double col_max(const std::string &col) const {
        auto type = require_numeric(col);
        if (type == "double") { const auto &v=df_.get_column<double>(col.c_str()); if(v.empty()) throw std::runtime_error("Empty"); return *std::max_element(v.begin(),v.end()); }
        const auto &v=df_.get_column<int64_t>(col.c_str()); if(v.empty()) throw std::runtime_error("Empty"); return static_cast<double>(*std::max_element(v.begin(),v.end()));
    }
    size_t count(const std::string &col) const {
        auto it = col_types_.find(col);
        if (it == col_types_.end()) throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        if (type=="double") return df_.get_column<double>(col.c_str()).size();
        if (type=="int64")  return df_.get_column<int64_t>(col.c_str()).size();
        if (type=="bool")   return df_.get_column<bool>(col.c_str()).size();
        return df_.get_column<std::string>(col.c_str()).size();
    }
    py::dict describe() {
        py::dict result;
        for (const auto &name : col_order_) {
            const std::string &type = col_types_.at(name);
            if (type != "double" && type != "int64") continue;
            py::dict stats;
            stats["count"] = count(name);
            stats["mean"]  = mean(name);
            stats["std"]   = std_dev(name);
            stats["min"]   = col_min(name);
            stats["max"]   = col_max(name);
            stats["sum"]   = sum(name);
            result[name.c_str()] = stats;
        }
        return result;
    }

    // ── quantile / correlation / covariance ──────────────────────────────────

    double quantile(const std::string &col, double q) const {
        require_numeric(col);
        if (q < 0.0 || q > 1.0) throw std::runtime_error("q must be in [0, 1]");
        std::vector<double> vals;
        const std::string &type = col_types_.at(col);
        if (type == "double") { const auto &v=df_.get_column<double>(col.c_str()); vals.assign(v.begin(),v.end()); }
        else { const auto &v=df_.get_column<int64_t>(col.c_str()); for (auto x:v) vals.push_back(static_cast<double>(x)); }
        if (vals.empty()) throw std::runtime_error("Column is empty: " + col);
        std::sort(vals.begin(), vals.end());
        double pos = q * (vals.size() - 1);
        size_t lo  = static_cast<size_t>(pos);
        double frac = pos - lo;
        if (lo + 1 < vals.size()) return vals[lo] + frac * (vals[lo + 1] - vals[lo]);
        return vals[lo];
    }

    double corr(const std::string &col1, const std::string &col2) const {
        require_numeric(col1); require_numeric(col2);
        auto as_dbl = [this](const std::string &c) {
            std::vector<double> r;
            if (col_types_.at(c) == "double") { const auto &v=df_.get_column<double>(c.c_str()); r.assign(v.begin(),v.end()); }
            else { const auto &v=df_.get_column<int64_t>(c.c_str()); for (auto x:v) r.push_back(static_cast<double>(x)); }
            return r;
        };
        auto v1 = as_dbl(col1), v2 = as_dbl(col2);
        size_t n = std::min(v1.size(), v2.size());
        if (n == 0) return 0.0;
        double m1 = std::accumulate(v1.begin(),v1.begin()+n,0.0)/n;
        double m2 = std::accumulate(v2.begin(),v2.begin()+n,0.0)/n;
        double cov=0, var1=0, var2=0;
        for (size_t i=0;i<n;++i) { double d1=v1[i]-m1, d2=v2[i]-m2; cov+=d1*d2; var1+=d1*d1; var2+=d2*d2; }
        double denom = std::sqrt(var1*var2);
        return denom > 0 ? cov/denom : 0.0;
    }

    double cov(const std::string &col1, const std::string &col2) const {
        require_numeric(col1); require_numeric(col2);
        auto as_dbl = [this](const std::string &c) {
            std::vector<double> r;
            if (col_types_.at(c) == "double") { const auto &v=df_.get_column<double>(c.c_str()); r.assign(v.begin(),v.end()); }
            else { const auto &v=df_.get_column<int64_t>(c.c_str()); for (auto x:v) r.push_back(static_cast<double>(x)); }
            return r;
        };
        auto v1 = as_dbl(col1), v2 = as_dbl(col2);
        size_t n = std::min(v1.size(), v2.size());
        if (n < 2) return 0.0;
        double m1 = std::accumulate(v1.begin(),v1.begin()+n,0.0)/n;
        double m2 = std::accumulate(v2.begin(),v2.begin()+n,0.0)/n;
        double cov_val = 0;
        for (size_t i=0;i<n;++i) cov_val += (v1[i]-m1)*(v2[i]-m2);
        return cov_val / (n - 1);
    }

    // ── rolling / cumulative / shift ─────────────────────────────────────────

    py::array_t<double> rolling(const std::string &col, size_t window,
                                 const std::string &func) const {
        require_numeric(col);
        size_t n = df_.get_index().size();
        py::array_t<double> result(static_cast<py::ssize_t>(n));
        auto buf = result.request();
        double *ptr = static_cast<double*>(buf.ptr);
        double nan = std::numeric_limits<double>::quiet_NaN();
        for (size_t i=0;i<n;++i) ptr[i] = nan;
        if (window == 0 || window > n) return result;

        auto fill = [&](auto &vec) {
            // Build initial window sum
            double ws = 0;
            for (size_t i=0;i<window;++i) ws += static_cast<double>(vec[i]);

            auto reduce = [&](size_t end_i) -> double {
                if (func == "mean") return ws / window;
                if (func == "sum")  return ws;
                if (func == "std") {
                    double m = ws / window;
                    double sq = 0;
                    for (size_t j=end_i-window+1;j<=end_i;++j) sq += (static_cast<double>(vec[j])-m)*(static_cast<double>(vec[j])-m);
                    return window > 1 ? std::sqrt(sq/(window-1)) : 0.0;
                }
                if (func == "min") { double v=static_cast<double>(vec[end_i-window+1]); for(size_t j=end_i-window+2;j<=end_i;++j) v=std::min(v,static_cast<double>(vec[j])); return v; }
                if (func == "max") { double v=static_cast<double>(vec[end_i-window+1]); for(size_t j=end_i-window+2;j<=end_i;++j) v=std::max(v,static_cast<double>(vec[j])); return v; }
                return ws / window;
            };

            ptr[window-1] = reduce(window-1);
            for (size_t i=window;i<n;++i) {
                ws += static_cast<double>(vec[i]) - static_cast<double>(vec[i-window]);
                ptr[i] = reduce(i);
            }
        };

        const std::string &type = col_types_.at(col);
        if (type == "double") { const auto &v=df_.get_column<double>(col.c_str()); fill(v); }
        else                  { const auto &v=df_.get_column<int64_t>(col.c_str()); fill(v); }
        return result;
    }

    py::array_t<double> cumulative(const std::string &col,
                                    const std::string &func) const {
        require_numeric(col);
        size_t n = df_.get_index().size();
        py::array_t<double> result(static_cast<py::ssize_t>(n));
        auto buf = result.request();
        double *ptr = static_cast<double*>(buf.ptr);

        auto fill = [&](auto &vec) {
            double running = (func=="prod") ? 1.0 : 0.0;
            for (size_t i=0;i<n;++i) {
                double x = static_cast<double>(vec[i]);
                if (func=="sum")  running += x;
                else if (func=="prod") running *= x;
                else if (func=="min")  running = (i==0) ? x : std::min(running,x);
                else if (func=="max")  running = (i==0) ? x : std::max(running,x);
                ptr[i] = running;
            }
        };
        const std::string &type = col_types_.at(col);
        if (type == "double") { const auto &v=df_.get_column<double>(col.c_str()); fill(v); }
        else                  { const auto &v=df_.get_column<int64_t>(col.c_str()); fill(v); }
        return result;
    }

    py::array_t<double> shift_col(const std::string &col, int n) const {
        auto it = col_types_.find(col);
        if (it == col_types_.end()) throw std::runtime_error("Column not found: " + col);
        size_t sz = df_.get_index().size();
        py::array_t<double> result(static_cast<py::ssize_t>(sz));
        auto buf = result.request();
        double *ptr = static_cast<double*>(buf.ptr);
        double nan = std::numeric_limits<double>::quiet_NaN();

        auto fill = [&](auto &vec) {
            if (n >= 0) {
                size_t sh = static_cast<size_t>(n);
                for (size_t i=0;i<std::min(sh,sz);++i) ptr[i] = nan;
                for (size_t i=sh;i<sz;++i) ptr[i] = static_cast<double>(vec[i-sh]);
            } else {
                size_t sh = static_cast<size_t>(-n);
                for (size_t i=0;i+sh<sz;++i) ptr[i] = static_cast<double>(vec[i+sh]);
                for (size_t i=(sz>sh ? sz-sh : 0);i<sz;++i) ptr[i] = nan;
            }
        };
        const std::string &type = it->second;
        if (type == "double") { const auto &v=df_.get_column<double>(col.c_str()); fill(v); }
        else if (type == "int64") { const auto &v=df_.get_column<int64_t>(col.c_str()); fill(v); }
        else { for (size_t i=0;i<sz;++i) ptr[i]=nan; }
        return result;
    }

    py::array_t<double> pct_change(const std::string &col) const {
        require_numeric(col);
        size_t n = df_.get_index().size();
        py::array_t<double> result(static_cast<py::ssize_t>(n));
        auto buf = result.request();
        double *ptr = static_cast<double*>(buf.ptr);
        double nan = std::numeric_limits<double>::quiet_NaN();
        ptr[0] = nan;

        auto fill = [&](auto &vec) {
            for (size_t i=1;i<n;++i) {
                double prev = static_cast<double>(vec[i-1]);
                ptr[i] = (prev != 0) ? (static_cast<double>(vec[i]) - prev) / prev : nan;
            }
        };
        const std::string &type = col_types_.at(col);
        if (type == "double") { const auto &v=df_.get_column<double>(col.c_str()); fill(v); }
        else                  { const auto &v=df_.get_column<int64_t>(col.c_str()); fill(v); }
        return result;
    }

    // ── sorting ──────────────────────────────────────────────────────────────

    // Sort by building a permutation index with C++17 parallel sort, then
    // scatter all columns in parallel.  Returns a NEW frame — never mutates
    // this one.  Callers should NOT pre-copy; that doubled work was the old
    // bottleneck (two full scatter passes for string-heavy frames).
    //
    // String sort uses string_view keys so the comparator avoids the
    // std::string SSO + heap-pointer indirection on every comparison.
    GrizzlarFrame sort_by(const std::string &col, bool ascending = true) const {
        auto it = col_types_.find(col);
        if (it == col_types_.end()) throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        const size_t n = df_.get_index().size();

        std::vector<size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);

        if (type == "string") {
            const auto &raw = df_.get_column<std::string>(col.c_str());
            // Build a string_view vector: avoids SSO/heap indirection per comparison
            std::vector<std::string_view> keys(n);
            for (size_t i = 0; i < n; ++i) keys[i] = raw[i];
            if (ascending)
                std::sort(std::execution::par_unseq, perm.begin(), perm.end(),
                    [&](size_t a, size_t b) { return keys[a] < keys[b]; });
            else
                std::sort(std::execution::par_unseq, perm.begin(), perm.end(),
                    [&](size_t a, size_t b) { return keys[a] > keys[b]; });
        } else if (type == "int64") {
            const auto &keys = df_.get_column<int64_t>(col.c_str());
            if (ascending)
                std::sort(std::execution::par_unseq, perm.begin(), perm.end(),
                    [&](size_t a, size_t b) { return keys[a] < keys[b]; });
            else
                std::sort(std::execution::par_unseq, perm.begin(), perm.end(),
                    [&](size_t a, size_t b) { return keys[a] > keys[b]; });
        } else if (type == "double") {
            const auto &keys = df_.get_column<double>(col.c_str());
            if (ascending)
                std::sort(std::execution::par_unseq, perm.begin(), perm.end(),
                    [&](size_t a, size_t b) { return keys[a] < keys[b]; });
            else
                std::sort(std::execution::par_unseq, perm.begin(), perm.end(),
                    [&](size_t a, size_t b) { return keys[a] > keys[b]; });
        } else {
            throw std::runtime_error("sort_by: unsortable column type: " + type);
        }

        return extract_rows_parallel(perm);
    }

    GrizzlarFrame sort_index(bool ascending = true) const {
        const size_t n = df_.get_index().size();
        std::vector<size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        const auto &idx = df_.get_index();
        if (ascending)
            std::sort(std::execution::par_unseq, perm.begin(), perm.end(),
                [&](size_t a, size_t b) { return idx[a] < idx[b]; });
        else
            std::sort(std::execution::par_unseq, perm.begin(), perm.end(),
                [&](size_t a, size_t b) { return idx[a] > idx[b]; });
        return extract_rows_parallel(perm);
    }

    // ── filtering ────────────────────────────────────────────────────────────

    // Filter rows using a Python boolean mask (list[bool] or numpy bool array).
    // Direct compress: no intermediate index vector, sequential access (SIMD-friendly).
    // For frames with >= 50K output rows, processes columns in parallel threads.
    GrizzlarFrame filter_by_mask(py::object mask_obj) const {
        const auto &idx = df_.get_index();
        const size_t n = idx.size();

        // Get raw bool pointer — avoid copying to std::vector<bool>
        auto arr = py::cast<py::array_t<bool, py::array::c_style | py::array::forcecast>>(mask_obj);
        auto buf_info = arr.request();
        if (static_cast<size_t>(buf_info.size) != n)
            throw std::runtime_error("mask length " + std::to_string(buf_info.size) +
                                      " != frame length " + std::to_string(n));
        const bool *m = static_cast<const bool *>(buf_info.ptr);

        // Count output rows
        size_t out_n = 0;
        for (size_t i = 0; i < n; ++i) out_n += static_cast<size_t>(m[i]);

        if (out_n == n) return deep_copy();  // all pass — fast path

        GrizzlarFrame out;
        out.col_order_ = col_order_;
        out.col_types_ = col_types_;
        const size_t ncols = col_order_.size();

        // Pre-allocate all output column storage so threads can write in parallel
        struct ColOut {
            std::vector<int64_t>     ints;
            std::vector<double>      dbls;
            std::vector<bool>        bools;
            std::vector<std::string> strs;
        };
        std::vector<ulong>  new_idx(out_n);
        std::vector<ColOut> col_outs(ncols);
        for (size_t ci = 0; ci < ncols; ++ci) {
            const std::string &type = col_types_.at(col_order_[ci]);
            if      (type == "double") col_outs[ci].dbls.resize(out_n);
            else if (type == "int64")  col_outs[ci].ints.resize(out_n);
            else if (type == "bool")   col_outs[ci].bools.resize(out_n, false);
            else                       col_outs[ci].strs.reserve(out_n);
        }

        // compress_unit: 0 = index, 1..ncols = columns
        // Each unit's inner loop is a sequential compress (fast / SIMD-vectorisable)
        auto compress_unit = [&](size_t unit) {
            if (unit == 0) {
                ulong *dst = new_idx.data();
                const ulong *src = idx.data();
                for (size_t i = 0; i < n; ++i) if (m[i]) *dst++ = src[i];
            } else {
                const size_t ci = unit - 1;
                const std::string &cname = col_order_[ci];
                const std::string &type  = col_types_.at(cname);
                if (type == "double") {
                    const auto &src = df_.get_column<double>(cname.c_str());
                    double *dp = col_outs[ci].dbls.data();
                    for (size_t i = 0; i < src.size() && i < n; ++i)
                        if (m[i]) *dp++ = src[i];
                } else if (type == "int64") {
                    const auto &src = df_.get_column<int64_t>(cname.c_str());
                    int64_t *dp = col_outs[ci].ints.data();
                    for (size_t i = 0; i < src.size() && i < n; ++i)
                        if (m[i]) *dp++ = src[i];
                } else if (type == "bool") {
                    const auto &src = df_.get_column<bool>(cname.c_str());
                    size_t w = 0;
                    for (size_t i = 0; i < src.size() && i < n; ++i)
                        if (m[i]) col_outs[ci].bools[w++] = src[i];
                } else {
                    const auto &src = df_.get_column<std::string>(cname.c_str());
                    auto &dst_v = col_outs[ci].strs;
                    for (size_t i = 0; i < src.size() && i < n; ++i)
                        if (m[i]) dst_v.push_back(src[i]);
                }
            }
        };

        const size_t total_units = ncols + 1;
        // Use parallel threads only for large frames (thread overhead amortised)
        const size_t nthreads = (out_n >= 50000 && ncols >= 2)
            ? std::min(total_units,
                       static_cast<size_t>(std::thread::hardware_concurrency()))
            : 1;

        if (nthreads <= 1) {
            for (size_t u = 0; u < total_units; ++u) compress_unit(u);
        } else {
            const size_t upt = (total_units + nthreads - 1) / nthreads;
            std::vector<std::future<void>> futs;
            for (size_t t = 0; t < nthreads; ++t) {
                size_t ustart = t * upt;
                if (ustart >= total_units) break;
                size_t uend = std::min(ustart + upt, total_units);
                futs.push_back(std::async(std::launch::async,
                    [ustart, uend, &compress_unit]() {
                        for (size_t u = ustart; u < uend; ++u) compress_unit(u);
                    }));
            }
            for (auto &f : futs) f.wait();
        }

        // Load into output frame (sequential — load_column is not thread-safe)
        out.df_.load_index(std::move(new_idx));
        for (size_t ci = 0; ci < ncols; ++ci) {
            const std::string &cname = col_order_[ci];
            const std::string &type  = col_types_.at(cname);
            if      (type == "double") out.df_.load_column<double>(cname.c_str(),      std::move(col_outs[ci].dbls));
            else if (type == "int64")  out.df_.load_column<int64_t>(cname.c_str(),     std::move(col_outs[ci].ints));
            else if (type == "bool")   out.df_.load_column<bool>(cname.c_str(),        std::move(col_outs[ci].bools));
            else                       out.df_.load_column<std::string>(cname.c_str(), std::move(col_outs[ci].strs));
        }
        return out;
    }

    // Slice rows by integer position [start, stop).
    GrizzlarFrame iloc(long start, long stop) const {
        long n = static_cast<long>(df_.get_index().size());
        if (start < 0) start = std::max(0L, n + start);
        if (stop  < 0) stop  = std::max(0L, n + stop);
        start = std::min(start, n);
        stop  = std::min(stop,  n);
        std::vector<size_t> locs;
        for (long i=start;i<stop;++i) locs.push_back(static_cast<size_t>(i));
        return extract_rows(locs);
    }

    // Return a deep copy of this frame using the hmdf copy constructor.
    // Faster than extract_rows(all_rows) for full-frame copies (e.g. before sort).
    GrizzlarFrame deep_copy() const {
        GrizzlarFrame out;
        out.df_ = df_;
        out.col_types_ = col_types_;
        out.col_order_ = col_order_;
        return out;
    }

    // Return a new frame with only the requested columns (projection).
    GrizzlarFrame select_columns(const std::vector<std::string> &names) const {
        GrizzlarFrame out;
        const auto &src_idx = df_.get_index();
        std::vector<ulong> new_idx(src_idx.begin(), src_idx.end());
        out.df_.load_index(std::move(new_idx));

        for (const auto &name : names) {
            auto it = col_types_.find(name);
            if (it == col_types_.end())
                throw std::runtime_error("Column not found: " + name);
            out.col_order_.push_back(name);
            const std::string &type = it->second;
            out.col_types_[name] = type;
            if (type=="double")     out.df_.load_column<double>(name.c_str(),     df_.get_column<double>(name.c_str()));
            else if (type=="int64") out.df_.load_column<int64_t>(name.c_str(),    df_.get_column<int64_t>(name.c_str()));
            else if (type=="bool")  out.df_.load_column<bool>(name.c_str(),       df_.get_column<bool>(name.c_str()));
            else                    out.df_.load_column<std::string>(name.c_str(),df_.get_column<std::string>(name.c_str()));
        }
        return out;
    }

    // ── groupby ──────────────────────────────────────────────────────────────

    // specs: list of (agg_col, func) pairs.
    // Supported funcs: "mean","sum","min","max","count","std","first","last"
    GrizzlarFrame groupby_agg(const std::string &by_col,
                               const std::vector<std::pair<std::string,std::string>> &specs) const {
        auto it = col_types_.find(by_col);
        if (it == col_types_.end()) throw std::runtime_error("Column not found: " + by_col);
        for (const auto &[col, _] : specs) require_numeric(col);

        const std::string &by_type = it->second;
        if (by_type == "double") {
            const auto &v = df_.get_column<double>(by_col.c_str());
            return do_groupby<double>(by_col, {v.begin(),v.end()}, specs);
        } else if (by_type == "int64") {
            const auto &v = df_.get_column<int64_t>(by_col.c_str());
            return do_groupby<int64_t>(by_col, {v.begin(),v.end()}, specs);
        } else if (by_type == "string") {
            const auto &v = df_.get_column<std::string>(by_col.c_str());
            // string_view keys: avoids copying 2M strings into a new vector
            std::vector<std::string_view> key_views;
            key_views.reserve(v.size());
            for (const auto &s : v) key_views.emplace_back(s);
            return do_groupby<std::string_view>(by_col, key_views, specs);
        }
        throw std::runtime_error("Cannot group by column of type: " + by_type);
    }

    // ── join ─────────────────────────────────────────────────────────────────

    // Hash join two frames on their shared index.
    // Builds an unordered_map from the right index, probes with the left index,
    // then scatters columns in parallel — O(n+m) with no sort required.
    // how: "inner" | "left" | "right" | "outer"
    GrizzlarFrame join_by_index(const GrizzlarFrame &rhs, const std::string &how) const {
        const bool do_inner = (how == "inner");
        const bool do_left  = (how == "left");
        const bool do_right = (how == "right");
        const bool do_outer = (how == "outer");
        if (!do_inner && !do_left && !do_right && !do_outer)
            throw std::runtime_error("Unknown join type: " + how +
                                     " (use inner/left/right/outer)");

        const auto &li = df_.get_index();
        const auto &ri = rhs.df_.get_index();
        constexpr size_t NO_MATCH = std::numeric_limits<size_t>::max();

        // Build hash map: right index value → right row position
        std::unordered_map<ulong, size_t> right_map;
        right_map.reserve(ri.size() * 2);
        for (size_t j = 0; j < ri.size(); ++j)
            right_map.emplace(ri[j], j);

        // Probe: assemble left_pos[] and right_pos[] index arrays
        std::vector<size_t> left_pos, right_pos;

        if (do_inner || do_left) {
            left_pos.reserve(li.size());
            right_pos.reserve(li.size());
            for (size_t i = 0; i < li.size(); ++i) {
                auto it = right_map.find(li[i]);
                if (it != right_map.end()) {
                    left_pos.push_back(i);
                    right_pos.push_back(it->second);
                } else if (do_left) {
                    left_pos.push_back(i);
                    right_pos.push_back(NO_MATCH);
                }
            }
        } else if (do_right) {
            std::unordered_map<ulong, size_t> left_map;
            left_map.reserve(li.size() * 2);
            for (size_t i = 0; i < li.size(); ++i)
                left_map.emplace(li[i], i);
            left_pos.reserve(ri.size());
            right_pos.reserve(ri.size());
            for (size_t j = 0; j < ri.size(); ++j) {
                auto it = left_map.find(ri[j]);
                left_pos.push_back(it != left_map.end() ? it->second : NO_MATCH);
                right_pos.push_back(j);
            }
        } else { // outer
            std::vector<bool> right_matched(ri.size(), false);
            left_pos.reserve(li.size());
            right_pos.reserve(li.size());
            for (size_t i = 0; i < li.size(); ++i) {
                auto it = right_map.find(li[i]);
                if (it != right_map.end()) {
                    left_pos.push_back(i);
                    right_pos.push_back(it->second);
                    right_matched[it->second] = true;
                } else {
                    left_pos.push_back(i);
                    right_pos.push_back(NO_MATCH);
                }
            }
            for (size_t j = 0; j < ri.size(); ++j) {
                if (!right_matched[j]) {
                    left_pos.push_back(NO_MATCH);
                    right_pos.push_back(j);
                }
            }
        }

        const size_t n      = left_pos.size();
        const size_t nleft  = col_order_.size();
        const size_t nright = rhs.col_order_.size();
        const size_t total_units = 1 + nleft + nright;

        // Output column order and types: left then right
        GrizzlarFrame out;
        out.col_order_ = col_order_;
        out.col_types_ = col_types_;
        for (const auto &name : rhs.col_order_) {
            out.col_order_.push_back(name);
            out.col_types_[name] = rhs.col_types_.at(name);
        }

        // Pre-allocate output column buffers (default = null/zero/empty)
        struct ColBuf {
            std::vector<int64_t>     ints;
            std::vector<double>      dbls;
            std::vector<bool>        bools;
            std::vector<std::string> strs;
        };
        std::vector<ulong>  new_idx(n);
        std::vector<ColBuf> col_bufs(nleft + nright);

        auto alloc_buf = [&](size_t ci, const std::string &type) {
            if      (type == "double") col_bufs[ci].dbls.resize(n, std::numeric_limits<double>::quiet_NaN());
            else if (type == "int64")  col_bufs[ci].ints.resize(n, 0);
            else if (type == "bool")   col_bufs[ci].bools.resize(n, false);
            else                       col_bufs[ci].strs.resize(n);
        };
        for (size_t ci = 0; ci < nleft; ++ci)
            alloc_buf(ci, col_types_.at(col_order_[ci]));
        for (size_t ci = 0; ci < nright; ++ci)
            alloc_buf(nleft + ci, rhs.col_types_.at(rhs.col_order_[ci]));

        // Parallel scatter: each "unit" fills one column (or the index)
        auto scatter_unit = [&](size_t unit) {
            if (unit == 0) {
                // Index: left side wins; outer unmatched-right rows use right index
                for (size_t j = 0; j < n; ++j)
                    new_idx[j] = (left_pos[j] != NO_MATCH) ? li[left_pos[j]] : ri[right_pos[j]];
            } else if (unit <= nleft) {
                const size_t ci    = unit - 1;
                const auto  &cname = col_order_[ci];
                const auto  &type  = col_types_.at(cname);
                if (type == "double") {
                    const double *src = df_.get_column<double>(cname.c_str()).data();
                    double       *dst = col_bufs[ci].dbls.data();
                    for (size_t j = 0; j < n; ++j)
                        if (left_pos[j] != NO_MATCH) dst[j] = src[left_pos[j]];
                } else if (type == "int64") {
                    const int64_t *src = df_.get_column<int64_t>(cname.c_str()).data();
                    int64_t       *dst = col_bufs[ci].ints.data();
                    for (size_t j = 0; j < n; ++j)
                        if (left_pos[j] != NO_MATCH) dst[j] = src[left_pos[j]];
                } else if (type == "bool") {
                    const auto &sv = df_.get_column<bool>(cname.c_str());
                    auto       &dv = col_bufs[ci].bools;
                    for (size_t j = 0; j < n; ++j)
                        if (left_pos[j] != NO_MATCH) dv[j] = sv[left_pos[j]];
                } else {
                    const auto &sv = df_.get_column<std::string>(cname.c_str());
                    auto       &dv = col_bufs[ci].strs;
                    for (size_t j = 0; j < n; ++j)
                        if (left_pos[j] != NO_MATCH) dv[j] = sv[left_pos[j]];
                }
            } else {
                const size_t ci    = unit - 1 - nleft;
                const auto  &cname = rhs.col_order_[ci];
                const auto  &type  = rhs.col_types_.at(cname);
                if (type == "double") {
                    const double *src = rhs.df_.get_column<double>(cname.c_str()).data();
                    double       *dst = col_bufs[nleft + ci].dbls.data();
                    for (size_t j = 0; j < n; ++j)
                        if (right_pos[j] != NO_MATCH) dst[j] = src[right_pos[j]];
                } else if (type == "int64") {
                    const int64_t *src = rhs.df_.get_column<int64_t>(cname.c_str()).data();
                    int64_t       *dst = col_bufs[nleft + ci].ints.data();
                    for (size_t j = 0; j < n; ++j)
                        if (right_pos[j] != NO_MATCH) dst[j] = src[right_pos[j]];
                } else if (type == "bool") {
                    const auto &sv = rhs.df_.get_column<bool>(cname.c_str());
                    auto       &dv = col_bufs[nleft + ci].bools;
                    for (size_t j = 0; j < n; ++j)
                        if (right_pos[j] != NO_MATCH) dv[j] = sv[right_pos[j]];
                } else {
                    const auto &sv = rhs.df_.get_column<std::string>(cname.c_str());
                    auto       &dv = col_bufs[nleft + ci].strs;
                    for (size_t j = 0; j < n; ++j)
                        if (right_pos[j] != NO_MATCH) dv[j] = sv[right_pos[j]];
                }
            }
        };

        const size_t nthreads = (n >= 10000 && total_units >= 2)
            ? std::min(total_units, (size_t)std::thread::hardware_concurrency())
            : 1;

        if (nthreads <= 1) {
            for (size_t u = 0; u < total_units; ++u) scatter_unit(u);
        } else {
            const size_t upt = (total_units + nthreads - 1) / nthreads;
            std::vector<std::future<void>> futs;
            futs.reserve(nthreads);
            for (size_t t = 0; t < nthreads; ++t) {
                size_t us = t * upt, ue = std::min(us + upt, total_units);
                if (us >= total_units) break;
                futs.push_back(std::async(std::launch::async,
                    [us, ue, &scatter_unit]() {
                        for (size_t u = us; u < ue; ++u) scatter_unit(u);
                    }));
            }
            for (auto &f : futs) f.wait();
        }

        out.df_.load_index(std::move(new_idx));
        for (size_t ci = 0; ci < nleft + nright; ++ci) {
            const auto &cname = (ci < nleft) ? col_order_[ci] : rhs.col_order_[ci - nleft];
            const auto &type  = out.col_types_.at(cname);
            if      (type == "double")  out.df_.load_column<double>     (cname.c_str(), std::move(col_bufs[ci].dbls));
            else if (type == "int64")   out.df_.load_column<int64_t>    (cname.c_str(), std::move(col_bufs[ci].ints));
            else if (type == "bool")    out.df_.load_column<bool>       (cname.c_str(), std::move(col_bufs[ci].bools));
            else                        out.df_.load_column<std::string>(cname.c_str(), std::move(col_bufs[ci].strs));
        }
        return out;
    }

    // ── concat ───────────────────────────────────────────────────────────────

    // Vertically concatenate two frames (append rows). Columns present in
    // both frames with the same type are combined; others are dropped.
    // Index is reset to 0..N-1.
    GrizzlarFrame concat_frame(const GrizzlarFrame &other) const {
        GrizzlarFrame out;
        size_t n1 = df_.get_index().size();
        size_t n2 = other.df_.get_index().size();
        size_t total = n1 + n2;
        std::vector<ulong> new_idx(total);
        std::iota(new_idx.begin(), new_idx.end(), 0);
        out.df_.load_index(std::move(new_idx));

        for (const auto &name : col_order_) {
            auto o = other.col_types_.find(name);
            if (o == other.col_types_.end()) continue;
            const std::string &type = col_types_.at(name);
            if (o->second != type) continue;

            out.col_order_.push_back(name);
            out.col_types_[name] = type;

            if (type == "double") {
                const auto &a=df_.get_column<double>(name.c_str());
                const auto &b=other.df_.get_column<double>(name.c_str());
                std::vector<double> combined; combined.reserve(total);
                combined.insert(combined.end(),a.begin(),a.end());
                combined.insert(combined.end(),b.begin(),b.end());
                out.df_.load_column<double>(name.c_str(), std::move(combined));
            } else if (type == "int64") {
                const auto &a=df_.get_column<int64_t>(name.c_str());
                const auto &b=other.df_.get_column<int64_t>(name.c_str());
                std::vector<int64_t> combined; combined.reserve(total);
                combined.insert(combined.end(),a.begin(),a.end());
                combined.insert(combined.end(),b.begin(),b.end());
                out.df_.load_column<int64_t>(name.c_str(), std::move(combined));
            } else if (type == "bool") {
                const auto &a=df_.get_column<bool>(name.c_str());
                const auto &b=other.df_.get_column<bool>(name.c_str());
                std::vector<bool> combined; combined.reserve(total);
                combined.insert(combined.end(),a.begin(),a.end());
                combined.insert(combined.end(),b.begin(),b.end());
                out.df_.load_column<bool>(name.c_str(), std::move(combined));
            } else {
                const auto &a=df_.get_column<std::string>(name.c_str());
                const auto &b=other.df_.get_column<std::string>(name.c_str());
                std::vector<std::string> combined; combined.reserve(total);
                combined.insert(combined.end(),a.begin(),a.end());
                combined.insert(combined.end(),b.begin(),b.end());
                out.df_.load_column<std::string>(name.c_str(), std::move(combined));
            }
        }
        return out;
    }

    // ── data cleaning ────────────────────────────────────────────────────────

    // Return a new frame with duplicate rows removed (keep first occurrence).
    GrizzlarFrame drop_duplicates(const std::string &col) const {
        auto it = col_types_.find(col);
        if (it == col_types_.end()) throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        std::vector<size_t> keep;

        if (type == "double") {
            std::unordered_set<double> seen;
            const auto &v=df_.get_column<double>(col.c_str());
            for (size_t i=0;i<v.size();++i) if(seen.insert(v[i]).second) keep.push_back(i);
        } else if (type == "int64") {
            std::unordered_set<int64_t> seen;
            const auto &v=df_.get_column<int64_t>(col.c_str());
            for (size_t i=0;i<v.size();++i) if(seen.insert(v[i]).second) keep.push_back(i);
        } else if (type == "string") {
            std::unordered_set<std::string> seen;
            const auto &v=df_.get_column<std::string>(col.c_str());
            for (size_t i=0;i<v.size();++i) if(seen.insert(v[i]).second) keep.push_back(i);
        } else {
            bool st=false, sf=false;
            const auto &v=df_.get_column<bool>(col.c_str());
            for (size_t i=0;i<v.size();++i) {
                if ((v[i]&&!st)||(!v[i]&&!sf)) { keep.push_back(i); if(v[i]) st=true; else sf=true; }
            }
        }
        return extract_rows(keep);
    }

    // Remove rows where the given column has a NaN (double) or empty string.
    GrizzlarFrame drop_na(const std::string &col) const {
        auto it = col_types_.find(col);
        if (it == col_types_.end()) throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        std::vector<size_t> keep;

        if (type == "double") {
            const auto &v=df_.get_column<double>(col.c_str());
            for (size_t i=0;i<v.size();++i) if(!std::isnan(v[i])) keep.push_back(i);
        } else if (type == "string") {
            const auto &v=df_.get_column<std::string>(col.c_str());
            for (size_t i=0;i<v.size();++i) if(!v[i].empty()) keep.push_back(i);
        } else {
            // int64 / bool — no NaN concept, return as-is
            size_t n = df_.get_index().size();
            for (size_t i=0;i<n;++i) keep.push_back(i);
        }
        return extract_rows(keep);
    }

    // Fill NaN (double) or empty string in-place.
    void fillna(const std::string &col, py::object value) {
        auto it = col_types_.find(col);
        if (it == col_types_.end()) throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        if (type == "double") {
            double fill = py::cast<double>(value);
            auto &v = df_.get_column<double>(col.c_str());
            for (auto &x : v) if (std::isnan(x)) x = fill;
        } else if (type == "string") {
            std::string fill = py::cast<std::string>(value);
            auto &v = df_.get_column<std::string>(col.c_str());
            for (auto &x : v) if (x.empty()) x = fill;
        }
    }

    // Rename a column in-place.
    void rename_col(const std::string &old_name, const std::string &new_name) {
        auto it = col_types_.find(old_name);
        if (it == col_types_.end()) throw std::runtime_error("Column not found: " + old_name);
        if (col_types_.count(new_name)) throw std::runtime_error("Column already exists: " + new_name);
        df_.rename_column(old_name.c_str(), new_name.c_str());
        std::string type = it->second;
        col_types_.erase(it);
        col_types_[new_name] = type;
        for (auto &n : col_order_) if (n == old_name) { n = new_name; break; }
    }

    // Remove a column in-place.
    void drop_column(const std::string &name) {
        auto it = col_types_.find(name);
        if (it == col_types_.end()) throw std::runtime_error("Column not found: " + name);
        const std::string &type = it->second;
        if (type=="double")     df_.remove_column<double>(name.c_str());
        else if (type=="int64") df_.remove_column<int64_t>(name.c_str());
        else if (type=="bool")  df_.remove_column<bool>(name.c_str());
        else                    df_.remove_column<std::string>(name.c_str());
        col_types_.erase(it);
        col_order_.erase(std::remove(col_order_.begin(), col_order_.end(), name), col_order_.end());
    }

    // ── utilities ────────────────────────────────────────────────────────────

    // Frequency count of each unique value; returns a frame with "value","count" cols.
    GrizzlarFrame value_counts(const std::string &col) const {
        auto it = col_types_.find(col);
        if (it == col_types_.end()) throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;

        std::vector<std::string> keys;
        std::vector<int64_t> cnts;

        auto add_counts = [&](auto &vec) {
            std::map<std::string, int64_t> m;
            for (const auto &x : vec) {
                if constexpr (std::is_same_v<std::decay_t<decltype(x)>, std::string>)
                    m[x]++;
                else
                    m[std::to_string(x)]++;
            }
            for (auto &[k,cv] : m) { keys.push_back(k); cnts.push_back(cv); }
        };

        if (type == "double") { const auto &v=df_.get_column<double>(col.c_str()); add_counts(v); }
        else if (type == "int64") { const auto &v=df_.get_column<int64_t>(col.c_str()); add_counts(v); }
        else if (type == "string") { const auto &v=df_.get_column<std::string>(col.c_str()); add_counts(v); }
        else {
            const auto &v=df_.get_column<bool>(col.c_str());
            int64_t t=0,f=0; for(bool b:v){if(b)t++;else f++;}
            keys={"false","true"}; cnts={f,t};
        }

        // Sort by count descending
        std::vector<size_t> order(keys.size());
        std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return cnts[a]>cnts[b];});

        GrizzlarFrame out;
        std::vector<ulong> idx(keys.size()); std::iota(idx.begin(),idx.end(),0);
        out.df_.load_index(std::move(idx));
        std::vector<std::string> sk; sk.reserve(keys.size());
        std::vector<int64_t> sc; sc.reserve(cnts.size());
        for (size_t i : order) { sk.push_back(keys[i]); sc.push_back(cnts[i]); }
        out.col_order_ = {"value","count"};
        out.col_types_ = {{"value","string"},{"count","int64"}};
        out.df_.load_column<std::string>("value", std::move(sk));
        out.df_.load_column<int64_t>("count", std::move(sc));
        return out;
    }

    // Sorted unique values for a column.
    py::object unique_values(const std::string &col) const {
        auto it = col_types_.find(col);
        if (it == col_types_.end()) throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        if (type == "double") {
            const auto &v=df_.get_column<double>(col.c_str());
            std::set<double> s(v.begin(),v.end());
            py::array_t<double> r(static_cast<py::ssize_t>(s.size()));
            std::copy(s.begin(),s.end(),static_cast<double*>(r.request().ptr));
            return r;
        }
        if (type == "int64") {
            const auto &v=df_.get_column<int64_t>(col.c_str());
            std::set<int64_t> s(v.begin(),v.end());
            py::array_t<int64_t> r(static_cast<py::ssize_t>(s.size()));
            std::copy(s.begin(),s.end(),static_cast<int64_t*>(r.request().ptr));
            return r;
        }
        py::list lst;
        if (type == "string") {
            const auto &v=df_.get_column<std::string>(col.c_str());
            std::set<std::string> s(v.begin(),v.end());
            for (const auto &x:s) lst.append(py::str(x));
        } else {
            const auto &v=df_.get_column<bool>(col.c_str());
            bool ht=false,hf=false; for(bool b:v){if(b)ht=true;else hf=true;}
            if(hf) lst.append(py::bool_(false));
            if(ht) lst.append(py::bool_(true));
        }
        return lst;
    }

    size_t nunique(const std::string &col) const {
        auto it = col_types_.find(col);
        if (it == col_types_.end()) throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        if (type=="double") { const auto &v=df_.get_column<double>(col.c_str()); return std::set<double>(v.begin(),v.end()).size(); }
        if (type=="int64")  { const auto &v=df_.get_column<int64_t>(col.c_str()); return std::set<int64_t>(v.begin(),v.end()).size(); }
        if (type=="string") { const auto &v=df_.get_column<std::string>(col.c_str()); return std::set<std::string>(v.begin(),v.end()).size(); }
        const auto &v=df_.get_column<bool>(col.c_str());
        bool ht=false,hf=false; for(bool b:v){if(b)ht=true;else hf=true;}
        return static_cast<size_t>(ht) + static_cast<size_t>(hf);
    }

    size_t n_missing(const std::string &col) const {
        auto it = col_types_.find(col);
        if (it == col_types_.end()) throw std::runtime_error("Column not found: " + col);
        if (it->second == "double") {
            const auto &v=df_.get_column<double>(col.c_str());
            return static_cast<size_t>(std::count_if(v.begin(),v.end(),[](double x){return std::isnan(x);}));
        }
        if (it->second == "string") {
            const auto &v=df_.get_column<std::string>(col.c_str());
            return static_cast<size_t>(std::count_if(v.begin(),v.end(),[](const std::string &s){return s.empty();}));
        }
        return 0;
    }

    // ── I/O ──────────────────────────────────────────────────────────────────

    void to_csv(const std::string &path, bool write_index = true) const {
        std::ofstream out(path);
        if (!out) throw std::runtime_error("Cannot open for writing: " + path);
        bool first = true;
        if (write_index) { out << "index"; first = false; }
        for (const auto &c : col_order_) {
            if (!first) out << ',';
            out << c; first = false;
        }
        out << '\n';
        const auto &idx = df_.get_index();
        size_t nrows = idx.size();
        for (size_t i=0;i<nrows;++i) {
            first = true;
            if (write_index) { out << idx[i]; first = false; }
            for (const auto &c : col_order_) {
                if (!first) out << ',';
                write_cell(out, c, i); first = false;
            }
            out << '\n';
        }
    }

private:
    void write_cell(std::ofstream &out, const std::string &col, size_t row) const {
        const std::string &type = col_types_.at(col);
        if (type == "double") {
            const auto &v=df_.get_column<double>(col.c_str());
            if (row<v.size()) out<<v[row];
        } else if (type == "int64") {
            const auto &v=df_.get_column<int64_t>(col.c_str());
            if (row<v.size()) out<<v[row];
        } else if (type == "bool") {
            const auto &v=df_.get_column<bool>(col.c_str());
            if (row<v.size()) out<<(v[row]?"true":"false");
        } else {
            const auto &v=df_.get_column<std::string>(col.c_str());
            if (row<v.size()) {
                const std::string &s=v[row];
                bool nq = s.find(',')!=std::string::npos || s.find('"')!=std::string::npos || s.find('\n')!=std::string::npos;
                if (nq) { out<<'"'; for(char ch:s){if(ch=='"')out<<'"';out<<ch;} out<<'"'; }
                else out<<s;
            }
        }
    }
};

// ─── module ──────────────────────────────────────────────────────────────────

PYBIND11_MODULE(_grizzlars, m) {
    m.doc() = "Grizzlar: Python bindings for the hmdf C++ DataFrame library";

    py::class_<GrizzlarFrame>(m, "GrizzlarFrame")
        .def(py::init<>())
        // loading
        .def("load_index",  &GrizzlarFrame::load_index,  py::arg("indices"))
        .def("load_column", &GrizzlarFrame::load_column, py::arg("name"), py::arg("data"))
        // access
        .def("get_index",        &GrizzlarFrame::get_index)
        .def("get_column",       &GrizzlarFrame::get_column,      py::arg("name"))
        .def("columns",          &GrizzlarFrame::columns)
        .def("shape",            &GrizzlarFrame::shape)
        .def("col_type",         &GrizzlarFrame::col_type,         py::arg("name"))
        .def("has_column",       &GrizzlarFrame::has_column,       py::arg("name"))
        // statistics
        .def("mean",     &GrizzlarFrame::mean,     py::arg("col"))
        .def("std",      &GrizzlarFrame::std_dev,  py::arg("col"))
        .def("sum",      &GrizzlarFrame::sum,       py::arg("col"))
        .def("min",      &GrizzlarFrame::col_min,  py::arg("col"))
        .def("max",      &GrizzlarFrame::col_max,  py::arg("col"))
        .def("count",    &GrizzlarFrame::count,    py::arg("col"))
        .def("describe", &GrizzlarFrame::describe)
        // advanced stats
        .def("quantile",   &GrizzlarFrame::quantile,   py::arg("col"), py::arg("q"))
        .def("corr",       &GrizzlarFrame::corr,       py::arg("col1"), py::arg("col2"))
        .def("cov",        &GrizzlarFrame::cov,        py::arg("col1"), py::arg("col2"))
        // time-series / window
        .def("rolling",    &GrizzlarFrame::rolling,    py::arg("col"), py::arg("window"), py::arg("func") = "mean")
        .def("cumulative", &GrizzlarFrame::cumulative, py::arg("col"), py::arg("func") = "sum")
        .def("shift_col",  &GrizzlarFrame::shift_col,  py::arg("col"), py::arg("n"))
        .def("pct_change", &GrizzlarFrame::pct_change, py::arg("col"))
        // sort
        .def("sort_by",    &GrizzlarFrame::sort_by,    py::arg("col"), py::arg("ascending") = true)
        .def("sort_index", &GrizzlarFrame::sort_index, py::arg("ascending") = true)
        // filter / copy
        .def("filter_by_mask",  &GrizzlarFrame::filter_by_mask,  py::arg("mask"))
        .def("deep_copy",       &GrizzlarFrame::deep_copy)
        .def("iloc",            &GrizzlarFrame::iloc,            py::arg("start"), py::arg("stop"))
        .def("select_columns",  &GrizzlarFrame::select_columns,  py::arg("names"))
        // groupby
        .def("groupby_agg", &GrizzlarFrame::groupby_agg, py::arg("by_col"), py::arg("specs"))
        // join / concat
        .def("join_by_index", &GrizzlarFrame::join_by_index, py::arg("rhs"), py::arg("how") = "inner")
        .def("concat_frame",  &GrizzlarFrame::concat_frame,  py::arg("other"))
        // data cleaning
        .def("drop_duplicates", &GrizzlarFrame::drop_duplicates, py::arg("col"))
        .def("drop_na",         &GrizzlarFrame::drop_na,         py::arg("col"))
        .def("fillna",          &GrizzlarFrame::fillna,          py::arg("col"), py::arg("value"))
        .def("rename_col",      &GrizzlarFrame::rename_col,      py::arg("old_name"), py::arg("new_name"))
        .def("drop_column",     &GrizzlarFrame::drop_column,     py::arg("name"))
        // utilities
        .def("value_counts",  &GrizzlarFrame::value_counts,  py::arg("col"))
        .def("unique_values", &GrizzlarFrame::unique_values,  py::arg("col"))
        .def("nunique",       &GrizzlarFrame::nunique,        py::arg("col"))
        .def("n_missing",     &GrizzlarFrame::n_missing,      py::arg("col"))
        // I/O
        .def("to_csv", &GrizzlarFrame::to_csv, py::arg("path"), py::arg("write_index") = true)
        // native C++ CSV loader (bypasses Python csv.DictReader for large files)
        .def_static("read_csv_native", &GrizzlarFrame::read_csv_native,
                    py::arg("path"), py::arg("index_col") = "");

    // Thread-pool controls
    m.def("set_thread_level",
          [](long n) { GDF::set_thread_level(n); },
          py::arg("n"),
          "Set the number of worker threads (0 = single-threaded).");
    m.def("set_optimum_thread_level",
          []() { GDF::set_optimum_thread_level(); },
          "Enable multithreading using all logical CPU cores.");
    m.def("get_thread_level",
          []() { return GDF::get_thread_level(); },
          "Return the current number of worker threads.");
}
