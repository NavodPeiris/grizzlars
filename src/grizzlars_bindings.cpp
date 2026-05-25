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
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#if defined(__has_include)
#if __has_include(<execution>)
#include <execution>
#if defined(__cpp_lib_execution)
#define GRIZZLAR_USE_EXECUTION 1
#endif
#endif
#endif

// Execution policy compatibility layer
#if defined(GRIZZLAR_USE_EXECUTION)
// par_unseq allows both parallelisation AND vectorisation (SIMD)
#ifdef __APPLE__
#define GRIZZLAR_EXEC_POLICY std::execution::seq
#else
#define GRIZZLAR_EXEC_POLICY std::execution::par_unseq
#endif
#define GRIZZLAR_SORT(policy, ...) std::sort(policy, __VA_ARGS__)
#else
#define GRIZZLAR_SORT(policy, ...) std::sort(__VA_ARGS__)
#endif
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
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace py = pybind11;
using namespace hmdf;

using ulong = unsigned long;
using GDF = StdDataFrame<ulong>;

// ─── type detection ──────────────────────────────────────────────────────────

static std::string detect_type(py::object data)
{
    if (py::isinstance<py::array>(data))
    {
        char kind = py::cast<py::array>(data).dtype().kind();
        if (kind == 'f')
            return "double";
        if (kind == 'i' || kind == 'u')
            return "int64";
        if (kind == 'b')
            return "bool";
        return "string";
    }
    if (py::isinstance<py::list>(data))
    {
        py::list lst = py::cast<py::list>(data);
        if (lst.empty())
            return "double";
        py::object first = lst[0];
        if (py::isinstance<py::bool_>(first))
            return "bool";
        if (py::isinstance<py::float_>(first))
            return "double";
        if (py::isinstance<py::int_>(first))
            return "int64";
        if (py::isinstance<py::str>(first))
            return "string";
    }
    return "double";
}

// ─── conversion helpers ──────────────────────────────────────────────────────

template <typename T>
static std::vector<T> to_vec(py::object obj)
{
    if (py::isinstance<py::array>(obj))
    {
        auto arr = py::cast<
            py::array_t<T, py::array::c_style | py::array::forcecast>>(obj);
        auto buf = arr.request();
        auto *ptr = static_cast<T *>(buf.ptr);
        return std::vector<T>(ptr, ptr + buf.size);
    }
    std::vector<T> result;
    for (auto item : py::cast<py::list>(obj))
        result.push_back(py::cast<T>(item));
    return result;
}

static std::vector<std::string> to_str_vec(py::object obj)
{
    std::vector<std::string> result;
    if (py::isinstance<py::array>(obj))
    {
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
static std::string strip_hmdf_annotation(const std::string &s)
{
    if (s.size() < 5 || s.back() != '>')
        return s;
    auto lt = s.rfind('<');
    if (lt == std::string::npos || lt < 2 || s[lt - 1] != ':')
        return s;
    size_t ed = lt - 2;
    if (!std::isdigit((unsigned char)s[ed]))
        return s;
    size_t sd = ed;
    while (sd > 0 && std::isdigit((unsigned char)s[sd - 1]))
        --sd;
    if (sd == 0 || s[sd - 1] != ':')
        return s;
    return s.substr(0, sd - 1);
}

// RFC-4180-compliant CSV row parser.
// Fast path: unquoted fields are assigned directly from a pointer range
// (one emplace_back + one memcpy) instead of N char-by-char appends.
// Slow path: quoted fields handle escaped double-quotes correctly.
static void parse_csv_row_fast(const char *p, size_t len,
                               std::vector<std::string> &fields)
{
    fields.clear();
    const char *end = p + len;
    if (end > p && *(end - 1) == '\r')
        --end;
    if (p == end)
        return;

    for (;;)
    {
        if (p >= end)
        {
            // trailing comma → empty last field
            fields.emplace_back();
            break;
        }
        if (*p == '"')
        {
            // Quoted field: slow path handles embedded commas and "" escapes
            ++p;
            std::string f;
            while (p < end)
            {
                char c = *p++;
                if (c == '"')
                {
                    if (p < end && *p == '"')
                    {
                        f += '"';
                        ++p;
                    }
                    else
                        break;
                }
                else
                {
                    f += c;
                }
            }
            fields.push_back(std::move(f));
        }
        else
        {
            // Unquoted field: scan to next comma, assign whole range at once
            const char *fs = p;
            while (p < end && *p != ',')
                ++p;
            fields.emplace_back(fs, static_cast<size_t>(p - fs));
        }
        if (p >= end)
            break;
        ++p; // skip comma separator
    }
}

// ─── memory-mapped file helpers ───────────────────────────────────────────────

struct MmapView
{
    const char *data{nullptr};
    size_t size{0};
    void *handle{nullptr}; // platform opaque handle
};

// Returns a read-only memory-mapped view of the file.
// Falls back gracefully if mmap fails (view.data == nullptr).
static MmapView mmap_open(const std::string &path)
{
    MmapView v;
#ifdef _WIN32
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
                               nullptr);
    if (hFile == INVALID_HANDLE_VALUE)
        return v;
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(hFile, &sz))
    {
        CloseHandle(hFile);
        return v;
    }
    v.size = static_cast<size_t>(sz.QuadPart);
    if (v.size == 0)
    {
        CloseHandle(hFile);
        return v;
    }
    HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(hFile);
    if (!hMap)
        return v;
    v.data = static_cast<const char *>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
    v.handle = hMap;
    if (!v.data)
    {
        CloseHandle(hMap);
        v.handle = nullptr;
    }
#else
    // POSIX mmap
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return v;
    struct stat st{};
    if (::fstat(fd, &st) != 0)
    {
        ::close(fd);
        return v;
    }
    v.size = static_cast<size_t>(st.st_size);
    if (v.size == 0)
    {
        ::close(fd);
        return v;
    }
    void *ptr = ::mmap(nullptr, v.size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (ptr == MAP_FAILED)
    {
        v.size = 0;
        return v;
    }
    v.data = static_cast<const char *>(ptr);
    v.handle = ptr;
#endif
    return v;
}

static void mmap_close(MmapView &v)
{
    if (!v.data)
        return;
#ifdef _WIN32
    UnmapViewOfFile(v.data);
    if (v.handle)
        CloseHandle(v.handle);
#else
    ::munmap(v.handle, v.size);
#endif
    v = MmapView{};
}

static bool csv_try_int64(const char *s, size_t len, int64_t &out)
{
    if (!len) return false;
    auto [end, ec] = std::from_chars(s, s + len, out);
    return ec == std::errc{} && end == s + len;
}
static bool csv_try_double(const char *s, size_t len, double &out)
{
    if (!len) return false;
#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION < 200000
    // Apple libc++ before LLVM 20 (Xcode ≤ 15) lacks floating-point from_chars
    char buf[64];
    if (len < sizeof(buf)) {
        std::memcpy(buf, s, len);
        buf[len] = '\0';
        char *e;
        out = std::strtod(buf, &e);
        return (size_t)(e - buf) == len;
    }
    std::string tmp(s, len);
    char *e;
    out = std::strtod(tmp.c_str(), &e);
    return (size_t)(e - tmp.c_str()) == len;
#else
    auto [end, ec] = std::from_chars(s, s + len, out);
    return ec == std::errc{} && end == s + len;
#endif
}

static bool is_na_raw(const char *s, size_t len)
{
    switch (len)
    {
    case 0: return true;
    case 2: return s[0]=='N' && s[1]=='A';
    case 3: return (s[0]=='N' && s[1]=='/' && s[2]=='A')
                || (s[0]=='n' && s[1]=='a' && s[2]=='n')
                || (s[0]=='N' && s[1]=='a' && s[2]=='N');
    case 4: return std::memcmp(s,"null",4)==0 || std::memcmp(s,"NULL",4)==0
                || std::memcmp(s,"None",4)==0;
    default: return false;
    }
}

// ─── StringArray ─────────────────────────────────────────────────────────────
// Compact flat-buffer string storage (Arrow-style).
// All string bytes live in one contiguous `data` vector.
// `offsets[i]` .. `offsets[i+1]` is the byte range for string i.
//
// Benefits vs std::vector<std::string>:
//   filter (compress): one memcpy per contiguous block of matching rows
//   sort   (gather):   one memcpy per output string, reusing one large buffer
//   memory: one big allocation instead of n individual heap allocations
struct StringArray {
    std::vector<char>     data;
    std::vector<uint32_t> offsets;  // n+1 entries

    StringArray() { offsets.push_back(0); }

    size_t size()  const { return offsets.size() - 1; }
    bool   empty() const { return offsets.size() <= 1; }

    std::string_view view(size_t i) const {
        return {data.data() + offsets[i], offsets[i+1] - offsets[i]};
    }
    std::string str(size_t i) const { return std::string(view(i)); }

    void push_back(const char *s, size_t len) {
        data.insert(data.end(), s, s + len);
        offsets.push_back(static_cast<uint32_t>(data.size()));
    }
    void push_back(std::string_view sv)      { push_back(sv.data(), sv.size()); }
    void push_back(const std::string &s)     { push_back(s.data(),  s.size());  }

    static StringArray from_strvec(const std::vector<std::string> &strs) {
        StringArray sa;
        size_t total = 0;
        for (auto &s : strs) total += s.size();
        sa.data.reserve(total);
        sa.offsets.reserve(strs.size() + 1);
        for (auto &s : strs) sa.push_back(s.data(), s.size());
        return sa;
    }
    static StringArray from_strvec(std::vector<std::string> &&strs) {
        return from_strvec(static_cast<const std::vector<std::string>&>(strs));
    }

    static StringArray from_py_list(const py::list &lst) {
        StringArray sa;
        sa.offsets.reserve(static_cast<size_t>(lst.size()) + 1);
        for (auto item : lst) {
            if (py::isinstance<py::none>(item)) { sa.push_back("", 0); continue; }
            auto s = py::cast<std::string>(item);
            sa.push_back(s.data(), s.size());
        }
        return sa;
    }

    py::list to_py_list() const {
        py::list lst;
        size_t n = size();
        for (size_t i = 0; i < n; ++i)
            lst.append(py::str(data.data() + offsets[i], offsets[i+1] - offsets[i]));
        return lst;
    }

    std::vector<std::string> to_strvec() const {
        size_t n = size();
        std::vector<std::string> r(n);
        for (size_t i = 0; i < n; ++i)
            r[i].assign(data.data() + offsets[i], offsets[i+1] - offsets[i]);
        return r;
    }

    // Compress: keep rows where mask[i] == true.
    // Copies contiguous blocks via memcpy — one allocation for the output buffer.
    StringArray compress(const uint8_t *mask, size_t n_rows) const {
        StringArray sa;
        uint32_t out_bytes = 0;
        size_t   out_n    = 0;
        for (size_t i = 0; i < n_rows; ++i) {
            if (mask[i]) { out_bytes += offsets[i+1] - offsets[i]; ++out_n; }
        }
        sa.data.reserve(out_bytes);
        sa.offsets.reserve(out_n + 1);
        size_t i = 0;
        while (i < n_rows) {
            if (!mask[i]) { ++i; continue; }
            size_t blk = i;
            while (i < n_rows && mask[i]) ++i;
            const char *src  = data.data() + offsets[blk];
            uint32_t    nb   = offsets[i] - offsets[blk];
            uint32_t    base = static_cast<uint32_t>(sa.data.size());
            sa.data.insert(sa.data.end(), src, src + nb);
            for (size_t j = blk; j < i; ++j)
                sa.offsets.push_back(base + offsets[j+1] - offsets[blk]);
        }
        return sa;
    }

    // Gather: reorder rows by permutation locs[0..n_out).
    StringArray gather(const size_t *locs, size_t n_out) const {
        StringArray sa;
        // When gathering all rows (sort), output size == input size — skip estimation pass.
        uint32_t est;
        if (n_out == size()) {
            est = static_cast<uint32_t>(data.size());
        } else {
            est = 0;
            for (size_t j = 0; j < n_out; ++j)
                est += offsets[locs[j]+1] - offsets[locs[j]];
        }
        sa.data.reserve(est);
        sa.offsets.reserve(n_out + 1);
        constexpr size_t PF = 8;
        for (size_t j = 0; j < n_out; ++j) {
            if (j + PF < n_out)
                HMDF_PREFETCH_R(data.data() + offsets[locs[j + PF]]);
            const char *s  = data.data() + offsets[locs[j]];
            uint32_t    nb = offsets[locs[j]+1] - offsets[locs[j]];
            sa.data.insert(sa.data.end(), s, s + nb);
            sa.offsets.push_back(static_cast<uint32_t>(sa.data.size()));
        }
        return sa;
    }

    // Concat: append other after this.
    StringArray concat_with(const StringArray &other) const {
        StringArray sa;   // constructor already pushed offsets[0]=0
        size_t n1 = size(), n2 = other.size();
        sa.data.reserve(data.size() + other.data.size());
        sa.offsets.reserve(n1 + n2 + 1);
        sa.data = data;
        // Start from i=1: skip offsets[0] which the constructor already added
        for (size_t i = 1; i <= n1; ++i) sa.offsets.push_back(offsets[i]);
        uint32_t base = static_cast<uint32_t>(data.size());
        sa.data.insert(sa.data.end(), other.data.begin(), other.data.end());
        for (size_t i = 1; i <= n2; ++i)
            sa.offsets.push_back(base + other.offsets[i]);
        return sa;
    }

    // Scatter for joins: positions[j] gives source row (NO_MATCH → empty string).
    StringArray scatter_join(const std::vector<size_t> &positions,
                              size_t NO_MATCH) const {
        StringArray sa;
        size_t n = positions.size();
        sa.offsets.reserve(n + 1);
        for (size_t j = 0; j < n; ++j) {
            if (positions[j] != NO_MATCH) {
                size_t r  = positions[j];
                const char *s  = data.data() + offsets[r];
                uint32_t    nb = offsets[r+1] - offsets[r];
                sa.data.insert(sa.data.end(), s, s + nb);
            }
            sa.offsets.push_back(static_cast<uint32_t>(sa.data.size()));
        }
        return sa;
    }

    // Rebuild with empty strings replaced by fill.
    StringArray with_fillna(std::string_view fill) const {
        StringArray sa;
        size_t n = size();
        sa.offsets.reserve(n + 1);
        for (size_t i = 0; i < n; ++i) {
            if (offsets[i+1] == offsets[i])
                sa.push_back(fill.data(), fill.size());
            else
                sa.push_back(data.data() + offsets[i], offsets[i+1] - offsets[i]);
        }
        return sa;
    }

    // Rebuild replacing strings according to a map.
    StringArray with_replace(const std::unordered_map<std::string, std::string> &m) const {
        StringArray sa;
        size_t n = size();
        sa.offsets.reserve(n + 1);
        for (size_t i = 0; i < n; ++i) {
            std::string key(data.data() + offsets[i], offsets[i+1] - offsets[i]);
            auto it = m.find(key);
            if (it != m.end()) sa.push_back(it->second.data(), it->second.size());
            else                sa.push_back(data.data() + offsets[i], offsets[i+1] - offsets[i]);
        }
        return sa;
    }
};

// ─── GrizzlarFrame ────────────────────────────────────────────────────────────

class GrizzlarFrame
{
public:
    GDF df_;
    std::unordered_map<std::string, std::string> col_types_;
    std::vector<std::string> col_order_;
    std::unordered_map<std::string, StringArray> str_cols_; // flat string storage

    // ── private helpers ──────────────────────────────────────────────────────

    // Build a new GrizzlarFrame containing only the rows at positions `locs`.
    // Uses resize+pointer-write (avoids push_back overhead) and prefetches
    // the next gather address to hide random-access latency.
    GrizzlarFrame extract_rows(const std::vector<size_t> &locs) const
    {
        GrizzlarFrame out;
        const size_t n = locs.size();
        const auto &src_idx = df_.get_index();

        // ── index ──────────────────────────────────────────────────────────────
        std::vector<ulong> new_idx(n);
        {
            ulong *dst = new_idx.data();
            const ulong *src = src_idx.data();
            for (size_t j = 0; j < n; ++j)
            {
                if (j + HMDF_PF_DIST < n)
                    HMDF_PREFETCH_R(src + locs[j + HMDF_PF_DIST]);
                dst[j] = src[locs[j]];
            }
        }
        out.df_.load_index(std::move(new_idx));

        // ── data columns ───────────────────────────────────────────────────────
        for (const auto &name : col_order_)
        {
            out.col_order_.push_back(name);
            const std::string &type = col_types_.at(name);
            out.col_types_[name] = type;

            if (type == "double")
            {
                const auto &v = df_.get_column<double>(name.c_str());
                std::vector<double> nv(n);
                double *dst = nv.data();
                const double *src = v.data();
                for (size_t j = 0; j < n; ++j)
                {
                    if (j + HMDF_PF_DIST < n)
                        HMDF_PREFETCH_R(src + locs[j + HMDF_PF_DIST]);
                    dst[j] = src[locs[j]];
                }
                out.df_.load_column<double>(name.c_str(), std::move(nv));
            }
            else if (type == "int64")
            {
                const auto &v = df_.get_column<int64_t>(name.c_str());
                std::vector<int64_t> nv(n);
                int64_t *dst = nv.data();
                const int64_t *src = v.data();
                for (size_t j = 0; j < n; ++j)
                {
                    if (j + HMDF_PF_DIST < n)
                        HMDF_PREFETCH_R(src + locs[j + HMDF_PF_DIST]);
                    dst[j] = src[locs[j]];
                }
                out.df_.load_column<int64_t>(name.c_str(), std::move(nv));
            }
            else if (type == "bool")
            {
                const auto &v = df_.get_column<bool>(name.c_str());
                std::vector<bool> nv(n);
                for (size_t j = 0; j < n; ++j)
                    nv[j] = v[locs[j]];
                out.df_.load_column<bool>(name.c_str(), std::move(nv));
            }
            else
            {
                out.str_cols_[name] = str_cols_.at(name).gather(locs.data(), n);
            }
        }
        return out;
    }

    // Parallel scatter: apply a row permutation/index list to all columns
    // simultaneously.  String columns use StringArray::gather (one buffer alloc)
    // instead of per-row std::string copies.
    GrizzlarFrame extract_rows_parallel(const std::vector<size_t> &locs) const
    {
        const size_t n_out = locs.size();
        const size_t ncols = col_order_.size();

        GrizzlarFrame out;
        out.col_order_ = col_order_;
        out.col_types_ = col_types_;

        // Separate string columns so they use StringArray::gather
        std::vector<size_t> str_ci;
        str_ci.reserve(ncols);
        for (size_t ci = 0; ci < ncols; ++ci)
            if (col_types_.at(col_order_[ci]) == "string") str_ci.push_back(ci);

        struct ColOut { std::vector<int64_t> ints; std::vector<double> dbls; std::vector<bool> bools; };
        std::vector<ulong> new_idx(n_out);
        std::vector<ColOut> col_outs(ncols);
        std::vector<StringArray> str_outs(str_ci.size());

        for (size_t ci = 0; ci < ncols; ++ci)
        {
            const std::string &type = col_types_.at(col_order_[ci]);
            if (type == "double")     col_outs[ci].dbls.resize(n_out);
            else if (type == "int64") col_outs[ci].ints.resize(n_out);
            else if (type == "bool")  col_outs[ci].bools.resize(n_out, false);
        }

        // gather_unit: 0=index, 1..ncols = numeric/bool columns only
        auto gather_unit = [&](size_t unit)
        {
            if (unit == 0)
            {
                const ulong *src = df_.get_index().data();
                for (size_t j = 0; j < n_out; ++j)
                {
                    if (j + HMDF_PF_DIST < n_out)
                        HMDF_PREFETCH_R(src + locs[j + HMDF_PF_DIST]);
                    new_idx[j] = src[locs[j]];
                }
            }
            else
            {
                const size_t ci = unit - 1;
                const auto &cname = col_order_[ci];
                const auto &type = col_types_.at(cname);
                if (type == "double")
                {
                    const double *src = df_.get_column<double>(cname.c_str()).data();
                    double *dst = col_outs[ci].dbls.data();
                    for (size_t j = 0; j < n_out; ++j)
                    {
                        if (j + HMDF_PF_DIST < n_out)
                            HMDF_PREFETCH_R(src + locs[j + HMDF_PF_DIST]);
                        dst[j] = src[locs[j]];
                    }
                }
                else if (type == "int64")
                {
                    const int64_t *src = df_.get_column<int64_t>(cname.c_str()).data();
                    int64_t *dst = col_outs[ci].ints.data();
                    for (size_t j = 0; j < n_out; ++j)
                    {
                        if (j + HMDF_PF_DIST < n_out)
                            HMDF_PREFETCH_R(src + locs[j + HMDF_PF_DIST]);
                        dst[j] = src[locs[j]];
                    }
                }
                else if (type == "bool")
                {
                    const auto &sv = df_.get_column<bool>(cname.c_str());
                    for (size_t j = 0; j < n_out; ++j)
                        col_outs[ci].bools[j] = sv[locs[j]];
                }
                // string columns handled in str_outs below
            }
        };

        const size_t total_units = ncols + 1;
        const bool do_parallel = (n_out >= 50000 && ncols >= 2);

#if defined(GRIZZLAR_USE_EXECUTION)
        if (do_parallel)
        {
            const size_t n_tasks = total_units + str_ci.size();
            std::vector<size_t> all_tasks(n_tasks);
            std::iota(all_tasks.begin(), all_tasks.end(), 0);
            std::for_each(std::execution::par, all_tasks.begin(), all_tasks.end(),
                [&](size_t tid) {
                    if (tid < total_units)
                        gather_unit(tid);
                    else
                    {
                        const size_t si = tid - total_units;
                        str_outs[si] = str_cols_.at(col_order_[str_ci[si]]).gather(locs.data(), n_out);
                    }
                });
        }
        else
        {
            for (size_t u = 0; u < total_units; ++u) gather_unit(u);
            for (size_t si = 0; si < str_ci.size(); ++si)
                str_outs[si] = str_cols_.at(col_order_[str_ci[si]]).gather(locs.data(), n_out);
        }
#else
        for (size_t u = 0; u < total_units; ++u) gather_unit(u);
        for (size_t si = 0; si < str_ci.size(); ++si)
            str_outs[si] = str_cols_.at(col_order_[str_ci[si]]).gather(locs.data(), n_out);
#endif

        out.df_.load_index(std::move(new_idx));
        for (size_t ci = 0; ci < ncols; ++ci)
        {
            const auto &cname = col_order_[ci];
            const auto &type = col_types_.at(cname);
            if (type == "double")
                out.df_.load_column<double>(cname.c_str(), std::move(col_outs[ci].dbls));
            else if (type == "int64")
                out.df_.load_column<int64_t>(cname.c_str(), std::move(col_outs[ci].ints));
            else if (type == "bool")
                out.df_.load_column<bool>(cname.c_str(), std::move(col_outs[ci].bools));
        }
        for (size_t si = 0; si < str_ci.size(); ++si)
            out.str_cols_[col_order_[str_ci[si]]] = std::move(str_outs[si]);
        return out;
    }

    // Wrap an hmdf GDF result (from join/concat) back into a GrizzlarFrame,
    // rediscovering column names and types via get_columns_info.
    static GrizzlarFrame from_gdf(GDF &&gdf)
    {
        GrizzlarFrame out;
        out.df_ = std::move(gdf);
        auto info =
            out.df_.get_columns_info<double, int64_t, bool, std::string>();
        for (const auto &[raw_name, idx, tidx] : info)
        {
            std::string name(raw_name.c_str());
            out.col_order_.push_back(name);
            if (tidx == std::type_index(typeid(double)))
                out.col_types_[name] = "double";
            else if (tidx == std::type_index(typeid(int64_t)))
                out.col_types_[name] = "int64";
            else if (tidx == std::type_index(typeid(bool)))
                out.col_types_[name] = "bool";
            else
                out.col_types_[name] = "string";
        }
        return out;
    }

    // Returns true for string values that pandas treats as NaN by default.
    static bool is_na_string(const std::string &s)
    {
        // Matches pandas default na_values set (case-sensitive subset that matters most)
        static const std::unordered_set<std::string> na_tokens = {
            "", "None", "none", "nan", "NaN", "NA", "N/A", "n/a", "na",
            "NULL", "null", "Null", "#N/A", "#NA", "<NA>", "-NaN", "-nan",
            "1.#IND", "1.#QNAN", "-1.#IND", "-1.#QNAN", "#N/A N/A"
        };
        return na_tokens.count(s) > 0;
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
                                         const std::string &index_col_name)
    {
        // ── Step 1: map whole file (zero-copy on OS page cache) ───────────────
        MmapView mmap = mmap_open(path);
        // Fallback to file.read() if mmap failed
        std::string fbuf_fallback;
        const char *data;
        size_t fsz;
        if (mmap.data)
        {
            data = mmap.data;
            fsz = mmap.size;
        }
        else
        {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file)
                throw std::runtime_error("Cannot open CSV: " + path);
            const std::streamsize fsz2 = file.tellg();
            file.seekg(0);
            fbuf_fallback.assign(static_cast<size_t>(fsz2) + 1, '\0');
            file.read(&fbuf_fallback[0], fsz2);
            data = fbuf_fallback.data();
            fsz = static_cast<size_t>(fsz2);
        }
        struct MmapGuard
        {
            MmapView &v;
            ~MmapGuard() { mmap_close(v); }
        } _guard{mmap};

        const char *fend = data + fsz;

        // ── Step 2: header ─────────────────────────────────────────────────────
        const char *hdr_nl = static_cast<const char *>(std::memchr(data, '\n', fend - data));
        if (!hdr_nl)
            return GrizzlarFrame{};

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
            for (int samp = 0; samp < 1000 && sp < fend; ++samp)
            {
                const char *nl = static_cast<const char *>(
                    std::memchr(sp, '\n', fend - sp));
                if (!nl)
                    nl = fend;
                parse_csv_row_fast(sp, static_cast<size_t>(nl - sp), row_buf);
                for (size_t c = 0; c < ncols && c < row_buf.size(); ++c)
                {
                    if (type_id[c] == 2)
                        continue;
                    const std::string &v = row_buf[c];
                    if (v.empty())
                        continue;
                    if (type_id[c] == 0)
                    {
                        int64_t x;
                        if (!csv_try_int64(v.c_str(), v.size(), x))
                        {
                            double d;
                            type_id[c] = csv_try_double(v.c_str(), v.size(), d) ? 1 : 2;
                        }
                    }
                    else
                    {
                        double d;
                        if (!csv_try_double(v.c_str(), v.size(), d))
                            type_id[c] = 2;
                    }
                }
                sp = nl + 1;
            }
        }

        // ── find index column ──────────────────────────────────────────────────
        size_t idx_col = ncols;
        for (size_t c = 0; c < ncols; ++c)
        {
            if (!index_col_name.empty() && headers[c] == index_col_name)
            {
                idx_col = c;
                break;
            }
        }

        // ── Step 3: chunk boundaries at newlines ───────────────────────────────
        const size_t nthreads = static_cast<size_t>(
            std::max(1u, std::thread::hardware_concurrency()));
        const size_t data_len = static_cast<size_t>(fend - data_start);
        const size_t chunk_size = (data_len + nthreads - 1) / nthreads;

        std::vector<const char *> chunk_starts(nthreads + 1);
        chunk_starts[0] = data_start;
        for (size_t t = 1; t < nthreads; ++t)
        {
            const char *split = data_start + t * chunk_size;
            if (split >= fend)
            {
                chunk_starts[t] = fend;
                continue;
            }
            const char *nl = static_cast<const char *>(
                std::memchr(split, '\n', fend - split));
            chunk_starts[t] = nl ? nl + 1 : fend;
        }
        chunk_starts[nthreads] = fend;

        // ── Step 4: parallel parse ─────────────────────────────────────────────
        struct ColBuf
        {
            int type_id{0};
            std::vector<int64_t> ints;
            std::vector<double> dbls;
            StringArray sa;   // flat-buffer string storage — no per-string heap alloc
        };
        struct ChunkResult
        {
            std::vector<ColBuf> cols;
            size_t nrows{0};
            explicit ChunkResult(size_t nc) : cols(nc) {}
            ChunkResult() = default;
        };

        std::vector<std::future<ChunkResult>> futures;
        futures.reserve(nthreads);
        for (size_t t = 0; t < nthreads; ++t)
        {
            const char *cs = chunk_starts[t];
            const char *ce = chunk_starts[t + 1];
            futures.push_back(std::async(std::launch::async,
                                         [cs, ce, ncols, &type_id]() -> ChunkResult
                                         {
                                             ChunkResult r(ncols);
                                             for (size_t c = 0; c < ncols; ++c)
                                                 r.cols[c].type_id = type_id[c];
                                             // Pre-reserve: eliminates O(log N) resize passes for 250 K+ rows
                                             const size_t est = static_cast<size_t>(ce > cs ? ce - cs : 0) / 30 + 256;
                                             for (size_t c = 0; c < ncols; ++c)
                                             {
                                                 if (type_id[c] == 0)
                                                     r.cols[c].ints.reserve(est);
                                                 else if (type_id[c] == 1)
                                                     r.cols[c].dbls.reserve(est);
                                                 else
                                                 {
                                                     r.cols[c].sa.offsets.reserve(est + 1);
                                                     r.cols[c].sa.data.reserve(est * 12);
                                                 }
                                             }

                                             // Inline field scanner: no std::string allocation for numeric fields.
                                             // String fields pushed directly into StringArray flat buffer.
                                             std::string quoted_buf;
                                             const char *p = cs;
                                             while (p < ce)
                                             {
                                                 const char *nl = static_cast<const char *>(
                                                     std::memchr(p, '\n', ce - p));
                                                 if (!nl)
                                                     nl = ce;
                                                 const char *row_end = nl;
                                                 if (row_end > p && *(row_end - 1) == '\r')
                                                     --row_end;

                                                 if (row_end > p)
                                                 {
                                                     const char *fp = p;
                                                     for (size_t c = 0; c < ncols; ++c)
                                                     {
                                                         const char *fs;
                                                         size_t flen;
                                                         if (fp < row_end && *fp == '"')
                                                         {
                                                             ++fp;
                                                             quoted_buf.clear();
                                                             while (fp < row_end)
                                                             {
                                                                 char ch = *fp++;
                                                                 if (ch == '"')
                                                                 {
                                                                     if (fp < row_end && *fp == '"') { quoted_buf += '"'; ++fp; }
                                                                     else break;
                                                                 }
                                                                 else quoted_buf += ch;
                                                             }
                                                             if (fp < row_end && *fp == ',') ++fp;
                                                             fs = quoted_buf.c_str();
                                                             flen = quoted_buf.size();
                                                         }
                                                         else
                                                         {
                                                             fs = fp;
                                                             while (fp < row_end && *fp != ',') ++fp;
                                                             flen = static_cast<size_t>(fp - fs);
                                                             if (fp < row_end) ++fp; // skip ','
                                                         }
                                                         switch (r.cols[c].type_id)
                                                         {
                                                         case 0:
                                                         {
                                                             int64_t x = std::numeric_limits<int64_t>::min();
                                                             csv_try_int64(fs, flen, x);
                                                             r.cols[c].ints.push_back(x);
                                                             break;
                                                         }
                                                         case 1:
                                                         {
                                                             double x = std::numeric_limits<double>::quiet_NaN();
                                                             csv_try_double(fs, flen, x);
                                                             r.cols[c].dbls.push_back(x);
                                                             break;
                                                         }
                                                         case 2:
                                                             r.cols[c].sa.push_back(
                                                                 is_na_raw(fs, flen) ? std::string_view{} : std::string_view{fs, flen});
                                                             break;
                                                         }
                                                     }
                                                     ++r.nrows;
                                                 }
                                                 p = nl + 1;
                                             }
                                             return r;
                                         }));
        }

        // Collect results
        std::vector<ChunkResult> chunks;
        chunks.reserve(nthreads);
        size_t total_rows = 0;
        for (auto &f : futures)
        {
            chunks.push_back(f.get());
            total_rows += chunks.back().nrows;
        }

        // ── Step 5: merge per-chunk columns ───────────────────────────────────
        struct MergedCol
        {
            int type_id{0};
            std::vector<int64_t> ints;
            std::vector<double> dbls;
            StringArray sa;
        };
        std::vector<MergedCol> merged(ncols);

        auto merge_col = [&](size_t c) {
            merged[c].type_id = type_id[c];
            switch (type_id[c])
            {
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
            case 2: {
                size_t total_bytes = 0;
                for (auto &ch : chunks) total_bytes += ch.cols[c].sa.data.size();
                merged[c].sa.data.reserve(total_bytes);
                merged[c].sa.offsets.reserve(total_rows + 1);
                for (auto &ch : chunks) {
                    uint32_t base = static_cast<uint32_t>(merged[c].sa.data.size());
                    merged[c].sa.data.insert(merged[c].sa.data.end(),
                                             ch.cols[c].sa.data.begin(), ch.cols[c].sa.data.end());
                    for (size_t i = 1; i <= ch.cols[c].sa.size(); ++i)
                        merged[c].sa.offsets.push_back(base + ch.cols[c].sa.offsets[i]);
                }
                break;
            }
            }
        };

#if defined(GRIZZLAR_USE_EXECUTION)
        {
            std::vector<size_t> col_ids(ncols);
            std::iota(col_ids.begin(), col_ids.end(), 0);
            std::for_each(std::execution::par, col_ids.begin(), col_ids.end(), merge_col);
        }
#else
        for (size_t c = 0; c < ncols; ++c) merge_col(c);
#endif

        // ── assemble GrizzlarFrame ─────────────────────────────────────────────
        GrizzlarFrame out;
        std::vector<ulong> idx_vec;
        if (idx_col < ncols && merged[idx_col].type_id == 0)
        {
            idx_vec.reserve(total_rows);
            for (auto x : merged[idx_col].ints)
                idx_vec.push_back(static_cast<ulong>(x));
        }
        else
        {
            idx_vec.resize(total_rows);
            std::iota(idx_vec.begin(), idx_vec.end(), 0);
        }
        out.df_.load_index(std::move(idx_vec));

        for (size_t c = 0; c < ncols; ++c)
        {
            if (c == idx_col)
                continue;
            const std::string &nm = headers[c];
            out.col_order_.push_back(nm);
            switch (merged[c].type_id)
            {
            case 0: {
                auto &ints = merged[c].ints;
                const int64_t missing_sentinel = std::numeric_limits<int64_t>::min();
                bool has_missing = std::any_of(ints.begin(), ints.end(),
                    [missing_sentinel](int64_t v) { return v == missing_sentinel; });
                if (has_missing) {
                    out.col_types_[nm] = "double";
                    std::vector<double> dbls;
                    dbls.reserve(ints.size());
                    for (int64_t v : ints)
                        dbls.push_back(v == missing_sentinel
                            ? std::numeric_limits<double>::quiet_NaN()
                            : static_cast<double>(v));
                    out.df_.load_column<double>(nm.c_str(), std::move(dbls));
                } else {
                    out.col_types_[nm] = "int64";
                    out.df_.load_column<int64_t>(nm.c_str(), std::move(ints));
                }
                break;
            }
            case 1:
                out.col_types_[nm] = "double";
                out.df_.load_column<double>(nm.c_str(), std::move(merged[c].dbls));
                break;
            case 2:
                out.col_types_[nm] = "string";
                out.str_cols_[nm] = std::move(merged[c].sa);
                break;
            }
        }
        return out;
    }

    // Per-group aggregation for groupby_agg.
    // Computes directly over the source column (no intermediate vals vector).
    double aggregate_group(const std::string &col,
                           const std::vector<size_t> &indices,
                           const std::string &func) const
    {
        const size_t cnt = indices.size();
        if (func == "count")
            return static_cast<double>(cnt);
        if (cnt == 0)
            return 0.0;

        const std::string &type = col_types_.at(col);
        if (type != "double" && type != "int64")
            throw std::runtime_error("Cannot aggregate non-numeric column: " + col);

        auto get = [&](size_t i) -> double
        {
            if (type == "double")
                return df_.get_column<double>(col.c_str())[indices[i]];
            return static_cast<double>(df_.get_column<int64_t>(col.c_str())[indices[i]]);
        };

        if (func == "first")
            return get(0);
        if (func == "last")
            return get(cnt - 1);

        double s = 0;
        double mn = get(0), mx = get(0);
        for (size_t i = 0; i < cnt; ++i)
        {
            double v = get(i);
            s += v;
            if (v < mn)
                mn = v;
            if (v > mx)
                mx = v;
        }
        if (func == "sum")
            return s;
        if (func == "mean")
            return s / static_cast<double>(cnt);
        if (func == "min")
            return mn;
        if (func == "max")
            return mx;
        if (func == "std")
        {
            double mean_v = s / static_cast<double>(cnt);
            double sq = 0;
            for (size_t i = 0; i < cnt; ++i)
            {
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
    template <typename K>
    GrizzlarFrame do_groupby(const std::string &by_col,
                             const std::vector<K> &key_vec,
                             const std::vector<std::pair<std::string, std::string>> &specs) const
    {
        const size_t nspecs = specs.size();
        const size_t N = key_vec.size();

        // Pre-fetch column data pointers
        struct ColPtr { const double *dbl{nullptr}; const int64_t *i64{nullptr}; };
        std::vector<ColPtr> col_ptrs(nspecs);
        for (size_t s = 0; s < nspecs; ++s)
        {
            const auto &col = specs[s].first;
            const std::string &type = col_types_.at(col);
            if (type == "double")
                col_ptrs[s].dbl = df_.get_column<double>(col.c_str()).data();
            else if (type == "int64")
                col_ptrs[s].i64 = df_.get_column<int64_t>(col.c_str()).data();
            else
                throw std::runtime_error("Cannot aggregate non-numeric column: " + col);
        }

        struct RunState
        {
            double sum{0}, min_v{1e300}, max_v{-1e300}, sum_sq{0};
            double first_v{0}, last_v{0};
            int64_t count{0};
            bool initialized{false};
        };

        auto update = [&](RunState &st, double v)
        {
            if (!st.initialized) { st.first_v = v; st.min_v = v; st.max_v = v; st.initialized = true; }
            ++st.count; st.sum += v; st.sum_sq += v * v; st.last_v = v;
            if (v < st.min_v) st.min_v = v;
            if (v > st.max_v) st.max_v = v;
        };

        auto finalize = [&](const RunState &st, const std::string &func) -> double
        {
            if (func == "count") return static_cast<double>(st.count);
            if (func == "sum")   return st.sum;
            if (func == "mean")  return st.sum / static_cast<double>(st.count);
            if (func == "min")   return st.min_v;
            if (func == "max")   return st.max_v;
            if (func == "first") return st.first_v;
            if (func == "last")  return st.last_v;
            if (func == "std")
            {
                double m = st.sum / static_cast<double>(st.count);
                double var = st.sum_sq / static_cast<double>(st.count) - m * m;
                if (st.count > 1) var = var * static_cast<double>(st.count) / static_cast<double>(st.count - 1);
                return st.count > 0 ? std::sqrt(std::max(0.0, var)) : 0.0;
            }
            throw std::runtime_error("Unknown aggregation function: " + func);
        };

        using GroupMap = std::unordered_map<K, std::vector<RunState>>;

        // ── Choose parallelism: power-of-2 partitions, one per thread. ──────────
        // Small datasets or low-cardinality groupby go single-threaded to avoid
        // hash + partition overhead.
        const size_t hw = std::thread::hardware_concurrency();
        size_t P = 1;
        if (N >= 50000 && hw >= 2)
        {
            while (P * 2 <= hw) P *= 2;  // largest power-of-2 <= hw
        }

        // ── Single-threaded fast path ────────────────────────────────────────────
        if (P == 1)
        {
            GroupMap groups;
            groups.reserve(std::min(N / 4 + 8, (size_t)65536));
            for (size_t i = 0; i < N; ++i)
            {
                auto &states = groups[key_vec[i]];
                if (states.empty()) states.resize(nspecs);
                for (size_t s = 0; s < nspecs; ++s)
                    update(states[s], col_ptrs[s].dbl ? col_ptrs[s].dbl[i]
                                                      : static_cast<double>(col_ptrs[s].i64[i]));
            }
            std::vector<K> sorted_keys;
            sorted_keys.reserve(groups.size());
            for (auto &[k, _] : groups) sorted_keys.push_back(k);
            std::sort(sorted_keys.begin(), sorted_keys.end());

            const size_t ng = sorted_keys.size();
            std::vector<K> result_keys; result_keys.reserve(ng);
            std::vector<std::vector<double>> agg_results(nspecs);
            for (auto &v : agg_results) v.reserve(ng);
            for (const K &key : sorted_keys)
            {
                result_keys.push_back(key);
                const auto &states = groups[key];
                for (size_t s = 0; s < nspecs; ++s)
                    agg_results[s].push_back(finalize(states[s], specs[s].second));
            }
            return build_groupby_frame(by_col, result_keys, agg_results, specs);
        }

        // ── Parallel hash-partition path (polars-style) ──────────────────────────
        // hash(key) & (P-1) → partition index. All rows with the same key go to
        // the same partition, so each thread's GroupMap has disjoint keys.
        // No merge step needed — just collect and sort at the end.
        const size_t mask = P - 1;

        // Step 1: compute partition for every row (vectorizable, O(N))
        std::vector<uint32_t> row_part(N);
        for (size_t i = 0; i < N; ++i)
            row_part[i] = static_cast<uint32_t>(std::hash<K>{}(key_vec[i]) & mask);

        // Step 2: scatter row indices into per-partition lists (O(N))
        std::vector<size_t> part_sizes(P, 0);
        for (size_t i = 0; i < N; ++i) ++part_sizes[row_part[i]];
        std::vector<std::vector<size_t>> parts(P);
        for (size_t p = 0; p < P; ++p) parts[p].reserve(part_sizes[p]);
        for (size_t i = 0; i < N; ++i) parts[row_part[i]].push_back(i);

        // Step 3: parallel aggregation — each partition owns disjoint keys
        std::vector<GroupMap> part_maps(P);
        {
#if defined(GRIZZLAR_USE_EXECUTION)
            std::vector<size_t> part_ids(P);
            std::iota(part_ids.begin(), part_ids.end(), 0);
            std::for_each(std::execution::par, part_ids.begin(), part_ids.end(),
                [&](size_t p)
                {
                    auto &gmap = part_maps[p];
                    gmap.reserve(parts[p].size() / 2 + 4);
                    for (size_t idx : parts[p])
                    {
                        auto &states = gmap[key_vec[idx]];
                        if (states.empty()) states.resize(nspecs);
                        for (size_t s = 0; s < nspecs; ++s)
                            update(states[s], col_ptrs[s].dbl ? col_ptrs[s].dbl[idx]
                                                              : static_cast<double>(col_ptrs[s].i64[idx]));
                    }
                });
#else
            for (size_t p = 0; p < P; ++p)
            {
                auto &gmap = part_maps[p];
                gmap.reserve(parts[p].size() / 2 + 4);
                for (size_t idx : parts[p])
                {
                    auto &states = gmap[key_vec[idx]];
                    if (states.empty()) states.resize(nspecs);
                    for (size_t s = 0; s < nspecs; ++s)
                        update(states[s], col_ptrs[s].dbl ? col_ptrs[s].dbl[idx]
                                                          : static_cast<double>(col_ptrs[s].i64[idx]));
                }
            }
#endif
        }

        // Step 4: collect all keys, sort for deterministic output
        std::vector<K> sorted_keys;
        sorted_keys.reserve(N / 4 + 4);
        for (size_t p = 0; p < P; ++p)
            for (auto &[k, _] : part_maps[p]) sorted_keys.push_back(k);
        std::sort(sorted_keys.begin(), sorted_keys.end());

        const size_t ng = sorted_keys.size();
        std::vector<K> result_keys; result_keys.reserve(ng);
        std::vector<std::vector<double>> agg_results(nspecs);
        for (auto &v : agg_results) v.reserve(ng);

        for (const K &key : sorted_keys)
        {
            result_keys.push_back(key);
            // O(1) lookup: same hash partition as during aggregation
            const auto &states = part_maps[std::hash<K>{}(key) & mask].at(key);
            for (size_t s = 0; s < nspecs; ++s)
                agg_results[s].push_back(finalize(states[s], specs[s].second));
        }
        return build_groupby_frame(by_col, result_keys, agg_results, specs);
    }

    // Build a GrizzlarFrame from sorted groupby result vectors.
    template<typename K>
    GrizzlarFrame build_groupby_frame(const std::string &by_col,
                                       std::vector<K> &result_keys,
                                       std::vector<std::vector<double>> &agg_results,
                                       const std::vector<std::pair<std::string, std::string>> &specs) const
    {
        const size_t ng = result_keys.size();
        GrizzlarFrame out;
        std::vector<ulong> new_idx(ng);
        std::iota(new_idx.begin(), new_idx.end(), 0);
        out.df_.load_index(std::move(new_idx));

        out.col_order_.push_back(by_col);
        if constexpr (std::is_same_v<K, std::string_view>)
        {
            out.col_types_[by_col] = "string";
            StringArray str_keys;
            for (auto sv : result_keys) str_keys.push_back(sv);
            out.str_cols_[by_col] = std::move(str_keys);
        }
        else
        {
            out.col_types_[by_col] = col_types_.at(by_col);
            out.df_.load_column<K>(by_col.c_str(), std::move(result_keys));
        }
        for (size_t s = 0; s < specs.size(); ++s)
        {
            const auto &col = specs[s].first;
            out.col_order_.push_back(col);
            out.col_types_[col] = "double";
            out.df_.load_column<double>(col.c_str(), std::move(agg_results[s]));
        }
        return out;
    }

    // ── require helpers ──────────────────────────────────────────────────────

    std::string require_numeric(const std::string &col) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        if (it->second != "double" && it->second != "int64")
            throw std::runtime_error(
                "Column '" + col + "' is not numeric (type: " + it->second + ")");
        return it->second;
    }

    // ── loading ──────────────────────────────────────────────────────────────

    void load_index(py::object indices)
    {
        auto vec = to_vec<ulong>(indices);
        df_.load_index(std::move(vec));
    }

    void load_column(const std::string &name, py::object data)
    {
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
            str_cols_[name] = StringArray::from_py_list(py::cast<py::list>(data));
    }

    // ── accessors ────────────────────────────────────────────────────────────

    py::array_t<ulong> get_index() const
    {
        const auto &vec = df_.get_index();
        py::array_t<ulong> result(static_cast<py::ssize_t>(vec.size()));
        auto buf = result.request();
        std::copy(vec.begin(), vec.end(), static_cast<ulong *>(buf.ptr));
        return result;
    }

    py::object get_column(const std::string &name) const
    {
        auto it = col_types_.find(name);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + name);
        const std::string &type = it->second;
        if (type == "double")
        {
            const auto &vec = df_.get_column<double>(name.c_str());
            py::array_t<double> r(static_cast<py::ssize_t>(vec.size()));
            std::copy(vec.begin(), vec.end(), static_cast<double *>(r.request().ptr));
            return r;
        }
        if (type == "int64")
        {
            const auto &vec = df_.get_column<int64_t>(name.c_str());
            py::array_t<int64_t> r(static_cast<py::ssize_t>(vec.size()));
            std::copy(vec.begin(), vec.end(), static_cast<int64_t *>(r.request().ptr));
            return r;
        }
        if (type == "bool")
        {
            const auto &vec = df_.get_column<bool>(name.c_str());
            py::list lst;
            for (bool v : vec)
                lst.append(py::bool_(v));
            return lst;
        }
        return str_cols_.at(name).to_py_list();
    }

    std::vector<std::string> columns() const { return col_order_; }

    py::tuple shape() const
    {
        size_t nrows = df_.get_index().size();
        size_t ncols = col_order_.size();
        return py::make_tuple(nrows, ncols);
    }

    bool has_column(const std::string &name) const
    {
        return col_types_.find(name) != col_types_.end();
    }

    std::string col_type(const std::string &name) const
    {
        auto it = col_types_.find(name);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + name);
        return it->second;
    }

    // ── statistics ───────────────────────────────────────────────────────────

    double mean(const std::string &col)
    {
        auto type = require_numeric(col);
        if (type == "double")
        {
            MeanVisitor<double, ulong> v;
            df_.single_act_visit<double>(col.c_str(), v);
            return v.get_result();
        }
        MeanVisitor<int64_t, ulong> v;
        df_.single_act_visit<int64_t>(col.c_str(), v);
        return static_cast<double>(v.get_result());
    }
    double std_dev(const std::string &col)
    {
        auto type = require_numeric(col);
        if (type == "double")
        {
            StdVisitor<double, ulong> v;
            df_.single_act_visit<double>(col.c_str(), v);
            return v.get_result();
        }
        StdVisitor<int64_t, ulong> v;
        df_.single_act_visit<int64_t>(col.c_str(), v);
        return static_cast<double>(v.get_result());
    }
    double sum(const std::string &col)
    {
        auto type = require_numeric(col);
        if (type == "double")
        {
            SumVisitor<double, ulong> v;
            df_.single_act_visit<double>(col.c_str(), v);
            return v.get_result();
        }
        SumVisitor<int64_t, ulong> v;
        df_.single_act_visit<int64_t>(col.c_str(), v);
        return static_cast<double>(v.get_result());
    }
    double col_min(const std::string &col) const
    {
        auto type = require_numeric(col);
        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            if (v.empty())
                throw std::runtime_error("Empty");
            return *std::min_element(v.begin(), v.end());
        }
        const auto &v = df_.get_column<int64_t>(col.c_str());
        if (v.empty())
            throw std::runtime_error("Empty");
        return static_cast<double>(*std::min_element(v.begin(), v.end()));
    }
    double col_max(const std::string &col) const
    {
        auto type = require_numeric(col);
        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            if (v.empty())
                throw std::runtime_error("Empty");
            return *std::max_element(v.begin(), v.end());
        }
        const auto &v = df_.get_column<int64_t>(col.c_str());
        if (v.empty())
            throw std::runtime_error("Empty");
        return static_cast<double>(*std::max_element(v.begin(), v.end()));
    }
    size_t count(const std::string &col) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        if (type == "double")
            return df_.get_column<double>(col.c_str()).size();
        if (type == "int64")
            return df_.get_column<int64_t>(col.c_str()).size();
        if (type == "bool")
            return df_.get_column<bool>(col.c_str()).size();
        return str_cols_.at(col).size();
    }
    py::dict describe()
    {
        // Collect numeric columns in definition order.
        std::vector<std::string> num_cols;
        for (const auto &name : col_order_)
        {
            const auto &t = col_types_.at(name);
            if (t == "double" || t == "int64") num_cols.push_back(name);
        }
        if (num_cols.empty()) return py::dict();

        const size_t nc = num_cols.size();
        std::vector<DescribeStats> raw(nc);

        // _describe_raw is pure C++ — safe to run without the GIL.
        // Release GIL and use par execution policy so all column sorts run concurrently
        // via the ConcRT/TBB thread pool (zero thread-creation overhead vs std::async).
        {
            py::gil_scoped_release release;
#if defined(GRIZZLAR_USE_EXECUTION)
            std::vector<size_t> col_ids(nc);
            std::iota(col_ids.begin(), col_ids.end(), 0);
            std::for_each(std::execution::par, col_ids.begin(), col_ids.end(),
                [&](size_t i) { raw[i] = _describe_raw(num_cols[i]); });
#else
            for (size_t i = 0; i < nc; ++i) raw[i] = _describe_raw(num_cols[i]);
#endif
        }

        // GIL re-acquired here — build Python dicts.
        py::dict result;
        for (size_t i = 0; i < nc; ++i)
        {
            const auto &r = raw[i];
            py::dict d;
            d["count"] = r.count; d["mean"]  = r.mean;  d["std"]  = r.std_v;
            d["min"]   = r.min_v; d["25%"]   = r.q25;   d["50%"]  = r.q50;
            d["75%"]   = r.q75;   d["max"]   = r.max_v;
            result[num_cols[i].c_str()] = d;
        }
        return result;
    }

    // ── quantile / correlation / covariance ──────────────────────────────────

    double quantile(const std::string &col, double q) const
    {
        require_numeric(col);
        if (q < 0.0 || q > 1.0)
            throw std::runtime_error("q must be in [0, 1]");
        std::vector<double> vals;
        const std::string &type = col_types_.at(col);
        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            vals.assign(v.begin(), v.end());
        }
        else
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            for (auto x : v)
                vals.push_back(static_cast<double>(x));
        }
        if (vals.empty())
            throw std::runtime_error("Column is empty: " + col);
        std::sort(vals.begin(), vals.end());
        double pos = q * (vals.size() - 1);
        size_t lo = static_cast<size_t>(pos);
        double frac = pos - lo;
        if (lo + 1 < vals.size())
            return vals[lo] + frac * (vals[lo + 1] - vals[lo]);
        return vals[lo];
    }

    double corr(const std::string &col1, const std::string &col2) const
    {
        require_numeric(col1);
        require_numeric(col2);
        auto as_dbl = [this](const std::string &c)
        {
            std::vector<double> r;
            if (col_types_.at(c) == "double")
            {
                const auto &v = df_.get_column<double>(c.c_str());
                r.assign(v.begin(), v.end());
            }
            else
            {
                const auto &v = df_.get_column<int64_t>(c.c_str());
                for (auto x : v)
                    r.push_back(static_cast<double>(x));
            }
            return r;
        };
        auto v1 = as_dbl(col1), v2 = as_dbl(col2);
        size_t n = std::min(v1.size(), v2.size());
        if (n == 0)
            return 0.0;
        double m1 = std::accumulate(v1.begin(), v1.begin() + n, 0.0) / n;
        double m2 = std::accumulate(v2.begin(), v2.begin() + n, 0.0) / n;
        double cov = 0, var1 = 0, var2 = 0;
        for (size_t i = 0; i < n; ++i)
        {
            double d1 = v1[i] - m1, d2 = v2[i] - m2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }
        double denom = std::sqrt(var1 * var2);
        return denom > 0 ? cov / denom : 0.0;
    }

    double cov(const std::string &col1, const std::string &col2) const
    {
        require_numeric(col1);
        require_numeric(col2);
        auto as_dbl = [this](const std::string &c)
        {
            std::vector<double> r;
            if (col_types_.at(c) == "double")
            {
                const auto &v = df_.get_column<double>(c.c_str());
                r.assign(v.begin(), v.end());
            }
            else
            {
                const auto &v = df_.get_column<int64_t>(c.c_str());
                for (auto x : v)
                    r.push_back(static_cast<double>(x));
            }
            return r;
        };
        auto v1 = as_dbl(col1), v2 = as_dbl(col2);
        size_t n = std::min(v1.size(), v2.size());
        if (n < 2)
            return 0.0;
        double m1 = std::accumulate(v1.begin(), v1.begin() + n, 0.0) / n;
        double m2 = std::accumulate(v2.begin(), v2.begin() + n, 0.0) / n;
        double cov_val = 0;
        for (size_t i = 0; i < n; ++i)
            cov_val += (v1[i] - m1) * (v2[i] - m2);
        return cov_val / (n - 1);
    }

    // ── rolling / cumulative / shift ─────────────────────────────────────────

    py::array_t<double> rolling(const std::string &col, size_t window,
                                const std::string &func) const
    {
        require_numeric(col);
        size_t n = df_.get_index().size();
        py::array_t<double> result(static_cast<py::ssize_t>(n));
        auto buf = result.request();
        double *ptr = static_cast<double *>(buf.ptr);
        double nan = std::numeric_limits<double>::quiet_NaN();
        for (size_t i = 0; i < n; ++i)
            ptr[i] = nan;
        if (window == 0 || window > n)
            return result;

        auto fill = [&](auto &vec)
        {
            // Build initial window sum
            double ws = 0;
            for (size_t i = 0; i < window; ++i)
                ws += static_cast<double>(vec[i]);

            auto reduce = [&](size_t end_i) -> double
            {
                if (func == "mean")
                    return ws / window;
                if (func == "sum")
                    return ws;
                if (func == "std")
                {
                    double m = ws / window;
                    double sq = 0;
                    for (size_t j = end_i - window + 1; j <= end_i; ++j)
                        sq += (static_cast<double>(vec[j]) - m) * (static_cast<double>(vec[j]) - m);
                    return window > 1 ? std::sqrt(sq / (window - 1)) : 0.0;
                }
                if (func == "min")
                {
                    double v = static_cast<double>(vec[end_i - window + 1]);
                    for (size_t j = end_i - window + 2; j <= end_i; ++j)
                        v = std::min(v, static_cast<double>(vec[j]));
                    return v;
                }
                if (func == "max")
                {
                    double v = static_cast<double>(vec[end_i - window + 1]);
                    for (size_t j = end_i - window + 2; j <= end_i; ++j)
                        v = std::max(v, static_cast<double>(vec[j]));
                    return v;
                }
                return ws / window;
            };

            ptr[window - 1] = reduce(window - 1);
            for (size_t i = window; i < n; ++i)
            {
                ws += static_cast<double>(vec[i]) - static_cast<double>(vec[i - window]);
                ptr[i] = reduce(i);
            }
        };

        const std::string &type = col_types_.at(col);
        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            fill(v);
        }
        else
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            fill(v);
        }
        return result;
    }

    py::array_t<double> cumulative(const std::string &col,
                                   const std::string &func) const
    {
        require_numeric(col);
        size_t n = df_.get_index().size();
        py::array_t<double> result(static_cast<py::ssize_t>(n));
        auto buf = result.request();
        double *ptr = static_cast<double *>(buf.ptr);

        auto fill = [&](auto &vec)
        {
            double running = (func == "prod") ? 1.0 : 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                double x = static_cast<double>(vec[i]);
                if (func == "sum")
                    running += x;
                else if (func == "prod")
                    running *= x;
                else if (func == "min")
                    running = (i == 0) ? x : std::min(running, x);
                else if (func == "max")
                    running = (i == 0) ? x : std::max(running, x);
                ptr[i] = running;
            }
        };
        const std::string &type = col_types_.at(col);
        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            fill(v);
        }
        else
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            fill(v);
        }
        return result;
    }

    py::array_t<double> shift_col(const std::string &col, int n) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        size_t sz = df_.get_index().size();
        py::array_t<double> result(static_cast<py::ssize_t>(sz));
        auto buf = result.request();
        double *ptr = static_cast<double *>(buf.ptr);
        double nan = std::numeric_limits<double>::quiet_NaN();

        auto fill = [&](auto &vec)
        {
            if (n >= 0)
            {
                size_t sh = static_cast<size_t>(n);
                for (size_t i = 0; i < std::min(sh, sz); ++i)
                    ptr[i] = nan;
                for (size_t i = sh; i < sz; ++i)
                    ptr[i] = static_cast<double>(vec[i - sh]);
            }
            else
            {
                size_t sh = static_cast<size_t>(-n);
                for (size_t i = 0; i + sh < sz; ++i)
                    ptr[i] = static_cast<double>(vec[i + sh]);
                for (size_t i = (sz > sh ? sz - sh : 0); i < sz; ++i)
                    ptr[i] = nan;
            }
        };
        const std::string &type = it->second;
        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            fill(v);
        }
        else if (type == "int64")
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            fill(v);
        }
        else
        {
            for (size_t i = 0; i < sz; ++i)
                ptr[i] = nan;
        }
        return result;
    }

    py::array_t<double> pct_change(const std::string &col) const
    {
        require_numeric(col);
        size_t n = df_.get_index().size();
        py::array_t<double> result(static_cast<py::ssize_t>(n));
        auto buf = result.request();
        double *ptr = static_cast<double *>(buf.ptr);
        double nan = std::numeric_limits<double>::quiet_NaN();
        ptr[0] = nan;

        auto fill = [&](auto &vec)
        {
            for (size_t i = 1; i < n; ++i)
            {
                double prev = static_cast<double>(vec[i - 1]);
                ptr[i] = (prev != 0) ? (static_cast<double>(vec[i]) - prev) / prev : nan;
            }
        };
        const std::string &type = col_types_.at(col);
        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            fill(v);
        }
        else
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            fill(v);
        }
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
    GrizzlarFrame sort_by(const std::string &col, bool ascending = true) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        const size_t n = df_.get_index().size();

        std::vector<size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);

        if (type == "string")
        {
            // (key, idx) pairs: string_view + index stored adjacently so the
            // comparator never dereferences through a separate keys array.
            struct PairSV { std::string_view key; uint32_t idx; };
            const StringArray &sa = str_cols_.at(col);
            std::vector<PairSV> pairs(n);
            for (size_t i = 0; i < n; ++i)
                pairs[i] = {sa.view(i), static_cast<uint32_t>(i)};
            if (ascending)
                GRIZZLAR_SORT(GRIZZLAR_EXEC_POLICY, pairs.begin(), pairs.end(),
                              [](const PairSV &a, const PairSV &b)
                              { return a.key < b.key; });
            else
                GRIZZLAR_SORT(GRIZZLAR_EXEC_POLICY, pairs.begin(), pairs.end(),
                              [](const PairSV &a, const PairSV &b)
                              { return a.key > b.key; });
            for (size_t i = 0; i < n; ++i) perm[i] = pairs[i].idx;
        }
        else if (type == "int64")
        {
            // (key, idx) pairs: comparator touches adjacent memory — no indirection.
            struct Pair64 { int64_t key; uint32_t idx; };
            const auto &raw = df_.get_column<int64_t>(col.c_str());
            std::vector<Pair64> pairs(n);
            for (size_t i = 0; i < n; ++i) pairs[i] = {raw[i], static_cast<uint32_t>(i)};
            if (ascending)
                GRIZZLAR_SORT(GRIZZLAR_EXEC_POLICY, pairs.begin(), pairs.end(),
                              [](const Pair64 &a, const Pair64 &b) { return a.key < b.key; });
            else
                GRIZZLAR_SORT(GRIZZLAR_EXEC_POLICY, pairs.begin(), pairs.end(),
                              [](const Pair64 &a, const Pair64 &b) { return a.key > b.key; });
            for (size_t i = 0; i < n; ++i) perm[i] = pairs[i].idx;
        }
        else if (type == "double")
        {
            struct PairDbl { double key; uint32_t idx; };
            const auto &raw = df_.get_column<double>(col.c_str());
            std::vector<PairDbl> pairs(n);
            for (size_t i = 0; i < n; ++i) pairs[i] = {raw[i], static_cast<uint32_t>(i)};
            if (ascending)
                GRIZZLAR_SORT(GRIZZLAR_EXEC_POLICY, pairs.begin(), pairs.end(),
                              [](const PairDbl &a, const PairDbl &b) { return a.key < b.key; });
            else
                GRIZZLAR_SORT(GRIZZLAR_EXEC_POLICY, pairs.begin(), pairs.end(),
                              [](const PairDbl &a, const PairDbl &b) { return a.key > b.key; });
            for (size_t i = 0; i < n; ++i) perm[i] = pairs[i].idx;
        }
        else
        {
            throw std::runtime_error("sort_by: unsortable column type: " + type);
        }

        return extract_rows_parallel(perm);
    }

    GrizzlarFrame sort_index(bool ascending = true) const
    {
        const size_t n = df_.get_index().size();
        std::vector<size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        const auto &idx = df_.get_index();
        if (ascending)
            GRIZZLAR_SORT(GRIZZLAR_EXEC_POLICY, perm.begin(), perm.end(),
                          [&](size_t a, size_t b)
                          { return idx[a] < idx[b]; });
        else
            GRIZZLAR_SORT(GRIZZLAR_EXEC_POLICY, perm.begin(), perm.end(),
                          [&](size_t a, size_t b)
                          { return idx[a] > idx[b]; });
        return extract_rows_parallel(perm);
    }

    // ── filtering ────────────────────────────────────────────────────────────

    // Filter rows using a Python boolean mask (list[bool] or numpy bool array).
    // Direct compress: no intermediate index vector, sequential access (SIMD-friendly).
    // For frames with >= 50K output rows, processes columns in parallel threads.
    // Internal compress: shared by filter_by_mask, filter_col_scalar, filter_by_mask_list.
    // mask is uint8_t (not bool) for SIMD-friendly comparison loops.
    GrizzlarFrame compress_by_uint8(const uint8_t *m, size_t n, size_t out_n) const
    {
        GrizzlarFrame out;
        out.col_order_ = col_order_;
        out.col_types_ = col_types_;
        const size_t ncols = col_order_.size();

        std::vector<size_t> str_ci;
        str_ci.reserve(ncols);
        for (size_t ci = 0; ci < ncols; ++ci)
            if (col_types_.at(col_order_[ci]) == "string") str_ci.push_back(ci);

        struct ColOut { std::vector<int64_t> ints; std::vector<double> dbls; std::vector<bool> bools; };
        std::vector<ulong> new_idx(out_n);
        std::vector<ColOut> col_outs(ncols);
        std::vector<StringArray> str_outs(str_ci.size());

        for (size_t ci = 0; ci < ncols; ++ci)
        {
            const std::string &type = col_types_.at(col_order_[ci]);
            if (type == "double")     col_outs[ci].dbls.resize(out_n);
            else if (type == "int64") col_outs[ci].ints.resize(out_n);
            else if (type == "bool")  col_outs[ci].bools.resize(out_n, false);
        }

        auto compress_unit = [&](size_t unit)
        {
            if (unit == 0)
            {
                ulong *dst = new_idx.data();
                const ulong *src = df_.get_index().data();
                for (size_t i = 0; i < n; ++i)
                    if (m[i]) *dst++ = src[i];
            }
            else
            {
                const size_t ci = unit - 1;
                const std::string &cname = col_order_[ci];
                const std::string &type  = col_types_.at(cname);
                if (type == "double")
                {
                    const double *src = df_.get_column<double>(cname.c_str()).data();
                    double *dp = col_outs[ci].dbls.data();
                    for (size_t i = 0; i < n; ++i)
                        if (m[i]) *dp++ = src[i];
                }
                else if (type == "int64")
                {
                    const int64_t *src = df_.get_column<int64_t>(cname.c_str()).data();
                    int64_t *dp = col_outs[ci].ints.data();
                    for (size_t i = 0; i < n; ++i)
                        if (m[i]) *dp++ = src[i];
                }
                else if (type == "bool")
                {
                    const auto &src = df_.get_column<bool>(cname.c_str());
                    size_t w = 0;
                    for (size_t i = 0; i < n; ++i)
                        if (m[i]) col_outs[ci].bools[w++] = src[i];
                }
                // string columns handled in str_outs below
            }
        };

        const size_t total_units = ncols + 1;
        const bool do_parallel = (out_n >= 50000 && ncols >= 2);

#if defined(GRIZZLAR_USE_EXECUTION)
        if (do_parallel)
        {
            // Unified task list: [0..total_units) = compress_unit(u),
            // [total_units..total_units+str_ci.size()) = StringArray::compress for string cols.
            // std::execution::par reuses the ConcRT/TBB thread pool — zero thread-creation overhead.
            const size_t n_tasks = total_units + str_ci.size();
            std::vector<size_t> all_tasks(n_tasks);
            std::iota(all_tasks.begin(), all_tasks.end(), 0);
            std::for_each(std::execution::par, all_tasks.begin(), all_tasks.end(),
                [&](size_t tid) {
                    if (tid < total_units)
                        compress_unit(tid);
                    else
                    {
                        const size_t si = tid - total_units;
                        str_outs[si] = str_cols_.at(col_order_[str_ci[si]]).compress(m, n);
                    }
                });
        }
        else
        {
            for (size_t u = 0; u < total_units; ++u) compress_unit(u);
            for (size_t si = 0; si < str_ci.size(); ++si)
                str_outs[si] = str_cols_.at(col_order_[str_ci[si]]).compress(m, n);
        }
#else
        for (size_t u = 0; u < total_units; ++u) compress_unit(u);
        for (size_t si = 0; si < str_ci.size(); ++si)
            str_outs[si] = str_cols_.at(col_order_[str_ci[si]]).compress(m, n);
#endif

        out.df_.load_index(std::move(new_idx));
        for (size_t ci = 0; ci < ncols; ++ci)
        {
            const std::string &cname = col_order_[ci];
            const std::string &type  = col_types_.at(cname);
            if (type == "double")
                out.df_.load_column<double>(cname.c_str(), std::move(col_outs[ci].dbls));
            else if (type == "int64")
                out.df_.load_column<int64_t>(cname.c_str(), std::move(col_outs[ci].ints));
            else if (type == "bool")
                out.df_.load_column<bool>(cname.c_str(), std::move(col_outs[ci].bools));
        }
        for (size_t si = 0; si < str_ci.size(); ++si)
            out.str_cols_[col_order_[str_ci[si]]] = std::move(str_outs[si]);
        return out;
    }

    GrizzlarFrame filter_by_mask(py::object mask_obj) const
    {
        const auto &idx = df_.get_index();
        const size_t n = idx.size();

        auto arr = py::cast<py::array_t<bool, py::array::c_style | py::array::forcecast>>(mask_obj);
        auto buf_info = arr.request();
        if (static_cast<size_t>(buf_info.size) != n)
            throw std::runtime_error("mask length " + std::to_string(buf_info.size) +
                                     " != frame length " + std::to_string(n));
        const bool *bm = static_cast<const bool *>(buf_info.ptr);

        // Convert bool→uint8_t for SIMD-friendly loops in compress_by_uint8
        std::vector<uint8_t> m(n);
        size_t out_n = 0;
        for (size_t i = 0; i < n; ++i) { m[i] = bm[i] ? 1 : 0; out_n += m[i]; }

        if (out_n == n) return deep_copy();
        return compress_by_uint8(m.data(), n, out_n);
    }

    // Slice rows by integer position [start, stop).
    GrizzlarFrame iloc(long start, long stop) const
    {
        long n = static_cast<long>(df_.get_index().size());
        if (start < 0)
            start = std::max(0L, n + start);
        if (stop < 0)
            stop = std::max(0L, n + stop);
        start = std::min(start, n);
        stop = std::min(stop, n);
        std::vector<size_t> locs;
        for (long i = start; i < stop; ++i)
            locs.push_back(static_cast<size_t>(i));
        return extract_rows(locs);
    }

    // Return a deep copy of this frame using the hmdf copy constructor.
    // Faster than extract_rows(all_rows) for full-frame copies (e.g. before sort).
    GrizzlarFrame deep_copy() const
    {
        GrizzlarFrame out;
        out.df_ = df_;
        out.col_types_ = col_types_;
        out.col_order_ = col_order_;
        out.str_cols_  = str_cols_;
        return out;
    }

    // Return a new frame with only the requested columns (projection).
    GrizzlarFrame select_columns(const std::vector<std::string> &names) const
    {
        GrizzlarFrame out;
        const auto &src_idx = df_.get_index();
        std::vector<ulong> new_idx(src_idx.begin(), src_idx.end());
        out.df_.load_index(std::move(new_idx));

        for (const auto &name : names)
        {
            auto it = col_types_.find(name);
            if (it == col_types_.end())
                throw std::runtime_error("Column not found: " + name);
            out.col_order_.push_back(name);
            const std::string &type = it->second;
            out.col_types_[name] = type;
            if (type == "double")
                out.df_.load_column<double>(name.c_str(), df_.get_column<double>(name.c_str()));
            else if (type == "int64")
                out.df_.load_column<int64_t>(name.c_str(), df_.get_column<int64_t>(name.c_str()));
            else if (type == "bool")
                out.df_.load_column<bool>(name.c_str(), df_.get_column<bool>(name.c_str()));
            else
                out.str_cols_[name] = str_cols_.at(name);
        }
        return out;
    }

    // ── groupby ──────────────────────────────────────────────────────────────

    // specs: list of (agg_col, func) pairs.
    // Supported funcs: "mean","sum","min","max","count","std","first","last"
    GrizzlarFrame groupby_agg(const std::string &by_col,
                              const std::vector<std::pair<std::string, std::string>> &specs) const
    {
        auto it = col_types_.find(by_col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + by_col);
        for (const auto &[col, _] : specs)
            require_numeric(col);

        const std::string &by_type = it->second;
        if (by_type == "double")
        {
            const auto &v = df_.get_column<double>(by_col.c_str());
            return do_groupby<double>(by_col, {v.begin(), v.end()}, specs);
        }
        else if (by_type == "int64")
        {
            const auto &v = df_.get_column<int64_t>(by_col.c_str());
            return do_groupby<int64_t>(by_col, {v.begin(), v.end()}, specs);
        }
        else if (by_type == "string")
        {
            const StringArray &sa = str_cols_.at(by_col);
            // string_view keys: avoids copying strings into a new vector
            std::vector<std::string_view> key_views;
            key_views.reserve(sa.size());
            for (size_t i = 0; i < sa.size(); ++i)
                key_views.emplace_back(sa.view(i));
            return do_groupby<std::string_view>(by_col, key_views, specs);
        }
        throw std::runtime_error("Cannot group by column of type: " + by_type);
    }

    // ── join ─────────────────────────────────────────────────────────────────

    // Hash join two frames on their shared index.
    // Builds an unordered_map from the right index, probes with the left index,
    // then scatters columns in parallel — O(n+m) with no sort required.
    // how: "inner" | "left" | "right" | "outer"
    GrizzlarFrame join_by_index(const GrizzlarFrame &rhs, const std::string &how) const
    {
        const bool do_inner = (how == "inner");
        const bool do_left = (how == "left");
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

        if (do_inner || do_left)
        {
            left_pos.reserve(li.size());
            right_pos.reserve(li.size());
            for (size_t i = 0; i < li.size(); ++i)
            {
                auto it = right_map.find(li[i]);
                if (it != right_map.end())
                {
                    left_pos.push_back(i);
                    right_pos.push_back(it->second);
                }
                else if (do_left)
                {
                    left_pos.push_back(i);
                    right_pos.push_back(NO_MATCH);
                }
            }
        }
        else if (do_right)
        {
            std::unordered_map<ulong, size_t> left_map;
            left_map.reserve(li.size() * 2);
            for (size_t i = 0; i < li.size(); ++i)
                left_map.emplace(li[i], i);
            left_pos.reserve(ri.size());
            right_pos.reserve(ri.size());
            for (size_t j = 0; j < ri.size(); ++j)
            {
                auto it = left_map.find(ri[j]);
                left_pos.push_back(it != left_map.end() ? it->second : NO_MATCH);
                right_pos.push_back(j);
            }
        }
        else
        { // outer
            std::vector<bool> right_matched(ri.size(), false);
            left_pos.reserve(li.size());
            right_pos.reserve(li.size());
            for (size_t i = 0; i < li.size(); ++i)
            {
                auto it = right_map.find(li[i]);
                if (it != right_map.end())
                {
                    left_pos.push_back(i);
                    right_pos.push_back(it->second);
                    right_matched[it->second] = true;
                }
                else
                {
                    left_pos.push_back(i);
                    right_pos.push_back(NO_MATCH);
                }
            }
            for (size_t j = 0; j < ri.size(); ++j)
            {
                if (!right_matched[j])
                {
                    left_pos.push_back(NO_MATCH);
                    right_pos.push_back(j);
                }
            }
        }

        const size_t n = left_pos.size();
        const size_t nleft = col_order_.size();
        const size_t nright = rhs.col_order_.size();
        const size_t total_units = 1 + nleft + nright;

        // Output column order and types: left then right
        GrizzlarFrame out;
        out.col_order_ = col_order_;
        out.col_types_ = col_types_;
        for (const auto &name : rhs.col_order_)
        {
            out.col_order_.push_back(name);
            out.col_types_[name] = rhs.col_types_.at(name);
        }

        // Pre-allocate output column buffers (default = null/zero/empty)
        struct ColBuf
        {
            std::vector<int64_t> ints;
            std::vector<double> dbls;
            std::vector<bool> bools;
        };
        std::vector<ulong> new_idx(n);
        std::vector<ColBuf> col_bufs(nleft + nright);

        auto alloc_buf = [&](size_t ci, const std::string &type)
        {
            if (type == "double")
                col_bufs[ci].dbls.resize(n, std::numeric_limits<double>::quiet_NaN());
            else if (type == "int64")
                col_bufs[ci].ints.resize(n, 0);
            else if (type == "bool")
                col_bufs[ci].bools.resize(n, false);
            // string cols handled via scatter_join after parallel scatter
        };
        for (size_t ci = 0; ci < nleft; ++ci)
            alloc_buf(ci, col_types_.at(col_order_[ci]));
        for (size_t ci = 0; ci < nright; ++ci)
            alloc_buf(nleft + ci, rhs.col_types_.at(rhs.col_order_[ci]));

        // Parallel scatter: each "unit" fills one column (or the index)
        auto scatter_unit = [&](size_t unit)
        {
            if (unit == 0)
            {
                // Index: left side wins; outer unmatched-right rows use right index
                for (size_t j = 0; j < n; ++j)
                    new_idx[j] = (left_pos[j] != NO_MATCH) ? li[left_pos[j]] : ri[right_pos[j]];
            }
            else if (unit <= nleft)
            {
                const size_t ci = unit - 1;
                const auto &cname = col_order_[ci];
                const auto &type = col_types_.at(cname);
                if (type == "double")
                {
                    const double *src = df_.get_column<double>(cname.c_str()).data();
                    double *dst = col_bufs[ci].dbls.data();
                    for (size_t j = 0; j < n; ++j)
                        if (left_pos[j] != NO_MATCH)
                            dst[j] = src[left_pos[j]];
                }
                else if (type == "int64")
                {
                    const int64_t *src = df_.get_column<int64_t>(cname.c_str()).data();
                    int64_t *dst = col_bufs[ci].ints.data();
                    for (size_t j = 0; j < n; ++j)
                        if (left_pos[j] != NO_MATCH)
                            dst[j] = src[left_pos[j]];
                }
                else if (type == "bool")
                {
                    const auto &sv = df_.get_column<bool>(cname.c_str());
                    auto &dv = col_bufs[ci].bools;
                    for (size_t j = 0; j < n; ++j)
                        if (left_pos[j] != NO_MATCH)
                            dv[j] = sv[left_pos[j]];
                }
                // string cols handled via scatter_join after parallel scatter
            }
            else
            {
                const size_t ci = unit - 1 - nleft;
                const auto &cname = rhs.col_order_[ci];
                const auto &type = rhs.col_types_.at(cname);
                if (type == "double")
                {
                    const double *src = rhs.df_.get_column<double>(cname.c_str()).data();
                    double *dst = col_bufs[nleft + ci].dbls.data();
                    for (size_t j = 0; j < n; ++j)
                        if (right_pos[j] != NO_MATCH)
                            dst[j] = src[right_pos[j]];
                }
                else if (type == "int64")
                {
                    const int64_t *src = rhs.df_.get_column<int64_t>(cname.c_str()).data();
                    int64_t *dst = col_bufs[nleft + ci].ints.data();
                    for (size_t j = 0; j < n; ++j)
                        if (right_pos[j] != NO_MATCH)
                            dst[j] = src[right_pos[j]];
                }
                else if (type == "bool")
                {
                    const auto &sv = rhs.df_.get_column<bool>(cname.c_str());
                    auto &dv = col_bufs[nleft + ci].bools;
                    for (size_t j = 0; j < n; ++j)
                        if (right_pos[j] != NO_MATCH)
                            dv[j] = sv[right_pos[j]];
                }
                // string cols handled via scatter_join after parallel scatter
            }
        };

        const bool do_parallel_scatter = (n >= 10000 && total_units >= 2);

#if defined(GRIZZLAR_USE_EXECUTION)
        if (do_parallel_scatter)
        {
            std::vector<size_t> all_tasks(total_units);
            std::iota(all_tasks.begin(), all_tasks.end(), 0);
            std::for_each(std::execution::par, all_tasks.begin(), all_tasks.end(),
                [&](size_t u) { scatter_unit(u); });
        }
        else
        {
            for (size_t u = 0; u < total_units; ++u) scatter_unit(u);
        }
#else
        for (size_t u = 0; u < total_units; ++u) scatter_unit(u);
#endif

        out.df_.load_index(std::move(new_idx));
        for (size_t ci = 0; ci < nleft + nright; ++ci)
        {
            const auto &cname = (ci < nleft) ? col_order_[ci] : rhs.col_order_[ci - nleft];
            const auto &type = out.col_types_.at(cname);
            if (type == "double")
                out.df_.load_column<double>(cname.c_str(), std::move(col_bufs[ci].dbls));
            else if (type == "int64")
                out.df_.load_column<int64_t>(cname.c_str(), std::move(col_bufs[ci].ints));
            else if (type == "bool")
                out.df_.load_column<bool>(cname.c_str(), std::move(col_bufs[ci].bools));
            // string cols scattered below
        }
        // Scatter string columns using flat-buffer scatter_join
        // Collect (column_name, side) pairs for string cols so we can parallelise
        struct StrTask { std::string name; bool is_right; };
        std::vector<StrTask> str_tasks;
        for (size_t ci = 0; ci < nleft; ++ci)
        {
            const auto &cname = col_order_[ci];
            if (col_types_.at(cname) == "string")
                str_tasks.push_back({cname, false});
        }
        for (size_t ci = 0; ci < nright; ++ci)
        {
            const auto &cname = rhs.col_order_[ci];
            if (rhs.col_types_.at(cname) == "string")
                str_tasks.push_back({cname, true});
        }

        // Pre-allocate result slots so parallel writes go to independent locations
        const size_t n_str = str_tasks.size();
        std::vector<StringArray> str_results(n_str);

#if defined(GRIZZLAR_USE_EXECUTION)
        if (do_parallel_scatter && n_str >= 2)
        {
            std::vector<size_t> str_ids(n_str);
            std::iota(str_ids.begin(), str_ids.end(), 0);
            std::for_each(std::execution::par, str_ids.begin(), str_ids.end(),
                [&](size_t i) {
                    const auto &t = str_tasks[i];
                    if (!t.is_right)
                        str_results[i] = str_cols_.at(t.name).scatter_join(left_pos, NO_MATCH);
                    else
                        str_results[i] = rhs.str_cols_.at(t.name).scatter_join(right_pos, NO_MATCH);
                });
        }
        else
        {
#endif
            for (size_t i = 0; i < n_str; ++i)
            {
                const auto &t = str_tasks[i];
                if (!t.is_right)
                    str_results[i] = str_cols_.at(t.name).scatter_join(left_pos, NO_MATCH);
                else
                    str_results[i] = rhs.str_cols_.at(t.name).scatter_join(right_pos, NO_MATCH);
            }
#if defined(GRIZZLAR_USE_EXECUTION)
        }
#endif

        for (size_t i = 0; i < n_str; ++i)
            out.str_cols_[str_tasks[i].name] = std::move(str_results[i]);
        return out;
    }

    // ── concat ───────────────────────────────────────────────────────────────

    // Vertically concatenate two frames (append rows). Columns present in
    // both frames with the same type are combined; others are dropped.
    // Index is reset to 0..N-1.
    GrizzlarFrame concat_frame(const GrizzlarFrame &other) const
    {
        GrizzlarFrame out;
        size_t n1 = df_.get_index().size();
        size_t n2 = other.df_.get_index().size();
        size_t total = n1 + n2;
        std::vector<ulong> new_idx(total);
        std::iota(new_idx.begin(), new_idx.end(), 0);
        out.df_.load_index(std::move(new_idx));

        for (const auto &name : col_order_)
        {
            auto o = other.col_types_.find(name);
            if (o == other.col_types_.end())
                continue;
            const std::string &type = col_types_.at(name);
            if (o->second != type)
                continue;

            out.col_order_.push_back(name);
            out.col_types_[name] = type;

            if (type == "double")
            {
                const auto &a = df_.get_column<double>(name.c_str());
                const auto &b = other.df_.get_column<double>(name.c_str());
                std::vector<double> combined;
                combined.reserve(total);
                combined.insert(combined.end(), a.begin(), a.end());
                combined.insert(combined.end(), b.begin(), b.end());
                out.df_.load_column<double>(name.c_str(), std::move(combined));
            }
            else if (type == "int64")
            {
                const auto &a = df_.get_column<int64_t>(name.c_str());
                const auto &b = other.df_.get_column<int64_t>(name.c_str());
                std::vector<int64_t> combined;
                combined.reserve(total);
                combined.insert(combined.end(), a.begin(), a.end());
                combined.insert(combined.end(), b.begin(), b.end());
                out.df_.load_column<int64_t>(name.c_str(), std::move(combined));
            }
            else if (type == "bool")
            {
                const auto &a = df_.get_column<bool>(name.c_str());
                const auto &b = other.df_.get_column<bool>(name.c_str());
                std::vector<bool> combined;
                combined.reserve(total);
                combined.insert(combined.end(), a.begin(), a.end());
                combined.insert(combined.end(), b.begin(), b.end());
                out.df_.load_column<bool>(name.c_str(), std::move(combined));
            }
            else
            {
                out.str_cols_[name] = str_cols_.at(name).concat_with(other.str_cols_.at(name));
            }
        }
        return out;
    }

    // ── data cleaning ────────────────────────────────────────────────────────

    // Return a new frame with duplicate rows removed (keep first occurrence).
    GrizzlarFrame drop_duplicates(const std::string &col) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        std::vector<size_t> keep;

        if (type == "double")
        {
            std::unordered_set<double> seen;
            const auto &v = df_.get_column<double>(col.c_str());
            for (size_t i = 0; i < v.size(); ++i)
                if (seen.insert(v[i]).second)
                    keep.push_back(i);
        }
        else if (type == "int64")
        {
            std::unordered_set<int64_t> seen;
            const auto &v = df_.get_column<int64_t>(col.c_str());
            for (size_t i = 0; i < v.size(); ++i)
                if (seen.insert(v[i]).second)
                    keep.push_back(i);
        }
        else if (type == "string")
        {
            std::unordered_set<std::string> seen;
            const StringArray &sa = str_cols_.at(col);
            for (size_t i = 0; i < sa.size(); ++i)
                if (seen.emplace(sa.str(i)).second)
                    keep.push_back(i);
        }
        else
        {
            bool st = false, sf = false;
            const auto &v = df_.get_column<bool>(col.c_str());
            for (size_t i = 0; i < v.size(); ++i)
            {
                if ((v[i] && !st) || (!v[i] && !sf))
                {
                    keep.push_back(i);
                    if (v[i])
                        st = true;
                    else
                        sf = true;
                }
            }
        }
        return extract_rows(keep);
    }

    // Remove rows where the given column has a NaN (double) or empty string.
    GrizzlarFrame drop_na(const std::string &col) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        std::vector<size_t> keep;

        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            for (size_t i = 0; i < v.size(); ++i)
                if (!std::isnan(v[i]))
                    keep.push_back(i);
        }
        else if (type == "string")
        {
            const StringArray &sa = str_cols_.at(col);
            for (size_t i = 0; i < sa.size(); ++i)
                if (!sa.view(i).empty())
                    keep.push_back(i);
        }
        else
        {
            // int64 / bool — no NaN concept, return as-is
            size_t n = df_.get_index().size();
            for (size_t i = 0; i < n; ++i)
                keep.push_back(i);
        }
        return extract_rows(keep);
    }

    // Fill NaN (double) or empty string in-place.
    void fillna(const std::string &col, py::object value)
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        if (type == "double")
        {
            double fill = py::cast<double>(value);
            auto &v = df_.get_column<double>(col.c_str());
            for (auto &x : v)
                if (std::isnan(x))
                    x = fill;
        }
        else if (type == "string")
        {
            std::string fill = py::cast<std::string>(value);
            str_cols_[col] = str_cols_.at(col).with_fillna(fill);
        }
    }

    // Rename a column in-place.
    void rename_col(const std::string &old_name, const std::string &new_name)
    {
        auto it = col_types_.find(old_name);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + old_name);
        if (col_types_.count(new_name))
            throw std::runtime_error("Column already exists: " + new_name);
        std::string type = it->second;
        col_types_.erase(it);
        col_types_[new_name] = type;
        if (type == "string")
        {
            str_cols_[new_name] = std::move(str_cols_.at(old_name));
            str_cols_.erase(old_name);
        }
        else
        {
            df_.rename_column(old_name.c_str(), new_name.c_str());
        }
        for (auto &n : col_order_)
            if (n == old_name)
            {
                n = new_name;
                break;
            }
    }

    // Remove a column in-place.
    void drop_column(const std::string &name)
    {
        auto it = col_types_.find(name);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + name);
        const std::string &type = it->second;
        if (type == "double")
            df_.remove_column<double>(name.c_str());
        else if (type == "int64")
            df_.remove_column<int64_t>(name.c_str());
        else if (type == "bool")
            df_.remove_column<bool>(name.c_str());
        else
            str_cols_.erase(name);
        col_types_.erase(it);
        col_order_.erase(std::remove(col_order_.begin(), col_order_.end(), name), col_order_.end());
    }

    // ── utilities ────────────────────────────────────────────────────────────

    // Frequency count of each unique value; returns a frame with "value","count" cols.
    GrizzlarFrame value_counts(const std::string &col) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;

        std::vector<std::string> keys;
        std::vector<int64_t> cnts;

        auto add_counts = [&](auto &vec)
        {
            std::map<std::string, int64_t> m;
            for (const auto &x : vec)
            {
                if constexpr (std::is_same_v<std::decay_t<decltype(x)>, std::string>)
                    m[x]++;
                else
                    m[std::to_string(x)]++;
            }
            for (auto &[k, cv] : m)
            {
                keys.push_back(k);
                cnts.push_back(cv);
            }
        };

        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            add_counts(v);
        }
        else if (type == "int64")
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            add_counts(v);
        }
        else if (type == "string")
        {
            const StringArray &sa = str_cols_.at(col);
            std::map<std::string, int64_t> m;
            for (size_t i = 0; i < sa.size(); ++i)
                m[std::string(sa.view(i))]++;
            for (auto &[k, cv] : m) { keys.push_back(k); cnts.push_back(cv); }
        }
        else
        {
            const auto &v = df_.get_column<bool>(col.c_str());
            int64_t t = 0, f = 0;
            for (bool b : v)
            {
                if (b)
                    t++;
                else
                    f++;
            }
            keys = {"false", "true"};
            cnts = {f, t};
        }

        // Sort by count descending
        std::vector<size_t> order(keys.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b)
                  { return cnts[a] > cnts[b]; });

        GrizzlarFrame out;
        std::vector<ulong> idx(keys.size());
        std::iota(idx.begin(), idx.end(), 0);
        out.df_.load_index(std::move(idx));
        std::vector<std::string> sk;
        sk.reserve(keys.size());
        std::vector<int64_t> sc;
        sc.reserve(cnts.size());
        for (size_t i : order)
        {
            sk.push_back(keys[i]);
            sc.push_back(cnts[i]);
        }
        out.col_order_ = {"value", "count"};
        out.col_types_ = {{"value", "string"}, {"count", "int64"}};
        out.str_cols_["value"] = StringArray::from_strvec(std::move(sk));
        out.df_.load_column<int64_t>("count", std::move(sc));
        return out;
    }

    // Sorted unique values for a column.
    py::object unique_values(const std::string &col) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            std::set<double> s(v.begin(), v.end());
            py::array_t<double> r(static_cast<py::ssize_t>(s.size()));
            std::copy(s.begin(), s.end(), static_cast<double *>(r.request().ptr));
            return r;
        }
        if (type == "int64")
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            std::set<int64_t> s(v.begin(), v.end());
            py::array_t<int64_t> r(static_cast<py::ssize_t>(s.size()));
            std::copy(s.begin(), s.end(), static_cast<int64_t *>(r.request().ptr));
            return r;
        }
        py::list lst;
        if (type == "string")
        {
            const StringArray &sa = str_cols_.at(col);
            std::set<std::string> s;
            for (size_t i = 0; i < sa.size(); ++i)
                s.insert(sa.str(i));
            for (const auto &x : s)
                lst.append(py::str(x));
        }
        else
        {
            const auto &v = df_.get_column<bool>(col.c_str());
            bool ht = false, hf = false;
            for (bool b : v)
            {
                if (b)
                    ht = true;
                else
                    hf = true;
            }
            if (hf)
                lst.append(py::bool_(false));
            if (ht)
                lst.append(py::bool_(true));
        }
        return lst;
    }

    size_t nunique(const std::string &col) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            return std::set<double>(v.begin(), v.end()).size();
        }
        if (type == "int64")
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            return std::set<int64_t>(v.begin(), v.end()).size();
        }
        if (type == "string")
        {
            const StringArray &sa = str_cols_.at(col);
            std::unordered_set<std::string_view> s;
            for (size_t i = 0; i < sa.size(); ++i)
                s.insert(sa.view(i));
            return s.size();
        }
        const auto &v = df_.get_column<bool>(col.c_str());
        bool ht = false, hf = false;
        for (bool b : v)
        {
            if (b)
                ht = true;
            else
                hf = true;
        }
        return static_cast<size_t>(ht) + static_cast<size_t>(hf);
    }

    size_t n_missing(const std::string &col) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        if (it->second == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            return static_cast<size_t>(std::count_if(v.begin(), v.end(), [](double x)
                                                     { return std::isnan(x); }));
        }
        if (it->second == "string")
        {
            const StringArray &sa = str_cols_.at(col);
            size_t cnt = 0;
            for (size_t i = 0; i < sa.size(); ++i)
                if (sa.view(i).empty()) ++cnt;
            return cnt;
        }
        return 0;
    }

    // ── I/O ──────────────────────────────────────────────────────────────────

    void to_csv(const std::string &path, bool write_index = true) const
    {
        std::ofstream out(path);
        if (!out)
            throw std::runtime_error("Cannot open for writing: " + path);
        bool first = true;
        if (write_index)
        {
            out << "index";
            first = false;
        }
        for (const auto &c : col_order_)
        {
            if (!first)
                out << ',';
            out << c;
            first = false;
        }
        out << '\n';
        const auto &idx = df_.get_index();
        size_t nrows = idx.size();
        for (size_t i = 0; i < nrows; ++i)
        {
            first = true;
            if (write_index)
            {
                out << idx[i];
                first = false;
            }
            for (const auto &c : col_order_)
            {
                if (!first)
                    out << ',';
                write_cell(out, c, i);
                first = false;
            }
            out << '\n';
        }
    }

    // take_rows — select rows by integer position in arbitrary order (fast path for sklearn)
    GrizzlarFrame take_rows(const std::vector<int64_t> &indices) const
    {
        size_t n = indices.size();
        size_t src_n = df_.get_index().size();
        GrizzlarFrame out;
        const auto &src_idx = df_.get_index();
        std::vector<ulong> new_idx(n);
        for (size_t j = 0; j < n; ++j)
        {
            int64_t raw = indices[j];
            size_t pos = (raw >= 0) ? static_cast<size_t>(raw)
                                    : static_cast<size_t>(static_cast<int64_t>(src_n) + raw);
            new_idx[j] = src_idx[pos];
        }
        out.df_.load_index(std::move(new_idx));
        for (const auto &name : col_order_)
        {
            const std::string &type = col_types_.at(name);
            out.col_types_[name] = type;
            out.col_order_.push_back(name);
            if (type == "double")
            {
                const auto &v = df_.get_column<double>(name.c_str());
                std::vector<double> nv(n);
                for (size_t j = 0; j < n; ++j) { int64_t r = indices[j]; size_t p = r >= 0 ? (size_t)r : (size_t)((int64_t)src_n + r); nv[j] = v[p]; }
                out.df_.load_column<double>(name.c_str(), std::move(nv));
            }
            else if (type == "int64")
            {
                const auto &v = df_.get_column<int64_t>(name.c_str());
                std::vector<int64_t> nv(n);
                for (size_t j = 0; j < n; ++j) { int64_t r = indices[j]; size_t p = r >= 0 ? (size_t)r : (size_t)((int64_t)src_n + r); nv[j] = v[p]; }
                out.df_.load_column<int64_t>(name.c_str(), std::move(nv));
            }
            else if (type == "bool")
            {
                const auto &v = df_.get_column<bool>(name.c_str());
                std::vector<bool> nv(n);
                for (size_t j = 0; j < n; ++j) { int64_t r = indices[j]; size_t p = r >= 0 ? (size_t)r : (size_t)((int64_t)src_n + r); nv[j] = v[p]; }
                out.df_.load_column<bool>(name.c_str(), std::move(nv));
            }
            else
            {
                std::vector<size_t> locs(n);
                for (size_t j = 0; j < n; ++j) { int64_t r = indices[j]; locs[j] = r >= 0 ? (size_t)r : (size_t)((int64_t)src_n + r); }
                out.str_cols_[name] = str_cols_.at(name).gather(locs.data(), n);
            }
        }
        return out;
    }

    // ── new bulk operations ──────────────────────────────────────────────────

    // 1. isna_frame — boolean GrizzlarFrame, True where value is NaN/empty
    GrizzlarFrame isna_frame() const
    {
        GrizzlarFrame out;
        const auto &src_idx = df_.get_index();
        std::vector<ulong> new_idx(src_idx.begin(), src_idx.end());
        out.df_.load_index(std::move(new_idx));
        for (const auto &name : col_order_)
        {
            out.col_order_.push_back(name);
            out.col_types_[name] = "bool";
            const std::string &type = col_types_.at(name);
            size_t n = df_.get_index().size();
            std::vector<bool> bv(n, false);
            if (type == "double")
            {
                const auto &v = df_.get_column<double>(name.c_str());
                for (size_t i = 0; i < v.size(); ++i)
                    bv[i] = std::isnan(v[i]);
            }
            else if (type == "string")
            {
                const StringArray &sa = str_cols_.at(name);
                for (size_t i = 0; i < sa.size(); ++i)
                    bv[i] = sa.view(i).empty();
            }
            // int64/bool: always false
            out.df_.load_column<bool>(name.c_str(), std::move(bv));
        }
        return out;
    }

    // 2. notna_frame — logical inverse of isna_frame
    GrizzlarFrame notna_frame() const
    {
        GrizzlarFrame out;
        const auto &src_idx = df_.get_index();
        std::vector<ulong> new_idx(src_idx.begin(), src_idx.end());
        out.df_.load_index(std::move(new_idx));
        for (const auto &name : col_order_)
        {
            out.col_order_.push_back(name);
            out.col_types_[name] = "bool";
            const std::string &type = col_types_.at(name);
            size_t n = df_.get_index().size();
            std::vector<bool> bv(n, true);
            if (type == "double")
            {
                const auto &v = df_.get_column<double>(name.c_str());
                for (size_t i = 0; i < v.size(); ++i)
                    bv[i] = !std::isnan(v[i]);
            }
            else if (type == "string")
            {
                const StringArray &sa = str_cols_.at(name);
                for (size_t i = 0; i < sa.size(); ++i)
                    bv[i] = !sa.view(i).empty();
            }
            // int64/bool: always true
            out.df_.load_column<bool>(name.c_str(), std::move(bv));
        }
        return out;
    }

    // 3. ffill_col — forward-fill NaN/empty in-place
    void ffill_col(const std::string &col)
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        if (it->second == "double")
        {
            auto &v = df_.get_column<double>(col.c_str());
            double last = std::numeric_limits<double>::quiet_NaN();
            for (auto &x : v)
            {
                if (!std::isnan(x))
                    last = x;
                else if (!std::isnan(last))
                    x = last;
            }
        }
        else if (it->second == "string")
        {
            StringArray &sa = str_cols_.at(col);
            StringArray out;
            std::string_view last;
            for (size_t i = 0; i < sa.size(); ++i)
            {
                auto v = sa.view(i);
                if (!v.empty()) last = v;
                else if (!last.empty()) v = last;
                out.push_back(v);
            }
            sa = std::move(out);
        }
    }

    // 4. bfill_col — backward-fill NaN/empty in-place
    void bfill_col(const std::string &col)
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        if (it->second == "double")
        {
            auto &v = df_.get_column<double>(col.c_str());
            double nxt = std::numeric_limits<double>::quiet_NaN();
            for (int64_t i = static_cast<int64_t>(v.size()) - 1; i >= 0; --i)
            {
                if (!std::isnan(v[i]))
                    nxt = v[i];
                else if (!std::isnan(nxt))
                    v[i] = nxt;
            }
        }
        else if (it->second == "string")
        {
            StringArray &sa = str_cols_.at(col);
            const int64_t sz = static_cast<int64_t>(sa.size());
            std::vector<std::string_view> vals(static_cast<size_t>(sz));
            std::string_view nxt;
            for (int64_t i = sz - 1; i >= 0; --i)
            {
                auto v = sa.view(static_cast<size_t>(i));
                if (!v.empty()) nxt = v;
                vals[static_cast<size_t>(i)] = !v.empty() ? v : nxt;
            }
            StringArray out;
            for (auto sv : vals) out.push_back(sv);
            sa = std::move(out);
        }
    }

    // 5. clip_col — clip double/int64 values to [lower, upper] in-place
    void clip_col(const std::string &col, double lower, double upper)
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        if (it->second == "double")
        {
            auto &v = df_.get_column<double>(col.c_str());
            for (auto &x : v)
                if (!std::isnan(x))
                    x = std::max(lower, std::min(upper, x));
        }
        else if (it->second == "int64")
        {
            auto &v = df_.get_column<int64_t>(col.c_str());
            for (auto &x : v)
                x = static_cast<int64_t>(std::max(lower, std::min(upper, static_cast<double>(x))));
        }
    }

    // 6. round_col — round double column to decimals places in-place
    void round_col(const std::string &col, int decimals)
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        if (it->second == "double")
        {
            double factor = std::pow(10.0, decimals);
            auto &v = df_.get_column<double>(col.c_str());
            for (auto &x : v)
                if (!std::isnan(x))
                    x = std::round(x * factor) / factor;
        }
    }

    // 7. abs_col — absolute value of double/int64 column in-place
    void abs_col(const std::string &col)
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        if (it->second == "double")
        {
            auto &v = df_.get_column<double>(col.c_str());
            for (auto &x : v)
                x = std::abs(x);
        }
        else if (it->second == "int64")
        {
            auto &v = df_.get_column<int64_t>(col.c_str());
            for (auto &x : v)
                x = std::abs(x);
        }
    }

    // 8. diff_col — discrete difference
    std::vector<double> diff_col(const std::string &col, int periods) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const double nan = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> result;
        if (it->second == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            result.resize(v.size(), nan);
            for (size_t i = static_cast<size_t>(periods); i < v.size(); ++i)
                result[i] = v[i] - v[i - static_cast<size_t>(periods)];
        }
        else if (it->second == "int64")
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            result.resize(v.size(), nan);
            for (size_t i = static_cast<size_t>(periods); i < v.size(); ++i)
                result[i] = static_cast<double>(v[i]) - static_cast<double>(v[i - static_cast<size_t>(periods)]);
        }
        return result;
    }

    // 9. isin_col — boolean membership test
    std::vector<bool> isin_col(const std::string &col, py::object values) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        std::vector<bool> result;

        if (type == "double")
        {
            std::unordered_set<double> val_set;
            for (auto item : py::cast<py::iterable>(values))
                val_set.insert(py::cast<double>(item));
            const auto &v = df_.get_column<double>(col.c_str());
            result.reserve(v.size());
            for (const auto &x : v)
                result.push_back(val_set.count(x) > 0);
        }
        else if (type == "int64")
        {
            std::unordered_set<int64_t> val_set;
            for (auto item : py::cast<py::iterable>(values))
                val_set.insert(py::cast<int64_t>(item));
            const auto &v = df_.get_column<int64_t>(col.c_str());
            result.reserve(v.size());
            for (const auto &x : v)
                result.push_back(val_set.count(x) > 0);
        }
        else if (type == "string")
        {
            std::unordered_set<std::string> val_set;
            for (auto item : py::cast<py::iterable>(values))
                val_set.insert(py::cast<std::string>(item));
            const StringArray &sa = str_cols_.at(col);
            result.reserve(sa.size());
            for (size_t i = 0; i < sa.size(); ++i)
                result.push_back(val_set.count(std::string(sa.view(i))) > 0);
        }
        else
        {
            std::unordered_set<int> val_set;
            for (auto item : py::cast<py::iterable>(values))
                val_set.insert(py::cast<bool>(item) ? 1 : 0);
            const auto &v = df_.get_column<bool>(col.c_str());
            result.reserve(v.size());
            for (bool x : v)
                result.push_back(val_set.count(x ? 1 : 0) > 0);
        }
        return result;
    }

    // 10. replace_col — replace values in one column via dict mapping
    void replace_col(const std::string &col, py::dict mapping)
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        if (type == "double")
        {
            std::unordered_map<double, double> m;
            for (auto item : mapping)
                m[py::cast<double>(item.first)] = py::cast<double>(item.second);
            auto &v = df_.get_column<double>(col.c_str());
            for (auto &x : v)
            {
                auto mi = m.find(x);
                if (mi != m.end())
                    x = mi->second;
            }
        }
        else if (type == "int64")
        {
            std::unordered_map<int64_t, int64_t> m;
            for (auto item : mapping)
                m[py::cast<int64_t>(item.first)] = py::cast<int64_t>(item.second);
            auto &v = df_.get_column<int64_t>(col.c_str());
            for (auto &x : v)
            {
                auto mi = m.find(x);
                if (mi != m.end())
                    x = mi->second;
            }
        }
        else if (type == "string")
        {
            std::unordered_map<std::string, std::string> m;
            for (auto item : mapping)
                m[py::cast<std::string>(item.first)] = py::cast<std::string>(item.second);
            str_cols_[col] = str_cols_.at(col).with_replace(m);
        }
    }

    // 11. replace_all_cols — replace_col applied to all columns
    void replace_all_cols(py::dict mapping)
    {
        for (const auto &name : col_order_)
        {
            try { replace_col(name, mapping); }
            catch (...) {}
        }
    }

    // 12. reduce_all — apply reduction function to all numeric columns, return 1-row frame
    GrizzlarFrame reduce_all(const std::string &func) const
    {
        GrizzlarFrame out;
        std::vector<ulong> idx = {0};
        out.df_.load_index(std::move(idx));
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (const auto &name : col_order_)
        {
            const std::string &type = col_types_.at(name);
            if (type != "double" && type != "int64" && type != "bool")
                continue;
            out.col_order_.push_back(name);
            out.col_types_[name] = "double";
            double val = nan;
            if (type == "bool")
            {
                const auto &bv = df_.get_column<bool>(name.c_str());
                size_t n = bv.size();
                if (n == 0) { out.df_.load_column<double>(name.c_str(), {val}); continue; }
                size_t trues = 0;
                for (bool b : bv) trues += b ? 1 : 0;
                if (func == "sum" || func == "count") val = static_cast<double>(trues);
                else if (func == "mean") val = static_cast<double>(trues) / n;
                else if (func == "min") val = trues > 0 ? 0.0 : 0.0; // at least one false
                else if (func == "max") val = trues == n ? 1.0 : (trues > 0 ? 1.0 : 0.0);
                else if (func == "std")
                {
                    double m = static_cast<double>(trues) / n;
                    double sq = 0;
                    for (bool b : bv) { double d = (b ? 1.0 : 0.0) - m; sq += d * d; }
                    val = n > 1 ? std::sqrt(sq / (n - 1)) : 0.0;
                }
                else if (func == "median")
                {
                    // sorted bool: 0..0..1..1 — median is just the middle value
                    std::vector<double> sorted_v(n);
                    for (size_t i = 0; i < n; ++i) sorted_v[i] = bv[i] ? 1.0 : 0.0;
                    std::sort(sorted_v.begin(), sorted_v.end());
                    double pos = 0.5 * (n - 1);
                    size_t lo = static_cast<size_t>(pos);
                    val = sorted_v[lo] + (pos - lo) * (lo + 1 < n ? sorted_v[lo+1] - sorted_v[lo] : 0.0);
                }
                else if (func == "var")
                {
                    double m = static_cast<double>(trues) / n;
                    double sq = 0;
                    for (bool b : bv) { double d = (b ? 1.0 : 0.0) - m; sq += d * d; }
                    val = n > 1 ? sq / (n - 1) : 0.0;
                }
            }
            else
            {
                // Use existing scalar methods (which handle double and int64)
                // We need a non-const self here; cast away const for these read-only visitors
                GrizzlarFrame *self = const_cast<GrizzlarFrame *>(this);
                if (func == "sum") val = self->sum(name);
                else if (func == "mean") val = self->mean(name);
                else if (func == "std") val = self->std_dev(name);
                else if (func == "min") val = self->col_min(name);
                else if (func == "max") val = self->col_max(name);
                else if (func == "count") val = static_cast<double>(self->count(name));
                else if (func == "median")
                {
                    std::vector<double> vals;
                    if (type == "double")
                    {
                        const auto &v = df_.get_column<double>(name.c_str());
                        for (double x : v) if (!std::isnan(x)) vals.push_back(x);
                    }
                    else
                    {
                        const auto &v = df_.get_column<int64_t>(name.c_str());
                        for (int64_t x : v) vals.push_back(static_cast<double>(x));
                    }
                    if (!vals.empty())
                    {
                        std::sort(vals.begin(), vals.end());
                        double pos = 0.5 * (vals.size() - 1);
                        size_t lo = static_cast<size_t>(pos);
                        val = vals[lo] + (pos - lo) * (lo + 1 < vals.size() ? vals[lo+1] - vals[lo] : 0.0);
                    }
                }
                else if (func == "var")
                {
                    double m = self->mean(name);
                    std::vector<double> vals;
                    if (type == "double")
                    {
                        const auto &v = df_.get_column<double>(name.c_str());
                        for (double x : v) if (!std::isnan(x)) vals.push_back(x);
                    }
                    else
                    {
                        const auto &v = df_.get_column<int64_t>(name.c_str());
                        for (int64_t x : v) vals.push_back(static_cast<double>(x));
                    }
                    size_t n = vals.size();
                    if (n > 1)
                    {
                        double sq = 0;
                        for (double x : vals) { double d = x - m; sq += d * d; }
                        val = sq / (n - 1);
                    }
                    else val = 0.0;
                }
            }
            out.df_.load_column<double>(name.c_str(), {val});
        }
        return out;
    }

    // 13. arith_scalar — arithmetic op with a scalar, returns new frame
    GrizzlarFrame arith_scalar(const std::string &op, double scalar) const
    {
        GrizzlarFrame out = deep_copy();
        for (const auto &name : col_order_)
        {
            const std::string &type = col_types_.at(name);
            if (type == "double")
            {
                const auto &src = df_.get_column<double>(name.c_str());
                std::vector<double> nv(src.size());
                for (size_t i = 0; i < src.size(); ++i)
                {
                    if (op == "+")       nv[i] = src[i] + scalar;
                    else if (op == "-")  nv[i] = src[i] - scalar;
                    else if (op == "*")  nv[i] = src[i] * scalar;
                    else if (op == "/")  nv[i] = src[i] / scalar;
                    else if (op == "//") nv[i] = std::floor(src[i] / scalar);
                    else if (op == "%")  nv[i] = std::fmod(src[i], scalar);
                    else if (op == "**") nv[i] = std::pow(src[i], scalar);
                    else nv[i] = src[i];
                }
                out.df_.load_column<double>(name.c_str(), std::move(nv));
            }
            else if (type == "int64")
            {
                const auto &src = df_.get_column<int64_t>(name.c_str());
                std::vector<double> nv(src.size());
                for (size_t i = 0; i < src.size(); ++i)
                {
                    double s = static_cast<double>(src[i]);
                    if (op == "+")       nv[i] = s + scalar;
                    else if (op == "-")  nv[i] = s - scalar;
                    else if (op == "*")  nv[i] = s * scalar;
                    else if (op == "/")  nv[i] = s / scalar;
                    else if (op == "//") nv[i] = std::floor(s / scalar);
                    else if (op == "%")  nv[i] = std::fmod(s, scalar);
                    else if (op == "**") nv[i] = std::pow(s, scalar);
                    else nv[i] = s;
                }
                out.col_types_[name] = "double";
                out.df_.remove_column<int64_t>(name.c_str());
                out.df_.load_column<double>(name.c_str(), std::move(nv));
            }
        }
        return out;
    }

    // 14. arith_frame_op — element-wise arithmetic between matching numeric columns
    GrizzlarFrame arith_frame_op(const std::string &op, const GrizzlarFrame &other) const
    {
        GrizzlarFrame out = deep_copy();
        for (const auto &name : col_order_)
        {
            const std::string &type = col_types_.at(name);
            if (type != "double" && type != "int64")
                continue;
            auto ot = other.col_types_.find(name);
            if (ot == other.col_types_.end())
                continue;
            if (ot->second != "double" && ot->second != "int64")
                continue;

            auto to_dbl = [](const GrizzlarFrame &f, const std::string &c, const std::string &t) -> std::vector<double>
            {
                if (t == "double")
                {
                    const auto &v = f.df_.get_column<double>(c.c_str());
                    return {v.begin(), v.end()};
                }
                const auto &v = f.df_.get_column<int64_t>(c.c_str());
                std::vector<double> r; r.reserve(v.size());
                for (auto x : v) r.push_back(static_cast<double>(x));
                return r;
            };

            auto a = to_dbl(*this, name, type);
            auto b = to_dbl(other, name, ot->second);
            size_t n = std::min(a.size(), b.size());
            std::vector<double> nv(a.size(), std::numeric_limits<double>::quiet_NaN());
            for (size_t i = 0; i < n; ++i)
            {
                if (op == "+")       nv[i] = a[i] + b[i];
                else if (op == "-")  nv[i] = a[i] - b[i];
                else if (op == "*")  nv[i] = a[i] * b[i];
                else if (op == "/")  nv[i] = a[i] / b[i];
                else if (op == "//") nv[i] = std::floor(a[i] / b[i]);
                else if (op == "%")  nv[i] = std::fmod(a[i], b[i]);
                else if (op == "**") nv[i] = std::pow(a[i], b[i]);
                else nv[i] = a[i];
            }
            if (type == "double")
                out.df_.load_column<double>(name.c_str(), std::move(nv));
            else
            {
                out.col_types_[name] = "double";
                out.df_.remove_column<int64_t>(name.c_str());
                out.df_.load_column<double>(name.c_str(), std::move(nv));
            }
        }
        return out;
    }

    // 15. compare_scalar — compare each numeric column value to scalar, returns bool frame
    GrizzlarFrame compare_scalar(const std::string &op, double scalar) const
    {
        GrizzlarFrame out;
        const auto &src_idx = df_.get_index();
        std::vector<ulong> new_idx(src_idx.begin(), src_idx.end());
        out.df_.load_index(std::move(new_idx));
        for (const auto &name : col_order_)
        {
            out.col_order_.push_back(name);
            const std::string &type = col_types_.at(name);
            size_t n = df_.get_index().size();
            std::vector<bool> bv(n, false);

            auto cmp = [&](double x) -> bool {
                if (op == "==")       return x == scalar;
                else if (op == "!=")  return x != scalar;
                else if (op == ">")   return x > scalar;
                else if (op == ">=")  return x >= scalar;
                else if (op == "<")   return x < scalar;
                else if (op == "<=")  return x <= scalar;
                return false;
            };

            if (type == "double")
            {
                const auto &v = df_.get_column<double>(name.c_str());
                for (size_t i = 0; i < v.size(); ++i)
                    bv[i] = cmp(v[i]);
            }
            else if (type == "int64")
            {
                const auto &v = df_.get_column<int64_t>(name.c_str());
                for (size_t i = 0; i < v.size(); ++i)
                    bv[i] = cmp(static_cast<double>(v[i]));
            }
            out.col_types_[name] = "bool";
            out.df_.load_column<bool>(name.c_str(), std::move(bv));
        }
        return out;
    }

    // 16. skew_col — sample skewness
    double skew_col(const std::string &col) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        std::vector<double> vals;
        if (it->second == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            for (double x : v) if (!std::isnan(x)) vals.push_back(x);
        }
        else if (it->second == "int64")
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            for (int64_t x : v) vals.push_back(static_cast<double>(x));
        }
        else return std::numeric_limits<double>::quiet_NaN();

        const size_t n = vals.size();
        if (n < 3) return std::numeric_limits<double>::quiet_NaN();
        double mu = std::accumulate(vals.begin(), vals.end(), 0.0) / n;
        double sq = 0, cu = 0;
        for (double x : vals) { double d = x - mu; sq += d * d; cu += d * d * d; }
        double s = std::sqrt(sq / (n - 1));
        if (s == 0) return std::numeric_limits<double>::quiet_NaN();
        return (static_cast<double>(n) / ((n-1.0) * (n-2.0))) * (cu / (s * s * s));
    }

    // 17. kurt_col — excess kurtosis
    double kurt_col(const std::string &col) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        std::vector<double> vals;
        if (it->second == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            for (double x : v) if (!std::isnan(x)) vals.push_back(x);
        }
        else if (it->second == "int64")
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            for (int64_t x : v) vals.push_back(static_cast<double>(x));
        }
        else return std::numeric_limits<double>::quiet_NaN();

        const size_t n = vals.size();
        if (n < 4) return std::numeric_limits<double>::quiet_NaN();
        double mu = std::accumulate(vals.begin(), vals.end(), 0.0) / n;
        double sq = 0, qu = 0;
        for (double x : vals) { double d = x - mu; sq += d * d; qu += d * d * d * d; }
        double s = std::sqrt(sq / (n - 1));
        if (s == 0) return std::numeric_limits<double>::quiet_NaN();
        double k4 = (static_cast<double>(n) * (n + 1.0)) / ((n-1.0) * (n-2.0) * (n-3.0)) * (qu / (s * s * s * s));
        double corr = 3.0 * (n - 1.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0));
        return k4 - corr;
    }

    // 18. mode_col — most frequent value(s), skipping NaN/""
    py::list mode_col(const std::string &col) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        py::list result;

        auto compute_mode_str = [&](auto &vec)
        {
            std::map<std::string, int64_t> cnt;
            for (const auto &x : vec)
            {
                std::string key;
                if constexpr (std::is_same_v<std::decay_t<decltype(x)>, std::string>)
                {
                    if (x.empty()) continue;
                    key = x;
                }
                else
                {
                    key = std::to_string(x);
                }
                cnt[key]++;
            }
            if (cnt.empty()) return;
            int64_t max_cnt = std::max_element(cnt.begin(), cnt.end(),
                [](const auto &a, const auto &b){ return a.second < b.second; })->second;
            for (const auto &[k, c] : cnt)
                if (c == max_cnt) result.append(py::str(k));
        };

        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            std::map<double, int64_t> cnt;
            for (double x : v) { if (!std::isnan(x)) cnt[x]++; }
            if (!cnt.empty())
            {
                int64_t mc = std::max_element(cnt.begin(), cnt.end(),
                    [](const auto &a, const auto &b){ return a.second < b.second; })->second;
                for (const auto &[k, c] : cnt)
                    if (c == mc) result.append(py::float_(k));
            }
        }
        else if (type == "int64")
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            std::map<int64_t, int64_t> cnt;
            for (int64_t x : v) cnt[x]++;
            if (!cnt.empty())
            {
                int64_t mc = std::max_element(cnt.begin(), cnt.end(),
                    [](const auto &a, const auto &b){ return a.second < b.second; })->second;
                for (const auto &[k, c] : cnt)
                    if (c == mc) result.append(py::int_(k));
            }
        }
        else if (type == "string")
        {
            auto sv = str_cols_.at(col).to_strvec();
            compute_mode_str(sv);
        }
        return result;
    }

    // 19. duplicated_rows — mark duplicate rows
    std::vector<bool> duplicated_rows(const std::vector<std::string> &cols, const std::string &keep) const
    {
        size_t n = df_.get_index().size();
        std::vector<bool> result(n, false);

        // Build per-row string keys
        auto row_key = [&](size_t i) -> std::string
        {
            std::string k;
            for (const auto &c : cols)
            {
                auto it = col_types_.find(c);
                if (it == col_types_.end()) continue;
                const std::string &type = it->second;
                if (type == "double")
                {
                    const auto &v = df_.get_column<double>(c.c_str());
                    k += std::to_string(v[i]) + "|";
                }
                else if (type == "int64")
                {
                    const auto &v = df_.get_column<int64_t>(c.c_str());
                    k += std::to_string(v[i]) + "|";
                }
                else if (type == "string")
                {
                    k += std::string(str_cols_.at(c).view(i)) + "|";
                }
                else
                {
                    const auto &v = df_.get_column<bool>(c.c_str());
                    k += (v[i] ? "1" : "0") + std::string("|");
                }
            }
            return k;
        };

        if (keep == "last")
        {
            std::unordered_map<std::string, size_t> seen; // key -> last seen position
            for (size_t i = 0; i < n; ++i)
            {
                std::string k = row_key(i);
                auto it = seen.find(k);
                if (it != seen.end())
                {
                    result[it->second] = true;  // mark previous as duplicate
                    it->second = i;             // update to current
                }
                else
                    seen[k] = i;
            }
        }
        else if (keep == "false")
        {
            std::unordered_map<std::string, std::vector<size_t>> positions;
            for (size_t i = 0; i < n; ++i)
                positions[row_key(i)].push_back(i);
            for (const auto &[k, pos] : positions)
                if (pos.size() > 1)
                    for (size_t p : pos) result[p] = true;
        }
        else // "first" (default)
        {
            std::unordered_set<std::string> seen;
            for (size_t i = 0; i < n; ++i)
            {
                std::string k = row_key(i);
                if (!seen.insert(k).second)
                    result[i] = true;
            }
        }
        return result;
    }

    // 20. melt_frame — unpivot wide to long
    GrizzlarFrame melt_frame(const std::vector<std::string> &id_cols,
                              const std::vector<std::string> &val_cols,
                              const std::string &var_name,
                              const std::string &value_name) const
    {
        size_t n = df_.get_index().size();
        size_t out_n = n * val_cols.size();

        GrizzlarFrame out;
        std::vector<ulong> new_idx(out_n);
        std::iota(new_idx.begin(), new_idx.end(), 0);
        out.df_.load_index(std::move(new_idx));

        // id columns
        for (const auto &ic : id_cols)
        {
            out.col_order_.push_back(ic);
            const std::string &type = col_types_.at(ic);
            out.col_types_[ic] = type;
            if (type == "double")
            {
                const auto &src = df_.get_column<double>(ic.c_str());
                std::vector<double> nv; nv.reserve(out_n);
                for (size_t vc_i = 0; vc_i < val_cols.size(); ++vc_i)
                    for (size_t r = 0; r < n; ++r) nv.push_back(src[r]);
                out.df_.load_column<double>(ic.c_str(), std::move(nv));
            }
            else if (type == "int64")
            {
                const auto &src = df_.get_column<int64_t>(ic.c_str());
                std::vector<int64_t> nv; nv.reserve(out_n);
                for (size_t vc_i = 0; vc_i < val_cols.size(); ++vc_i)
                    for (size_t r = 0; r < n; ++r) nv.push_back(src[r]);
                out.df_.load_column<int64_t>(ic.c_str(), std::move(nv));
            }
            else if (type == "bool")
            {
                const auto &src = df_.get_column<bool>(ic.c_str());
                std::vector<bool> nv; nv.reserve(out_n);
                for (size_t vc_i = 0; vc_i < val_cols.size(); ++vc_i)
                    for (size_t r = 0; r < n; ++r) nv.push_back(src[r]);
                out.df_.load_column<bool>(ic.c_str(), std::move(nv));
            }
            else
            {
                const StringArray &src = str_cols_.at(ic);
                StringArray nv;
                for (size_t vc_i = 0; vc_i < val_cols.size(); ++vc_i)
                    for (size_t r = 0; r < n; ++r) nv.push_back(src.view(r));
                out.str_cols_[ic] = std::move(nv);
            }
        }

        // variable column
        out.col_order_.push_back(var_name);
        out.col_types_[var_name] = "string";
        std::vector<std::string> var_col; var_col.reserve(out_n);
        for (const auto &vc : val_cols)
            for (size_t r = 0; r < n; ++r) var_col.push_back(vc);
        out.str_cols_[var_name] = StringArray::from_strvec(std::move(var_col));

        // value column — use double as common type
        out.col_order_.push_back(value_name);
        out.col_types_[value_name] = "double";
        std::vector<double> val_col_data; val_col_data.reserve(out_n);
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (const auto &vc : val_cols)
        {
            auto vt = col_types_.find(vc);
            const std::string &vtype = vt != col_types_.end() ? vt->second : "double";
            if (vtype == "double")
            {
                const auto &src = df_.get_column<double>(vc.c_str());
                for (size_t r = 0; r < n; ++r) val_col_data.push_back(src[r]);
            }
            else if (vtype == "int64")
            {
                const auto &src = df_.get_column<int64_t>(vc.c_str());
                for (size_t r = 0; r < n; ++r) val_col_data.push_back(static_cast<double>(src[r]));
            }
            else if (vtype == "bool")
            {
                const auto &src = df_.get_column<bool>(vc.c_str());
                for (size_t r = 0; r < n; ++r) val_col_data.push_back(src[r] ? 1.0 : 0.0);
            }
            else
            {
                for (size_t r = 0; r < n; ++r) val_col_data.push_back(nan);
            }
        }
        out.df_.load_column<double>(value_name.c_str(), std::move(val_col_data));
        return out;
    }

    // 21. transpose_frame — rows become columns, columns become rows
    GrizzlarFrame transpose_frame() const
    {
        const size_t n = df_.get_index().size();
        const size_t ncols = col_order_.size();
        GrizzlarFrame out;
        std::vector<ulong> new_idx(ncols);
        std::iota(new_idx.begin(), new_idx.end(), 0);
        out.df_.load_index(std::move(new_idx));

        // New column names: old index values as strings (col named "0","1","2",...)
        const auto &src_idx = df_.get_index();
        for (size_t i = 0; i < n; ++i)
        {
            std::string cname = std::to_string(src_idx[i]);
            out.col_order_.push_back(cname);
            out.col_types_[cname] = "double";
            std::vector<double> col_data(ncols, std::numeric_limits<double>::quiet_NaN());
            for (size_t j = 0; j < ncols; ++j)
            {
                const std::string &type = col_types_.at(col_order_[j]);
                if (type == "double")
                {
                    const auto &v = df_.get_column<double>(col_order_[j].c_str());
                    if (i < v.size()) col_data[j] = v[i];
                }
                else if (type == "int64")
                {
                    const auto &v = df_.get_column<int64_t>(col_order_[j].c_str());
                    if (i < v.size()) col_data[j] = static_cast<double>(v[i]);
                }
                else if (type == "bool")
                {
                    const auto &v = df_.get_column<bool>(col_order_[j].c_str());
                    if (i < v.size()) col_data[j] = v[i] ? 1.0 : 0.0;
                }
                // string: leave as NaN
            }
            out.df_.load_column<double>(cname.c_str(), std::move(col_data));
        }
        return out;
    }

    // 22. set_index_col — use column values as index
    GrizzlarFrame set_index_col(const std::string &col, bool drop) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        size_t n = df_.get_index().size();
        std::vector<ulong> new_idx(n);
        if (type == "int64")
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            for (size_t i = 0; i < n; ++i)
                new_idx[i] = static_cast<ulong>(v[i]);
        }
        else if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            for (size_t i = 0; i < n; ++i)
                new_idx[i] = static_cast<ulong>(v[i]);
        }
        else
        {
            for (size_t i = 0; i < n; ++i) new_idx[i] = static_cast<ulong>(i);
        }

        GrizzlarFrame out = deep_copy();
        out.df_.load_index(std::move(new_idx));
        if (drop)
            out.drop_column(col);
        return out;
    }

    // 23. reset_index_frame — reset index to 0..N-1
    GrizzlarFrame reset_index_frame(bool drop) const
    {
        GrizzlarFrame out = deep_copy();
        size_t n = df_.get_index().size();
        if (!drop)
        {
            const auto &old_idx = df_.get_index();
            std::vector<int64_t> idx_vals(n);
            for (size_t i = 0; i < n; ++i)
                idx_vals[i] = static_cast<int64_t>(old_idx[i]);
            // Insert at front: rebuild with "index" col first
            GrizzlarFrame rebuilt;
            std::vector<ulong> new_idx(n);
            std::iota(new_idx.begin(), new_idx.end(), 0);
            rebuilt.df_.load_index(std::move(new_idx));
            rebuilt.col_order_.push_back("index");
            rebuilt.col_types_["index"] = "int64";
            rebuilt.df_.load_column<int64_t>("index", std::move(idx_vals));
            for (const auto &nm : col_order_)
            {
                rebuilt.col_order_.push_back(nm);
                rebuilt.col_types_[nm] = col_types_.at(nm);
                const std::string &type = col_types_.at(nm);
                if (type == "double")
                    rebuilt.df_.load_column<double>(nm.c_str(), df_.get_column<double>(nm.c_str()));
                else if (type == "int64")
                    rebuilt.df_.load_column<int64_t>(nm.c_str(), df_.get_column<int64_t>(nm.c_str()));
                else if (type == "bool")
                    rebuilt.df_.load_column<bool>(nm.c_str(), df_.get_column<bool>(nm.c_str()));
                else
                    rebuilt.str_cols_[nm] = str_cols_.at(nm);
            }
            return rebuilt;
        }
        std::vector<ulong> new_idx(n);
        std::iota(new_idx.begin(), new_idx.end(), 0);
        out.df_.load_index(std::move(new_idx));
        return out;
    }

    // 24. astype_col — cast a column to target type in-place
    void astype_col(const std::string &col, const std::string &target_type)
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &src_type = it->second;
        if (src_type == target_type) return;

        size_t n = df_.get_index().size();
        const double nan = std::numeric_limits<double>::quiet_NaN();

        if (target_type == "double")
        {
            std::vector<double> nv(n, nan);
            if (src_type == "int64") { const auto &v = df_.get_column<int64_t>(col.c_str()); for (size_t i=0;i<n;++i) nv[i]=static_cast<double>(v[i]); }
            else if (src_type == "bool") { const auto &v = df_.get_column<bool>(col.c_str()); for (size_t i=0;i<n;++i) nv[i]=v[i]?1.0:0.0; }
            else if (src_type == "string") { const StringArray &sa = str_cols_.at(col); for (size_t i=0;i<n;++i) { auto s=sa.str(i); char *e; double d=std::strtod(s.c_str(),&e); nv[i]=(e!=s.c_str())?d:nan; } }
            if (src_type == "int64") df_.remove_column<int64_t>(col.c_str());
            else if (src_type == "bool") df_.remove_column<bool>(col.c_str());
            else if (src_type == "string") str_cols_.erase(col);
            df_.load_column<double>(col.c_str(), std::move(nv));
            col_types_[col] = "double";
        }
        else if (target_type == "int64")
        {
            std::vector<int64_t> nv(n, 0);
            if (src_type == "double") { const auto &v = df_.get_column<double>(col.c_str()); for (size_t i=0;i<n;++i) nv[i]=static_cast<int64_t>(v[i]); }
            else if (src_type == "bool") { const auto &v = df_.get_column<bool>(col.c_str()); for (size_t i=0;i<n;++i) nv[i]=v[i]?1:0; }
            else if (src_type == "string") { const StringArray &sa = str_cols_.at(col); for (size_t i=0;i<n;++i) { auto s=sa.str(i); char *e; long long d=std::strtoll(s.c_str(),&e,10); nv[i]=(e!=s.c_str())?static_cast<int64_t>(d):0; } }
            if (src_type == "double") df_.remove_column<double>(col.c_str());
            else if (src_type == "bool") df_.remove_column<bool>(col.c_str());
            else if (src_type == "string") str_cols_.erase(col);
            df_.load_column<int64_t>(col.c_str(), std::move(nv));
            col_types_[col] = "int64";
        }
        else if (target_type == "string")
        {
            std::vector<std::string> nv(n);
            if (src_type == "double") { const auto &v = df_.get_column<double>(col.c_str()); for (size_t i=0;i<n;++i) nv[i]=std::isnan(v[i])?"":std::to_string(v[i]); }
            else if (src_type == "int64") { const auto &v = df_.get_column<int64_t>(col.c_str()); for (size_t i=0;i<n;++i) nv[i]=std::to_string(v[i]); }
            else if (src_type == "bool") { const auto &v = df_.get_column<bool>(col.c_str()); for (size_t i=0;i<n;++i) nv[i]=v[i]?"true":"false"; }
            if (src_type == "double") df_.remove_column<double>(col.c_str());
            else if (src_type == "int64") df_.remove_column<int64_t>(col.c_str());
            else if (src_type == "bool") df_.remove_column<bool>(col.c_str());
            str_cols_[col] = StringArray::from_strvec(std::move(nv));
            col_types_[col] = "string";
        }
        else if (target_type == "bool")
        {
            std::vector<bool> nv(n, false);
            if (src_type == "double") { const auto &v = df_.get_column<double>(col.c_str()); for (size_t i=0;i<n;++i) nv[i]=(v[i]!=0.0&&!std::isnan(v[i])); }
            else if (src_type == "int64") { const auto &v = df_.get_column<int64_t>(col.c_str()); for (size_t i=0;i<n;++i) nv[i]=(v[i]!=0); }
            else if (src_type == "string") { const StringArray &sa = str_cols_.at(col); for (size_t i=0;i<n;++i) { auto sv=sa.view(i); nv[i]=(!sv.empty()&&sv!="false"&&sv!="0"); } }
            if (src_type == "double") df_.remove_column<double>(col.c_str());
            else if (src_type == "int64") df_.remove_column<int64_t>(col.c_str());
            else if (src_type == "string") str_cols_.erase(col);
            df_.load_column<bool>(col.c_str(), std::move(nv));
            col_types_[col] = "bool";
        }
    }

    // 25. where_frame — replace values where cond_frame is false with fill_val
    GrizzlarFrame where_frame(const GrizzlarFrame &cond_frame, double fill_val) const
    {
        GrizzlarFrame out = deep_copy();
        for (const auto &name : col_order_)
        {
            auto ct = cond_frame.col_types_.find(name);
            if (ct == cond_frame.col_types_.end()) continue;
            if (ct->second != "bool") continue;
            const auto &mask = cond_frame.df_.get_column<bool>(name.c_str());
            const std::string &type = col_types_.at(name);
            if (type == "double")
            {
                const auto &src = df_.get_column<double>(name.c_str());
                std::vector<double> nv(src.size());
                for (size_t i = 0; i < src.size(); ++i)
                    nv[i] = (i < mask.size() && mask[i]) ? src[i] : fill_val;
                out.df_.load_column<double>(name.c_str(), std::move(nv));
            }
            else if (type == "int64")
            {
                const auto &src = df_.get_column<int64_t>(name.c_str());
                std::vector<double> nv(src.size());
                for (size_t i = 0; i < src.size(); ++i)
                    nv[i] = (i < mask.size() && mask[i]) ? static_cast<double>(src[i]) : fill_val;
                out.col_types_[name] = "double";
                out.df_.remove_column<int64_t>(name.c_str());
                out.df_.load_column<double>(name.c_str(), std::move(nv));
            }
        }
        return out;
    }

    // 26. corr_matrix — full Pearson correlation matrix
    GrizzlarFrame corr_matrix() const
    {
        std::vector<std::string> num_cols;
        for (const auto &name : col_order_)
        {
            const std::string &t = col_types_.at(name);
            if (t == "double" || t == "int64") num_cols.push_back(name);
        }
        const size_t k = num_cols.size();
        GrizzlarFrame out;
        std::vector<ulong> new_idx(k);
        std::iota(new_idx.begin(), new_idx.end(), 0);
        out.df_.load_index(std::move(new_idx));

        // Leading label column
        out.col_order_.push_back("");
        out.col_types_[""] = "string";
        out.str_cols_[""] = StringArray::from_strvec(std::vector<std::string>(num_cols.begin(), num_cols.end()));

        for (size_t i = 0; i < k; ++i)
        {
            const std::string &ci = num_cols[i];
            out.col_order_.push_back(ci);
            out.col_types_[ci] = "double";
            std::vector<double> col_vals(k);
            for (size_t j = 0; j < k; ++j)
            {
                if (i == j) col_vals[j] = 1.0;
                else col_vals[j] = corr(ci, num_cols[j]);
            }
            out.df_.load_column<double>(ci.c_str(), std::move(col_vals));
        }
        return out;
    }

    // 27. cov_matrix — full covariance matrix
    GrizzlarFrame cov_matrix() const
    {
        std::vector<std::string> num_cols;
        for (const auto &name : col_order_)
        {
            const std::string &t = col_types_.at(name);
            if (t == "double" || t == "int64") num_cols.push_back(name);
        }
        const size_t k = num_cols.size();
        GrizzlarFrame out;
        std::vector<ulong> new_idx(k);
        std::iota(new_idx.begin(), new_idx.end(), 0);
        out.df_.load_index(std::move(new_idx));

        out.col_order_.push_back("");
        out.col_types_[""] = "string";
        out.str_cols_[""] = StringArray::from_strvec(std::vector<std::string>(num_cols.begin(), num_cols.end()));

        for (size_t i = 0; i < k; ++i)
        {
            const std::string &ci = num_cols[i];
            out.col_order_.push_back(ci);
            out.col_types_[ci] = "double";
            std::vector<double> col_vals(k);
            for (size_t j = 0; j < k; ++j)
                col_vals[j] = cov(ci, num_cols[j]);
            out.df_.load_column<double>(ci.c_str(), std::move(col_vals));
        }
        return out;
    }

    // 28. filter_by_mask_list — filter using vector<bool> (no numpy required)
    GrizzlarFrame filter_by_mask_list(const std::vector<bool> &mask) const
    {
        const size_t n = df_.get_index().size();
        if (mask.size() != n)
            throw std::runtime_error("mask length " + std::to_string(mask.size()) +
                                     " != frame length " + std::to_string(n));
        std::vector<uint8_t> m(n);
        size_t out_n = 0;
        for (size_t i = 0; i < n; ++i) { m[i] = mask[i] ? 1 : 0; out_n += m[i]; }
        if (out_n == n) return deep_copy();
        return compress_by_uint8(m.data(), n, out_n);
    }

    // 28b. describe_col / _describe_raw — one copy + one sort per column.
    //      _describe_raw is pure C++ (no pybind11 objects) so it is safe to call
    //      from threads without the GIL.  describe() launches all numeric columns
    //      in parallel and assembles the result dict after re-acquiring the GIL.

    struct DescribeStats {
        double count, mean, std_v, min_v, q25, q50, q75, max_v;
    };

    DescribeStats _describe_raw(const std::string &col) const
    {
        const std::string &type = col_types_.at(col);
        const double nan = std::numeric_limits<double>::quiet_NaN();

        std::vector<double> vals;
        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            vals.reserve(v.size());
            for (double x : v)
                if (!std::isnan(x)) vals.push_back(x);
        }
        else
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            vals.reserve(v.size());
            for (int64_t x : v) vals.push_back(static_cast<double>(x));
        }

        const size_t cnt = vals.size();
        if (cnt == 0) return {0.0, nan, nan, nan, nan, nan, nan, nan};

        double sum_v = 0, sum_sq = 0;
        double mn =  std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        for (double v : vals) { sum_v += v; sum_sq += v * v; if (v < mn) mn = v; if (v > mx) mx = v; }
        double mean_v = sum_v / cnt;
        double var    = cnt > 1 ? (sum_sq - sum_v * sum_v / cnt) / (cnt - 1) : 0.0;
        double std_v  = cnt > 1 ? std::sqrt(var) : 0.0;

        std::sort(vals.begin(), vals.end());
        auto interp = [&](double q) -> double {
            double pos = q * (cnt - 1);
            size_t lo  = static_cast<size_t>(pos);
            double frac = pos - lo;
            return (lo + 1 < cnt) ? vals[lo] + frac * (vals[lo + 1] - vals[lo]) : vals[lo];
        };
        return {static_cast<double>(cnt), mean_v, std_v, mn,
                interp(0.25), interp(0.50), interp(0.75), mx};
    }

    py::dict describe_col(const std::string &col) const
    {
        require_numeric(col);
        const auto r = _describe_raw(col);
        py::dict d;
        d["count"] = r.count; d["mean"]  = r.mean;  d["std"]  = r.std_v;
        d["min"]   = r.min_v; d["25%"]   = r.q25;   d["50%"]  = r.q50;
        d["75%"]   = r.q75;   d["max"]   = r.max_v;
        return d;
    }

    // 29. multi_stat_col — compute count/mean/std/min/max/sum in one C++ pass.
    //     Returns a Python dict.  Cuts the pybind11 call overhead from 10 to 1
    //     compared with calling mean()/sum()/std()/min()/max() separately.
    py::dict multi_stat_col(const std::string &col) const
    {
        require_numeric(col);
        const std::string &type = col_types_.at(col);
        const size_t n = df_.get_index().size();
        const double inf  =  std::numeric_limits<double>::infinity();
        const double nan  =  std::numeric_limits<double>::quiet_NaN();

        double sum_v = 0, sum_sq = 0, min_v = inf, max_v = -inf;
        size_t cnt = 0;

        // Fast path: scan for NaN first (vectorizable single pass).
        // Real-world numeric columns (sales, prices, volumes) rarely contain NaN,
        // so the no-NaN branch runs as a tight branchless loop that MSVC /arch:AVX2
        // can auto-vectorize to VADDPD / VMULPD / VMINPD / VMAXPD.
        if (type == "double")
        {
            const auto &raw = df_.get_column<double>(col.c_str());
            bool has_nan = false;
            for (size_t i = 0; i < n && !has_nan; ++i) has_nan = std::isnan(raw[i]);

            if (!has_nan)
            {
                cnt = n;
                for (size_t i = 0; i < n; ++i)
                {
                    sum_v  += raw[i];
                    sum_sq += raw[i] * raw[i];
                    if (raw[i] < min_v) min_v = raw[i];
                    if (raw[i] > max_v) max_v = raw[i];
                }
            }
            else
            {
                for (size_t i = 0; i < n; ++i)
                {
                    const double v = raw[i];
                    if (!std::isnan(v)) { ++cnt; sum_v += v; sum_sq += v * v; if (v < min_v) min_v = v; if (v > max_v) max_v = v; }
                }
            }
        }
        else // int64 — no NaN sentinel in range we expose; process all
        {
            cnt = n;
            for (int64_t v : df_.get_column<int64_t>(col.c_str()))
            {
                const double d = static_cast<double>(v);
                sum_v  += d; sum_sq += d * d;
                if (d < min_v) min_v = d;
                if (d > max_v) max_v = d;
            }
        }

        const double dcnt = static_cast<double>(cnt);
        double mean_v = cnt ? sum_v / dcnt : nan;
        double var_v  = cnt > 1 ? (sum_sq - sum_v * sum_v / dcnt) / (dcnt - 1) : 0.0;
        double std_v  = cnt > 1 ? std::sqrt(var_v) : 0.0;

        py::dict d;
        d["count"] = dcnt;
        d["mean"]  = mean_v;
        d["std"]   = std_v;
        d["min"]   = cnt ? min_v : nan;
        d["max"]   = cnt ? max_v : nan;
        d["sum"]   = sum_v;
        return d;
    }

    // 30. compare_col_scalar — return a bool mask for col op scalar entirely in C++
    //     Avoids materializing 100K Python objects for the comparison step.
    std::vector<bool> compare_col_scalar(const std::string &col,
                                         const std::string &op,
                                         double scalar) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        const size_t n = df_.get_index().size();
        std::vector<bool> mask(n, false);

        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            if (op == ">")       for (size_t i = 0; i < n; ++i) mask[i] = v[i] > scalar;
            else if (op == ">=") for (size_t i = 0; i < n; ++i) mask[i] = v[i] >= scalar;
            else if (op == "<")  for (size_t i = 0; i < n; ++i) mask[i] = v[i] < scalar;
            else if (op == "<=") for (size_t i = 0; i < n; ++i) mask[i] = v[i] <= scalar;
            else if (op == "==") for (size_t i = 0; i < n; ++i) mask[i] = v[i] == scalar;
            else if (op == "!=") for (size_t i = 0; i < n; ++i) mask[i] = v[i] != scalar;
        }
        else if (type == "int64")
        {
            int64_t s = static_cast<int64_t>(scalar);
            const auto &v = df_.get_column<int64_t>(col.c_str());
            if (op == ">")       for (size_t i = 0; i < n; ++i) mask[i] = v[i] > s;
            else if (op == ">=") for (size_t i = 0; i < n; ++i) mask[i] = v[i] >= s;
            else if (op == "<")  for (size_t i = 0; i < n; ++i) mask[i] = v[i] < s;
            else if (op == "<=") for (size_t i = 0; i < n; ++i) mask[i] = v[i] <= s;
            else if (op == "==") for (size_t i = 0; i < n; ++i) mask[i] = v[i] == s;
            else if (op == "!=") for (size_t i = 0; i < n; ++i) mask[i] = v[i] != s;
        }
        else
        {
            throw std::runtime_error("compare_col_scalar: unsupported type " + type);
        }
        return mask;
    }

    // 30. filter_col_scalar — compare column vs scalar, compress-filter all columns.
    //     Uses uint8_t mask + compress_by_uint8 — no per-row string heap allocation.
    GrizzlarFrame filter_col_scalar(const std::string &col,
                                     const std::string &op,
                                     double scalar) const
    {
        auto it = col_types_.find(col);
        if (it == col_types_.end())
            throw std::runtime_error("Column not found: " + col);
        const std::string &type = it->second;
        const size_t n = df_.get_index().size();

        std::vector<uint8_t> mask(n, 0);

        if (type == "double")
        {
            const double *v = df_.get_column<double>(col.c_str()).data();
            if (op == ">")       for (size_t i = 0; i < n; ++i) mask[i] = v[i] > scalar;
            else if (op == ">=") for (size_t i = 0; i < n; ++i) mask[i] = v[i] >= scalar;
            else if (op == "<")  for (size_t i = 0; i < n; ++i) mask[i] = v[i] < scalar;
            else if (op == "<=") for (size_t i = 0; i < n; ++i) mask[i] = v[i] <= scalar;
            else if (op == "==") for (size_t i = 0; i < n; ++i) mask[i] = v[i] == scalar;
            else if (op == "!=") for (size_t i = 0; i < n; ++i) mask[i] = v[i] != scalar;
            else throw std::runtime_error("filter_col_scalar: unknown op: " + op);
        }
        else if (type == "int64")
        {
            int64_t s = static_cast<int64_t>(scalar);
            const int64_t *v = df_.get_column<int64_t>(col.c_str()).data();
            if (op == ">")       for (size_t i = 0; i < n; ++i) mask[i] = v[i] > s;
            else if (op == ">=") for (size_t i = 0; i < n; ++i) mask[i] = v[i] >= s;
            else if (op == "<")  for (size_t i = 0; i < n; ++i) mask[i] = v[i] < s;
            else if (op == "<=") for (size_t i = 0; i < n; ++i) mask[i] = v[i] <= s;
            else if (op == "==") for (size_t i = 0; i < n; ++i) mask[i] = v[i] == s;
            else if (op == "!=") for (size_t i = 0; i < n; ++i) mask[i] = v[i] != s;
            else throw std::runtime_error("filter_col_scalar: unknown op: " + op);
        }
        else
        {
            throw std::runtime_error("filter_col_scalar: unsupported type " + type +
                                     " (only double/int64 supported)");
        }

        size_t out_n = 0;
        for (size_t i = 0; i < n; ++i) out_n += mask[i];
        if (out_n == n) return deep_copy();
        return compress_by_uint8(mask.data(), n, out_n);
    }

private:
    void write_cell(std::ofstream &out, const std::string &col, size_t row) const
    {
        const std::string &type = col_types_.at(col);
        if (type == "double")
        {
            const auto &v = df_.get_column<double>(col.c_str());
            if (row < v.size())
                out << v[row];
        }
        else if (type == "int64")
        {
            const auto &v = df_.get_column<int64_t>(col.c_str());
            if (row < v.size())
                out << v[row];
        }
        else if (type == "bool")
        {
            const auto &v = df_.get_column<bool>(col.c_str());
            if (row < v.size())
                out << (v[row] ? "true" : "false");
        }
        else
        {
            const StringArray &sa = str_cols_.at(col);
            if (row < sa.size())
            {
                auto sv = sa.view(row);
                bool nq = sv.find(',') != std::string_view::npos || sv.find('"') != std::string_view::npos || sv.find('\n') != std::string_view::npos;
                if (nq)
                {
                    out << '"';
                    for (char ch : sv)
                    {
                        if (ch == '"')
                            out << '"';
                        out << ch;
                    }
                    out << '"';
                }
                else
                    out << sv;
            }
        }
    }
};

// ─── module ──────────────────────────────────────────────────────────────────

PYBIND11_MODULE(_grizzlars, m)
{
    m.doc() = "Grizzlar: Python bindings for the hmdf C++ DataFrame library";

    py::class_<GrizzlarFrame>(m, "GrizzlarFrame")
        .def(py::init<>())
        // loading
        .def("load_index", &GrizzlarFrame::load_index, py::arg("indices"))
        .def("load_column", &GrizzlarFrame::load_column, py::arg("name"), py::arg("data"))
        // access
        .def("get_index", &GrizzlarFrame::get_index)
        .def("get_column", &GrizzlarFrame::get_column, py::arg("name"))
        .def("columns", &GrizzlarFrame::columns)
        .def("shape", &GrizzlarFrame::shape)
        .def("col_type", &GrizzlarFrame::col_type, py::arg("name"))
        .def("has_column", &GrizzlarFrame::has_column, py::arg("name"))
        // statistics
        .def("mean", &GrizzlarFrame::mean, py::arg("col"))
        .def("std", &GrizzlarFrame::std_dev, py::arg("col"))
        .def("sum", &GrizzlarFrame::sum, py::arg("col"))
        .def("min", &GrizzlarFrame::col_min, py::arg("col"))
        .def("max", &GrizzlarFrame::col_max, py::arg("col"))
        .def("count", &GrizzlarFrame::count, py::arg("col"))
        .def("describe", &GrizzlarFrame::describe)
        // advanced stats
        .def("quantile", &GrizzlarFrame::quantile, py::arg("col"), py::arg("q"))
        .def("corr", &GrizzlarFrame::corr, py::arg("col1"), py::arg("col2"))
        .def("cov", &GrizzlarFrame::cov, py::arg("col1"), py::arg("col2"))
        // time-series / window
        .def("rolling", &GrizzlarFrame::rolling, py::arg("col"), py::arg("window"), py::arg("func") = "mean")
        .def("cumulative", &GrizzlarFrame::cumulative, py::arg("col"), py::arg("func") = "sum")
        .def("shift_col", &GrizzlarFrame::shift_col, py::arg("col"), py::arg("n"))
        .def("pct_change", &GrizzlarFrame::pct_change, py::arg("col"))
        // sort
        .def("sort_by", &GrizzlarFrame::sort_by, py::arg("col"), py::arg("ascending") = true)
        .def("sort_index", &GrizzlarFrame::sort_index, py::arg("ascending") = true)
        // filter / copy
        .def("filter_by_mask", &GrizzlarFrame::filter_by_mask, py::arg("mask"))
        .def("deep_copy", &GrizzlarFrame::deep_copy)
        .def("iloc", &GrizzlarFrame::iloc, py::arg("start"), py::arg("stop"))
        .def("select_columns", &GrizzlarFrame::select_columns, py::arg("names"))
        // groupby
        .def("groupby_agg", &GrizzlarFrame::groupby_agg, py::arg("by_col"), py::arg("specs"))
        // join / concat
        .def("join_by_index", &GrizzlarFrame::join_by_index, py::arg("rhs"), py::arg("how") = "inner")
        .def("concat_frame", &GrizzlarFrame::concat_frame, py::arg("other"))
        // data cleaning
        .def("drop_duplicates", &GrizzlarFrame::drop_duplicates, py::arg("col"))
        .def("drop_na", &GrizzlarFrame::drop_na, py::arg("col"))
        .def("fillna", &GrizzlarFrame::fillna, py::arg("col"), py::arg("value"))
        .def("rename_col", &GrizzlarFrame::rename_col, py::arg("old_name"), py::arg("new_name"))
        .def("drop_column", &GrizzlarFrame::drop_column, py::arg("name"))
        // utilities
        .def("value_counts", &GrizzlarFrame::value_counts, py::arg("col"))
        .def("unique_values", &GrizzlarFrame::unique_values, py::arg("col"))
        .def("nunique", &GrizzlarFrame::nunique, py::arg("col"))
        .def("n_missing", &GrizzlarFrame::n_missing, py::arg("col"))
        // I/O
        .def("to_csv", &GrizzlarFrame::to_csv, py::arg("path"), py::arg("write_index") = true)
        // native C++ CSV loader (bypasses Python csv.DictReader for large files)
        .def_static("read_csv_native", &GrizzlarFrame::read_csv_native,
                    py::arg("path"), py::arg("index_col") = "")
        // new bulk operations
        .def("isna_frame", &GrizzlarFrame::isna_frame)
        .def("notna_frame", &GrizzlarFrame::notna_frame)
        .def("ffill_col", &GrizzlarFrame::ffill_col, py::arg("col"))
        .def("bfill_col", &GrizzlarFrame::bfill_col, py::arg("col"))
        .def("clip_col", &GrizzlarFrame::clip_col, py::arg("col"), py::arg("lower"), py::arg("upper"))
        .def("round_col", &GrizzlarFrame::round_col, py::arg("col"), py::arg("decimals"))
        .def("abs_col", &GrizzlarFrame::abs_col, py::arg("col"))
        .def("diff_col", &GrizzlarFrame::diff_col, py::arg("col"), py::arg("periods") = 1)
        .def("isin_col", &GrizzlarFrame::isin_col, py::arg("col"), py::arg("values"))
        .def("replace_col", &GrizzlarFrame::replace_col, py::arg("col"), py::arg("mapping"))
        .def("replace_all_cols", &GrizzlarFrame::replace_all_cols, py::arg("mapping"))
        .def("reduce_all", &GrizzlarFrame::reduce_all, py::arg("func"))
        .def("arith_scalar", &GrizzlarFrame::arith_scalar, py::arg("op"), py::arg("scalar"))
        .def("arith_frame_op", &GrizzlarFrame::arith_frame_op, py::arg("op"), py::arg("other"))
        .def("compare_scalar", &GrizzlarFrame::compare_scalar, py::arg("op"), py::arg("scalar"))
        .def("skew_col", &GrizzlarFrame::skew_col, py::arg("col"))
        .def("kurt_col", &GrizzlarFrame::kurt_col, py::arg("col"))
        .def("mode_col", &GrizzlarFrame::mode_col, py::arg("col"))
        .def("duplicated_rows", &GrizzlarFrame::duplicated_rows, py::arg("cols"), py::arg("keep") = "first")
        .def("melt_frame", &GrizzlarFrame::melt_frame,
             py::arg("id_cols"), py::arg("val_cols"), py::arg("var_name") = "variable", py::arg("value_name") = "value")
        .def("transpose_frame", &GrizzlarFrame::transpose_frame)
        .def("set_index_col", &GrizzlarFrame::set_index_col, py::arg("col"), py::arg("drop") = true)
        .def("reset_index_frame", &GrizzlarFrame::reset_index_frame, py::arg("drop") = false)
        .def("astype_col", &GrizzlarFrame::astype_col, py::arg("col"), py::arg("target_type"))
        .def("where_frame", &GrizzlarFrame::where_frame, py::arg("cond_frame"), py::arg("fill_val") = 0.0)
        .def("corr_matrix", &GrizzlarFrame::corr_matrix)
        .def("cov_matrix", &GrizzlarFrame::cov_matrix)
        .def("filter_by_mask_list", &GrizzlarFrame::filter_by_mask_list, py::arg("mask"))
        .def("take_rows", &GrizzlarFrame::take_rows, py::arg("indices"))
        .def("compare_col_scalar", &GrizzlarFrame::compare_col_scalar,
             py::arg("col"), py::arg("op"), py::arg("scalar"))
        .def("filter_col_scalar", &GrizzlarFrame::filter_col_scalar,
             py::arg("col"), py::arg("op"), py::arg("scalar"))
        .def("multi_stat_col", &GrizzlarFrame::multi_stat_col, py::arg("col"))
        .def("describe_col", &GrizzlarFrame::describe_col, py::arg("col"));

    // Thread-pool controls
    m.def("set_thread_level", [](long n)
          { GDF::set_thread_level(n); }, py::arg("n"), "Set the number of worker threads (0 = single-threaded).");
    m.def("set_optimum_thread_level", []()
          { GDF::set_optimum_thread_level(); }, "Enable multithreading using all logical CPU cores.");
    m.def("get_thread_level", []()
          { return GDF::get_thread_level(); }, "Return the current number of worker threads.");
}
