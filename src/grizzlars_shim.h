// grizzlars_shim.h — thin, concrete-signature wrapper over the real hmdf
// DataFrame API (cpp_lib/DataFrame, unedited upstream). Every public method
// here delegates to a real hmdf method/visitor; this header is parsed by
// litgen (a syntactic, srcML-based parser) to auto-generate nanobind glue,
// so every public method must have a concrete (non-template) signature.
//
// Supported column types: double, int64_t, bool (as uint8_t), std::string.
// Index type: unsigned long (matches hmdf's StdDataFrame<unsigned long>),
// exposed to Python as uint64_t.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <DataFrame/DataFrame.h>

class GrizzlarFrame
{
public:
    GrizzlarFrame() = default;

    // ── construction / loading ────────────────────────────────────────────
    // Taken by value (not const&): nanobind already materializes a fresh
    // std::vector<T> converting from the Python list/array regardless, so
    // taking it by value costs nothing extra and lets the implementation
    // std::move it straight into hmdf's storage instead of copying again.
    void load_index(std::vector<uint64_t> indices);
    void load_column_double(const std::string & name, std::vector<double> values);
    void load_column_int64(const std::string & name, std::vector<int64_t> values);
    void load_column_bool(const std::string & name, std::vector<uint8_t> values);
    void load_column_string(const std::string & name, std::vector<std::string> values);

    // ── accessors ──────────────────────────────────────────────────────────
    GrizzlarFrame deep_copy() const;
    std::vector<uint64_t> get_index() const;
    std::vector<std::string> columns() const;
    std::pair<size_t, size_t> shape() const;
    bool has_column(const std::string & name) const;
    std::string col_type(const std::string & name) const;
    void drop_column(const std::string & name);

    std::vector<double> get_column_double(const std::string & name) const;
    std::vector<int64_t> get_column_int64(const std::string & name) const;
    std::vector<uint8_t> get_column_bool(const std::string & name) const;
    std::vector<std::string> get_column_string(const std::string & name) const;

    // ── scalar statistics (real hmdf visitors) ────────────────────────────
    double mean(const std::string & col) const;
    double std_dev(const std::string & col) const;
    double sum(const std::string & col) const;
    double col_min(const std::string & col) const;
    double col_max(const std::string & col) const;
    double quantile(const std::string & col, double q) const;
    double corr(const std::string & col1, const std::string & col2) const;
    double cov(const std::string & col1, const std::string & col2) const;
    double skew_col(const std::string & col) const;
    double kurt_col(const std::string & col) const;
    std::vector<double> mode_col_double(const std::string & col) const;
    std::vector<int64_t> mode_col_int64(const std::string & col) const;
    std::vector<std::string> mode_col_string(const std::string & col) const;

    size_t nunique(const std::string & col) const;
    size_t n_missing(const std::string & col) const;
    size_t count(const std::string & col) const;
    std::vector<double> unique_double(const std::string & col) const;
    std::vector<int64_t> unique_int64(const std::string & col) const;
    std::vector<std::string> unique_string(const std::string & col) const;
    GrizzlarFrame value_counts_double(const std::string & col) const;
    GrizzlarFrame value_counts_int64(const std::string & col) const;
    GrizzlarFrame value_counts_string(const std::string & col) const;
    std::map<std::string, std::map<std::string, double>> describe() const;

    // ── sorting / row selection / filtering ───────────────────────────────
    GrizzlarFrame sort_by(const std::string & by, bool ascending) const;
    GrizzlarFrame sort_index(bool ascending) const;
    GrizzlarFrame iloc(int64_t start, int64_t stop) const;
    GrizzlarFrame take_rows(const std::vector<int64_t> & positions) const;
    GrizzlarFrame select_columns(const std::vector<std::string> & names) const;
    GrizzlarFrame filter_by_mask_list(const std::vector<uint8_t> & mask) const;
    GrizzlarFrame filter_col_scalar_double(const std::string & col, const std::string & op, double scalar) const;
    std::vector<uint8_t> compare_col_scalar_double(const std::string & col, const std::string & op, double scalar) const;

    // ── groupby / join / concat ────────────────────────────────────────────
    GrizzlarFrame groupby_agg(
        const std::string & by_col,
        const std::vector<std::string> & agg_cols,
        const std::vector<std::string> & agg_funcs) const;
    GrizzlarFrame join_by_index(const GrizzlarFrame & rhs, const std::string & how) const;
    GrizzlarFrame concat_frame(const GrizzlarFrame & other) const;

    // ── missing data ───────────────────────────────────────────────────────
    void fillna_double(const std::string & col, double value);
    void fillna_string(const std::string & col, const std::string & value);
    void ffill_col(const std::string & col);
    void bfill_col(const std::string & col);
    GrizzlarFrame drop_na(const std::string & col) const;
    GrizzlarFrame drop_duplicates(const std::string & col) const;

    // ── window functions ───────────────────────────────────────────────────
    std::vector<double> cumulative(const std::string & col, const std::string & func) const;
    std::vector<double> shift_col(const std::string & col, int64_t n) const;
    std::vector<double> pct_change(const std::string & col) const;
    std::vector<double> rolling(const std::string & col, size_t window, const std::string & func) const;

    // ── CSV I/O ────────────────────────────────────────────────────────────
    // A straightforward sequential reader/writer (not memory-mapped or
    // multithreaded, unlike the previous hand-rolled implementation) — hmdf's
    // real read()/write() are schema/index-centric in a way that doesn't fit
    // grizzlars' "no index column given" case cleanly, so this trades the
    // old CSV-loading performance edge for a simple, predictable path.
    void to_csv(const std::string & path, bool write_index) const;
    static GrizzlarFrame read_csv_native(const std::string & path, const std::string & index_col_name);

    // ── data cleaning ──────────────────────────────────────────────────────
    void clip_col(const std::string & col, double lower, double upper);
    void round_col(const std::string & col, int decimals);
    void abs_col(const std::string & col);
    void rename_col(const std::string & old_name, const std::string & new_name);
    void astype_col(const std::string & col, const std::string & target_type);
    void replace_col_double(const std::string & col, const std::vector<double> & from, const std::vector<double> & to);
    void replace_col_string(const std::string & col, const std::vector<std::string> & from, const std::vector<std::string> & to);
    std::vector<uint8_t> isin_col_double(const std::string & col, const std::vector<double> & values) const;
    std::vector<uint8_t> isin_col_string(const std::string & col, const std::vector<std::string> & values) const;
    std::vector<uint8_t> duplicated_rows(const std::vector<std::string> & cols, const std::string & keep) const;
    std::vector<double> diff_col(const std::string & col, int64_t periods) const;

    // ── boolean-frame / reduction ops ──────────────────────────────────────
    GrizzlarFrame isna_frame() const;
    GrizzlarFrame notna_frame() const;
    GrizzlarFrame where_frame(const GrizzlarFrame & cond_frame, double fill_val) const;
    GrizzlarFrame arith_scalar(const std::string & op, double scalar) const;
    GrizzlarFrame arith_frame_op(const std::string & op, const GrizzlarFrame & other) const;
    GrizzlarFrame compare_scalar(const std::string & op, double scalar) const;
    GrizzlarFrame reduce_all(const std::string & func) const;

    // ── reshaping ───────────────────────────────────────────────────────────
    GrizzlarFrame transpose_frame() const;
    GrizzlarFrame set_index_col(const std::string & col, bool drop) const;
    GrizzlarFrame reset_index_frame(bool drop) const;
    GrizzlarFrame melt_frame(
        const std::vector<std::string> & id_cols,
        const std::vector<std::string> & val_cols,
        const std::string & var_name,
        const std::string & value_name) const;

private:
    // Rediscovers col_order_/col_types_ from df_'s real column storage via
    // hmdf's get_columns_info() — used after operations (join, concat) whose
    // real hmdf implementation may add/rename/reorder columns in ways this
    // shim's own bookkeeping can't predict ahead of time.
    void sync_from_df();

    // Row projection shared by iloc/take_rows/filter_*: builds a new frame
    // containing only the rows at `positions` (hmdf's real get_data_by_loc,
    // which natively supports arbitrary/unsorted order and negative indices).
    GrizzlarFrame from_positions(const std::vector<long> & positions) const;

    // Reads a numeric (double or int64) column and returns it as double,
    // for the rare case two columns being paired (corr/cov) don't share a
    // single C++ type (hmdf's visitors require both columns to match).
    std::vector<double> get_column_double_or_cast(const std::string & name) const;

    hmdf::StdDataFrame<unsigned long> df_;
    std::vector<std::string> col_order_;
    std::unordered_map<std::string, std::string> col_types_;
};
