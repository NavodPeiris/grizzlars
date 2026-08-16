// grizzlars_shim_reshape.cpp — reshaping: transpose_frame, set_index_col,
// reset_index_frame, melt_frame.
#include "grizzlars_shim.h"

#include <limits>

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
        out.df_.load_column<double>(new_name.c_str(), std::move(col));
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
        out.df_.load_column<int64_t>("index", std::move(idx_as_int64));
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
            out.df_.load_column<double>(id_col.c_str(), std::move(rep));
        }
        else if (type == "int64")
        {
            const auto & c = df_.get_column<int64_t>(id_col.c_str());
            std::vector<int64_t> rep(out_n);
            for (size_t v = 0; v < nval; ++v)
                for (size_t i = 0; i < n; ++i) rep[v * n + i] = c[i];
            out.df_.load_column<int64_t>(id_col.c_str(), std::move(rep));
        }
        else if (type == "string")
        {
            const auto & c = df_.get_column<std::string>(id_col.c_str());
            std::vector<std::string> rep(out_n);
            for (size_t v = 0; v < nval; ++v)
                for (size_t i = 0; i < n; ++i) rep[v * n + i] = c[i];
            out.df_.load_column<std::string>(id_col.c_str(), std::move(rep));
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
    out.df_.load_column<std::string>(var_name.c_str(), std::move(var_col));
    out.df_.load_column<double>(value_name.c_str(), std::move(value_col));
    out.col_order_.push_back(var_name);
    out.col_types_[var_name] = "string";
    out.col_order_.push_back(value_name);
    out.col_types_[value_name] = "double";
    return out;
}
