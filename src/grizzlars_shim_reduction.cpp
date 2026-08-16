// grizzlars_shim_reduction.cpp — boolean-frame ops (isna/notna/where) and
// elementwise/reduction ops (arith_scalar, arith_frame_op, compare_scalar,
// reduce_all).
#include "grizzlars_shim.h"
#include "grizzlars_shim_internal.h"

#include <cmath>
#include <limits>

using grizzlars_detail::apply_op;

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
        out.df_.load_column<uint8_t>(name.c_str(), std::move(mask));
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
        out.df_.load_column<uint8_t>(name.c_str(), std::move(mask));
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

