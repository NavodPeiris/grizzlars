// grizzlars_shim_window.cpp — cumulative, shift_col, pct_change, rolling.
#include "grizzlars_shim.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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
