#include <boost/python.hpp>
#include <dials/model/data/shoebox.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/algorithms/profile_model/gaussian_rs/mask_calculator.h>
#include <dials/algorithms/profile_model/gaussian_rs/modeller.h>
#include <dials/algorithms/profile_model/modeller/multi_experiment_modeller.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace dials_gpu {

using dials::model::Shoebox;
using dials::ProfileFloatType;

// 1. Extract valid 3D background pixels and per-shoebox medians in parallel (for GLM constant3d)
boost::python::tuple extract_glm_constant3d_bg(
    dials::af::ref<Shoebox<ProfileFloatType>> self,
    int min_pixels,
    int mask_bg_valid,
    int mask_overlapped,
    int mask_bg_used,
    std::size_t counts_addr,
    std::size_t medians_addr,
    std::size_t has_neg_addr)
{
    int64_t* counts_out = reinterpret_cast<int64_t*>(counts_addr);
    double* medians_out = reinterpret_cast<double*>(medians_addr);
    uint8_t* has_neg_out = reinterpret_cast<uint8_t*>(has_neg_addr);
    uint8_t m_bg_val = static_cast<uint8_t>(mask_bg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    uint8_t m_used = static_cast<uint8_t>(mask_bg_used);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    #pragma omp parallel for schedule(guided)
    for (std::int64_t r = 0; r < n; ++r) {
        auto& sb = self[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            counts_out[r] = 0;
            medians_out[r] = 1.0;
            has_neg_out[r] = 0;
            continue;
        }
        const float* d_ptr = sb.data.begin();
        uint8_t* m_ptr = sb.mask.begin();
        int64_t cnt = 0;
        uint8_t neg = 0;
        for (std::size_t i = 0; i < sz; ++i) {
            uint8_t m = m_ptr[i];
            if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                m_ptr[i] = m | m_used;
                cnt++;
                if (d_ptr[i] < 0.0f) neg = 1;
            }
        }
        counts_out[r] = cnt;
        has_neg_out[r] = neg;
        if (cnt < min_pixels || neg) {
            medians_out[r] = 1.0;
            if (sb.background.size() == sz) {
                std::fill(sb.background.begin(), sb.background.end(), 0.0f);
            }
        }
    }

    std::vector<int64_t> offsets(n + 1, 0);
    for (std::int64_t r = 0; r < n; ++r) {
        int64_t c = (counts_out[r] >= min_pixels && !has_neg_out[r]) ? counts_out[r] : 0;
        offsets[r + 1] = offsets[r] + c;
    }
    int64_t total_valid = offsets[n];
    return boost::python::make_tuple(total_valid);
}

void fill_glm_constant3d_bg(
    dials::af::const_ref<Shoebox<ProfileFloatType>> self,
    int min_pixels,
    int mask_bg_valid,
    int mask_overlapped,
    std::size_t counts_addr,
    std::size_t has_neg_addr,
    std::size_t medians_addr,
    std::size_t vals_addr,
    std::size_t idx_addr)
{
    const int64_t* counts = reinterpret_cast<const int64_t*>(counts_addr);
    const uint8_t* has_neg = reinterpret_cast<const uint8_t*>(has_neg_addr);
    double* medians_out = reinterpret_cast<double*>(medians_addr);
    double* vals_out = reinterpret_cast<double*>(vals_addr);
    int64_t* idx_out = reinterpret_cast<int64_t*>(idx_addr);
    uint8_t m_bg_val = static_cast<uint8_t>(mask_bg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    std::vector<int64_t> offsets(n + 1, 0);
    for (std::int64_t r = 0; r < n; ++r) {
        int64_t c = (counts[r] >= min_pixels && !has_neg[r]) ? counts[r] : 0;
        offsets[r + 1] = offsets[r] + c;
    }

    #pragma omp parallel for schedule(guided)
    for (std::int64_t r = 0; r < n; ++r) {
        int64_t cnt = counts[r];
        if (cnt < min_pixels || has_neg[r]) continue;
        const auto& sb = self[r];
        std::size_t sz = sb.data.size();
        const float* d_ptr = sb.data.begin();
        const uint8_t* m_ptr = sb.mask.begin();
        int64_t off = offsets[r];
        double* v_dst = vals_out + off;
        int64_t* i_dst = idx_out + off;
        int64_t k = 0;
        for (std::size_t i = 0; i < sz; ++i) {
            uint8_t m = m_ptr[i];
            if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                v_dst[k] = static_cast<double>(d_ptr[i]);
                i_dst[k] = r;
                k++;
            }
        }
        // Compute exact C++ median (nth_element at cnt / 2) in thread-local vector
        std::vector<double> tmp(v_dst, v_dst + cnt);
        std::nth_element(tmp.begin(), tmp.begin() + cnt / 2, tmp.end());
        double med = tmp[cnt / 2];
        medians_out[r] = (med <= 0.0) ? 1.0 : med;
    }
}

// 1b. Histogram + sparse tail extraction for GLM constant3d (compresses ~10M pixels into (n_refl, 128) hist)
std::size_t extract_glm_constant3d_hist(
    dials::af::ref<Shoebox<ProfileFloatType>> self,
    int min_pixels,
    int mask_bg_valid,
    int mask_overlapped,
    int mask_bg_used,
    std::size_t counts_addr,
    std::size_t medians_addr,
    std::size_t has_neg_addr,
    std::size_t hist_addr,
    std::size_t tail_counts_addr)
{
    int64_t* counts_out = reinterpret_cast<int64_t*>(counts_addr);
    double* medians_out = reinterpret_cast<double*>(medians_addr);
    uint8_t* has_neg_out = reinterpret_cast<uint8_t*>(has_neg_addr);
    int32_t* hist_out = reinterpret_cast<int32_t*>(hist_addr);
    int64_t* tail_counts_out = reinterpret_cast<int64_t*>(tail_counts_addr);
    uint8_t m_bg_val = static_cast<uint8_t>(mask_bg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    uint8_t m_used = static_cast<uint8_t>(mask_bg_used);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    #pragma omp parallel for schedule(guided)
    for (std::int64_t r = 0; r < n; ++r) {
        auto& sb = self[r];
        std::size_t sz = sb.data.size();
        int32_t* h_r = hist_out + r * 128;
        std::memset(h_r, 0, 128 * sizeof(int32_t));
        if (sz == 0) {
            counts_out[r] = 0;
            medians_out[r] = 1.0;
            has_neg_out[r] = 0;
            tail_counts_out[r] = 0;
            continue;
        }
        const float* d_ptr = sb.data.begin();
        uint8_t* m_ptr = sb.mask.begin();
        int64_t cnt = 0;
        int64_t tail_cnt = 0;
        uint8_t neg = 0;
        for (std::size_t i = 0; i < sz; ++i) {
            uint8_t m = m_ptr[i];
            if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                m_ptr[i] = m | m_used;
                cnt++;
                float v = d_ptr[i];
                if (v < 0.0f) {
                    neg = 1;
                } else {
                    int iv = static_cast<int>(v);
                    if (static_cast<float>(iv) == v && iv >= 0 && iv < 128) {
                        h_r[iv]++;
                    } else {
                        tail_cnt++;
                    }
                }
            }
        }
        counts_out[r] = cnt;
        has_neg_out[r] = neg;
        if (cnt < min_pixels || neg) {
            medians_out[r] = 1.0;
            tail_counts_out[r] = 0;
            std::memset(h_r, 0, 128 * sizeof(int32_t));
            if (sb.background.size() == sz) {
                std::fill(sb.background.begin(), sb.background.end(), 0.0f);
            }
        } else if (tail_cnt == 0) {
            tail_counts_out[r] = 0;
            int64_t target = cnt / 2;
            int64_t acc = 0;
            double med = 0.0;
            for (int k = 0; k < 128; ++k) {
                acc += h_r[k];
                if (acc > target) {
                    med = static_cast<double>(k);
                    break;
                }
            }
            medians_out[r] = (med <= 0.0) ? 1.0 : med;
        } else {
            tail_counts_out[r] = tail_cnt;
            medians_out[r] = 1.0;
        }
    }

    std::size_t total_tail = 0;
    for (std::int64_t r = 0; r < n; ++r) {
        total_tail += static_cast<std::size_t>(tail_counts_out[r]);
    }
    return total_tail;
}

void fill_glm_constant3d_tail(
    dials::af::const_ref<Shoebox<ProfileFloatType>> self,
    int min_pixels,
    int mask_bg_valid,
    int mask_overlapped,
    std::size_t counts_addr,
    std::size_t has_neg_addr,
    std::size_t tail_counts_addr,
    std::size_t medians_addr,
    std::size_t tail_vals_addr,
    std::size_t tail_idx_addr)
{
    const int64_t* counts = reinterpret_cast<const int64_t*>(counts_addr);
    const uint8_t* has_neg = reinterpret_cast<const uint8_t*>(has_neg_addr);
    const int64_t* tail_counts = reinterpret_cast<const int64_t*>(tail_counts_addr);
    double* medians_out = reinterpret_cast<double*>(medians_addr);
    double* tail_vals_out = reinterpret_cast<double*>(tail_vals_addr);
    int64_t* tail_idx_out = reinterpret_cast<int64_t*>(tail_idx_addr);
    uint8_t m_bg_val = static_cast<uint8_t>(mask_bg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    std::vector<int64_t> offsets(n + 1, 0);
    for (std::int64_t r = 0; r < n; ++r) {
        offsets[r + 1] = offsets[r] + tail_counts[r];
    }

    #pragma omp parallel for schedule(guided)
    for (std::int64_t r = 0; r < n; ++r) {
        if (tail_counts[r] == 0) continue;
        int64_t cnt = counts[r];
        if (cnt < min_pixels || has_neg[r]) continue;
        const auto& sb = self[r];
        std::size_t sz = sb.data.size();
        const float* d_ptr = sb.data.begin();
        const uint8_t* m_ptr = sb.mask.begin();
        int64_t off = offsets[r];
        std::vector<double> tmp;
        tmp.reserve(static_cast<std::size_t>(cnt));
        int64_t tk = 0;
        for (std::size_t i = 0; i < sz; ++i) {
            uint8_t m = m_ptr[i];
            if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                float v = d_ptr[i];
                double dv = static_cast<double>(v);
                tmp.push_back(dv);
                int iv = static_cast<int>(v);
                if (!(static_cast<float>(iv) == v && iv >= 0 && iv < 128)) {
                    tail_vals_out[off + tk] = dv;
                    tail_idx_out[off + tk] = r;
                    tk++;
                }
            }
        }
        std::nth_element(tmp.begin(), tmp.begin() + cnt / 2, tmp.end());
        double med = tmp[cnt / 2];
        medians_out[r] = (med <= 0.0) ? 1.0 : med;
    }
}

void unpack_constant3d_bg_and_mask(
    dials::af::ref<Shoebox<ProfileFloatType>> self,
    std::size_t means_addr,
    std::size_t conv_addr,
    int mask_bg_used)
{
    const double* means = reinterpret_cast<const double*>(means_addr);
    const uint8_t* conv = reinterpret_cast<const uint8_t*>(conv_addr);
    uint8_t m_not_used = static_cast<uint8_t>(255 ^ static_cast<uint8_t>(mask_bg_used));
    std::int64_t n = static_cast<std::int64_t>(self.size());

    #pragma omp parallel for schedule(guided)
    for (std::int64_t r = 0; r < n; ++r) {
        auto& sb = self[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) continue;
        float* bg_ptr = sb.background.begin();
        uint8_t* m_ptr = sb.mask.begin();
        if (conv[r]) {
            float m_val = static_cast<float>(means[r]);
            std::fill(bg_ptr, bg_ptr + sz, m_val);
        } else {
            std::fill(bg_ptr, bg_ptr + sz, 0.0f);
            for (std::size_t i = 0; i < sz; ++i) {
                m_ptr[i] &= m_not_used;
            }
        }
    }
}

// 1c. Fast OpenMP outlier rejection + moment extraction for SimpleBackground (constant3d)
// outlier_code: 0 = null, 1 = nsigma, 2 = tukey, 3 = truncated
void extract_simple_constant3d_bg(
    dials::af::ref<Shoebox<ProfileFloatType>> self,
    int outlier_code,
    double lower,
    double upper,
    int min_pixels,
    int mask_bg_valid,
    int mask_overlapped,
    int mask_bg_used,
    std::size_t sum_x_addr,
    std::size_t sum_s_addr,
    std::size_t sum_x2_addr,
    std::size_t count_addr)
{
    double* sum_x_out = reinterpret_cast<double*>(sum_x_addr);
    double* sum_s_out = reinterpret_cast<double*>(sum_s_addr);
    double* sum_x2_out = reinterpret_cast<double*>(sum_x2_addr);
    int64_t* count_out = reinterpret_cast<int64_t*>(count_addr);
    uint8_t m_bg_val = static_cast<uint8_t>(mask_bg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    uint8_t m_used = static_cast<uint8_t>(mask_bg_used);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    #pragma omp parallel for schedule(guided)
    for (std::int64_t r = 0; r < n; ++r) {
        auto& sb = self[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            sum_x_out[r] = 0.0;
            sum_s_out[r] = 0.0;
            sum_x2_out[r] = 0.0;
            count_out[r] = 0;
            continue;
        }
        const float* d_ptr = sb.data.begin();
        uint8_t* m_ptr = sb.mask.begin();

        if (outlier_code == 0) {
            for (std::size_t i = 0; i < sz; ++i) {
                uint8_t m = m_ptr[i];
                if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                    m_ptr[i] = m | m_used;
                }
            }
        } else if (outlier_code == 1) {
            std::vector<double> vals;
            for (std::size_t i = 0; i < sz; ++i) {
                uint8_t m = m_ptr[i];
                if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                    vals.push_back(static_cast<double>(d_ptr[i]));
                }
            }
            if (vals.size() <= 1) {
                sum_x_out[r] = 0.0;
                sum_s_out[r] = 0.0;
                sum_x2_out[r] = 0.0;
                count_out[r] = 0;
                continue;
            }
            double s1 = 0.0;
            for (double v : vals) s1 += v;
            double mean = s1 / static_cast<double>(vals.size());
            double s2 = 0.0;
            for (double v : vals) {
                double d = v - mean;
                s2 += d * d;
            }
            double sigma = std::sqrt(s2 / static_cast<double>(vals.size() - 1));
            double p0 = mean - lower * sigma;
            double p1 = mean + upper * sigma;
            for (std::size_t i = 0; i < sz; ++i) {
                uint8_t m = m_ptr[i];
                if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                    double v = static_cast<double>(d_ptr[i]);
                    if (p0 <= v && v <= p1) {
                        m_ptr[i] = m | m_used;
                    }
                }
            }
        } else if (outlier_code == 2) {
            std::vector<double> vals;
            for (std::size_t i = 0; i < sz; ++i) {
                uint8_t m = m_ptr[i];
                if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                    vals.push_back(static_cast<double>(d_ptr[i]));
                }
            }
            if (vals.size() < 4) {
                sum_x_out[r] = 0.0;
                sum_s_out[r] = 0.0;
                sum_x2_out[r] = 0.0;
                count_out[r] = 0;
                continue;
            }
            std::size_t mid = vals.size() / 2;
            std::size_t q1i = mid / 2;
            std::size_t q3i = mid + (vals.size() - mid) / 2;
            std::nth_element(vals.begin(), vals.begin() + q1i, vals.end());
            double q1 = vals[q1i];
            std::nth_element(vals.begin() + q1i + 1, vals.begin() + q3i, vals.end());
            double q3 = vals[q3i];
            double iqr = q3 - q1;
            double lo = q1 - lower * iqr;
            double hi = q3 + upper * iqr;
            for (std::size_t i = 0; i < sz; ++i) {
                uint8_t m = m_ptr[i];
                if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                    double v = static_cast<double>(d_ptr[i]);
                    if (lo <= v && v <= hi) {
                        m_ptr[i] = m | m_used;
                    }
                }
            }
        } else if (outlier_code == 3) {
            std::vector<std::size_t> idxs;
            for (std::size_t i = 0; i < sz; ++i) {
                uint8_t m = m_ptr[i];
                if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                    idxs.push_back(i);
                }
            }
            std::stable_sort(idxs.begin(), idxs.end(), [&](std::size_t a, std::size_t b) {
                return d_ptr[a] < d_ptr[b];
            });
            std::size_t num_data = idxs.size();
            std::size_t i0 = static_cast<std::size_t>(lower * num_data / 2.0);
            std::size_t i1 = num_data - static_cast<std::size_t>(upper * num_data / 2.0);
            for (std::size_t k = i0; k < i1; ++k) {
                m_ptr[idxs[k]] |= m_used;
            }
        }

        double M = 0.0, S = 0.0, sx = 0.0, sx2 = 0.0;
        int64_t count = 0;
        for (std::size_t i = 0; i < sz; ++i) {
            if ((m_ptr[i] & m_used) != 0) {
                count++;
                double x = static_cast<double>(d_ptr[i]);
                sx += x;
                sx2 += x * x;
                double oldM = M;
                M = M + (x - M) / static_cast<double>(count);
                S = S + (x - M) * (x - oldM);
            }
        }
        sum_x_out[r] = sx;
        sum_s_out[r] = S;
        sum_x2_out[r] = sx2;
        count_out[r] = count;
    }
}

// 2. Pack / unpack 3D bucket tensors (R, Z, Y, X) for SimpleBackgroundGPU
void get_shoebox_shapes(
    dials::af::const_ref<Shoebox<ProfileFloatType>> self,
    std::size_t shapes_addr,
    std::size_t volumes_addr)
{
    int64_t* shapes = reinterpret_cast<int64_t*>(shapes_addr);
    int64_t* volumes = reinterpret_cast<int64_t*>(volumes_addr);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    for (std::int64_t r = 0; r < n; ++r) {
        const auto& sb = self[r];
        std::size_t sz = sb.data.size();
        if (sz > 0) {
            auto acc = sb.data.accessor();
            shapes[3 * r + 0] = static_cast<int64_t>(acc[0]);
            shapes[3 * r + 1] = static_cast<int64_t>(acc[1]);
            shapes[3 * r + 2] = static_cast<int64_t>(acc[2]);
            volumes[r] = static_cast<int64_t>(sz);
        } else {
            shapes[3 * r + 0] = 0;
            shapes[3 * r + 1] = 0;
            shapes[3 * r + 2] = 0;
            volumes[r] = 0;
        }
    }
}

void pack_bucket_3d(
    dials::af::const_ref<Shoebox<ProfileFloatType>> self,
    std::size_t bucket_addr,
    std::size_t R,
    std::size_t Z,
    std::size_t Y,
    std::size_t X,
    std::size_t data_pad_addr,
    std::size_t mask_pad_addr)
{
    const int64_t* bucket = reinterpret_cast<const int64_t*>(bucket_addr);
    double* data_pad = reinterpret_cast<double*>(data_pad_addr);
    int32_t* mask_pad = reinterpret_cast<int32_t*>(mask_pad_addr);
    std::int64_t R_s = static_cast<std::int64_t>(R);
    std::size_t stride_z = Y * X;
    std::size_t stride_r = Z * stride_z;

    #pragma omp parallel for schedule(guided)
    for (std::int64_t bi = 0; bi < R_s; ++bi) {
        std::int64_t r = bucket[bi];
        const auto& sb = self[r];
        auto acc = sb.data.accessor();
        std::size_t nz = acc[0], ny = acc[1], nx = acc[2];
        const float* d_src = sb.data.begin();
        const uint8_t* m_src = sb.mask.begin();
        double* d_dst_r = data_pad + static_cast<std::size_t>(bi) * stride_r;
        int32_t* m_dst_r = mask_pad + static_cast<std::size_t>(bi) * stride_r;
        for (std::size_t k = 0; k < nz; ++k) {
            for (std::size_t j = 0; j < ny; ++j) {
                const float* d_row = d_src + (k * ny + j) * nx;
                const uint8_t* m_row = m_src + (k * ny + j) * nx;
                double* d_out = d_dst_r + k * stride_z + j * X;
                int32_t* m_out = m_dst_r + k * stride_z + j * X;
                for (std::size_t i = 0; i < nx; ++i) {
                    d_out[i] = static_cast<double>(d_row[i]);
                    m_out[i] = static_cast<int32_t>(m_row[i]);
                }
            }
        }
    }
}

void unpack_bucket_3d(
    dials::af::ref<Shoebox<ProfileFloatType>> self,
    std::size_t bucket_addr,
    std::size_t R,
    std::size_t Z,
    std::size_t Y,
    std::size_t X,
    std::size_t bg_pad_addr,
    std::size_t mask_pad_addr)
{
    const int64_t* bucket = reinterpret_cast<const int64_t*>(bucket_addr);
    const float* bg_pad = reinterpret_cast<const float*>(bg_pad_addr);
    const uint8_t* mask_pad = reinterpret_cast<const uint8_t*>(mask_pad_addr);
    std::int64_t R_s = static_cast<std::int64_t>(R);
    std::size_t stride_z = Y * X;
    std::size_t stride_r = Z * stride_z;

    #pragma omp parallel for schedule(guided)
    for (std::int64_t bi = 0; bi < R_s; ++bi) {
        std::int64_t r = bucket[bi];
        auto& sb = self[r];
        auto acc = sb.data.accessor();
        std::size_t nz = acc[0], ny = acc[1], nx = acc[2];
        float* b_dst = sb.background.begin();
        uint8_t* m_dst = sb.mask.begin();
        const float* b_src_r = bg_pad + static_cast<std::size_t>(bi) * stride_r;
        const uint8_t* m_src_r = mask_pad + static_cast<std::size_t>(bi) * stride_r;
        for (std::size_t k = 0; k < nz; ++k) {
            for (std::size_t j = 0; j < ny; ++j) {
                float* b_row = b_dst + (k * ny + j) * nx;
                uint8_t* m_row = m_dst + (k * ny + j) * nx;
                const float* b_in = b_src_r + k * stride_z + j * X;
                const uint8_t* m_in = m_src_r + k * stride_z + j * X;
                std::memcpy(b_row, b_in, nx * sizeof(float));
                std::memcpy(m_row, m_in, nx * sizeof(uint8_t));
            }
        }
    }
}

// 3. Extract foreground pixels for SimpleCentroidGPU (~3-5% of shoebox volume)
std::size_t count_foreground_pixels(
    dials::af::const_ref<Shoebox<ProfileFloatType>> self,
    int mask_fg_valid,
    int mask_overlapped,
    std::size_t fg_counts_addr,
    std::size_t flat_flags_addr)
{
    int64_t* fg_counts = reinterpret_cast<int64_t*>(fg_counts_addr);
    uint8_t* flat_flags = reinterpret_cast<uint8_t*>(flat_flags_addr);
    uint8_t m_fg_val = static_cast<uint8_t>(mask_fg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    #pragma omp parallel for schedule(guided)
    for (std::int64_t r = 0; r < n; ++r) {
        const auto& sb = self[r];
        flat_flags[r] = sb.flat ? 1 : 0;
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            fg_counts[r] = 0;
            continue;
        }
        const float* d_ptr = sb.data.begin();
        const float* b_ptr = sb.background.begin();
        const uint8_t* m_ptr = sb.mask.begin();
        int64_t cnt = 0;
        for (std::size_t i = 0; i < sz; ++i) {
            uint8_t m = m_ptr[i];
            if (((m & m_fg_val) == m_fg_val) && ((m & m_ovlp) == 0)) {
                if (d_ptr[i] - b_ptr[i] > 0.0f) {
                    cnt++;
                }
            }
        }
        fg_counts[r] = cnt;
    }

    std::size_t total = 0;
    for (std::int64_t r = 0; r < n; ++r) {
        total += static_cast<std::size_t>(fg_counts[r]);
    }
    return total;
}

void fill_foreground_pixels(
    dials::af::const_ref<Shoebox<ProfileFloatType>> self,
    int mask_fg_valid,
    int mask_overlapped,
    std::size_t fg_counts_addr,
    std::size_t w_addr,
    std::size_t x_addr,
    std::size_t y_addr,
    std::size_t z_addr,
    std::size_t idx_addr)
{
    const int64_t* fg_counts = reinterpret_cast<const int64_t*>(fg_counts_addr);
    double* w_out = reinterpret_cast<double*>(w_addr);
    double* x_out = reinterpret_cast<double*>(x_addr);
    double* y_out = reinterpret_cast<double*>(y_addr);
    double* z_out = reinterpret_cast<double*>(z_addr);
    int64_t* idx_out = reinterpret_cast<int64_t*>(idx_addr);
    uint8_t m_fg_val = static_cast<uint8_t>(mask_fg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    std::vector<int64_t> offsets(n + 1, 0);
    for (std::int64_t r = 0; r < n; ++r) {
        offsets[r + 1] = offsets[r] + fg_counts[r];
    }

    #pragma omp parallel for schedule(guided)
    for (std::int64_t r = 0; r < n; ++r) {
        if (fg_counts[r] == 0) continue;
        const auto& sb = self[r];
        auto acc = sb.data.accessor();
        std::size_t nz = acc[0], ny = acc[1], nx = acc[2];
        const float* d_ptr = sb.data.begin();
        const float* b_ptr = sb.background.begin();
        const uint8_t* m_ptr = sb.mask.begin();
        int64_t pos = offsets[r];
        for (std::size_t k = 0; k < nz; ++k) {
            double zc = static_cast<double>(k) + 0.5;
            for (std::size_t j = 0; j < ny; ++j) {
                double yc = static_cast<double>(j) + 0.5;
                std::size_t row_off = (k * ny + j) * nx;
                for (std::size_t i = 0; i < nx; ++i) {
                    uint8_t m = m_ptr[row_off + i];
                    if (((m & m_fg_val) == m_fg_val) && ((m & m_ovlp) == 0)) {
                        double val = static_cast<double>(d_ptr[row_off + i]) - static_cast<double>(b_ptr[row_off + i]);
                        if (val > 0.0) {
                            w_out[pos] = val;
                            x_out[pos] = static_cast<double>(i) + 0.5;
                            y_out[pos] = yc;
                            z_out[pos] = zc;
                            idx_out[pos] = r;
                            pos++;
                        }
                    }
                }
            }
        }
    }
}

// 4. Extract foreground pixels + background counts for SummationGPU (~3-5% of shoebox volume)
std::size_t extract_summation_stats(
    dials::af::const_ref<Shoebox<ProfileFloatType>> self,
    int mask_fg,
    int mask_fg_valid,
    int mask_overlapped,
    int mask_bg_used_valid,
    std::size_t fg_counts_addr,
    std::size_t n_bg_addr,
    std::size_t bad_fg_addr)
{
    int64_t* fg_counts = reinterpret_cast<int64_t*>(fg_counts_addr);
    double* n_bg_out = reinterpret_cast<double*>(n_bg_addr);
    int32_t* bad_fg_out = reinterpret_cast<int32_t*>(bad_fg_addr);
    uint8_t m_fg = static_cast<uint8_t>(mask_fg);
    uint8_t m_fg_val = static_cast<uint8_t>(mask_fg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    uint8_t m_bg_used = static_cast<uint8_t>(mask_bg_used_valid);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    #pragma omp parallel for schedule(guided)
    for (std::int64_t r = 0; r < n; ++r) {
        const auto& sb = self[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            fg_counts[r] = 0;
            n_bg_out[r] = 0.0;
            bad_fg_out[r] = 1;
            continue;
        }
        const uint8_t* m_ptr = sb.mask.begin();
        int64_t fg_cnt = 0;
        int64_t bg_cnt = 0;
        int32_t bad_cnt = 0;
        for (std::size_t i = 0; i < sz; ++i) {
            uint8_t m = m_ptr[i];
            bool is_fg = (m & m_fg) == m_fg;
            if (is_fg) {
                if (((m & m_fg_val) == m_fg_val) && ((m & m_ovlp) == 0)) {
                    fg_cnt++;
                } else {
                    bad_cnt++;
                }
            } else if ((m & m_bg_used) == m_bg_used) {
                bg_cnt++;
            }
        }
        fg_counts[r] = fg_cnt;
        n_bg_out[r] = static_cast<double>(bg_cnt);
        bad_fg_out[r] = bad_cnt;
    }

    std::size_t total = 0;
    for (std::int64_t r = 0; r < n; ++r) {
        total += static_cast<std::size_t>(fg_counts[r]);
    }
    return total;
}

void fill_summation_fg_pixels(
    dials::af::const_ref<Shoebox<ProfileFloatType>> self,
    int mask_fg_valid,
    int mask_overlapped,
    std::size_t fg_counts_addr,
    std::size_t d_fg_addr,
    std::size_t b_fg_addr,
    std::size_t idx_addr)
{
    const int64_t* fg_counts = reinterpret_cast<const int64_t*>(fg_counts_addr);
    double* d_fg_out = reinterpret_cast<double*>(d_fg_addr);
    double* b_fg_out = reinterpret_cast<double*>(b_fg_addr);
    int64_t* idx_out = reinterpret_cast<int64_t*>(idx_addr);
    uint8_t m_fg_val = static_cast<uint8_t>(mask_fg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    std::vector<int64_t> offsets(n + 1, 0);
    for (std::int64_t r = 0; r < n; ++r) {
        offsets[r + 1] = offsets[r] + fg_counts[r];
    }

    #pragma omp parallel for schedule(guided)
    for (std::int64_t r = 0; r < n; ++r) {
        if (fg_counts[r] == 0) continue;
        const auto& sb = self[r];
        std::size_t sz = sb.data.size();
        const float* d_ptr = sb.data.begin();
        const float* b_ptr = sb.background.begin();
        const uint8_t* m_ptr = sb.mask.begin();
        int64_t pos = offsets[r];
        for (std::size_t i = 0; i < sz; ++i) {
            uint8_t m = m_ptr[i];
            if (((m & m_fg_val) == m_fg_val) && ((m & m_ovlp) == 0)) {
                d_fg_out[pos] = static_cast<double>(d_ptr[i]);
                b_fg_out[pos] = static_cast<double>(b_ptr[i]);
                idx_out[pos] = r;
                pos++;
            }
        }
    }
}

// 3b. Extract foreground centroid moments per shoebox in a single L1-cached OpenMP pass
void extract_centroid_moments(
    dials::af::const_ref<Shoebox<ProfileFloatType>> self,
    int mask_fg_valid,
    int mask_overlapped,
    std::size_t fg_counts_addr,
    std::size_t flat_flags_addr,
    std::size_t moments_addr)
{
    int64_t* fg_counts = reinterpret_cast<int64_t*>(fg_counts_addr);
    uint8_t* flat_flags = reinterpret_cast<uint8_t*>(flat_flags_addr);
    double* moments = reinterpret_cast<double*>(moments_addr); // shape (n_refl, 8): sum_w, sum_wsq, mean_x, mean_y, mean_z, sum_wdx2, sum_wdy2, sum_wdz2
    uint8_t m_fg_val = static_cast<uint8_t>(mask_fg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    #pragma omp parallel for schedule(guided)
    for (std::int64_t r = 0; r < n; ++r) {
        const auto& sb = self[r];
        flat_flags[r] = sb.flat ? 1 : 0;
        double* mom_r = moments + r * 8;
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            fg_counts[r] = 0;
            std::memset(mom_r, 0, 8 * sizeof(double));
            continue;
        }
        auto acc = sb.data.accessor();
        std::size_t nz = acc[0], ny = acc[1], nx = acc[2];
        const float* d_ptr = sb.data.begin();
        const float* b_ptr = sb.background.begin();
        const uint8_t* m_ptr = sb.mask.begin();

        int64_t cnt = 0;
        double sum_w = 0.0, sum_wsq = 0.0, sum_wx = 0.0, sum_wy = 0.0, sum_wz = 0.0;
        for (std::size_t k = 0; k < nz; ++k) {
            double zc = static_cast<double>(k) + 0.5;
            for (std::size_t j = 0; j < ny; ++j) {
                double yc = static_cast<double>(j) + 0.5;
                std::size_t row_off = (k * ny + j) * nx;
                for (std::size_t i = 0; i < nx; ++i) {
                    uint8_t m = m_ptr[row_off + i];
                    if (((m & m_fg_val) == m_fg_val) && ((m & m_ovlp) == 0)) {
                        double val = static_cast<double>(d_ptr[row_off + i]) - static_cast<double>(b_ptr[row_off + i]);
                        if (val > 0.0) {
                            double xc = static_cast<double>(i) + 0.5;
                            cnt++;
                            sum_w += val;
                            sum_wsq += val * val;
                            sum_wx += val * xc;
                            sum_wy += val * yc;
                            sum_wz += val * zc;
                        }
                    }
                }
            }
        }
        fg_counts[r] = cnt;
        if (cnt == 0 || sum_w <= 0.0) {
            std::memset(mom_r, 0, 8 * sizeof(double));
            continue;
        }
        double inv_w = 1.0 / sum_w;
        double mean_x = sum_wx * inv_w;
        double mean_y = sum_wy * inv_w;
        double mean_z = sum_wz * inv_w;
        double sum_wdx2 = 0.0, sum_wdy2 = 0.0, sum_wdz2 = 0.0;
        if (cnt > 1) {
            for (std::size_t k = 0; k < nz; ++k) {
                double dz = (static_cast<double>(k) + 0.5) - mean_z;
                double dz2 = dz * dz;
                for (std::size_t j = 0; j < ny; ++j) {
                    double dy = (static_cast<double>(j) + 0.5) - mean_y;
                    double dy2 = dy * dy;
                    std::size_t row_off = (k * ny + j) * nx;
                    for (std::size_t i = 0; i < nx; ++i) {
                        uint8_t m = m_ptr[row_off + i];
                        if (((m & m_fg_val) == m_fg_val) && ((m & m_ovlp) == 0)) {
                            double val = static_cast<double>(d_ptr[row_off + i]) - static_cast<double>(b_ptr[row_off + i]);
                            if (val > 0.0) {
                                double dx = (static_cast<double>(i) + 0.5) - mean_x;
                                sum_wdx2 += val * (dx * dx);
                                sum_wdy2 += val * dy2;
                                sum_wdz2 += val * dz2;
                            }
                        }
                    }
                }
            }
        }
        mom_r[0] = sum_w;
        mom_r[1] = sum_wsq;
        mom_r[2] = mean_x;
        mom_r[3] = mean_y;
        mom_r[4] = mean_z;
        mom_r[5] = sum_wdx2;
        mom_r[6] = sum_wdy2;
        mom_r[7] = sum_wdz2;
    }
}

// 4b. Extract summation totals per shoebox in a single OpenMP pass
void extract_summation_totals(
    dials::af::const_ref<Shoebox<ProfileFloatType>> self,
    int mask_fg,
    int mask_fg_valid,
    int mask_overlapped,
    int mask_bg_used_valid,
    std::size_t sum_p_addr,
    std::size_t sum_b_addr,
    std::size_t n_sig_addr,
    std::size_t n_bg_addr,
    std::size_t bad_fg_addr)
{
    double* sum_p_out = reinterpret_cast<double*>(sum_p_addr);
    double* sum_b_out = reinterpret_cast<double*>(sum_b_addr);
    double* n_sig_out = reinterpret_cast<double*>(n_sig_addr);
    double* n_bg_out = reinterpret_cast<double*>(n_bg_addr);
    int32_t* bad_fg_out = reinterpret_cast<int32_t*>(bad_fg_addr);
    uint8_t m_fg = static_cast<uint8_t>(mask_fg);
    uint8_t m_fg_val = static_cast<uint8_t>(mask_fg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    uint8_t m_bg_used = static_cast<uint8_t>(mask_bg_used_valid);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    #pragma omp parallel for schedule(guided)
    for (std::int64_t r = 0; r < n; ++r) {
        const auto& sb = self[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            sum_p_out[r] = 0.0;
            sum_b_out[r] = 0.0;
            n_sig_out[r] = 0.0;
            n_bg_out[r] = 0.0;
            bad_fg_out[r] = 1;
            continue;
        }
        const float* d_ptr = sb.data.begin();
        const float* b_ptr = sb.background.begin();
        const uint8_t* m_ptr = sb.mask.begin();
        double sum_p = 0.0;
        double sum_b = 0.0;
        int64_t fg_cnt = 0;
        int64_t bg_cnt = 0;
        int32_t bad_cnt = 0;
        for (std::size_t i = 0; i < sz; ++i) {
            uint8_t m = m_ptr[i];
            bool is_fg = (m & m_fg) == m_fg;
            if (is_fg) {
                if (((m & m_fg_val) == m_fg_val) && ((m & m_ovlp) == 0)) {
                    sum_p += static_cast<double>(d_ptr[i]);
                    sum_b += static_cast<double>(b_ptr[i]);
                    fg_cnt++;
                } else {
                    bad_cnt++;
                }
            } else if ((m & m_bg_used) == m_bg_used) {
                bg_cnt++;
            }
        }
        sum_p_out[r] = sum_p;
        sum_b_out[r] = sum_b;
        n_sig_out[r] = static_cast<double>(fg_cnt);
        n_bg_out[r] = static_cast<double>(bg_cnt);
        bad_fg_out[r] = bad_cnt;
    }
}

// 5. Fast OpenMP shoebox extraction directly from a 3D (B, H, W) int32 raw volume
std::size_t extract_shoeboxes_from_volume(
    dials::af::ref<Shoebox<ProfileFloatType>> self,
    std::size_t vol_addr,
    std::size_t static_mask_addr,
    std::size_t rejected_addr,
    int tr_min,
    int tr_max,
    int frame0,
    int frame1,
    std::size_t H,
    std::size_t W,
    std::size_t valid_flags_addr)
{
    const int32_t* vol = reinterpret_cast<const int32_t*>(vol_addr);
    const uint8_t* static_mask = reinterpret_cast<const uint8_t*>(static_mask_addr);
    const uint8_t* rejected = reinterpret_cast<const uint8_t*>(rejected_addr);
    uint8_t* valid_flags = reinterpret_cast<uint8_t*>(valid_flags_addr);
    std::int64_t n = static_cast<std::int64_t>(self.size());
    int yi = static_cast<int>(H);
    int xi = static_cast<int>(W);
    std::size_t stride_f = H * W;

    #pragma omp parallel for schedule(guided)
    for (std::int64_t r = 0; r < n; ++r) {
        auto& sbox = self[r];
        sbox.allocate_data();
        auto b = sbox.bbox;
        int x0 = b[0];
        int x1 = b[1];
        int y0 = b[2];
        int y1 = b[3];
        int z0 = b[4];
        int z1 = b[5];
        int xs = x1 - x0;
        int ys = y1 - y0;
        int zs = z1 - z0;
        int xb = x0 >= 0 ? 0 : std::abs(x0);
        int yb = y0 >= 0 ? 0 : std::abs(y0);
        int xe = x1 <= xi ? xs : xs - (x1 - xi);
        int ye = y1 <= yi ? ys : ys - (y1 - yi);
        if (yb >= ye || xb >= xe || z0 < frame0 || z1 > frame1) {
            valid_flags[r] = 0;
            continue;
        }
        valid_flags[r] = 1;
        sbox.allocate_background();
        float* sdata = sbox.data.begin();
        uint8_t* smask = sbox.mask.begin();
        std::size_t s_stride_z = static_cast<std::size_t>(ys) * static_cast<std::size_t>(xs);
        std::size_t s_stride_y = static_cast<std::size_t>(xs);

        if (sbox.flat) {
            for (int z = 0; z < zs; ++z) {
                int vf = (z0 + z) - frame0;
                bool is_rej = (rejected[vf] != 0);
                const int32_t* vol_f = vol + static_cast<std::size_t>(vf) * stride_f;
                for (int y = yb; y < ye; ++y) {
                    const int32_t* vol_row = vol_f + static_cast<std::size_t>(y + y0) * W;
                    const uint8_t* sm_row = static_mask + static_cast<std::size_t>(y + y0) * W;
                    float* sd_row = sdata + static_cast<std::size_t>(y) * s_stride_y;
                    uint8_t* smk_row = smask + static_cast<std::size_t>(y) * s_stride_y;
                    if (is_rej) {
                        for (int x = xb; x < xe; ++x) {
                            sd_row[x] += static_cast<float>(vol_row[x + x0]);
                            smk_row[x] = 0;
                        }
                    } else {
                        for (int x = xb; x < xe; ++x) {
                            int32_t raw_val = vol_row[x + x0];
                            sd_row[x] += static_cast<float>(raw_val);
                            bool sv = (smk_row[x] & dials::model::Valid) != 0;
                            bool mv = (sm_row[x + x0] != 0) && (raw_val >= tr_min) && (raw_val <= tr_max);
                            smk_row[x] = (mv && (z == 0 ? true : sv)) ? dials::model::Valid : 0;
                        }
                    }
                }
            }
        } else {
            for (int z = 0; z < zs; ++z) {
                int vf = (z0 + z) - frame0;
                bool is_rej = (rejected[vf] != 0);
                const int32_t* vol_f = vol + static_cast<std::size_t>(vf) * stride_f;
                float* sd_z = sdata + static_cast<std::size_t>(z) * s_stride_z;
                uint8_t* smk_z = smask + static_cast<std::size_t>(z) * s_stride_z;
                for (int y = yb; y < ye; ++y) {
                    const int32_t* vol_row = vol_f + static_cast<std::size_t>(y + y0) * W;
                    const uint8_t* sm_row = static_mask + static_cast<std::size_t>(y + y0) * W;
                    float* sd_row = sd_z + static_cast<std::size_t>(y) * s_stride_y;
                    uint8_t* smk_row = smk_z + static_cast<std::size_t>(y) * s_stride_y;
                    if (is_rej) {
                        for (int x = xb; x < xe; ++x) {
                            sd_row[x] = static_cast<float>(vol_row[x + x0]);
                            smk_row[x] = 0;
                        }
                    } else {
                        for (int x = xb; x < xe; ++x) {
                            int32_t raw_val = vol_row[x + x0];
                            sd_row[x] = static_cast<float>(raw_val);
                            bool mv = (sm_row[x + x0] != 0) && (raw_val >= tr_min) && (raw_val <= tr_max);
                            smk_row[x] = mv ? dials::model::Valid : 0;
                        }
                    }
                }
            }
        }
    }

    std::size_t n_valid = 0;
    for (std::int64_t r = 0; r < n; ++r) {
        if (valid_flags[r]) n_valid++;
    }
    return n_valid;
}

// 6. OpenMP parallel foreground/background mask calculation for GaussianRS
void parallel_compute_mask(
    const dials::algorithms::profile_model::gaussian_rs::MaskCalculatorIface& calc,
    dials::af::ref<Shoebox<ProfileFloatType>> shoeboxes,
    const dials::af::const_ref<scitbx::vec3<double>>& s1,
    const dials::af::const_ref<double>& frame,
    const dials::af::const_ref<std::size_t>& panel)
{
    DIALS_ASSERT(shoeboxes.size() == s1.size());
    DIALS_ASSERT(shoeboxes.size() == frame.size());
    DIALS_ASSERT(shoeboxes.size() == panel.size());
    std::int64_t n = static_cast<std::int64_t>(shoeboxes.size());
    #pragma omp parallel for schedule(guided)
    for (std::int64_t i = 0; i < n; ++i) {
        calc.single(shoeboxes[i], s1[i], frame[i], panel[i]);
    }
}

// 7. OpenMP parallel GaussianRS profile fitting and modelling (thread-safe, zero Scan/flex_table access in threads)
struct ThreadSafeTransformForward {
    dials::af::versa<bool, dials::af::c_grid<3>> mask_;
    dials::af::versa<double, dials::af::c_grid<3>> profile_;
    dials::af::versa<double, dials::af::c_grid<3>> background_;

    ThreadSafeTransformForward(
        const dxtbx::model::Panel& panel,
        int scan_index0,
        double phi0,
        double dphi,
        double sigma_b,
        double sigma_m,
        double n_sigma,
        std::size_t half_grid_size,
        const dials::algorithms::profile_model::gaussian_rs::CoordinateSystem& cs,
        scitbx::af::int6 bbox,
        const dials::af::const_ref<double, dials::af::c_grid<3>>& image,
        const dials::af::const_ref<double, dials::af::c_grid<3>>* bkgrd_ptr,
        const dials::af::const_ref<bool, dials::af::c_grid<3>>& mask)
    {
        using namespace dials::algorithms::profile_model::gaussian_rs::transform;
        using dials::algorithms::polygon::clip::vert4;
        using dials::algorithms::polygon::spatial_interpolation::Match;
        using dials::algorithms::polygon::spatial_interpolation::quad_to_grid;
        using scitbx::vec2;
        using scitbx::vec3;
        using scitbx::af::int3;
        using scitbx::af::double3;

        int x0 = bbox[0];
        int y0 = bbox[2];
        int3 shoebox_size(bbox[5] - bbox[4], bbox[3] - bbox[2], bbox[1] - bbox[0]);
        DIALS_ASSERT(shoebox_size.all_gt(0));
        int g_dim = static_cast<int>(2 * half_grid_size + 1);
        int3 grid_size(g_dim, g_dim, g_dim);
        double step_m = sigma_m * n_sigma / (static_cast<double>(half_grid_size) + 0.5);
        double step_b = sigma_b * n_sigma / (static_cast<double>(half_grid_size) + 0.5);
        double3 step_size(step_m, step_b, step_b);
        double g_cent = static_cast<double>(half_grid_size) + 0.5;
        double3 grid_cent(g_cent, g_cent, g_cent);

        vec3<double> s1 = cs.s1();
        double s1_len = s1.length();
        DIALS_ASSERT(s1_len > 0);
        vec3<double> e1 = cs.e1_axis() / s1_len;
        vec3<double> e2 = cs.e2_axis() / s1_len;

        vec2<int> zrange(bbox[4], bbox[5]);
        MapFramesForward<double> map_frames_forward(
            scan_index0, phi0, dphi, sigma_m, n_sigma, grid_size[2] / 2
        );
        dials::af::versa<double, dials::af::c_grid<2>> zfraction_arr =
            map_frames_forward(zrange, cs.phi(), cs.zeta());

        MapFramesReverse<double> map_frames_backward(
            scan_index0, phi0, dphi, sigma_m, n_sigma, grid_size[2] / 2
        );
        dials::af::versa<double, dials::af::c_grid<2>> efraction_arr =
            map_frames_backward(zrange, cs.phi(), cs.zeta());

        dials::af::const_ref<double, dials::af::c_grid<2>> zfraction = zfraction_arr.const_ref();
        dials::af::const_ref<double, dials::af::c_grid<2>> efraction = efraction_arr.const_ref();

        dials::af::c_grid<3> accessor(grid_size);
        profile_ = dials::af::versa<double, dials::af::c_grid<3>>(accessor, 0.0);
        if (bkgrd_ptr != nullptr) {
            background_ = dials::af::versa<double, dials::af::c_grid<3>>(accessor, 0.0);
        }
        mask_ = dials::af::versa<bool, dials::af::c_grid<3>>(accessor, false);

        for (std::size_t kk = 0; kk < static_cast<std::size_t>(grid_size[0]); ++kk) {
            double sumk = 0.0;
            for (std::size_t k = 0; k < static_cast<std::size_t>(shoebox_size[0]); ++k) {
                sumk += efraction(kk, k);
            }
            if (sumk > 0.99) {
                for (std::size_t j = 0; j < static_cast<std::size_t>(grid_size[1]); ++j) {
                    for (std::size_t i = 0; i < static_cast<std::size_t>(grid_size[2]); ++i) {
                        mask_(kk, j, i) = true;
                    }
                }
            }
        }

        vec2<double> shoebox_centroid_px = panel.get_ray_intersection_px(s1);
        double attenuation_length = panel.attenuation_length(shoebox_centroid_px);

        dials::af::versa<vec2<double>, dials::af::c_grid<2>> gc_array(
            dials::af::c_grid<2>(shoebox_size[1] + 1, shoebox_size[2] + 1)
        );
        for (int j = 0; j <= shoebox_size[1]; ++j) {
            for (int i = 0; i <= shoebox_size[2]; ++i) {
                vec3<double> sp = panel.get_pixel_lab_coord(
                    vec2<double>(x0 + i, y0 + j), attenuation_length
                );
                vec3<double> ds = sp.normalize() * s1_len - s1;
                gc_array(j, i) = vec2<double>(
                    grid_cent[2] + (e1 * ds) / step_size[2],
                    grid_cent[1] + (e2 * ds) / step_size[1]
                );
            }
        }

        dials::af::c_grid<2> grid_size2(grid_size[1], grid_size[2]);
        int img_w = static_cast<int>(panel.get_image_size()[0]);
        int img_h = static_cast<int>(panel.get_image_size()[1]);

        if (bkgrd_ptr != nullptr) {
            const auto& bkgrd = *bkgrd_ptr;
            for (int j = 0; j < shoebox_size[1]; ++j) {
                if (y0 + j < 0 || y0 + j >= img_h) continue;
                for (int i = 0; i < shoebox_size[2]; ++i) {
                    if (x0 + i < 0 || x0 + i >= img_w) continue;
                    vert4 input(
                        gc_array(j, i),
                        gc_array(j, i + 1),
                        gc_array(j + 1, i + 1),
                        gc_array(j + 1, i)
                    );
                    dials::af::shared<Match> matches = quad_to_grid(input, grid_size2, 0);
                    for (std::size_t m = 0; m < matches.size(); ++m) {
                        double fraction = matches[m].fraction;
                        int idx = matches[m].out;
                        int jj = idx / grid_size[2];
                        int ii = idx % grid_size[2];
                        for (int k = 0; k < shoebox_size[0]; ++k) {
                            if (mask(k, j, i)) {
                                double ivalue = image(k, j, i) * fraction;
                                double bvalue = bkgrd(k, j, i) * fraction;
                                for (int kk = 0; kk < grid_size[0]; ++kk) {
                                    double zf = zfraction(k, kk);
                                    profile_(kk, jj, ii) += ivalue * zf;
                                    background_(kk, jj, ii) += bvalue * zf;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            for (int j = 0; j < shoebox_size[1]; ++j) {
                if (y0 + j < 0 || y0 + j >= img_h) continue;
                for (int i = 0; i < shoebox_size[2]; ++i) {
                    if (x0 + i < 0 || x0 + i >= img_w) continue;
                    vert4 input(
                        gc_array(j, i),
                        gc_array(j, i + 1),
                        gc_array(j + 1, i + 1),
                        gc_array(j + 1, i)
                    );
                    dials::af::shared<Match> matches = quad_to_grid(input, grid_size2, 0);
                    for (std::size_t m = 0; m < matches.size(); ++m) {
                        double fraction = matches[m].fraction;
                        int idx = matches[m].out;
                        int jj = idx / grid_size[2];
                        int ii = idx % grid_size[2];
                        for (int k = 0; k < shoebox_size[0]; ++k) {
                            if (mask(k, j, i)) {
                                double value = image(k, j, i) * fraction;
                                for (int kk = 0; kk < grid_size[0]; ++kk) {
                                    profile_(kk, jj, ii) += value * zfraction(k, kk);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

struct ParallelGaussianRS : public dials::algorithms::GaussianRSProfileModeller {
    dials::af::shared<bool> fit_parallel_cols(
        dials::af::const_ref<Shoebox<ProfileFloatType>> sbox,
        const dials::af::const_ref<scitbx::vec3<double>>& s1,
        const dials::af::const_ref<scitbx::vec3<double>>& xyzpx,
        const dials::af::const_ref<scitbx::vec3<double>>& xyzmm,
        dials::af::ref<std::size_t> flags,
        dials::af::ref<double> intensity_val,
        dials::af::ref<double> intensity_var,
        dials::af::ref<double> reference_cor) const
    {
        using namespace dials::algorithms;
        using dials::model::Foreground;
        using dials::model::Valid;
        using dials::algorithms::profile_model::gaussian_rs::CoordinateSystem;

        scitbx::vec3<double> m2 = goniometer_.get_rotation_axis();
        scitbx::vec3<double> s0 = beam_->get_s0();
        int scan_index0 = scan_.get_array_range()[0];
        scitbx::vec2<double> scan_osc = scan_.get_oscillation();
        double phi0 = scan_osc[0];
        double dphi = scan_osc[1];
        double sb = sigma_b_;
        double sm = sigma_m_;
        double ns = n_sigma_;
        std::size_t hgs = grid_size_;

        std::size_t n_prof = size();
        std::vector<data_const_reference> prof_data(n_prof);
        std::vector<mask_const_reference> prof_mask(n_prof);
        std::vector<uint8_t> prof_valid(n_prof, 0);
        for (std::size_t k = 0; k < n_prof; ++k) {
            if (data_[k].size() != 0 && mask_[k].size() != 0) {
                prof_data[k] = data_[k].const_ref();
                prof_mask[k] = mask_[k].const_ref();
                prof_valid[k] = 1;
            }
        }

        std::int64_t n = static_cast<std::int64_t>(sbox.size());
        dials::af::shared<bool> success(sbox.size(), false);
        bool* succ_ptr = success.begin();

        if (fit_method_ == ReciprocalSpace) {
            #pragma omp parallel for schedule(guided)
            for (std::int64_t i = 0; i < n; ++i) {
                intensity_val[i] = 0.0;
                intensity_var[i] = -1.0;
                reference_cor[i] = 0.0;
                flags[i] &= ~dials::af::IntegratedPrf;
                bool integrate = !(flags[i] & dials::af::DontIntegrate);
                if (!integrate) continue;
                try {
                    std::size_t index = sampler_->nearest(sbox[i].panel, xyzpx[i]);
                    if (index >= n_prof || !prof_valid[index]) continue;
                    data_const_reference p = prof_data[index];
                    mask_const_reference mask1 = prof_mask[index];
                    CoordinateSystem cs(m2, s0, s1[i], xyzmm[i][2]);

                    dials::af::versa<double, dials::af::c_grid<3>> d_arr(sbox[i].data.accessor());
                    std::copy(sbox[i].data.begin(), sbox[i].data.end(), d_arr.begin());

                    dials::af::versa<double, dials::af::c_grid<3>> b_arr(sbox[i].background.accessor());
                    std::copy(sbox[i].background.begin(), sbox[i].background.end(), b_arr.begin());

                    dials::af::versa<bool, dials::af::c_grid<3>> m_arr(sbox[i].mask.accessor());
                    std::transform(
                        sbox[i].mask.begin(),
                        sbox[i].mask.end(),
                        m_arr.begin(),
                        detail::check_mask_code(Valid | Foreground)
                    );

                    auto b_cref = b_arr.const_ref();
                    ThreadSafeTransformForward transform(
                        detector_[sbox[i].panel],
                        scan_index0, phi0, dphi, sb, sm, ns, hgs,
                        cs, sbox[i].bbox,
                        d_arr.const_ref(), &b_cref, m_arr.const_ref()
                    );

                    data_const_reference c = transform.profile_.const_ref();
                    data_const_reference b = transform.background_.const_ref();
                    mask_const_reference mask2 = transform.mask_.const_ref();
                    dials::af::versa<bool, dials::af::c_grid<3>> m(mask2.accessor());
                    for (std::size_t j = 0; j < m.size(); ++j) {
                        m[j] = mask1[j] && mask2[j];
                    }

                    ProfileFitter<double> fit(c, b, m.const_ref(), p, 1e-3, 100);
                    intensity_val[i] = fit.intensity()[0];
                    intensity_var[i] = fit.variance()[0];
                    reference_cor[i] = fit.correlation();
                    flags[i] |= dials::af::IntegratedPrf;
                    succ_ptr[i] = true;
                } catch (...) {
                    continue;
                }
            }
        } else {
            throw DIALS_ERROR("DetectorSpace fallback to serial");
        }
        return success;
    }

    void model_parallel_cols(
        dials::af::const_ref<Shoebox<ProfileFloatType>> sbox,
        const dials::af::const_ref<double>& partiality,
        const dials::af::const_ref<scitbx::vec3<double>>& s1,
        const dials::af::const_ref<scitbx::vec3<double>>& xyzpx,
        const dials::af::const_ref<scitbx::vec3<double>>& xyzmm,
        dials::af::ref<std::size_t> flags)
    {
        using namespace dials::algorithms;
        using dials::model::Foreground;
        using dials::model::Valid;
        using dials::algorithms::profile_model::gaussian_rs::CoordinateSystem;

        scitbx::vec3<double> m2 = goniometer_.get_rotation_axis();
        scitbx::vec3<double> s0 = beam_->get_s0();
        int scan_index0 = scan_.get_array_range()[0];
        scitbx::vec2<double> scan_osc = scan_.get_oscillation();
        double phi0 = scan_osc[0];
        double dphi = scan_osc[1];
        double sb = sigma_b_;
        double sm = sigma_m_;
        double ns = n_sigma_;
        std::size_t hgs = grid_size_;

        std::int64_t n = static_cast<std::int64_t>(sbox.size());

        #pragma omp parallel for schedule(guided)
        for (std::int64_t i = 0; i < n; ++i) {
            bool full = partiality[i] > 0.99;
            bool integrated = (flags[i] & dials::af::IntegratedSum) != 0;
            if (!full || !integrated) continue;
            const auto& panel_obj = detector_[sbox[i].panel];
            bool bbox_valid = (
                sbox[i].bbox[0] >= 0 && sbox[i].bbox[2] >= 0 &&
                sbox[i].bbox[1] <= static_cast<int>(panel_obj.get_image_size()[0]) &&
                sbox[i].bbox[3] <= static_cast<int>(panel_obj.get_image_size()[1])
            );
            if (!bbox_valid) continue;
            bool pixels_valid = true;
            for (std::size_t k = 0; k < sbox[i].mask.size(); ++k) {
                if ((sbox[i].mask[k] & Foreground) && !(sbox[i].mask[k] & Valid)) {
                    pixels_valid = false;
                    break;
                }
            }
            if (!pixels_valid) continue;

            try {
                CoordinateSystem cs(m2, s0, s1[i], xyzmm[i][2]);
                dials::af::versa<double, dials::af::c_grid<3>> d_arr(sbox[i].data.accessor());
                std::transform(
                    sbox[i].data.begin(), sbox[i].data.end(),
                    sbox[i].background.begin(), d_arr.begin(),
                    std::minus<double>()
                );
                dials::af::versa<bool, dials::af::c_grid<3>> m_arr(sbox[i].mask.accessor());
                std::transform(
                    sbox[i].mask.begin(), sbox[i].mask.end(),
                    m_arr.begin(), detail::check_mask_code(Valid | Foreground)
                );

                ThreadSafeTransformForward transform(
                    panel_obj,
                    scan_index0, phi0, dphi, sb, sm, ns, hgs,
                    cs, sbox[i].bbox,
                    d_arr.const_ref(), nullptr, m_arr.const_ref()
                );

                dials::af::shared<std::size_t> indices = sampler_->nearest_n(sbox[i].panel, xyzpx[i]);
                dials::af::shared<double> weights(indices.size());
                for (std::size_t j = 0; j < indices.size(); ++j) {
                    weights[j] = sampler_->weight(indices[j], sbox[i].panel, xyzpx[i]);
                }

                #pragma omp critical
                {
                    add(indices.const_ref(), weights.const_ref(), transform.profile_.const_ref());
                }
                flags[i] |= dials::af::UsedInModelling;
            } catch (...) {
                continue;
            }
        }
    }
};

dials::af::shared<bool> parallel_fit_multi(
    const dials::algorithms::MultiExpProfileModeller& fitter,
    dials::af::const_ref<Shoebox<ProfileFloatType>> sbox,
    const dials::af::const_ref<scitbx::vec3<double>>& s1,
    const dials::af::const_ref<scitbx::vec3<double>>& xyzpx,
    const dials::af::const_ref<scitbx::vec3<double>>& xyzmm,
    dials::af::ref<std::size_t> flags,
    dials::af::ref<double> intensity_val,
    dials::af::ref<double> intensity_var,
    dials::af::ref<double> reference_cor)
{
    DIALS_ASSERT(fitter.size() == 1);
    const auto* grs = dynamic_cast<const dials::algorithms::GaussianRSProfileModeller*>(fitter[0].get());
    DIALS_ASSERT(grs != nullptr);
    return static_cast<const ParallelGaussianRS*>(grs)->fit_parallel_cols(
        sbox, s1, xyzpx, xyzmm, flags, intensity_val, intensity_var, reference_cor
    );
}

void parallel_model_multi(
    dials::algorithms::MultiExpProfileModeller& fitter,
    dials::af::const_ref<Shoebox<ProfileFloatType>> sbox,
    const dials::af::const_ref<double>& partiality,
    const dials::af::const_ref<scitbx::vec3<double>>& s1,
    const dials::af::const_ref<scitbx::vec3<double>>& xyzpx,
    const dials::af::const_ref<scitbx::vec3<double>>& xyzmm,
    dials::af::ref<std::size_t> flags)
{
    DIALS_ASSERT(fitter.size() == 1);
    auto* grs = dynamic_cast<dials::algorithms::GaussianRSProfileModeller*>(fitter[0].get());
    DIALS_ASSERT(grs != nullptr);
    static_cast<ParallelGaussianRS*>(grs)->model_parallel_cols(
        sbox, partiality, s1, xyzpx, xyzmm, flags
    );
}

} // namespace dials_gpu

BOOST_PYTHON_MODULE(_shoebox_pack_ext)
{
    using namespace boost::python;
    def("extract_glm_constant3d_bg", &dials_gpu::extract_glm_constant3d_bg);
    def("fill_glm_constant3d_bg", &dials_gpu::fill_glm_constant3d_bg);
    def("extract_glm_constant3d_hist", &dials_gpu::extract_glm_constant3d_hist);
    def("fill_glm_constant3d_tail", &dials_gpu::fill_glm_constant3d_tail);
    def("unpack_constant3d_bg_and_mask", &dials_gpu::unpack_constant3d_bg_and_mask);
    def("extract_simple_constant3d_bg", &dials_gpu::extract_simple_constant3d_bg);
    def("get_shoebox_shapes", &dials_gpu::get_shoebox_shapes);
    def("pack_bucket_3d", &dials_gpu::pack_bucket_3d);
    def("unpack_bucket_3d", &dials_gpu::unpack_bucket_3d);
    def("count_foreground_pixels", &dials_gpu::count_foreground_pixels);
    def("fill_foreground_pixels", &dials_gpu::fill_foreground_pixels);
    def("extract_centroid_moments", &dials_gpu::extract_centroid_moments);
    def("extract_summation_stats", &dials_gpu::extract_summation_stats);
    def("fill_summation_fg_pixels", &dials_gpu::fill_summation_fg_pixels);
    def("extract_summation_totals", &dials_gpu::extract_summation_totals);
    def("extract_shoeboxes_from_volume", &dials_gpu::extract_shoeboxes_from_volume);
    def("parallel_compute_mask", &dials_gpu::parallel_compute_mask);
    def("parallel_fit_multi", &dials_gpu::parallel_fit_multi);
    def("parallel_model_multi", &dials_gpu::parallel_model_multi);
}
