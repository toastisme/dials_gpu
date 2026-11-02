#include <boost/python.hpp>
#include <dials/algorithms/profile_model/gaussian_rs/bbox_calculator.h>
#include <dials/algorithms/profile_model/gaussian_rs/partiality_calculator.h>
#include <dials/algorithms/spot_prediction/pixel_to_miller_index.h>
#include <cctbx/miller.h>
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
#include <thread>
#include <atomic>
#include <type_traits>

namespace dials_gpu {

using dials::model::Shoebox;
using dials::ProfileFloatType;

inline int num_worker_threads() {
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 8;
    return std::min(hw, 20);
}

template <typename Func>
inline void parallel_for(std::int64_t n, Func&& fn, std::int64_t min_per_thread = 32) {
    if (n <= 0) return;
    int max_threads = num_worker_threads();
    int n_threads = static_cast<int>((n + min_per_thread - 1) / min_per_thread);
    if (n_threads > max_threads) n_threads = max_threads;
    if (n_threads <= 1) {
        for (std::int64_t i = 0; i < n; ++i) {
            if constexpr (std::is_invocable_v<Func, std::int64_t, int>) {
                fn(i, 0);
            } else {
                fn(i);
            }
        }
        return;
    }
    std::int64_t chunk = std::max<std::int64_t>(1, n / (n_threads * 8));
    if (chunk > 64) chunk = 64;
    std::atomic<std::int64_t> next_idx{0};
    auto worker = [&](int tid) {
        while (true) {
            std::int64_t start = next_idx.fetch_add(chunk, std::memory_order_relaxed);
            if (start >= n) break;
            std::int64_t end = std::min(n, start + chunk);
            for (std::int64_t i = start; i < end; ++i) {
                if constexpr (std::is_invocable_v<Func, std::int64_t, int>) {
                    fn(i, tid);
                } else {
                    fn(i);
                }
            }
        }
    };
    std::vector<std::thread> threads;
    threads.reserve(n_threads - 1);
    for (int t = 1; t < n_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    worker(0);
    for (auto& th : threads) {
        th.join();
    }
}


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

    parallel_for(n, [&](std::int64_t r, int) {
        auto& sb = self[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            counts_out[r] = 0;
            medians_out[r] = 1.0;
            has_neg_out[r] = 0;
            return;
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
    });

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

    parallel_for(n, [&](std::int64_t r, int) {
        int64_t cnt = counts[r];
        if (cnt < min_pixels || has_neg[r]) return;
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
    });
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

    parallel_for(n, [&](std::int64_t r, int) {
        auto& sb = self[r];
        std::size_t sz = sb.data.size();
        int32_t* h_r = hist_out + r * 128;
        std::memset(h_r, 0, 128 * sizeof(int32_t));
        if (sz == 0) {
            counts_out[r] = 0;
            medians_out[r] = 1.0;
            has_neg_out[r] = 0;
            tail_counts_out[r] = 0;
            return;
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
    });

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

    parallel_for(n, [&](std::int64_t r, int) {
        if (tail_counts[r] == 0) return;
        int64_t cnt = counts[r];
        if (cnt < min_pixels || has_neg[r]) return;
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
    });
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

    parallel_for(n, [&](std::int64_t r, int) {
        auto& sb = self[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) return;
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
    });
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

    parallel_for(n, [&](std::int64_t r, int) {
        auto& sb = self[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            sum_x_out[r] = 0.0;
            sum_s_out[r] = 0.0;
            sum_x2_out[r] = 0.0;
            count_out[r] = 0;
            return;
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
                return;
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
                return;
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
    });
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

    parallel_for(R_s, [&](std::int64_t bi) {
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
    });
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

    parallel_for(R_s, [&](std::int64_t bi) {
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
    });
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

    parallel_for(n, [&](std::int64_t r) {
        const auto& sb = self[r];
        flat_flags[r] = sb.flat ? 1 : 0;
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            fg_counts[r] = 0;
            return;
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
    });

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

    parallel_for(n, [&](std::int64_t r) {
        if (fg_counts[r] == 0) return;
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
    });
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

    parallel_for(n, [&](std::int64_t r) {
        const auto& sb = self[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            fg_counts[r] = 0;
            n_bg_out[r] = 0.0;
            bad_fg_out[r] = 1;
            return;
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
    });

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

    parallel_for(n, [&](std::int64_t r) {
        if (fg_counts[r] == 0) return;
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
    });
}

// 3b. Extract foreground centroid moments per shoebox in a single L1-cached parallel pass
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
    double* moments = reinterpret_cast<double*>(moments_addr);
    uint8_t m_fg_val = static_cast<uint8_t>(mask_fg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    parallel_for(n, [&](std::int64_t r) {
        const auto& sb = self[r];
        flat_flags[r] = sb.flat ? 1 : 0;
        double* mom_r = moments + r * 8;
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            fg_counts[r] = 0;
            std::memset(mom_r, 0, 8 * sizeof(double));
            return;
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
            return;
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
    });
}

// 4b. Extract summation totals per shoebox in a single parallel pass
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

    parallel_for(n, [&](std::int64_t r) {
        const auto& sb = self[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            sum_p_out[r] = 0.0;
            sum_b_out[r] = 0.0;
            n_sig_out[r] = 0.0;
            n_bg_out[r] = 0.0;
            bad_fg_out[r] = 1;
            return;
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
    });
}

// 5. Fast parallel shoebox extraction directly from a 3D (B, H, W) int32 raw volume
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

    parallel_for(n, [&](std::int64_t r) {
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
            return;
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
    });

    std::size_t n_valid = 0;
    for (std::int64_t r = 0; r < n; ++r) {
        if (valid_flags[r]) n_valid++;
    }
    return n_valid;
}

void extract_wide_shoeboxes_slice_from_volume(
    dials::af::ref<Shoebox<ProfileFloatType>> self,
    std::size_t vol_addr,
    std::size_t static_mask_addr,
    std::size_t rejected_addr,
    int tr_min,
    int tr_max,
    int chunk_frame0,
    int chunk_frame1,
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

    parallel_for(n, [&](std::int64_t r) {
        auto& sbox = self[r];
        auto b = sbox.bbox;
        int x0 = b[0], x1 = b[1], y0 = b[2], y1 = b[3], z0 = b[4], z1 = b[5];
        int xs = x1 - x0, ys = y1 - y0;
        int xb = x0 >= 0 ? 0 : std::abs(x0);
        int yb = y0 >= 0 ? 0 : std::abs(y0);
        int xe = x1 <= xi ? xs : xs - (x1 - xi);
        int ye = y1 <= yi ? ys : ys - (y1 - yi);
        if (yb >= ye || xb >= xe) {
            valid_flags[r] = 0;
            return;
        }
        valid_flags[r] = 1;
        int f_start = std::max(z0, chunk_frame0);
        int f_end = std::min(z1, chunk_frame1);
        if (f_start >= f_end) return;

        float* sdata = sbox.data.begin();
        uint8_t* smask = sbox.mask.begin();
        std::size_t s_stride_z = static_cast<std::size_t>(ys) * static_cast<std::size_t>(xs);
        std::size_t s_stride_y = static_cast<std::size_t>(xs);

        if (sbox.flat) {
            for (int f = f_start; f < f_end; ++f) {
                int z = f - z0;
                int vf = f - chunk_frame0;
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
            for (int f = f_start; f < f_end; ++f) {
                int z = f - z0;
                int vf = f - chunk_frame0;
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
    });
}

// 6. Parallel foreground/background mask calculation for GaussianRS
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
    parallel_for(n, [&](std::int64_t i) {
        calc.single(shoeboxes[i], s1[i], frame[i], panel[i]);
    });
}

void parallel_compute_mask_vec3(
    const dials::algorithms::profile_model::gaussian_rs::MaskCalculatorIface& calc,
    dials::af::ref<Shoebox<ProfileFloatType>> shoeboxes,
    const dials::af::const_ref<scitbx::vec3<double>>& s1,
    const dials::af::const_ref<scitbx::vec3<double>>& xyzcal_px,
    const dials::af::const_ref<std::size_t>& panel)
{
    DIALS_ASSERT(shoeboxes.size() == s1.size());
    DIALS_ASSERT(shoeboxes.size() == xyzcal_px.size());
    DIALS_ASSERT(shoeboxes.size() == panel.size());
    std::int64_t n = static_cast<std::int64_t>(shoeboxes.size());
    parallel_for(n, [&](std::int64_t i) {
        calc.single(shoeboxes[i], s1[i], xyzcal_px[i][2], panel[i]);
    });
}

void parallel_compute_mask_3d_fast(
    const dxtbx::model::BeamBase& beam,
    const dxtbx::model::Detector& detector,
    const dxtbx::model::Goniometer& gonio,
    const dxtbx::model::Scan& scan,
    double delta_b,
    dials::af::ref<Shoebox<ProfileFloatType>> shoeboxes,
    const dials::af::const_ref<scitbx::vec3<double>>& s1,
    const dials::af::const_ref<scitbx::vec3<double>>& xyzcal_px,
    const dials::af::const_ref<std::size_t>& panel_arr)
{
    using dials::algorithms::profile_model::gaussian_rs::CoordinateSystem;
    using dials::model::Foreground;
    using dials::model::Background;
    using scitbx::vec2;
    using scitbx::vec3;

    DIALS_ASSERT(delta_b > 0.0);
    DIALS_ASSERT(shoeboxes.size() == s1.size());
    DIALS_ASSERT(shoeboxes.size() == xyzcal_px.size());
    DIALS_ASSERT(shoeboxes.size() == panel_arr.size());

    vec3<double> m2 = gonio.get_rotation_axis();
    vec3<double> s0 = beam.get_s0();
    double s0_length = s0.length();
    double phi0 = scan.get_oscillation()[0];
    double dphi = scan.get_oscillation()[1];
    int index0 = scan.get_array_range()[0];
    int index1 = scan.get_array_range()[1];
    double delta_b_r = 1.0 / delta_b;
    double delta_b_r2 = delta_b_r * delta_b_r;

    std::int64_t n = static_cast<std::int64_t>(shoeboxes.size());
    std::vector<std::vector<double>> thread_dxy_bufs(20);
    parallel_for(n, [&](std::int64_t idx, int tid) {
        auto& dxy_buf = thread_dxy_bufs[tid];
        auto& shoebox = shoeboxes[idx];
        auto mask = shoebox.mask.ref();
        scitbx::af::int6 bbox = shoebox.bbox;
        double frame = xyzcal_px[idx][2];
        double phi = phi0 + (frame - index0) * dphi;
        int x0 = bbox[0], x1 = bbox[1];
        int y0 = bbox[2], y1 = bbox[3];
        int z0 = bbox[4], z1 = bbox[5];
        int xsize = x1 - x0;
        int ysize = y1 - y0;
        int zsize = z1 - z0;

        const dxtbx::model::Panel& panel = detector[panel_arr[idx]];
        CoordinateSystem cs(m2, s0, s1[idx], phi);

        std::size_t stride_x = static_cast<std::size_t>(xsize + 1);
        std::size_t dxy_sz = static_cast<std::size_t>(ysize + 1) * stride_x;
        if (dxy_buf.size() < dxy_sz) {
            dxy_buf.resize(dxy_sz);
        }

        if (shoebox.flat) {
            for (int j = 0; j <= ysize; ++j) {
                for (int i = 0; i <= xsize; ++i) {
                    vec2<double> gxy = cs.from_beam_vector(
                        panel.get_pixel_lab_coord(vec2<double>(x0 + i, y0 + j)).normalize() * s0_length);
                    dxy_buf[static_cast<std::size_t>(j) * stride_x + i] =
                        (gxy[0] * gxy[0] + gxy[1] * gxy[1]) * delta_b_r2;
                }
            }
            uint8_t* m_ptr = mask.begin();
            for (int j = 0; j < ysize; ++j) {
                const double* r0 = &dxy_buf[static_cast<std::size_t>(j) * stride_x];
                const double* r1 = r0 + stride_x;
                uint8_t* m_row = m_ptr + static_cast<std::size_t>(j) * xsize;
                for (int i = 0; i < xsize; ++i) {
                    double dxy = std::min(std::min(r0[i], r1[i]), std::min(r0[i + 1], r1[i + 1]));
                    m_row[i] |= (dxy <= 1.0) ? Foreground : Background;
                }
            }
        } else {
            vec2<double> shoebox_centroid_px = panel.get_ray_intersection_px(s1[idx]);
            double attenuation_length = panel.attenuation_length(shoebox_centroid_px);
            for (int j = 0; j <= ysize; ++j) {
                for (int i = 0; i <= xsize; ++i) {
                    vec2<double> gxy = cs.from_beam_vector(
                        panel.get_pixel_lab_coord(vec2<double>(x0 + i, y0 + j), attenuation_length).normalize() * s0_length);
                    dxy_buf[static_cast<std::size_t>(j) * stride_x + i] =
                        (gxy[0] * gxy[0] + gxy[1] * gxy[1]) * delta_b_r2;
                }
            }
            int k_start = std::max(0, index0 - z0);
            int k_end = std::min(zsize, index1 - z0);
            if (k_start < k_end) {
                uint8_t* m_ptr = mask.begin();
                std::size_t slice_stride = static_cast<std::size_t>(ysize) * static_cast<std::size_t>(xsize);
                uint8_t* m_first = m_ptr + static_cast<std::size_t>(k_start) * slice_stride;
                int n_k = k_end - k_start;
                for (int j = 0; j < ysize; ++j) {
                    const double* r0 = &dxy_buf[static_cast<std::size_t>(j) * stride_x];
                    const double* r1 = r0 + stride_x;
                    uint8_t* m_row0 = m_first + static_cast<std::size_t>(j) * xsize;
                    for (int i = 0; i < xsize; ++i) {
                        double dxy = std::min(std::min(r0[i], r1[i]), std::min(r0[i + 1], r1[i + 1]));
                        uint8_t mv = (dxy <= 1.0) ? Foreground : Background;
                        for (int k = 0; k < n_k; ++k) {
                            m_row0[static_cast<std::size_t>(k) * slice_stride + i] |= mv;
                        }
                    }
                }
            }
        }
    });
}

// 6b. Fused pre-background mask validation + foreground count + post-integration mask counts
void parallel_pre_background_mask_flags(
    dials::af::const_ref<Shoebox<ProfileFloatType>> shoeboxes,
    dials::af::ref<std::size_t> flags,
    double valid_foreground_threshold,
    dials::af::ref<int> nvalfg_out)
{
    using dials::model::Valid;
    using dials::model::Background;
    using dials::model::Foreground;
    DIALS_ASSERT(shoeboxes.size() == flags.size());
    DIALS_ASSERT(shoeboxes.size() == nvalfg_out.size());
    std::int64_t n = static_cast<std::int64_t>(shoeboxes.size());
    constexpr uint8_t bg_val = Background | Valid;
    constexpr uint8_t fg_val = Foreground | Valid;

    parallel_for(n, [&](std::int64_t r) {
        const auto& sb = shoeboxes[r];
        const uint8_t* m_ptr = sb.mask.begin();
        std::size_t sz = sb.mask.size();
        int nbackg = 0, nvalbg = 0, nforeg = 0, nvalfg = 0;
        for (std::size_t i = 0; i < sz; ++i) {
            uint8_t m = m_ptr[i];
            if (m & Background) {
                nbackg++;
                if ((m & bg_val) == bg_val) nvalbg++;
            }
            if (m & Foreground) {
                nforeg++;
                if ((m & fg_val) == fg_val) nvalfg++;
            }
        }
        std::size_t fl = flags[r];
        if (nbackg > nvalbg) fl |= dials::af::BackgroundIncludesBadPixels;
        if (nforeg > nvalfg) fl |= dials::af::ForegroundIncludesBadPixels;
        double frac = static_cast<double>(nvalfg) / static_cast<double>(nforeg);
        if (frac < valid_foreground_threshold) fl |= dials::af::DontIntegrate;
        flags[r] = fl;
        nvalfg_out[r] = nvalfg;
    });
}

void parallel_post_mask_counts(
    dials::af::const_ref<Shoebox<ProfileFloatType>> shoeboxes,
    dials::af::ref<int> nvalid_out,
    dials::af::ref<int> nbg_out,
    dials::af::ref<int> nbg_used_out)
{
    using dials::model::Valid;
    using dials::model::Background;
    using dials::model::BackgroundUsed;
    DIALS_ASSERT(shoeboxes.size() == nvalid_out.size());
    DIALS_ASSERT(shoeboxes.size() == nbg_out.size());
    DIALS_ASSERT(shoeboxes.size() == nbg_used_out.size());
    std::int64_t n = static_cast<std::int64_t>(shoeboxes.size());
    constexpr uint8_t m_val = Valid;
    constexpr uint8_t m_bg = Valid | Background;
    constexpr uint8_t m_bg_used = Valid | Background | BackgroundUsed;

    parallel_for(n, [&](std::int64_t r) {
        const auto& sb = shoeboxes[r];
        const uint8_t* m_ptr = sb.mask.begin();
        std::size_t sz = sb.mask.size();
        int nv = 0, nb = 0, nbu = 0;
        for (std::size_t i = 0; i < sz; ++i) {
            uint8_t m = m_ptr[i];
            if ((m & m_val) == m_val) {
                nv++;
                if ((m & m_bg) == m_bg) {
                    nb++;
                    if ((m & m_bg_used) == m_bg_used) {
                        nbu++;
                    }
                }
            }
        }
        nvalid_out[r] = nv;
        nbg_out[r] = nb;
        nbg_used_out[r] = nbu;
    });
}

void parallel_check_overloads_single_panel(
    dials::af::ref<Shoebox<ProfileFloatType>> shoeboxes,
    dials::af::ref<std::size_t> flags,
    double saturation)
{
    using dials::model::Valid;
    DIALS_ASSERT(shoeboxes.size() == flags.size());
    std::int64_t n = static_cast<std::int64_t>(shoeboxes.size());
    float sat_f = static_cast<float>(saturation);
    parallel_for(n, [&](std::int64_t r) {
        auto& sb = shoeboxes[r];
        const float* d_ptr = sb.data.begin();
        uint8_t* m_ptr = sb.mask.begin();
        std::size_t sz = sb.data.size();
        bool ovl = false;
        for (std::size_t i = 0; i < sz; ++i) {
            if (d_ptr[i] > sat_f) {
                m_ptr[i] &= ~Valid;
                ovl = true;
            }
        }
        if (ovl) {
            flags[r] |= dials::af::Overloaded;
        }
    });
}

// 7. Parallel GaussianRS profile fitting and modelling (zero per-reflection data/mask/Match heap allocations)
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
        const Shoebox<ProfileFloatType>& sbox,
        bool use_bkgrd)
    {
        using namespace dials::algorithms::profile_model::gaussian_rs::transform;
        using dials::algorithms::polygon::clip::vert4;
        using dials::algorithms::polygon::spatial_interpolation::quad_grid_range;
        using dials::algorithms::polygon::spatial_interpolation::reverse_quad_inplace_if_backward;
        using dials::algorithms::polygon::spatial_interpolation::quad_grid_intersection_area;
        using dials::model::Foreground;
        using dials::model::Valid;
        using scitbx::vec2;
        using scitbx::vec3;
        using scitbx::af::int3;
        using scitbx::af::int4;
        using scitbx::af::double3;

        scitbx::af::int6 bbox = sbox.bbox;
        int x0 = bbox[0];
        int y0 = bbox[2];
        int nz = bbox[5] - bbox[4];
        int ny = bbox[3] - bbox[2];
        int nx = bbox[1] - bbox[0];
        int3 shoebox_size(nz, ny, nx);
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
        if (use_bkgrd) {
            background_ = dials::af::versa<double, dials::af::c_grid<3>>(accessor, 0.0);
        }
        mask_ = dials::af::versa<bool, dials::af::c_grid<3>>(accessor, false);

        for (std::size_t kk = 0; kk < static_cast<std::size_t>(grid_size[0]); ++kk) {
            double sumk = 0.0;
            for (std::size_t k = 0; k < static_cast<std::size_t>(nz); ++k) {
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

        const float* sdata = sbox.data.begin();
        const float* sbkgrd = sbox.background.begin();
        const uint8_t* smask = sbox.mask.begin();
        constexpr uint8_t fg_valid = Valid | Foreground;
        std::size_t stride_z = static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);
        std::size_t stride_y = static_cast<std::size_t>(nx);

        int img_w = static_cast<int>(panel.get_image_size()[0]);
        int img_h = static_cast<int>(panel.get_image_size()[1]);

        // Find 2D bounding box of valid foreground pixels in (j, i) so we only evaluate
        // get_pixel_lab_coord and polygon clipping where foreground pixels actually exist
        int j_min = ny, j_max = -1, i_min = nx, i_max = -1;
        for (int j = 0; j < ny; ++j) {
            if (y0 + j < 0 || y0 + j >= img_h) continue;
            for (int i = 0; i < nx; ++i) {
                if (x0 + i < 0 || x0 + i >= img_w) continue;
                std::size_t col_off = static_cast<std::size_t>(j) * stride_y + static_cast<std::size_t>(i);
                for (int k = 0; k < nz; ++k) {
                    if ((smask[static_cast<std::size_t>(k) * stride_z + col_off] & fg_valid) == fg_valid) {
                        if (j < j_min) j_min = j;
                        if (j > j_max) j_max = j;
                        if (i < i_min) i_min = i;
                        if (i > i_max) i_max = i;
                        break;
                    }
                }
            }
        }
        if (j_min > j_max || i_min > i_max) {
            return;
        }

        vec2<double> shoebox_centroid_px = panel.get_ray_intersection_px(s1);
        double attenuation_length = panel.attenuation_length(shoebox_centroid_px);

        int sub_ny = (j_max - j_min) + 2;
        int sub_nx = (i_max - i_min) + 2;
        std::vector<vec2<double>> gc_sub(static_cast<std::size_t>(sub_ny * sub_nx));
        double inv_step2 = 1.0 / step_size[2];
        double inv_step1 = 1.0 / step_size[1];
        for (int jj = 0; jj < sub_ny; ++jj) {
            int j = j_min + jj;
            for (int ii = 0; ii < sub_nx; ++ii) {
                int i = i_min + ii;
                vec3<double> sp = panel.get_pixel_lab_coord(
                    vec2<double>(x0 + i, y0 + j), attenuation_length
                );
                vec3<double> ds = sp.normalize() * s1_len - s1;
                gc_sub[static_cast<std::size_t>(jj * sub_nx + ii)] = vec2<double>(
                    grid_cent[2] + (e1 * ds) * inv_step2,
                    grid_cent[1] + (e2 * ds) * inv_step1
                );
            }
        }

        dials::af::c_grid<2> grid_size2(grid_size[1], grid_size[2]);

        if (use_bkgrd) {
            for (int j = j_min; j <= j_max; ++j) {
                if (y0 + j < 0 || y0 + j >= img_h) continue;
                int jj_sub = j - j_min;
                for (int i = i_min; i <= i_max; ++i) {
                    if (x0 + i < 0 || x0 + i >= img_w) continue;
                    std::size_t col_off = static_cast<std::size_t>(j) * stride_y + static_cast<std::size_t>(i);
                    bool any_k = false;
                    for (int k = 0; k < nz; ++k) {
                        if ((smask[static_cast<std::size_t>(k) * stride_z + col_off] & fg_valid) == fg_valid) {
                            any_k = true;
                            break;
                        }
                    }
                    if (!any_k) continue;

                    int ii_sub = i - i_min;
                    vert4 input(
                        gc_sub[static_cast<std::size_t>(jj_sub * sub_nx + ii_sub)],
                        gc_sub[static_cast<std::size_t>(jj_sub * sub_nx + ii_sub + 1)],
                        gc_sub[static_cast<std::size_t>((jj_sub + 1) * sub_nx + ii_sub + 1)],
                        gc_sub[static_cast<std::size_t>((jj_sub + 1) * sub_nx + ii_sub)]
                    );
                    int4 range = quad_grid_range(input, grid_size2);
                    if (range[0] >= range[1] || range[2] >= range[3]) continue;
                    double target_area = reverse_quad_inplace_if_backward(input);
                    double inv_target_area = 1.0 / target_area;

                    for (int jj = range[2]; jj < range[3]; ++jj) {
                        for (int ii = range[0]; ii < range[1]; ++ii) {
                            double result_area = quad_grid_intersection_area(input, ii, jj);
                            if (result_area > 0.0) {
                                double fraction = result_area * inv_target_area;
                                for (int k = 0; k < nz; ++k) {
                                    std::size_t off = static_cast<std::size_t>(k) * stride_z + col_off;
                                    if ((smask[off] & fg_valid) == fg_valid) {
                                        double ivalue = static_cast<double>(sdata[off]) * fraction;
                                        double bvalue = static_cast<double>(sbkgrd[off]) * fraction;
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
                }
            }
        } else {
            for (int j = j_min; j <= j_max; ++j) {
                if (y0 + j < 0 || y0 + j >= img_h) continue;
                int jj_sub = j - j_min;
                for (int i = i_min; i <= i_max; ++i) {
                    if (x0 + i < 0 || x0 + i >= img_w) continue;
                    std::size_t col_off = static_cast<std::size_t>(j) * stride_y + static_cast<std::size_t>(i);
                    bool any_k = false;
                    for (int k = 0; k < nz; ++k) {
                        if ((smask[static_cast<std::size_t>(k) * stride_z + col_off] & fg_valid) == fg_valid) {
                            any_k = true;
                            break;
                        }
                    }
                    if (!any_k) continue;

                    int ii_sub = i - i_min;
                    vert4 input(
                        gc_sub[static_cast<std::size_t>(jj_sub * sub_nx + ii_sub)],
                        gc_sub[static_cast<std::size_t>(jj_sub * sub_nx + ii_sub + 1)],
                        gc_sub[static_cast<std::size_t>((jj_sub + 1) * sub_nx + ii_sub + 1)],
                        gc_sub[static_cast<std::size_t>((jj_sub + 1) * sub_nx + ii_sub)]
                    );
                    int4 range = quad_grid_range(input, grid_size2);
                    if (range[0] >= range[1] || range[2] >= range[3]) continue;
                    double target_area = reverse_quad_inplace_if_backward(input);
                    double inv_target_area = 1.0 / target_area;

                    for (int jj = range[2]; jj < range[3]; ++jj) {
                        for (int ii = range[0]; ii < range[1]; ++ii) {
                            double result_area = quad_grid_intersection_area(input, ii, jj);
                            if (result_area > 0.0) {
                                double fraction = result_area * inv_target_area;
                                for (int k = 0; k < nz; ++k) {
                                    std::size_t off = static_cast<std::size_t>(k) * stride_z + col_off;
                                    if ((smask[off] & fg_valid) == fg_valid) {
                                        double value = (static_cast<double>(sdata[off]) - static_cast<double>(sbkgrd[off])) * fraction;
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
            parallel_for(n, [&](std::int64_t i) {
                intensity_val[i] = 0.0;
                intensity_var[i] = -1.0;
                reference_cor[i] = 0.0;
                flags[i] &= ~dials::af::IntegratedPrf;
                bool integrate = !(flags[i] & dials::af::DontIntegrate);
                if (!integrate) return;
                try {
                    std::size_t index = sampler_->nearest(sbox[i].panel, xyzpx[i]);
                    if (index >= n_prof || !prof_valid[index]) return;
                    data_const_reference p = prof_data[index];
                    mask_const_reference mask1 = prof_mask[index];
                    CoordinateSystem cs(m2, s0, s1[i], xyzmm[i][2]);

                    ThreadSafeTransformForward transform(
                        detector_[sbox[i].panel],
                        scan_index0, phi0, dphi, sb, sm, ns, hgs,
                        cs, sbox[i], true
                    );

                    data_const_reference c = transform.profile_.const_ref();
                    data_const_reference b = transform.background_.const_ref();
                    bool* m2_ptr = transform.mask_.begin();
                    std::size_t m_sz = transform.mask_.size();
                    for (std::size_t j = 0; j < m_sz; ++j) {
                        m2_ptr[j] = mask1[j] && m2_ptr[j];
                    }

                    ProfileFitter<double> fit(c, b, transform.mask_.const_ref(), p, 1e-3, 100);
                    intensity_val[i] = fit.intensity()[0];
                    intensity_var[i] = fit.variance()[0];
                    reference_cor[i] = fit.correlation();
                    flags[i] |= dials::af::IntegratedPrf;
                    succ_ptr[i] = true;
                } catch (...) {
                    return;
                }
            });
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
        if (n <= 0) return;

        std::size_t n_prof = data_.size();
        std::size_t prof_elems = accessor_.size_1d();
        int nt = num_worker_threads();
        if (nt > n) nt = static_cast<int>(n);

        // Thread-local profile accumulators to eliminate lock contention completely
        std::vector<std::vector<std::vector<double>>> thread_data(
            static_cast<std::size_t>(nt),
            std::vector<std::vector<double>>(n_prof)
        );
        std::vector<std::vector<std::size_t>> thread_n_refl(
            static_cast<std::size_t>(nt),
            std::vector<std::size_t>(n_prof, 0)
        );

        std::atomic<std::int64_t> next_idx{0};
        constexpr std::int64_t chunk = 16;

        auto worker = [&](int tid) {
            auto& loc_data = thread_data[static_cast<std::size_t>(tid)];
            auto& loc_n = thread_n_refl[static_cast<std::size_t>(tid)];
            while (true) {
                std::int64_t start = next_idx.fetch_add(chunk, std::memory_order_relaxed);
                if (start >= n) break;
                std::int64_t end = std::min(start + chunk, n);
                for (std::int64_t i = start; i < end; ++i) {
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
                    const uint8_t* m_ptr = sbox[i].mask.begin();
                    std::size_t m_sz = sbox[i].mask.size();
                    bool pixels_valid = true;
                    for (std::size_t k = 0; k < m_sz; ++k) {
                        if ((m_ptr[k] & Foreground) && !(m_ptr[k] & Valid)) {
                            pixels_valid = false;
                            break;
                        }
                    }
                    if (!pixels_valid) continue;

                    try {
                        CoordinateSystem cs(m2, s0, s1[i], xyzmm[i][2]);
                        ThreadSafeTransformForward transform(
                            panel_obj,
                            scan_index0, phi0, dphi, sb, sm, ns, hgs,
                            cs, sbox[i], false
                        );

                        const double* prof_ptr = transform.profile_.begin();
                        double sum_data = 0.0;
                        for (std::size_t p_i = 0; p_i < prof_elems; ++p_i) {
                            sum_data += prof_ptr[p_i];
                        }
                        if (sum_data > 0.0) {
                            double inv_sum = 1.0 / sum_data;
                            dials::af::shared<std::size_t> indices = sampler_->nearest_n(sbox[i].panel, xyzpx[i]);
                            for (std::size_t j = 0; j < indices.size(); ++j) {
                                std::size_t idx = indices[j];
                                double w = sampler_->weight(idx, sbox[i].panel, xyzpx[i]) * inv_sum;
                                auto& vec = loc_data[idx];
                                if (vec.empty()) {
                                    vec.assign(prof_elems, 0.0);
                                }
                                for (std::size_t p_i = 0; p_i < prof_elems; ++p_i) {
                                    vec[p_i] += w * prof_ptr[p_i];
                                }
                                loc_n[idx]++;
                            }
                        }
                        flags[i] |= dials::af::UsedInModelling;
                    } catch (...) {
                        continue;
                    }
                }
            }
        };

        std::vector<std::thread> pool;
        pool.reserve(static_cast<std::size_t>(nt - 1));
        for (int t = 1; t < nt; ++t) {
            pool.emplace_back(worker, t);
        }
        worker(0);
        for (auto& th : pool) {
            th.join();
        }

        // Merge thread-local accumulators into EmpiricalProfileModeller
        for (std::size_t idx = 0; idx < n_prof; ++idx) {
            std::size_t tot_n = 0;
            for (int t = 0; t < nt; ++t) {
                tot_n += thread_n_refl[static_cast<std::size_t>(t)][idx];
            }
            if (tot_n == 0) continue;
            if (data_[idx].size() == 0) {
                data_[idx] = data_type(accessor_, 0.0);
                mask_[idx] = mask_type(accessor_, true);
            }
            double* dst = data_[idx].begin();
            for (int t = 0; t < nt; ++t) {
                const auto& vec = thread_data[static_cast<std::size_t>(t)][idx];
                if (!vec.empty()) {
                    for (std::size_t p_i = 0; p_i < prof_elems; ++p_i) {
                        dst[p_i] += vec[p_i];
                    }
                }
            }
            n_reflections_[idx] += tot_n;
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

// 8. Fast C++ multi-threaded extractors for GaussianRS profile parameter estimation (sigma_b and sigma_m)
void compute_beam_divergence_variances(
    const dxtbx::model::Detector& detector,
    dials::af::const_ref<Shoebox<ProfileFloatType>> shoeboxes,
    const dials::af::const_ref<scitbx::vec3<double>>& s1_all,
    const dials::af::const_ref<scitbx::vec3<double>>& xyz_all,
    bool is_com,
    std::size_t indices_addr,
    std::size_t n_sel,
    std::size_t out_var_addr,
    std::size_t out_valid_addr)
{
    const int64_t* indices = reinterpret_cast<const int64_t*>(indices_addr);
    double* out_var = reinterpret_cast<double*>(out_var_addr);
    uint8_t* out_valid = reinterpret_cast<uint8_t*>(out_valid_addr);
    std::int64_t n = static_cast<std::int64_t>(n_sel);

    parallel_for(n, [&](std::int64_t idx_i) {
        std::int64_t r = indices[idx_i];
        const auto& sb = shoeboxes[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            out_valid[idx_i] = 0;
            return;
        }
        const float* d_ptr = sb.data.begin();
        const uint8_t* m_ptr = sb.mask.begin();
        double sum_v = 0.0;
        for (std::size_t i = 0; i < sz; ++i) {
            if (m_ptr[i] != 0) {
                sum_v += static_cast<double>(d_ptr[i]);
            }
        }
        if (sum_v <= 1.0) {
            out_valid[idx_i] = 0;
            return;
        }
        const auto& panel = detector[sb.panel];
        scitbx::vec3<double> s1_cent = is_com
            ? panel.get_pixel_lab_coord(scitbx::vec2<double>(xyz_all[r][0], xyz_all[r][1]))
            : s1_all[r];

        auto acc = sb.data.accessor();
        std::size_t nz = acc[0], ny = acc[1], nx = acc[2];
        int x0 = sb.bbox[0], y0 = sb.bbox[2];
        double sum_v_theta2 = 0.0;
        for (std::size_t j = 0; j < ny; ++j) {
            double py = static_cast<double>(y0 + static_cast<int>(j)) + 0.5;
            for (std::size_t i = 0; i < nx; ++i) {
                double col_v = 0.0;
                for (std::size_t k = 0; k < nz; ++k) {
                    std::size_t off = (k * ny + j) * nx + i;
                    if (m_ptr[off] != 0) {
                        col_v += static_cast<double>(d_ptr[off]);
                    }
                }
                if (col_v != 0.0) {
                    double px = static_cast<double>(x0 + static_cast<int>(i)) + 0.5;
                    scitbx::vec3<double> s1_px = panel.get_pixel_lab_coord(scitbx::vec2<double>(px, py));
                    double angle = s1_px.angle(s1_cent);
                    sum_v_theta2 += col_v * (angle * angle);
                }
            }
        }
        out_var[idx_i] = sum_v_theta2 / (sum_v - 1.0);
        out_valid[idx_i] = 1;
    });
}

std::size_t count_extended_sigma_m_frames(
    dials::af::const_ref<Shoebox<ProfileFloatType>> shoeboxes,
    std::size_t indices_addr,
    std::size_t n_sel,
    std::size_t pos_counts_addr)
{
    const int64_t* indices = reinterpret_cast<const int64_t*>(indices_addr);
    int64_t* pos_counts = reinterpret_cast<int64_t*>(pos_counts_addr);
    std::int64_t n = static_cast<std::int64_t>(n_sel);
    constexpr uint8_t mask_code = dials::model::Valid | dials::model::Foreground;

    parallel_for(n, [&](std::int64_t idx_i) {
        std::int64_t r = indices[idx_i];
        const auto& sb = shoeboxes[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            pos_counts[idx_i] = 0;
            return;
        }
        auto acc = sb.data.accessor();
        std::size_t nz = acc[0], ny_nx = acc[1] * acc[2];
        const float* d_ptr = sb.data.begin();
        const uint8_t* m_ptr = sb.mask.begin();
        int64_t cnt = 0;
        for (std::size_t k = 0; k < nz; ++k) {
            double fsum = 0.0;
            std::size_t z_off = k * ny_nx;
            for (std::size_t p = 0; p < ny_nx; ++p) {
                if (m_ptr[z_off + p] == mask_code) {
                    fsum += static_cast<double>(d_ptr[z_off + p]);
                }
            }
            if (fsum > 0.0) cnt++;
        }
        pos_counts[idx_i] = cnt;
    });

    std::size_t total = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        total += static_cast<std::size_t>(pos_counts[i]);
    }
    return total;
}

std::size_t fill_extended_sigma_m_frames(
    dials::af::const_ref<Shoebox<ProfileFloatType>> shoeboxes,
    std::size_t indices_addr,
    std::size_t n_sel,
    std::size_t pos_counts_addr,
    std::size_t phi_addr,
    std::size_t zeta_addr,
    double osc0,
    double dosc,
    std::size_t e1_addr,
    std::size_t e2_addr,
    std::size_t n_out_addr,
    std::size_t refl_idx_addr,
    std::size_t K_out_addr)
{
    const int64_t* indices = reinterpret_cast<const int64_t*>(indices_addr);
    const int64_t* pos_counts = reinterpret_cast<const int64_t*>(pos_counts_addr);
    const double* phi_all = reinterpret_cast<const double*>(phi_addr);
    const double* zeta_all = reinterpret_cast<const double*>(zeta_addr);
    double* e1_out = reinterpret_cast<double*>(e1_addr);
    double* e2_out = reinterpret_cast<double*>(e2_addr);
    double* n_out = reinterpret_cast<double*>(n_out_addr);
    int64_t* refl_idx_out = reinterpret_cast<int64_t*>(refl_idx_addr);
    double* K_out = reinterpret_cast<double*>(K_out_addr);

    std::int64_t n = static_cast<std::int64_t>(n_sel);
    std::vector<int64_t> offsets(n + 1, 0);
    std::vector<int64_t> valid_ids(n, -1);
    int64_t valid_refl_count = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        offsets[i + 1] = offsets[i] + pos_counts[i];
        if (pos_counts[i] > 0) {
            valid_ids[i] = valid_refl_count++;
        }
    }

    constexpr uint8_t mask_code = dials::model::Valid | dials::model::Foreground;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const double dphi_half = dosc * 0.5;

    parallel_for(n, [&](std::int64_t idx_i) {
        if (pos_counts[idx_i] == 0) return;
        std::int64_t r = indices[idx_i];
        const auto& sb = shoeboxes[r];
        auto acc = sb.data.accessor();
        std::size_t nz = acc[0], ny_nx = acc[1] * acc[2];
        const float* d_ptr = sb.data.begin();
        const uint8_t* m_ptr = sb.mask.begin();
        int z0 = sb.bbox[4];
        double phi_r = phi_all[r];
        double factor = std::abs(zeta_all[r]) * inv_sqrt2;
        int64_t pos = offsets[idx_i];
        int64_t v_id = valid_ids[idx_i];
        double k_sum = 0.0;

        for (std::size_t k = 0; k < nz; ++k) {
            double fsum = 0.0;
            std::size_t z_off = k * ny_nx;
            for (std::size_t p = 0; p < ny_nx; ++p) {
                if (m_ptr[z_off + p] == mask_code) {
                    fsum += static_cast<double>(d_ptr[z_off + p]);
                }
            }
            if (fsum > 0.0) {
                int f_idx = z0 + static_cast<int>(k);
                double phi0_f = osc0 + static_cast<double>(f_idx) * dosc;
                double tau_f = phi0_f + dphi_half - phi_r;
                e1_out[pos] = (tau_f + dphi_half) * factor;
                e2_out[pos] = (tau_f - dphi_half) * factor;
                n_out[pos] = fsum;
                refl_idx_out[pos] = v_id;
                k_sum += fsum;
                pos++;
            }
        }
        K_out[v_id] = k_sum;
    });

    return static_cast<std::size_t>(valid_refl_count);
}

dials::af::shared<scitbx::af::int6> parallel_compute_bbox_3d_fast(
    const dxtbx::model::BeamBase& beam,
    const dxtbx::model::Detector& detector,
    const dxtbx::model::Goniometer& gonio,
    const dxtbx::model::Scan& scan,
    double delta_d,
    double delta_m,
    const dials::af::const_ref<scitbx::vec3<double>>& s1,
    const dials::af::const_ref<scitbx::vec3<double>>& xyzcal_px,
    const dials::af::const_ref<std::size_t>& panel_arr)
{
    using dials::algorithms::profile_model::gaussian_rs::CoordinateSystem;
    DIALS_ASSERT(delta_d > 0.0);
    DIALS_ASSERT(delta_m > 0.0);
    DIALS_ASSERT(s1.size() == xyzcal_px.size());
    DIALS_ASSERT(s1.size() == panel_arr.size());

    scitbx::vec3<double> s0 = beam.get_s0();
    scitbx::vec3<double> m2 = gonio.get_rotation_axis();
    scitbx::vec2<double> osc = scan.get_oscillation();
    scitbx::vec2<int> img_range = scan.get_image_range();
    scitbx::vec2<int> array_range = scan.get_array_range();
    double osc0 = osc[0], osc1 = osc[1];
    int img0 = img_range[0];
    int arr0 = array_range[0], arr1 = array_range[1];

    dials::af::shared<scitbx::af::int6> result(
        s1.size(), dials::af::init_functor_null<scitbx::af::int6>()
    );
    scitbx::af::int6* out_ptr = result.begin();
    std::int64_t n = static_cast<std::int64_t>(s1.size());
    parallel_for(n, [&](std::int64_t i) {
        double frame = xyzcal_px[i][2];
        double phi = osc0 + ((frame + 1.0) - img0) * osc1;
        CoordinateSystem xcs(m2, s0, s1[i], phi);
        double point = delta_d;
        scitbx::af::double3 sdash1 = xcs.to_beam_vector(scitbx::af::double2(-point, -point));
        scitbx::af::double3 sdash2 = xcs.to_beam_vector(scitbx::af::double2(+point, -point));
        scitbx::af::double3 sdash3 = xcs.to_beam_vector(scitbx::af::double2(-point, +point));
        scitbx::af::double3 sdash4 = xcs.to_beam_vector(scitbx::af::double2(+point, +point));
        const dxtbx::model::Panel& panel = detector[panel_arr[i]];
        scitbx::af::double2 xy1 = panel.get_ray_intersection_px(sdash1);
        scitbx::af::double2 xy2 = panel.get_ray_intersection_px(sdash2);
        scitbx::af::double2 xy3 = panel.get_ray_intersection_px(sdash3);
        scitbx::af::double2 xy4 = panel.get_ray_intersection_px(sdash4);
        double phi1 = xcs.to_rotation_angle_fast(-delta_m);
        double phi2 = xcs.to_rotation_angle_fast(+delta_m);
        double z1 = (img0 + (phi1 - osc0) / osc1) - 1.0;
        double z2 = (img0 + (phi2 - osc0) / osc1) - 1.0;
        scitbx::af::double4 x(xy1[0], xy2[0], xy3[0], xy4[0]);
        scitbx::af::double4 y(xy1[1], xy2[1], xy3[1], xy4[1]);
        scitbx::af::double2 z(z1, z2);
        scitbx::af::int6 bbox(
            static_cast<int>(std::floor(scitbx::af::min(x))),
            static_cast<int>(std::ceil(scitbx::af::max(x))),
            static_cast<int>(std::floor(scitbx::af::min(y))),
            static_cast<int>(std::ceil(scitbx::af::max(y))),
            static_cast<int>(std::floor(scitbx::af::min(z))),
            static_cast<int>(std::ceil(scitbx::af::max(z)))
        );
        DIALS_ASSERT(bbox[4] <= frame && frame < bbox[5]);
        bbox[4] = std::max(bbox[4], arr0);
        bbox[4] = std::min(bbox[4], arr1 - 1);
        bbox[5] = std::min(bbox[5], arr1);
        bbox[5] = std::max(bbox[5], arr0 + 1);
        out_ptr[i] = bbox;
    });
    return result;
}

dials::af::shared<bool> parallel_mask_neighbouring(
    dials::af::ref<Shoebox<ProfileFloatType>> self,
    const dials::af::const_ref<cctbx::miller::index<>>& hkl,
    const dxtbx::model::BeamBase& beam,
    const dxtbx::model::Detector& detector,
    const dxtbx::model::Goniometer& goniometer,
    const dxtbx::model::Scan& scan,
    const dxtbx::model::CrystalBase& crystal)
{
    DIALS_ASSERT(self.size() == hkl.size());
    scitbx::vec3<double> s0 = beam.get_s0();
    double s0_len = s0.length();
    scitbx::vec3<double> m2 = goniometer.get_rotation_axis_datum();
    DIALS_ASSERT(!(m2[0] == 0 && m2[1] == 0 && m2[2] == 0));
    scitbx::mat3<double> S_inv = goniometer.get_setting_rotation().inverse();
    scitbx::mat3<double> AF_inv = crystal.get_A().inverse() * goniometer.get_fixed_rotation().inverse();
    scitbx::vec2<double> osc = scan.get_oscillation();
    scitbx::vec2<int> img_range = scan.get_image_range();
    double osc0 = osc[0], osc1 = osc[1];
    int img0 = img_range[0];

    dials::af::shared<bool> modified(self.size(), false);
    bool* mod_ptr = modified.begin();
    std::int64_t n = static_cast<std::int64_t>(self.size());
    constexpr uint8_t mask_code = dials::model::Valid | dials::model::Background;

    parallel_for(n, [&](std::int64_t r) {
        auto& sb = self[r];
        int x0 = sb.bbox[0], x1 = sb.bbox[1];
        int y0 = sb.bbox[2], y1 = sb.bbox[3];
        int z0 = sb.bbox[4], z1 = sb.bbox[5];
        DIALS_ASSERT(x0 < x1);
        DIALS_ASSERT(y0 < y1);
        DIALS_ASSERT(z0 < z1);
        if (sb.mask.size() == 0) return;
        std::size_t xsize = sb.xsize(), ysize = sb.ysize(), zsize = sb.zsize();
        const auto& panel = detector[sb.panel];
        const auto& h_ref = hkl[r];
        int h0 = h_ref[0], k0 = h_ref[1], l0 = h_ref[2];

        std::vector<scitbx::mat3<double>> M_z(zsize + 1);
        for (std::size_t kz = 0; kz <= zsize; ++kz) {
            double z_frame = static_cast<double>(z0 + static_cast<int>(kz));
            double angle = osc0 + ((z_frame + 1.0) - img0) * osc1;
            scitbx::mat3<double> R = scitbx::math::r3_rotation::axis_and_angle_as_matrix(m2, angle);
            M_z[kz] = (AF_inv * R.transpose()) * S_inv;
        }

        uint8_t* m_ptr = sb.mask.begin();
        std::size_t stride_z = ysize * xsize;
        bool mod = false;
        std::vector<uint8_t> bad_z(zsize + 1);

        for (std::size_t y = 0; y < ysize; ++y) {
            double py = static_cast<double>(y0 + static_cast<int>(y)) + 0.5;
            std::size_t row_off = y * xsize;
            for (std::size_t x = 0; x < xsize; ++x) {
                double px = static_cast<double>(x0 + static_cast<int>(x)) + 0.5;
                scitbx::vec3<double> s1 = panel.get_pixel_lab_coord(scitbx::vec2<double>(px, py)).normalize() * s0_len;
                scitbx::vec3<double> r_vec = s1 - s0;
                for (std::size_t kz = 0; kz <= zsize; ++kz) {
                    scitbx::vec3<double> ph = M_z[kz] * r_vec;
                    int hk = static_cast<int>(std::floor(ph[0] + 0.5));
                    int kk = static_cast<int>(std::floor(ph[1] + 0.5));
                    int lk = static_cast<int>(std::floor(ph[2] + 0.5));
                    bad_z[kz] = (hk != h0 || kk != k0 || lk != l0) ? 1 : 0;
                }
                for (std::size_t z = 0; z < zsize; ++z) {
                    if (bad_z[z] | bad_z[z + 1]) {
                        m_ptr[z * stride_z + row_off + x] = mask_code;
                        mod = true;
                    }
                }
            }
        }
        mod_ptr[r] = mod;
    });
    return modified;
}

boost::python::tuple match_with_reference_indices(
    const dials::af::const_ref<int>& id1,
    const dials::af::const_ref<cctbx::miller::index<>>& h1,
    const dials::af::const_ref<bool>& e1,
    const dials::af::const_ref<scitbx::vec3<double>>& xyz1,
    const dials::af::const_ref<std::size_t>& p1,
    const dials::af::const_ref<int>& id2,
    const dials::af::const_ref<cctbx::miller::index<>>& h2,
    const dials::af::const_ref<bool>& e2,
    const dials::af::const_ref<scitbx::vec3<double>>& xyz2,
    const dials::af::const_ref<std::size_t>& p2)
{
    std::size_t n1 = id1.size();
    std::size_t n2 = id2.size();
    DIALS_ASSERT(h1.size() == n1 && e1.size() == n1 && xyz1.size() == n1 && p1.size() == n1);
    DIALS_ASSERT(h2.size() == n2 && e2.size() == n2 && xyz2.size() == n2 && p2.size() == n2);

    dials::af::shared<std::size_t> sind;
    dials::af::shared<std::size_t> oind;
    if (n1 == 0 || n2 == 0) {
        return boost::python::make_tuple(sind, oind);
    }

    std::size_t cap = 16;
    while (cap < n2 * 2) cap <<= 1;
    std::size_t mask = cap - 1;
    std::vector<int32_t> head_b(cap, -1);
    std::vector<int32_t> tail_b(n2, -1);
    std::vector<int32_t> next_b(n2, -1);

    auto hash_key = [](int h, int k, int l, int e, int id, std::size_t p) -> std::size_t {
        uint64_t u = static_cast<uint64_t>(static_cast<uint32_t>(h)) * 0x9e3779b97f4a7c15ULL;
        u ^= static_cast<uint64_t>(static_cast<uint32_t>(k)) * 0xbf58476d1ce4e5b9ULL;
        u ^= static_cast<uint64_t>(static_cast<uint32_t>(l)) * 0x94d049bb133111ebULL;
        u ^= static_cast<uint64_t>(static_cast<uint32_t>((id << 8) ^ (static_cast<int>(p) << 1) ^ e)) * 0x517cc1b727220a95ULL;
        u ^= (u >> 30);
        return static_cast<std::size_t>(u);
    };

    for (std::size_t j = 0; j < n2; ++j) {
        const auto& hj = h2[j];
        int ej = e2[j] ? 1 : 0;
        int idj = id2[j];
        std::size_t pj = p2[j];
        std::size_t slot = hash_key(hj[0], hj[1], hj[2], ej, idj, pj) & mask;
        while (true) {
            int32_t h_idx = head_b[slot];
            if (h_idx == -1) {
                head_b[slot] = static_cast<int32_t>(j);
                tail_b[j] = static_cast<int32_t>(j);
                break;
            }
            const auto& h_head = h2[h_idx];
            if (h_head[0] == hj[0] && h_head[1] == hj[1] && h_head[2] == hj[2] &&
                (e2[h_idx] ? 1 : 0) == ej && id2[h_idx] == idj && p2[h_idx] == pj) {
                int32_t t_idx = tail_b[h_idx];
                next_b[t_idx] = static_cast<int32_t>(j);
                tail_b[h_idx] = static_cast<int32_t>(j);
                break;
            }
            slot = (slot + 1) & mask;
        }
    }

    std::vector<int32_t> best_i_for_other(n2, -1);
    std::vector<double> best_d_for_other(n2, 0.0);

    for (std::size_t i = 0; i < n1; ++i) {
        const auto& hi = h1[i];
        int ei = e1[i] ? 1 : 0;
        int idi = id1[i];
        std::size_t pi = p1[i];
        std::size_t slot = hash_key(hi[0], hi[1], hi[2], ei, idi, pi) & mask;
        int32_t h_idx = -1;
        while (true) {
            int32_t cand = head_b[slot];
            if (cand == -1) break;
            const auto& hc = h2[cand];
            if (hc[0] == hi[0] && hc[1] == hi[1] && hc[2] == hi[2] &&
                (e2[cand] ? 1 : 0) == ei && id2[cand] == idi && p2[cand] == pi) {
                h_idx = cand;
                break;
            }
            slot = (slot + 1) & mask;
        }
        if (h_idx == -1) continue;

        const auto& xyzi = xyz1[i];
        int32_t best_j = -1;
        double min_d = 0.0;
        for (int32_t j = h_idx; j != -1; j = next_b[j]) {
            const auto& xyzj = xyz2[j];
            double dx = xyzi[0] - xyzj[0];
            double dy = xyzi[1] - xyzj[1];
            double dz = xyzi[2] - xyzj[2];
            double d = dx * dx + dy * dy + dz * dz;
            if (best_j == -1 || d < min_d) {
                best_j = j;
                min_d = d;
            }
        }
        if (best_i_for_other[best_j] == -1 || min_d < best_d_for_other[best_j]) {
            best_i_for_other[best_j] = static_cast<int32_t>(i);
            best_d_for_other[best_j] = min_d;
        }
    }

    std::vector<int32_t> matched_j_for_self(n1, -1);
    std::size_t n_matched = 0;
    for (std::size_t j = 0; j < n2; ++j) {
        int32_t i_match = best_i_for_other[j];
        if (i_match != -1) {
            matched_j_for_self[i_match] = static_cast<int32_t>(j);
            n_matched++;
        }
    }

    sind.resize(n_matched);
    oind.resize(n_matched);
    std::size_t* s_ptr = sind.begin();
    std::size_t* o_ptr = oind.begin();
    std::size_t k = 0;
    for (std::size_t i = 0; i < n1; ++i) {
        int32_t j_match = matched_j_for_self[i];
        if (j_match != -1) {
            s_ptr[k] = i;
            o_ptr[k] = static_cast<std::size_t>(j_match);
            k++;
        }
    }
    return boost::python::make_tuple(sind, oind);
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
    def("extract_wide_shoeboxes_slice_from_volume", &dials_gpu::extract_wide_shoeboxes_slice_from_volume);
    def("parallel_compute_mask", &dials_gpu::parallel_compute_mask);
    def("parallel_compute_mask_vec3", &dials_gpu::parallel_compute_mask_vec3);
    def("parallel_compute_mask_3d_fast", &dials_gpu::parallel_compute_mask_3d_fast);
    def("parallel_compute_bbox_3d_fast", &dials_gpu::parallel_compute_bbox_3d_fast);
    def("parallel_pre_background_mask_flags", &dials_gpu::parallel_pre_background_mask_flags);
    def("parallel_post_mask_counts", &dials_gpu::parallel_post_mask_counts);
    def("parallel_check_overloads_single_panel", &dials_gpu::parallel_check_overloads_single_panel);
    def("parallel_fit_multi", &dials_gpu::parallel_fit_multi);
    def("parallel_model_multi", &dials_gpu::parallel_model_multi);
    def("compute_beam_divergence_variances", &dials_gpu::compute_beam_divergence_variances);
    def("count_extended_sigma_m_frames", &dials_gpu::count_extended_sigma_m_frames);
    def("fill_extended_sigma_m_frames", &dials_gpu::fill_extended_sigma_m_frames);
    def("parallel_mask_neighbouring", &dials_gpu::parallel_mask_neighbouring);
    def("match_with_reference_indices", &dials_gpu::match_with_reference_indices);
}
