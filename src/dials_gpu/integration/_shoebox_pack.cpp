#include <boost/python.hpp>
#include <dials/algorithms/profile_model/gaussian_rs/bbox_calculator.h>
#include <dials/algorithms/profile_model/gaussian_rs/partiality_calculator.h>
#include <dials/algorithms/spot_prediction/pixel_to_miller_index.h>
#include <dials/algorithms/spot_prediction/reflection_predictor.h>
#include <cctbx/miller.h>
#include <dials/model/data/shoebox.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/algorithms/profile_model/gaussian_rs/mask_calculator.h>
#include <dials/algorithms/profile_model/gaussian_rs/modeller.h>
#include <dials/algorithms/profile_model/modeller/multi_experiment_modeller.h>
#include <scitbx/glmtbx/robust_glm.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <type_traits>
#include <typeinfo>

namespace dials_gpu {

using dials::model::Shoebox;
using dials::ProfileFloatType;

constexpr int MAX_WORKER_THREADS = 64;

inline int num_worker_threads() {
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 8;
    return std::min(hw, MAX_WORKER_THREADS);
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
    std::mutex exc_mtx;
    std::exception_ptr first_exc = nullptr;
    auto worker = [&](int tid) {
        try {
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
        } catch (...) {
            std::lock_guard<std::mutex> lock(exc_mtx);
            if (!first_exc) {
                first_exc = std::current_exception();
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
    if (first_exc) {
        std::rethrow_exception(first_exc);
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

inline bool solve_robust_poisson_hist(
    const int32_t* h_r,
    int max_bin,
    const std::vector<double>& tail,
    int64_t cnt,
    double mean0,
    double c,
    double tolerance,
    std::size_t max_iter,
    double& mean_out)
{
    typedef scitbx::glmtbx::poisson family;
    if (cnt <= 0 || !(mean0 > 0.0) || !(c > 0.0) || !(tolerance > 0.0) || max_iter == 0) {
        return false;
    }
    double beta = std::log(mean0);
    std::size_t niter = 0;
    for (niter = 0; niter < max_iter; ++niter) {
        double eta = beta;
        double mu = family::linkinv(eta);
        double var = family::variance(mu);
        double dmu = family::dmu_deta(eta);
        double phi = family::dispersion();
        if (!(phi > 0.0) || !(var > 0.0) || !std::isfinite(mu)) {
            return false;
        }
        double svar = std::sqrt(phi * var);
        scitbx::glmtbx::expectation<family> epsi(mu, svar, c);
        double b = epsi.epsi2 * dmu * dmu / svar;
        double scale_q = dmu / svar;
        double U = 0.0;
        for (int k = 0; k <= max_bin; ++k) {
            int32_t hk = h_r[k];
            if (hk > 0) {
                double res = (static_cast<double>(k) - mu) / svar;
                double psi = scitbx::glmtbx::huber(res, c);
                double q = (psi - epsi.epsi1) * scale_q;
                U += static_cast<double>(hk) * q;
            }
        }
        for (double y : tail) {
            double res = (y - mu) / svar;
            double psi = scitbx::glmtbx::huber(res, c);
            double q = (psi - epsi.epsi1) * scale_q;
            U += q;
        }
        double H = static_cast<double>(cnt) * b;
        if (H == 0.0 || !std::isfinite(H)) {
            return false;
        }
        U = U / H;
        double sum_delta_sq = U * U;
        double sum_beta_sq = beta * beta;
        beta += U;
        double error = std::sqrt(sum_delta_sq / std::max(1e-10, sum_beta_sq));
        if (error < tolerance) {
            break;
        }
    }
    if (niter >= max_iter || !(beta > -300.0 && beta < 300.0)) {
        return false;
    }
    mean_out = std::exp(beta);
    return true;
}

boost::python::tuple compute_glm_constant3d_fast(
    dials::af::ref<Shoebox<ProfileFloatType>> self,
    dials::af::ref<std::size_t> flags,
    int min_pixels,
    double tuning_constant,
    std::size_t max_iter,
    int mask_bg_valid,
    int mask_overlapped,
    int mask_bg_used)
{
    uint8_t m_bg_val = static_cast<uint8_t>(mask_bg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    uint8_t m_used = static_cast<uint8_t>(mask_bg_used);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    dials::af::shared<double> bg_mean(self.size(), 0.0);
    dials::af::shared<bool> success(self.size(), false);
    double* mean_ptr = bg_mean.begin();
    bool* succ_ptr = success.begin();
    std::size_t* fl_ptr = flags.begin();

    parallel_for(n, [&](std::int64_t r, int) {
        auto& sb = self[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            fl_ptr[r] |= dials::af::DontIntegrate;
            return;
        }
        const float* d_ptr = sb.data.begin();
        uint8_t* m_ptr = sb.mask.begin();
        float* bg_ptr = sb.background.begin();

        int32_t h_r[128];
        std::memset(h_r, 0, sizeof(h_r));
        int max_bin = -1;
        int64_t cnt = 0;
        int64_t tail_cnt = 0;
        bool neg = false;

        for (std::size_t i = 0; i < sz; ++i) {
            uint8_t m = m_ptr[i];
            if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                cnt++;
                float v = d_ptr[i];
                if (v < 0.0f) {
                    neg = true;
                } else {
                    int iv = static_cast<int>(v);
                    if (static_cast<float>(iv) == v && iv >= 0 && iv < 128) {
                        h_r[iv]++;
                        if (iv > max_bin) max_bin = iv;
                    } else {
                        tail_cnt++;
                    }
                }
            }
        }

        if (cnt < min_pixels || neg) {
            if (sb.background.size() == sz) {
                std::fill(bg_ptr, bg_ptr + sz, 0.0f);
            }
            fl_ptr[r] |= dials::af::DontIntegrate;
            return;
        }

        std::vector<double> tail;
        double med = 0.0;
        if (tail_cnt == 0) {
            int64_t target = cnt / 2;
            int64_t acc = 0;
            for (int k = 0; k <= max_bin; ++k) {
                acc += h_r[k];
                if (acc > target) {
                    med = static_cast<double>(k);
                    break;
                }
            }
        } else {
            tail.reserve(static_cast<std::size_t>(tail_cnt));
            std::vector<double> tmp;
            tmp.reserve(static_cast<std::size_t>(cnt));
            for (std::size_t i = 0; i < sz; ++i) {
                uint8_t m = m_ptr[i];
                if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                    float v = d_ptr[i];
                    double dv = static_cast<double>(v);
                    tmp.push_back(dv);
                    int iv = static_cast<int>(v);
                    if (!(static_cast<float>(iv) == v && iv >= 0 && iv < 128)) {
                        tail.push_back(dv);
                    }
                }
            }
            std::nth_element(tmp.begin(), tmp.begin() + cnt / 2, tmp.end());
            med = tmp[cnt / 2];
        }
        if (med <= 0.0) med = 1.0;

        double m_val_d = 0.0;
        bool ok = solve_robust_poisson_hist(
            h_r, max_bin, tail, cnt, med, tuning_constant, 1e-3, max_iter, m_val_d
        );
        if (ok) {
            float m_val_f = static_cast<float>(m_val_d);
            std::fill(bg_ptr, bg_ptr + sz, m_val_f);
            for (std::size_t i = 0; i < sz; ++i) {
                uint8_t m = m_ptr[i];
                if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                    m_ptr[i] = m | m_used;
                }
            }
            mean_ptr[r] = static_cast<double>(m_val_f);
            succ_ptr[r] = true;
        } else {
            if (sb.background.size() == sz) {
                std::fill(bg_ptr, bg_ptr + sz, 0.0f);
            }
            fl_ptr[r] |= dials::af::DontIntegrate;
        }
    });

    return boost::python::make_tuple(bg_mean, success);
}

boost::python::tuple compute_glm_constant2d_fast(
    dials::af::ref<Shoebox<ProfileFloatType>> self,
    dials::af::ref<std::size_t> flags,
    int min_pixels,
    double tuning_constant,
    std::size_t max_iter,
    int mask_bg_valid,
    int mask_overlapped,
    int mask_bg_used)
{
    uint8_t m_bg_val = static_cast<uint8_t>(mask_bg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    uint8_t m_used = static_cast<uint8_t>(mask_bg_used);
    uint8_t m_not_used = static_cast<uint8_t>(255 ^ m_used);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    dials::af::shared<double> bg_mean(self.size(), 0.0);
    dials::af::shared<bool> success(self.size(), false);
    double* mean_ptr = bg_mean.begin();
    bool* succ_ptr = success.begin();
    std::size_t* fl_ptr = flags.begin();

    parallel_for(n, [&](std::int64_t r, int) {
        auto& sb = self[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            fl_ptr[r] |= dials::af::DontIntegrate;
            return;
        }
        auto acc = sb.data.accessor();
        std::size_t nz = acc[0];
        std::size_t slice_sz = acc[1] * acc[2];
        const float* d_ptr = sb.data.begin();
        uint8_t* m_ptr = sb.mask.begin();
        float* bg_ptr = sb.background.begin();

        bool all_ok = true;
        double sum_bg = 0.0;
        int64_t total_used = 0;

        for (std::size_t k = 0; k < nz; ++k) {
            std::size_t z_off = k * slice_sz;
            const float* dz = d_ptr + z_off;
            uint8_t* mz = m_ptr + z_off;
            float* bgz = bg_ptr + z_off;

            int32_t h_r[128];
            std::memset(h_r, 0, sizeof(h_r));
            int max_bin = -1;
            int64_t cnt = 0;
            int64_t tail_cnt = 0;
            bool neg = false;

            for (std::size_t i = 0; i < slice_sz; ++i) {
                uint8_t m = mz[i];
                if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                    cnt++;
                    float v = dz[i];
                    if (v < 0.0f) {
                        neg = true;
                    } else {
                        int iv = static_cast<int>(v);
                        if (static_cast<float>(iv) == v && iv >= 0 && iv < 128) {
                            h_r[iv]++;
                            if (iv > max_bin) max_bin = iv;
                        } else {
                            tail_cnt++;
                        }
                    }
                }
            }

            if (cnt < min_pixels || neg) {
                all_ok = false;
                break;
            }

            std::vector<double> tail;
            double med = 0.0;
            if (tail_cnt == 0) {
                int64_t target = cnt / 2;
                int64_t acc_c = 0;
                for (int b = 0; b <= max_bin; ++b) {
                    acc_c += h_r[b];
                    if (acc_c > target) {
                        med = static_cast<double>(b);
                        break;
                    }
                }
            } else {
                tail.reserve(static_cast<std::size_t>(tail_cnt));
                std::vector<double> tmp;
                tmp.reserve(static_cast<std::size_t>(cnt));
                for (std::size_t i = 0; i < slice_sz; ++i) {
                    uint8_t m = mz[i];
                    if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                        float v = dz[i];
                        double dv = static_cast<double>(v);
                        tmp.push_back(dv);
                        int iv = static_cast<int>(v);
                        if (!(static_cast<float>(iv) == v && iv >= 0 && iv < 128)) {
                            tail.push_back(dv);
                        }
                    }
                }
                std::nth_element(tmp.begin(), tmp.begin() + cnt / 2, tmp.end());
                med = tmp[cnt / 2];
            }
            if (med <= 0.0) med = 1.0;

            double m_val_d = 0.0;
            bool ok = solve_robust_poisson_hist(
                h_r, max_bin, tail, cnt, med, tuning_constant, 1e-3, max_iter, m_val_d
            );
            if (!ok) {
                all_ok = false;
                break;
            }
            float m_val_f = static_cast<float>(m_val_d);
            std::fill(bgz, bgz + slice_sz, m_val_f);
            for (std::size_t i = 0; i < slice_sz; ++i) {
                uint8_t m = mz[i];
                if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                    mz[i] = m | m_used;
                }
            }
            sum_bg += static_cast<double>(m_val_f) * static_cast<double>(cnt);
            total_used += cnt;
        }

        if (all_ok && total_used > 0) {
            mean_ptr[r] = sum_bg / static_cast<double>(total_used);
            succ_ptr[r] = true;
        } else {
            if (sb.background.size() == sz) {
                std::fill(bg_ptr, bg_ptr + sz, 0.0f);
            }
            for (std::size_t i = 0; i < sz; ++i) {
                m_ptr[i] &= m_not_used;
            }
            fl_ptr[r] |= dials::af::DontIntegrate;
        }
    });

    return boost::python::make_tuple(bg_mean, success);
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

// 3c. Complete parallel C++ centroid calculation returning flex.vec3_double directly
boost::python::tuple compute_centroid_fast(
    dials::af::const_ref<Shoebox<ProfileFloatType>> self,
    int mask_fg_valid,
    int mask_overlapped,
    bool has_single_panel_px2mm,
    int px2mm_mode, // 0 = SimplePxMmStrategy, 1 = ParallaxCorrectedPxMmStrategy
    double px_size_x,
    double px_size_y,
    double mu,
    double t0,
    scitbx::vec3<double> fast_axis,
    scitbx::vec3<double> slow_axis,
    scitbx::vec3<double> origin,
    double osc0,
    double osc_w,
    double arr0)
{
    uint8_t m_fg_val = static_cast<uint8_t>(mask_fg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    std::int64_t n = static_cast<std::int64_t>(self.size());

    dials::af::shared<scitbx::vec3<double>> px_pos(
        self.size(), dials::af::init_functor_null<scitbx::vec3<double>>()
    );
    dials::af::shared<scitbx::vec3<double>> px_var(
        self.size(), dials::af::init_functor_null<scitbx::vec3<double>>()
    );
    scitbx::vec3<double>* p_pos = px_pos.begin();
    scitbx::vec3<double>* p_var = px_var.begin();

    dials::af::shared<scitbx::vec3<double>> mm_pos;
    dials::af::shared<scitbx::vec3<double>> mm_var;
    scitbx::vec3<double>* m_pos = nullptr;
    scitbx::vec3<double>* m_var = nullptr;
    if (has_single_panel_px2mm) {
        mm_pos = dials::af::shared<scitbx::vec3<double>>(
            self.size(), dials::af::init_functor_null<scitbx::vec3<double>>()
        );
        mm_var = dials::af::shared<scitbx::vec3<double>>(
            self.size(), dials::af::init_functor_null<scitbx::vec3<double>>()
        );
        m_pos = mm_pos.begin();
        m_var = mm_var.begin();
    }

    constexpr double half_sq = 1.0 / 12.0;
    double px_size_x2 = px_size_x * px_size_x;
    double px_size_y2 = px_size_y * px_size_y;
    double osc_w2 = osc_w * osc_w;
    scitbx::vec3<double> n_vec = fast_axis.cross(slow_axis);
    if (origin * n_vec < 0.0) {
        n_vec = -n_vec;
    }

    parallel_for(n, [&](std::int64_t r) {
        const auto& sb = self[r];
        auto b = sb.bbox;
        int x0 = b[0], x1 = b[1], y0 = b[2], y1 = b[3], z0 = b[4], z1 = b[5];
        double bx = static_cast<double>(x0);
        double by = static_cast<double>(y0);
        double bz = static_cast<double>(sb.flat ? ((z1 + z0) / 2) : z0);
        std::size_t sz = sb.data.size();

        double cx = static_cast<double>(x1 + x0) * 0.5;
        double cy = static_cast<double>(y1 + y0) * 0.5;
        double cz = static_cast<double>(z1 + z0) * 0.5;
        double vx = 0.0, vy = 0.0, vz = 0.0;

        if (sz > 0) {
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
            if (cnt > 0 && sum_w > 0.0) {
                double inv_w = 1.0 / sum_w;
                double mean_x = sum_wx * inv_w;
                double mean_y = sum_wy * inv_w;
                double mean_z = sum_wz * inv_w;
                cx = mean_x + bx;
                cy = mean_y + by;
                cz = mean_z + bz;
                double denom = (sum_w * sum_w) - sum_wsq;
                if (cnt > 1 && denom > 0.0) {
                    double sum_wdx2 = 0.0, sum_wdy2 = 0.0, sum_wdz2 = 0.0;
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
                    double scale = 1.0 / denom;
                    vx = sum_wdx2 * scale + half_sq;
                    vy = sum_wdy2 * scale + half_sq;
                    vz = sum_wdz2 * scale + half_sq;
                }
            }
        }
        p_pos[r] = scitbx::vec3<double>(cx, cy, cz);
        p_var[r] = scitbx::vec3<double>(vx, vy, vz);

        if (has_single_panel_px2mm) {
            double mx = cx * px_size_x;
            double my = cy * px_size_y;
            if (px2mm_mode == 1) {
                scitbx::vec3<double> s1 = origin + mx * fast_axis + my * slow_axis;
                double slen = s1.length();
                if (slen > 0.0) {
                    scitbx::vec3<double> s1_n = s1 * (1.0 / slen);
                    double cos_t = s1_n * n_vec;
                    if (mu > 0.0 && cos_t != 0.0) {
                        double inv_mu = 1.0 / mu;
                        double t0_over_cos_t = t0 / cos_t;
                        double atten_o = inv_mu - (t0_over_cos_t + inv_mu) * std::exp(-mu * t0_over_cos_t);
                        mx -= (s1_n * fast_axis) * atten_o;
                        my -= (s1_n * slow_axis) * atten_o;
                    }
                }
            }
            double mz = (osc_w == 0.0 && osc0 == 0.0) ? 0.0 : (osc0 + (cz - arr0) * osc_w);
            m_pos[r] = scitbx::vec3<double>(mx, my, mz);
            m_var[r] = scitbx::vec3<double>(vx * px_size_x2, vy * px_size_y2, vz * osc_w2);
        }
    });

    if (has_single_panel_px2mm) {
        return boost::python::make_tuple(px_pos, px_var, mm_pos, mm_var);
    }
    return boost::python::make_tuple(px_pos, px_var);
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

// 4c. Complete parallel C++ summation integration returning flex.double directly and updating flags
boost::python::tuple compute_summation_fast(
    dials::af::const_ref<Shoebox<ProfileFloatType>> self,
    dials::af::ref<std::size_t> flags,
    int mask_fg,
    int mask_fg_valid,
    int mask_overlapped,
    int mask_bg_used_valid)
{
    uint8_t m_fg = static_cast<uint8_t>(mask_fg);
    uint8_t m_fg_val = static_cast<uint8_t>(mask_fg_valid);
    uint8_t m_ovlp = static_cast<uint8_t>(mask_overlapped);
    uint8_t m_bg_used = static_cast<uint8_t>(mask_bg_used_valid);
    std::int64_t n = static_cast<std::int64_t>(self.size());
    DIALS_ASSERT(flags.size() == self.size());

    dials::af::shared<double> intensity_val(self.size(), dials::af::init_functor_null<double>());
    dials::af::shared<double> intensity_var(self.size(), dials::af::init_functor_null<double>());
    dials::af::shared<double> background_val(self.size(), dials::af::init_functor_null<double>());
    dials::af::shared<double> background_var(self.size(), dials::af::init_functor_null<double>());
    double* i_val = intensity_val.begin();
    double* i_var = intensity_var.begin();
    double* b_val = background_val.begin();
    double* b_var = background_var.begin();
    std::size_t* fl_ptr = flags.begin();

    constexpr std::size_t clear_mask = ~(dials::af::IntegratedSum | dials::af::FailedDuringSummation);

    parallel_for(n, [&](std::int64_t r) {
        const auto& sb = self[r];
        std::size_t sz = sb.data.size();
        if (sz == 0) {
            i_val[r] = 0.0;
            i_var[r] = 0.0;
            b_val[r] = 0.0;
            b_var[r] = 0.0;
            fl_ptr[r] |= dials::af::FailedDuringSummation;
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
        double m_ratio = (bg_cnt > 0) ? (static_cast<double>(fg_cnt) / static_cast<double>(bg_cnt)) : 0.0;
        double ival = sum_p - sum_b;
        double bg_v = std::abs(sum_b) * (1.0 + m_ratio);
        i_val[r] = ival;
        i_var[r] = std::abs(ival) + bg_v;
        b_val[r] = sum_b;
        b_var[r] = bg_v;
        if (bad_cnt == 0) {
            fl_ptr[r] |= dials::af::IntegratedSum;
        } else {
            fl_ptr[r] |= dials::af::FailedDuringSummation;
        }
    });

    return boost::python::make_tuple(intensity_val, intensity_var, background_val, background_var);
}

// 5. Fast parallel shoebox extraction directly from a 3D (B, H, W) raw volume
inline int32_t read_raw_pixel_i32(
    const void* row_ptr, int x, int dtype_code, int32_t invalid_top
) {
    if (dtype_code == 4) {
        int32_t v = static_cast<const int32_t*>(row_ptr)[x];
        if (__builtin_expect(invalid_top > 0 && v >= invalid_top - 2, 0)) {
            if (v == invalid_top - 1) return -1;
            if (v == invalid_top - 2) return -2;
        }
        return v;
    } else if (dtype_code == 1) {
        int32_t v = static_cast<int32_t>(static_cast<const uint16_t*>(row_ptr)[x]);
        if (__builtin_expect(invalid_top > 0 && v >= invalid_top - 2, 0)) {
            return v - invalid_top;
        }
        return v;
    } else if (dtype_code == 2) {
        int32_t v = static_cast<int32_t>(static_cast<const int16_t*>(row_ptr)[x]);
        if (__builtin_expect(invalid_top > 0 && v >= invalid_top - 2, 0)) {
            return v - invalid_top;
        }
        return v;
    } else {
        int32_t v = static_cast<int32_t>(static_cast<const uint32_t*>(row_ptr)[x]);
        if (__builtin_expect(invalid_top > 0 && v >= invalid_top - 2, 0)) {
            if (v == invalid_top - 1) return -1;
            if (v == invalid_top - 2) return -2;
        }
        return v;
    }
}

std::size_t extract_shoeboxes_from_volume_typed(
    dials::af::ref<Shoebox<ProfileFloatType>> self,
    std::size_t vol_addr,
    std::size_t static_mask_addr,
    std::size_t rejected_addr,
    int dtype_code,
    int invalid_top,
    int tr_min,
    int tr_max,
    int frame0,
    int frame1,
    std::size_t H,
    std::size_t W,
    std::size_t valid_flags_addr)
{
    const uint8_t* vol_bytes = reinterpret_cast<const uint8_t*>(vol_addr);
    const uint8_t* static_mask = reinterpret_cast<const uint8_t*>(static_mask_addr);
    const uint8_t* rejected = reinterpret_cast<const uint8_t*>(rejected_addr);
    uint8_t* valid_flags = reinterpret_cast<uint8_t*>(valid_flags_addr);
    std::int64_t n = static_cast<std::int64_t>(self.size());
    int yi = static_cast<int>(H);
    int xi = static_cast<int>(W);
    std::size_t elem_size = (dtype_code <= 2) ? 2 : 4;
    std::size_t row_bytes = W * elem_size;
    std::size_t stride_f_bytes = H * row_bytes;
    int32_t inv_top_i32 = static_cast<int32_t>(invalid_top);

    parallel_for(n, [&](std::int64_t r) {
        auto& sbox = self[r];
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
            sbox.allocate_data();
            valid_flags[r] = 0;
            return;
        }
        valid_flags[r] = 1;
        bool full_interior = (!sbox.flat && xb == 0 && xe == xs && yb == 0 && ye == ys);
        if (full_interior) {
            dials::af::c_grid<3> grid(zs, ys, xs);
            sbox.data = dials::af::versa<ProfileFloatType, dials::af::c_grid<3>>(
                grid, dials::af::init_functor_null<ProfileFloatType>()
            );
            sbox.mask = dials::af::versa<uint8_t, dials::af::c_grid<3>>(
                grid, dials::af::init_functor_null<uint8_t>()
            );
            sbox.background = dials::af::versa<ProfileFloatType, dials::af::c_grid<3>>(
                grid, 0.0f
            );
        } else {
            sbox.allocate_data();
            sbox.allocate_background();
        }
        float* sdata = sbox.data.begin();
        uint8_t* smask = sbox.mask.begin();
        std::size_t s_stride_z = static_cast<std::size_t>(ys) * static_cast<std::size_t>(xs);
        std::size_t s_stride_y = static_cast<std::size_t>(xs);

        if (sbox.flat) {
            for (int z = 0; z < zs; ++z) {
                int vf = (z0 + z) - frame0;
                bool is_rej = (rejected[vf] != 0);
                const uint8_t* vol_f = vol_bytes + static_cast<std::size_t>(vf) * stride_f_bytes;
                for (int y = yb; y < ye; ++y) {
                    const void* vol_row = vol_f + static_cast<std::size_t>(y + y0) * row_bytes;
                    const uint8_t* sm_row = static_mask + static_cast<std::size_t>(y + y0) * W;
                    float* sd_row = sdata + static_cast<std::size_t>(y) * s_stride_y;
                    uint8_t* smk_row = smask + static_cast<std::size_t>(y) * s_stride_y;
                    if (is_rej) {
                        for (int x = xb; x < xe; ++x) {
                            int32_t raw_val = read_raw_pixel_i32(vol_row, x + x0, dtype_code, inv_top_i32);
                            sd_row[x] += static_cast<float>(raw_val);
                            smk_row[x] = 0;
                        }
                    } else {
                        for (int x = xb; x < xe; ++x) {
                            int32_t raw_val = read_raw_pixel_i32(vol_row, x + x0, dtype_code, inv_top_i32);
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
                const uint8_t* vol_f = vol_bytes + static_cast<std::size_t>(vf) * stride_f_bytes;
                float* sd_z = sdata + static_cast<std::size_t>(z) * s_stride_z;
                uint8_t* smk_z = smask + static_cast<std::size_t>(z) * s_stride_z;
                for (int y = yb; y < ye; ++y) {
                    const void* vol_row = vol_f + static_cast<std::size_t>(y + y0) * row_bytes;
                    const uint8_t* sm_row = static_mask + static_cast<std::size_t>(y + y0) * W;
                    float* sd_row = sd_z + static_cast<std::size_t>(y) * s_stride_y;
                    uint8_t* smk_row = smk_z + static_cast<std::size_t>(y) * s_stride_y;
                    if (is_rej) {
                        for (int x = xb; x < xe; ++x) {
                            int32_t raw_val = read_raw_pixel_i32(vol_row, x + x0, dtype_code, inv_top_i32);
                            sd_row[x] = static_cast<float>(raw_val);
                            smk_row[x] = 0;
                        }
                    } else {
                        for (int x = xb; x < xe; ++x) {
                            int32_t raw_val = read_raw_pixel_i32(vol_row, x + x0, dtype_code, inv_top_i32);
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
    return extract_shoeboxes_from_volume_typed(
        self, vol_addr, static_mask_addr, rejected_addr,
        4, 0, tr_min, tr_max, frame0, frame1, H, W, valid_flags_addr
    );
}

void extract_wide_shoeboxes_slice_from_volume_typed(
    dials::af::ref<Shoebox<ProfileFloatType>> self,
    std::size_t vol_addr,
    std::size_t static_mask_addr,
    std::size_t rejected_addr,
    int dtype_code,
    int invalid_top,
    int tr_min,
    int tr_max,
    int chunk_frame0,
    int chunk_frame1,
    std::size_t H,
    std::size_t W,
    std::size_t valid_flags_addr)
{
    const uint8_t* vol_bytes = reinterpret_cast<const uint8_t*>(vol_addr);
    const uint8_t* static_mask = reinterpret_cast<const uint8_t*>(static_mask_addr);
    const uint8_t* rejected = reinterpret_cast<const uint8_t*>(rejected_addr);
    uint8_t* valid_flags = reinterpret_cast<uint8_t*>(valid_flags_addr);
    std::int64_t n = static_cast<std::int64_t>(self.size());
    int yi = static_cast<int>(H);
    int xi = static_cast<int>(W);
    std::size_t elem_size = (dtype_code <= 2) ? 2 : 4;
    std::size_t row_bytes = W * elem_size;
    std::size_t stride_f_bytes = H * row_bytes;
    int32_t inv_top_i32 = static_cast<int32_t>(invalid_top);

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
                const uint8_t* vol_f = vol_bytes + static_cast<std::size_t>(vf) * stride_f_bytes;
                for (int y = yb; y < ye; ++y) {
                    const void* vol_row = vol_f + static_cast<std::size_t>(y + y0) * row_bytes;
                    const uint8_t* sm_row = static_mask + static_cast<std::size_t>(y + y0) * W;
                    float* sd_row = sdata + static_cast<std::size_t>(y) * s_stride_y;
                    uint8_t* smk_row = smask + static_cast<std::size_t>(y) * s_stride_y;
                    if (is_rej) {
                        for (int x = xb; x < xe; ++x) {
                            int32_t raw_val = read_raw_pixel_i32(vol_row, x + x0, dtype_code, inv_top_i32);
                            sd_row[x] += static_cast<float>(raw_val);
                            smk_row[x] = 0;
                        }
                    } else {
                        for (int x = xb; x < xe; ++x) {
                            int32_t raw_val = read_raw_pixel_i32(vol_row, x + x0, dtype_code, inv_top_i32);
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
                const uint8_t* vol_f = vol_bytes + static_cast<std::size_t>(vf) * stride_f_bytes;
                float* sd_z = sdata + static_cast<std::size_t>(z) * s_stride_z;
                uint8_t* smk_z = smask + static_cast<std::size_t>(z) * s_stride_z;
                for (int y = yb; y < ye; ++y) {
                    const void* vol_row = vol_f + static_cast<std::size_t>(y + y0) * row_bytes;
                    const uint8_t* sm_row = static_mask + static_cast<std::size_t>(y + y0) * W;
                    float* sd_row = sd_z + static_cast<std::size_t>(y) * s_stride_y;
                    uint8_t* smk_row = smk_z + static_cast<std::size_t>(y) * s_stride_y;
                    if (is_rej) {
                        for (int x = xb; x < xe; ++x) {
                            int32_t raw_val = read_raw_pixel_i32(vol_row, x + x0, dtype_code, inv_top_i32);
                            sd_row[x] = static_cast<float>(raw_val);
                            smk_row[x] = 0;
                        }
                    } else {
                        for (int x = xb; x < xe; ++x) {
                            int32_t raw_val = read_raw_pixel_i32(vol_row, x + x0, dtype_code, inv_top_i32);
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
    extract_wide_shoeboxes_slice_from_volume_typed(
        self, vol_addr, static_mask_addr, rejected_addr,
        4, 0, tr_min, tr_max, chunk_frame0, chunk_frame1, H, W, valid_flags_addr
    );
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

    struct PanelFastInfo {
        vec3<double> origin;
        vec3<double> fast;
        vec3<double> slow;
        double px_x;
        double px_y;
        bool is_pure_simple;
        bool is_pure_parallax;
    };
    std::size_t n_panels = detector.size();
    std::vector<PanelFastInfo> panel_infos(n_panels);
    for (std::size_t p = 0; p < n_panels; ++p) {
        const auto& pan = detector[p];
        panel_infos[p].origin = pan.get_origin();
        panel_infos[p].fast = pan.get_fast_axis();
        panel_infos[p].slow = pan.get_slow_axis();
        panel_infos[p].px_x = pan.get_pixel_size()[0];
        panel_infos[p].px_y = pan.get_pixel_size()[1];
        auto strat = pan.get_px_mm_strategy();
        bool is_parallax = (dynamic_cast<const dxtbx::model::ParallaxCorrectedPxMmStrategy*>(strat.get()) != nullptr);
        bool is_offset = (dynamic_cast<const dxtbx::model::OffsetPxMmStrategy*>(strat.get()) != nullptr);
        bool is_simple = (dynamic_cast<const dxtbx::model::SimplePxMmStrategy*>(strat.get()) != nullptr);
        panel_infos[p].is_pure_simple = (is_simple && !is_parallax && !is_offset);
        panel_infos[p].is_pure_parallax = (is_parallax && !is_offset);
    }

    std::int64_t n = static_cast<std::int64_t>(shoeboxes.size());
    std::vector<std::vector<double>> thread_dxy_bufs(MAX_WORKER_THREADS);
    std::vector<std::vector<uint8_t>> thread_mv_bufs(MAX_WORKER_THREADS);
    parallel_for(n, [&](std::int64_t idx, int tid) {
        auto& dxy_buf = thread_dxy_bufs[tid];
        auto& mv_buf = thread_mv_bufs[tid];
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

        std::size_t p_idx = panel_arr[idx];
        const dxtbx::model::Panel& panel = detector[p_idx];
        const PanelFastInfo& pinfo = panel_infos[p_idx];
        vec3<double> s1_r = s1[idx];
        CoordinateSystem cs(m2, s0, s1_r, phi);
        double s1_len = cs.s1().length();
        vec3<double> scaled_e1 = cs.e1_axis() * (1.0 / s1_len);
        vec3<double> scaled_e2 = cs.e2_axis() * (1.0 / s1_len);

        std::size_t stride_x = static_cast<std::size_t>(xsize + 1);
        std::size_t dxy_sz = static_cast<std::size_t>(ysize + 1) * stride_x;
        if (dxy_buf.size() < dxy_sz) {
            dxy_buf.resize(dxy_sz);
        }

        if (shoebox.flat) {
            if (pinfo.is_pure_simple) {
                for (int j = 0; j <= ysize; ++j) {
                    double my = static_cast<double>(y0 + j) * pinfo.px_y;
                    vec3<double> row_base = pinfo.origin + pinfo.slow * my;
                    for (int i = 0; i <= xsize; ++i) {
                        double mx = static_cast<double>(x0 + i) * pinfo.px_x;
                        vec3<double> lab = row_base + pinfo.fast * mx;
                        vec3<double> ds1 = lab.normalize() * s0_length - s1_r;
                        double gx = scaled_e1 * ds1;
                        double gy = scaled_e2 * ds1;
                        dxy_buf[static_cast<std::size_t>(j) * stride_x + i] =
                            (gx * gx + gy * gy) * delta_b_r2;
                    }
                }
            } else {
                for (int j = 0; j <= ysize; ++j) {
                    for (int i = 0; i <= xsize; ++i) {
                        vec3<double> ds1 =
                            panel.get_pixel_lab_coord(vec2<double>(x0 + i, y0 + j)).normalize() * s0_length - s1_r;
                        double gx = scaled_e1 * ds1;
                        double gy = scaled_e2 * ds1;
                        dxy_buf[static_cast<std::size_t>(j) * stride_x + i] =
                            (gx * gx + gy * gy) * delta_b_r2;
                    }
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
            if (pinfo.is_pure_simple) {
                for (int j = 0; j <= ysize; ++j) {
                    double my = static_cast<double>(y0 + j) * pinfo.px_y;
                    vec3<double> row_base = pinfo.origin + pinfo.slow * my;
                    for (int i = 0; i <= xsize; ++i) {
                        double mx = static_cast<double>(x0 + i) * pinfo.px_x;
                        vec3<double> lab = row_base + pinfo.fast * mx;
                        vec3<double> ds1 = lab.normalize() * s0_length - s1_r;
                        double gx = scaled_e1 * ds1;
                        double gy = scaled_e2 * ds1;
                        dxy_buf[static_cast<std::size_t>(j) * stride_x + i] =
                            (gx * gx + gy * gy) * delta_b_r2;
                    }
                }
            } else if (pinfo.is_pure_parallax) {
                vec2<double> shoebox_centroid_px = panel.get_ray_intersection_px(s1_r);
                double attenuation_length = panel.attenuation_length(shoebox_centroid_px);
                for (int j = 0; j <= ysize; ++j) {
                    double my = static_cast<double>(y0 + j) * pinfo.px_y;
                    vec3<double> row_base = pinfo.origin + my * pinfo.slow;
                    for (int i = 0; i <= xsize; ++i) {
                        double mx = static_cast<double>(x0 + i) * pinfo.px_x;
                        vec3<double> s1_raw = row_base + mx * pinfo.fast;
                        vec3<double> s1_n = s1_raw.normalize();
                        double cx = mx - (s1_n * pinfo.fast) * attenuation_length;
                        double cy = my - (s1_n * pinfo.slow) * attenuation_length;
                        vec3<double> lab = pinfo.origin + cx * pinfo.fast + cy * pinfo.slow;
                        vec3<double> ds1 = lab.normalize() * s0_length - s1_r;
                        double gx = scaled_e1 * ds1;
                        double gy = scaled_e2 * ds1;
                        dxy_buf[static_cast<std::size_t>(j) * stride_x + i] =
                            (gx * gx + gy * gy) * delta_b_r2;
                    }
                }
            } else {
                vec2<double> shoebox_centroid_px = panel.get_ray_intersection_px(s1_r);
                double attenuation_length = panel.attenuation_length(shoebox_centroid_px);
                for (int j = 0; j <= ysize; ++j) {
                    for (int i = 0; i <= xsize; ++i) {
                        vec3<double> ds1 =
                            panel.get_pixel_lab_coord(vec2<double>(x0 + i, y0 + j), attenuation_length).normalize() * s0_length - s1_r;
                        double gx = scaled_e1 * ds1;
                        double gy = scaled_e2 * ds1;
                        dxy_buf[static_cast<std::size_t>(j) * stride_x + i] =
                            (gx * gx + gy * gy) * delta_b_r2;
                    }
                }
            }
            int k_start = std::max(0, index0 - z0);
            int k_end = std::min(zsize, index1 - z0);
            if (k_start < k_end) {
                uint8_t* m_ptr = mask.begin();
                std::size_t slice_stride = static_cast<std::size_t>(ysize) * static_cast<std::size_t>(xsize);
                if (mv_buf.size() < slice_stride) {
                    mv_buf.resize(slice_stride);
                }
                for (int j = 0; j < ysize; ++j) {
                    const double* r0 = &dxy_buf[static_cast<std::size_t>(j) * stride_x];
                    const double* r1 = r0 + stride_x;
                    uint8_t* mv_row = mv_buf.data() + static_cast<std::size_t>(j) * xsize;
                    for (int i = 0; i < xsize; ++i) {
                        double dxy = std::min(std::min(r0[i], r1[i]), std::min(r0[i + 1], r1[i + 1]));
                        mv_row[i] = (dxy <= 1.0) ? Foreground : Background;
                    }
                }
                uint8_t* m_first = m_ptr + static_cast<std::size_t>(k_start) * slice_stride;
                int n_k = k_end - k_start;
                const uint8_t* mv_data = mv_buf.data();
                for (int k = 0; k < n_k; ++k) {
                    uint8_t* m_slice = m_first + static_cast<std::size_t>(k) * slice_stride;
                    for (std::size_t p = 0; p < slice_stride; ++p) {
                        m_slice[p] |= mv_data[p];
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

void parallel_deallocate_shoeboxes(
    dials::af::ref<Shoebox<ProfileFloatType>> shoeboxes)
{
    std::int64_t n = static_cast<std::int64_t>(shoeboxes.size());
    parallel_for(n, [&](std::int64_t r) {
        shoeboxes[r].deallocate();
    });
}

void apply_pedestal_gain_shoeboxes(
    dials::af::ref<Shoebox<ProfileFloatType>> shoeboxes,
    double ped,
    double gain)
{
    bool has_ped = (std::abs(ped) > 1e-7);
    bool has_gain = (gain > 0.0 && std::abs(gain - 1.0) > 1e-7);
    if (!has_ped && !has_gain) return;
    std::int64_t n = static_cast<std::int64_t>(shoeboxes.size());
    parallel_for(n, [&](std::int64_t r) {
        auto& sb = shoeboxes[r];
        float* d_ptr = sb.data.begin();
        std::size_t sz = sb.data.size();
        for (std::size_t i = 0; i < sz; ++i) {
            double dv = static_cast<double>(d_ptr[i]);
            if (has_ped) dv -= ped;
            if (has_gain) dv /= gain;
            d_ptr[i] = static_cast<float>(dv);
        }
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

// 7. Parallel GaussianRS profile fitting and modelling (zero per-reflection heap allocations, zero inner-loop erf)
struct PanelTransformInfo {
    scitbx::vec3<double> origin;
    scitbx::vec3<double> fast;
    scitbx::vec3<double> slow;
    double px_x;
    double px_y;
    int img_w;
    int img_h;
    bool is_pure_simple;
    bool is_pure_parallax;
};

inline std::vector<PanelTransformInfo> extract_panel_transform_infos(const dxtbx::model::Detector& detector) {
    std::size_t n_panels = detector.size();
    std::vector<PanelTransformInfo> infos(n_panels);
    for (std::size_t p = 0; p < n_panels; ++p) {
        const auto& pan = detector[p];
        infos[p].origin = pan.get_origin();
        infos[p].fast = pan.get_fast_axis();
        infos[p].slow = pan.get_slow_axis();
        infos[p].px_x = pan.get_pixel_size()[0];
        infos[p].px_y = pan.get_pixel_size()[1];
        infos[p].img_w = static_cast<int>(pan.get_image_size()[0]);
        infos[p].img_h = static_cast<int>(pan.get_image_size()[1]);
        auto strat = pan.get_px_mm_strategy();
        bool is_parallax = (dynamic_cast<const dxtbx::model::ParallaxCorrectedPxMmStrategy*>(strat.get()) != nullptr);
        bool is_offset = (dynamic_cast<const dxtbx::model::OffsetPxMmStrategy*>(strat.get()) != nullptr);
        bool is_simple = (dynamic_cast<const dxtbx::model::SimplePxMmStrategy*>(strat.get()) != nullptr);
        infos[p].is_pure_simple = (is_simple && !is_parallax && !is_offset);
        infos[p].is_pure_parallax = (is_parallax && !is_offset);
    }
    return infos;
}

struct ThreadTransformScratch {
    std::vector<double> profile;
    std::vector<double> background;
    std::vector<double> net_profile;
    std::vector<double> av3;
    std::vector<double> bv3;
    std::vector<double> erf_av3;
    std::vector<double> erf_bv3;
    std::vector<double> aj;
    std::vector<double> bj;
    std::vector<double> erf_aj;
    std::vector<double> erf_bj;
    std::vector<double> zfraction;
    std::vector<scitbx::vec2<double>> gc_sub;
    std::vector<double> col_prof;
    std::vector<double> col_bg;
    std::vector<double> col_net;
    int valid_kk_list[64];
    int n_valid_kk = 0;
    uint32_t valid_kk_mask = 0;
    bool has_fg_pixels = false;

    std::vector<double> c_act;
    std::vector<double> b_act;
    std::vector<double> p_act;
    std::vector<double> cb_p;
    std::vector<double> p2;
};

inline double quad_unit_square_intersection_area(
    const dials::algorithms::polygon::clip::vert4& a,
    int ii,
    int jj,
    double q_xmin,
    double q_xmax,
    double q_ymin,
    double q_ymax,
    double target_area)
{
    using scitbx::vec2;
    using dials::algorithms::polygon::intersection;
    double x0 = static_cast<double>(ii);
    double x1 = static_cast<double>(ii + 1);
    double y0 = static_cast<double>(jj);
    double y1 = static_cast<double>(jj + 1);

    if (q_xmin > x0 && q_xmax < x1 && q_ymin > y0 && q_ymax < y1) {
        return target_area;
    }

    vec2<double> buf0[8];
    vec2<double> buf1[8];
    const vec2<double>* in_pts = &a[0];
    vec2<double>* out_pts = buf0;
    int in_sz = 4;

    if (!(q_xmin > x0)) {
        if (q_xmax <= x0) return 0.0;
        vec2<double> e1(x0, y1), e2(x0, y0);
        int out_sz = 0;
        vec2<double> p1 = in_pts[in_sz - 1];
        bool in1 = (p1[0] > x0);
        for (int k = 0; k < in_sz; ++k) {
            vec2<double> p2 = in_pts[k];
            bool in2 = (p2[0] > x0);
            if (in2) {
                if (!in1) out_pts[out_sz++] = intersection(p1, p2, e1, e2);
                out_pts[out_sz++] = p2;
            } else if (in1) {
                out_pts[out_sz++] = intersection(p1, p2, e1, e2);
            }
            p1 = p2;
            in1 = in2;
        }
        if (out_sz < 3) return 0.0;
        in_pts = out_pts;
        out_pts = (out_pts == buf0) ? buf1 : buf0;
        in_sz = out_sz;
    }

    if (!(q_ymin > y0)) {
        if (q_ymax <= y0) return 0.0;
        vec2<double> e1(x0, y0), e2(x1, y0);
        int out_sz = 0;
        vec2<double> p1 = in_pts[in_sz - 1];
        bool in1 = (p1[1] > y0);
        for (int k = 0; k < in_sz; ++k) {
            vec2<double> p2 = in_pts[k];
            bool in2 = (p2[1] > y0);
            if (in2) {
                if (!in1) out_pts[out_sz++] = intersection(p1, p2, e1, e2);
                out_pts[out_sz++] = p2;
            } else if (in1) {
                out_pts[out_sz++] = intersection(p1, p2, e1, e2);
            }
            p1 = p2;
            in1 = in2;
        }
        if (out_sz < 3) return 0.0;
        in_pts = out_pts;
        out_pts = (out_pts == buf0) ? buf1 : buf0;
        in_sz = out_sz;
    }

    if (!(q_xmax < x1)) {
        if (q_xmin >= x1) return 0.0;
        vec2<double> e1(x1, y0), e2(x1, y1);
        int out_sz = 0;
        vec2<double> p1 = in_pts[in_sz - 1];
        bool in1 = (p1[0] < x1);
        for (int k = 0; k < in_sz; ++k) {
            vec2<double> p2 = in_pts[k];
            bool in2 = (p2[0] < x1);
            if (in2) {
                if (!in1) out_pts[out_sz++] = intersection(p1, p2, e1, e2);
                out_pts[out_sz++] = p2;
            } else if (in1) {
                out_pts[out_sz++] = intersection(p1, p2, e1, e2);
            }
            p1 = p2;
            in1 = in2;
        }
        if (out_sz < 3) return 0.0;
        in_pts = out_pts;
        out_pts = (out_pts == buf0) ? buf1 : buf0;
        in_sz = out_sz;
    }

    if (!(q_ymax < y1)) {
        if (q_ymin >= y1) return 0.0;
        vec2<double> e1(x1, y1), e2(x0, y1);
        int out_sz = 0;
        vec2<double> p1 = in_pts[in_sz - 1];
        bool in1 = (p1[1] < y1);
        for (int k = 0; k < in_sz; ++k) {
            vec2<double> p2 = in_pts[k];
            bool in2 = (p2[1] < y1);
            if (in2) {
                if (!in1) out_pts[out_sz++] = intersection(p1, p2, e1, e2);
                out_pts[out_sz++] = p2;
            } else if (in1) {
                out_pts[out_sz++] = intersection(p1, p2, e1, e2);
            }
            p1 = p2;
            in1 = in2;
        }
        if (out_sz < 3) return 0.0;
        in_pts = out_pts;
        in_sz = out_sz;
    }

    double area = 0.0;
    vec2<double> v0 = in_pts[in_sz - 1];
    for (int k = 0; k < in_sz; ++k) {
        vec2<double> v1 = in_pts[k];
        area += v0[0] * v1[1] - v1[0] * v0[1];
        v0 = v1;
    }
    return area * 0.5;
}

inline void run_transform_forward_raw(
    const dxtbx::model::Panel& panel,
    const PanelTransformInfo& pinfo,
    int scan_index0,
    double phi0,
    double dphi,
    double sigma_b,
    double sigma_m,
    double n_sigma,
    std::size_t half_grid_size,
    const dials::algorithms::profile_model::gaussian_rs::CoordinateSystem& cs,
    const scitbx::af::int6& bbox,
    const float* sdata,
    const float* sbkgrd,
    const uint8_t* smask,
    bool use_bkgrd,
    bool compute_net,
    ThreadTransformScratch& scratch)
{
    using boost::math::erf;
    using dials::algorithms::polygon::clip::vert4;
    using dials::algorithms::polygon::spatial_interpolation::quad_grid_range;
    using dials::algorithms::polygon::spatial_interpolation::reverse_quad_inplace_if_backward;
    using dials::algorithms::polygon::spatial_interpolation::quad_grid_intersection_area;
    using dials::model::Foreground;
    using dials::model::Valid;
    using scitbx::vec2;
    using scitbx::vec3;
    using scitbx::af::int4;

    scratch.n_valid_kk = 0;
    scratch.valid_kk_mask = 0;
    scratch.has_fg_pixels = false;

    int x0 = bbox[0];
    int y0 = bbox[2];
    int j0 = bbox[4];
    int j1 = bbox[5];
    int nz = j1 - j0;
    int ny = bbox[3] - bbox[2];
    int nx = bbox[1] - bbox[0];
    if (nz <= 0 || ny <= 0 || nx <= 0) return;

    int g_dim = static_cast<int>(2 * half_grid_size + 1);
    std::size_t g_stride_k = static_cast<std::size_t>(g_dim) * static_cast<std::size_t>(g_dim);
    std::size_t g_stride_j = static_cast<std::size_t>(g_dim);
    std::size_t g_vol = static_cast<std::size_t>(g_dim) * g_stride_k;

    double phi = cs.phi();
    double zeta = cs.zeta();
    if (zeta == 0.0 || sigma_m <= 0.0) return;

    double delta_mosaicity = sigma_m * n_sigma;
    double step_size_e3 = 2.0 * delta_mosaicity / static_cast<double>(g_dim);
    double sigr2 = std::abs(zeta) / (std::sqrt(2.0) * sigma_m);
    int offset = static_cast<int>(half_grid_size);

    if (scratch.av3.size() < static_cast<std::size_t>(g_dim)) {
        scratch.av3.resize(static_cast<std::size_t>(g_dim));
        scratch.bv3.resize(static_cast<std::size_t>(g_dim));
        scratch.erf_av3.resize(static_cast<std::size_t>(g_dim));
        scratch.erf_bv3.resize(static_cast<std::size_t>(g_dim));
        scratch.col_prof.resize(static_cast<std::size_t>(g_dim));
        scratch.col_bg.resize(static_cast<std::size_t>(g_dim));
        scratch.col_net.resize(static_cast<std::size_t>(g_dim));
    }
    if (scratch.aj.size() < static_cast<std::size_t>(nz)) {
        scratch.aj.resize(static_cast<std::size_t>(nz));
        scratch.bj.resize(static_cast<std::size_t>(nz));
        scratch.erf_aj.resize(static_cast<std::size_t>(nz));
        scratch.erf_bj.resize(static_cast<std::size_t>(nz));
    }

    for (int kk = 0; kk < g_dim; ++kk) {
        double b_v = ((kk - offset - 0.5) * step_size_e3) / zeta + phi;
        double a_v = ((kk - offset + 0.5) * step_size_e3) / zeta + phi;
        if (a_v > b_v) std::swap(a_v, b_v);
        scratch.av3[kk] = a_v;
        scratch.bv3[kk] = b_v;
        scratch.erf_av3[kk] = erf((a_v - phi) * sigr2);
        scratch.erf_bv3[kk] = erf((b_v - phi) * sigr2);
    }

    for (int k = 0; k < nz; ++k) {
        double a_j = phi0 + ((j0 + k) - scan_index0) * dphi;
        double b_j = a_j + dphi;
        scratch.aj[k] = a_j;
        scratch.bj[k] = b_j;
        scratch.erf_aj[k] = erf((a_j - phi) * sigr2);
        scratch.erf_bj[k] = erf((b_j - phi) * sigr2);
    }

    if (use_bkgrd) {
        for (int kk = 0; kk < g_dim; ++kk) {
            double int_v3 = scratch.erf_bv3[kk] - scratch.erf_av3[kk];
            if (int_v3 == 0.0) continue;
            double int_v3_r = 1.0 / int_v3;
            double a_v = scratch.av3[kk];
            double b_v = scratch.bv3[kk];
            double e_av = scratch.erf_av3[kk];
            double e_bv = scratch.erf_bv3[kk];
            double sumk = 0.0;
            for (int k = 0; k < nz; ++k) {
                double a_j = scratch.aj[k];
                double b_j = scratch.bj[k];
                double av3j = std::max(a_v, a_j);
                double bv3j = std::min(b_v, b_j);
                if (av3j < bv3j) {
                    double e_hi = (b_v < b_j) ? e_bv : scratch.erf_bj[k];
                    double e_lo = (a_v > a_j) ? e_av : scratch.erf_aj[k];
                    sumk += (e_hi - e_lo) * int_v3_r;
                }
            }
            if (sumk > 0.99) {
                scratch.valid_kk_list[scratch.n_valid_kk++] = kk;
                scratch.valid_kk_mask |= (1u << kk);
            }
        }
        if (scratch.n_valid_kk == 0 && !compute_net) {
            return;
        }
    }

    constexpr uint8_t fg_valid = Valid | Foreground;
    std::size_t stride_z = static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);
    std::size_t stride_y = static_cast<std::size_t>(nx);

    int img_w = pinfo.img_w;
    int img_h = pinfo.img_h;

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
    scratch.has_fg_pixels = true;

    if (scratch.profile.size() < g_vol) {
        scratch.profile.resize(g_vol);
    }
    if (use_bkgrd && scratch.background.size() < g_vol) {
        scratch.background.resize(g_vol);
    }
    if (compute_net && scratch.net_profile.size() < g_vol) {
        scratch.net_profile.resize(g_vol);
    }

    double* prof_ptr = scratch.profile.data();
    double* bg_ptr = use_bkgrd ? scratch.background.data() : nullptr;
    double* net_ptr = compute_net ? scratch.net_profile.data() : nullptr;

    if (use_bkgrd && !compute_net) {
        for (int idx_kk = 0; idx_kk < scratch.n_valid_kk; ++idx_kk) {
            std::size_t kk_off = static_cast<std::size_t>(scratch.valid_kk_list[idx_kk]) * g_stride_k;
            std::memset(prof_ptr + kk_off, 0, g_stride_k * sizeof(double));
            std::memset(bg_ptr + kk_off, 0, g_stride_k * sizeof(double));
        }
    } else {
        std::memset(prof_ptr, 0, g_vol * sizeof(double));
        if (use_bkgrd) {
            std::memset(bg_ptr, 0, g_vol * sizeof(double));
        }
        if (compute_net) {
            std::memset(net_ptr, 0, g_vol * sizeof(double));
        }
    }

    std::size_t zf_sz = static_cast<std::size_t>(nz) * static_cast<std::size_t>(g_dim);
    if (scratch.zfraction.size() < zf_sz) {
        scratch.zfraction.resize(zf_sz);
    }
    double* zf_ptr = scratch.zfraction.data();
    for (int k = 0; k < nz; ++k) {
        double int_j = scratch.erf_bj[k] - scratch.erf_aj[k];
        double* zf_row = zf_ptr + static_cast<std::size_t>(k) * static_cast<std::size_t>(g_dim);
        if (int_j == 0.0) {
            for (int kk = 0; kk < g_dim; ++kk) {
                zf_row[kk] = 0.0;
            }
        } else {
            double int_j_r = 1.0 / int_j;
            double a_j = scratch.aj[k];
            double b_j = scratch.bj[k];
            double e_aj = scratch.erf_aj[k];
            double e_bj = scratch.erf_bj[k];
            for (int kk = 0; kk < g_dim; ++kk) {
                double a_v = scratch.av3[kk];
                double b_v = scratch.bv3[kk];
                double av3j = std::max(a_v, a_j);
                double bv3j = std::min(b_v, b_j);
                if (av3j >= bv3j) {
                    zf_row[kk] = 0.0;
                } else {
                    double e_hi = (b_v < b_j) ? scratch.erf_bv3[kk] : e_bj;
                    double e_lo = (a_v > a_j) ? scratch.erf_av3[kk] : e_aj;
                    zf_row[kk] = (e_hi - e_lo) * int_j_r;
                }
            }
        }
    }

    double g_cent = static_cast<double>(half_grid_size) + 0.5;
    double step_b = sigma_b * n_sigma / g_cent;
    double inv_step_b = 1.0 / step_b;

    vec3<double> s1 = cs.s1();
    double s1_len = s1.length();
    vec3<double> e1_scaled = (cs.e1_axis() / s1_len) * inv_step_b;
    vec3<double> e2_scaled = (cs.e2_axis() / s1_len) * inv_step_b;

    int sub_ny = (j_max - j_min) + 2;
    int sub_nx = (i_max - i_min) + 2;
    std::size_t gc_sz = static_cast<std::size_t>(sub_ny * sub_nx);
    if (scratch.gc_sub.size() < gc_sz) {
        scratch.gc_sub.resize(gc_sz);
    }
    vec2<double>* gc_sub = scratch.gc_sub.data();

    if (pinfo.is_pure_simple) {
        for (int jj = 0; jj < sub_ny; ++jj) {
            int j = j_min + jj;
            double my = static_cast<double>(y0 + j) * pinfo.px_y;
            vec3<double> row_base = pinfo.origin + my * pinfo.slow;
            for (int ii = 0; ii < sub_nx; ++ii) {
                int i = i_min + ii;
                double mx = static_cast<double>(x0 + i) * pinfo.px_x;
                vec3<double> sp = row_base + mx * pinfo.fast;
                vec3<double> ds = sp.normalize() * s1_len - s1;
                gc_sub[static_cast<std::size_t>(jj * sub_nx + ii)] = vec2<double>(
                    g_cent + (e1_scaled * ds),
                    g_cent + (e2_scaled * ds)
                );
            }
        }
    } else if (pinfo.is_pure_parallax) {
        vec2<double> shoebox_centroid_px = panel.get_ray_intersection_px(s1);
        double attenuation_length = panel.attenuation_length(shoebox_centroid_px);
        for (int jj = 0; jj < sub_ny; ++jj) {
            int j = j_min + jj;
            double my = static_cast<double>(y0 + j) * pinfo.px_y;
            vec3<double> row_base = pinfo.origin + my * pinfo.slow;
            for (int ii = 0; ii < sub_nx; ++ii) {
                int i = i_min + ii;
                double mx = static_cast<double>(x0 + i) * pinfo.px_x;
                vec3<double> s1_raw = row_base + mx * pinfo.fast;
                vec3<double> s1_n = s1_raw.normalize();
                double cx = mx - (s1_n * pinfo.fast) * attenuation_length;
                double cy = my - (s1_n * pinfo.slow) * attenuation_length;
                vec3<double> sp = pinfo.origin + cx * pinfo.fast + cy * pinfo.slow;
                vec3<double> ds = sp.normalize() * s1_len - s1;
                gc_sub[static_cast<std::size_t>(jj * sub_nx + ii)] = vec2<double>(
                    g_cent + (e1_scaled * ds),
                    g_cent + (e2_scaled * ds)
                );
            }
        }
    } else {
        vec2<double> shoebox_centroid_px = panel.get_ray_intersection_px(s1);
        double attenuation_length = panel.attenuation_length(shoebox_centroid_px);
        for (int jj = 0; jj < sub_ny; ++jj) {
            int j = j_min + jj;
            for (int ii = 0; ii < sub_nx; ++ii) {
                int i = i_min + ii;
                vec3<double> sp = panel.get_pixel_lab_coord(
                    vec2<double>(x0 + i, y0 + j), attenuation_length
                );
                vec3<double> ds = sp.normalize() * s1_len - s1;
                gc_sub[static_cast<std::size_t>(jj * sub_nx + ii)] = vec2<double>(
                    g_cent + (e1_scaled * ds),
                    g_cent + (e2_scaled * ds)
                );
            }
        }
    }

    dials::af::c_grid<2> grid_size2(g_dim, g_dim);
    double* col_prof = scratch.col_prof.data();
    double* col_bg = scratch.col_bg.data();
    double* col_net = scratch.col_net.data();
    const int* vkk = scratch.valid_kk_list;
    int n_vkk = scratch.n_valid_kk;

    if (use_bkgrd && !compute_net) {
        for (int j = j_min; j <= j_max; ++j) {
            if (y0 + j < 0 || y0 + j >= img_h) continue;
            int jj_sub = j - j_min;
            for (int i = i_min; i <= i_max; ++i) {
                if (x0 + i < 0 || x0 + i >= img_w) continue;
                std::size_t col_off = static_cast<std::size_t>(j) * stride_y + static_cast<std::size_t>(i);
                bool any_k = false;
                for (int idx_kk = 0; idx_kk < n_vkk; ++idx_kk) {
                    col_prof[idx_kk] = 0.0;
                    col_bg[idx_kk] = 0.0;
                }
                for (int k = 0; k < nz; ++k) {
                    std::size_t off = static_cast<std::size_t>(k) * stride_z + col_off;
                    if ((smask[off] & fg_valid) == fg_valid) {
                        any_k = true;
                        double d_val = static_cast<double>(sdata[off]);
                        double b_val = static_cast<double>(sbkgrd[off]);
                        const double* zf_row = zf_ptr + static_cast<std::size_t>(k) * static_cast<std::size_t>(g_dim);
                        for (int idx_kk = 0; idx_kk < n_vkk; ++idx_kk) {
                            double zf = zf_row[vkk[idx_kk]];
                            col_prof[idx_kk] += d_val * zf;
                            col_bg[idx_kk] += b_val * zf;
                        }
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
                double q_xmin = std::min(std::min(input[0][0], input[1][0]), std::min(input[2][0], input[3][0]));
                double q_xmax = std::max(std::max(input[0][0], input[1][0]), std::max(input[2][0], input[3][0]));
                double q_ymin = std::min(std::min(input[0][1], input[1][1]), std::min(input[2][1], input[3][1]));
                double q_ymax = std::max(std::max(input[0][1], input[1][1]), std::max(input[2][1], input[3][1]));
                int ox0 = std::max(0, static_cast<int>(std::floor(q_xmin)));
                int oy0 = std::max(0, static_cast<int>(std::floor(q_ymin)));
                int ox1 = std::min(g_dim, static_cast<int>(std::ceil(q_xmax)));
                int oy1 = std::min(g_dim, static_cast<int>(std::ceil(q_ymax)));
                if (ox0 >= ox1 || oy0 >= oy1) continue;
                double target_area = reverse_quad_inplace_if_backward(input);
                double inv_target_area = 1.0 / target_area;

                for (int jj = oy0; jj < oy1; ++jj) {
                    for (int ii = ox0; ii < ox1; ++ii) {
                        double result_area = quad_unit_square_intersection_area(
                            input, ii, jj, q_xmin, q_xmax, q_ymin, q_ymax, target_area
                        );
                        if (result_area > 0.0) {
                            double fraction = result_area * inv_target_area;
                            std::size_t cell_off = static_cast<std::size_t>(jj) * g_stride_j + static_cast<std::size_t>(ii);
                            for (int idx_kk = 0; idx_kk < n_vkk; ++idx_kk) {
                                std::size_t idx3 = static_cast<std::size_t>(vkk[idx_kk]) * g_stride_k + cell_off;
                                prof_ptr[idx3] += col_prof[idx_kk] * fraction;
                                bg_ptr[idx3] += col_bg[idx_kk] * fraction;
                            }
                        }
                    }
                }
            }
        }
    } else if (use_bkgrd && compute_net) {
        for (int j = j_min; j <= j_max; ++j) {
            if (y0 + j < 0 || y0 + j >= img_h) continue;
            int jj_sub = j - j_min;
            for (int i = i_min; i <= i_max; ++i) {
                if (x0 + i < 0 || x0 + i >= img_w) continue;
                std::size_t col_off = static_cast<std::size_t>(j) * stride_y + static_cast<std::size_t>(i);
                bool any_k = false;
                for (int kk = 0; kk < g_dim; ++kk) {
                    col_prof[kk] = 0.0;
                    col_bg[kk] = 0.0;
                    col_net[kk] = 0.0;
                }
                for (int k = 0; k < nz; ++k) {
                    std::size_t off = static_cast<std::size_t>(k) * stride_z + col_off;
                    if ((smask[off] & fg_valid) == fg_valid) {
                        any_k = true;
                        double d_val = static_cast<double>(sdata[off]);
                        double b_val = static_cast<double>(sbkgrd[off]);
                        double n_val = d_val - b_val;
                        const double* zf_row = zf_ptr + static_cast<std::size_t>(k) * static_cast<std::size_t>(g_dim);
                        for (int kk = 0; kk < g_dim; ++kk) {
                            double zf = zf_row[kk];
                            col_prof[kk] += d_val * zf;
                            col_bg[kk] += b_val * zf;
                            col_net[kk] += n_val * zf;
                        }
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
                double q_xmin = std::min(std::min(input[0][0], input[1][0]), std::min(input[2][0], input[3][0]));
                double q_xmax = std::max(std::max(input[0][0], input[1][0]), std::max(input[2][0], input[3][0]));
                double q_ymin = std::min(std::min(input[0][1], input[1][1]), std::min(input[2][1], input[3][1]));
                double q_ymax = std::max(std::max(input[0][1], input[1][1]), std::max(input[2][1], input[3][1]));
                int ox0 = std::max(0, static_cast<int>(std::floor(q_xmin)));
                int oy0 = std::max(0, static_cast<int>(std::floor(q_ymin)));
                int ox1 = std::min(g_dim, static_cast<int>(std::ceil(q_xmax)));
                int oy1 = std::min(g_dim, static_cast<int>(std::ceil(q_ymax)));
                if (ox0 >= ox1 || oy0 >= oy1) continue;
                double target_area = reverse_quad_inplace_if_backward(input);
                double inv_target_area = 1.0 / target_area;

                for (int jj = oy0; jj < oy1; ++jj) {
                    for (int ii = ox0; ii < ox1; ++ii) {
                        double result_area = quad_unit_square_intersection_area(
                            input, ii, jj, q_xmin, q_xmax, q_ymin, q_ymax, target_area
                        );
                        if (result_area > 0.0) {
                            double fraction = result_area * inv_target_area;
                            std::size_t cell_off = static_cast<std::size_t>(jj) * g_stride_j + static_cast<std::size_t>(ii);
                            for (int kk = 0; kk < g_dim; ++kk) {
                                std::size_t idx3 = static_cast<std::size_t>(kk) * g_stride_k + cell_off;
                                prof_ptr[idx3] += col_prof[kk] * fraction;
                                bg_ptr[idx3] += col_bg[kk] * fraction;
                                net_ptr[idx3] += col_net[kk] * fraction;
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
                for (int kk = 0; kk < g_dim; ++kk) {
                    col_prof[kk] = 0.0;
                }
                for (int k = 0; k < nz; ++k) {
                    std::size_t off = static_cast<std::size_t>(k) * stride_z + col_off;
                    if ((smask[off] & fg_valid) == fg_valid) {
                        any_k = true;
                        double val = static_cast<double>(sdata[off]) - static_cast<double>(sbkgrd[off]);
                        const double* zf_row = zf_ptr + static_cast<std::size_t>(k) * static_cast<std::size_t>(g_dim);
                        for (int kk = 0; kk < g_dim; ++kk) {
                            col_prof[kk] += val * zf_row[kk];
                        }
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
                double q_xmin = std::min(std::min(input[0][0], input[1][0]), std::min(input[2][0], input[3][0]));
                double q_xmax = std::max(std::max(input[0][0], input[1][0]), std::max(input[2][0], input[3][0]));
                double q_ymin = std::min(std::min(input[0][1], input[1][1]), std::min(input[2][1], input[3][1]));
                double q_ymax = std::max(std::max(input[0][1], input[1][1]), std::max(input[2][1], input[3][1]));
                int ox0 = std::max(0, static_cast<int>(std::floor(q_xmin)));
                int oy0 = std::max(0, static_cast<int>(std::floor(q_ymin)));
                int ox1 = std::min(g_dim, static_cast<int>(std::ceil(q_xmax)));
                int oy1 = std::min(g_dim, static_cast<int>(std::ceil(q_ymax)));
                if (ox0 >= ox1 || oy0 >= oy1) continue;
                double target_area = reverse_quad_inplace_if_backward(input);
                double inv_target_area = 1.0 / target_area;

                for (int jj = oy0; jj < oy1; ++jj) {
                    for (int ii = ox0; ii < ox1; ++ii) {
                        double result_area = quad_unit_square_intersection_area(
                            input, ii, jj, q_xmin, q_xmax, q_ymin, q_ymax, target_area
                        );
                        if (result_area > 0.0) {
                            double fraction = result_area * inv_target_area;
                            std::size_t cell_off = static_cast<std::size_t>(jj) * g_stride_j + static_cast<std::size_t>(ii);
                            for (int kk = 0; kk < g_dim; ++kk) {
                                prof_ptr[static_cast<std::size_t>(kk) * g_stride_k + cell_off] += col_prof[kk] * fraction;
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void run_transform_forward_fast(
    const dxtbx::model::Panel& panel,
    const PanelTransformInfo& pinfo,
    int scan_index0,
    double phi0,
    double dphi,
    double sigma_b,
    double sigma_m,
    double n_sigma,
    std::size_t half_grid_size,
    const dials::algorithms::profile_model::gaussian_rs::CoordinateSystem& cs,
    const Shoebox<ProfileFloatType>& sbox,
    bool use_bkgrd,
    bool compute_net,
    ThreadTransformScratch& scratch)
{
    run_transform_forward_raw(
        panel, pinfo, scan_index0, phi0, dphi, sigma_b, sigma_m, n_sigma,
        half_grid_size, cs, sbox.bbox,
        sbox.data.begin(), sbox.background.begin(), sbox.mask.begin(),
        use_bkgrd, compute_net, scratch
    );
}

// Exception-free 1D profile fitter over valid kk slices matching DIALS ProfileFitter<double> bit-for-bit
inline bool fit_1d_noexcept_slices(
    const double* c_full_or_compact,
    const double* b_full_or_compact,
    bool is_compact_slices,
    const bool* mask1,
    const int* valid_kk_list,
    int n_valid_kk,
    std::size_t g_stride_k,
    const double* p_raw,
    std::size_t sz,
    double eps,
    std::size_t max_iter,
    ThreadTransformScratch& scratch,
    double& out_intensity,
    double& out_variance,
    double& out_correlation)
{
    if (n_valid_kk <= 0) return false;
    if (scratch.c_act.size() < sz) {
        scratch.c_act.resize(sz);
        scratch.b_act.resize(sz);
        scratch.p_act.resize(sz);
        scratch.cb_p.resize(sz);
        scratch.p2.resize(sz);
    }
    double* c_act = scratch.c_act.data();
    double* b_act = scratch.b_act.data();
    double* p_act = scratch.p_act.data();
    double* cb_p = scratch.cb_p.data();
    double* p2 = scratch.p2.data();

    std::size_t m = 0;
    double sumd = 0.0;
    double sumb = 0.0;
    double sump = 0.0;
    for (int idx_kk = 0; idx_kk < n_valid_kk; ++idx_kk) {
        std::size_t full_off = static_cast<std::size_t>(valid_kk_list[idx_kk]) * g_stride_k;
        std::size_t cb_off = is_compact_slices
            ? (static_cast<std::size_t>(idx_kk) * g_stride_k)
            : full_off;
        const double* c_sl = c_full_or_compact + cb_off;
        const double* b_sl = b_full_or_compact + cb_off;
        const double* p_sl = p_raw + full_off;
        const bool* m1_sl = mask1 + full_off;
        for (std::size_t j = 0; j < g_stride_k; ++j) {
            if (m1_sl[j]) {
                double pj = p_sl[j];
                if (pj < 0.0) return false;
                double dj = c_sl[j];
                double bj = b_sl[j];
                c_act[m] = dj;
                b_act[m] = bj;
                p_act[m] = pj;
                cb_p[m] = (dj - bj) * pj;
                p2[m] = pj * pj;
                sumd += dj;
                sumb += bj;
                sump += pj;
                m++;
            }
        }
    }
    if (sumb < 0.0 || sumd < 0.0 || sump <= 0.0 || m == 0) return false;

    double I0 = sumd - sumb;
    double I = 0.0;
    double V = 0.0;
    std::size_t n_iter = 0;
    for (; n_iter < max_iter; ++n_iter) {
        double sum1 = 0.0;
        double sum2 = 0.0;
        for (std::size_t k = 0; k < m; ++k) {
            double pk = p_act[k];
            if (pk > 0.0) {
                double v = 1e-10 + std::abs(b_act[k]) + std::abs(I0 * pk);
                if (v <= 0.0) return false;
                sum1 += cb_p[k] / v;
                sum2 += p2[k] / v;
            }
        }
        if (sum2 <= 0.0) return false;
        I = sum1 / sum2;
        V = std::abs(I) + std::abs(sumb);
        if (std::abs(I - I0) < eps) break;
        I0 = I;
    }
    if (n_iter >= max_iter) {
        I = sumd - sumb;
        V = std::abs(I) + std::abs(sumb);
    }
    if (!(V >= 0.0 && V >= I)) return false;

    // Pearson correlation matching ProfileFitter::compute_correlation
    double xb = 0.0, yb = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
        xb += I * p_act[k] + b_act[k];
        yb += c_act[k];
    }
    double inv_m = 1.0 / static_cast<double>(m);
    xb *= inv_m;
    yb *= inv_m;
    double sdxdy = 0.0, sdx2 = 0.0, sdy2 = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
        double dx = (I * p_act[k] + b_act[k]) - xb;
        double dy = c_act[k] - yb;
        sdxdy += dx * dy;
        sdx2 += dx * dx;
        sdy2 += dy * dy;
    }
    double corr = 0.0;
    if (sdx2 > 0.0 && sdy2 > 0.0) {
        corr = sdxdy / (std::sqrt(sdx2) * std::sqrt(sdy2));
    }
    out_intensity = I;
    out_variance = V;
    out_correlation = corr;
    return true;
}

struct CachedTransformStore {
    std::size_t n_total = 0;
    std::size_t g_dim = 0;
    std::size_t g_stride_k = 0;
    std::vector<uint8_t> was_processed;
    std::vector<uint32_t> valid_kk_mask;
    std::vector<uint8_t> tid_of_r;
    std::vector<uint32_t> slice_offset_of_r;
    std::vector<std::vector<double>> thread_c_slices;
    std::vector<std::vector<double>> thread_b_slices;

    CachedTransformStore(std::size_t n, std::size_t gd)
        : n_total(n),
          g_dim(gd),
          g_stride_k(gd * gd),
          was_processed(n, 0),
          valid_kk_mask(n, 0),
          tid_of_r(n, 0),
          slice_offset_of_r(n, 0),
          thread_c_slices(MAX_WORKER_THREADS),
          thread_b_slices(MAX_WORKER_THREADS)
    {
        int nt = num_worker_threads();
        std::size_t est_slices_per_thread = (n / static_cast<std::size_t>(nt) + 64) * 2;
        for (int t = 0; t < nt; ++t) {
            thread_c_slices[t].reserve(est_slices_per_thread * g_stride_k);
            thread_b_slices[t].reserve(est_slices_per_thread * g_stride_k);
        }
    }
};

static std::mutex g_transform_cache_mtx;
static std::unordered_map<std::size_t, std::shared_ptr<CachedTransformStore>> g_transform_caches;
static std::size_t g_next_cache_id = 1;

void free_transform_cache(std::size_t cache_id) {
    std::lock_guard<std::mutex> lock(g_transform_cache_mtx);
    g_transform_caches.erase(cache_id);
}

struct ParallelGaussianRS : public dials::algorithms::GaussianRSProfileModeller {
    std::size_t get_half_grid_size() const {
        return grid_size_;
    }
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
        std::size_t g_dim = 2 * hgs + 1;
        std::size_t g_stride_k = g_dim * g_dim;
        std::size_t g_vol = g_dim * g_stride_k;

        std::vector<PanelTransformInfo> panel_infos = extract_panel_transform_infos(detector_);

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
            std::vector<ThreadTransformScratch> thread_scratch(MAX_WORKER_THREADS);
            parallel_for(n, [&](std::int64_t i, int tid) {
                intensity_val[i] = 0.0;
                intensity_var[i] = -1.0;
                reference_cor[i] = 0.0;
                flags[i] &= ~dials::af::IntegratedPrf;
                bool integrate = !(flags[i] & dials::af::DontIntegrate);
                if (!integrate) return;
                try {
                    std::size_t p_idx = sbox[i].panel;
                    std::size_t index = sampler_->nearest(p_idx, xyzpx[i]);
                    if (index >= n_prof || !prof_valid[index]) return;
                    data_const_reference p = prof_data[index];
                    mask_const_reference mask1 = prof_mask[index];
                    CoordinateSystem cs(m2, s0, s1[i], xyzmm[i][2]);

                    auto& scratch = thread_scratch[tid];
                    run_transform_forward_fast(
                        detector_[p_idx],
                        panel_infos[p_idx],
                        scan_index0, phi0, dphi, sb, sm, ns, hgs,
                        cs, sbox[i], true, false, scratch
                    );
                    if (scratch.n_valid_kk == 0 || !scratch.has_fg_pixels) {
                        return;
                    }

                    double fit_i = 0.0, fit_v = -1.0, fit_c = 0.0;
                    if (fit_1d_noexcept_slices(
                            scratch.profile.data(),
                            scratch.background.data(),
                            false,
                            mask1.begin(),
                            scratch.valid_kk_list,
                            scratch.n_valid_kk,
                            g_stride_k,
                            p.begin(),
                            g_vol,
                            1e-3,
                            100,
                            scratch,
                            fit_i,
                            fit_v,
                            fit_c)) {
                        intensity_val[i] = fit_i;
                        intensity_var[i] = fit_v;
                        reference_cor[i] = fit_c;
                        flags[i] |= dials::af::IntegratedPrf;
                        succ_ptr[i] = true;
                    }
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
        dials::af::ref<std::size_t> flags,
        bool require_ref_spot = false)
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

        std::vector<PanelTransformInfo> panel_infos = extract_panel_transform_infos(detector_);

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
        std::vector<ThreadTransformScratch> thread_scratch(static_cast<std::size_t>(nt));

        std::atomic<std::int64_t> next_idx{0};
        constexpr std::int64_t chunk = 16;

        auto worker = [&](int tid) {
            auto& loc_data = thread_data[static_cast<std::size_t>(tid)];
            auto& loc_n = thread_n_refl[static_cast<std::size_t>(tid)];
            auto& scratch = thread_scratch[static_cast<std::size_t>(tid)];
            while (true) {
                std::int64_t start = next_idx.fetch_add(chunk, std::memory_order_relaxed);
                if (start >= n) break;
                std::int64_t end = std::min(start + chunk, n);
                for (std::int64_t i = start; i < end; ++i) {
                    if (require_ref_spot && !(flags[i] & dials::af::ReferenceSpot)) continue;
                    bool full = partiality[i] > 0.99;
                    bool integrated = (flags[i] & dials::af::IntegratedSum) != 0;
                    if (!full || !integrated) continue;
                    std::size_t p_idx = sbox[i].panel;
                    const PanelTransformInfo& pinfo = panel_infos[p_idx];
                    bool bbox_valid = (
                        sbox[i].bbox[0] >= 0 && sbox[i].bbox[2] >= 0 &&
                        sbox[i].bbox[1] <= pinfo.img_w &&
                        sbox[i].bbox[3] <= pinfo.img_h
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
                        run_transform_forward_fast(
                            detector_[p_idx],
                            pinfo,
                            scan_index0, phi0, dphi, sb, sm, ns, hgs,
                            cs, sbox[i], false, false, scratch
                        );
                        if (!scratch.has_fg_pixels) {
                            flags[i] |= dials::af::UsedInModelling;
                            continue;
                        }

                        const double* prof_ptr = scratch.profile.data();
                        double sum_data = 0.0;
                        for (std::size_t p_i = 0; p_i < prof_elems; ++p_i) {
                            sum_data += prof_ptr[p_i];
                        }
                        if (sum_data > 0.0) {
                            double inv_sum = 1.0 / sum_data;
                            dials::af::shared<std::size_t> indices = sampler_->nearest_n(p_idx, xyzpx[i]);
                            for (std::size_t j = 0; j < indices.size(); ++j) {
                                std::size_t idx = indices[j];
                                double w = sampler_->weight(idx, p_idx, xyzpx[i]) * inv_sum;
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

    void transform_and_model_slab_cols(
        CachedTransformStore& store,
        const dials::af::const_ref<std::size_t>& slab_ind,
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

        if (fit_method_ != ReciprocalSpace) {
            throw DIALS_ERROR("DetectorSpace not supported in single-pass slab cache");
        }

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
        std::size_t g_dim = 2 * hgs + 1;
        std::size_t g_stride_k = g_dim * g_dim;

        std::int64_t n = static_cast<std::int64_t>(sbox.size());
        if (n <= 0) return;

        std::vector<PanelTransformInfo> panel_infos = extract_panel_transform_infos(detector_);

        std::size_t n_prof = data_.size();
        std::size_t prof_elems = accessor_.size_1d();
        int nt = num_worker_threads();
        if (nt > n) nt = static_cast<int>(n);

        std::vector<std::vector<std::vector<double>>> thread_data(
            static_cast<std::size_t>(nt),
            std::vector<std::vector<double>>(n_prof)
        );
        std::vector<std::vector<std::size_t>> thread_n_refl(
            static_cast<std::size_t>(nt),
            std::vector<std::size_t>(n_prof, 0)
        );
        std::vector<ThreadTransformScratch> thread_scratch(static_cast<std::size_t>(nt));

        std::atomic<std::int64_t> next_idx{0};
        constexpr std::int64_t chunk = 16;

        auto worker = [&](int tid) {
            auto& loc_data = thread_data[static_cast<std::size_t>(tid)];
            auto& loc_n = thread_n_refl[static_cast<std::size_t>(tid)];
            auto& scratch = thread_scratch[static_cast<std::size_t>(tid)];
            auto& c_store = store.thread_c_slices[static_cast<std::size_t>(tid)];
            auto& b_store = store.thread_b_slices[static_cast<std::size_t>(tid)];

            while (true) {
                std::int64_t start = next_idx.fetch_add(chunk, std::memory_order_relaxed);
                if (start >= n) break;
                std::int64_t end = std::min(start + chunk, n);
                for (std::int64_t i = start; i < end; ++i) {
                    std::size_t global_r = slab_ind[i];
                    store.was_processed[global_r] = 1;
                    std::size_t fl = flags[i];
                    bool do_fit = (fl & dials::af::DontIntegrate) == 0;
                    bool is_ref = ((fl & dials::af::ReferenceSpot) != 0)
                                  && (partiality[i] > 0.99)
                                  && ((fl & dials::af::IntegratedSum) != 0);
                    std::size_t p_idx = sbox[i].panel;
                    const PanelTransformInfo& pinfo = panel_infos[p_idx];
                    if (is_ref) {
                        bool bbox_valid = (
                            sbox[i].bbox[0] >= 0 && sbox[i].bbox[2] >= 0 &&
                            sbox[i].bbox[1] <= pinfo.img_w &&
                            sbox[i].bbox[3] <= pinfo.img_h
                        );
                        if (!bbox_valid) {
                            is_ref = false;
                        } else {
                            const uint8_t* m_ptr = sbox[i].mask.begin();
                            std::size_t m_sz = sbox[i].mask.size();
                            for (std::size_t k = 0; k < m_sz; ++k) {
                                if ((m_ptr[k] & Foreground) && !(m_ptr[k] & Valid)) {
                                    is_ref = false;
                                    break;
                                }
                            }
                        }
                    }
                    if (!do_fit && !is_ref) continue;

                    try {
                        CoordinateSystem cs(m2, s0, s1[i], xyzmm[i][2]);
                        run_transform_forward_fast(
                            detector_[p_idx],
                            pinfo,
                            scan_index0, phi0, dphi, sb, sm, ns, hgs,
                            cs, sbox[i], true, is_ref, scratch
                        );
                        if (!scratch.has_fg_pixels) {
                            continue;
                        }

                        if (is_ref) {
                            const double* net_ptr = scratch.net_profile.data();
                            double sum_data = 0.0;
                            for (std::size_t p_i = 0; p_i < prof_elems; ++p_i) {
                                sum_data += net_ptr[p_i];
                            }
                            if (sum_data > 0.0) {
                                double inv_sum = 1.0 / sum_data;
                                dials::af::shared<std::size_t> indices = sampler_->nearest_n(p_idx, xyzpx[i]);
                                for (std::size_t j = 0; j < indices.size(); ++j) {
                                    std::size_t idx = indices[j];
                                    double w = sampler_->weight(idx, p_idx, xyzpx[i]) * inv_sum;
                                    auto& vec = loc_data[idx];
                                    if (vec.empty()) {
                                        vec.assign(prof_elems, 0.0);
                                    }
                                    for (std::size_t p_i = 0; p_i < prof_elems; ++p_i) {
                                        vec[p_i] += w * net_ptr[p_i];
                                    }
                                    loc_n[idx]++;
                                }
                            }
                        }

                        if (do_fit && scratch.n_valid_kk > 0) {
                            uint32_t sl_off = static_cast<uint32_t>(c_store.size() / g_stride_k);
                            std::size_t add_elems = static_cast<std::size_t>(scratch.n_valid_kk) * g_stride_k;
                            std::size_t old_sz = c_store.size();
                            c_store.resize(old_sz + add_elems);
                            b_store.resize(old_sz + add_elems);
                            double* c_dst = c_store.data() + old_sz;
                            double* b_dst = b_store.data() + old_sz;
                            const double* prof_ptr = scratch.profile.data();
                            const double* bg_ptr = scratch.background.data();
                            for (int idx_kk = 0; idx_kk < scratch.n_valid_kk; ++idx_kk) {
                                std::size_t kk_off = static_cast<std::size_t>(scratch.valid_kk_list[idx_kk]) * g_stride_k;
                                std::memcpy(
                                    c_dst + static_cast<std::size_t>(idx_kk) * g_stride_k,
                                    prof_ptr + kk_off,
                                    g_stride_k * sizeof(double)
                                );
                                std::memcpy(
                                    b_dst + static_cast<std::size_t>(idx_kk) * g_stride_k,
                                    bg_ptr + kk_off,
                                    g_stride_k * sizeof(double)
                                );
                            }
                            store.valid_kk_mask[global_r] = scratch.valid_kk_mask;
                            store.tid_of_r[global_r] = static_cast<uint8_t>(tid);
                            store.slice_offset_of_r[global_r] = sl_off;
                        }
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

    dials::af::shared<bool> fit_from_cache_cols(
        const CachedTransformStore& store,
        const dials::af::const_ref<std::size_t>& panel,
        const dials::af::const_ref<scitbx::vec3<double>>& xyzpx,
        dials::af::ref<std::size_t> flags,
        dials::af::ref<double> intensity_val,
        dials::af::ref<double> intensity_var,
        dials::af::ref<double> reference_cor) const
    {
        std::size_t hgs = grid_size_;
        std::size_t g_dim = 2 * hgs + 1;
        std::size_t g_stride_k = g_dim * g_dim;
        std::size_t g_vol = g_dim * g_stride_k;

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

        std::int64_t n = static_cast<std::int64_t>(panel.size());
        dials::af::shared<bool> success(panel.size(), false);
        bool* succ_ptr = success.begin();

        std::vector<ThreadTransformScratch> thread_scratch(MAX_WORKER_THREADS);
        parallel_for(n, [&](std::int64_t i, int tid) {
            if (!store.was_processed[static_cast<std::size_t>(i)]) {
                return;
            }
            intensity_val[i] = 0.0;
            intensity_var[i] = -1.0;
            reference_cor[i] = 0.0;
            flags[i] &= ~dials::af::IntegratedPrf;
            if (flags[i] & dials::af::DontIntegrate) {
                flags[i] |= dials::af::FailedDuringProfileFitting;
                return;
            }
            uint32_t vmask = store.valid_kk_mask[static_cast<std::size_t>(i)];
            if (vmask == 0) {
                flags[i] |= dials::af::FailedDuringProfileFitting;
                return;
            }

            try {
                std::size_t index = sampler_->nearest(panel[i], xyzpx[i]);
                if (index >= n_prof || !prof_valid[index]) {
                    flags[i] |= dials::af::FailedDuringProfileFitting;
                    return;
                }
                data_const_reference p = prof_data[index];
                mask_const_reference mask1 = prof_mask[index];

                auto& scratch = thread_scratch[tid];
                int n_vkk = 0;
                for (int kk = 0; kk < static_cast<int>(g_dim); ++kk) {
                    if (vmask & (1u << kk)) {
                        scratch.valid_kk_list[n_vkk++] = kk;
                    }
                }
                uint8_t src_tid = store.tid_of_r[static_cast<std::size_t>(i)];
                std::size_t byte_off = static_cast<std::size_t>(store.slice_offset_of_r[static_cast<std::size_t>(i)]) * g_stride_k;
                const double* c_compact = store.thread_c_slices[src_tid].data() + byte_off;
                const double* b_compact = store.thread_b_slices[src_tid].data() + byte_off;

                double fit_i = 0.0, fit_v = -1.0, fit_c = 0.0;
                if (fit_1d_noexcept_slices(
                        c_compact,
                        b_compact,
                        true,
                        mask1.begin(),
                        scratch.valid_kk_list,
                        n_vkk,
                        g_stride_k,
                        p.begin(),
                        g_vol,
                        1e-3,
                        100,
                        scratch,
                        fit_i,
                        fit_v,
                        fit_c)) {
                    intensity_val[i] = fit_i;
                    intensity_var[i] = fit_v;
                    reference_cor[i] = fit_c;
                    flags[i] |= dials::af::IntegratedPrf;
                    succ_ptr[i] = true;
                } else {
                    flags[i] |= dials::af::FailedDuringProfileFitting;
                }
            } catch (...) {
                flags[i] |= dials::af::FailedDuringProfileFitting;
                return;
            }
        });
        return success;
    }

    dials::af::shared<bool> model_and_fit_circular_slab_cols(
        std::size_t prof_begin,
        std::size_t prof_end,
        dials::af::const_ref<Shoebox<ProfileFloatType>> sbox,
        const dials::af::const_ref<double>& partiality,
        const dials::af::const_ref<scitbx::vec3<double>>& s1,
        const dials::af::const_ref<scitbx::vec3<double>>& xyzpx,
        const dials::af::const_ref<scitbx::vec3<double>>& xyzmm,
        dials::af::ref<std::size_t> flags,
        dials::af::ref<double> intensity_val,
        dials::af::ref<double> intensity_var,
        dials::af::ref<double> reference_cor)
    {
        model_parallel_cols(sbox, partiality, s1, xyzpx, xyzmm, flags, true);
        std::size_t end_idx = std::min(prof_end, data_.size());
        for (std::size_t idx = prof_begin; idx < end_idx; ++idx) {
            if (data_[idx].size() != 0) {
                finalize(idx);
            }
        }
        return fit_parallel_cols(sbox, s1, xyzpx, xyzmm, flags, intensity_val, intensity_var, reference_cor);
    }

    void mark_finalized() {
        set_finalized(true);
    }

    static void integrate_slab_from_volume_cols(
        ParallelGaussianRS* self,
        const dials::af::const_ref<std::size_t>& slab_ind,
        boost::python::tuple cols,
        boost::python::tuple params)
    {
        using namespace boost::python;
        using namespace dials::algorithms;
        using dials::model::Valid;
        using dials::model::Background;
        using dials::model::Foreground;
        using dials::model::Overlapped;
        using dials::model::BackgroundUsed;
        using dials::algorithms::profile_model::gaussian_rs::CoordinateSystem;
        using scitbx::vec2;
        using scitbx::vec3;

        dials::af::const_ref<std::size_t> panel_col = extract<dials::af::const_ref<std::size_t>>(cols[0]);
        dials::af::const_ref<scitbx::af::int6> bbox_col = extract<dials::af::const_ref<scitbx::af::int6>>(cols[1]);
        dials::af::const_ref<vec3<double>> s1_col = extract<dials::af::const_ref<vec3<double>>>(cols[2]);
        dials::af::const_ref<vec3<double>> xyzcal_px_col = extract<dials::af::const_ref<vec3<double>>>(cols[3]);
        dials::af::const_ref<vec3<double>> xyzcal_mm_col = extract<dials::af::const_ref<vec3<double>>>(cols[4]);
        dials::af::const_ref<double> partiality_col = extract<dials::af::const_ref<double>>(cols[5]);
        dials::af::ref<std::size_t> flags_col = extract<dials::af::ref<std::size_t>>(cols[6]);
        dials::af::ref<double> bg_mean_col = extract<dials::af::ref<double>>(cols[7]);
        dials::af::ref<double> int_sum_val_col = extract<dials::af::ref<double>>(cols[8]);
        dials::af::ref<double> int_sum_var_col = extract<dials::af::ref<double>>(cols[9]);
        dials::af::ref<double> bg_sum_val_col = extract<dials::af::ref<double>>(cols[10]);
        dials::af::ref<double> bg_sum_var_col = extract<dials::af::ref<double>>(cols[11]);
        dials::af::ref<vec3<double>> xyzobs_px_val_col = extract<dials::af::ref<vec3<double>>>(cols[12]);
        dials::af::ref<vec3<double>> xyzobs_px_var_col = extract<dials::af::ref<vec3<double>>>(cols[13]);
        dials::af::ref<vec3<double>> xyzobs_mm_val_col = extract<dials::af::ref<vec3<double>>>(cols[14]);
        dials::af::ref<vec3<double>> xyzobs_mm_var_col = extract<dials::af::ref<vec3<double>>>(cols[15]);
        dials::af::ref<int> nvalid_col = extract<dials::af::ref<int>>(cols[16]);
        dials::af::ref<int> nbg_col = extract<dials::af::ref<int>>(cols[17]);
        dials::af::ref<int> nbg_used_col = extract<dials::af::ref<int>>(cols[18]);
        dials::af::ref<int> nforeg_col = extract<dials::af::ref<int>>(cols[19]);
        dials::af::ref<double> int_prf_val_col = extract<dials::af::ref<double>>(cols[20]);
        dials::af::ref<double> int_prf_var_col = extract<dials::af::ref<double>>(cols[21]);
        dials::af::ref<double> prf_corr_col = extract<dials::af::ref<double>>(cols[22]);

        const dxtbx::model::BeamBase& beam = extract<const dxtbx::model::BeamBase&>(params[0]);
        const dxtbx::model::Detector& detector = extract<const dxtbx::model::Detector&>(params[1]);
        const dxtbx::model::Goniometer& goniometer = extract<const dxtbx::model::Goniometer&>(params[2]);
        const dxtbx::model::Scan& scan = extract<const dxtbx::model::Scan&>(params[3]);
        std::size_t vol_addr = extract<std::size_t>(params[4]);
        std::size_t static_mask_addr = extract<std::size_t>(params[5]);
        std::size_t rejected_addr = extract<std::size_t>(params[6]);
        int tr_min = extract<int>(params[7]);
        int tr_max = extract<int>(params[8]);
        int frame0 = extract<int>(params[9]);
        int frame1 = extract<int>(params[10]);
        std::size_t img_h = extract<std::size_t>(params[11]);
        std::size_t img_w = extract<std::size_t>(params[12]);
        int vol_dtype_code = extract<int>(params[13]);
        int vol_invalid_top = extract<int>(params[14]);
        double saturation = extract<double>(params[15]);
        double delta_b = extract<double>(params[16]);
        double valid_fg_threshold = extract<double>(params[17]);
        int bg_alg_code = extract<int>(params[18]);
        int bg_min_pixels = extract<int>(params[19]);
        double bg_tuning_constant = extract<double>(params[20]);
        std::size_t bg_max_iter = extract<std::size_t>(params[21]);
        int px2mm_mode = extract<int>(params[22]);
        double mu = extract<double>(params[23]);
        double t0 = extract<double>(params[24]);
        int mode = extract<int>(params[25]);
        std::size_t cache_id = extract<std::size_t>(params[26]);
        std::size_t prof_begin = extract<std::size_t>(params[27]);
        std::size_t prof_end = extract<std::size_t>(params[28]);

        std::int64_t n = static_cast<std::int64_t>(slab_ind.size());
        if (n <= 0) return;

        const void* vol_raw = reinterpret_cast<const void*>(vol_addr);
        const uint8_t* static_mask = reinterpret_cast<const uint8_t*>(static_mask_addr);
        const uint8_t* rejected = reinterpret_cast<const uint8_t*>(rejected_addr);
        std::size_t frame_stride = img_h * img_w;
        int H = static_cast<int>(img_h);
        int W = static_cast<int>(img_w);
        float sat_f = static_cast<float>(saturation);

        vec3<double> s0 = beam.get_s0();
        vec3<double> m2 = goniometer.get_rotation_axis();
        double s0_length = s0.length();
        double delta_b_r = 1.0 / delta_b;
        double delta_b_r2 = delta_b_r * delta_b_r;
        vec2<int> scan_arr_range = scan.get_array_range();
        int scan_index0 = scan_arr_range[0];
        int scan_index1 = scan_arr_range[1];
        vec2<double> scan_osc = scan.get_oscillation();
        double phi0 = scan_osc[0];
        double dphi = scan_osc[1];
        double arr0 = static_cast<double>(scan_index0);

        std::vector<PanelTransformInfo> panel_infos = extract_panel_transform_infos(detector);
        const PanelTransformInfo& pinfo0 = panel_infos[0];
        double panel_gain = detector[0].get_gain();
        double panel_ped = detector[0].get_pedestal();
        bool has_ped = (std::abs(panel_ped) > 1e-7);
        bool has_gain = (panel_gain > 0.0 && std::abs(panel_gain - 1.0) > 1e-7);
        bool has_ped_or_gain = (has_ped || has_gain);
        double px_size_x = pinfo0.px_x;
        double px_size_y = pinfo0.px_y;
        double px_size_x2 = px_size_x * px_size_x;
        double px_size_y2 = px_size_y * px_size_y;
        double osc_w2 = dphi * dphi;
        vec3<double> fast_axis = pinfo0.fast;
        vec3<double> slow_axis = pinfo0.slow;
        vec3<double> origin = pinfo0.origin;
        vec3<double> n_vec = fast_axis.cross(slow_axis);
        if (origin * n_vec < 0.0) {
            n_vec = -n_vec;
        }
        constexpr double half_sq = 1.0 / 12.0;

        double sb = 0.0, sm = 0.0, ns = 0.0;
        std::size_t hgs = 0, g_dim = 0, g_stride_k = 0, g_vol = 0, n_prof = 0, prof_elems = 0;
        std::vector<data_const_reference> prof_data;
        std::vector<mask_const_reference> prof_mask;
        std::vector<uint8_t> prof_valid;

        if (self != nullptr) {
            if (self->fit_method_ != ReciprocalSpace) {
                throw DIALS_ERROR("DetectorSpace not supported in integrate_slab_from_volume_fast");
            }
            sb = self->sigma_b_;
            sm = self->sigma_m_;
            ns = self->n_sigma_;
            hgs = self->grid_size_;
            g_dim = 2 * hgs + 1;
            g_stride_k = g_dim * g_dim;
            g_vol = g_dim * g_stride_k;
            n_prof = self->data_.size();
            prof_elems = self->accessor_.size_1d();
            if (mode == 1) {
                prof_data.resize(n_prof);
                prof_mask.resize(n_prof);
                prof_valid.assign(n_prof, 0);
                for (std::size_t k = 0; k < n_prof; ++k) {
                    if (self->data_[k].size() != 0 && self->mask_[k].size() != 0) {
                        prof_data[k] = self->data_[k].const_ref();
                        prof_mask[k] = self->mask_[k].const_ref();
                        prof_valid[k] = 1;
                    }
                }
            }
        }

        std::shared_ptr<CachedTransformStore> global_store;
        if (mode == 2) {
            std::lock_guard<std::mutex> lock(g_transform_cache_mtx);
            auto it = g_transform_caches.find(cache_id);
            DIALS_ASSERT(it != g_transform_caches.end());
            global_store = it->second;
        }
        std::unique_ptr<CachedTransformStore> local_slab_store;
        if (mode == 3) {
            local_slab_store = std::make_unique<CachedTransformStore>(
                static_cast<std::size_t>(n), g_dim
            );
        }
        CachedTransformStore* active_store = (mode == 2) ? global_store.get()
                                           : (mode == 3) ? local_slab_store.get()
                                           : nullptr;

        int nt = num_worker_threads();
        if (nt > n) nt = static_cast<int>(n);

        struct ThreadSlabScratch {
            std::vector<float> sb_data;
            std::vector<float> sb_bg;
            std::vector<uint8_t> sb_mask;
            std::vector<double> dxy_buf;
            std::vector<uint8_t> mv_buf;
            ThreadTransformScratch trans;
        };

        std::vector<ThreadSlabScratch> thread_scratch(static_cast<std::size_t>(nt));
        std::vector<std::vector<std::vector<double>>> thread_data;
        std::vector<std::vector<std::size_t>> thread_n_refl;
        if (mode == 0 || mode == 2 || mode == 3) {
            thread_data.assign(
                static_cast<std::size_t>(nt),
                std::vector<std::vector<double>>(n_prof)
            );
            thread_n_refl.assign(
                static_cast<std::size_t>(nt),
                std::vector<std::size_t>(n_prof, 0)
            );
        }

        constexpr uint8_t m_bg_val = Background | Valid;
        constexpr uint8_t m_fg_val = Foreground | Valid;
        constexpr uint8_t m_ovlp = Overlapped;
        constexpr uint8_t m_used = BackgroundUsed;
        constexpr uint8_t m_not_used = static_cast<uint8_t>(255 ^ m_used);
        constexpr uint8_t m_bg_used_val = Valid | Background | BackgroundUsed;

        std::atomic<std::int64_t> next_idx{0};
        constexpr std::int64_t chunk = 16;

        auto worker = [&](int tid) {
            auto& sc = thread_scratch[static_cast<std::size_t>(tid)];
            while (true) {
                std::int64_t start = next_idx.fetch_add(chunk, std::memory_order_relaxed);
                if (start >= n) break;
                std::int64_t end = std::min(start + chunk, n);
                for (std::int64_t i = start; i < end; ++i) {
                    std::size_t r = slab_ind[static_cast<std::size_t>(i)];
                    scitbx::af::int6 b = bbox_col[r];
                    int x0 = b[0], x1 = b[1], y0 = b[2], y1 = b[3], z0 = b[4], z1 = b[5];
                    int nx = x1 - x0, ny = y1 - y0, nz = z1 - z0;
                    if (nx <= 0 || ny <= 0 || nz <= 0 || z0 < frame0 || z1 > frame1) {
                        continue;
                    }

                    std::size_t slice_sz = static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);
                    std::size_t sz = static_cast<std::size_t>(nz) * slice_sz;
                    if (sc.sb_data.size() < sz) {
                        sc.sb_data.resize(sz);
                        sc.sb_bg.resize(sz);
                        sc.sb_mask.resize(sz);
                    }
                    float* d_ptr = sc.sb_data.data();
                    float* bg_ptr = sc.sb_bg.data();
                    uint8_t* m_ptr = sc.sb_mask.data();
                    std::size_t fl = flags_col[r];

                    // Step A: Extract shoebox from volume + overload check
                    bool ovl = false;
                    for (int kz = 0; kz < nz; ++kz) {
                        int f_rel = (z0 + kz) - frame0;
                        bool f_rej = rejected[f_rel] != 0;
                        std::size_t f_base = static_cast<std::size_t>(f_rel) * frame_stride;
                        std::size_t z_off = static_cast<std::size_t>(kz) * slice_sz;
                        for (int ky = 0; ky < ny; ++ky) {
                            int iy = y0 + ky;
                            std::size_t y_off = z_off + static_cast<std::size_t>(ky) * static_cast<std::size_t>(nx);
                            if (iy < 0 || iy >= H) {
                                for (int kx = 0; kx < nx; ++kx) {
                                    d_ptr[y_off + kx] = 0.0f;
                                    m_ptr[y_off + kx] = 0;
                                }
                                continue;
                            }
                            std::size_t row_base = f_base + static_cast<std::size_t>(iy) * img_w;
                            const uint8_t* sm_row = static_mask + static_cast<std::size_t>(iy) * img_w;
                            for (int kx = 0; kx < nx; ++kx) {
                                int ix = x0 + kx;
                                std::size_t out_idx = y_off + static_cast<std::size_t>(kx);
                                if (ix < 0 || ix >= W) {
                                    d_ptr[out_idx] = 0.0f;
                                    m_ptr[out_idx] = 0;
                                } else {
                                    int32_t pv = read_raw_pixel_i32(
                                        vol_raw, row_base + static_cast<std::size_t>(ix),
                                        vol_dtype_code, vol_invalid_top
                                    );
                                    float fv;
                                    if (__builtin_expect(has_ped_or_gain, 0)) {
                                        double dv = static_cast<double>(pv);
                                        if (has_ped) dv -= panel_ped;
                                        if (has_gain) dv /= panel_gain;
                                        fv = static_cast<float>(dv);
                                    } else {
                                        fv = static_cast<float>(pv);
                                    }
                                    d_ptr[out_idx] = fv;
                                    bool valid = (!f_rej) && (sm_row[ix] != 0) && (pv >= tr_min) && (pv <= tr_max);
                                    if (fv > sat_f) {
                                        valid = false;
                                        ovl = true;
                                    }
                                    m_ptr[out_idx] = valid ? Valid : 0;
                                }
                            }
                        }
                    }
                    if (ovl) {
                        fl |= dials::af::Overloaded;
                    }

                    // Step B: 3D GaussianRS shoebox mask
                    std::size_t p_idx = panel_col[r];
                    const dxtbx::model::Panel& panel = detector[p_idx];
                    const PanelTransformInfo& pinfo = panel_infos[p_idx];
                    double frame = xyzcal_px_col[r][2];
                    double phi = phi0 + (frame - scan_index0) * dphi;
                    vec3<double> s1_r = s1_col[r];
                    CoordinateSystem cs_mask(m2, s0, s1_r, phi);
                    double s1_len = cs_mask.s1().length();
                    vec3<double> scaled_e1 = cs_mask.e1_axis() * (1.0 / s1_len);
                    vec3<double> scaled_e2 = cs_mask.e2_axis() * (1.0 / s1_len);

                    std::size_t stride_x = static_cast<std::size_t>(nx + 1);
                    std::size_t dxy_sz = static_cast<std::size_t>(ny + 1) * stride_x;
                    if (sc.dxy_buf.size() < dxy_sz) {
                        sc.dxy_buf.resize(dxy_sz);
                    }
                    if (pinfo.is_pure_simple) {
                        for (int j = 0; j <= ny; ++j) {
                            double my = static_cast<double>(y0 + j) * pinfo.px_y;
                            vec3<double> row_base = pinfo.origin + pinfo.slow * my;
                            for (int ix_c = 0; ix_c <= nx; ++ix_c) {
                                double mx = static_cast<double>(x0 + ix_c) * pinfo.px_x;
                                vec3<double> lab = row_base + pinfo.fast * mx;
                                vec3<double> ds1 = lab.normalize() * s0_length - s1_r;
                                double gx = scaled_e1 * ds1;
                                double gy = scaled_e2 * ds1;
                                sc.dxy_buf[static_cast<std::size_t>(j) * stride_x + ix_c] =
                                    (gx * gx + gy * gy) * delta_b_r2;
                            }
                        }
                    } else if (pinfo.is_pure_parallax) {
                        vec2<double> shoebox_centroid_px = panel.get_ray_intersection_px(s1_r);
                        double attenuation_length = panel.attenuation_length(shoebox_centroid_px);
                        for (int j = 0; j <= ny; ++j) {
                            double my = static_cast<double>(y0 + j) * pinfo.px_y;
                            vec3<double> row_base = pinfo.origin + my * pinfo.slow;
                            for (int ix_c = 0; ix_c <= nx; ++ix_c) {
                                double mx = static_cast<double>(x0 + ix_c) * pinfo.px_x;
                                vec3<double> s1_raw = row_base + mx * pinfo.fast;
                                vec3<double> s1_n = s1_raw.normalize();
                                double cx_p = mx - (s1_n * pinfo.fast) * attenuation_length;
                                double cy_p = my - (s1_n * pinfo.slow) * attenuation_length;
                                vec3<double> lab = pinfo.origin + cx_p * pinfo.fast + cy_p * pinfo.slow;
                                vec3<double> ds1 = lab.normalize() * s0_length - s1_r;
                                double gx = scaled_e1 * ds1;
                                double gy = scaled_e2 * ds1;
                                sc.dxy_buf[static_cast<std::size_t>(j) * stride_x + ix_c] =
                                    (gx * gx + gy * gy) * delta_b_r2;
                            }
                        }
                    } else {
                        vec2<double> shoebox_centroid_px = panel.get_ray_intersection_px(s1_r);
                        double attenuation_length = panel.attenuation_length(shoebox_centroid_px);
                        for (int j = 0; j <= ny; ++j) {
                            for (int ix_c = 0; ix_c <= nx; ++ix_c) {
                                vec3<double> ds1 =
                                    panel.get_pixel_lab_coord(vec2<double>(x0 + ix_c, y0 + j), attenuation_length).normalize() * s0_length - s1_r;
                                double gx = scaled_e1 * ds1;
                                double gy = scaled_e2 * ds1;
                                sc.dxy_buf[static_cast<std::size_t>(j) * stride_x + ix_c] =
                                    (gx * gx + gy * gy) * delta_b_r2;
                            }
                        }
                    }

                    int k_start = std::max(0, scan_index0 - z0);
                    int k_end = std::min(nz, scan_index1 - z0);
                    if (k_start < k_end) {
                        if (sc.mv_buf.size() < slice_sz) {
                            sc.mv_buf.resize(slice_sz);
                        }
                        for (int j = 0; j < ny; ++j) {
                            const double* r0 = &sc.dxy_buf[static_cast<std::size_t>(j) * stride_x];
                            const double* r1 = r0 + stride_x;
                            uint8_t* mv_row = sc.mv_buf.data() + static_cast<std::size_t>(j) * nx;
                            for (int ix_c = 0; ix_c < nx; ++ix_c) {
                                double dxy = std::min(std::min(r0[ix_c], r1[ix_c]), std::min(r0[ix_c + 1], r1[ix_c + 1]));
                                mv_row[ix_c] = (dxy <= 1.0) ? Foreground : Background;
                            }
                        }
                        uint8_t* m_first = m_ptr + static_cast<std::size_t>(k_start) * slice_sz;
                        int n_k = k_end - k_start;
                        const uint8_t* mv_data = sc.mv_buf.data();
                        for (int k = 0; k < n_k; ++k) {
                            uint8_t* m_slice = m_first + static_cast<std::size_t>(k) * slice_sz;
                            for (std::size_t p = 0; p < slice_sz; ++p) {
                                m_slice[p] |= mv_data[p];
                            }
                        }
                    }

                    // Step C: Pre-background mask validation + foreground count
                    if (mode != 0) {
                        int nbackg = 0, nvalbg = 0, nforeg = 0, nvalfg = 0;
                        for (std::size_t p = 0; p < sz; ++p) {
                            uint8_t m = m_ptr[p];
                            if (m & Background) {
                                nbackg++;
                                if ((m & m_bg_val) == m_bg_val) nvalbg++;
                            }
                            if (m & Foreground) {
                                nforeg++;
                                if ((m & m_fg_val) == m_fg_val) nvalfg++;
                            }
                        }
                        if (nbackg > nvalbg) fl |= dials::af::BackgroundIncludesBadPixels;
                        if (nforeg > nvalfg) fl |= dials::af::ForegroundIncludesBadPixels;
                        double frac = static_cast<double>(nvalfg) / static_cast<double>(nforeg);
                        if (frac < valid_fg_threshold) fl |= dials::af::DontIntegrate;
                        nforeg_col[r] = nvalfg;
                    }

                    // Step D: GLM Background (constant3d or constant2d)
                    if (bg_alg_code == 0) {
                        int32_t h_r[128];
                        std::memset(h_r, 0, sizeof(h_r));
                        int max_bin = -1;
                        int64_t cnt = 0;
                        int64_t tail_cnt = 0;
                        bool neg = false;
                        for (std::size_t p = 0; p < sz; ++p) {
                            uint8_t m = m_ptr[p];
                            if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                                cnt++;
                                float v = d_ptr[p];
                                if (v < 0.0f) {
                                    neg = true;
                                } else {
                                    int iv = static_cast<int>(v);
                                    if (static_cast<float>(iv) == v && iv >= 0 && iv < 128) {
                                        h_r[iv]++;
                                        if (iv > max_bin) max_bin = iv;
                                    } else {
                                        tail_cnt++;
                                    }
                                }
                            }
                        }
                        bool bg_ok = false;
                        float m_val_f = 0.0f;
                        if (cnt >= bg_min_pixels && !neg) {
                            std::vector<double> tail;
                            double med = 0.0;
                            if (tail_cnt == 0) {
                                int64_t target = cnt / 2;
                                int64_t acc = 0;
                                for (int k = 0; k <= max_bin; ++k) {
                                    acc += h_r[k];
                                    if (acc > target) {
                                        med = static_cast<double>(k);
                                        break;
                                    }
                                }
                            } else {
                                tail.reserve(static_cast<std::size_t>(tail_cnt));
                                std::vector<double> tmp;
                                tmp.reserve(static_cast<std::size_t>(cnt));
                                for (std::size_t p = 0; p < sz; ++p) {
                                    uint8_t m = m_ptr[p];
                                    if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                                        float v = d_ptr[p];
                                        double dv = static_cast<double>(v);
                                        tmp.push_back(dv);
                                        int iv = static_cast<int>(v);
                                        if (!(static_cast<float>(iv) == v && iv >= 0 && iv < 128)) {
                                            tail.push_back(dv);
                                        }
                                    }
                                }
                                std::nth_element(tmp.begin(), tmp.begin() + cnt / 2, tmp.end());
                                med = tmp[cnt / 2];
                            }
                            if (med <= 0.0) med = 1.0;
                            double m_val_d = 0.0;
                            bg_ok = solve_robust_poisson_hist(
                                h_r, max_bin, tail, cnt, med, bg_tuning_constant, 1e-3, bg_max_iter, m_val_d
                            );
                            if (bg_ok) {
                                m_val_f = static_cast<float>(m_val_d);
                            }
                        }
                        if (bg_ok) {
                            std::fill(bg_ptr, bg_ptr + sz, m_val_f);
                            for (std::size_t p = 0; p < sz; ++p) {
                                uint8_t m = m_ptr[p];
                                if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                                    m_ptr[p] = m | m_used;
                                }
                            }
                            if (mode != 0) bg_mean_col[r] = static_cast<double>(m_val_f);
                        } else {
                            std::fill(bg_ptr, bg_ptr + sz, 0.0f);
                            if (mode != 0) bg_mean_col[r] = 0.0;
                            fl |= (dials::af::DontIntegrate | dials::af::FailedDuringBackgroundModelling);
                        }
                    } else {
                        bool all_ok = true;
                        double sum_bg = 0.0;
                        int64_t total_used = 0;
                        for (int kz = 0; kz < nz; ++kz) {
                            std::size_t z_off = static_cast<std::size_t>(kz) * slice_sz;
                            const float* dz = d_ptr + z_off;
                            uint8_t* mz = m_ptr + z_off;
                            float* bgz = bg_ptr + z_off;
                            int32_t h_r[128];
                            std::memset(h_r, 0, sizeof(h_r));
                            int max_bin = -1;
                            int64_t cnt = 0;
                            int64_t tail_cnt = 0;
                            bool neg = false;
                            for (std::size_t p = 0; p < slice_sz; ++p) {
                                uint8_t m = mz[p];
                                if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                                    cnt++;
                                    float v = dz[p];
                                    if (v < 0.0f) {
                                        neg = true;
                                    } else {
                                        int iv = static_cast<int>(v);
                                        if (static_cast<float>(iv) == v && iv >= 0 && iv < 128) {
                                            h_r[iv]++;
                                            if (iv > max_bin) max_bin = iv;
                                        } else {
                                            tail_cnt++;
                                        }
                                    }
                                }
                            }
                            if (cnt < bg_min_pixels || neg) {
                                all_ok = false;
                                break;
                            }
                            std::vector<double> tail;
                            double med = 0.0;
                            if (tail_cnt == 0) {
                                int64_t target = cnt / 2;
                                int64_t acc_c = 0;
                                for (int bk = 0; bk <= max_bin; ++bk) {
                                    acc_c += h_r[bk];
                                    if (acc_c > target) {
                                        med = static_cast<double>(bk);
                                        break;
                                    }
                                }
                            } else {
                                tail.reserve(static_cast<std::size_t>(tail_cnt));
                                std::vector<double> tmp;
                                tmp.reserve(static_cast<std::size_t>(cnt));
                                for (std::size_t p = 0; p < slice_sz; ++p) {
                                    uint8_t m = mz[p];
                                    if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                                        float v = dz[p];
                                        double dv = static_cast<double>(v);
                                        tmp.push_back(dv);
                                        int iv = static_cast<int>(v);
                                        if (!(static_cast<float>(iv) == v && iv >= 0 && iv < 128)) {
                                            tail.push_back(dv);
                                        }
                                    }
                                }
                                std::nth_element(tmp.begin(), tmp.begin() + cnt / 2, tmp.end());
                                med = tmp[cnt / 2];
                            }
                            if (med <= 0.0) med = 1.0;
                            double m_val_d = 0.0;
                            bool ok = solve_robust_poisson_hist(
                                h_r, max_bin, tail, cnt, med, bg_tuning_constant, 1e-3, bg_max_iter, m_val_d
                            );
                            if (!ok) {
                                all_ok = false;
                                break;
                            }
                            float m_val_f = static_cast<float>(m_val_d);
                            std::fill(bgz, bgz + slice_sz, m_val_f);
                            for (std::size_t p = 0; p < slice_sz; ++p) {
                                uint8_t m = mz[p];
                                if (((m & m_bg_val) == m_bg_val) && ((m & m_ovlp) == 0)) {
                                    mz[p] = m | m_used;
                                }
                            }
                            sum_bg += static_cast<double>(m_val_f) * static_cast<double>(cnt);
                            total_used += cnt;
                        }
                        if (all_ok && total_used > 0) {
                            if (mode != 0) bg_mean_col[r] = sum_bg / static_cast<double>(total_used);
                        } else {
                            std::fill(bg_ptr, bg_ptr + sz, 0.0f);
                            for (std::size_t p = 0; p < sz; ++p) {
                                m_ptr[p] &= m_not_used;
                            }
                            if (mode != 0) bg_mean_col[r] = 0.0;
                            fl |= (dials::af::DontIntegrate | dials::af::FailedDuringBackgroundModelling);
                        }
                    }

                    // Step E: Summation + Centroid + Post-mask counts
                    double sum_p = 0.0, sum_b = 0.0;
                    int64_t fg_cnt = 0, bg_cnt = 0;
                    int32_t bad_cnt = 0;
                    int nv_post = 0, nb_post = 0, nbu_post = 0;
                    for (std::size_t p = 0; p < sz; ++p) {
                        uint8_t m = m_ptr[p];
                        if (m & Valid) {
                            nv_post++;
                            if (m & Background) {
                                nb_post++;
                                if (m & BackgroundUsed) nbu_post++;
                            }
                        }
                        if (m & Foreground) {
                            if (((m & m_fg_val) == m_fg_val) && ((m & m_ovlp) == 0)) {
                                sum_p += static_cast<double>(d_ptr[p]);
                                sum_b += static_cast<double>(bg_ptr[p]);
                                fg_cnt++;
                            } else {
                                bad_cnt++;
                            }
                        } else if ((m & m_bg_used_val) == m_bg_used_val) {
                            bg_cnt++;
                        }
                    }

                    if (bad_cnt == 0) {
                        fl |= dials::af::IntegratedSum;
                    } else {
                        fl |= dials::af::FailedDuringSummation;
                    }

                    if (mode != 0) {
                        double m_ratio = (bg_cnt > 0) ? (static_cast<double>(fg_cnt) / static_cast<double>(bg_cnt)) : 0.0;
                        double ival = sum_p - sum_b;
                        double bg_v = std::abs(sum_b) * (1.0 + m_ratio);
                        int_sum_val_col[r] = ival;
                        int_sum_var_col[r] = std::abs(ival) + bg_v;
                        bg_sum_val_col[r] = sum_b;
                        bg_sum_var_col[r] = bg_v;
                        nvalid_col[r] = nv_post;
                        nbg_col[r] = nb_post;
                        nbg_used_col[r] = nbu_post;

                        // Centroid
                        double bx = static_cast<double>(x0);
                        double by = static_cast<double>(y0);
                        double bz = static_cast<double>(z0);
                        double cx = static_cast<double>(x1 + x0) * 0.5;
                        double cy = static_cast<double>(y1 + y0) * 0.5;
                        double cz = static_cast<double>(z1 + z0) * 0.5;
                        double vx = 0.0, vy = 0.0, vz = 0.0;
                        int64_t c_cnt = 0;
                        double sum_w = 0.0, sum_wsq = 0.0, sum_wx = 0.0, sum_wy = 0.0, sum_wz = 0.0;
                        for (int kz = 0; kz < nz; ++kz) {
                            double zc = static_cast<double>(kz) + 0.5;
                            for (int ky = 0; ky < ny; ++ky) {
                                double yc = static_cast<double>(ky) + 0.5;
                                std::size_t row_off = (static_cast<std::size_t>(kz) * static_cast<std::size_t>(ny) + static_cast<std::size_t>(ky)) * static_cast<std::size_t>(nx);
                                for (int kx = 0; kx < nx; ++kx) {
                                    uint8_t m = m_ptr[row_off + kx];
                                    if (((m & m_fg_val) == m_fg_val) && ((m & m_ovlp) == 0)) {
                                        double val = static_cast<double>(d_ptr[row_off + kx]) - static_cast<double>(bg_ptr[row_off + kx]);
                                        if (val > 0.0) {
                                            double xc = static_cast<double>(kx) + 0.5;
                                            c_cnt++;
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
                        if (c_cnt > 0 && sum_w > 0.0) {
                            double inv_w = 1.0 / sum_w;
                            double mean_x = sum_wx * inv_w;
                            double mean_y = sum_wy * inv_w;
                            double mean_z = sum_wz * inv_w;
                            cx = mean_x + bx;
                            cy = mean_y + by;
                            cz = mean_z + bz;
                            double denom = (sum_w * sum_w) - sum_wsq;
                            if (c_cnt > 1 && denom > 0.0) {
                                double sum_wdx2 = 0.0, sum_wdy2 = 0.0, sum_wdz2 = 0.0;
                                for (int kz = 0; kz < nz; ++kz) {
                                    double dz = (static_cast<double>(kz) + 0.5) - mean_z;
                                    double dz2 = dz * dz;
                                    for (int ky = 0; ky < ny; ++ky) {
                                        double dy = (static_cast<double>(ky) + 0.5) - mean_y;
                                        double dy2 = dy * dy;
                                        std::size_t row_off = (static_cast<std::size_t>(kz) * static_cast<std::size_t>(ny) + static_cast<std::size_t>(ky)) * static_cast<std::size_t>(nx);
                                        for (int kx = 0; kx < nx; ++kx) {
                                            uint8_t m = m_ptr[row_off + kx];
                                            if (((m & m_fg_val) == m_fg_val) && ((m & m_ovlp) == 0)) {
                                                double val = static_cast<double>(d_ptr[row_off + kx]) - static_cast<double>(bg_ptr[row_off + kx]);
                                                if (val > 0.0) {
                                                    double dx = (static_cast<double>(kx) + 0.5) - mean_x;
                                                    sum_wdx2 += val * (dx * dx);
                                                    sum_wdy2 += val * dy2;
                                                    sum_wdz2 += val * dz2;
                                                }
                                            }
                                        }
                                    }
                                }
                                double scale = 1.0 / denom;
                                vx = sum_wdx2 * scale + half_sq;
                                vy = sum_wdy2 * scale + half_sq;
                                vz = sum_wdz2 * scale + half_sq;
                            }
                        }
                        xyzobs_px_val_col[r] = vec3<double>(cx, cy, cz);
                        xyzobs_px_var_col[r] = vec3<double>(vx, vy, vz);

                        double mx = cx * px_size_x;
                        double my = cy * px_size_y;
                        if (px2mm_mode == 1) {
                            vec3<double> s1_c = origin + mx * fast_axis + my * slow_axis;
                            double slen = s1_c.length();
                            if (slen > 0.0) {
                                vec3<double> s1_n = s1_c * (1.0 / slen);
                                double cos_t = s1_n * n_vec;
                                if (mu > 0.0 && cos_t != 0.0) {
                                    double inv_mu = 1.0 / mu;
                                    double t0_over_cos_t = t0 / cos_t;
                                    double atten_o = inv_mu - (t0_over_cos_t + inv_mu) * std::exp(-mu * t0_over_cos_t);
                                    mx -= (s1_n * fast_axis) * atten_o;
                                    my -= (s1_n * slow_axis) * atten_o;
                                }
                            }
                        }
                        double mz = (dphi == 0.0 && phi0 == 0.0) ? 0.0 : (phi0 + (cz - arr0) * dphi);
                        xyzobs_mm_val_col[r] = vec3<double>(mx, my, mz);
                        xyzobs_mm_var_col[r] = vec3<double>(vx * px_size_x2, vy * px_size_y2, vz * osc_w2);
                    }

                    // Step F: GaussianRS Transform / Profile Modelling / Profile Fitting
                    if (self != nullptr) {
                        if (mode == 0) {
                            bool full = partiality_col[r] > 0.99;
                            bool integrated = (fl & dials::af::IntegratedSum) != 0;
                            bool bbox_valid = (x0 >= 0 && y0 >= 0 && x1 <= pinfo.img_w && y1 <= pinfo.img_h);
                            if (full && integrated && bbox_valid && bad_cnt == 0) {
                                try {
                                    CoordinateSystem cs(m2, s0, s1_r, xyzcal_mm_col[r][2]);
                                    run_transform_forward_raw(
                                        panel, pinfo, scan_index0, phi0, dphi, sb, sm, ns, hgs,
                                        cs, b, d_ptr, bg_ptr, m_ptr, false, false, sc.trans
                                    );
                                    if (sc.trans.has_fg_pixels) {
                                        const double* prof_ptr = sc.trans.profile.data();
                                        double sum_data = 0.0;
                                        for (std::size_t p_i = 0; p_i < prof_elems; ++p_i) {
                                            sum_data += prof_ptr[p_i];
                                        }
                                        if (sum_data > 0.0) {
                                            double inv_sum = 1.0 / sum_data;
                                            auto& loc_data = thread_data[static_cast<std::size_t>(tid)];
                                            auto& loc_n = thread_n_refl[static_cast<std::size_t>(tid)];
                                            dials::af::shared<std::size_t> indices = self->sampler_->nearest_n(p_idx, xyzcal_px_col[r]);
                                            for (std::size_t j = 0; j < indices.size(); ++j) {
                                                std::size_t idx = indices[j];
                                                double w = self->sampler_->weight(idx, p_idx, xyzcal_px_col[r]) * inv_sum;
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
                                    }
                                } catch (...) {}
                            }
                        } else if (mode == 1) {
                            int_prf_val_col[r] = 0.0;
                            int_prf_var_col[r] = -1.0;
                            prf_corr_col[r] = 0.0;
                            fl &= ~dials::af::IntegratedPrf;
                            bool fit_ok = false;
                            if (!(fl & dials::af::DontIntegrate)) {
                                try {
                                    std::size_t index = self->sampler_->nearest(p_idx, xyzcal_px_col[r]);
                                    if (index < n_prof && prof_valid[index]) {
                                        CoordinateSystem cs(m2, s0, s1_r, xyzcal_mm_col[r][2]);
                                        run_transform_forward_raw(
                                            panel, pinfo, scan_index0, phi0, dphi, sb, sm, ns, hgs,
                                            cs, b, d_ptr, bg_ptr, m_ptr, true, false, sc.trans
                                        );
                                        if (sc.trans.n_valid_kk > 0 && sc.trans.has_fg_pixels) {
                                            double fit_i = 0.0, fit_v = -1.0, fit_c = 0.0;
                                            if (fit_1d_noexcept_slices(
                                                    sc.trans.profile.data(),
                                                    sc.trans.background.data(),
                                                    false,
                                                    prof_mask[index].begin(),
                                                    sc.trans.valid_kk_list,
                                                    sc.trans.n_valid_kk,
                                                    g_stride_k,
                                                    prof_data[index].begin(),
                                                    g_vol,
                                                    1e-3, 100, sc.trans,
                                                    fit_i, fit_v, fit_c)) {
                                                int_prf_val_col[r] = fit_i;
                                                int_prf_var_col[r] = fit_v;
                                                prf_corr_col[r] = fit_c;
                                                fl |= dials::af::IntegratedPrf;
                                                fit_ok = true;
                                            }
                                        }
                                    }
                                } catch (...) {}
                            }
                            if (!fit_ok) {
                                fl |= dials::af::FailedDuringProfileFitting;
                            }
                        } else if (mode == 2 || mode == 3) {
                            std::size_t store_idx = (mode == 2) ? r : static_cast<std::size_t>(i);
                            active_store->was_processed[store_idx] = 1;
                            bool do_fit = (fl & dials::af::DontIntegrate) == 0;
                            bool is_ref = ((fl & dials::af::ReferenceSpot) != 0)
                                          && (partiality_col[r] > 0.99)
                                          && ((fl & dials::af::IntegratedSum) != 0)
                                          && (x0 >= 0 && y0 >= 0 && x1 <= pinfo.img_w && y1 <= pinfo.img_h)
                                          && (bad_cnt == 0);
                            if (do_fit || is_ref) {
                                try {
                                    CoordinateSystem cs(m2, s0, s1_r, xyzcal_mm_col[r][2]);
                                    run_transform_forward_raw(
                                        panel, pinfo, scan_index0, phi0, dphi, sb, sm, ns, hgs,
                                        cs, b, d_ptr, bg_ptr, m_ptr, true, is_ref, sc.trans
                                    );
                                    if (is_ref) {
                                        if (mode == 3) {
                                            fl |= dials::af::UsedInModelling;
                                        }
                                        if (sc.trans.has_fg_pixels) {
                                            const double* net_ptr = sc.trans.net_profile.data();
                                            double sum_data = 0.0;
                                            for (std::size_t p_i = 0; p_i < prof_elems; ++p_i) {
                                                sum_data += net_ptr[p_i];
                                            }
                                            if (sum_data > 0.0) {
                                                double inv_sum = 1.0 / sum_data;
                                                auto& loc_data = thread_data[static_cast<std::size_t>(tid)];
                                                auto& loc_n = thread_n_refl[static_cast<std::size_t>(tid)];
                                                dials::af::shared<std::size_t> indices = self->sampler_->nearest_n(p_idx, xyzcal_px_col[r]);
                                                for (std::size_t j = 0; j < indices.size(); ++j) {
                                                    std::size_t idx = indices[j];
                                                    double w = self->sampler_->weight(idx, p_idx, xyzcal_px_col[r]) * inv_sum;
                                                    auto& vec = loc_data[idx];
                                                    if (vec.empty()) {
                                                        vec.assign(prof_elems, 0.0);
                                                    }
                                                    for (std::size_t p_i = 0; p_i < prof_elems; ++p_i) {
                                                        vec[p_i] += w * net_ptr[p_i];
                                                    }
                                                    loc_n[idx]++;
                                                }
                                            }
                                        }
                                    }
                                    if (do_fit && sc.trans.has_fg_pixels && sc.trans.n_valid_kk > 0) {
                                        auto& c_store = active_store->thread_c_slices[static_cast<std::size_t>(tid)];
                                        auto& b_store = active_store->thread_b_slices[static_cast<std::size_t>(tid)];
                                        uint32_t sl_off = static_cast<uint32_t>(c_store.size() / g_stride_k);
                                        std::size_t add_elems = static_cast<std::size_t>(sc.trans.n_valid_kk) * g_stride_k;
                                        std::size_t old_sz = c_store.size();
                                        c_store.resize(old_sz + add_elems);
                                        b_store.resize(old_sz + add_elems);
                                        double* c_dst = c_store.data() + old_sz;
                                        double* b_dst = b_store.data() + old_sz;
                                        const double* prof_ptr = sc.trans.profile.data();
                                        const double* bg_t_ptr = sc.trans.background.data();
                                        for (int idx_kk = 0; idx_kk < sc.trans.n_valid_kk; ++idx_kk) {
                                            std::size_t kk_off = static_cast<std::size_t>(sc.trans.valid_kk_list[idx_kk]) * g_stride_k;
                                            std::memcpy(
                                                c_dst + static_cast<std::size_t>(idx_kk) * g_stride_k,
                                                prof_ptr + kk_off,
                                                g_stride_k * sizeof(double)
                                            );
                                            std::memcpy(
                                                b_dst + static_cast<std::size_t>(idx_kk) * g_stride_k,
                                                bg_t_ptr + kk_off,
                                                g_stride_k * sizeof(double)
                                            );
                                        }
                                        active_store->valid_kk_mask[store_idx] = sc.trans.valid_kk_mask;
                                        active_store->tid_of_r[store_idx] = static_cast<uint8_t>(tid);
                                        active_store->slice_offset_of_r[store_idx] = sl_off;
                                    }
                                } catch (...) {}
                            }
                        }
                    }

                    if (mode != 0) {
                        flags_col[r] = fl;
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

        if (self != nullptr && (mode == 0 || mode == 2 || mode == 3)) {
            for (std::size_t idx = 0; idx < n_prof; ++idx) {
                std::size_t tot_n = 0;
                for (int t = 0; t < nt; ++t) {
                    tot_n += thread_n_refl[static_cast<std::size_t>(t)][idx];
                }
                if (tot_n == 0) continue;
                if (self->data_[idx].size() == 0) {
                    self->data_[idx] = data_type(self->accessor_, 0.0);
                    self->mask_[idx] = mask_type(self->accessor_, true);
                }
                double* dst = self->data_[idx].begin();
                for (int t = 0; t < nt; ++t) {
                    const auto& vec = thread_data[static_cast<std::size_t>(t)][idx];
                    if (!vec.empty()) {
                        for (std::size_t p_i = 0; p_i < prof_elems; ++p_i) {
                            dst[p_i] += vec[p_i];
                        }
                    }
                }
                self->n_reflections_[idx] += tot_n;
            }
        }

        if (self != nullptr && mode == 3) {
            std::size_t end_idx = std::min(prof_end, self->data_.size());
            for (std::size_t idx = prof_begin; idx < end_idx; ++idx) {
                if (self->data_[idx].size() != 0) {
                    self->finalize(idx);
                }
            }
            prof_data.resize(n_prof);
            prof_mask.resize(n_prof);
            prof_valid.assign(n_prof, 0);
            for (std::size_t k = 0; k < n_prof; ++k) {
                if (self->data_[k].size() != 0 && self->mask_[k].size() != 0) {
                    prof_data[k] = self->data_[k].const_ref();
                    prof_mask[k] = self->mask_[k].const_ref();
                    prof_valid[k] = 1;
                }
            }
            std::vector<ThreadTransformScratch> fit_scratch(MAX_WORKER_THREADS);
            parallel_for(n, [&](std::int64_t i, int tid) {
                std::size_t si = static_cast<std::size_t>(i);
                if (!local_slab_store->was_processed[si]) {
                    return;
                }
                std::size_t r = slab_ind[si];
                int_prf_val_col[r] = 0.0;
                int_prf_var_col[r] = -1.0;
                prf_corr_col[r] = 0.0;
                flags_col[r] &= ~dials::af::IntegratedPrf;
                if (flags_col[r] & dials::af::DontIntegrate) {
                    flags_col[r] |= dials::af::FailedDuringProfileFitting;
                    return;
                }
                uint32_t vmask = local_slab_store->valid_kk_mask[si];
                if (vmask == 0) {
                    flags_col[r] |= dials::af::FailedDuringProfileFitting;
                    return;
                }
                try {
                    std::size_t index = self->sampler_->nearest(panel_col[r], xyzcal_px_col[r]);
                    if (index >= n_prof || !prof_valid[index]) {
                        flags_col[r] |= dials::af::FailedDuringProfileFitting;
                        return;
                    }
                    data_const_reference p = prof_data[index];
                    mask_const_reference mask1 = prof_mask[index];
                    auto& scratch = fit_scratch[tid];
                    int n_vkk = 0;
                    for (int kk = 0; kk < static_cast<int>(g_dim); ++kk) {
                        if (vmask & (1u << kk)) {
                            scratch.valid_kk_list[n_vkk++] = kk;
                        }
                    }
                    uint8_t src_tid = local_slab_store->tid_of_r[si];
                    std::size_t byte_off = static_cast<std::size_t>(local_slab_store->slice_offset_of_r[si]) * g_stride_k;
                    const double* c_compact = local_slab_store->thread_c_slices[src_tid].data() + byte_off;
                    const double* b_compact = local_slab_store->thread_b_slices[src_tid].data() + byte_off;
                    double fit_i = 0.0, fit_v = -1.0, fit_c = 0.0;
                    if (fit_1d_noexcept_slices(
                            c_compact, b_compact, true,
                            mask1.begin(), scratch.valid_kk_list, n_vkk,
                            g_stride_k, p.begin(), g_vol,
                            1e-3, 100, scratch,
                            fit_i, fit_v, fit_c)) {
                        int_prf_val_col[r] = fit_i;
                        int_prf_var_col[r] = fit_v;
                        prf_corr_col[r] = fit_c;
                        flags_col[r] |= dials::af::IntegratedPrf;
                    } else {
                        flags_col[r] |= dials::af::FailedDuringProfileFitting;
                    }
                } catch (...) {
                    flags_col[r] |= dials::af::FailedDuringProfileFitting;
                }
            });
        }
    }
};

void integrate_slab_from_volume_fast(
    boost::python::object fitter_obj,
    const dials::af::const_ref<std::size_t>& slab_ind,
    boost::python::tuple cols,
    boost::python::tuple params)
{
    ParallelGaussianRS* grs_ptr = nullptr;
    if (!fitter_obj.is_none()) {
        dials::algorithms::MultiExpProfileModeller& fitter =
            boost::python::extract<dials::algorithms::MultiExpProfileModeller&>(fitter_obj);
        DIALS_ASSERT(fitter.size() == 1);
        auto* grs = dynamic_cast<dials::algorithms::GaussianRSProfileModeller*>(fitter[0].get());
        DIALS_ASSERT(grs != nullptr);
        grs_ptr = static_cast<ParallelGaussianRS*>(grs);
    }
    ParallelGaussianRS::integrate_slab_from_volume_cols(grs_ptr, slab_ind, cols, params);
}

dials::af::shared<bool> parallel_model_and_fit_circular_slab(
    dials::algorithms::MultiExpProfileModeller& fitter,
    std::size_t prof_begin,
    std::size_t prof_end,
    dials::af::const_ref<Shoebox<ProfileFloatType>> sbox,
    const dials::af::const_ref<double>& partiality,
    const dials::af::const_ref<scitbx::vec3<double>>& s1,
    const dials::af::const_ref<scitbx::vec3<double>>& xyzpx,
    const dials::af::const_ref<scitbx::vec3<double>>& xyzmm,
    dials::af::ref<std::size_t> flags,
    dials::af::ref<double> intensity_val,
    dials::af::ref<double> intensity_var,
    dials::af::ref<double> reference_cor)
{
    DIALS_ASSERT(fitter.size() == 1);
    auto* grs = dynamic_cast<dials::algorithms::GaussianRSProfileModeller*>(fitter[0].get());
    DIALS_ASSERT(grs != nullptr);
    return static_cast<ParallelGaussianRS*>(grs)->model_and_fit_circular_slab_cols(
        prof_begin, prof_end, sbox, partiality, s1, xyzpx, xyzmm, flags, intensity_val, intensity_var, reference_cor
    );
}

void mark_modeller_finalized(dials::algorithms::MultiExpProfileModeller& fitter) {
    DIALS_ASSERT(fitter.size() == 1);
    auto* grs = dynamic_cast<dials::algorithms::GaussianRSProfileModeller*>(fitter[0].get());
    DIALS_ASSERT(grs != nullptr);
    static_cast<ParallelGaussianRS*>(grs)->mark_finalized();
}

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

std::size_t create_transform_cache(
    const dials::algorithms::MultiExpProfileModeller& fitter,
    std::size_t n_total)
{
    DIALS_ASSERT(fitter.size() == 1);
    const auto* grs = dynamic_cast<const dials::algorithms::GaussianRSProfileModeller*>(fitter[0].get());
    DIALS_ASSERT(grs != nullptr);
    std::size_t half_grid_size = static_cast<const ParallelGaussianRS*>(grs)->get_half_grid_size();
    std::size_t gd = 2 * half_grid_size + 1;
    auto ptr = std::make_shared<CachedTransformStore>(n_total, gd);
    std::lock_guard<std::mutex> lock(g_transform_cache_mtx);
    std::size_t cid = g_next_cache_id++;
    g_transform_caches[cid] = std::move(ptr);
    return cid;
}

void parallel_transform_and_model_slab(
    dials::algorithms::MultiExpProfileModeller& fitter,
    std::size_t cache_id,
    const dials::af::const_ref<std::size_t>& slab_ind,
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
    std::shared_ptr<CachedTransformStore> store;
    {
        std::lock_guard<std::mutex> lock(g_transform_cache_mtx);
        auto it = g_transform_caches.find(cache_id);
        DIALS_ASSERT(it != g_transform_caches.end());
        store = it->second;
    }
    static_cast<ParallelGaussianRS*>(grs)->transform_and_model_slab_cols(
        *store, slab_ind, sbox, partiality, s1, xyzpx, xyzmm, flags
    );
}

dials::af::shared<bool> parallel_fit_from_cache(
    const dials::algorithms::MultiExpProfileModeller& fitter,
    std::size_t cache_id,
    const dials::af::const_ref<std::size_t>& panel,
    const dials::af::const_ref<scitbx::vec3<double>>& xyzpx,
    dials::af::ref<std::size_t> flags,
    dials::af::ref<double> intensity_val,
    dials::af::ref<double> intensity_var,
    dials::af::ref<double> reference_cor)
{
    DIALS_ASSERT(fitter.size() == 1);
    const auto* grs = dynamic_cast<const dials::algorithms::GaussianRSProfileModeller*>(fitter[0].get());
    DIALS_ASSERT(grs != nullptr);
    std::shared_ptr<CachedTransformStore> store;
    {
        std::lock_guard<std::mutex> lock(g_transform_cache_mtx);
        auto it = g_transform_caches.find(cache_id);
        DIALS_ASSERT(it != g_transform_caches.end());
        store = it->second;
    }
    return static_cast<const ParallelGaussianRS*>(grs)->fit_from_cache_cols(
        *store, panel, xyzpx, flags, intensity_val, intensity_var, reference_cor
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
                    double angle = std::atan2(s1_px.cross(s1_cent).length(), s1_px * s1_cent);
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

struct ThreadPredVecs {
    std::vector<cctbx::miller::index<>> hkl;
    std::vector<std::size_t> panel;
    std::vector<bool> enter;
    std::vector<scitbx::vec3<double>> s1;
    std::vector<scitbx::vec3<double>> xyz_px;
    std::vector<scitbx::vec3<double>> xyz_mm;
    std::vector<std::size_t> flags;
};

boost::python::tuple parallel_predict_scan_varying(
    const dxtbx::model::BeamBase& beam,
    const dxtbx::model::Detector& detector,
    const dxtbx::model::Goniometer& goniometer,
    const dxtbx::model::Scan& scan,
    const cctbx::sgtbx::space_group_type& space_group_type,
    double dmin,
    std::size_t margin,
    double padding,
    const dials::af::const_ref<scitbx::mat3<double>>& A)
{
    using namespace dials::algorithms;
    using dials::model::Ray;
    using scitbx::vec2;
    using scitbx::vec3;
    using scitbx::mat3;
    using scitbx::constants::pi;
    using scitbx::math::r3_rotation::axis_and_angle_as_matrix;

    DIALS_ASSERT(A.size() == static_cast<std::size_t>(scan.get_num_images() + 1));

    vec2<int> array_range = scan.get_array_range();
    double a0 = scan.get_oscillation_range()[0];
    double a1 = scan.get_oscillation_range()[1];
    int z0 = static_cast<int>(std::floor(scan.get_array_index_from_angle(a0 - padding * pi / 180.0) + 0.5));
    int z1 = static_cast<int>(std::floor(scan.get_array_index_from_angle(a1 + padding * pi / 180.0) + 0.5));
    const int offset = array_range[0];
    std::int64_t n_frames = static_cast<std::int64_t>(std::max(0, z1 - z0));

    vec3<double> m2 = goniometer.get_rotation_axis_datum();
    vec3<double> s0 = beam.get_s0();
    mat3<double> r_fixed = goniometer.get_fixed_rotation();
    mat3<double> r_setting = goniometer.get_setting_rotation();
    // Scan accessors must not be called from the worker threads: Scan::get_oscillation()
    // copies a scitbx::af::shared out of the scan's properties table, and scitbx's
    // non-atomic reference counting then races (heap-use-after-free, found with ASan).
    // Read the values once here and reproduce dxtbx's conversions exactly.
    const vec2<double> osc = scan.get_oscillation();
    const int image0 = scan.get_image_range()[0];
    auto angle_from_array_index = [osc, image0](int array_index) {
        double index = static_cast<double>(array_index) + 1;  // Scan::get_angle_from_array_index
        return osc[0] + (index - image0) * osc[1];
    };
    auto array_index_from_angle = [osc, image0](double angle) {
        return (image0 + (angle - osc[0]) / osc[1]) - 1;  // Scan::get_array_index_from_angle
    };
    ScanVaryingRayPredictor predict_rays(s0, m2, array_range[0], osc, dmin);

    cctbx::sgtbx::space_group sg_base = space_group_type.group();
    std::vector<ThreadPredVecs> frame_preds(static_cast<std::size_t>(n_frames));
    parallel_for(n_frames, [&](std::int64_t f_idx) {
        int frame = z0 + static_cast<int>(f_idx);
        int i = frame - offset;
        if (i < 0) i = 0;
        if (i >= static_cast<int>(A.size()) - 1) i = static_cast<int>(A.size()) - 2;

        double phi_beg = angle_from_array_index(frame);
        double phi_end = angle_from_array_index(frame + 1);
        mat3<double> r_beg = axis_and_angle_as_matrix(m2, phi_beg);
        mat3<double> r_end = axis_and_angle_as_matrix(m2, phi_end);
        mat3<double> A1 = r_setting * r_beg * r_fixed * A[i];
        mat3<double> A2 = r_setting * r_end * r_fixed * A[i + 1];

        auto& out = frame_preds[static_cast<std::size_t>(f_idx)];
        ReekeModel model(A1, A2, m2, -s0, -s0, dmin, static_cast<int>(margin));
        mat3<std::size_t> perm = model.permutation();
        vec2<int> p_lim = model.p_limits();
        for (int p0 = p_lim[0]; p0 < p_lim[1]; ++p0) {
            vec2<int> q_lim = model.q_limits(p0);
            for (int q0 = q_lim[0]; q0 < q_lim[1]; ++q0) {
                dials::af::small<vec2<int>, 2> r_lim = model.r_limits(p0, q0);
                for (std::size_t ridx = 0; ridx < r_lim.size(); ++ridx) {
                    for (int r0 = r_lim[ridx][0]; r0 < r_lim[ridx][1]; ++r0) {
                        cctbx::miller::index<> pqr(p0, q0, r0);
                        if (pqr.is_zero()) continue;
                        cctbx::miller::index<> h = perm * pqr;
                        if (h.is_zero() || sg_base.is_sys_absent(h)) continue;
                        boost::optional<Ray> ray = predict_rays(h, A1, A2, frame, 1);
                        if (ray) {
                            auto impact = detector.try_get_ray_intersection(ray->s1);
                            if (!impact) continue;
                            std::size_t panel_idx = impact->first;
                            vec2<double> mm = impact->second;
                            vec2<double> px = detector[panel_idx].millimeter_to_pixel(mm);
                            double f_obs = array_index_from_angle(ray->angle);
                            out.hkl.push_back(h);
                            out.enter.push_back(ray->entering);
                            out.s1.push_back(ray->s1);
                            out.xyz_mm.push_back(vec3<double>(mm[0], mm[1], ray->angle));
                            out.xyz_px.push_back(vec3<double>(px[0], px[1], f_obs));
                            out.panel.push_back(panel_idx);
                            out.flags.push_back(dials::af::Predicted);
                        }
                    }
                }
            }
        }
    });

    std::size_t total = 0;
    for (std::int64_t f_idx = 0; f_idx < n_frames; ++f_idx) {
        total += frame_preds[static_cast<std::size_t>(f_idx)].hkl.size();
    }

    dials::af::shared<cctbx::miller::index<>> hkl_col(total, dials::af::init_functor_null<cctbx::miller::index<>>());
    dials::af::shared<std::size_t> panel_col(total, dials::af::init_functor_null<std::size_t>());
    dials::af::shared<bool> enter_col(total, dials::af::init_functor_null<bool>());
    dials::af::shared<vec3<double>> s1_col(total, dials::af::init_functor_null<vec3<double>>());
    dials::af::shared<vec3<double>> xyz_px_col(total, dials::af::init_functor_null<vec3<double>>());
    dials::af::shared<vec3<double>> xyz_mm_col(total, dials::af::init_functor_null<vec3<double>>());
    dials::af::shared<std::size_t> flags_col(total, dials::af::init_functor_null<std::size_t>());
    dials::af::shared<int> id_col(total, 0);

    std::size_t pos = 0;
    for (std::int64_t f_idx = 0; f_idx < n_frames; ++f_idx) {
        const auto& fp = frame_preds[static_cast<std::size_t>(f_idx)];
        std::size_t m = fp.hkl.size();
        for (std::size_t k = 0; k < m; ++k) {
            hkl_col[pos] = fp.hkl[k];
            panel_col[pos] = fp.panel[k];
            enter_col[pos] = fp.enter[k];
            s1_col[pos] = fp.s1[k];
            xyz_px_col[pos] = fp.xyz_px[k];
            xyz_mm_col[pos] = fp.xyz_mm[k];
            flags_col[pos] = fp.flags[k];
            pos++;
        }
    }
    return boost::python::make_tuple(
        hkl_col, panel_col, enter_col, s1_col, xyz_px_col, xyz_mm_col, flags_col, id_col
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
    def("compute_glm_constant3d_fast", &dials_gpu::compute_glm_constant3d_fast);
    def("compute_glm_constant2d_fast", &dials_gpu::compute_glm_constant2d_fast);
    def("unpack_constant3d_bg_and_mask", &dials_gpu::unpack_constant3d_bg_and_mask);
    def("extract_simple_constant3d_bg", &dials_gpu::extract_simple_constant3d_bg);
    def("get_shoebox_shapes", &dials_gpu::get_shoebox_shapes);
    def("pack_bucket_3d", &dials_gpu::pack_bucket_3d);
    def("unpack_bucket_3d", &dials_gpu::unpack_bucket_3d);
    def("count_foreground_pixels", &dials_gpu::count_foreground_pixels);
    def("fill_foreground_pixels", &dials_gpu::fill_foreground_pixels);
    def("extract_centroid_moments", &dials_gpu::extract_centroid_moments);
    def("compute_centroid_fast", &dials_gpu::compute_centroid_fast);
    def("extract_summation_stats", &dials_gpu::extract_summation_stats);
    def("fill_summation_fg_pixels", &dials_gpu::fill_summation_fg_pixels);
    def("extract_summation_totals", &dials_gpu::extract_summation_totals);
    def("compute_summation_fast", &dials_gpu::compute_summation_fast);
    def("extract_shoeboxes_from_volume", &dials_gpu::extract_shoeboxes_from_volume);
    def("extract_shoeboxes_from_volume_typed", &dials_gpu::extract_shoeboxes_from_volume_typed);
    def("extract_wide_shoeboxes_slice_from_volume", &dials_gpu::extract_wide_shoeboxes_slice_from_volume);
    def("extract_wide_shoeboxes_slice_from_volume_typed", &dials_gpu::extract_wide_shoeboxes_slice_from_volume_typed);
    def("integrate_slab_from_volume_fast", &dials_gpu::integrate_slab_from_volume_fast);
    def("parallel_compute_mask", &dials_gpu::parallel_compute_mask);
    def("parallel_compute_mask_vec3", &dials_gpu::parallel_compute_mask_vec3);
    def("parallel_compute_mask_3d_fast", &dials_gpu::parallel_compute_mask_3d_fast);
    def("parallel_compute_bbox_3d_fast", &dials_gpu::parallel_compute_bbox_3d_fast);
    def("parallel_pre_background_mask_flags", &dials_gpu::parallel_pre_background_mask_flags);
    def("parallel_post_mask_counts", &dials_gpu::parallel_post_mask_counts);
    def("parallel_deallocate_shoeboxes", &dials_gpu::parallel_deallocate_shoeboxes);
    def("apply_pedestal_gain_shoeboxes", &dials_gpu::apply_pedestal_gain_shoeboxes);
    def("parallel_check_overloads_single_panel", &dials_gpu::parallel_check_overloads_single_panel);
    def("parallel_fit_multi", &dials_gpu::parallel_fit_multi);
    def("parallel_model_multi", &dials_gpu::parallel_model_multi);
    def("create_transform_cache", &dials_gpu::create_transform_cache);
    def("free_transform_cache", &dials_gpu::free_transform_cache);
    def("parallel_transform_and_model_slab", &dials_gpu::parallel_transform_and_model_slab);
    def("parallel_fit_from_cache", &dials_gpu::parallel_fit_from_cache);
    def("parallel_model_and_fit_circular_slab", &dials_gpu::parallel_model_and_fit_circular_slab);
    def("mark_modeller_finalized", &dials_gpu::mark_modeller_finalized);
    def("compute_beam_divergence_variances", &dials_gpu::compute_beam_divergence_variances);
    def("count_extended_sigma_m_frames", &dials_gpu::count_extended_sigma_m_frames);
    def("fill_extended_sigma_m_frames", &dials_gpu::fill_extended_sigma_m_frames);
    def("parallel_mask_neighbouring", &dials_gpu::parallel_mask_neighbouring);
    def("match_with_reference_indices", &dials_gpu::match_with_reference_indices);
    def("parallel_predict_scan_varying", &dials_gpu::parallel_predict_scan_varying);
}
