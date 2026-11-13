#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <libdeflate.h>
#include <zlib.h>

static int decode_cbf_byte_offset(
    const uint8_t* __restrict__ src,
    size_t src_len,
    int32_t* __restrict__ dst,
    size_t n_elems
) {
    size_t search_limit = src_len > 16384 ? 16384 : src_len;
    const uint8_t magic[4] = {0x0c, 0x1a, 0x04, 0xd5};
    const void* found_ptr = memmem(src, search_limit, magic, 4);
    if (!found_ptr && src_len > search_limit) {
        found_ptr = memmem(src, src_len, magic, 4);
    }
    if (!found_ptr) return -1;
    size_t pos = static_cast<const uint8_t*>(found_ptr) - src + 4;

    if (memmem(src, pos, "x-CBF_BYTE_OFFSET", 17) == nullptr) {
        return -2;
    }
    if (memmem(src, pos, "signed 32-bit integer", 21) == nullptr) {
        return -2;
    }
    if (memmem(src, pos, "LITTLE_ENDIAN", 13) == nullptr) {
        return -2;
    }
    const char* n_elem_hdr = static_cast<const char*>(
        memmem(src, pos, "X-Binary-Number-of-Elements:", 28)
    );
    if (!n_elem_hdr) return -2;
    long long hdr_elems = strtoll(n_elem_hdr + 28, nullptr, 10);
    if (hdr_elems != static_cast<long long>(n_elems)) {
        return -2;
    }

    int32_t current = 0;
    size_t out_idx = 0;
    while (out_idx < n_elems && pos < src_len) {
        while (out_idx + 8 <= n_elems && pos + 8 <= src_len) {
            int8_t b0 = static_cast<int8_t>(src[pos]);
            int8_t b1 = static_cast<int8_t>(src[pos + 1]);
            int8_t b2 = static_cast<int8_t>(src[pos + 2]);
            int8_t b3 = static_cast<int8_t>(src[pos + 3]);
            int8_t b4 = static_cast<int8_t>(src[pos + 4]);
            int8_t b5 = static_cast<int8_t>(src[pos + 5]);
            int8_t b6 = static_cast<int8_t>(src[pos + 6]);
            int8_t b7 = static_cast<int8_t>(src[pos + 7]);
            if (__builtin_expect(
                    (b0 == -128) | (b1 == -128) | (b2 == -128) | (b3 == -128) |
                    (b4 == -128) | (b5 == -128) | (b6 == -128) | (b7 == -128),
                    0)) {
                break;
            }
            current += b0; dst[out_idx]     = current;
            current += b1; dst[out_idx + 1] = current;
            current += b2; dst[out_idx + 2] = current;
            current += b3; dst[out_idx + 3] = current;
            current += b4; dst[out_idx + 4] = current;
            current += b5; dst[out_idx + 5] = current;
            current += b6; dst[out_idx + 6] = current;
            current += b7; dst[out_idx + 7] = current;
            pos += 8;
            out_idx += 8;
        }
        if (out_idx >= n_elems || pos >= src_len) break;
        int8_t b = static_cast<int8_t>(src[pos++]);
        if (__builtin_expect(b != -128, 1)) {
            current += b;
            dst[out_idx++] = current;
        } else {
            if (pos + 2 > src_len) return -3;
            int16_t s;
            memcpy(&s, src + pos, 2);
            pos += 2;
            if (__builtin_expect(s != -32768, 1)) {
                current += s;
                dst[out_idx++] = current;
            } else {
                if (pos + 4 > src_len) return -4;
                int32_t l;
                memcpy(&l, src + pos, 4);
                pos += 4;
                if (l != -2147483648LL) {
                    current += l;
                    dst[out_idx++] = current;
                } else {
                    if (pos + 8 > src_len) return -5;
                    int64_t ll;
                    memcpy(&ll, src + pos, 8);
                    pos += 8;
                    current = static_cast<int32_t>(static_cast<int64_t>(current) + ll);
                    dst[out_idx++] = current;
                }
            }
        }
    }
    return (out_idx == n_elems) ? 0 : -6;
}

extern "C" int read_cbf_batch_omp(
    const char** paths,
    int n_files,
    int64_t n_elems,
    int32_t* out_vol,
    int n_threads
) {
    if (n_files <= 0) return 0;
    int actual_threads = n_threads < n_files ? n_threads : n_files;
    if (actual_threads < 1) actual_threads = 1;

    std::atomic<int> next_idx{0};
    std::atomic<int> err{0};

    auto worker = [&]() {
        std::vector<uint8_t> comp_buf;
        std::vector<uint8_t> decomp_buf(static_cast<size_t>(n_elems) * 2 + 65536);
        struct libdeflate_decompressor* d = libdeflate_alloc_decompressor();

        while (true) {
            int i = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (i >= n_files) break;
            if (err.load(std::memory_order_relaxed) != 0) continue;

            const char* p = paths[i];
            size_t len = strlen(p);
            size_t total_read = 0;
            if (len > 7 && strcmp(p + len - 7, ".cbf.gz") == 0) {
                FILE* fp = fopen(p, "rb");
                if (!fp) { err.fetch_or(1, std::memory_order_relaxed); continue; }
                fseek(fp, 0, SEEK_END);
                long fsize = ftell(fp);
                fseek(fp, 0, SEEK_SET);
                if (fsize < 18) { fclose(fp); err.fetch_or(2, std::memory_order_relaxed); continue; }
                if (static_cast<size_t>(fsize) > comp_buf.size()) {
                    comp_buf.resize(fsize);
                }
                size_t nr = fread(comp_buf.data(), 1, fsize, fp);
                fclose(fp);
                if (nr != static_cast<size_t>(fsize)) { err.fetch_or(2, std::memory_order_relaxed); continue; }
                uint32_t isize = 0;
                memcpy(&isize, comp_buf.data() + fsize - 4, 4);
                if (isize > decomp_buf.size() && isize < static_cast<size_t>(n_elems) * 8) {
                    decomp_buf.resize(isize + 4096);
                }
                size_t actual_out = 0;
                enum libdeflate_result res = libdeflate_gzip_decompress(
                    d, comp_buf.data(), fsize, decomp_buf.data(), decomp_buf.size(), &actual_out
                );
                if (res == LIBDEFLATE_SUCCESS) {
                    total_read = actual_out;
                } else {
                    gzFile gz = gzopen(p, "rb");
                    if (!gz) { err.fetch_or(1, std::memory_order_relaxed); continue; }
                    while (true) {
                        if (total_read + 262144 > decomp_buf.size()) {
                            decomp_buf.resize(decomp_buf.size() * 2);
                        }
                        int r = gzread(gz, decomp_buf.data() + total_read, 262144);
                        if (r < 0) { err.fetch_or(2, std::memory_order_relaxed); break; }
                        if (r == 0) break;
                        total_read += r;
                    }
                    gzclose(gz);
                }
            } else if (len > 4 && strcmp(p + len - 4, ".cbf") == 0) {
                FILE* fp = fopen(p, "rb");
                if (!fp) { err.fetch_or(1, std::memory_order_relaxed); continue; }
                fseek(fp, 0, SEEK_END);
                long fsize = ftell(fp);
                fseek(fp, 0, SEEK_SET);
                if (fsize <= 0) { fclose(fp); err.fetch_or(2, std::memory_order_relaxed); continue; }
                if (static_cast<size_t>(fsize) > decomp_buf.size()) {
                    decomp_buf.resize(fsize);
                }
                total_read = fread(decomp_buf.data(), 1, fsize, fp);
                fclose(fp);
            } else {
                FILE* fp = fopen(p, "rb");
                if (!fp) { err.fetch_or(1, std::memory_order_relaxed); continue; }
                uint8_t hdr[4096];
                if (fread(hdr, 1, 4096, fp) != 4096) {
                    fclose(fp);
                    err.fetch_or(8, std::memory_order_relaxed);
                    continue;
                }
                uint32_t r_width = 0, r_height = 0, r_depth = 0;
                uint32_t r_nimg = 0, r_orig = 0, r_orient = 0, r_view = 0;
                int32_t r_bias_raw = 0;
                memcpy(&r_width, hdr + 1104, 4);
                memcpy(&r_height, hdr + 1108, 4);
                memcpy(&r_depth, hdr + 1112, 4);
                memcpy(&r_nimg, hdr + 1136, 4);
                memcpy(&r_orig, hdr + 1140, 4);
                memcpy(&r_orient, hdr + 1144, 4);
                memcpy(&r_view, hdr + 1148, 4);
                memcpy(&r_bias_raw, hdr + 1804, 4);
                if (hdr[0] == 'I' && hdr[1] == 'I' && hdr[2] == 42 && hdr[3] == 0 &&
                    r_width == r_height && (r_depth == 2 || r_depth == 16) &&
                    static_cast<int64_t>(r_width) * static_cast<int64_t>(r_height) == n_elems &&
                    r_nimg == 1 && r_orig == 0 && r_orient == 0 && r_view == 0) {
                    size_t raw_bytes = static_cast<size_t>(n_elems) * 2;
                    if (decomp_buf.size() < raw_bytes) {
                        decomp_buf.resize(raw_bytes);
                    }
                    size_t nr = fread(decomp_buf.data(), 1, raw_bytes, fp);
                    fclose(fp);
                    if (nr != raw_bytes) {
                        err.fetch_or(2, std::memory_order_relaxed);
                        continue;
                    }
                    int32_t bias = static_cast<int32_t>(
                        std::llround(static_cast<double>(r_bias_raw) * 1.0e-3)
                    );
                    const uint16_t* src16 = reinterpret_cast<const uint16_t*>(decomp_buf.data());
                    int32_t* dst32 = out_vol + static_cast<int64_t>(i) * n_elems;
                    for (int64_t k = 0; k < n_elems; ++k) {
                        dst32[k] = static_cast<int32_t>(src16[k]) - bias;
                    }
                    continue;
                }
                fclose(fp);
                err.fetch_or(8, std::memory_order_relaxed);
                continue;
            }
            if (err.load(std::memory_order_relaxed) == 0) {
                int rc = decode_cbf_byte_offset(
                    decomp_buf.data(),
                    total_read,
                    out_vol + static_cast<int64_t>(i) * n_elems,
                    static_cast<size_t>(n_elems)
                );
                if (rc != 0) err.fetch_or(4, std::memory_order_relaxed);
            }
        }
        libdeflate_free_decompressor(d);
    };

    std::vector<std::thread> threads;
    threads.reserve(actual_threads - 1);
    for (int t = 0; t < actual_threads - 1; ++t) {
        threads.emplace_back(worker);
    }
    worker();
    for (auto& th : threads) {
        th.join();
    }
    return err.load(std::memory_order_relaxed);
}

typedef int64_t (*bshuf_fn_t)(
    const void* in, void* out, size_t size, size_t elem_size, size_t block_size
);

static inline uint32_t read_be32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) |
           static_cast<uint32_t>(p[3]);
}

static inline int decode_bshuf_frame_tiled_i32(
    bshuf_fn_t bshuf_fn,
    const uint8_t* cptr,
    size_t clen,
    size_t n_elems,
    int dtype_code, // 1=uint16, 2=int16, 3=uint32, 4=int32
    int64_t invalid_top,
    int32_t* dst_frame,
    std::vector<uint8_t>& tile_buf
) {
    if (!cptr || clen < 12) return -1;
    size_t elem_size = (dtype_code <= 2) ? 2 : 4;
    uint32_t block_bytes = read_be32(cptr + 8);
    size_t block_elems = block_bytes / elem_size;
    if (block_elems == 0) block_elems = 4096;
    size_t tile_blocks = std::max<size_t>(1, (65536 + block_elems - 1) / block_elems);
    size_t tile_elems = tile_blocks * block_elems;
    if (elem_size == 2 && tile_buf.size() < tile_elems * 2) {
        tile_buf.resize(tile_elems * 2);
    }

    const uint8_t* in_cur = cptr + 12;
    size_t done = 0;
    while (done < n_elems) {
        size_t rem = n_elems - done;
        size_t cur = (rem > tile_elems + block_elems) ? tile_elems : rem;
        int32_t* dst_tile = dst_frame + done;
        if (elem_size == 4) {
            int64_t consumed = bshuf_fn(in_cur, dst_tile, cur, 4, block_elems);
            if (consumed < 0) return -2;
            in_cur += consumed;
            if (invalid_top > 0) {
                int32_t top_m1 = static_cast<int32_t>(invalid_top - 1);
                int32_t top_m2 = static_cast<int32_t>(invalid_top - 2);
                for (size_t k = 0; k < cur; ++k) {
                    int32_t v = dst_tile[k];
                    if (__builtin_expect(v >= top_m2, 0)) {
                        if (v == top_m1) dst_tile[k] = -1;
                        else if (v == top_m2) dst_tile[k] = -2;
                    }
                }
            }
        } else {
            int64_t consumed = bshuf_fn(in_cur, tile_buf.data(), cur, 2, block_elems);
            if (consumed < 0) return -2;
            in_cur += consumed;
            if (dtype_code == 1) {
                const uint16_t* s = reinterpret_cast<const uint16_t*>(tile_buf.data());
                if (invalid_top > 0 && invalid_top <= 65536) {
                    int32_t top_i = static_cast<int32_t>(invalid_top);
                    int32_t top_m2 = top_i - 2;
                    for (size_t k = 0; k < cur; ++k) {
                        int32_t v = static_cast<int32_t>(s[k]);
                        dst_tile[k] = (v >= top_m2) ? (v - top_i) : v;
                    }
                } else {
                    for (size_t k = 0; k < cur; ++k) {
                        dst_tile[k] = static_cast<int32_t>(s[k]);
                    }
                }
            } else {
                const int16_t* s = reinterpret_cast<const int16_t*>(tile_buf.data());
                if (invalid_top > 0 && invalid_top <= 65536) {
                    int32_t top_i = static_cast<int32_t>(invalid_top);
                    int32_t top_m2 = top_i - 2;
                    for (size_t k = 0; k < cur; ++k) {
                        int32_t v = static_cast<int32_t>(s[k]);
                        dst_tile[k] = (v >= top_m2) ? (v - top_i) : v;
                    }
                } else {
                    for (size_t k = 0; k < cur; ++k) {
                        dst_tile[k] = static_cast<int32_t>(s[k]);
                    }
                }
            }
        }
        done += cur;
    }
    return 0;
}

static std::mutex g_bshuf_scratch_mtx;
static std::vector<std::vector<uint8_t>> g_bshuf_comp_bufs(64);
static std::vector<std::vector<uint8_t>> g_bshuf_tile_bufs(64);

extern "C" int bshuf_decode_batch_i32(
    bshuf_fn_t bshuf_fn,
    const uint8_t** chunk_ptrs,
    const size_t* chunk_lens,
    int n_frames,
    size_t n_elems,
    int dtype_code, // 1=uint16, 2=int16, 3=uint32, 4=int32
    int64_t invalid_top,
    int32_t* out_vol,
    int n_threads
) {
    if (n_frames <= 0) return 0;
    std::lock_guard<std::mutex> lock(g_bshuf_scratch_mtx);
    int actual_threads = std::min(n_frames, std::min(32, std::max(1, n_threads)));
    std::atomic<int> next_idx{0};
    std::atomic<int> err{0};

    auto worker = [&](int tid) {
        auto& tile_buf = g_bshuf_tile_bufs[tid];
        while (true) {
            int i = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (i >= n_frames) break;
            if (err.load(std::memory_order_relaxed) != 0) continue;
            int rc = decode_bshuf_frame_tiled_i32(
                bshuf_fn,
                chunk_ptrs[i],
                chunk_lens[i],
                n_elems,
                dtype_code,
                invalid_top,
                out_vol + static_cast<size_t>(i) * n_elems,
                tile_buf
            );
            if (rc != 0) err.fetch_or(2, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(actual_threads - 1);
    for (int t = 1; t < actual_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    worker(0);
    for (auto& th : threads) {
        th.join();
    }
    return err.load(std::memory_order_relaxed);
}

extern "C" int bshuf_pread_direct_batch(
    bshuf_fn_t bshuf_fn,
    const int* fds,
    const int64_t* offsets,
    const size_t* sizes,
    int n_frames,
    size_t n_elems,
    size_t elem_size,
    uint8_t* out_vol,
    int n_threads
) {
    if (n_frames <= 0) return 0;
    std::lock_guard<std::mutex> lock(g_bshuf_scratch_mtx);
    int actual_threads = std::min(n_frames, std::min(32, std::max(1, n_threads)));
    std::atomic<int> next_idx{0};
    std::atomic<int> err{0};

    auto worker = [&](int tid) {
        auto& comp_buf = g_bshuf_comp_bufs[tid];
        while (true) {
            int i = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (i >= n_frames) break;
            if (err.load(std::memory_order_relaxed) != 0) continue;
            size_t sz = sizes[i];
            if (sz < 12) { err.fetch_or(1, std::memory_order_relaxed); continue; }
            if (comp_buf.size() < sz) comp_buf.resize(sz + 65536);
            int fd = fds[i];
            int64_t off = offsets[i];
            size_t got = 0;
            while (got < sz) {
                ssize_t r = pread(fd, comp_buf.data() + got, sz - got, off + static_cast<int64_t>(got));
                if (r <= 0) { err.fetch_or(4, std::memory_order_relaxed); break; }
                got += static_cast<size_t>(r);
            }
            if (got < sz) continue;
            uint32_t block_bytes = read_be32(comp_buf.data() + 8);
            size_t block_elems = block_bytes / elem_size;
            uint8_t* dst = out_vol + static_cast<size_t>(i) * n_elems * elem_size;
            int64_t res = bshuf_fn(comp_buf.data() + 12, dst, n_elems, elem_size, block_elems);
            if (res < 0) err.fetch_or(2, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(actual_threads - 1);
    for (int t = 1; t < actual_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    worker(0);
    for (auto& th : threads) {
        th.join();
    }
    return err.load(std::memory_order_relaxed);
}

extern "C" int bshuf_decode_direct_batch(
    bshuf_fn_t bshuf_fn,
    const uint8_t** chunk_ptrs,
    const size_t* chunk_lens,
    int n_frames,
    size_t n_elems,
    size_t elem_size,
    uint8_t* out_vol,
    int n_threads
) {
    if (n_frames <= 0) return 0;
    int actual_threads = std::min(n_frames, std::min(32, std::max(1, n_threads)));
    std::atomic<int> next_idx{0};
    std::atomic<int> err{0};

    auto worker = [&]() {
        while (true) {
            int i = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (i >= n_frames) break;
            if (err.load(std::memory_order_relaxed) != 0) continue;
            const uint8_t* cptr = chunk_ptrs[i];
            size_t clen = chunk_lens[i];
            if (!cptr || clen < 12) { err.fetch_or(1, std::memory_order_relaxed); continue; }
            uint32_t block_bytes = read_be32(cptr + 8);
            size_t block_elems = block_bytes / elem_size;
            uint8_t* dst = out_vol + static_cast<size_t>(i) * n_elems * elem_size;
            int64_t res = bshuf_fn(cptr + 12, dst, n_elems, elem_size, block_elems);
            if (res < 0) err.fetch_or(2, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(actual_threads - 1);
    for (int t = 1; t < actual_threads; ++t) {
        threads.emplace_back(worker);
    }
    worker();
    for (auto& th : threads) {
        th.join();
    }
    return err.load(std::memory_order_relaxed);
}
