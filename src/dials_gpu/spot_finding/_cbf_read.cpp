#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <thread>
#include <atomic>
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
