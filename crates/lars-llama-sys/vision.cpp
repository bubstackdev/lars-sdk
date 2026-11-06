// Narrow, exception-contained bridge to the pinned mtmd API. Each instance belongs to
// the model worker. Encoded inputs are decoded once per request, never on the caller thread.
#include "mtmd.h"
#include "mtmd-helper.h"
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <vector>
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb/stb_image.h"

static constexpr size_t MAX_BYTES = 20 * 1024 * 1024;
static constexpr size_t MAX_PIXELS = 40000000;
static constexpr const char * MARKER = "\x1e" "LARS_IMAGE" "\x1e";
static std::once_flag logging_initialized;
static void quiet_log(ggml_log_level, const char *, void *) {}
struct lars_vision {
    mtmd_context * ctx = nullptr;
    mtmd_bitmap * bitmap = nullptr;
    std::string error;
    ~lars_vision() { if (bitmap) mtmd_bitmap_free(bitmap); if (ctx) mtmd_free(ctx); }
};
using chunks_ptr = std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;

extern "C" {
void * lars_vision_open(const char * path, const llama_model * model, bool gpu, int threads, int max_tokens) noexcept {
    try {
        std::call_once(logging_initialized, [] { mtmd_helper_log_set(quiet_log, nullptr); });
        auto v = std::make_unique<lars_vision>();
        auto params = mtmd_context_params_default();
        params.use_gpu = gpu; params.n_threads = threads; params.warmup = false;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.image_max_tokens = max_tokens;
        v->ctx = mtmd_init_from_file(path, model, params);
        if (!v->ctx || !mtmd_support_vision(v->ctx)) return nullptr;
        return v.release();
    } catch (...) { return nullptr; }
}
void lars_vision_close(void * ptr) noexcept { delete static_cast<lars_vision *>(ptr); }
const char * lars_vision_error(const void * ptr) noexcept { return static_cast<const lars_vision *>(ptr)->error.c_str(); }
void lars_vision_clear(void * ptr) noexcept {
    auto & v = *static_cast<lars_vision *>(ptr);
    if (v.bitmap) mtmd_bitmap_free(v.bitmap);
    v.bitmap = nullptr;
}
int lars_vision_set_image(void * ptr, const unsigned char * data, size_t size) noexcept {
    auto & v = *static_cast<lars_vision *>(ptr);
    lars_vision_clear(ptr);
    try {
        if (!data || size == 0 || size > MAX_BYTES) throw std::runtime_error("image must contain 1 byte to 20 MiB");
        const bool png = size >= 8 && std::memcmp(data, "\x89PNG\r\n\x1a\n", 8) == 0;
        const bool jpeg = size >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff;
        if (!png && !jpeg) throw std::runtime_error("only JPEG and PNG images are supported");
        // Reject APNG before decoding its first frame as though it were a still image.
        if (png) for (size_t at = 8; at + 12 <= size;) {
            size_t n = (size_t(data[at]) << 24) | (size_t(data[at+1]) << 16) | (size_t(data[at+2]) << 8) | data[at+3];
            if (n > size - at - 12) throw std::runtime_error("truncated PNG chunk");
            if (std::memcmp(data + at + 4, "acTL", 4) == 0) throw std::runtime_error("animated PNG is not supported");
            at += n + 12;
        }
        int w, h, channels;
        if (!stbi_info_from_memory(data, int(size), &w, &h, &channels) || w <= 0 || h <= 0 || size_t(w) * size_t(h) > MAX_PIXELS)
            throw std::runtime_error("invalid image or image exceeds 40 million pixels");
        std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(
            stbi_load_from_memory(data, int(size), &w, &h, &channels, 4), stbi_image_free);
        if (!pixels) throw std::runtime_error("could not decode image");
        std::vector<unsigned char> rgb(size_t(w) * h * 3);
        for (size_t i = 0; i < size_t(w) * h; ++i) for (size_t c = 0; c < 3; ++c) {
            unsigned a = pixels.get()[4*i+3];
            rgb[3*i+c] = (pixels.get()[4*i+c] * a + 255 * (255-a) + 127) / 255;
        }
        v.bitmap = mtmd_bitmap_init(w, h, rgb.data());
        if (!v.bitmap) throw std::runtime_error("could not create image bitmap");
        return 0;
    } catch (const std::exception & e) { v.error = e.what(); return -1; }
      catch (...) { v.error = "image preparation failed"; return -1; }
}
}

static chunks_ptr tokenize(lars_vision & v, const char * prompt, size_t size) {
    std::string text(prompt, size);
    auto at = text.find(MARKER);
    if (!v.bitmap || at == std::string::npos || text.find(MARKER, at + std::strlen(MARKER)) != std::string::npos)
        throw std::runtime_error("image prompt must have exactly one internal image placeholder");
    std::string before = text.substr(0, at), after = text.substr(at + std::strlen(MARKER));
    mtmd_input_text a{before.data(), before.size(), false, true};
    mtmd_input_text b{after.data(), after.size(), false, true};
    mtmd_input_part pa{&a, nullptr}, image{nullptr, v.bitmap}, pb{&b, nullptr};
    const mtmd_input_part * parts[]{&pa, &image, &pb};
    chunks_ptr chunks(mtmd_input_chunks_init(), mtmd_input_chunks_free);
    if (!chunks || mtmd_tokenize_from_parts(v.ctx, chunks.get(), parts, 3, true))
        throw std::runtime_error("multimodal tokenization failed");
    return chunks;
}
extern "C" {
int lars_vision_count(void * ptr, const char * text, size_t size, size_t * count) noexcept {
    auto & v = *static_cast<lars_vision *>(ptr);
    try { auto chunks = tokenize(v, text, size); *count = mtmd_helper_get_n_tokens(chunks.get()); return 0; }
    catch (const std::exception & e) { v.error = e.what(); return -1; }
    catch (...) { v.error = "image token counting failed"; return -1; }
}
int lars_vision_score(void * ptr, llama_context * ctx, const char * text, size_t size, int batch) noexcept {
    auto & v = *static_cast<lars_vision *>(ptr);
    try {
        auto chunks = tokenize(v, text, size);
        if (mtmd_helper_get_n_tokens(chunks.get()) > llama_n_ctx(ctx)) throw std::runtime_error("image prompt exceeds model context capacity");
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_pos next = 0;
        if (mtmd_helper_eval_chunks(v.ctx, ctx, chunks.get(), 0, 0, batch, true, &next))
            throw std::runtime_error("multimodal evaluation failed");
        return 0;
    } catch (const std::exception & e) { v.error = e.what(); return -1; }
      catch (...) { v.error = "image scoring failed"; return -1; }
}
}
