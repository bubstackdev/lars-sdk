#include "session.hpp"
#include "model.h"
#include "metal.h"
#include "llama.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <optional>
#include <sys/sysctl.h>
#include <thread>

namespace teacher {
class VocabEngine : public Engine {
protected:
    struct FreeModel { void operator()(llama_model *p) { if (p) llama_model_free(p); } };
    std::unique_ptr<llama_model, FreeModel> vocab_model_;
    const llama_vocab *vocab_;
    std::string path_, architecture_;
public:
    explicit VocabEngine(const std::string &path) : path_(path) {
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true; mp.n_gpu_layers = 0; mp.load_mode = LLAMA_LOAD_MODE_NONE;
        // Vocab-only loading needs no backend registration and never loads weights.
        ggml_backend_dev_t no_devices[] = {nullptr}; mp.devices = no_devices;
        vocab_model_.reset(llama_model_load_from_file(path.c_str(), mp));
        if (!vocab_model_) throw std::runtime_error("CPU vocabulary load failed");
        vocab_ = llama_model_get_vocab(vocab_model_.get());
        char value[256] = {};
        if (llama_model_meta_val_str(vocab_model_.get(), "general.architecture", value, sizeof(value)) < 0)
            throw std::runtime_error("missing GGUF architecture");
        architecture_ = value;
    }
    json info() const override {
        return {{"mode", "tokenizer-only"}, {"synthetic", false}, {"gpu_loaded", false},
                {"model", path_}, {"architecture", architecture_}, {"vocab_size", llama_vocab_n_tokens(vocab_)}};
    }
    bool can_score() const override { return false; }
    std::vector<int32_t> tokenize(const std::string &text, bool special) override {
        std::vector<int32_t> result(text.size() + 16);
        int n = llama_tokenize(vocab_, text.data(), (int)text.size(), result.data(), (int)result.size(), false, special);
        if (n < 0) {
            result.resize(-n);
            n = llama_tokenize(vocab_, text.data(), (int)text.size(), result.data(), (int)result.size(), false, special);
        }
        if (n < 0) throw std::runtime_error("tokenization failed");
        result.resize(n); return result;
    }
    void reset() override {}
    void drop_snapshot() override {}
    void append(const std::vector<int32_t> &) override { throw std::runtime_error("GPU not enabled"); }
    void save() override { throw std::runtime_error("GPU not enabled"); }
    void restore() override { throw std::runtime_error("GPU not enabled"); }
    std::vector<double> score(const std::vector<int32_t> &) override { throw std::runtime_error("GPU not enabled"); }
};

// This watchdog can exit only its OWN process. It never signals another job.
// Start before allocating any model buffers; remain enabled during load/dispatch.
class PressureGuard {
    std::atomic<bool> stop_{false};
    std::thread thread_;
    static bool normal() {
        int level = 0; size_t size = sizeof(level);
        return sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size, nullptr, 0) == 0 && level == 1;
    }
public:
    PressureGuard() {
        if (!normal()) throw Fatal("memory guard: pressure is not normal or cannot be read");
        thread_ = std::thread([this] {
            while (!stop_.load()) {
                if (!normal()) {
                    std::fprintf(stderr, "memory guard: exiting own Flash teacher worker\n");
                    std::_Exit(75);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
        });
    }
    ~PressureGuard() { stop_ = true; if (thread_.joinable()) thread_.join(); }
};

class FlashEngine : public VocabEngine {
    PressureGuard guard_; // Constructed before Metal/model; destroyed after them.
    flash::Metal metal_;
    std::unique_ptr<flash::Model> model_;
    std::optional<flash::Model::Snapshot> snapshot_;
    std::vector<float> last_, prefix_logits_;
public:
    FlashEngine(const std::string &path, int context) : VocabEngine(path) {
        if (architecture_ != "qwen4exp") throw std::invalid_argument("Flash worker requires qwen4exp GGUF");
        metal_.load_kernels();
        model_ = std::make_unique<flash::Model>(metal_, path, context, 16, true);
        if (model_->hparams().vocab != llama_vocab_n_tokens(vocab_)) throw std::runtime_error("model/tokenizer vocab mismatch");
    }
    json info() const override {
        auto value = VocabEngine::info(); value["mode"] = "flash"; value["gpu_loaded"] = true;
        value["memory_guard"] = true; value["snapshot_slots"] = 1; return value;
    }
    bool can_score() const override { return true; }
    void reset() override { model_->reset(); last_.clear(); }
    void drop_snapshot() override { snapshot_.reset(); prefix_logits_.clear(); }
    void append(const std::vector<int32_t> &ids) override {
        // Score only the trunk at the last position; no sampling or speculative MTP.
        for (size_t i = 0; i < ids.size(); i += model_->ubatch()) {
            const int n = (int)std::min(ids.size() - i, (size_t)model_->ubatch());
            const float *p = model_->forward(ids.data() + i, n, true, true);
            last_.assign(p, p + model_->hparams().vocab);
        }
    }
    void save() override { snapshot_.emplace(model_->snapshot()); prefix_logits_ = last_; }
    void restore() override {
        if (!snapshot_) throw std::runtime_error("missing prefix snapshot");
        model_->restore(*snapshot_); last_ = prefix_logits_;
    }
    std::vector<double> score(const std::vector<int32_t> &ids) override {
        if (last_.empty()) throw std::runtime_error("no evaluated prompt");
        std::vector<double> z;
        for (int32_t id : ids) {
            if (id < 0 || id >= (int32_t)last_.size()) throw std::runtime_error("invalid label token id");
            z.push_back(last_[id]);
        }
        return z;
    }
    int32_t greedy() override {
        if (last_.empty() || std::any_of(last_.begin(), last_.end(), [](float v) { return !std::isfinite(v); }))
            throw std::runtime_error("invalid generation logits");
        return (int32_t)(std::max_element(last_.begin(), last_.end()) - last_.begin());
    }
    bool eog(int32_t token) const override { return llama_vocab_is_eog(vocab_, token); }
    std::string piece(int32_t token) const override {
        std::vector<char> bytes(256);
        int n = llama_token_to_piece(vocab_, token, bytes.data(), (int)bytes.size(), 0, true);
        if (n < 0) {
            bytes.resize(-n);
            n = llama_token_to_piece(vocab_, token, bytes.data(), (int)bytes.size(), 0, true);
        }
        if (n < 0) throw std::runtime_error("token decoding failed");
        return std::string(bytes.data(), n);
    }
};
std::unique_ptr<Engine> runtime_engine(const std::string &model, int context, bool gpu) {
    if (gpu) return std::make_unique<FlashEngine>(model, context);
    return std::make_unique<VocabEngine>(model);
}
} // namespace teacher
