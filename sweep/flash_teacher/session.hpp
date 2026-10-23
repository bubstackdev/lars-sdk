#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace teacher {
using json = nlohmann::json;
struct Fatal : std::runtime_error { using std::runtime_error::runtime_error; };
struct Engine {
    virtual ~Engine() = default;
    virtual json info() const = 0;
    virtual std::vector<int32_t> tokenize(const std::string &, bool special) = 0;
    virtual bool can_score() const = 0;
    virtual void reset() = 0;  // Active continuation only; snapshot has separate ownership.
    virtual void drop_snapshot() = 0;
    virtual void append(const std::vector<int32_t> &) = 0;
    virtual void save() = 0;
    virtual void restore() = 0;
    virtual std::vector<double> score(const std::vector<int32_t> &) = 0;
    virtual int32_t greedy() { throw std::runtime_error("generation unavailable"); }
    virtual bool eog(int32_t) const { return false; }
    virtual std::string piece(int32_t) const { throw std::runtime_error("decoding unavailable"); }
};

// CPU test double: deliberately merges AB across concatenation boundaries and has
// logits sensitive to the entire active token history. It is not a language model.
struct MockEngine : Engine {
    std::vector<int32_t> state, saved;
    size_t generated = 0;
    json info() const override { return {{"mode", "mock"}, {"synthetic", true}, {"gpu_loaded", false}}; }
    bool can_score() const override { return true; }
    std::vector<int32_t> tokenize(const std::string &s, bool) override {
        std::vector<int32_t> ids;
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == ' ' && i + 1 < s.size() && s[i + 1] >= 'A' && s[i + 1] <= 'Z') {
                ids.push_back(1000 + s[i + 1]); ++i;
            } else if (s[i] == 'A' && i + 1 < s.size() && s[i + 1] == 'B') { ids.push_back(500); ++i; }
            else ids.push_back((unsigned char)s[i]);
        }
        return ids;
    }
    void reset() override { state.clear(); generated = 0; }
    void drop_snapshot() override { saved.clear(); }
    void append(const std::vector<int32_t> &ids) override { state.insert(state.end(), ids.begin(), ids.end()); }
    void save() override { saved = state; }
    void restore() override { state = saved; }
    std::vector<double> score(const std::vector<int32_t> &ids) override {
        uint64_t hash = 1469598103934665603ULL;
        for (int32_t t : state) hash = (hash ^ (uint32_t)t) * 1099511628211ULL;
        std::vector<double> result;
        for (int32_t t : ids) result.push_back(double((hash ^ ((uint64_t)t * 2654435761ULL)) % 100003) / 10000.);
        return result;
    }
    int32_t greedy() override {
        const std::string text = "brief</think>\nFINAL: A";
        return generated < text.size() ? (unsigned char)text[generated++] : 100000;
    }
    bool eog(int32_t token) const override { return token == 100000; }
    std::string piece(int32_t token) const override { return std::string(1, (char)token); }
};

class Session {
    Engine &engine_;
    int context_;
    bool cached_ = false;
    std::vector<int32_t> prefix_;
public:
    Session(Engine &engine, int context) : engine_(engine), context_(context) {}
    void clear() { cached_ = false; prefix_.clear(); engine_.drop_snapshot(); engine_.reset(); }
    json dispatch(const json &request) {
        const std::string op = request.at("op").get<std::string>();
        if (op == "info") return engine_.info();
        if (op == "reset") { clear(); return {{"reset", true}}; }
        if (op == "quit") return {{"closed", true}};
        if (op == "tokenize") {
            auto tokens = engine_.tokenize(request.at("text").get<std::string>(), request.value("special", true));
            return {{"tokens", tokens}};
        }
        if (op == "generate") {
            if (!engine_.can_score()) throw std::invalid_argument("tokenizer-only mode cannot generate; GPU loading requires --allow-gpu");
            const auto prompt = engine_.tokenize(request.at("prompt").get<std::string>(), true);
            const int budget = request.value("max_new_tokens", 256);
            if (budget < 1 || budget > 2048 || prompt.empty() || prompt.size() + budget > (size_t)context_)
                throw std::invalid_argument("invalid generation budget or prompt/context limit");
            clear(); engine_.append(prompt);
            std::vector<int32_t> tokens;
            std::vector<uint8_t> bytes;
            std::string finish = "length";
            for (int i = 0; i < budget; ++i) {
                const auto token = engine_.greedy();
                if (engine_.eog(token)) { finish = "eos"; break; }
                tokens.push_back(token);
                const auto piece = engine_.piece(token);
                bytes.insert(bytes.end(), piece.begin(), piece.end());
                if (i + 1 < budget) engine_.append({token});
            }
            // Bytes avoid invalid UTF-8 JSON when a token budget cuts a code point.
            clear();
            return {{"tokens", tokens}, {"bytes", bytes}, {"finish_reason", finish},
                    {"prompt_tokens", prompt.size()}, {"completion_tokens", tokens.size()}};
        }
        if (op != "label_logits") throw std::invalid_argument("unknown operation");
        if (!engine_.can_score()) throw std::invalid_argument("tokenizer-only mode cannot score; GPU loading requires --allow-gpu");
        const auto prefix_text = request.at("prefix").get<std::string>();
        const auto suffixes = request.at("suffixes").get<std::vector<std::string>>();
        const auto labels = request.at("labels").get<std::vector<std::vector<std::string>>>();
        const auto label_prefix = request.value("label_prefix", std::string(" "));
        if (label_prefix != " " && !label_prefix.empty())
            throw std::invalid_argument("label_prefix must be empty or a single space");
        if (suffixes.empty() || suffixes.size() != labels.size() || suffixes.size() > 128)
            throw std::invalid_argument("need 1..128 suffix rows and matching label rows");
        std::vector<std::vector<int32_t>> fulls, label_ids;
        const auto prefix_ids = engine_.tokenize(prefix_text, true);
        // Validate every row before mutating the engine or doing GPU work.
        for (size_t i = 0; i < suffixes.size(); ++i) {
            auto full = engine_.tokenize(prefix_text + suffixes[i], true);
            if (full.empty() || full.size() > (size_t)context_) throw std::invalid_argument("empty prompt or context limit exceeded");
            if (labels[i].empty() || labels[i].size() > 512) throw std::invalid_argument("need 1..512 labels per row");
            std::vector<int32_t> ids;
            std::set<int32_t> distinct;
            for (const auto &label : labels[i]) {
                if (label.empty()) throw std::invalid_argument("empty label");
                auto token = engine_.tokenize(label_prefix + label, false);
                if (token.size() != 1) throw std::invalid_argument("label is not one token: " + label);
                if (!distinct.insert(token[0]).second) throw std::invalid_argument("duplicate label token");
                ids.push_back(token[0]);
            }
            fulls.push_back(std::move(full)); label_ids.push_back(std::move(ids));
        }
        const bool use_cache = request.value("use_cache", true);
        bool reused = use_cache && cached_ && prefix_ids == prefix_;
        size_t forwarded = 0, fallbacks = 0;
        if (use_cache && !reused) {
            clear();
            // A merged full prompt can be shorter than its prefix. Do not load an
            // oversized prefix just to serve such a row; use full-pass fallback.
            if (prefix_ids.size() <= (size_t)context_) {
                engine_.append(prefix_ids); forwarded += prefix_ids.size();
                engine_.save(); prefix_ = prefix_ids; cached_ = true;
            }
        } else if (!use_cache) clear();
        json rows = json::array();
        for (size_t i = 0; i < fulls.size(); ++i) {
            const auto &full = fulls[i];
            const bool boundary_ok = full.size() >= prefix_ids.size() && std::equal(prefix_ids.begin(), prefix_ids.end(), full.begin());
            if (use_cache && cached_ && boundary_ok) {
                engine_.restore();
                std::vector<int32_t> tail(full.begin() + prefix_ids.size(), full.end());
                engine_.append(tail); forwarded += tail.size();
            } else {
                engine_.reset(); engine_.append(full); forwarded += full.size(); ++fallbacks;
            }
            auto z = engine_.score(label_ids[i]);
            if (z.size() != label_ids[i].size() || std::any_of(z.begin(), z.end(), [](double v) { return !std::isfinite(v); }))
                throw std::runtime_error("invalid model logits");
            rows.push_back({{"logits", z}, {"label_ids", label_ids[i]}, {"prompt_tokens", full.size()}});
        }
        return {{"rows", rows}, {"prefix_reused", reused}, {"prefix_tokens", prefix_ids.size()},
                {"forwarded_tokens", forwarded}, {"fallback_rows", fallbacks}};
    }
};
std::unique_ptr<Engine> runtime_engine(const std::string &model, int context, bool gpu);
} // namespace teacher
