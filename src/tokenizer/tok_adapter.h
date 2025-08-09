#pragma once
#include "bpe_tokenizer.hpp"  // Base BPE tokenizer
#include <opennmt/Tokenizer.h> // OpenNMT's tokenizer
#include <mutex>

namespace lm::tokenizer {

class OpenNMTAdapter : public BPETokenizer {
public:
    // Supported tokenization modes
    enum class Mode { BPE, SENTENCEPIECE, WORDPIECE };

    // Initialize with OpenNMT config file
    explicit OpenNMTAdapter(const std::filesystem::path& config_path);
    
    // Tokenization with mode selection
    std::vector<int32_t> encode(std::string_view text, 
                               Mode mode = Mode::BPE,
                               const SamplingOptions& opts = {}) const override;

    // Conversion utilities
    static std::vector<std::string> to_opennmt_tokens(const std::vector<int32_t>& our_tokens);
    static std::vector<int32_t> from_opennmt_tokens(const std::vector<std::string>& opennmt_tokens);

    // Configuration
    void set_mode(Mode mode) { mode_ = mode; }

private:
    mutable std::mutex mutex_;  // Thread safety for OpenNMT's tokenizer
    Mode mode_ = Mode::BPE;
    std::unique_ptr<openmt::Tokenizer> opennmt_tokenizer_;

    // Internal implementations
    std::vector<int32_t> encode_bpe(std::string_view text, const SamplingOptions& opts) const;
    std::vector<int32_t> encode_sp(std::string_view text) const;
    std::vector<int32_t> encode_wp(std::string_view text) const;
};

} // namespace lm::tokenizer
