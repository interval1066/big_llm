#include "tokenizer/opennmt_adapter.hpp"
#include <gtest/gtest.h>

TEST(OpenNMTAdapter, ModeSwitch) {
    OpenNMTAdapter tokenizer("config.json");
    
    auto bpe_tokens = tokenizer.encode("hello", OpenNMTAdapter::Mode::BPE);
    auto sp_tokens = tokenizer.encode("hello", OpenNMTAdapter::Mode::SENTENCEPIECE);
    
    EXPECT_NE(bpe_tokens, sp_tokens); // Different tokenization schemes
}

