#include "sentencepiece.h"
#include <sentencepiece_processor.h>
#include <cstdio>
#include <string>
#include <vector>

struct SPTokenizer {
    sentencepiece::SentencePieceProcessor processor;
};

extern "C" {

SPTokenizer *sp_load_mem(const void *data, size_t size) {
    auto *sp = new SPTokenizer();
    std::string serialized(reinterpret_cast<const char*>(data), size);
    auto status = sp->processor.LoadFromSerializedProto(
        serialized);
    if (!status.ok()) {
        fprintf(stderr, "SentencePiece load from memory failed: %s\n", status.ToString().c_str());
        delete sp;
        return nullptr;
    }
    return sp;
}

SPTokenizer *sp_load(const char *model_path) {
    auto *sp = new SPTokenizer();
    auto status = sp->processor.Load(std::string(model_path));
    if (!status.ok()) {
        fprintf(stderr, "SentencePiece load failed: %s\n", status.ToString().c_str());
        delete sp;
        return nullptr;
    }
    return sp;
}

int sp_encode(const SPTokenizer *sp, const char *text, int *ids_out, int max_tokens) {
    std::vector<int> ids;
    auto status = sp->processor.Encode(std::string(text), &ids);
    if (!status.ok()) {
        fprintf(stderr, "SentencePiece encode failed: %s\n", status.ToString().c_str());
        return -1;
    }
    int n = (int)ids.size();
    if (n > max_tokens) n = max_tokens;
    for (int i = 0; i < n; i++) ids_out[i] = ids[i];
    return n;
}

void sp_free(SPTokenizer *sp) {
    delete sp;
}

} /* extern "C" */
