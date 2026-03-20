#include <nuketorch/Protocol.h>

#include <arpa/inet.h>

#include <cstring>
#include <stdexcept>

namespace nuketorch {
namespace {

void appendU32(std::string& out, uint32_t v) {
    const uint32_t net = htonl(v);
    out.append(reinterpret_cast<const char*>(&net), sizeof(net));
}

void appendI32(std::string& out, int32_t v) {
    appendU32(out, static_cast<uint32_t>(v));
}

void appendU8(std::string& out, uint8_t v) {
    out.push_back(static_cast<char>(v));
}

void appendBlob(std::string& out, const std::string& s) {
    if (s.size() > 0xFFFFFFFFu) {
        throw std::runtime_error("string too long for protocol");
    }
    appendU32(out, static_cast<uint32_t>(s.size()));
    out.append(s);
}

bool readU32(const std::string& in, size_t& off, uint32_t& out, std::string& error) {
    if (off + sizeof(uint32_t) > in.size()) {
        error = "truncated u32";
        return false;
    }
    uint32_t net = 0;
    std::memcpy(&net, in.data() + off, sizeof(net));
    out = ntohl(net);
    off += sizeof(net);
    return true;
}

bool readI32(const std::string& in, size_t& off, int32_t& out, std::string& error) {
    uint32_t u = 0;
    if (!readU32(in, off, u, error)) {
        return false;
    }
    out = static_cast<int32_t>(u);
    return true;
}

bool readU8(const std::string& in, size_t& off, uint8_t& out, std::string& error) {
    if (off + 1 > in.size()) {
        error = "truncated u8";
        return false;
    }
    out = static_cast<uint8_t>(static_cast<unsigned char>(in[off]));
    ++off;
    return true;
}

bool readBlob(const std::string& in, size_t& off, std::string& out, std::string& error) {
    uint32_t len = 0;
    if (!readU32(in, off, len, error)) {
        return false;
    }
    if (off + len > in.size()) {
        error = "truncated blob";
        return false;
    }
    out.assign(in.data() + off, len);
    off += len;
    return true;
}

}  // namespace

std::string serialize(const InferenceRequest& req) {
    std::string out;
    out.append(kInferenceWireMagic, sizeof(kInferenceWireMagic));
    appendU32(out, 1u);  // version

    appendU32(out, static_cast<uint32_t>(req.shm_inputs.size()));
    for (const auto& name : req.shm_inputs) {
        appendBlob(out, name);
    }
    appendBlob(out, req.shm_output);
    appendBlob(out, req.header.model_path);

    appendI32(out, req.header.width);
    appendI32(out, req.header.height);
    appendI32(out, req.header.channels);
    appendU8(out, req.header.use_gpu ? 1u : 0u);
    appendU8(out, req.header.mixed_precision ? 1u : 0u);
    appendU8(out, req.header.debug ? 1u : 0u);

    const uint32_t n_params = static_cast<uint32_t>(req.params.size());
    appendU32(out, n_params);
    for (const auto& kv : req.params) {
        appendBlob(out, kv.first);
        appendBlob(out, kv.second);
    }
    return out;
}

bool deserialize(const std::string& payload, InferenceRequest& out, std::string& error) {
    size_t off = 0;
    if (payload.size() < sizeof(kInferenceWireMagic)) {
        error = "payload too small";
        return false;
    }
    if (std::memcmp(payload.data(), kInferenceWireMagic, sizeof(kInferenceWireMagic)) != 0) {
        error = "bad magic";
        return false;
    }
    off += sizeof(kInferenceWireMagic);

    uint32_t version = 0;
    if (!readU32(payload, off, version, error)) {
        return false;
    }
    if (version != 1u) {
        error = "unsupported version";
        return false;
    }

    uint32_t num_inputs = 0;
    if (!readU32(payload, off, num_inputs, error)) {
        return false;
    }
    out.shm_inputs.clear();
    out.shm_inputs.reserve(num_inputs);
    for (uint32_t i = 0; i < num_inputs; ++i) {
        std::string name;
        if (!readBlob(payload, off, name, error)) {
            return false;
        }
        out.shm_inputs.push_back(std::move(name));
    }

    if (!readBlob(payload, off, out.shm_output, error)) {
        return false;
    }
    if (!readBlob(payload, off, out.header.model_path, error)) {
        return false;
    }

    int32_t w = 0;
    int32_t h = 0;
    int32_t c = 0;
    if (!readI32(payload, off, w, error)) {
        return false;
    }
    if (!readI32(payload, off, h, error)) {
        return false;
    }
    if (!readI32(payload, off, c, error)) {
        return false;
    }
    out.header.width = w;
    out.header.height = h;
    out.header.channels = c;

    uint8_t ug = 0;
    uint8_t mp = 0;
    uint8_t db = 0;
    if (!readU8(payload, off, ug, error)) {
        return false;
    }
    if (!readU8(payload, off, mp, error)) {
        return false;
    }
    if (!readU8(payload, off, db, error)) {
        return false;
    }
    out.header.use_gpu = ug != 0;
    out.header.mixed_precision = mp != 0;
    out.header.debug = db != 0;

    uint32_t n_params = 0;
    if (!readU32(payload, off, n_params, error)) {
        return false;
    }
    out.params.clear();
    for (uint32_t i = 0; i < n_params; ++i) {
        std::string k;
        std::string v;
        if (!readBlob(payload, off, k, error)) {
            return false;
        }
        if (!readBlob(payload, off, v, error)) {
            return false;
        }
        out.params[std::move(k)] = std::move(v);
    }

    if (off != payload.size()) {
        error = "trailing garbage";
        return false;
    }

    error.clear();
    return true;
}

}  // namespace nuketorch
