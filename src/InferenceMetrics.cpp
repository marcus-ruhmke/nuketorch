#include <nuketorch/InferenceMetrics.h>

#include <arpa/inet.h>

#include <cstring>
#include <sstream>
#include <stdexcept>

namespace nuketorch {
namespace {

constexpr char kMetricsWireMagic[4] = {'N', 'T', 'M', '1'};

void appendU32(std::string& out, uint32_t v) {
    const uint32_t net = htonl(v);
    out.append(reinterpret_cast<const char*>(&net), sizeof(net));
}

void appendBlob(std::string& out, const std::string& s) {
    if (s.size() > 0xFFFFFFFFu) {
        throw std::runtime_error("string too long for metrics wire format");
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

std::string doubleToWire(double v) {
    std::ostringstream oss;
    oss.precision(17);
    oss << v;
    return oss.str();
}

bool wireToDouble(const std::string& s, double& out) {
    std::istringstream iss(s);
    iss >> out;
    return !iss.fail() && iss.eof();
}

bool wireToInt64(const std::string& s, int64_t& out) {
    std::istringstream iss(s);
    iss >> out;
    return !iss.fail() && iss.eof();
}

}  // namespace

std::string serializeMetrics(const InferenceMetrics& m) {
    std::string out;
    out.append(kMetricsWireMagic, sizeof(kMetricsWireMagic));
    appendU32(out, 1u);  // version

    std::ostringstream peak_oss;
    peak_oss << m.peak_gpu_memory_bytes;
    const std::string peak_str = peak_oss.str();

    struct Entry {
        const char* key;
        std::string value;
    };
    const Entry entries[] = {
        {"backend_forward_ms", doubleToWire(m.backend_forward_ms)},
        {"gpu_compute_ms", doubleToWire(m.gpu_compute_ms)},
        {"tensor_prep_ms", doubleToWire(m.tensor_prep_ms)},
        {"output_copy_ms", doubleToWire(m.output_copy_ms)},
        {"model_load_ms", doubleToWire(m.model_load_ms)},
        {"backend", m.backend},
        {"device", m.device},
        {"dtype", m.dtype},
        {"peak_gpu_memory_bytes", peak_str},
    };

    appendU32(out, static_cast<uint32_t>(sizeof(entries) / sizeof(entries[0])));
    for (const auto& e : entries) {
        appendBlob(out, std::string(e.key));
        appendBlob(out, e.value);
    }
    return out;
}

bool parseMetricsPayload(const std::string& payload, InferenceMetrics& out, std::string& error) {
    size_t off = 0;
    if (payload.size() < sizeof(kMetricsWireMagic)) {
        error = "metrics payload too small";
        return false;
    }
    if (std::memcmp(payload.data(), kMetricsWireMagic, sizeof(kMetricsWireMagic)) != 0) {
        error = "bad metrics magic";
        return false;
    }
    off += sizeof(kMetricsWireMagic);

    uint32_t version = 0;
    if (!readU32(payload, off, version, error)) {
        return false;
    }
    if (version != 1u) {
        error = "unsupported metrics version";
        return false;
    }

    uint32_t n = 0;
    if (!readU32(payload, off, n, error)) {
        return false;
    }

    InferenceMetrics m;
    for (uint32_t i = 0; i < n; ++i) {
        std::string k;
        std::string v;
        if (!readBlob(payload, off, k, error)) {
            return false;
        }
        if (!readBlob(payload, off, v, error)) {
            return false;
        }
        if (k == "backend_forward_ms") {
            if (!wireToDouble(v, m.backend_forward_ms)) {
                error = "bad backend_forward_ms";
                return false;
            }
        } else if (k == "gpu_compute_ms") {
            if (!wireToDouble(v, m.gpu_compute_ms)) {
                error = "bad gpu_compute_ms";
                return false;
            }
        } else if (k == "tensor_prep_ms") {
            if (!wireToDouble(v, m.tensor_prep_ms)) {
                error = "bad tensor_prep_ms";
                return false;
            }
        } else if (k == "output_copy_ms") {
            if (!wireToDouble(v, m.output_copy_ms)) {
                error = "bad output_copy_ms";
                return false;
            }
        } else if (k == "model_load_ms") {
            if (!wireToDouble(v, m.model_load_ms)) {
                error = "bad model_load_ms";
                return false;
            }
        } else if (k == "backend") {
            m.backend = std::move(v);
        } else if (k == "device") {
            m.device = std::move(v);
        } else if (k == "dtype") {
            m.dtype = std::move(v);
        } else if (k == "peak_gpu_memory_bytes") {
            if (!wireToInt64(v, m.peak_gpu_memory_bytes)) {
                error = "bad peak_gpu_memory_bytes";
                return false;
            }
        } else {
            error = "unknown metrics key: " + k;
            return false;
        }
    }

    if (off != payload.size()) {
        error = "trailing garbage in metrics payload";
        return false;
    }

    out = std::move(m);
    error.clear();
    return true;
}

bool parseInferenceOkResponse(const std::string& response, InferenceMetrics& out, std::string& error) {
    if (response.size() < 2u || response[0] != 'O' || response[1] != 'K') {
        error = "response is not OK";
        return false;
    }
    if (response.size() == 2u) {
        out = InferenceMetrics{};
        error.clear();
        return true;
    }
    const std::string payload = response.substr(2);
    return parseMetricsPayload(payload, out, error);
}

}  // namespace nuketorch
