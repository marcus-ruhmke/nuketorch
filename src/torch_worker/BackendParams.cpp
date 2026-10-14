#include <nuketorch/torch_worker/BackendParams.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace nuketorch::torch_worker {

namespace {

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string trim(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = s.find_last_not_of(" \t");
    return s.substr(begin, end - begin + 1);
}

const std::string* findParam(const std::unordered_map<std::string, std::string>& params, const char* key) {
    const auto it = params.find(key);
    return it == params.end() ? nullptr : &it->second;
}

/// `precision` value -> weights/autocast plus the TF32 implied by the
/// `tf32` / `ieee` sugar.
struct PrecisionValue {
    bool half_weights = false;
    AutocastDtype autocast = AutocastDtype::none;
    Tf32 implied_tf32 = Tf32::keep;
};

PrecisionValue parsePrecisionValue(const std::string& raw) {
    const std::string v = toLower(raw);
    PrecisionValue p;
    if (v == "half" || v == "fp16" || v == "float16") {
        p.half_weights = true;
    } else if (v == "autocast" || v == "amp") {
        p.autocast = AutocastDtype::float16;
    } else if (v == "bf16" || v == "bfloat16") {
        p.autocast = AutocastDtype::bfloat16;
    } else if (v == "float32" || v == "fp32" || v == "float" || v == "full") {
    } else if (v == "tf32") {
        p.implied_tf32 = Tf32::on;
    } else if (v == "ieee") {
        p.implied_tf32 = Tf32::off;
    } else {
        throw std::invalid_argument("unknown precision \"" + raw +
                                    "\" (expected half, autocast, bf16, float32, tf32, or ieee)");
    }
    return p;
}

bool parseFlag(const char* key, const std::string& raw) {
    const std::string v = toLower(raw);
    if (v == "1" || v == "on" || v == "true") {
        return true;
    }
    if (v == "0" || v == "off" || v == "false") {
        return false;
    }
    throw std::invalid_argument(std::string("unknown ") + key + " \"" + raw + "\" (expected 0 or 1)");
}

Tf32 parseTf32(const std::string& raw) {
    const std::string v = toLower(raw);
    if (v == "keep") {
        return Tf32::keep;
    }
    if (v == "on" || v == "1" || v == "true") {
        return Tf32::on;
    }
    if (v == "off" || v == "0" || v == "false") {
        return Tf32::off;
    }
    throw std::invalid_argument("unknown tf32 \"" + raw + "\" (expected on or off)");
}

bool isIdentifier(const std::string& s) {
    if (s.empty() || !(std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '_')) {
        return false;
    }
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isalnum(c) || c == '_'; });
}

std::vector<std::string> parseMethods(const std::string& raw) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        const size_t comma = raw.find(',', start);
        const size_t len = comma == std::string::npos ? std::string::npos : comma - start;
        const std::string name = trim(raw.substr(start, len));
        if (!isIdentifier(name)) {
            throw std::invalid_argument("bad autocast_methods \"" + raw +
                                        "\" (expected comma-separated method names; omit it to "
                                        "autocast every method)");
        }
        out.push_back(name);
        if (comma == std::string::npos) {
            return out;
        }
        start = comma + 1;
    }
}

std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) {
            out += sep;
        }
        out += v[i];
    }
    return out;
}

/// Method scope for labels: names while short, a count otherwise, so labels
/// stay inside the C API's 64-byte metrics strings.
std::string methodsScope(const std::vector<std::string>& methods) {
    std::string s = join(methods, ",");
    if (s.size() > 24) {
        s = std::to_string(methods.size()) + " methods";
    }
    return s;
}

const char* autocastName(AutocastDtype d) {
    return d == AutocastDtype::bfloat16 ? "bfloat16" : "float16";
}

std::string deviceName(const PrecisionTarget& t) {
    return t.device == PrecisionTarget::Device::cpu ? "cpu" : "a non-CUDA GPU";
}

}  // namespace

std::string backendNameFromParams(const std::unordered_map<std::string, std::string>& params) {
    const auto it = params.find("backend");
    if (it == params.end()) {
        return kBackendTorchScript;
    }
    const std::string v = toLower(it->second);
    if (v == "aotinductor" || v == "aoti" || v == "inductor") {
        return kBackendAOTInductor;
    }
    if (v == "tensorrt" || v == "trt") {
        return kBackendTensorRT;
    }
    if (v == "torchscript" || v == "jit" || v == "ts") {
        return kBackendTorchScript;
    }
    // A typo must not silently fall back to TorchScript and then fail with a
    // baffling "invalid TorchScript archive" on a .engine/.pt2 file.
    throw std::invalid_argument("unknown backend \"" + it->second +
                                "\" (expected torchscript, aotinductor, or tensorrt)");
}

bool PrecisionSpec::autocastApplies(const std::string& method) const {
    if (autocast == AutocastDtype::none) {
        return false;
    }
    return autocast_methods.empty() ||
           std::find(autocast_methods.begin(), autocast_methods.end(), method) != autocast_methods.end();
}

PrecisionSpec PrecisionSpec::ieeeFloat32() {
    PrecisionSpec s;
    s.tf32 = Tf32::off;
    return s;
}

bool PrecisionSpec::operator==(const PrecisionSpec& o) const {
    return half_weights == o.half_weights && autocast == o.autocast &&
           autocast_methods == o.autocast_methods && tf32 == o.tf32 && cudnn_benchmark == o.cudnn_benchmark &&
           jit_optimize == o.jit_optimize && policy == o.policy;
}

PrecisionSpec precisionSpecFromParams(const std::unordered_map<std::string, std::string>& params,
                                      bool mixed_precision) {
    PrecisionSpec s;
    const std::string* precision = findParam(params, "precision");
    PrecisionValue pv;
    if (precision) {
        pv = parsePrecisionValue(*precision);
    } else {
        pv.half_weights = mixed_precision;
    }
    s.half_weights = pv.half_weights;
    s.autocast = pv.autocast;
    s.tf32 = pv.implied_tf32;

    if (const std::string* v = findParam(params, "tf32")) {
        const Tf32 t = parseTf32(*v);
        if (pv.implied_tf32 != Tf32::keep && t != pv.implied_tf32) {
            throw std::invalid_argument("precision \"" + *precision + "\" contradicts tf32 \"" + *v + "\"");
        }
        s.tf32 = t;
    }
    if (const std::string* v = findParam(params, "autocast_methods")) {
        s.autocast_methods = parseMethods(*v);
        if (s.autocast == AutocastDtype::none) {
            const std::string got = precision ? *precision : (mixed_precision ? "half" : "float32");
            throw std::invalid_argument("autocast_methods needs an autocast precision (bf16 or autocast), "
                                        "got precision \"" + got + "\"");
        }
    }
    if (const std::string* v = findParam(params, "jit_optimize")) {
        s.jit_optimize = parseFlag("jit_optimize", *v);
    }
    if (const std::string* v = findParam(params, "cudnn_benchmark")) {
        s.cudnn_benchmark = parseFlag("cudnn_benchmark", *v);
    }
    if (const std::string* v = findParam(params, "precision_policy")) {
        const std::string p = toLower(*v);
        if (p == "error") {
            s.policy = PrecisionPolicy::error;
        } else if (p == "fallback_fp32") {
            s.policy = PrecisionPolicy::fallback_fp32;
        } else {
            throw std::invalid_argument("unknown precision_policy \"" + *v +
                                        "\" (expected error or fallback_fp32)");
        }
    }
    return s;
}

PrecisionSpec resolvePrecision(const PrecisionSpec& requested, const PrecisionTarget& target,
                               std::string* fallback_note) {
    using Device = PrecisionTarget::Device;
    PrecisionSpec s = requested;
    if (fallback_note) {
        fallback_note->clear();
    }

    // Pre-0.3 rules, every backend: FP16 weights on GPUs only; plain autocast
    // runs float32 off CUDA.
    if (target.device == Device::cpu) {
        s.half_weights = false;
    }
    if (target.device != Device::cuda && s.autocast == AutocastDtype::float16 && s.autocast_methods.empty()) {
        s.autocast = AutocastDtype::none;
    }

    if (target.baked) {
        // The artifact's compiled precision wins; TF32/bf16 are not checked here.
        if (!s.autocast_methods.empty()) {
            throw std::invalid_argument(
                "autocast_methods needs the torchscript backend (AOTInductor/TensorRT artifacts run forward "
                "with their compiled precision)");
        }
        return s;
    }

    const bool cuda = target.device == Device::cuda;
    const bool ampere = target.cc_major >= 8;
    const std::string cc = std::to_string(target.cc_major) + "." + std::to_string(target.cc_minor);
    std::vector<std::string> dropped;
    const auto refuse = [&](const std::string& mode, const std::string& note, const std::string& why) {
        if (requested.policy == PrecisionPolicy::error) {
            throw std::invalid_argument(mode + " " + why +
                                        " (precision_policy=fallback_fp32 runs float32 instead)");
        }
        dropped.push_back(note);
    };

    if (s.autocast != AutocastDtype::none) {
        const bool bf16 = s.autocast == AutocastDtype::bfloat16;
        std::string mode = std::string("precision ") + (bf16 ? "bf16" : "autocast");
        std::string note = bf16 ? "bf16" : "fp16";
        if (!s.autocast_methods.empty()) {
            mode += " (autocast_methods=" + join(s.autocast_methods, ",") + ")";
            note += ";" + methodsScope(s.autocast_methods);
        }
        if (!cuda) {
            refuse(mode, note, "needs a CUDA device (got " + deviceName(target) + ")");
        } else if (bf16 && !ampere) {
            refuse(mode, note, "needs CUDA compute capability >= 8.0 (got " + cc + ")");
        }
    }
    if (s.tf32 == Tf32::on) {
        if (!cuda) {
            refuse("tf32=on", "tf32", "needs a CUDA device (got " + deviceName(target) + ")");
        } else if (!ampere) {
            refuse("tf32=on", "tf32", "needs CUDA compute capability >= 8.0 (got " + cc + ")");
        }
    }

    if (!dropped.empty()) {
        PrecisionSpec fallback = PrecisionSpec::ieeeFloat32();
        fallback.cudnn_benchmark = s.cudnn_benchmark;
        fallback.jit_optimize = s.jit_optimize;
        fallback.policy = s.policy;
        s = fallback;
        if (fallback_note) {
            *fallback_note = join(dropped, "+");
        }
    }
    return s;
}

std::string precisionLabel(const PrecisionSpec& s, bool baked, const std::string& fallback_note) {
    const bool legacy =
        s.tf32 == Tf32::keep && s.autocast != AutocastDtype::bfloat16 && s.autocast_methods.empty();
    std::string label;
    if (baked && !legacy) {
        label = "baked";
    } else if (s.autocast != AutocastDtype::none) {
        label = std::string("autocast(") + autocastName(s.autocast);
        if (!s.autocast_methods.empty()) {
            label += ";" + methodsScope(s.autocast_methods);
        }
        label += ")";
    } else if (s.half_weights) {
        label = "float16";
    } else {
        label = s.tf32 == Tf32::on ? "float32/tf32" : s.tf32 == Tf32::off ? "float32/ieee" : "float32";
    }
    if (!baked && s.tf32 == Tf32::on && (s.autocast != AutocastDtype::none || s.half_weights)) {
        label += "/tf32";
    }
    if (!fallback_note.empty()) {
        label += "(fallback:" + fallback_note + ")";
    }
    if (!baked && s.jit_optimize == false) {
        label += ",jit0";
    }
    if (label.size() > 63) {
        label.resize(63);
    }
    return label;
}

PrecisionMode precisionModeFromParams(const std::unordered_map<std::string, std::string>& params,
                                      bool mixed_precision) {
    const std::string* precision = findParam(params, "precision");
    if (!precision) {
        return mixed_precision ? PrecisionMode::half : PrecisionMode::full;
    }
    const PrecisionValue pv = parsePrecisionValue(*precision);
    if (pv.half_weights) {
        return PrecisionMode::half;
    }
    return pv.autocast == AutocastDtype::none ? PrecisionMode::full : PrecisionMode::autocast;
}

}  // namespace nuketorch::torch_worker
