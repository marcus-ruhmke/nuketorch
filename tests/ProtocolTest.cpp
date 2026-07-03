#include <gtest/gtest.h>

#include <nuketorch/Protocol.h>

TEST(ProtocolTest, RoundTripWithArbitraryParams) {
    nuketorch::InferenceRequest req;
    req.header.model_path = "/tmp/model.pt";
    req.header.width = 1920;
    req.header.height = 1080;
    req.header.channels = 3;
    req.header.use_gpu = true;
    req.header.mixed_precision = false;
    req.header.debug = true;
    req.request_id = 12345678901234567ULL;
    req.num_inputs = 2;
    req.params["timestep"] = "0.25";
    req.params["max_depth"] = "5";

    const std::string payload = nuketorch::serialize(req);
    nuketorch::InferenceRequest parsed;
    std::string error;
    ASSERT_TRUE(nuketorch::deserialize(payload, parsed, error)) << error;
    EXPECT_TRUE(error.empty());

    EXPECT_EQ(parsed.request_id, req.request_id);
    EXPECT_EQ(parsed.num_inputs, req.num_inputs);
    EXPECT_EQ(parsed.header.model_path, req.header.model_path);
    EXPECT_EQ(parsed.header.width, req.header.width);
    EXPECT_EQ(parsed.header.height, req.header.height);
    EXPECT_EQ(parsed.header.channels, req.header.channels);
    EXPECT_EQ(parsed.header.use_gpu, req.header.use_gpu);
    EXPECT_EQ(parsed.header.mixed_precision, req.header.mixed_precision);
    EXPECT_EQ(parsed.header.debug, req.header.debug);
    EXPECT_EQ(parsed.params.at("timestep"), "0.25");
    EXPECT_EQ(parsed.params.at("max_depth"), "5");
}

TEST(ProtocolTest, EmptyParamsRoundTrip) {
    nuketorch::InferenceRequest req;
    req.header.model_path = "m";
    req.header.width = 1;
    req.header.height = 1;
    req.header.channels = 1;
    req.request_id = 1;
    req.num_inputs = 1;

    const std::string payload = nuketorch::serialize(req);
    nuketorch::InferenceRequest parsed;
    std::string error;
    ASSERT_TRUE(nuketorch::deserialize(payload, parsed, error));
    EXPECT_TRUE(parsed.params.empty());
}

TEST(ProtocolTest, RejectsGarbagePayload) {
    nuketorch::InferenceRequest parsed;
    std::string error;
    EXPECT_FALSE(nuketorch::deserialize("not-a-message", parsed, error));
    EXPECT_FALSE(error.empty());
}

TEST(ProtocolTest, RejectsWrongVersion) {
    nuketorch::InferenceRequest req;
    req.header.model_path = "m";
    req.header.width = 1;
    req.header.height = 1;
    req.header.channels = 1;
    req.num_inputs = 1;

    std::string payload = nuketorch::serialize(req);
    // Corrupt the big-endian version field right after the 4-byte magic.
    payload[4] = 0;
    payload[5] = 0;
    payload[6] = 0;
    payload[7] = 99;

    nuketorch::InferenceRequest parsed;
    std::string error;
    EXPECT_FALSE(nuketorch::deserialize(payload, parsed, error));
    EXPECT_NE(error.find("unsupported version"), std::string::npos);
}

TEST(ProtocolTest, RejectsTrailingGarbage) {
    nuketorch::InferenceRequest req;
    req.header.model_path = "m";
    req.header.width = 1;
    req.header.height = 1;
    req.header.channels = 1;
    req.num_inputs = 1;

    std::string payload = nuketorch::serialize(req) + "extra";
    nuketorch::InferenceRequest parsed;
    std::string error;
    EXPECT_FALSE(nuketorch::deserialize(payload, parsed, error));
    EXPECT_EQ(error, "trailing garbage");
}

TEST(ProtocolTest, RejectsTruncatedPayload) {
    nuketorch::InferenceRequest req;
    req.header.model_path = "some/longer/model/path.pt";
    req.header.width = 8;
    req.header.height = 4;
    req.header.channels = 3;
    req.num_inputs = 2;

    const std::string payload = nuketorch::serialize(req);
    nuketorch::InferenceRequest parsed;
    std::string error;
    // Every strict prefix must fail cleanly, never crash.
    for (size_t len = 0; len < payload.size(); ++len) {
        EXPECT_FALSE(nuketorch::deserialize(payload.substr(0, len), parsed, error)) << len;
    }
}

TEST(ProtocolTest, ParamsWithSpecialCharacters) {
    nuketorch::InferenceRequest req;
    req.header.model_path = "/tmp/x";
    req.header.width = 2;
    req.header.height = 1;
    req.header.channels = 3;
    req.num_inputs = 2;
    req.params["note"] = "a|b=c\nd";

    const std::string payload = nuketorch::serialize(req);
    nuketorch::InferenceRequest parsed;
    std::string error;
    ASSERT_TRUE(nuketorch::deserialize(payload, parsed, error)) << error;
    EXPECT_EQ(parsed.params.at("note"), "a|b=c\nd");
}
