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
    req.shm_inputs = {"/in0", "/in1"};
    req.shm_output = "/out";
    req.params["timestep"] = "0.25";
    req.params["max_depth"] = "5";

    const std::string payload = nuketorch::serialize(req);
    nuketorch::InferenceRequest parsed;
    std::string error;
    ASSERT_TRUE(nuketorch::deserialize(payload, parsed, error)) << error;
    EXPECT_TRUE(error.empty());

    EXPECT_EQ(parsed.shm_inputs, req.shm_inputs);
    EXPECT_EQ(parsed.shm_output, req.shm_output);
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
    req.shm_inputs = {"/a"};
    req.shm_output = "/b";

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

TEST(ProtocolTest, ParamsWithSpecialCharacters) {
    nuketorch::InferenceRequest req;
    req.header.model_path = "/tmp/x";
    req.header.width = 2;
    req.header.height = 1;
    req.header.channels = 3;
    req.shm_inputs = {"/i0", "/i1"};
    req.shm_output = "/o";
    req.params["note"] = "a|b=c\nd";

    const std::string payload = nuketorch::serialize(req);
    nuketorch::InferenceRequest parsed;
    std::string error;
    ASSERT_TRUE(nuketorch::deserialize(payload, parsed, error)) << error;
    EXPECT_EQ(parsed.params.at("note"), "a|b=c\nd");
}

TEST(ProtocolTest, SingleInputRoundTrip) {
    nuketorch::InferenceRequest req;
    req.header.model_path = "/m.pt";
    req.header.width = 4;
    req.header.height = 2;
    req.header.channels = 1;
    req.shm_inputs = {"/only"};
    req.shm_output = "/out";

    nuketorch::InferenceRequest parsed;
    std::string error;
    ASSERT_TRUE(nuketorch::deserialize(nuketorch::serialize(req), parsed, error));
    ASSERT_EQ(parsed.shm_inputs.size(), 1u);
    EXPECT_EQ(parsed.shm_inputs[0], "/only");
}
