package openai

import (
	"testing"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestToBifrostListModelsResponse_DecodesProviderMetadata pins the llmgateway /
// vLLM-style /v1/models payload: context_length, max_output, pricing and
// supported_parameters must survive the decode + conversion into schemas.Model.
// Before this change the fields were silently dropped at OpenAIModel.
func TestToBifrostListModelsResponse_DecodesProviderMetadata(t *testing.T) {
	t.Parallel()

	raw := []byte(`{"object":"list","data":[
		{"id":"glm-5.3-flash","object":"model","created":1787702400,
		 "context_length":1048576,"max_output":131072,
		 "pricing":{"prompt":"0.0000001","completion":"0.00000025","input_cache_read":"0.00000002"},
		 "supported_parameters":["temperature","top_p"]},
		{"id":"bare-model","object":"model","created":1700000000}
	]}`)

	var resp OpenAIListModelsResponse
	require.NoError(t, sonic.Unmarshal(raw, &resp))

	out := resp.ToBifrostListModelsResponse("llmgateway", nil, nil, nil, true)
	require.Len(t, out.Data, 2)

	meta := out.Data[0]
	assert.Equal(t, "llmgateway/glm-5.3-flash", meta.ID)
	require.NotNil(t, meta.ContextLength)
	assert.Equal(t, 1048576, *meta.ContextLength)
	require.NotNil(t, meta.MaxOutputTokens)
	assert.Equal(t, 131072, *meta.MaxOutputTokens)
	assert.Equal(t, []string{"temperature", "top_p"}, meta.SupportedParameters)
	require.NotNil(t, meta.Pricing)
	require.NotNil(t, meta.Pricing.Prompt)
	assert.Equal(t, "0.0000001", *meta.Pricing.Prompt)
	require.NotNil(t, meta.Pricing.Completion)
	assert.Equal(t, "0.00000025", *meta.Pricing.Completion)
	require.NotNil(t, meta.Pricing.InputCacheRead)
	assert.Equal(t, "0.00000002", *meta.Pricing.InputCacheRead)
	assert.Nil(t, meta.Pricing.Request)

	// An entry without metadata still round-trips with nil optional fields.
	plain := out.Data[1]
	assert.Equal(t, "llmgateway/bare-model", plain.ID)
	assert.Nil(t, plain.ContextLength)
	assert.Nil(t, plain.MaxOutputTokens)
	assert.Nil(t, plain.Pricing)
	assert.Nil(t, plain.SupportedParameters)
}

// TestToBifrostListModelsResponse_ContextWindowFallback keeps the GROQ-only
// context_window mapping working when the newer context_length is absent.
func TestToBifrostListModelsResponse_ContextWindowFallback(t *testing.T) {
	t.Parallel()

	raw := []byte(`{"object":"list","data":[
		{"id":"gpt-oss-120b","object":"model","context_window":131072}
	]}`)

	var resp OpenAIListModelsResponse
	require.NoError(t, sonic.Unmarshal(raw, &resp))

	out := resp.ToBifrostListModelsResponse(schemas.Groq, nil, nil, nil, true)
	require.Len(t, out.Data, 1)
	require.NotNil(t, out.Data[0].ContextLength)
	assert.Equal(t, 131072, *out.Data[0].ContextLength)
	assert.Nil(t, out.Data[0].Pricing)
}

// TestToBifrostListModelsResponse_ContextLengthWinsOverContextWindow pins
// precedence when a provider emits both.
func TestToBifrostListModelsResponse_ContextLengthWinsOverContextWindow(t *testing.T) {
	t.Parallel()

	raw := []byte(`{"object":"list","data":[
		{"id":"m","object":"model","context_window":4096,"context_length":8192}
	]}`)

	var resp OpenAIListModelsResponse
	require.NoError(t, sonic.Unmarshal(raw, &resp))

	out := resp.ToBifrostListModelsResponse(schemas.OpenAI, nil, nil, nil, true)
	require.Len(t, out.Data, 1)
	require.NotNil(t, out.Data[0].ContextLength)
	assert.Equal(t, 8192, *out.Data[0].ContextLength)
}

// TestToBifrostListModelsResponse_ReasoningEfforts pins both provider wire
// shapes for thinking levels: the llmgateway-style per-provider
// `providers[].reasoning_efforts` union and the OpenRouter-style top-level
// `reasoning` object, plus absence → nil and reverse-conversion passthrough.
func TestToBifrostListModelsResponse_ReasoningEfforts(t *testing.T) {
	t.Parallel()

	raw := []byte(`{"object":"list","data":[
		{"id":"gpt-5","object":"model","providers":[
			{"reasoning_efforts":["minimal","low","medium","high"]},
			{"reasoning_efforts":["low","medium","high","xhigh"]}]},
		{"id":"gemini-3-pro","object":"model","reasoning":{
			"supported_efforts":["low","high"],"default_effort":"low","default_enabled":true}},
		{"id":"plain-model","object":"model","providers":[{"reasoning_efforts":[]}]}
	]}`)

	var resp OpenAIListModelsResponse
	require.NoError(t, sonic.Unmarshal(raw, &resp))

	out := resp.ToBifrostListModelsResponse("llmgateway", nil, nil, nil, true)
	require.Len(t, out.Data, 3)

	// Ordered union across providers, provider order preserved, deduped.
	union := out.Data[0]
	require.NotNil(t, union.Reasoning)
	assert.Equal(t, []string{"minimal", "low", "medium", "high", "xhigh"}, union.Reasoning.SupportedEfforts)

	// Top-level reasoning object cloned through, other fields kept.
	or := out.Data[1]
	require.NotNil(t, or.Reasoning)
	assert.Equal(t, []string{"low", "high"}, or.Reasoning.SupportedEfforts)
	require.NotNil(t, or.Reasoning.DefaultEffort)
	assert.Equal(t, "low", *or.Reasoning.DefaultEffort)
	require.NotNil(t, or.Reasoning.DefaultEnabled)
	assert.True(t, *or.Reasoning.DefaultEnabled)

	// Neither shape → nil.
	assert.Nil(t, out.Data[2].Reasoning)

	// Entries must not alias the decoded payload.
	resp.Data[0].Providers[0].ReasoningEfforts[0] = "mutated"
	resp.Data[1].Reasoning.SupportedEfforts[0] = "mutated"
	assert.Equal(t, "minimal", union.Reasoning.SupportedEfforts[0])
	assert.Equal(t, "low", or.Reasoning.SupportedEfforts[0])

	// Reverse conversion re-advertises Reasoning (the forward converter already
	// flattened the providers[] union into entry.Reasoning).
	back := ToOpenAIListModelsResponse(out)
	require.Len(t, back.Data, 3)
	require.NotNil(t, back.Data[0].Reasoning)
	assert.Equal(t, []string{"minimal", "low", "medium", "high", "xhigh"}, back.Data[0].Reasoning.SupportedEfforts)
	require.NotNil(t, back.Data[1].Reasoning)
	assert.Equal(t, []string{"low", "high"}, back.Data[1].Reasoning.SupportedEfforts)
	assert.Nil(t, back.Data[2].Reasoning)

	// Clone discipline: mutating the reverse output must not touch the source.
	back.Data[1].Reasoning.SupportedEfforts[0] = "mutated"
	assert.Equal(t, "low", or.Reasoning.SupportedEfforts[0])
}
