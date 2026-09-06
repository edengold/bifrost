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
