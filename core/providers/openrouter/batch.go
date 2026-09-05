package openrouter

import (
	"net/http"
	"strings"

	"github.com/maximhq/bifrost/core/providers/openai"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// OpenRouter Batch API (beta) types and provider methods.
// Client surface stays OpenAI-shaped (BifrostBatch* DTOs); on the wire Bifrost
// calls OpenRouter's /api/beta/batches endpoints. Differences from OpenAI:
//   - create takes inline `requests` (no file upload, no input_file_id) and a
//     batch-level `model`; the server stream-parses, so `endpoint` and `model`
//     MUST serialize before `requests`.
//   - a terminal `completed` batch carries its `results` array inline on the
//     batch object (no output-file download).
//   - completion_window is fixed at 24h (field not sent).
//   - supported endpoints: /v1/chat/completions, /v1/responses, /v1/messages,
//     /v1/embeddings (NOT /v1/completions).

// OpenRouterBatchRequest is the create request body.
// Field order is load-bearing: OpenRouter stream-parses and requires
// endpoint, model before requests. Struct order (not MarshalSorted) controls it.
type OpenRouterBatchRequest struct {
	Endpoint string                     `json:"endpoint"`
	Model    string                     `json:"model"`
	Requests []schemas.BatchRequestItem `json:"requests"`
}

// OpenRouterBatchResponse embeds the OpenAI wire type so list/retrieve/cancel
// decoding is shared; adds OpenRouter-specific fields.
type OpenRouterBatchResponse struct {
	openai.OpenAIBatchResponse
	Model       string                    `json:"model,omitempty"`
	FinalizedAt *int64                    `json:"finalized_at,omitempty"`
	Usage       *OpenRouterBatchUsage     `json:"usage,omitempty"`
	Results     []schemas.BatchResultItem `json:"results,omitempty"`
}

// OpenRouterBatchUsage is the aggregate usage reported on a completed batch.
type OpenRouterBatchUsage struct {
	PromptTokens     int      `json:"prompt_tokens"`
	CompletionTokens int      `json:"completion_tokens"`
	TotalTokens      int      `json:"total_tokens"`
	Cost             *float64 `json:"cost,omitempty"`
	IsBYOK           bool     `json:"is_byok,omitempty"`
}

// openRouterBatchPathOverride is the default path appended to BaseURL for batch ops.
const openRouterBatchesPath = "/beta/batches"

// normalizeOpenRouterBatchModel strips the "openrouter/" provider prefix and a
// trailing ":batch" variant suffix so a model copied from Bifrost's /v1/models
// (which advertises `openrouter/<slug>:batch` rows) submits as the bare slug.
func normalizeOpenRouterBatchModel(model string) string {
	providerPrefix := string(schemas.OpenRouter) + "/"
	if strings.HasPrefix(strings.ToLower(model), strings.ToLower(providerPrefix)) {
		model = model[len(providerPrefix):]
	}
	return strings.TrimSuffix(model, ":batch")
}

// batchSharedConfig builds the shared OpenAI-compatible list/retrieve/cancel
// config for this provider.
func (provider *OpenRouterProvider) batchSharedConfig() openai.BatchSharedConfig {
	return openai.BatchSharedConfig{
		Client:              provider.client,
		BaseURL:             provider.networkConfig.BaseURL,
		BatchesPath:         openRouterBatchesPath,
		ExtraHeaders:        provider.networkConfig.ExtraHeaders,
		Provider:            schemas.OpenRouter,
		SendBackRawRequest:  provider.sendBackRawRequest,
		SendBackRawResponse: provider.sendBackRawResponse,
		Logger:              provider.logger,
	}
}

// BatchCreate creates a new batch job on OpenRouter's beta Batch API.
func (provider *OpenRouterProvider) BatchCreate(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostBatchCreateRequest) (*schemas.BifrostBatchCreateResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.OpenRouter, nil, schemas.BatchCreateRequest); err != nil {
		return nil, err
	}

	if request.InputFileID != "" {
		return nil, providerUtils.NewBifrostOperationError("openrouter batch api does not support file uploads; pass inline requests instead", nil)
	}
	if len(request.Requests) == 0 {
		return nil, providerUtils.NewBifrostOperationError("requests array is required for openrouter batch api", nil)
	}
	if request.Endpoint == "" {
		return nil, providerUtils.NewBifrostOperationError("endpoint is required for openrouter batch api", nil)
	}
	switch request.Endpoint {
	case schemas.BatchEndpointChatCompletions, schemas.BatchEndpointResponses, schemas.BatchEndpointEmbeddings, schemas.BatchEndpointMessages:
	default:
		return nil, providerUtils.NewBifrostOperationError("openrouter batch api does not support endpoint "+string(request.Endpoint), nil)
	}

	// Derive the batch-level model: explicit request.Model wins, else the first
	// per-request body's model. Bodies with a ":batch" model are normalized to
	// match the batch-level value — OpenRouter rejects a body model that differs.
	batchModel := ""
	if request.Model != nil && *request.Model != "" {
		batchModel = normalizeOpenRouterBatchModel(*request.Model)
	}
	for i := range request.Requests {
		if request.Requests[i].Body == nil {
			continue
		}
		if raw, ok := request.Requests[i].Body["model"].(string); ok && raw != "" {
			normalized := normalizeOpenRouterBatchModel(raw)
			request.Requests[i].Body["model"] = normalized
			if batchModel == "" {
				batchModel = normalized
			}
		}
	}
	if batchModel == "" {
		return nil, providerUtils.NewBifrostOperationError("model is required for openrouter batch api", nil)
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)
	req.SetRequestURI(provider.networkConfig.BaseURL + providerUtils.GetPathFromContext(ctx, openRouterBatchesPath))
	req.Header.SetMethod(http.MethodPost)
	req.Header.SetContentType("application/json")

	if key.Value.GetValue() != "" {
		req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
	}

	openRouterReq := &OpenRouterBatchRequest{
		Endpoint: string(request.Endpoint),
		Model:    batchModel,
		Requests: request.Requests,
	}

	jsonData, err := providerUtils.MarshalSorted(openRouterReq)
	if err != nil {
		return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderRequestMarshal, err)
	}
	req.SetBody(jsonData)

	sendBackRawRequest := providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest)
	sendBackRawResponse := providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse)

	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, nil, sendBackRawRequest, sendBackRawResponse, latency)
	}

	// OpenRouter answers 202 Accepted (OpenAI: 200) — accept both.
	if resp.StatusCode() != fasthttp.StatusOK && resp.StatusCode() != fasthttp.StatusAccepted {
		return nil, providerUtils.EnrichError(ctx, openai.ParseOpenAIError(resp), jsonData, nil, sendBackRawRequest, sendBackRawResponse, latency)
	}

	body, err := providerUtils.CheckAndDecodeBody(resp)
	if err != nil {
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseDecode, err), jsonData, nil, sendBackRawRequest, sendBackRawResponse, latency)
	}

	var openRouterResp OpenRouterBatchResponse
	rawRequest, rawResponse, bifrostErr := providerUtils.HandleProviderResponse(body, &openRouterResp, jsonData, sendBackRawRequest, sendBackRawResponse)
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, body, sendBackRawRequest, sendBackRawResponse, latency)
	}

	return openRouterResp.OpenAIBatchResponse.ToBifrostBatchCreateResponse(latency, sendBackRawRequest, sendBackRawResponse, rawRequest, rawResponse), nil
}

// BatchList lists batch jobs using serial pagination across keys.
func (provider *OpenRouterProvider) BatchList(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostBatchListRequest) (*schemas.BifrostBatchListResponse, *schemas.BifrostError) {
	return openai.HandleOpenAIBatchListRequest(ctx, provider.batchSharedConfig(), keys, request)
}

// BatchRetrieve retrieves a specific batch job by trying each key until found.
func (provider *OpenRouterProvider) BatchRetrieve(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostBatchRetrieveRequest) (*schemas.BifrostBatchRetrieveResponse, *schemas.BifrostError) {
	return openai.HandleOpenAIBatchRetrieveRequest(ctx, provider.batchSharedConfig(), keys, request)
}

// BatchCancel cancels a batch job by trying each key until successful.
func (provider *OpenRouterProvider) BatchCancel(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostBatchCancelRequest) (*schemas.BifrostBatchCancelResponse, *schemas.BifrostError) {
	return openai.HandleOpenAIBatchCancelRequest(ctx, provider.batchSharedConfig(), keys, request)
}

// BatchResults retrieves batch results — OpenRouter returns the results array
// inline on the batch object once the batch is completed (no output file).
// Does NOT delegate to BatchRetrieve: the shared OpenAI response type drops
// the `results` field.
func (provider *OpenRouterProvider) BatchResults(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostBatchResultsRequest) (*schemas.BifrostBatchResultsResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.OpenRouter, nil, schemas.BatchResultsRequest); err != nil {
		return nil, err
	}

	if request.BatchID == "" {
		return nil, providerUtils.NewBifrostOperationError("batch_id is required", nil)
	}

	sendBackRawRequest := providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest)
	sendBackRawResponse := providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse)

	var lastErr *schemas.BifrostError
	for _, key := range keys {
		req := fasthttp.AcquireRequest()
		resp := fasthttp.AcquireResponse()

		providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)
		req.SetRequestURI(provider.networkConfig.BaseURL + openRouterBatchesPath + "/" + request.BatchID)
		req.Header.SetMethod(http.MethodGet)
		req.Header.SetContentType("application/json")

		if key.Value.GetValue() != "" {
			req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
		}

		latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
		wait()
		if bifrostErr != nil {
			fasthttp.ReleaseRequest(req)
			fasthttp.ReleaseResponse(resp)
			lastErr = bifrostErr
			continue
		}

		if resp.StatusCode() != fasthttp.StatusOK && resp.StatusCode() != fasthttp.StatusAccepted {
			lastErr = openai.ParseOpenAIError(resp)
			fasthttp.ReleaseRequest(req)
			fasthttp.ReleaseResponse(resp)
			continue
		}

		body, err := providerUtils.CheckAndDecodeBody(resp)
		if err != nil {
			fasthttp.ReleaseRequest(req)
			fasthttp.ReleaseResponse(resp)
			lastErr = providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseDecode, err)
			continue
		}

		var openRouterResp OpenRouterBatchResponse
		_, _, bifrostErr = providerUtils.HandleProviderResponse(body, &openRouterResp, nil, sendBackRawRequest, sendBackRawResponse)
		if bifrostErr != nil {
			fasthttp.ReleaseRequest(req)
			fasthttp.ReleaseResponse(resp)
			lastErr = bifrostErr
			continue
		}

		fasthttp.ReleaseRequest(req)
		fasthttp.ReleaseResponse(resp)

		// Results are only present once the batch reaches `completed`;
		// failed/expired/cancelled batches carry no results array either.
		if openRouterResp.Status != string(schemas.BatchStatusCompleted) {
			return nil, providerUtils.NewBifrostOperationError("batch results not available: batch status is "+openRouterResp.Status, nil)
		}

		resultsResp := &schemas.BifrostBatchResultsResponse{
			BatchID:  request.BatchID,
			Endpoint: schemas.BatchEndpoint(openRouterResp.Endpoint),
			Results:  openRouterResp.Results,
			ExtraFields: schemas.BifrostResponseExtraFields{
				Latency: latency.Milliseconds(),
			},
		}
		if sendBackRawResponse {
			resultsResp.ExtraFields.RawResponse = openRouterResp.Results
		}
		return resultsResp, nil
	}

	return nil, lastErr
}

// BatchDelete is not supported by OpenRouter provider.
func (provider *OpenRouterProvider) BatchDelete(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostBatchDeleteRequest) (*schemas.BifrostBatchDeleteResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.BatchDeleteRequest, provider.GetProviderKey())
}
