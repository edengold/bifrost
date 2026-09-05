package openai

import (
	"fmt"
	"net/http"
	"net/url"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// Shared HTTP flow for the OpenAI-compatible batch list/retrieve/cancel endpoints.
// Wire-identical for any provider exposing OpenAI's /v1/batches shape (OpenRouter
// passes "/beta/batches"); the batch-object decoding tolerates extra fields
// (OpenRouter's model/usage/results/finalized_at) via Go's ignore-unknown decode.

// BatchSharedConfig carries the per-provider parameters for the shared batch
// list/retrieve/cancel helpers.
type BatchSharedConfig struct {
	Client               *fasthttp.Client
	BaseURL              string // e.g. "https://api.openai.com" / "https://openrouter.ai/api"
	BatchesPath          string // appended to BaseURL: OpenAI "/v1/batches", OpenRouter "/beta/batches"
	ExtraHeaders         map[string]string
	CustomProviderConfig *schemas.CustomProviderConfig // may be nil (nil = all ops allowed)
	Provider             schemas.ModelProvider         // used by CheckOperationAllowed + errors
	SendBackRawRequest   bool
	SendBackRawResponse  bool
	Logger               schemas.Logger
}

// buildBatchURL constructs the batch URL honoring a path override from context,
// mirroring OpenAIProvider.buildRequestURL.
func (cfg BatchSharedConfig) buildBatchURL(ctx *schemas.BifrostContext, defaultPath string, requestType schemas.RequestType) string {
	path, isCompleteURL := providerUtils.GetRequestPath(ctx, defaultPath, cfg.CustomProviderConfig, requestType)
	if isCompleteURL {
		return path
	}
	return cfg.BaseURL + path
}

// HandleOpenAIBatchListRequest lists batch jobs using serial pagination across keys.
// Exhausts all pages from one key before moving to the next.
func HandleOpenAIBatchListRequest(ctx *schemas.BifrostContext, cfg BatchSharedConfig, keys []schemas.Key, request *schemas.BifrostBatchListRequest) (*schemas.BifrostBatchListResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(cfg.Provider, cfg.CustomProviderConfig, schemas.BatchListRequest); err != nil {
		return nil, err
	}

	sendBackRawRequest := providerUtils.ShouldSendBackRawRequest(ctx, cfg.SendBackRawRequest)
	sendBackRawResponse := providerUtils.ShouldSendBackRawResponse(ctx, cfg.SendBackRawResponse)

	// Initialize serial pagination helper
	helper, err := providerUtils.NewSerialListHelper(keys, request.After, cfg.Logger, true)
	if err != nil {
		return nil, providerUtils.NewBifrostOperationError("invalid pagination cursor", err)
	}

	// Get current key to query
	key, nativeCursor, ok := helper.GetCurrentKey()
	if !ok {
		// All keys exhausted
		return &schemas.BifrostBatchListResponse{
			Object:  "list",
			Data:    []schemas.BifrostBatchRetrieveResponse{},
			HasMore: false,
		}, nil
	}

	// Create request
	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	// Build URL with query params
	baseURL := cfg.buildBatchURL(ctx, cfg.BatchesPath, schemas.BatchListRequest)
	values := url.Values{}
	if request.Limit > 0 {
		values.Set("limit", fmt.Sprintf("%d", request.Limit))
	}
	// Use native cursor from serial helper instead of request.After
	if nativeCursor != "" {
		values.Set("after", nativeCursor)
	}
	requestURL := baseURL
	if encodedValues := values.Encode(); encodedValues != "" {
		requestURL += "?" + encodedValues
	}

	// Set headers
	providerUtils.SetExtraHeaders(ctx, req, cfg.ExtraHeaders, nil)
	req.SetRequestURI(requestURL)
	req.Header.SetMethod(http.MethodGet)
	req.Header.SetContentType("application/json")

	if key.Value.GetValue() != "" {
		req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
	}

	// Make request
	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, cfg.Client, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	// Handle error response
	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, providerUtils.SetErrorLatency(ParseOpenAIError(resp), latency)
	}

	body, decodeErr := providerUtils.CheckAndDecodeBody(resp)
	if decodeErr != nil {
		return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseDecode, decodeErr)
	}

	var openAIResp OpenAIBatchListResponse
	rawRequest, rawResponse, bifrostErr := providerUtils.HandleProviderResponse(body, &openAIResp, nil, sendBackRawRequest, sendBackRawResponse)
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	// Convert batches to Bifrost format
	batches := make([]schemas.BifrostBatchRetrieveResponse, 0, len(openAIResp.Data))
	var lastBatchID string
	for _, batch := range openAIResp.Data {
		batches = append(batches, *batch.ToBifrostBatchRetrieveResponse(latency, sendBackRawRequest, sendBackRawResponse, rawRequest, rawResponse))
		lastBatchID = batch.ID
	}

	// Build cursor for next request
	// OpenAI uses LastID as the cursor for pagination
	nextCursor, hasMore := helper.BuildNextCursor(openAIResp.HasMore, lastBatchID)

	// Convert to Bifrost response
	bifrostResp := &schemas.BifrostBatchListResponse{
		Object:  "list",
		Data:    batches,
		HasMore: hasMore,
		ExtraFields: schemas.BifrostResponseExtraFields{
			Latency: latency.Milliseconds(),
		},
	}
	if nextCursor != "" {
		bifrostResp.NextCursor = &nextCursor
	}

	return bifrostResp, nil
}

// HandleOpenAIBatchRetrieveRequest retrieves a specific batch job by trying each key until found.
func HandleOpenAIBatchRetrieveRequest(ctx *schemas.BifrostContext, cfg BatchSharedConfig, keys []schemas.Key, request *schemas.BifrostBatchRetrieveRequest) (*schemas.BifrostBatchRetrieveResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(cfg.Provider, cfg.CustomProviderConfig, schemas.BatchRetrieveRequest); err != nil {
		return nil, err
	}

	if request.BatchID == "" {
		return nil, providerUtils.NewBifrostOperationError("batch_id is required", nil)
	}

	sendBackRawRequest := providerUtils.ShouldSendBackRawRequest(ctx, cfg.SendBackRawRequest)
	sendBackRawResponse := providerUtils.ShouldSendBackRawResponse(ctx, cfg.SendBackRawResponse)

	var lastErr *schemas.BifrostError
	for _, key := range keys {
		// Create request
		req := fasthttp.AcquireRequest()
		resp := fasthttp.AcquireResponse()

		// Set headers
		providerUtils.SetExtraHeaders(ctx, req, cfg.ExtraHeaders, nil)
		req.SetRequestURI(cfg.BaseURL + cfg.BatchesPath + "/" + request.BatchID)
		req.Header.SetMethod(http.MethodGet)
		req.Header.SetContentType("application/json")

		if key.Value.GetValue() != "" {
			req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
		}

		// Make request
		latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, cfg.Client, req, resp)
		wait()
		if bifrostErr != nil {
			fasthttp.ReleaseRequest(req)
			fasthttp.ReleaseResponse(resp)
			lastErr = bifrostErr
			continue
		}

		// Handle error response
		if resp.StatusCode() != fasthttp.StatusOK {
			lastErr = ParseOpenAIError(resp)
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

		var openAIResp OpenAIBatchResponse
		rawRequest, rawResponse, bifrostErr := providerUtils.HandleProviderResponse(body, &openAIResp, nil, sendBackRawRequest, sendBackRawResponse)
		if bifrostErr != nil {
			fasthttp.ReleaseRequest(req)
			fasthttp.ReleaseResponse(resp)
			lastErr = bifrostErr
			continue
		}

		fasthttp.ReleaseRequest(req)
		fasthttp.ReleaseResponse(resp)

		result := openAIResp.ToBifrostBatchRetrieveResponse(latency, sendBackRawRequest, sendBackRawResponse, rawRequest, rawResponse)
		return result, nil
	}

	return nil, lastErr
}

// HandleOpenAIBatchCancelRequest cancels a batch job by trying each key until successful.
func HandleOpenAIBatchCancelRequest(ctx *schemas.BifrostContext, cfg BatchSharedConfig, keys []schemas.Key, request *schemas.BifrostBatchCancelRequest) (*schemas.BifrostBatchCancelResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(cfg.Provider, cfg.CustomProviderConfig, schemas.BatchCancelRequest); err != nil {
		return nil, err
	}

	if request.BatchID == "" {
		return nil, providerUtils.NewBifrostOperationError("batch_id is required", nil)
	}

	sendBackRawRequest := providerUtils.ShouldSendBackRawRequest(ctx, cfg.SendBackRawRequest)
	sendBackRawResponse := providerUtils.ShouldSendBackRawResponse(ctx, cfg.SendBackRawResponse)

	var lastErr *schemas.BifrostError
	for _, key := range keys {
		// Create request
		req := fasthttp.AcquireRequest()
		resp := fasthttp.AcquireResponse()

		// Set headers
		providerUtils.SetExtraHeaders(ctx, req, cfg.ExtraHeaders, nil)
		req.SetRequestURI(cfg.BaseURL + cfg.BatchesPath + "/" + request.BatchID + "/cancel")
		req.Header.SetMethod(http.MethodPost)
		req.Header.SetContentType("application/json")

		if key.Value.GetValue() != "" {
			req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
		}

		// Make request
		latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, cfg.Client, req, resp)
		wait()
		if bifrostErr != nil {
			fasthttp.ReleaseRequest(req)
			fasthttp.ReleaseResponse(resp)
			lastErr = bifrostErr
			continue
		}

		// Handle error response
		if resp.StatusCode() != fasthttp.StatusOK {
			lastErr = ParseOpenAIError(resp)
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

		var openAIResp OpenAIBatchResponse
		rawRequest, rawResponse, bifrostErr := providerUtils.HandleProviderResponse(body, &openAIResp, nil, sendBackRawRequest, sendBackRawResponse)
		if bifrostErr != nil {
			fasthttp.ReleaseRequest(req)
			fasthttp.ReleaseResponse(resp)
			lastErr = bifrostErr
			continue
		}

		fasthttp.ReleaseRequest(req)
		fasthttp.ReleaseResponse(resp)

		result := &schemas.BifrostBatchCancelResponse{
			ID:           openAIResp.ID,
			Object:       openAIResp.Object,
			Status:       ToBifrostBatchStatus(openAIResp.Status),
			CancellingAt: openAIResp.CancellingAt,
			CancelledAt:  openAIResp.CancelledAt,
			ExtraFields: schemas.BifrostResponseExtraFields{
				Latency: latency.Milliseconds(),
			},
		}

		if openAIResp.RequestCounts != nil {
			result.RequestCounts = schemas.BatchRequestCounts{
				Total:     openAIResp.RequestCounts.Total,
				Completed: openAIResp.RequestCounts.Completed,
				Failed:    openAIResp.RequestCounts.Failed,
			}
		}

		if sendBackRawRequest {
			result.ExtraFields.RawRequest = rawRequest
		}

		if sendBackRawResponse {
			result.ExtraFields.RawResponse = rawResponse
		}

		return result, nil
	}

	return nil, lastErr
}
