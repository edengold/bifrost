package openrouter_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/maximhq/bifrost/core/providers/openrouter"
	"github.com/maximhq/bifrost/core/schemas"
)

type batchTestLogger struct{}

func (batchTestLogger) Debug(string, ...any)                   {}
func (batchTestLogger) Info(string, ...any)                    {}
func (batchTestLogger) Warn(string, ...any)                    {}
func (batchTestLogger) Error(string, ...any)                   {}
func (batchTestLogger) Fatal(string, ...any)                   {}
func (batchTestLogger) SetLevel(schemas.LogLevel)              {}
func (batchTestLogger) SetOutputType(schemas.LoggerOutputType) {}
func (batchTestLogger) LogHTTPRequest(schemas.LogLevel, string) schemas.LogEventBuilder {
	return schemas.NoopLogEvent
}

// capturedRequest records exact wire bytes seen by the fake OpenRouter server.
type capturedRequest struct {
	Method string
	Path   string
	Body   []byte
}

type requestRecorder struct {
	mu       sync.Mutex
	requests []capturedRequest
}

func (rec *requestRecorder) wrap(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = r.Body.Close()
		rec.mu.Lock()
		rec.requests = append(rec.requests, capturedRequest{Method: r.Method, Path: r.URL.Path, Body: body})
		rec.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		next(w, r)
	}
}

func (rec *requestRecorder) last() capturedRequest {
	rec.mu.Lock()
	defer rec.mu.Unlock()
	return rec.requests[len(rec.requests)-1]
}

func (rec *requestRecorder) count() int {
	rec.mu.Lock()
	defer rec.mu.Unlock()
	return len(rec.requests)
}

func newTestProvider(baseURL string) *openrouter.OpenRouterProvider {
	return openrouter.NewOpenRouterProvider(&schemas.ProviderConfig{
		NetworkConfig: schemas.NetworkConfig{
			BaseURL:                        baseURL,
			DefaultRequestTimeoutInSeconds: 10,
		},
	}, batchTestLogger{})
}

func newTestKey() schemas.Key {
	return schemas.Key{
		ID:             "test-key",
		Value:          *schemas.NewSecretVar("test-api-key"),
		Models:         schemas.WhiteList{"*"},
		Weight:         1.0,
		UseForBatchAPI: schemas.Ptr(true),
	}
}

func testCtx() *schemas.BifrostContext {
	return schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
}

func errMessage(err *schemas.BifrostError) string {
	if err == nil || err.Error == nil {
		return ""
	}
	return err.Error.Message
}

const batchObjectJSON = `{
	"id": "batch_123",
	"object": "batch",
	"endpoint": "/v1/chat/completions",
	"model": "openai/gpt-4o",
	"completion_window": "24h",
	"status": "validating",
	"created_at": 1782097200,
	"finalized_at": null,
	"request_counts": { "total": 2, "completed": 0, "failed": 0 },
	"usage": null,
	"results": null,
	"error": null
}`

const completedBatchWithResultsJSON = `{
	"id": "batch_done",
	"object": "batch",
	"endpoint": "/v1/chat/completions",
	"model": "openai/gpt-4o",
	"completion_window": "24h",
	"status": "completed",
	"created_at": 1782097200,
	"finalized_at": 1782100800,
	"request_counts": { "total": 2, "completed": 2, "failed": 0 },
	"usage": { "prompt_tokens": 40, "completion_tokens": 80, "total_tokens": 120, "cost": 0.000225, "is_byok": false },
	"results": [
		{
			"custom_id": "req-1",
			"response": { "status_code": 200, "request_id": "gen-1", "body": { "id": "gen-1", "object": "chat.completion", "model": "openai/gpt-4o", "usage": { "prompt_tokens": 20, "completion_tokens": 40, "total_tokens": 60 } } },
			"error": null
		},
		{
			"custom_id": "req-2",
			"response": { "status_code": 200, "request_id": "gen-2", "body": { "id": "gen-2", "object": "chat.completion", "model": "openai/gpt-4o", "usage": { "prompt_tokens": 20, "completion_tokens": 40, "total_tokens": 60 } } },
			"error": null
		}
	],
	"error": null
}`

func TestOpenRouterBatchCreate_InlineRequests(t *testing.T) {
	t.Parallel()

	rec := &requestRecorder{}
	srv := httptest.NewServer(rec.wrap(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted) // OpenRouter answers 202, not 200
		_, _ = w.Write([]byte(batchObjectJSON))
	}))
	t.Cleanup(srv.Close)

	provider := newTestProvider(srv.URL)
	resp, err := provider.BatchCreate(testCtx(), newTestKey(), &schemas.BifrostBatchCreateRequest{
		Provider: schemas.OpenRouter,
		Model:    schemas.Ptr("openai/gpt-4o"),
		Endpoint: schemas.BatchEndpointChatCompletions,
		Requests: []schemas.BatchRequestItem{
			{CustomID: "req-1", Body: map[string]any{"messages": []any{map[string]any{"role": "user", "content": "hi"}}}},
			{CustomID: "req-2", Body: map[string]any{"messages": []any{map[string]any{"role": "user", "content": "bye"}}}},
		},
	})
	if err != nil {
		t.Fatalf("BatchCreate failed: %s", errMessage(err))
	}
	if resp.ID != "batch_123" {
		t.Fatalf("expected batch ID batch_123, got %q", resp.ID)
	}
	if resp.Status != schemas.BatchStatusValidating {
		t.Fatalf("expected status validating, got %q", resp.Status)
	}
	if resp.RequestCounts.Total != 2 {
		t.Fatalf("expected total=2, got %d", resp.RequestCounts.Total)
	}

	req := rec.last()
	if req.Method != http.MethodPost || req.Path != "/beta/batches" {
		t.Fatalf("expected POST /beta/batches, got %s %s", req.Method, req.Path)
	}
	if !strings.Contains(string(req.Body), `"custom_id":"req-1"`) || !strings.Contains(string(req.Body), `"custom_id":"req-2"`) {
		t.Fatalf("expected inline requests on the wire, got %s", string(req.Body))
	}
}

// TestOpenRouterBatchCreate_FieldOrder asserts the wire body serializes
// endpoint and model BEFORE requests — OpenRouter's stream parser 400s otherwise.
func TestOpenRouterBatchCreate_FieldOrder(t *testing.T) {
	t.Parallel()

	rec := &requestRecorder{}
	srv := httptest.NewServer(rec.wrap(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte(batchObjectJSON))
	}))
	t.Cleanup(srv.Close)

	provider := newTestProvider(srv.URL)
	_, err := provider.BatchCreate(testCtx(), newTestKey(), &schemas.BifrostBatchCreateRequest{
		Provider: schemas.OpenRouter,
		Model:    schemas.Ptr("openai/gpt-4o"),
		Endpoint: schemas.BatchEndpointChatCompletions,
		Requests: []schemas.BatchRequestItem{
			{CustomID: "req-1", Body: map[string]any{"messages": []any{}}},
		},
	})
	if err != nil {
		t.Fatalf("BatchCreate failed: %s", errMessage(err))
	}

	raw := string(rec.last().Body)
	iEndpoint := strings.Index(raw, `"endpoint"`)
	iModel := strings.Index(raw, `"model"`)
	iRequests := strings.Index(raw, `"requests"`)
	if iEndpoint < 0 || iModel < 0 || iRequests < 0 {
		t.Fatalf("wire body missing required keys: %s", raw)
	}
	if !(iEndpoint < iModel && iModel < iRequests) {
		t.Fatalf("wire body must serialize endpoint, model before requests; got %s", raw)
	}
}

func TestOpenRouterBatchCreate_RejectsFileUpload(t *testing.T) {
	t.Parallel()

	rec := &requestRecorder{}
	srv := httptest.NewServer(rec.wrap(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("unexpected request to %s", r.URL.Path)
	}))
	t.Cleanup(srv.Close)

	provider := newTestProvider(srv.URL)
	_, err := provider.BatchCreate(testCtx(), newTestKey(), &schemas.BifrostBatchCreateRequest{
		Provider:    schemas.OpenRouter,
		Model:       schemas.Ptr("openai/gpt-4o"),
		Endpoint:    schemas.BatchEndpointChatCompletions,
		InputFileID: "file_abc",
	})
	if err == nil {
		t.Fatal("expected error for file-based create")
	}
	if !strings.Contains(errMessage(err), "does not support file uploads") {
		t.Fatalf("unexpected error: %s", errMessage(err))
	}
	if rec.count() != 0 {
		t.Fatalf("no HTTP call expected when rejecting file upload, got %d", rec.count())
	}
}

func TestOpenRouterBatchCreate_RejectsUnsupportedEndpoint(t *testing.T) {
	t.Parallel()

	rec := &requestRecorder{}
	srv := httptest.NewServer(rec.wrap(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("unexpected request to %s", r.URL.Path)
	}))
	t.Cleanup(srv.Close)

	provider := newTestProvider(srv.URL)
	_, err := provider.BatchCreate(testCtx(), newTestKey(), &schemas.BifrostBatchCreateRequest{
		Provider: schemas.OpenRouter,
		Model:    schemas.Ptr("openai/gpt-4o"),
		Endpoint: schemas.BatchEndpointCompletions,
		Requests: []schemas.BatchRequestItem{{CustomID: "req-1", Body: map[string]any{}}},
	})
	if err == nil {
		t.Fatal("expected error for /v1/completions endpoint")
	}
	if !strings.Contains(errMessage(err), "does not support endpoint") {
		t.Fatalf("unexpected error: %s", errMessage(err))
	}
	if rec.count() != 0 {
		t.Fatalf("no HTTP call expected on endpoint rejection, got %d", rec.count())
	}
}

// TestOpenRouterBatchCreate_ModelNormalization covers the :batch/prefix strips
// and the per-request-body model fallback.
func TestOpenRouterBatchCreate_ModelNormalization(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		model        *string
		bodyModels   []string
		wantWire     string
		wantErr      string
		wantBody0Mdl string
	}{
		{
			name:     "prefix and batch suffix stripped from request model",
			model:    schemas.Ptr("openrouter/google/gemini-3.5-flash-lite:batch"),
			wantWire: "google/gemini-3.5-flash-lite",
		},
		{
			name:         "body models used and normalized when request model absent",
			bodyModels:   []string{"openai/gpt-4o:batch"},
			wantWire:     "openai/gpt-4o",
			wantBody0Mdl: "openai/gpt-4o",
		},
		{
			name:    "no model anywhere errors",
			wantErr: "model is required",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			rec := &requestRecorder{}
			srv := httptest.NewServer(rec.wrap(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusAccepted)
				_, _ = w.Write([]byte(batchObjectJSON))
			}))
			t.Cleanup(srv.Close)

			provider := newTestProvider(srv.URL)
			requests := make([]schemas.BatchRequestItem, 0, len(tc.bodyModels)+1)
			if len(tc.bodyModels) > 0 {
				for i, m := range tc.bodyModels {
					requests = append(requests, schemas.BatchRequestItem{
						CustomID: "req-" + string(rune('1'+i)),
						Body:     map[string]any{"model": m, "messages": []any{}},
					})
				}
			} else {
				requests = append(requests, schemas.BatchRequestItem{CustomID: "req-1", Body: map[string]any{"messages": []any{}}})
			}
			_, err := provider.BatchCreate(testCtx(), newTestKey(), &schemas.BifrostBatchCreateRequest{
				Provider: schemas.OpenRouter,
				Model:    tc.model,
				Endpoint: schemas.BatchEndpointChatCompletions,
				Requests: requests,
			})
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(errMessage(err), tc.wantErr) {
					t.Fatalf("expected error %q, got %v", tc.wantErr, errMessage(err))
				}
				if rec.count() != 0 {
					t.Fatalf("no HTTP call expected when model missing, got %d", rec.count())
				}
				return
			}
			if err != nil {
				t.Fatalf("BatchCreate failed: %s", errMessage(err))
			}

			var wire struct {
				Model    string `json:"model"`
				Requests []struct {
					Body map[string]any `json:"body"`
				} `json:"requests"`
			}
			if decErr := json.Unmarshal(rec.last().Body, &wire); decErr != nil {
				t.Fatalf("decode wire body: %v body=%s", decErr, string(rec.last().Body))
			}
			if wire.Model != tc.wantWire {
				t.Fatalf("expected batch-level model %q, got %q", tc.wantWire, wire.Model)
			}
			if tc.wantBody0Mdl != "" {
				got, _ := wire.Requests[0].Body["model"].(string)
				if got != tc.wantBody0Mdl {
					t.Fatalf("expected body[0].model %q, got %q", tc.wantBody0Mdl, got)
				}
			}
		})
	}
}

func TestOpenRouterBatchResults_Completed(t *testing.T) {
	t.Parallel()

	rec := &requestRecorder{}
	srv := httptest.NewServer(rec.wrap(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(completedBatchWithResultsJSON))
	}))
	t.Cleanup(srv.Close)

	provider := newTestProvider(srv.URL)
	resp, err := provider.BatchResults(testCtx(), []schemas.Key{newTestKey()}, &schemas.BifrostBatchResultsRequest{
		Provider: schemas.OpenRouter,
		BatchID:  "batch_done",
	})
	if err != nil {
		t.Fatalf("BatchResults failed: %s", errMessage(err))
	}
	req := rec.last()
	if req.Method != http.MethodGet || req.Path != "/beta/batches/batch_done" {
		t.Fatalf("expected GET /beta/batches/batch_done, got %s %s", req.Method, req.Path)
	}
	if resp.BatchID != "batch_done" {
		t.Fatalf("expected batch id echoed, got %q", resp.BatchID)
	}
	if len(resp.Results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(resp.Results))
	}
	if resp.Results[0].CustomID != "req-1" || resp.Results[1].CustomID != "req-2" {
		t.Fatalf("custom_id mapping broken: %q %q", resp.Results[0].CustomID, resp.Results[1].CustomID)
	}
	if resp.Results[0].Response == nil || resp.Results[0].Response.StatusCode != 200 {
		t.Fatalf("expected per-item response with status 200, got %#v", resp.Results[0].Response)
	}
}

func TestOpenRouterBatchResults_NonCompletedErrors(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(batchObjectJSON)) // status validating, results null
	}))
	t.Cleanup(srv.Close)

	provider := newTestProvider(srv.URL)
	_, err := provider.BatchResults(testCtx(), []schemas.Key{newTestKey()}, &schemas.BifrostBatchResultsRequest{
		Provider: schemas.OpenRouter,
		BatchID:  "batch_123",
	})
	if err == nil {
		t.Fatal("expected error for non-completed batch")
	}
	if !strings.Contains(errMessage(err), "batch results not available") {
		t.Fatalf("unexpected error: %s", errMessage(err))
	}
}

// TestOpenRouterBatchSharedHelpers asserts list/retrieve/cancel hit
// /beta/batches (not /v1/batches) through the extracted OpenAI helpers.
func TestOpenRouterBatchSharedHelpers(t *testing.T) {
	t.Parallel()

	listJSON := `{"object":"list","data":[` + batchObjectJSON + `],"first_id":"batch_123","last_id":"batch_123","has_more":false}`
	cancelJSON := `{"id":"batch_123","object":"batch","status":"cancelled","cancelled_at":1782097300,"request_counts":{"total":2,"completed":1,"failed":0}}`

	rec := &requestRecorder{}
	srv := httptest.NewServer(rec.wrap(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/beta/batches":
			_, _ = w.Write([]byte(listJSON))
		case r.Method == http.MethodGet && r.URL.Path == "/beta/batches/batch_123":
			_, _ = w.Write([]byte(batchObjectJSON))
		case r.Method == http.MethodPost && r.URL.Path == "/beta/batches/batch_123/cancel":
			_, _ = w.Write([]byte(cancelJSON))
		default:
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"message":"no route"}}`))
		}
	}))
	t.Cleanup(srv.Close)

	provider := newTestProvider(srv.URL)
	ctx := testCtx()
	keys := []schemas.Key{newTestKey()}

	listResp, err := provider.BatchList(ctx, keys, &schemas.BifrostBatchListRequest{Provider: schemas.OpenRouter})
	if err != nil {
		t.Fatalf("BatchList failed: %s", errMessage(err))
	}
	if len(listResp.Data) != 1 || listResp.Data[0].ID != "batch_123" {
		t.Fatalf("unexpected list response: %#v", listResp.Data)
	}

	getResp, err := provider.BatchRetrieve(ctx, keys, &schemas.BifrostBatchRetrieveRequest{Provider: schemas.OpenRouter, BatchID: "batch_123"})
	if err != nil {
		t.Fatalf("BatchRetrieve failed: %s", errMessage(err))
	}
	if getResp.Status != schemas.BatchStatusValidating {
		t.Fatalf("unexpected status %q", getResp.Status)
	}

	cancelResp, err := provider.BatchCancel(ctx, keys, &schemas.BifrostBatchCancelRequest{Provider: schemas.OpenRouter, BatchID: "batch_123"})
	if err != nil {
		t.Fatalf("BatchCancel failed: %s", errMessage(err))
	}
	if cancelResp.Status != schemas.BatchStatusCancelled {
		t.Fatalf("unexpected cancel status %q", cancelResp.Status)
	}

	want := []string{"GET /beta/batches", "GET /beta/batches/batch_123", "POST /beta/batches/batch_123/cancel"}
	got := []string{}
	rec.mu.Lock()
	for _, rq := range rec.requests {
		got = append(got, rq.Method+" "+rq.Path)
	}
	rec.mu.Unlock()
	if len(got) != len(want) {
		t.Fatalf("expected %d requests, got %v", len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("request %d: expected %q, got %q", i, want[i], got[i])
		}
	}
}
