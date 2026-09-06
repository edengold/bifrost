// Provider-reported list-models metadata ("live meta") overlay surface.
//
// The live cache captures what each provider's /v1/models endpoint advertises
// per model (context limits, supported parameters, rates). Once captured, it
// wins per-field over the Bifrost datasheet: operator pricing overrides still
// apply last, and fields a provider doesn't report fall back to the datasheet.
//
// This file sits in the composer package because it needs both the datasheet
// and live packages; the datasheet reaches the same data through the
// SetProviderMetaResolver hook installed by Init (cost path), so the dependency
// stays acyclic.
package modelcatalog

import (
	"slices"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/modelcatalog/live"
)

// providerMeta resolves provider-reported metadata for (provider, model).
//
// The live store keys entries by the configured provider name verbatim
// (FetchAndStoreLiveForKey passes schemas.ModelProvider straight through), so
// callers must pass the same name they used when upserting. The cost path
// calls this with what it computed as catalogProvider; custom providers are
// keyed by their configured name end-to-end, no folding needed.
func (mc *ModelCatalog) providerMeta(provider, model string) *live.ModelMeta {
	if mc == nil || mc.live == nil || model == "" {
		return nil
	}
	meta := mc.live.MetaForProvider(schemas.ModelProvider(provider))
	if meta == nil {
		return nil
	}
	return meta[model]
}

// GetLiveModelMeta returns provider-reported metadata for one model of one
// provider, or nil when the live cache has nothing for it. Exported for the
// HTTP handlers' model-details surface.
func (mc *ModelCatalog) GetLiveModelMeta(provider schemas.ModelProvider, model string) *live.ModelMeta {
	if mc == nil {
		return nil
	}
	return mc.providerMeta(string(provider), model)
}

// overlayLiveModelInfo replaces fields on a display-facing *schemas.Model with
// provider-reported live values, per-field: a non-nil meta field wins, and a
// reported zero is a real price. The meta pointer values are cloned so callers
// never alias live-cache state. Rates convert to the API's string form via
// formatCost, mirroring ApplyModelInfo's datasheet mapping.
func overlayLiveModelInfo(model *schemas.Model, meta *live.ModelMeta) {
	if model == nil || meta == nil {
		return
	}
	if meta.ContextLength != nil {
		model.ContextLength = new(*meta.ContextLength)
	}
	if meta.MaxInputTokens != nil {
		model.MaxInputTokens = new(*meta.MaxInputTokens)
	}
	if meta.MaxOutputTokens != nil {
		model.MaxOutputTokens = new(*meta.MaxOutputTokens)
	}
	if len(meta.SupportedParameters) > 0 {
		model.SupportedParameters = slices.Clone(meta.SupportedParameters)
	}
	p := meta.Pricing
	if p == nil {
		return
	}
	pricing := model.Pricing
	if pricing == nil {
		pricing = &schemas.Pricing{}
		model.Pricing = pricing
	}
	if p.PromptPerToken != nil {
		pricing.Prompt = new(formatCost(*p.PromptPerToken))
	}
	if p.CompletionPerToken != nil {
		pricing.Completion = new(formatCost(*p.CompletionPerToken))
	}
	if p.CacheReadPerToken != nil {
		pricing.InputCacheRead = new(formatCost(*p.CacheReadPerToken))
	}
	if p.CacheWritePerToken != nil {
		pricing.InputCacheWrite = new(formatCost(*p.CacheWritePerToken))
	}
	if p.PerRequest != nil {
		pricing.Request = new(formatCost(*p.PerRequest))
	}
	if p.PerImage != nil {
		pricing.Image = new(formatCost(*p.PerImage))
	}
	if p.PerWebSearchQuery != nil {
		pricing.WebSearch = new(formatCost(*p.PerWebSearchQuery))
	}
}
