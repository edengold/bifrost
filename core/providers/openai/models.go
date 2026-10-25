package openai

import (
	"slices"
	"strings"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// unionReasoningEfforts merges the per-provider reasoning_efforts ladders from
// an llmgateway-style providers[] array into one ordered union. Provider order
// and each ladder's published order are preserved (providers emit ascending
// ladders); duplicates across providers are dropped.
func unionReasoningEfforts(providers []OpenAIModelProvider) []string {
	var union []string
	for _, p := range providers {
		for _, effort := range p.ReasoningEfforts {
			if !slices.Contains(union, effort) {
				union = append(union, effort)
			}
		}
	}
	return union
}

// ToBifrostListModelsResponse converts an OpenAI list models response to a Bifrost list models response
func (response *OpenAIListModelsResponse) ToBifrostListModelsResponse(providerKey schemas.ModelProvider, allowedModels schemas.WhiteList, blacklistedModels schemas.BlackList, aliases schemas.KeyAliases, unfiltered bool) *schemas.BifrostListModelsResponse {
	if response == nil {
		return nil
	}

	bifrostResponse := &schemas.BifrostListModelsResponse{
		Data: make([]schemas.Model, 0, len(response.Data)),
	}

	pipeline := &providerUtils.ListModelsPipeline{
		AllowedModels:     allowedModels,
		BlacklistedModels: blacklistedModels,
		Aliases:           aliases,
		Unfiltered:        unfiltered,
		ProviderKey:       providerKey,
		MatchFns:          providerUtils.DefaultMatchFns(),
	}
	if pipeline.ShouldEarlyExit() {
		return bifrostResponse
	}

	included := make(map[string]bool)

	for _, model := range response.Data {
		for _, result := range pipeline.FilterModel(model.ID) {
			entry := schemas.Model{
				ID:            string(providerKey) + "/" + result.ResolvedID,
				Created:       model.Created,
				OwnedBy:       schemas.Ptr(model.OwnedBy),
				ContextLength: model.ContextWindow,
			}
			// Provider-reported metadata wins over the GROQ-style context_window
			// fallback; keep ContextWindow mapping for providers that only emit it.
			if model.ContextLength != nil {
				entry.ContextLength = model.ContextLength
			}
			if model.MaxOutputTokens != nil {
				entry.MaxOutputTokens = model.MaxOutputTokens
			}
			if len(model.SupportedParameters) > 0 {
				entry.SupportedParameters = model.SupportedParameters
			}
			if model.Pricing != nil {
				// Clone so entries never alias the decoded struct.
				pricing := *model.Pricing
				entry.Pricing = &pricing
			}
			// Provider-reported thinking levels: OpenRouter-style top-level
			// `reasoning` object wins; otherwise union the llmgateway-style
			// per-provider `reasoning_efforts` ladders in provider order.
			if model.Reasoning != nil {
				reasoning := *model.Reasoning
				reasoning.SupportedEfforts = slices.Clone(model.Reasoning.SupportedEfforts)
				entry.Reasoning = &reasoning
			} else if efforts := unionReasoningEfforts(model.Providers); len(efforts) > 0 {
				entry.Reasoning = &schemas.ModelReasoning{SupportedEfforts: efforts}
			}
			if result.AliasValue != "" {
				entry.Alias = schemas.Ptr(result.AliasValue)
			}
			bifrostResponse.Data = append(bifrostResponse.Data, entry)
			included[strings.ToLower(result.ResolvedID)] = true
		}
	}

	bifrostResponse.Data = append(bifrostResponse.Data,
		pipeline.BackfillModels(included)...)

	return bifrostResponse
}

// ToBifrostModelRetrieveResponse converts an OpenAI model object to a Bifrost model retrieve response
func (model *OpenAIModel) ToBifrostModelRetrieveResponse(providerKey schemas.ModelProvider) *schemas.BifrostModelRetrieveResponse {
	if model == nil {
		return nil
	}

	return &schemas.BifrostModelRetrieveResponse{
		Model: schemas.Model{
			ID:            string(providerKey) + "/" + model.ID,
			Created:       model.Created,
			OwnedBy:       schemas.Ptr(model.OwnedBy),
			ContextLength: model.ContextWindow,
			ShutdownDate:  model.ShutdownDate,
		},
	}
}

// ToOpenAIModelRetrieveResponse converts a Bifrost model retrieve response to an OpenAI model object.
// The ID keeps its provider prefix so this surface agrees with ToOpenAIListModelsResponse.
func ToOpenAIModelRetrieveResponse(response *schemas.BifrostModelRetrieveResponse) *OpenAIModel {
	if response == nil {
		return nil
	}

	openaiModel := &OpenAIModel{
		ID:           response.ID,
		Object:       "model",
		Created:      response.Created,
		ShutdownDate: response.ShutdownDate,
	}
	if response.OwnedBy != nil {
		openaiModel.OwnedBy = *response.OwnedBy
	}
	if response.ContextLength != nil {
		openaiModel.ContextWindow = response.ContextLength
	} else if response.MaxInputTokens != nil {
		openaiModel.ContextWindow = response.MaxInputTokens // Fallback to MaxInputTokens if ContextLength is not set
	}
	return openaiModel
}

// ToOpenAIListModelsResponse converts a Bifrost list models response to an OpenAI list models response
func ToOpenAIListModelsResponse(response *schemas.BifrostListModelsResponse) *OpenAIListModelsResponse {
	if response == nil {
		return nil
	}
	openaiResponse := &OpenAIListModelsResponse{
		Data: make([]OpenAIModel, 0, len(response.Data)),
	}
	for _, model := range response.Data {
		openaiModel := OpenAIModel{
			ID:     model.ID,
			Object: "model",
		}
		if model.Created != nil {
			openaiModel.Created = model.Created
		}
		if model.OwnedBy != nil {
			openaiModel.OwnedBy = *model.OwnedBy
		}
		if model.ContextLength != nil {
			openaiModel.ContextWindow = model.ContextLength
		} else if model.MaxInputTokens != nil {
			openaiModel.ContextWindow = model.MaxInputTokens // Fallback to MaxInputTokens if ContextLength is not set
		}
		if model.Reasoning != nil {
			// Clone so the response never aliases the source model.
			reasoning := *model.Reasoning
			reasoning.SupportedEfforts = slices.Clone(model.Reasoning.SupportedEfforts)
			openaiModel.Reasoning = &reasoning
		}

		openaiResponse.Data = append(openaiResponse.Data, openaiModel)

	}
	return openaiResponse
}
