package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"kiro-go/config"
	accountpool "kiro-go/pool"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestClaudeToKiroInjectsPrivateFinishToolAndHandlesNameCollision(t *testing.T) {
	req := &ClaudeRequest{
		Model:    "claude-opus-5",
		Messages: []ClaudeMessage{{Role: "user", Content: "do the work"}},
		Tools: []ClaudeTool{
			{Name: defaultInternalFinishToolName, Description: "client tool", InputSchema: map[string]interface{}{"type": "object"}},
		},
	}
	payload := ClaudeToKiro(req, false)
	if !payload.StrictCompletion {
		t.Fatal("tool-enabled Claude request must enable strict completion")
	}
	if payload.InternalFinishToolName == defaultInternalFinishToolName {
		t.Fatalf("internal tool must avoid client name collision, got %q", payload.InternalFinishToolName)
	}
	ctx := payload.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext
	if ctx == nil || len(ctx.Tools) != 2 {
		t.Fatalf("expected client and internal tools, got %#v", ctx)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if bytes.Contains(raw, []byte("StrictCompletion")) || bytes.Contains(raw, []byte("InternalFinishToolName")) {
		t.Fatalf("private completion metadata leaked into Kiro JSON: %s", raw)
	}
	decision := classifyClaudeCompletionRound(payload, &claudeCompletionRound{toolUses: []KiroToolUse{{Name: defaultInternalFinishToolName}}})
	if decision.outcome != "tool_use" || decision.clientTools != 1 {
		t.Fatalf("restored same-name client tool was misclassified: %#v", decision)
	}
}

func TestClaudeToKiroDoesNotInjectFinishToolWithoutClientTools(t *testing.T) {
	payload := ClaudeToKiro(&ClaudeRequest{
		Model:    "claude-opus-5",
		Messages: []ClaudeMessage{{Role: "user", Content: "hello"}},
	}, false)
	if payload.StrictCompletion || payload.InternalFinishToolName != "" {
		t.Fatalf("plain request unexpectedly enabled completion contract: %#v", payload)
	}
}

func TestClaudeCompletionContractContinuesProgressThenReturnsClientTool(t *testing.T) {
	payload := newStrictCompletionTestPayload()
	var calls int
	var secondPayload KiroPayload
	installKiroStreamTestClient(t, roundTripFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		if calls == 2 {
			decodeKiroRequestBody(t, req, &secondPayload)
			return kiroStreamTestResponse(bytes.NewReader(bytes.Join([][]byte{
				awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
					"toolUseId": "tool_client", "name": "Bash", "input": `{"command":"go test"}`, "stop": true,
				}),
				awsEventStreamFrame(t, "meteringEvent", map[string]interface{}{"usage": 0.5}),
			}, nil))), nil
		}
		return kiroStreamTestResponse(bytes.NewReader(bytes.Join([][]byte{
			awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{"content": "Now I will inspect the remaining files."}),
			awsEventStreamFrame(t, "meteringEvent", map[string]interface{}{"usage": 0.25}),
			awsEventStreamStopFrame(t, "END_TURN"),
		}, nil))), nil
	}))
	installKiroRetryWait(t, func(time.Duration) {})

	var text string
	var tools []KiroToolUse
	var stopReason string
	var completed int
	var credits float64
	err := callKiroForClaude(context.Background(), newKiroRetryTestAPIKeyAccount(""), payload, &KiroStreamCallback{
		OnText:       func(chunk string, _ bool) { text += chunk },
		OnToolUse:    func(tool KiroToolUse) { tools = append(tools, tool) },
		OnStopReason: func(reason string) { stopReason = reason },
		OnComplete:   func(int, int) { completed++ },
		OnCredits:    func(value float64) { credits = value },
	})
	if err != nil {
		t.Fatalf("unexpected completion error: %v", err)
	}
	if calls != 2 || text != "" || len(tools) != 1 || tools[0].Name != "Bash" || stopReason != "TOOL_USE" || completed != 1 {
		t.Fatalf("calls=%d text=%q tools=%#v stop=%q completed=%d", calls, text, tools, stopReason, completed)
	}
	if credits != 0.75 {
		t.Fatalf("credits=%v, want accumulated 0.75", credits)
	}
	if len(secondPayload.ConversationState.History) < 2 {
		t.Fatalf("continuation did not retain progress in history: %#v", secondPayload.ConversationState.History)
	}
	lastAssistant := secondPayload.ConversationState.History[len(secondPayload.ConversationState.History)-1].AssistantResponseMessage
	if lastAssistant == nil || !strings.Contains(lastAssistant.Content, "remaining files") {
		t.Fatalf("partial assistant progress missing from continuation history: %#v", lastAssistant)
	}
	if !strings.Contains(secondPayload.ConversationState.CurrentMessage.UserInputMessage.Content, payload.InternalFinishToolName) {
		t.Fatalf("continuation instruction does not name finish tool: %q", secondPayload.ConversationState.CurrentMessage.UserInputMessage.Content)
	}
	if len(payload.ConversationState.History) != 0 {
		t.Fatal("completion controller must not mutate the caller payload")
	}
}

func TestClaudeCompletionContractPreservesMaxTokenPartialForContinuation(t *testing.T) {
	payload := newStrictCompletionTestPayload()
	var calls int
	var secondPayload KiroPayload
	installKiroStreamTestClient(t, roundTripFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		if calls == 1 {
			return kiroStreamTestResponse(bytes.NewReader(bytes.Join([][]byte{
				awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{"content": "partial result"}),
				awsEventStreamStopFrame(t, "MAX_TOKENS"),
			}, nil))), nil
		}
		decodeKiroRequestBody(t, req, &secondPayload)
		return kiroStreamTestResponse(bytes.NewReader(internalFinishFrame(t, payload.InternalFinishToolName, "completed", "final result"))), nil
	}))
	installKiroRetryWait(t, func(time.Duration) {})

	var text, stopReason string
	err := callKiroForClaude(context.Background(), newKiroRetryTestAPIKeyAccount(""), payload, &KiroStreamCallback{
		OnText:       func(chunk string, _ bool) { text += chunk },
		OnStopReason: func(reason string) { stopReason = reason },
	})
	if err != nil {
		t.Fatalf("unexpected completion error: %v", err)
	}
	if text != "final result" || stopReason != "END_TURN" || calls != 2 {
		t.Fatalf("text=%q stop=%q calls=%d", text, stopReason, calls)
	}
	lastAssistant := secondPayload.ConversationState.History[len(secondPayload.ConversationState.History)-1].AssistantResponseMessage
	if lastAssistant == nil || lastAssistant.Content != "partial result" {
		t.Fatalf("MAX_TOKENS partial output not preserved: %#v", lastAssistant)
	}
}

func TestClaudeCompletionContractAcceptsCompletedAndBlocked(t *testing.T) {
	for _, test := range []struct {
		status  string
		message string
	}{
		{"completed", "all work is done"},
		{"blocked", "Which environment should I modify?"},
	} {
		t.Run(test.status, func(t *testing.T) {
			payload := newStrictCompletionTestPayload()
			installKiroStreamTestClient(t, roundTripFunc(func(*http.Request) (*http.Response, error) {
				return kiroStreamTestResponse(bytes.NewReader(internalFinishFrame(t, payload.InternalFinishToolName, test.status, test.message))), nil
			}))
			var text, stopReason string
			var tools int
			err := callKiroForClaude(context.Background(), newKiroRetryTestAPIKeyAccount(""), payload, &KiroStreamCallback{
				OnText:       func(chunk string, _ bool) { text += chunk },
				OnToolUse:    func(KiroToolUse) { tools++ },
				OnStopReason: func(reason string) { stopReason = reason },
			})
			if err != nil || text != test.message || tools != 0 || stopReason != "END_TURN" {
				t.Fatalf("err=%v text=%q tools=%d stop=%q", err, text, tools, stopReason)
			}
		})
	}
}

func TestClaudeCompletionContractRejectsInvalidFinishAndContinues(t *testing.T) {
	for _, invalidInput := range []map[string]interface{}{
		{"status": "done", "message": "not valid"},
		{"status": "completed", "message": ""},
	} {
		t.Run(invalidInput["status"].(string)+"_message", func(t *testing.T) {
			payload := newStrictCompletionTestPayload()
			var calls int
			installKiroStreamTestClient(t, roundTripFunc(func(*http.Request) (*http.Response, error) {
				calls++
				if calls == 1 {
					return kiroStreamTestResponse(bytes.NewReader(bytes.Join([][]byte{
						awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
							"toolUseId": "finish_invalid", "name": payload.InternalFinishToolName, "input": invalidInput, "stop": true,
						}),
						awsEventStreamStopFrame(t, "END_TURN"),
					}, nil))), nil
				}
				return kiroStreamTestResponse(bytes.NewReader(internalFinishFrame(t, payload.InternalFinishToolName, "completed", "recovered"))), nil
			}))

			var text string
			err := callKiroForClaude(context.Background(), newKiroRetryTestAPIKeyAccount(""), payload, &KiroStreamCallback{
				OnText: func(chunk string, _ bool) { text += chunk },
			})
			if err != nil || calls != 2 || text != "recovered" {
				t.Fatalf("err=%v calls=%d text=%q", err, calls, text)
			}
		})
	}
}

func TestClaudeCompletionContractClientToolWinsOverInternalFinish(t *testing.T) {
	payload := newStrictCompletionTestPayload()
	installKiroStreamTestClient(t, roundTripFunc(func(*http.Request) (*http.Response, error) {
		return kiroStreamTestResponse(bytes.NewReader(bytes.Join([][]byte{
			internalFinishFrame(t, payload.InternalFinishToolName, "completed", "must be hidden"),
			awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
				"toolUseId": "client", "name": "Bash", "input": `{"command":"pwd"}`, "stop": true,
			}),
		}, nil))), nil
	}))

	var text string
	var tools []KiroToolUse
	err := callKiroForClaude(context.Background(), newKiroRetryTestAPIKeyAccount(""), payload, &KiroStreamCallback{
		OnText:    func(chunk string, _ bool) { text += chunk },
		OnToolUse: func(tool KiroToolUse) { tools = append(tools, tool) },
	})
	if err != nil || text != "" || len(tools) != 1 || tools[0].Name != "Bash" {
		t.Fatalf("err=%v text=%q tools=%#v", err, text, tools)
	}
}

func TestClaudeCompletionContractMissingStopReasonRetriesOriginalRequest(t *testing.T) {
	payload := newStrictCompletionTestPayload()
	var calls int
	var requests []KiroPayload
	installKiroStreamTestClient(t, roundTripFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		var decoded KiroPayload
		decodeKiroRequestBody(t, req, &decoded)
		requests = append(requests, decoded)
		if calls == 1 {
			return kiroStreamTestResponse(bytes.NewReader(awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
				"content": "truncated sentence",
			}))), nil
		}
		return kiroStreamTestResponse(bytes.NewReader(awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
			"toolUseId": "client", "name": "Bash", "input": `{"command":"pwd"}`, "stop": true,
		}))), nil
	}))
	installKiroRetryWait(t, func(time.Duration) {})

	err := callKiroForClaude(context.Background(), newKiroRetryTestAPIKeyAccount(""), payload, &KiroStreamCallback{})
	if err != nil || calls != 2 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
	if len(requests[0].ConversationState.History) != len(requests[1].ConversationState.History) ||
		requests[0].ConversationState.CurrentMessage.UserInputMessage.Content != requests[1].ConversationState.CurrentMessage.UserInputMessage.Content {
		t.Fatalf("truncated response altered retry payload: first=%#v second=%#v", requests[0].ConversationState, requests[1].ConversationState)
	}
}

func TestClaudeCompletionContractStopsAfterFourRoundsWithoutReleasingProgress(t *testing.T) {
	payload := newStrictCompletionTestPayload()
	var calls int
	installKiroStreamTestClient(t, roundTripFunc(func(*http.Request) (*http.Response, error) {
		calls++
		return kiroStreamTestResponse(bytes.NewReader(bytes.Join([][]byte{
			awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{"content": "I will continue checking."}),
			awsEventStreamStopFrame(t, "END_TURN"),
		}, nil))), nil
	}))

	var outputCallbacks, completed int
	err := callKiroForClaude(context.Background(), newKiroRetryTestAPIKeyAccount(""), payload, &KiroStreamCallback{
		OnText:     func(string, bool) { outputCallbacks++ },
		OnToolUse:  func(KiroToolUse) { outputCallbacks++ },
		OnComplete: func(int, int) { completed++ },
	})
	if !errors.Is(err, errClaudeCompletionContract) || calls != maxClaudeCompletionRounds {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
	if outputCallbacks != 0 || completed != 0 {
		t.Fatalf("unaccepted rounds leaked callbacks: output=%d completed=%d", outputCallbacks, completed)
	}
}

func TestClaudeCompletionContractAcceptsExplicitRefusalWithoutText(t *testing.T) {
	payload := newStrictCompletionTestPayload()
	installKiroStreamTestClient(t, roundTripFunc(func(*http.Request) (*http.Response, error) {
		return kiroStreamTestResponse(bytes.NewReader(bytes.Join([][]byte{
			awsEventStreamFrame(t, "reasoningContentEvent", map[string]interface{}{"text": "policy check"}),
			awsEventStreamStopFrame(t, "GUARDRAIL_INTERVENED"),
		}, nil))), nil
	}))
	var stopReason string
	var thinking string
	var completed int
	err := callKiroForClaude(context.Background(), newKiroRetryTestAPIKeyAccount(""), payload, &KiroStreamCallback{
		OnText: func(text string, isThinking bool) {
			if isThinking {
				thinking += text
			}
		},
		OnStopReason: func(reason string) { stopReason = reason },
		OnComplete:   func(int, int) { completed++ },
	})
	if err != nil || thinking != "policy check" || stopReason != "GUARDRAIL_INTERVENED" || completed != 1 {
		t.Fatalf("err=%v thinking=%q stop=%q completed=%d", err, thinking, stopReason, completed)
	}
}

func TestCallKiroAPIContextStopsOnClientCancellation(t *testing.T) {
	payload := newStrictCompletionTestPayload()
	var calls int
	installKiroStreamTestClient(t, roundTripFunc(func(req *http.Request) (*http.Response, error) {
		calls++
		<-req.Context().Done()
		return nil, req.Context().Err()
	}))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := callKiroForClaude(ctx, newKiroRetryTestAPIKeyAccount(""), payload, &KiroStreamCallback{})
	if !errors.Is(err, context.Canceled) || calls != 0 {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
}

func TestClaudeCompletionBudgetIsSharedAcrossControllerInvocations(t *testing.T) {
	var calls int
	installKiroStreamTestClient(t, roundTripFunc(func(*http.Request) (*http.Response, error) {
		calls++
		return kiroStreamTestResponse(bytes.NewReader(awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
			"toolUseId": "client", "name": "Bash", "input": `{"command":"pwd"}`, "stop": true,
		}))), nil
	}))
	ctx := withClaudeCompletionBudget(context.Background())
	for round := 0; round < maxClaudeCompletionRounds; round++ {
		if err := callKiroForClaude(ctx, newKiroRetryTestAPIKeyAccount(""), newStrictCompletionTestPayload(), &KiroStreamCallback{}); err != nil {
			t.Fatalf("round %d failed early: %v", round+1, err)
		}
	}
	err := callKiroForClaude(ctx, newKiroRetryTestAPIKeyAccount(""), newStrictCompletionTestPayload(), &KiroStreamCallback{})
	if !errors.Is(err, errClaudeCompletionContract) || calls != maxClaudeCompletionRounds {
		t.Fatalf("err=%v calls=%d", err, calls)
	}
}

func TestClaudeStreamCompletionContractExhaustionEmitsOnlyAPIError(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(bytes.Join([][]byte{
			awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{"content": "Now I will inspect the remaining implementation."}),
			awsEventStreamStopFrame(t, "END_TURN"),
		}, nil))
	}))
	defer server.Close()

	h := setupClaudeCompletionHandlerTest(t, server)
	recorder := httptest.NewRecorder()
	h.handleClaudeStream(context.Background(), recorder, newStrictCompletionTestPayload(), "claude-opus-5", false, claudeThinkingResponseOptions{}, 1, nil, "")
	body := recorder.Body.String()
	if calls != maxClaudeCompletionRounds {
		t.Fatalf("upstream calls=%d, want %d", calls, maxClaudeCompletionRounds)
	}
	if !strings.Contains(body, "event: error") || !strings.Contains(body, `"type":"api_error"`) {
		t.Fatalf("expected SSE api_error, body=%s", body)
	}
	for _, forbidden := range []string{"event: message_start", "event: content_block", "event: message_delta", "event: message_stop", "remaining implementation"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("unaccepted output leaked via %q, body=%s", forbidden, body)
		}
	}
}

func TestClaudeMixedWebSearchCompletionContractExhaustionEmitsOnlyAPIError(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(bytes.Join([][]byte{
			awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{"content": "Now I will inspect the remaining implementation."}),
			awsEventStreamStopFrame(t, "END_TURN"),
		}, nil))
	}))
	defer server.Close()

	h := setupClaudeCompletionHandlerTest(t, server)
	recorder := httptest.NewRecorder()
	req := &ClaudeRequest{
		Model:    "claude-opus-5",
		Messages: []ClaudeMessage{{Role: "user", Content: "research and complete the task"}},
		Stream:   true,
		Tools: []ClaudeTool{
			{Type: "web_search_20250305", Name: webSearchToolName, MaxUses: 2},
			{Name: "Bash", Description: "Run a command", InputSchema: map[string]interface{}{"type": "object"}},
		},
	}
	h.runWebSearchLoop(context.Background(), recorder, req, false, 1, "")

	body := recorder.Body.String()
	if calls != maxClaudeCompletionRounds {
		t.Fatalf("upstream calls=%d, want %d", calls, maxClaudeCompletionRounds)
	}
	if contentType := recorder.Header().Get("Content-Type"); !strings.HasPrefix(contentType, "text/event-stream") {
		t.Fatalf("content type=%q, want SSE", contentType)
	}
	if !strings.Contains(body, "event: error") || !strings.Contains(body, `"type":"api_error"`) {
		t.Fatalf("expected SSE api_error, body=%s", body)
	}
	for _, forbidden := range []string{"event: message_start", "event: content_block", "event: message_delta", "event: message_stop", "remaining implementation"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("unaccepted mixed-tool output leaked via %q, body=%s", forbidden, body)
		}
	}
}

func TestClaudeStreamCompletionContractReleasesOnlyAcceptedToolRound(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusOK)
		if calls == 1 {
			_, _ = w.Write(bytes.Join([][]byte{
				awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{"content": "I will inspect the remaining implementation."}),
				awsEventStreamStopFrame(t, "END_TURN"),
			}, nil))
			return
		}
		_, _ = w.Write(awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
			"toolUseId": "client", "name": "Bash", "input": `{"command":"go test ./..."}`, "stop": true,
		}))
	}))
	defer server.Close()

	h := setupClaudeCompletionHandlerTest(t, server)
	recorder := httptest.NewRecorder()
	h.handleClaudeStream(context.Background(), recorder, newStrictCompletionTestPayload(), "claude-opus-5", false, claudeThinkingResponseOptions{}, 1, nil, "")
	body := recorder.Body.String()
	if calls != 2 {
		t.Fatalf("upstream calls=%d, want 2", calls)
	}
	if strings.Contains(body, "remaining implementation") {
		t.Fatalf("discarded progress round leaked, body=%s", body)
	}
	if !strings.Contains(body, `"name":"Bash"`) || !strings.Contains(body, `"stop_reason":"tool_use"`) || !strings.Contains(body, "event: message_stop") {
		t.Fatalf("accepted tool round was not completed normally, body=%s", body)
	}
	if strings.Contains(body, "event: error") {
		t.Fatalf("successful recovery emitted error, body=%s", body)
	}
}

func newStrictCompletionTestPayload() *KiroPayload {
	payload := &KiroPayload{
		StrictCompletion:       true,
		InternalFinishToolName: defaultInternalFinishToolName,
	}
	payload.ConversationState.ChatTriggerType = "MANUAL"
	payload.ConversationState.ConversationID = "completion-test"
	payload.ConversationState.CurrentMessage.UserInputMessage = KiroUserInputMessage{
		Content: "complete the requested work",
		ModelID: "claude-opus-5",
		Origin:  "AI_EDITOR",
		UserInputMessageContext: &UserInputMessageContext{Tools: []KiroToolWrapper{
			completionTestClientTool(),
			buildInternalFinishTool(defaultInternalFinishToolName),
		}},
	}
	return payload
}

func completionTestClientTool() KiroToolWrapper {
	tool := KiroToolWrapper{}
	tool.ToolSpecification.Name = "Bash"
	tool.ToolSpecification.Description = "Run a command"
	tool.ToolSpecification.InputSchema = InputSchema{JSON: map[string]interface{}{"type": "object"}}
	return tool
}

func internalFinishFrame(t *testing.T, name, status, message string) []byte {
	t.Helper()
	return bytes.Join([][]byte{
		awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
			"toolUseId": "finish", "name": name,
			"input": map[string]interface{}{"status": status, "message": message}, "stop": true,
		}),
		awsEventStreamStopFrame(t, "END_TURN"),
	}, nil)
}

func decodeKiroRequestBody(t *testing.T, req *http.Request, target *KiroPayload) {
	t.Helper()
	raw, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("read request body: %v", err)
	}
	if err := json.Unmarshal(raw, target); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
}

func setupClaudeCompletionHandlerTest(t *testing.T, server *httptest.Server) *Handler {
	t.Helper()
	cfgFile := t.TempDir() + "/config.json"
	if err := config.Init(cfgFile); err != nil {
		t.Fatalf("config.Init: %v", err)
	}
	if err := config.AddAccount(config.Account{
		ID:          "completion-contract",
		Enabled:     true,
		AccessToken: "token-completion-contract",
		ProfileArn:  "arn:aws:codewhisperer:profile/completion-contract",
	}); err != nil {
		t.Fatalf("config.AddAccount: %v", err)
	}
	if err := config.UpdatePreferredEndpoint("kiro"); err != nil {
		t.Fatalf("set preferred endpoint: %v", err)
	}
	if err := config.UpdateEndpointFallback(false); err != nil {
		t.Fatalf("disable endpoint fallback: %v", err)
	}

	oldEndpoints := kiroEndpoints
	kiroEndpoints = []kiroEndpoint{{URL: server.URL, Origin: "AI_EDITOR", Name: "test"}}
	t.Cleanup(func() { kiroEndpoints = oldEndpoints })
	oldClient := kiroHttpStore.Load()
	kiroHttpStore.Store(&http.Client{Timeout: time.Second, Transport: &http.Transport{}})
	t.Cleanup(func() { kiroHttpStore.Store(oldClient) })

	p := accountpool.GetPool()
	p.Reload()
	return &Handler{pool: p, promptCache: newPromptCacheTracker(defaultPromptCacheTTL)}
}
