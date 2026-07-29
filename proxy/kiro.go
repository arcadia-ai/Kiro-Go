// Package proxy is the core proxy layer for the Kiro API.
// It handles streaming API calls to the Kiro backend and parses AWS Event Stream responses.
package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"kiro-go/config"
	"kiro-go/logger"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

const (
	// streamRetryBackoff spaces out a retry of a stream that died before
	// delivering any output callback. Upstream drops cluster in time, so an
	// immediate retry tends to hit the same blip.
	streamRetryBackoff = 700 * time.Millisecond

	// maxStreamAttemptsPerEndpoint includes the initial request. One retry
	// covers the observed transient drops without multiplying the existing
	// endpoint and account fallback budgets more than necessary.
	maxStreamAttemptsPerEndpoint = 2
	maxEventStreamMessageSize    = 16 * 1024 * 1024
)

var (
	errEmptyKiroStream           = errors.New("upstream stream ended before any output")
	errUpstreamTruncatedResponse = errors.New("upstream truncated response without stop reason")
	errIncompleteKiroResponse    = errUpstreamTruncatedResponse
	errIncompleteKiroToolInput   = errors.New("upstream stream ended with incomplete tool input")
	errInvalidKiroEventStream    = errors.New("invalid upstream event stream")
	errKiroEventStreamUpstream   = errors.New("upstream event stream error")
	streamRetryWait              = waitForStreamRetry
	resolveKiroEndpoints         = endpointsForAccount
)

// Endpoint configuration (auto-fallback on quota exhaustion).
type kiroEndpoint struct {
	URL       string
	Origin    string
	AmzTarget string
	Name      string
}

var kiroEndpoints = []kiroEndpoint{
	{
		URL:       "https://q.us-east-1.amazonaws.com/generateAssistantResponse",
		Origin:    "AI_EDITOR",
		AmzTarget: "",
		Name:      "Kiro IDE",
	},
	{
		URL:       "https://codewhisperer.us-east-1.amazonaws.com/generateAssistantResponse",
		Origin:    "AI_EDITOR",
		AmzTarget: "AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
		Name:      "CodeWhisperer",
	},
	{
		URL:       "https://q.us-east-1.amazonaws.com/generateAssistantResponse",
		Origin:    "AI_EDITOR",
		AmzTarget: "AmazonQDeveloperStreamingService.SendMessage",
		Name:      "AmazonQ",
	},
}

// kiroCLIEndpoint is the headless / API Key path used by Kiro CLI:
// POST https://runtime.{region}.kiro.dev/ with AWS JSON 1.0 protocol.
var kiroCLIEndpoint = kiroEndpoint{
	URL:       "https://runtime.us-east-1.kiro.dev/",
	Origin:    "KIRO_CLI",
	AmzTarget: "AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
	Name:      "Kiro CLI",
}

// Global HTTP clients, swappable at runtime to apply proxy reconfiguration without restart.
var kiroHttpStore atomic.Pointer[http.Client]
var kiroRestHttpStore atomic.Pointer[http.Client]

// proxyClientCache caches http.Client instances keyed by proxy URL for per-account proxy support.
var proxyClientCache sync.Map

func init() {
	InitKiroHttpClient("")
}

// GetClientForProxy returns an http.Client configured for the given proxy URL.
// If proxyURL is empty, returns the global kiro HTTP client.
func GetClientForProxy(proxyURL string) *http.Client {
	if proxyURL == "" {
		return kiroHttpStore.Load()
	}
	if cached, ok := proxyClientCache.Load(proxyURL); ok {
		return cached.(*http.Client)
	}
	client := &http.Client{
		Timeout:   5 * time.Minute,
		Transport: buildKiroTransport(proxyURL),
	}
	proxyClientCache.Store(proxyURL, client)
	return client
}

// GetRestClientForProxy returns a rest http.Client (30s timeout) for the given proxy URL.
// If proxyURL is empty, returns the global kiro REST HTTP client.
func GetRestClientForProxy(proxyURL string) *http.Client {
	if proxyURL == "" {
		return kiroRestHttpStore.Load()
	}
	cacheKey := "rest:" + proxyURL
	if cached, ok := proxyClientCache.Load(cacheKey); ok {
		return cached.(*http.Client)
	}
	client := &http.Client{
		Timeout:   30 * time.Second,
		Transport: buildKiroTransport(proxyURL),
	}
	proxyClientCache.Store(cacheKey, client)
	return client
}

// ResolveAccountProxyURL returns the effective proxy URL for an account.
// Falls back to global config.GetProxyURL() if the account has no per-account proxy.
func ResolveAccountProxyURL(account *config.Account) string {
	if account != nil && account.ProxyURL != "" {
		return account.ProxyURL
	}
	return config.GetProxyURL()
}

// buildKiroTransport constructs an HTTP Transport with optional outbound proxy support.
func buildKiroTransport(proxyURL string) *http.Transport {
	t := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 20,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression:  false,
		ForceAttemptHTTP2:   true,
	}
	if proxyURL != "" {
		if u, err := url.Parse(proxyURL); err == nil {
			t.Proxy = http.ProxyURL(u)
			// Proxied connections cannot negotiate HTTP/2.
			t.ForceAttemptHTTP2 = false
		}
	} else {
		t.Proxy = http.ProxyFromEnvironment
	}
	return t
}

// InitKiroHttpClient initializes (or reinitializes) the HTTP clients used for Kiro API requests.
func InitKiroHttpClient(proxyURL string) {
	client := &http.Client{
		Timeout:   5 * time.Minute,
		Transport: buildKiroTransport(proxyURL),
	}
	kiroHttpStore.Store(client)

	restClient := &http.Client{
		Timeout:   30 * time.Second,
		Transport: buildKiroTransport(proxyURL),
	}
	kiroRestHttpStore.Store(restClient)
}

// ==================== Request Structs ====================

// KiroPayload is the top-level request body sent to the Kiro API.
type KiroPayload struct {
	ConversationState struct {
		AgentContinuationId string `json:"agentContinuationId,omitempty"`
		AgentTaskType       string `json:"agentTaskType,omitempty"`
		ChatTriggerType     string `json:"chatTriggerType"`
		ConversationID      string `json:"conversationId"`
		CurrentMessage      struct {
			UserInputMessage KiroUserInputMessage `json:"userInputMessage"`
		} `json:"currentMessage"`
		History []KiroHistoryMessage `json:"history,omitempty"`
	} `json:"conversationState"`
	ProfileArn      string           `json:"profileArn,omitempty"`
	InferenceConfig *InferenceConfig `json:"inferenceConfig,omitempty"`

	// ToolNameMap maps sanitized tool names (sent to Kiro) back to the
	// original names supplied by the client. Used to restore original names
	// in tool_use responses so the client can match them to its tool registry.
	// Not serialized to the Kiro API request body.
	ToolNameMap            map[string]string `json:"-"`
	StrictCompletion       bool              `json:"-"`
	InternalFinishToolName string            `json:"-"`
}

type KiroUserInputMessage struct {
	Content                 string                   `json:"content"`
	ModelID                 string                   `json:"modelId,omitempty"`
	Origin                  string                   `json:"origin"`
	Images                  []KiroImage              `json:"images,omitempty"`
	UserInputMessageContext *UserInputMessageContext `json:"userInputMessageContext,omitempty"`
}

type UserInputMessageContext struct {
	Tools       []KiroToolWrapper `json:"tools,omitempty"`
	ToolResults []KiroToolResult  `json:"toolResults,omitempty"`
}

type KiroToolWrapper struct {
	ToolSpecification struct {
		Name        string      `json:"name"`
		Description string      `json:"description"`
		InputSchema InputSchema `json:"inputSchema"`
	} `json:"toolSpecification"`
}

type InputSchema struct {
	JSON interface{} `json:"json"`
}

type KiroToolResult struct {
	ToolUseID string              `json:"toolUseId"`
	Content   []KiroResultContent `json:"content"`
	Status    string              `json:"status"`
}

type KiroResultContent struct {
	Text string `json:"text"`
}

type KiroImage struct {
	Format string `json:"format"`
	Source struct {
		Bytes string `json:"bytes"`
	} `json:"source"`
}

type KiroHistoryMessage struct {
	UserInputMessage         *KiroUserInputMessage         `json:"userInputMessage,omitempty"`
	AssistantResponseMessage *KiroAssistantResponseMessage `json:"assistantResponseMessage,omitempty"`
}

type KiroAssistantResponseMessage struct {
	Content  string        `json:"content"`
	ToolUses []KiroToolUse `json:"toolUses,omitempty"`
}

type KiroToolUse struct {
	ToolUseID string                 `json:"toolUseId"`
	Name      string                 `json:"name"`
	Input     map[string]interface{} `json:"input"`
}

type InferenceConfig struct {
	MaxTokens   int     `json:"maxTokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	TopP        float64 `json:"topP,omitempty"`
}

// ==================== Stream Callbacks ====================

// KiroStreamCallback stream response callbacks
type KiroStreamCallback struct {
	OnText         func(text string, isThinking bool)
	OnToolUse      func(toolUse KiroToolUse)
	OnProgress     func()
	OnComplete     func(inputTokens, outputTokens int)
	OnError        func(err error)
	OnCredits      func(credits float64)
	OnContextUsage func(percentage float64)
	OnStopReason   func(reason string)
}

type bufferedKiroOutput struct {
	text       string
	isThinking bool
	toolUse    *KiroToolUse
}

// kiroStreamDiagnostics captures stream shape without retaining response
// content. It is logged once per upstream attempt to diagnose silent
// truncation while keeping prompts, model output, and credentials private.
type kiroStreamDiagnostics struct {
	FrameCount         int
	LastEventType      string
	AssistantEvents    int
	AssistantBytes     int
	ReasoningEvents    int
	ReasoningBytes     int
	ToolEvents         int
	ToolInputBytes     int
	CompletedToolUses  int
	MeteringEvents     int
	ContextUsageEvents int
	MetadataEvents     int
	StopReason         string
	Integrity          string
	OutputsReleased    bool
	PendingToolUse     bool
}

func (d kiroStreamDiagnostics) summary() string {
	return fmt.Sprintf(
		"frames=%d last_event=%q assistant_events=%d assistant_bytes=%d reasoning_events=%d reasoning_bytes=%d tool_events=%d tool_input_bytes=%d completed_tools=%d pending_tool=%t metering_events=%d context_usage_events=%d metadata_events=%d stop_reason=%q integrity=%q outputs_released=%t",
		d.FrameCount,
		d.LastEventType,
		d.AssistantEvents,
		d.AssistantBytes,
		d.ReasoningEvents,
		d.ReasoningBytes,
		d.ToolEvents,
		d.ToolInputBytes,
		d.CompletedToolUses,
		d.PendingToolUse,
		d.MeteringEvents,
		d.ContextUsageEvents,
		d.MetadataEvents,
		d.StopReason,
		d.Integrity,
		d.OutputsReleased,
	)
}

type assistantCompletionDetector struct {
	inThinking bool
	hasText    bool
	pending    string
}

func (d *assistantCompletionDetector) Add(content string) bool {
	const (
		openTag  = "<thinking>"
		closeTag = "</thinking>"
	)

	if d.hasText {
		return true
	}
	d.pending += content

	for len(d.pending) > 0 {
		if d.inThinking {
			end := strings.Index(d.pending, closeTag)
			if end == -1 {
				d.pending = tagPrefixSuffix(d.pending, closeTag)
				return false
			}
			d.pending = d.pending[end+len(closeTag):]
			d.inThinking = false
			continue
		}

		start := strings.Index(d.pending, openTag)
		if start >= 0 {
			if strings.TrimSpace(d.pending[:start]) != "" {
				d.hasText = true
				d.pending = ""
				return true
			}
			d.pending = d.pending[start+len(openTag):]
			d.inThinking = true
			continue
		}

		tagPrefix := tagPrefixSuffix(d.pending, openTag)
		visibleEnd := len(d.pending) - len(tagPrefix)
		if strings.TrimSpace(d.pending[:visibleEnd]) != "" {
			d.hasText = true
			d.pending = ""
			return true
		}
		d.pending = tagPrefix
		return false
	}
	return false
}

func tagPrefixSuffix(content, tag string) string {
	maxLength := min(len(content), len(tag)-1)
	for length := maxLength; length > 0; length-- {
		if strings.HasSuffix(content, tag[:length]) {
			return content[len(content)-length:]
		}
	}
	return ""
}

// ==================== API Call ====================

func setPayloadProfileArnForAccount(payload *KiroPayload, account *config.Account) {
	if payload == nil {
		return
	}

	// API Key credentials must not carry IDE/profile semantics.
	if config.IsAPIKeyAccount(account) {
		payload.ProfileArn = ""
		return
	}

	payload.ProfileArn = strings.TrimSpace(payload.ProfileArn)
	if account != nil {
		if profileArn := strings.TrimSpace(account.ProfileArn); profileArn != "" {
			payload.ProfileArn = profileArn
		}
	}
}

// endpointsForAccount returns the upstream endpoint list for a credential.
// API Key accounts always use the CLI runtime protocol; OAuth accounts keep
// the configured preferred-endpoint fallback chain.
func endpointsForAccount(account *config.Account) []kiroEndpoint {
	if config.IsAPIKeyAccount(account) {
		return []kiroEndpoint{kiroCLIEndpoint}
	}
	return getSortedEndpoints(config.GetPreferredEndpoint())
}

// cliRuntimeURL builds the regional Kiro CLI runtime URL.
func cliRuntimeURL(account *config.Account) string {
	region := "us-east-1"
	if account != nil {
		if r := strings.TrimSpace(account.Region); r != "" {
			region = r
		}
	}
	return fmt.Sprintf("https://runtime.%s.kiro.dev/", region)
}

// getSortedEndpoints returns endpoints ordered by user preference, with optional fallback.
func getSortedEndpoints(preferred string) []kiroEndpoint {
	fallback := config.GetEndpointFallback()

	var primary int
	switch preferred {
	case "kiro":
		primary = 0
	case "codewhisperer":
		primary = 1
	case "amazonq":
		primary = 2
	default:
		// "auto": Kiro first, then fallback to others
		return []kiroEndpoint{kiroEndpoints[0], kiroEndpoints[1], kiroEndpoints[2]}
	}

	if !fallback {
		// No fallback: only use the selected endpoint
		return []kiroEndpoint{kiroEndpoints[primary]}
	}

	// With fallback: selected first, then others in order
	result := []kiroEndpoint{kiroEndpoints[primary]}
	for i, ep := range kiroEndpoints {
		if i != primary {
			result = append(result, ep)
		}
	}
	return result
}

// CallKiroAPI calls the Kiro streaming API, trying each configured endpoint with automatic fallback.
func CallKiroAPI(account *config.Account, payload *KiroPayload, callback *KiroStreamCallback) error {
	return CallKiroAPIContext(context.Background(), account, payload, callback)
}

// CallKiroAPIContext calls Kiro while respecting cancellation from the client request.
func CallKiroAPIContext(ctx context.Context, account *config.Account, payload *KiroPayload, callback *KiroStreamCallback) error {
	return callKiroAPIContext(ctx, account, payload, callback, nil)
}

func callKiroAPIContext(ctx context.Context, account *config.Account, payload *KiroPayload, callback *KiroStreamCallback, beforeAttempt func() error) error {
	if ctx == nil {
		ctx = context.Background()
	}
	originalProfileArn := ""
	if payload != nil {
		originalProfileArn = payload.ProfileArn
		defer func() {
			payload.ProfileArn = originalProfileArn
		}()
	}
	setPayloadProfileArnForAccount(payload, account)

	if _, err := json.Marshal(payload); err != nil {
		return err
	}

	// Debug: dump full payload for troubleshooting upstream rejections
	if payloadJSON, err := json.Marshal(payload); err == nil {
		logger.Debugf("[KiroAPI] Request payload: %s", string(payloadJSON))
	}

	// Wrap OnToolUse to restore original tool names for the client.
	if callback != nil && callback.OnToolUse != nil && len(payload.ToolNameMap) > 0 {
		originalOnToolUse := callback.OnToolUse
		nameMap := payload.ToolNameMap
		wrapped := *callback
		wrapped.OnToolUse = func(tu KiroToolUse) {
			if original, ok := nameMap[tu.Name]; ok {
				tu.Name = original
			}
			originalOnToolUse(tu)
		}
		callback = &wrapped
	}

	if payload != nil && strings.TrimSpace(payload.ProfileArn) == "" && !config.IsAPIKeyAccount(account) {
		if profileArn, err := ResolveProfileArn(account); err == nil {
			payload.ProfileArn = profileArn
		} else if isProfileArnResolutionSoftError(err) {
			logger.Debugf("[ProfileArn] Skipped profile ARN resolution for %s: %v", accountEmailForLog(account), err)
		} else {
			logger.Warnf("[ProfileArn] Failed to resolve profile ARN for %s: %v", accountEmailForLog(account), err)
		}
	}

	// Build endpoint list ordered by configuration / credential type.
	endpoints := resolveKiroEndpoints(account)
	isAPIKey := config.IsAPIKeyAccount(account)

	var lastErr error
endpointLoop:
	for epIndex, ep := range endpoints {
		if err := ctx.Err(); err != nil {
			return err
		}
		// Update the origin field for the selected endpoint.
		payload.ConversationState.CurrentMessage.UserInputMessage.Origin = ep.Origin

		// Target the profile's data-plane region; endpoint URLs are declared for us-east-1.
		// API Key accounts use the CLI runtime host instead of IDE/Q hosts.
		epURL := regionalizeURLForProfile(ep.URL, account, payload.ProfileArn)
		if isAPIKey {
			epURL = cliRuntimeURL(account)
		}

		reqBody, _ := json.Marshal(payload)
		host := ""
		if parsedURL, parseErr := url.Parse(epURL); parseErr == nil {
			host = parsedURL.Host
		}
		headerValues := buildStreamingHeaderValues(account, host)
		invocationID := uuid.New().String()

		for streamAttempt := 1; streamAttempt <= maxStreamAttemptsPerEndpoint; streamAttempt++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			if beforeAttempt != nil {
				if err := beforeAttempt(); err != nil {
					if lastErr != nil {
						return fmt.Errorf("%w: last upstream error: %v", err, lastErr)
					}
					return err
				}
			}
			// Requests and bodies cannot be reused after an HTTP attempt.
			req, err := http.NewRequestWithContext(ctx, "POST", epURL, bytes.NewReader(reqBody))
			if err != nil {
				lastErr = err
				continue endpointLoop
			}

			if isAPIKey {
				req.Header.Set("Content-Type", "application/x-amz-json-1.0")
			} else {
				req.Header.Set("Content-Type", "application/json")
			}
			req.Header.Set("Accept", "*/*")
			if ep.AmzTarget != "" {
				req.Header.Set("X-Amz-Target", ep.AmzTarget)
			}
			applyKiroBaseHeaders(req, account, headerValues)
			if !isAPIKey {
				req.Header.Set("x-amzn-kiro-agent-mode", "vibe")
			}
			// CLI captures use optout=false; IDE path keeps true.
			if isAPIKey {
				req.Header.Set("x-amzn-codewhisperer-optout", "false")
			} else {
				req.Header.Set("x-amzn-codewhisperer-optout", "true")
			}
			req.Header.Set("Amz-Sdk-Request", fmt.Sprintf("attempt=%d; max=%d", streamAttempt, maxStreamAttemptsPerEndpoint))
			req.Header.Set("Amz-Sdk-Invocation-Id", invocationID)

			resp, err := GetClientForProxy(ResolveAccountProxyURL(account)).Do(req)
			if err != nil {
				if ctxErr := ctx.Err(); ctxErr != nil {
					return ctxErr
				}
				lastErr = err
				logger.Warnf("[KiroAPI] Endpoint %s failed: %v", ep.Name, err)
				if !isRetryableStreamError(err) {
					return err
				}
				continue endpointLoop
			}

			if resp.StatusCode == 429 {
				resp.Body.Close()
				logger.Warnf("[KiroAPI] Endpoint %s quota exhausted (429), trying next...", ep.Name)
				lastErr = fmt.Errorf("quota exhausted on %s", ep.Name)
				continue endpointLoop
			}

			if resp.StatusCode != 200 {
				errBody, _ := io.ReadAll(resp.Body)
				resp.Body.Close()
				lastErr = fmt.Errorf("HTTP %d from %s: %s", resp.StatusCode, ep.Name, string(errBody))
				// Authentication errors and payment errors are not retried across endpoints.
				if resp.StatusCode == 401 || resp.StatusCode == 403 || resp.StatusCode == 402 {
					return lastErr
				}
				logger.Warnf("[KiroAPI] Endpoint %s error: %v", ep.Name, lastErr)
				continue endpointLoop
			}

			var diagnostics kiroStreamDiagnostics
			emitted, err := parseEventStreamTrackedWithDiagnostics(resp.Body, callback, &diagnostics)
			resp.Body.Close()
			if err == nil {
				logger.Infof("[KiroStream] endpoint=%q invocation_id=%s attempt=%d/%d result=complete emitted=%t %s",
					ep.Name, invocationID, streamAttempt, maxStreamAttemptsPerEndpoint, emitted, diagnostics.summary())
				return nil
			}
			logger.Warnf("[KiroStream] endpoint=%q invocation_id=%s attempt=%d/%d result=error emitted=%t error=%q %s",
				ep.Name, invocationID, streamAttempt, maxStreamAttemptsPerEndpoint, emitted, err.Error(), diagnostics.summary())
			lastErr = err
			if ctxErr := ctx.Err(); ctxErr != nil {
				return ctxErr
			}
			// "Emitted" deliberately means that an output callback ran. This
			// conservative boundary also protects buffered/non-stream callers:
			// retrying after their callback mutated state would concatenate two
			// attempts even if no network bytes were flushed yet.
			if emitted {
				return err
			}
			if !isRetryableStreamError(err) {
				return err
			}

			hasSameEndpointRetry := streamAttempt < maxStreamAttemptsPerEndpoint
			hasEndpointFallback := epIndex+1 < len(endpoints)
			if !hasSameEndpointRetry && !hasEndpointFallback {
				if errors.Is(err, errIncompleteKiroResponse) {
					logger.Warnf("[KiroAPI] Endpoint %s reasoning-only response retries exhausted (attempt %d/%d)",
						ep.Name, streamAttempt, maxStreamAttemptsPerEndpoint)
				}
				break endpointLoop
			}

			if errors.Is(err, errIncompleteKiroResponse) {
				logger.Warnf("[KiroAPI] Endpoint %s returned a reasoning-only response before output; retrying (attempt %d/%d)",
					ep.Name, streamAttempt, maxStreamAttemptsPerEndpoint)
			} else {
				logger.Warnf("[KiroAPI] Endpoint %s stream failed before any output callback (attempt %d/%d): %v",
					ep.Name, streamAttempt, maxStreamAttemptsPerEndpoint, err)
			}
			if err := streamRetryWait(ctx, streamRetryBackoff); err != nil {
				return err
			}
			if !hasSameEndpointRetry {
				continue endpointLoop
			}
		}
	}

	if lastErr != nil {
		return lastErr
	}
	return fmt.Errorf("all endpoints failed")
}

func waitForStreamRetry(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func isRetryableStreamError(err error) bool {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var netErr net.Error
	return !errors.As(err, &netErr) || !netErr.Timeout()
}

func accountEmailForLog(account *config.Account) string {
	if account == nil {
		return "<nil>"
	}
	return account.Email
}

// ==================== Event Stream Parsing ====================

// parseEventStream decodes an AWS binary Event Stream response body.
func parseEventStream(body io.Reader, callback *KiroStreamCallback) error {
	_, err := parseEventStreamTracked(body, callback)
	return err
}

// parseEventStreamTracked is parseEventStream plus a report of whether an
// output callback was invoked. A failure before the first callback is safe to
// retry because it cannot duplicate or concatenate caller state.
func parseEventStreamTracked(body io.Reader, callback *KiroStreamCallback) (emitted bool, err error) {
	return parseEventStreamTrackedWithDiagnostics(body, callback, nil)
}

func parseEventStreamTrackedWithDiagnostics(body io.Reader, callback *KiroStreamCallback, diagnostics *kiroStreamDiagnostics) (emitted bool, err error) {
	if callback == nil {
		callback = &KiroStreamCallback{}
	}
	if diagnostics == nil {
		diagnostics = &kiroStreamDiagnostics{}
	} else {
		*diagnostics = kiroStreamDiagnostics{}
	}

	// Read directly without bufio to avoid buffering latency in streaming responses.
	var inputTokens, outputTokens int
	var totalCredits float64
	var contextUsagePercentages []float64
	var sawOutput bool
	var sawReasoning bool
	var sawAssistantContent bool
	var sawToolUse bool
	var stopReason string
	var completionDetector assistantCompletionDetector
	pendingTools := &pendingToolUses{}
	var pendingToolUse bool
	var pendingOutputs []bufferedKiroOutput
	outputsReleased := false
	defer func() {
		diagnostics.OutputsReleased = outputsReleased
		diagnostics.PendingToolUse = pendingToolUse
		switch {
		case err == nil:
			diagnostics.Integrity = "valid"
		case isKiroStreamIntegrityError(err):
			diagnostics.Integrity = "invalid"
		default:
			diagnostics.Integrity = "error"
		}
	}()

	deliverOutput := func(output bufferedKiroOutput) {
		if output.toolUse != nil {
			if callback.OnToolUse != nil {
				emitted = true
				callback.OnToolUse(*output.toolUse)
			}
			return
		}
		if callback.OnText != nil {
			emitted = true
			callback.OnText(output.text, output.isThinking)
		}
	}
	releasePendingOutputs := func() {
		if outputsReleased {
			return
		}
		outputsReleased = true
		for _, output := range pendingOutputs {
			deliverOutput(output)
		}
		pendingOutputs = nil
	}
	queueOutput := func(output bufferedKiroOutput) {
		if outputsReleased {
			deliverOutput(output)
			return
		}
		pendingOutputs = append(pendingOutputs, output)
	}

	trackedCallback := *callback
	trackedCallback.OnToolUse = func(toolUse KiroToolUse) {
		sawOutput = true
		sawToolUse = true
		pendingToolUse = false
		diagnostics.CompletedToolUses++
		toolUseCopy := toolUse
		queueOutput(bufferedKiroOutput{toolUse: &toolUseCopy})
		releasePendingOutputs()
	}
	toolCallback := &trackedCallback

	for {
		// Prelude: total length, headers length, and prelude CRC.
		prelude := make([]byte, 12)
		_, readErr := io.ReadFull(body, prelude)
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return emitted, readErr
		}

		totalLength := int(eventStreamUint32(prelude[0:4]))
		headersLength := int(eventStreamUint32(prelude[4:8]))
		if totalLength < 16 || totalLength > maxEventStreamMessageSize {
			return emitted, fmt.Errorf("%w: invalid frame length %d", errInvalidKiroEventStream, totalLength)
		}
		if headersLength > totalLength-16 {
			return emitted, fmt.Errorf("%w: invalid headers length %d", errInvalidKiroEventStream, headersLength)
		}
		if got, want := eventStreamUint32(prelude[8:12]), crc32.ChecksumIEEE(prelude[:8]); got != want {
			return emitted, fmt.Errorf("%w: prelude CRC mismatch", errInvalidKiroEventStream)
		}

		message := make([]byte, totalLength-len(prelude))
		if _, err = io.ReadFull(body, message); err != nil {
			return emitted, err
		}
		if got, want := eventStreamUint32(message[len(message)-4:]), eventStreamChecksum(prelude, message[:len(message)-4]); got != want {
			return emitted, fmt.Errorf("%w: message CRC mismatch", errInvalidKiroEventStream)
		}

		headers, headerErr := parseEventStreamHeaders(message[:headersLength])
		if headerErr != nil {
			return emitted, fmt.Errorf("%w: %v", errInvalidKiroEventStream, headerErr)
		}
		eventType := headers[":event-type"]
		diagnostics.FrameCount++
		diagnostics.LastEventType = eventType
		if callback.OnProgress != nil {
			callback.OnProgress()
		}
		payloadBytes := message[headersLength : len(message)-4]
		event := make(map[string]interface{})
		if len(payloadBytes) > 0 {
			if err := json.Unmarshal(payloadBytes, &event); err != nil {
				return emitted, fmt.Errorf("%w: decode payload: %v", errInvalidKiroEventStream, err)
			}
		}
		if messageType := headers[":message-type"]; messageType == "error" || messageType == "exception" {
			detail := firstStringField(event, "message", "errorMessage", "error")
			if detail == "" {
				detail = messageType
			}
			return emitted, fmt.Errorf("%w: %s", errKiroEventStreamUpstream, detail)
		}

		inputTokens, outputTokens = updateTokensFromEvent(event, inputTokens, outputTokens)

		// Dispatch by event type.
		switch eventType {
		// Both text streams are preserved verbatim. Output is held only until a
		// final assistant response or complete tool use proves the attempt is usable.
		// Kiro sends
		// assistantResponseEvent and reasoningContentEvent as pure incremental deltas
		// (verified against real upstream traffic), never as cumulative snapshots, and
		// never replays a chunk: ordering and at-most-once delivery are already
		// guaranteed by TCP, and a dropped stream is retried as a whole new request
		// rather than resumed.
		//
		// Do NOT reintroduce content-based de-duplication here. At the string level a
		// replayed chunk is indistinguishable from text that simply repeats itself, and
		// the wire protocol carries no sequence number or message id to tell them apart
		// (the AWS event-stream base spec defines none), so such a heuristic can only
		// guess -- and when it guesses wrong it silently eats real output. The previous
		// implementation turned "6666666666" into "666", "abababab" into "abab" and
		// "1833" into "183", on both streams.
		case "assistantResponseEvent":
			diagnostics.AssistantEvents++
			if content, ok := event["content"].(string); ok && content != "" {
				diagnostics.AssistantBytes += len(content)
				sawOutput = true
				sawAssistantContent = true
				queueOutput(bufferedKiroOutput{text: content})
				if completionDetector.Add(content) {
					releasePendingOutputs()
				}
			}
		case "reasoningContentEvent":
			diagnostics.ReasoningEvents++
			if text, ok := event["text"].(string); ok && text != "" {
				diagnostics.ReasoningBytes += len(text)
				sawOutput = true
				sawReasoning = true
				queueOutput(bufferedKiroOutput{text: text, isThinking: true})
			}
		case "toolUseEvent":
			diagnostics.ToolEvents++
			pendingToolUse = true
			if input, ok := event["input"].(string); ok {
				diagnostics.ToolInputBytes += len(input)
			} else if input, ok := event["input"].(map[string]interface{}); ok {
				if data, marshalErr := json.Marshal(input); marshalErr == nil {
					diagnostics.ToolInputBytes += len(data)
				}
			}
			if toolErr := handleToolUseEvent(event, pendingTools, toolCallback); toolErr != nil {
				return emitted, toolErr
			}
			pendingToolUse = pendingTools.len() > 0
		case "meteringEvent":
			diagnostics.MeteringEvents++
			if usage, ok := event["usage"].(float64); ok {
				totalCredits += usage
			}
		case "contextUsageEvent":
			diagnostics.ContextUsageEvents++
			if pct, ok := event["contextUsagePercentage"].(float64); ok {
				contextUsagePercentages = append(contextUsagePercentages, pct)
			}
		case "metadataEvent":
			diagnostics.MetadataEvents++
			if reason := firstStringField(event, "stopReason", "stop_reason"); reason != "" {
				stopReason = reason
				diagnostics.StopReason = reason
				if callback.OnStopReason != nil {
					callback.OnStopReason(reason)
				}
			}
		}
	}

	if pendingTools.len() > 0 {
		if err := pendingTools.flushAll(toolCallback); err != nil {
			return emitted, err
		}
		pendingToolUse = false
	}

	if !sawOutput && !isExplicitKiroTerminalReason(stopReason) {
		return emitted, errEmptyKiroStream
	}

	hasFinalAssistantText := completionDetector.hasText
	if !hasFinalAssistantText && !sawToolUse && (sawReasoning || sawAssistantContent) && !isExplicitKiroTerminalReason(stopReason) {
		return emitted, errIncompleteKiroResponse
	}
	if !sawToolUse && strings.TrimSpace(stopReason) == "" {
		return emitted, errUpstreamTruncatedResponse
	}
	if !sawToolUse && !isKnownKiroStopReason(stopReason) {
		return emitted, fmt.Errorf("%w: unknown stop reason %q", errInvalidKiroEventStream, stopReason)
	}
	releasePendingOutputs()

	if callback.OnCredits != nil && totalCredits > 0 {
		callback.OnCredits(totalCredits)
	}

	if callback.OnContextUsage != nil {
		for _, percentage := range contextUsagePercentages {
			callback.OnContextUsage(percentage)
		}
	}

	if callback.OnComplete != nil {
		callback.OnComplete(inputTokens, outputTokens)
	}
	return emitted, nil
}

func updateTokensFromEvent(event map[string]interface{}, currentInputTokens, currentOutputTokens int) (int, int) {
	candidates := []map[string]interface{}{event}
	collectUsageMaps(event, &candidates)

	inputTokens := currentInputTokens
	outputTokens := currentOutputTokens

	for _, usage := range candidates {
		if usage == nil {
			continue
		}

		if v, ok := readTokenNumber(usage,
			"outputTokens", "completionTokens", "totalOutputTokens",
			"output_tokens", "completion_tokens", "total_output_tokens",
		); ok {
			outputTokens = v
		}

		if v, ok := readTokenNumber(usage,
			"inputTokens", "promptTokens", "totalInputTokens",
			"input_tokens", "prompt_tokens", "total_input_tokens",
		); ok {
			inputTokens = v
			continue
		}

		uncached, _ := readTokenNumber(usage, "uncachedInputTokens", "uncached_input_tokens")
		cacheRead, _ := readTokenNumber(usage, "cacheReadInputTokens", "cache_read_input_tokens")
		cacheWrite, _ := readTokenNumber(usage, "cacheWriteInputTokens", "cache_write_input_tokens", "cacheCreationInputTokens", "cache_creation_input_tokens")
		if uncached+cacheRead+cacheWrite > 0 {
			inputTokens = uncached + cacheRead + cacheWrite
			continue
		}

		total, ok := readTokenNumber(usage, "totalTokens", "total_tokens")
		if ok && total > 0 {
			candidateOutput := outputTokens
			if v, vok := readTokenNumber(usage,
				"outputTokens", "completionTokens", "totalOutputTokens",
				"output_tokens", "completion_tokens", "total_output_tokens",
			); vok {
				candidateOutput = v
			}
			if total-candidateOutput > 0 {
				inputTokens = total - candidateOutput
			}
		}
	}

	return inputTokens, outputTokens
}

// getContextWindowSize returns the context window size (in tokens) for a model.
//
// Per Kiro's ListAvailableModels, the 1M-token context window applies to
// Claude 4.6 and newer (sonnet-4.6, opus-4.6, opus-4.7, opus-4.8, and future
// 4.x releases), while 4.5 and earlier (opus-4.5, sonnet-4.5, sonnet-4,
// haiku-4.5) use a 200K window. This value is used to convert the upstream
// contextUsagePercentage into an absolute input-token count that clients rely
// on to decide when to compact; an undersized window under-reports tokens and
// prevents clients from compacting in time.
func getContextWindowSize(model string) int {
	if isLargeContextModel(model) {
		return 1_000_000
	}
	return 200_000
}

// claudeVersionExtractor matches "claude-<family>-<major>[.<minor>]" (dot or
// dash form) and is used to classify 1M-window models by version. The minor
// component is optional so major-only identifiers such as "claude-opus-5"
// classify correctly instead of falling through to the 200K default.
var claudeVersionExtractor = regexp.MustCompile(`claude-(?:opus|sonnet|haiku)-(\d+)(?:[.-](\d+))?`)

func isLargeContextModel(model string) bool {
	m := strings.ToLower(model)
	if match := claudeVersionExtractor.FindStringSubmatch(m); match != nil {
		major, errMaj := strconv.Atoi(match[1])
		if errMaj == nil {
			// 1M window for any major >= 5 (claude-opus-5, claude-opus-5.1, ...).
			if major > 4 {
				return true
			}
			// Within Claude 4.x the window depends on the minor version, so an
			// absent minor (claude-sonnet-4) is treated as 4.0 -> 200K.
			minor := 0
			if match[2] != "" {
				parsed, errMin := strconv.Atoi(match[2])
				if errMin != nil {
					return false
				}
				minor = parsed
			}
			return major == 4 && minor >= 6
		}
	}
	// Fallback substring checks for non-standard identifiers.
	for _, tag := range []string{"4.6", "4-6", "4.7", "4-7", "4.8", "4-8", "4.9", "4-9"} {
		if strings.Contains(m, tag) {
			return true
		}
	}
	return false
}

func collectUsageMaps(v interface{}, out *[]map[string]interface{}) {
	switch t := v.(type) {
	case map[string]interface{}:
		for k, child := range t {
			lk := strings.ToLower(k)
			if lk == "usage" || lk == "tokenusage" || lk == "token_usage" {
				if m, ok := child.(map[string]interface{}); ok {
					*out = append(*out, m)
				}
			}
			collectUsageMaps(child, out)
		}
	case []interface{}:
		for _, child := range t {
			collectUsageMaps(child, out)
		}
	}
}

func readTokenNumber(m map[string]interface{}, keys ...string) (int, bool) {
	for _, k := range keys {
		v, ok := m[k]
		if !ok {
			continue
		}
		switch n := v.(type) {
		case float64:
			return int(n), true
		case int:
			return n, true
		case int64:
			return int(n), true
		case json.Number:
			if parsed, err := n.Int64(); err == nil {
				return int(parsed), true
			}
		case string:
			if parsed, err := strconv.Atoi(n); err == nil {
				return parsed, true
			}
			if parsed, err := strconv.ParseFloat(n, 64); err == nil {
				return int(parsed), true
			}
		}
	}
	return 0, false
}

// ==================== Tool Use Handling ====================

type toolUseState struct {
	ToolUseID   string
	Name        string
	InputBuffer strings.Builder
	GeneratedID bool
}

// pendingToolUses keeps interleaved tool fragments independent while preserving
// arrival order for clients that execute calls sequentially.
type pendingToolUses struct {
	byID   map[string]*toolUseState
	order  []string
	lastID string
}

func (p *pendingToolUses) len() int {
	return len(p.order)
}

func (p *pendingToolUses) get(id string) *toolUseState {
	if p.byID == nil {
		return nil
	}
	return p.byID[id]
}

func (p *pendingToolUses) add(state *toolUseState) {
	if p.byID == nil {
		p.byID = make(map[string]*toolUseState)
	}
	p.byID[state.ToolUseID] = state
	p.order = append(p.order, state.ToolUseID)
	p.lastID = state.ToolUseID
}

func (p *pendingToolUses) rekey(state *toolUseState, newID string) {
	oldID := state.ToolUseID
	delete(p.byID, oldID)
	for i, id := range p.order {
		if id == oldID {
			p.order[i] = newID
			break
		}
	}
	state.ToolUseID = newID
	state.GeneratedID = false
	p.byID[newID] = state
	if p.lastID == oldID {
		p.lastID = newID
	}
}

func (p *pendingToolUses) remove(id string) {
	delete(p.byID, id)
	for i, existing := range p.order {
		if existing == id {
			p.order = append(p.order[:i], p.order[i+1:]...)
			break
		}
	}
	if p.lastID == id {
		p.lastID = ""
		if len(p.order) > 0 {
			p.lastID = p.order[len(p.order)-1]
		}
	}
}

func (p *pendingToolUses) flushAll(callback *KiroStreamCallback) error {
	order := p.order
	byID := p.byID
	p.byID = nil
	p.order = nil
	p.lastID = ""
	for _, id := range order {
		state := byID[id]
		if state == nil {
			continue
		}
		if err := finishToolUse(state, callback); err != nil {
			return err
		}
	}
	return nil
}

func handleToolUseEvent(event map[string]interface{}, pending *pendingToolUses, callback *KiroStreamCallback) error {
	toolUseID := firstStringField(event, "toolUseId", "toolUseID", "tool_use_id", "id")
	name := firstStringField(event, "name", "toolName", "tool_name")
	isStop := firstBoolField(event, "stop", "isStop", "done")

	var state *toolUseState
	switch {
	case toolUseID != "":
		state = pending.get(toolUseID)
		if state == nil && pending.lastID != "" {
			if previous := pending.get(pending.lastID); previous != nil && previous.GeneratedID && (name == "" || previous.Name == name) {
				pending.rekey(previous, toolUseID)
				state = previous
			}
		}
		if state == nil {
			if name == "" {
				return nil
			}
			state = &toolUseState{ToolUseID: toolUseID, Name: name}
			pending.add(state)
		} else {
			if name != "" && state.Name == "" {
				state.Name = name
			}
			pending.lastID = state.ToolUseID
		}
	case pending.lastID != "" && pending.get(pending.lastID) != nil:
		state = pending.get(pending.lastID)
		if name != "" && state.Name != name {
			if err := finishToolUse(state, callback); err != nil {
				return err
			}
			pending.remove(state.ToolUseID)
			state = &toolUseState{ToolUseID: "toolu_" + uuid.New().String(), Name: name, GeneratedID: true}
			pending.add(state)
		}
	case name != "":
		state = &toolUseState{ToolUseID: "toolu_" + uuid.New().String(), Name: name, GeneratedID: true}
		pending.add(state)
	default:
		return nil
	}

	if input, ok := event["input"].(string); ok {
		state.InputBuffer.WriteString(input)
	} else if inputObj, ok := event["input"].(map[string]interface{}); ok {
		data, _ := json.Marshal(inputObj)
		state.InputBuffer.Reset()
		state.InputBuffer.Write(data)
	}

	if isStop {
		if err := finishToolUse(state, callback); err != nil {
			return err
		}
		pending.remove(state.ToolUseID)
	}
	return nil
}

func finishToolUse(state *toolUseState, callback *KiroStreamCallback) error {
	if state == nil || state.Name == "" {
		return nil
	}
	if state.ToolUseID == "" {
		state.ToolUseID = "toolu_" + uuid.New().String()
	}
	var input map[string]interface{}
	if state.InputBuffer.Len() > 0 {
		if err := json.Unmarshal([]byte(state.InputBuffer.String()), &input); err != nil {
			return fmt.Errorf("%w: %v", errIncompleteKiroToolInput, err)
		}
	}
	if input == nil {
		input = make(map[string]interface{})
	}
	if callback == nil || callback.OnToolUse == nil {
		return nil
	}
	callback.OnToolUse(KiroToolUse{
		ToolUseID: state.ToolUseID,
		Name:      state.Name,
		Input:     input,
	})
	return nil
}

func firstStringField(m map[string]interface{}, keys ...string) string {
	for _, key := range keys {
		if v, ok := m[key].(string); ok && v != "" {
			return v
		}
	}
	return ""
}

func firstBoolField(m map[string]interface{}, keys ...string) bool {
	for _, key := range keys {
		if v, ok := m[key].(bool); ok {
			return v
		}
	}
	return false
}

func eventStreamUint32(data []byte) uint32 {
	return uint32(data[0])<<24 | uint32(data[1])<<16 | uint32(data[2])<<8 | uint32(data[3])
}

func eventStreamChecksum(prelude, message []byte) uint32 {
	checksum := crc32.Update(0, crc32.IEEETable, prelude)
	return crc32.Update(checksum, crc32.IEEETable, message)
}

func parseEventStreamHeaders(data []byte) (map[string]string, error) {
	headers := make(map[string]string)
	for offset := 0; offset < len(data); {
		nameLength := int(data[offset])
		offset++
		if nameLength == 0 || offset+nameLength >= len(data) {
			return nil, errors.New("malformed event header name")
		}
		name := string(data[offset : offset+nameLength])
		offset += nameLength
		valueType := data[offset]
		offset++

		valueLength := 0
		switch valueType {
		case 0, 1:
			continue
		case 2:
			valueLength = 1
		case 3:
			valueLength = 2
		case 4:
			valueLength = 4
		case 5, 8:
			valueLength = 8
		case 9:
			valueLength = 16
		case 6, 7:
			if offset+2 > len(data) {
				return nil, errors.New("malformed variable event header")
			}
			valueLength = int(data[offset])<<8 | int(data[offset+1])
			offset += 2
		default:
			return nil, fmt.Errorf("unsupported event header type %d", valueType)
		}
		if offset+valueLength > len(data) {
			return nil, errors.New("truncated event header value")
		}
		if valueType == 7 {
			headers[name] = string(data[offset : offset+valueLength])
		}
		offset += valueLength
	}
	return headers, nil
}

func normalizeKiroStopReason(reason string) string {
	return strings.ToUpper(strings.TrimSpace(reason))
}

func isKnownKiroStopReason(reason string) bool {
	switch normalizeKiroStopReason(reason) {
	case "END_TURN", "MAX_TOKENS", "MAX_OUTPUT_TOKENS", "CONTEXT_WINDOW_EXCEEDED",
		"CONTENT_FILTERED", "GUARDRAIL_INTERVENED", "PAUSE_TURN", "TOOL_USE":
		return true
	default:
		return false
	}
}

// extractEventType extracts the event type string from AWS Event Stream message headers.
func extractEventType(headers []byte) string {
	offset := 0
	for offset < len(headers) {
		if offset >= len(headers) {
			break
		}
		nameLen := int(headers[offset])
		offset++
		if offset+nameLen > len(headers) {
			break
		}
		name := string(headers[offset : offset+nameLen])
		offset += nameLen
		if offset >= len(headers) {
			break
		}
		valueType := headers[offset]
		offset++

		if valueType == 7 { // String
			if offset+2 > len(headers) {
				break
			}
			valueLen := int(headers[offset])<<8 | int(headers[offset+1])
			offset += 2
			if offset+valueLen > len(headers) {
				break
			}
			value := string(headers[offset : offset+valueLen])
			offset += valueLen
			if name == ":event-type" {
				return value
			}
			continue
		}

		// Skip other value types by their fixed byte widths.
		skipSizes := map[byte]int{0: 0, 1: 0, 2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16}
		if valueType == 6 {
			if offset+2 > len(headers) {
				break
			}
			l := int(headers[offset])<<8 | int(headers[offset+1])
			offset += 2 + l
		} else if skip, ok := skipSizes[valueType]; ok {
			offset += skip
		} else {
			break
		}
	}
	return ""
}
