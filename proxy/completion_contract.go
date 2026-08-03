package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"kiro-go/config"
	"kiro-go/logger"
	"strings"

	"github.com/google/uuid"
)

const maxClaudeCompletionRounds = 4

var (
	errClaudeCompletionContract        = errors.New("Claude completion contract failed")
	errClaudeCompletionRoundsExhausted = errors.New("Claude completion control rounds exhausted")
)

type claudeCompletionRound struct {
	outputs      []bufferedKiroOutput
	text         strings.Builder
	thinking     strings.Builder
	toolUses     []KiroToolUse
	inputTokens  int
	outputTokens int
	credits      float64
	contextUsage []float64
	stopReason   string
	completed    bool
}

type claudeCompletionDecision struct {
	outcome       string
	stopReason    string
	finishMessage string
	clientTools   int
	internalTools int
}

type claudeCompletionBudget struct {
	attempts int
}

type claudeCompletionBudgetContextKey struct{}

func withClaudeCompletionBudget(ctx context.Context) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	if _, ok := ctx.Value(claudeCompletionBudgetContextKey{}).(*claudeCompletionBudget); ok {
		return ctx
	}
	return context.WithValue(ctx, claudeCompletionBudgetContextKey{}, &claudeCompletionBudget{})
}

// callKiroForClaude applies the structured completion contract only to Claude
// requests that exposed client tools. All output remains buffered until a
// client tool call, an internal finish call, or an explicit refusal/overflow
// proves that the round is terminal.
func callKiroForClaude(ctx context.Context, account *config.Account, payload *KiroPayload, callback *KiroStreamCallback) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if payload == nil || !payload.StrictCompletion {
		return CallKiroAPIContext(ctx, account, payload, callback)
	}

	working, err := cloneKiroPayload(payload)
	if err != nil {
		return fmt.Errorf("%w: clone request: %v", errClaudeCompletionContract, err)
	}

	budget, _ := ctx.Value(claudeCompletionBudgetContextKey{}).(*claudeCompletionBudget)
	if budget == nil {
		budget = &claudeCompletionBudget{}
	}
	recoveryRounds := 0
	totalCredits := 0.0
	beforeAttempt := func() error {
		if budget.attempts >= maxClaudeCompletionRounds {
			return errClaudeCompletionRoundsExhausted
		}
		budget.attempts++
		return nil
	}

	for {
		round := &claudeCompletionRound{}
		roundCallback := round.callback(callback)
		err = callKiroAPIContext(ctx, account, working, roundCallback, beforeAttempt)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return err
			}
			if errors.Is(err, errClaudeCompletionRoundsExhausted) {
				return completionRoundsError(budget.attempts, err)
			}
			if errors.Is(err, errUpstreamTruncatedResponse) && strings.TrimSpace(round.text.String()) != "" {
				recoveryRounds++
				logger.Warnf("[ClaudeCompletionContract] round=%d outcome=continue cause=missing_stop_reason assistant_bytes=%d recovery_rounds=%d",
					budget.attempts, len([]byte(round.text.String())), recoveryRounds)
				observeLegacyProgressRule(working, round.text.String(), round.toolUses, budget.attempts)
				if budget.attempts >= maxClaudeCompletionRounds {
					return completionRoundsError(budget.attempts, err)
				}
				// Kiro IDE can cleanly end a text response without metadata. Keep
				// that text as internal progress and ask for a structured terminal
				// action instead of repeating the unchanged request.
				if appendErr := appendClaudeCompletionContinuation(working, round, false); appendErr != nil {
					return fmt.Errorf("%w: build missing-stop continuation: %v", errClaudeCompletionContract, appendErr)
				}
				continue
			}
			if isKiroStreamIntegrityError(err) {
				logger.Warnf("[ClaudeCompletionContract] round=%d outcome=continue cause=integrity error=%q", budget.attempts, err.Error())
				if budget.attempts >= maxClaudeCompletionRounds {
					return completionRoundsError(budget.attempts, err)
				}
				// The buffered round is intentionally discarded. Retrying the same
				// payload avoids adding a truncated sentence to Kiro history.
				continue
			}
			return err
		}

		totalCredits += round.credits
		decision := classifyClaudeCompletionRound(working, round)
		if decision.clientTools > 0 && decision.internalTools > 0 {
			logger.Warnf("[ClaudeCompletionContract] round=%d conflict=client_and_internal_finish client_tools=%d internal_tools=%d resolution=client_tool", budget.attempts, decision.clientTools, decision.internalTools)
		}

		switch decision.outcome {
		case "tool_use", "finish", "terminal":
			logger.Infof("[ClaudeCompletionContract] round=%d outcome=%s stop_reason=%q recovery_rounds=%d", budget.attempts, decision.outcome, decision.stopReason, recoveryRounds)
			replayClaudeCompletionRound(callback, working, round, decision, totalCredits)
			return nil
		case "continue":
			recoveryRounds++
			logger.Infof("[ClaudeCompletionContract] round=%d outcome=continue stop_reason=%q recovery_rounds=%d", budget.attempts, normalizeKiroStopReason(round.stopReason), recoveryRounds)
			observeLegacyProgressRule(working, round.text.String(), round.toolUses, budget.attempts)
			if budget.attempts >= maxClaudeCompletionRounds {
				return completionRoundsError(budget.attempts, errors.New("upstream did not call a client tool or the internal finish tool"))
			}
			if err := appendClaudeCompletionContinuation(working, round, decision.internalTools > 0); err != nil {
				return fmt.Errorf("%w: build continuation: %v", errClaudeCompletionContract, err)
			}
		default:
			return fmt.Errorf("%w: unknown controller outcome %q", errClaudeCompletionContract, decision.outcome)
		}
	}
}

func (r *claudeCompletionRound) callback(downstream *KiroStreamCallback) *KiroStreamCallback {
	return &KiroStreamCallback{
		OnProgress: func() {
			if downstream != nil && downstream.OnProgress != nil {
				downstream.OnProgress()
			}
		},
		OnText: func(text string, isThinking bool) {
			r.outputs = append(r.outputs, bufferedKiroOutput{text: text, isThinking: isThinking})
			if isThinking {
				r.thinking.WriteString(text)
			} else {
				r.text.WriteString(text)
			}
		},
		OnToolUse: func(toolUse KiroToolUse) {
			r.toolUses = append(r.toolUses, toolUse)
			toolUseCopy := toolUse
			r.outputs = append(r.outputs, bufferedKiroOutput{toolUse: &toolUseCopy})
		},
		OnComplete: func(inputTokens, outputTokens int) {
			r.inputTokens = inputTokens
			r.outputTokens = outputTokens
			r.completed = true
		},
		OnCredits: func(credits float64) {
			r.credits += credits
		},
		OnContextUsage: func(percentage float64) {
			r.contextUsage = append(r.contextUsage, percentage)
		},
		OnStopReason: func(reason string) {
			r.stopReason = reason
		},
	}
}

func classifyClaudeCompletionRound(payload *KiroPayload, round *claudeCompletionRound) claudeCompletionDecision {
	decision := claudeCompletionDecision{outcome: "continue", stopReason: normalizeKiroStopReason(round.stopReason)}
	internalName := payload.InternalFinishToolName
	var internalTools []KiroToolUse
	for _, toolUse := range round.toolUses {
		if toolUse.Name == internalName {
			internalTools = append(internalTools, toolUse)
		} else {
			decision.clientTools++
		}
	}
	decision.internalTools = len(internalTools)

	// A client tool always wins. This preserves the client's tool loop even if
	// the model incorrectly emits the internal finish tool in the same response.
	if decision.clientTools > 0 {
		decision.outcome = "tool_use"
		decision.stopReason = "TOOL_USE"
		return decision
	}

	for _, toolUse := range internalTools {
		if message, ok := validInternalFinishMessage(toolUse); ok {
			decision.outcome = "finish"
			decision.stopReason = "END_TURN"
			decision.finishMessage = message
			return decision
		}
	}

	if isExplicitKiroTerminalReason(decision.stopReason) {
		decision.outcome = "terminal"
		return decision
	}
	return decision
}

func validInternalFinishMessage(toolUse KiroToolUse) (string, bool) {
	status, statusOK := toolUse.Input["status"].(string)
	message, messageOK := toolUse.Input["message"].(string)
	status = strings.ToLower(strings.TrimSpace(status))
	message = strings.TrimSpace(message)
	if !statusOK || !messageOK || message == "" {
		return "", false
	}
	if status != "completed" && status != "blocked" {
		return "", false
	}
	return message, true
}

func replayClaudeCompletionRound(callback *KiroStreamCallback, payload *KiroPayload, round *claudeCompletionRound, decision claudeCompletionDecision, totalCredits float64) {
	if callback == nil {
		return
	}

	for _, output := range round.outputs {
		if output.toolUse != nil {
			if output.toolUse.Name == payload.InternalFinishToolName || decision.outcome == "finish" {
				continue
			}
			if callback.OnToolUse != nil {
				callback.OnToolUse(*output.toolUse)
			}
			continue
		}
		if decision.outcome == "finish" && !output.isThinking {
			continue
		}
		if callback.OnText != nil {
			callback.OnText(output.text, output.isThinking)
		}
	}

	if decision.outcome == "finish" && callback.OnText != nil {
		callback.OnText(decision.finishMessage, false)
	}
	if callback.OnCredits != nil && totalCredits > 0 {
		callback.OnCredits(totalCredits)
	}
	if callback.OnContextUsage != nil {
		for _, percentage := range round.contextUsage {
			callback.OnContextUsage(percentage)
		}
	}
	if callback.OnStopReason != nil {
		callback.OnStopReason(decision.stopReason)
	}
	if callback.OnComplete != nil {
		callback.OnComplete(round.inputTokens, round.outputTokens)
	}
}

func appendClaudeCompletionContinuation(payload *KiroPayload, round *claudeCompletionRound, invalidFinish bool) error {
	if payload == nil {
		return errors.New("nil payload")
	}

	current := payload.ConversationState.CurrentMessage.UserInputMessage
	historyUser := current
	if current.UserInputMessageContext != nil {
		historyContext := *current.UserInputMessageContext
		historyContext.Tools = nil
		historyUser.UserInputMessageContext = &historyContext
	}
	payload.ConversationState.History = append(payload.ConversationState.History, KiroHistoryMessage{
		UserInputMessage: &historyUser,
	})

	partial := strings.TrimSpace(round.text.String())
	if partial == "" {
		partial = "[The previous response did not contain a usable final answer.]"
	}
	payload.ConversationState.History = append(payload.ConversationState.History, KiroHistoryMessage{
		AssistantResponseMessage: &KiroAssistantResponseMessage{Content: partial},
	})

	instruction := fmt.Sprintf(
		"Continue executing the original request now. The previous assistant response was only partial progress. Use a client tool for the next actionable step, or call %s only when the task is complete or genuinely blocked.",
		payload.InternalFinishToolName,
	)
	if invalidFinish {
		instruction = fmt.Sprintf(
			"The previous %s call was invalid. Continue the original task, then call it with status completed or blocked and a non-empty message only when that terminal state is true.",
			payload.InternalFinishToolName,
		)
	}

	var tools []KiroToolWrapper
	if current.UserInputMessageContext != nil {
		tools = current.UserInputMessageContext.Tools
	}
	payload.ConversationState.CurrentMessage.UserInputMessage = KiroUserInputMessage{
		Content: instruction,
		ModelID: current.ModelID,
		Origin:  current.Origin,
		UserInputMessageContext: &UserInputMessageContext{
			Tools: tools,
		},
	}
	payload.ConversationState.AgentContinuationId = uuid.New().String()
	truncatePayloadToLimit(payload, true)
	return nil
}

func cloneKiroPayload(payload *KiroPayload) (*KiroPayload, error) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	clone := &KiroPayload{}
	if err := json.Unmarshal(raw, clone); err != nil {
		return nil, err
	}
	clone.StrictCompletion = payload.StrictCompletion
	clone.InternalFinishToolName = payload.InternalFinishToolName
	if len(payload.ToolNameMap) > 0 {
		clone.ToolNameMap = make(map[string]string, len(payload.ToolNameMap))
		for sanitized, original := range payload.ToolNameMap {
			clone.ToolNameMap[sanitized] = original
		}
	}
	return clone, nil
}

func isExplicitKiroTerminalReason(reason string) bool {
	switch normalizeKiroStopReason(reason) {
	case "CONTEXT_WINDOW_EXCEEDED", "CONTENT_FILTERED", "GUARDRAIL_INTERVENED":
		return true
	default:
		return false
	}
}

func isKiroStreamIntegrityError(err error) bool {
	return errors.Is(err, errEmptyKiroStream) ||
		errors.Is(err, errUpstreamTruncatedResponse) ||
		errors.Is(err, errIncompleteKiroResponse) ||
		errors.Is(err, errIncompleteKiroToolInput) ||
		errors.Is(err, errInvalidKiroEventStream) ||
		errors.Is(err, io.ErrUnexpectedEOF)
}

func isSoftKiroCompletionError(err error) bool {
	return errors.Is(err, errClaudeCompletionContract) || isKiroStreamIntegrityError(err)
}

func completionRoundsError(attempts int, cause error) error {
	return fmt.Errorf("%w after %d control rounds: %w", errClaudeCompletionContract, attempts, cause)
}

func observeLegacyProgressRule(payload *KiroPayload, text string, toolUses []KiroToolUse, round int) {
	if payload == nil {
		return
	}
	observationPayload := *payload
	observationPayload.StrictCompletion = false
	if rule, matched := shouldRejectClaudeProgressOnly(&observationPayload, text, toolUses); matched {
		logger.Infof("[ClaudeProgressGuard] rule=%q observation_only=true round=%d assistant_bytes=%d", rule, round, len([]byte(text)))
	}
}
