package broker

import (
	"errors"
	"fmt"
	"net/http"
)

// ActionErrorKind classifies action failures taken under a node lock
type ActionErrorKind int

const (
	// ActionErrorTransport indicates a transport-level failure (e.g. POST could not connect/timeout)
	ActionErrorTransport ActionErrorKind = iota
	// ActionErrorApplication indicates an application-level failure that should not be retried by default
	ActionErrorApplication
)

// ActionError wraps an underlying error with a classification
type ActionError struct {
	Kind ActionErrorKind
	Err  error
}

func (e *ActionError) Error() string {
	if e == nil {
		return "<nil>"
	}
	return e.Err.Error()
}

func (e *ActionError) Unwrap() error { return e.Err }

func NewTransportActionError(err error) *ActionError {
	if err == nil {
		err = errors.New("transport error")
	}
	return &ActionError{Kind: ActionErrorTransport, Err: err}
}

func NewApplicationActionError(err error) *ActionError {
	if err == nil {
		err = errors.New("application error")
	}
	return &ActionError{Kind: ActionErrorApplication, Err: err}
}

// DoWithLockedNodeHTTPRetry is a convenience helper for HTTP calls under a node lock.
// It centralizes retry and status re-check logic:
// - Transport errors (no HTTP response) trigger status re-check, node skip and retry.
// - HTTP 5xx responses trigger status re-check, node skip and retry.
// - HTTP 4xx responses are returned as-is without retry.
// - 2xx responses are returned.
func DoWithLockedNodeHTTPRetry(
	b *Broker,
	model string,
	skipNodeIDs []string,
	maxAttempts int,
	doPost func(node *Node) (*http.Response, *ActionError),
) (*http.Response, error) {
	var zero *http.Response
	if maxAttempts <= 0 {
		maxAttempts = 1
	}

	skip := make(map[string]struct{}, len(skipNodeIDs))
	orderedSkip := make([]string, 0, len(skipNodeIDs))
	for _, id := range skipNodeIDs {
		if id != "" {
			if _, seen := skip[id]; !seen {
				skip[id] = struct{}{}
				orderedSkip = append(orderedSkip, id)
			}
		}
	}

	var lastErr error
	attempts := 0

	for attempts < maxAttempts {
		attempts++

		nodeChan := make(chan *Node, 2)
		if err := b.QueueMessage(LockAvailableNode{Model: model, Response: nodeChan, SkipNodeIDs: orderedSkip}); err != nil {
			return zero, err
		}
		node := <-nodeChan
		if node == nil {
			if lastErr != nil {
				return zero, lastErr
			}
			return zero, ErrNoNodesAvailable
		}

		resp, aerr := doPost(node)

		// Decide outcome and retry policy
		retry := false
		triggerRecheck := false
		fatal := false

		if aerr != nil {
			if aerr.Kind == ActionErrorTransport {
				// Transport error: retry and recheck
				retry = true
				triggerRecheck = true
				fatal = false
				lastErr = fmt.Errorf("node %s transport failure: %w", node.Id, aerr)
			} else {
				// Application error: do not retry
				retry = false
				triggerRecheck = false
				fatal = true
				lastErr = aerr
			}
		} else if resp != nil {
			if resp.StatusCode >= 500 {
				// Server error: retry and recheck
				retry = true
				triggerRecheck = true
				fatal = false
				lastErr = fmt.Errorf("node %s server error: status=%d", node.Id, resp.StatusCode)
			} else if resp.StatusCode < 200 || resp.StatusCode >= 300 {
				// 4xx or other non-success (non-retryable)
				retry = false
				triggerRecheck = false
				fatal = true // request problem
			}
		}

		// Release lock with outcome immediately
		var outcome InferenceResult
		if aerr == nil && resp != nil && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			outcome = InferenceSuccess{Response: nil}
		} else {
			// Compose a concise message
			msg := ""
			if aerr != nil {
				msg = aerr.Error()
			} else if resp != nil {
				msg = fmt.Sprintf("http status %d", resp.StatusCode)
			} else {
				msg = "unknown error"
			}
			outcome = InferenceError{Message: msg, IsFatal: fatal}
		}
		_ = b.QueueMessage(ReleaseNode{NodeId: node.Id, Outcome: outcome, Response: make(chan bool, 2)})

		if retry {
			if triggerRecheck {
				b.TriggerStatusQuery()
			}
			if resp != nil && resp.Body != nil {
				// Ensure we don't leak the body before retrying
				_ = resp.Body.Close()
			}
			if _, seen := skip[node.Id]; !seen {
				skip[node.Id] = struct{}{}
				orderedSkip = append(orderedSkip, node.Id)
			}
			// Continue to next attempt
			continue
		}

		// No retry: return
		if aerr != nil {
			return zero, aerr
		}
		return resp, nil
	}

	if lastErr != nil {
		return zero, lastErr
	}
	return zero, ErrNoNodesAvailable
}
