package broker

import (
	"errors"
	"fmt"
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

// DoWithLockedNodeRetry locks a node supporting the requested model, runs the action, and
// retries on transport-level failures. It skips nodes listed in skipNodeIDs and any nodes
// that fail with transport errors during this call. On a transport failure it triggers a
// node status query to refresh health.
func DoWithLockedNodeRetry[T any](
	b *Broker,
	model string,
	skipNodeIDs []string,
	maxAttempts int,
	action func(node *Node) (T, *ActionError),
) (T, error) {
	var zero T
	if maxAttempts <= 0 {
		maxAttempts = 1
	}

	// Build skip set and a slice we pass to the lock command
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

		// Skip unwanted nodes immediately
		if _, shouldSkip := skip[node.Id]; shouldSkip {
			// release with non-fatal error outcome
			_ = b.QueueMessage(ReleaseNode{
				NodeId:   node.Id,
				Outcome:  InferenceError{Message: "skipped by DoWithLockedNodeRetry", IsFatal: false},
				Response: make(chan bool, 2),
			})
			lastErr = fmt.Errorf("node %s skipped by policy", node.Id)
			continue
		}

		// Execute action
		value, aerr := action(node)

		// Always release the lock
		var outcome InferenceResult
		if aerr == nil {
			outcome = InferenceSuccess{Response: nil}
		} else {
			outcome = InferenceError{Message: aerr.Error(), IsFatal: aerr.Kind == ActionErrorApplication}
		}
		_ = b.QueueMessage(ReleaseNode{
			NodeId:   node.Id,
			Outcome:  outcome,
			Response: make(chan bool, 2),
		})

		if aerr == nil {
			return value, nil
		}

		// Transport failure: trigger status refresh, skip this node, and retry
		if aerr.Kind == ActionErrorTransport {
			b.TriggerStatusQuery()
			if _, seen := skip[node.Id]; !seen {
				skip[node.Id] = struct{}{}
				orderedSkip = append(orderedSkip, node.Id)
			}
			lastErr = fmt.Errorf("node %s transport failure: %w", node.Id, aerr)
			continue
		}

		// Non-transport failure: do not retry
		return zero, aerr
	}

	if lastErr != nil {
		return zero, lastErr
	}
	return zero, ErrNoNodesAvailable
}
