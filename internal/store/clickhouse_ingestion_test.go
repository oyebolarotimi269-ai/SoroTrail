package store

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeClickHouse is a minimal stand-in for ClickHouse's HTTP interface. It
// understands exactly the statements the ingestion-state pair and Ping
// issue (SELECT ... FROM ingestion_state FINAL, an INSERT ... JSONEachRow
// into the same table, and the scalar SELECT 1), so the tests exercise the
// real encoding/decoding paths rather than a translation the production
// code never takes. Rows are held in memory; the FINAL query returns the
// row with the greatest updated_at, which is how ReplacingMergeTree FINAL
// collapses concurrent versions in the real server.
type fakeClickHouse struct {
	// failWith, when non-zero, is returned for every statement, leaving
	// the stored rows untouched. It simulates a live-but-broken server.
	failWith int

	// select1Result replaces the "1" the healthy SELECT 1 answers: a
	// server that answers a scalar query with something else is exactly
	// the failure a TCP-dial Ping would miss.
	select1Result string

	server *httptest.Server

	versions []map[string]any
}

// newFakeClickHouseWithStore starts the fake server and returns both it and
// a ClickHouse store configured to reach it over the HTTP port. The URL's
// host carries the fake's port as the native port too; the HTTP port
// override is what the executor (and therefore everything these tests
// touch) actually connects to.
func newFakeClickHouseWithStore(t *testing.T) (*fakeClickHouse, *ClickHouse) {
	t.Helper()

	f := &fakeClickHouse{select1Result: "1"}
	f.server = httptest.NewServer(http.HandlerFunc(f.handle))
	t.Cleanup(f.server.Close)

	raw := fmt.Sprintf("clickhouse://user:pass@%s/sorotrail?http_port=%s",
		f.server.Listener.Addr().String(), portOf(f.server.Listener.Addr().String()))
	cfg, err := parseClickHouseConfig(raw)
	require.NoError(t, err)
	return f, &ClickHouse{host: cfg.host, port: cfg.port, cfg: cfg}
}

func portOf(hostPort string) string {
	_, p, _ := strings.Cut(hostPort, ":")
	return p
}

func (f *fakeClickHouse) handle(w http.ResponseWriter, r *http.Request) {
	body := readAll(r)
	if f.failWith != 0 {
		http.Error(w, "simulated server failure", f.failWith)
		return
	}
	stmt := strings.TrimSpace(body)
	switch {
	case strings.HasPrefix(stmt, "INSERT INTO ingestion_state"):
		_, payload, ok := strings.Cut(stmt, "FORMAT JSONEachRow")
		if !ok {
			http.Error(w, "fake: insert missing JSONEachRow format", http.StatusBadRequest)
			return
		}
		for _, line := range strings.Split(payload, "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			var row map[string]any
			if err := json.Unmarshal([]byte(line), &row); err != nil {
				http.Error(w, "fake: bad insert payload: "+line, http.StatusBadRequest)
				return
			}
			f.versions = append(f.versions, row)
		}
		w.WriteHeader(http.StatusOK)
	case strings.Contains(stmt, "FROM ingestion_state FINAL"):
		row, ok := latestClickHouseVersion(f.versions)
		if !ok {
			// An empty result set is how ClickHouse answers a query
			// matching no rows; JSONEachRow emits nothing.
			return
		}
		encoded, err := json.Marshal(row)
		if err != nil {
			http.Error(w, "fake: encoding row", http.StatusInternalServerError)
			return
		}
		_, _ = w.Write(append(encoded, '\n'))
	case stmt == "SELECT 1 FORMAT TabSeparated":
		_, _ = w.Write([]byte(f.select1Result + "\n"))
	default:
		http.Error(w, "fake: unexpected statement: "+truncateForError(stmt), http.StatusInternalServerError)
	}
}

// latestClickHouseVersion emulates `FINAL` for an append-only
// ReplacingMergeTree: the row with the greatest updated_at wins, and on an
// updated_at tie the later-inserted row wins — the "newer save supersedes
// the older" rule the save path relies on for its empty-result second
// save. updated_at strings share one fixed millisecond layout, so
// lexicographic order is chronological order.
func latestClickHouseVersion(versions []map[string]any) (map[string]any, bool) {
	var best map[string]any
	var stamp string
	for _, v := range versions {
		key, _ := v["updated_at"].(string)
		if best == nil || key >= stamp {
			best, stamp = v, key
		}
	}
	return best, best != nil
}

func readAll(r *http.Request) string {
	var b strings.Builder
	buf := make([]byte, 4096)
	for {
		n, err := r.Body.Read(buf)
		b.Write(buf[:n])
		if err != nil {
			break
		}
	}
	return b.String()
}

// TestClickHouseIngestionStateRoundTrip covers #523's save-then-load
// acceptance: state persisted now must come back after the process "went
// away" (a fresh executor inside Get), and re-saving must supersede the
// earlier version the way the ReplacingMergeTree's FINAL read does in
// production — three saves appended while it runs; it always reads back the
// newest.
func TestClickHouseIngestionStateRoundTrip(t *testing.T) {
	f, st := newFakeClickHouseWithStore(t)
	ctx := context.Background()

	// A fresh instance has no resume position; the caller is told that
	// plainly rather than being handed a zero state that reads as
	// "ingested through ledger 0".
	_, err := st.GetIngestionState(ctx)
	assert.ErrorIs(t, err, ErrNotFound, "cold start must report ErrNotFound")

	first := time.Date(2026, 10, 19, 9, 0, 0, 0, time.UTC)
	second := first.Add(17 * time.Minute)
	lastPoll := second.Add(-2 * time.Second)

	require.NoError(t, st.SaveIngestionState(ctx, IngestionState{
		LastIngestedLedger: 100,
		LastCursor:         "cursor@100",
		LastSuccessfulPoll: &first,
	}), "save #1")

	got, err := st.GetIngestionState(ctx)
	require.NoError(t, err, "load #1 (round-trip the save that just landed)")
	assert.Equal(t, IngestionState{
		Network:            "default",
		LastIngestedLedger: 100,
		LastCursor:         "cursor@100",
		LastSuccessfulPoll: &first,
		UpdatedAt:          got.UpdatedAt,
	}, got, "every persisted field must survive the save-then-load round trip")
	assert.WithinDuration(t, time.Now(), got.UpdatedAt, 5*time.Second,
		"updated_at must reflect the save time, the reload marker a restart resumes from")

	// Save #2 carries only the ledger: the newer version wins wholesale,
	// including for fields the caller left unset, exactly as the
	// Postgres ON CONFLICT DO UPDATE overwrites the row.
	require.NoError(t, st.SaveIngestionState(ctx, IngestionState{
		Network:            "default",
		LastIngestedLedger: 2222,
		LastSuccessfulPoll: &lastPoll,
	}))
	got, err = st.GetIngestionState(ctx)
	require.NoError(t, err, "load #2 (newest version wins)")
	assert.Equal(t, int64(2222), got.LastIngestedLedger, "the newest save must be the one a restart resumes from")
	assert.Empty(t, got.LastCursor, "unset fields must not inherit stale values from the older version")
	require.NotNil(t, got.LastSuccessfulPoll, "last_successful_poll set on the newest save must round-trip")
	assert.True(t, got.LastSuccessfulPoll.Equal(lastPoll), "last_successful_poll must round-trip at millisecond precision")

	// The fake only ever answered the two statements this backend issues;
	// anything else means the SQL and the fake drifted apart, which this
	// assertion keeps silent-hook-patches (a handler matching too
	// generously) from hiding.
	assert.GreaterOrEqual(t, len(f.versions), 2, "each save must actually append a version, not mutate in place")
}

// TestClickHouseIngestionStateFailuresAreLoud pins the one option #523
// rules out: a backend failure must surface as an error on both the save
// and the load, never as a silent nil-plus-zero-struct that reads as
// "caught up from the retention window".
func TestClickHouseIngestionStateFailuresAreLoud(t *testing.T) {
	f, st := newFakeClickHouseWithStore(t)
	f.failWith = http.StatusInternalServerError
	ctx := context.Background()

	err := st.SaveIngestionState(ctx, IngestionState{LastIngestedLedger: 7})
	require.Error(t, err, "a failed save must return an error, not a silent nil")
	assert.Contains(t, err.Error(), "500", "the error should name the server's failure, not bury it")

	_, err = st.GetIngestionState(ctx)
	require.Error(t, err, "a failed load must return an error")
	assert.NotErrorIs(t, err, ErrNotFound, "a broken server is not the same thing as a cold start")

	// Nothing was written while failing: a retried process must not see
	// partial state from the broken window.
	f.failWith = 0
	_, err = st.GetIngestionState(ctx)
	assert.ErrorIs(t, err, ErrNotFound, "failed saves must not leave half-written state behind")
}

// TestClickHousePingRequiresServingClickHouse covers #524: readiness is
// only real when the server answers an actual query. Each scenario fails a
// bare TCP dial would have passed.
func TestClickHousePingRequiresServingClickHouse(t *testing.T) {
	ctx := context.Background()

	t.Run("healthy server answers SELECT 1", func(t *testing.T) {
		_, st := newFakeClickHouseWithStore(t)
		require.NoError(t, st.Ping(ctx))
	})

	t.Run("live listener that cannot serve queries is not ready", func(t *testing.T) {
		f, st := newFakeClickHouseWithStore(t)
		f.failWith = http.StatusServiceUnavailable
		err := st.Ping(ctx)
		require.Error(t, err, "a listening-but-broken server must fail the readiness probe")
		assert.Contains(t, err.Error(), "503", "the error should name the server's failure")
	})

	t.Run("server answering a wrong scalar is not ready", func(t *testing.T) {
		f, st := newFakeClickHouseWithStore(t)
		f.select1Result = "0"
		err := st.Ping(ctx)
		require.Error(t, err, "a contradictory answer is proof of a wrong or poisoned server")
		assert.Contains(t, err.Error(), "SELECT 1", "the error should say which probe failed")
	})

	t.Run("unreachable server is not ready", func(t *testing.T) {
		_, st := newFakeClickHouseWithStore(t)
		st.host = "127.0.0.1"
		st.cfg.httpPort = 1 // nothing serves http://127.0.0.1:1 in CI
		err := st.Ping(ctx)
		require.Error(t, err, "pinging a dead endpoint must fail the probe")
	})

	t.Run("empty host is refused without dialing", func(t *testing.T) {
		_, st := newFakeClickHouseWithStore(t)
		st.host = ""
		assert.Error(t, st.Ping(ctx))
	})

	t.Run("probe honours the caller's context deadline", func(t *testing.T) {
		f, st := newFakeClickHouseWithStore(t)
		f.failWith = 0
		// A healthy endpoint whose answer takes longer than the probe's
		// budget is *unready* — that is the whole point of a timed probe.
		f.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			time.Sleep(500 * time.Millisecond)
			_, _ = w.Write([]byte("1\n"))
		})
		deadlineCtx, cancel := context.WithTimeout(ctx, 20*time.Millisecond)
		defer cancel()
		err := st.Ping(deadlineCtx)
		require.Error(t, err, "an over-deadline probe must fail, not wait for the answer")
	})
}
