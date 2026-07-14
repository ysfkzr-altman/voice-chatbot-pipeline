#!/usr/bin/env bash
# Runs every eval scenario in evals/ against a FRESH bot.py process each
# time (running multiple scenarios against one long-lived process has
# caused real problems before - e.g. Kokoro TTS silently stopped firing
# after several turns, and the eval harness's own event-matching gets
# confused by leftover state from a previous scenario). Restarting fresh
# per scenario trades a bit of wall-clock time for actually trustworthy
# results.
#
# Usage:
#   scripts/run_all_evals.sh              # run every scenario in evals/
#   scripts/run_all_evals.sh foo bar       # run only evals/foo.yaml, evals/bar.yaml
#
# Exit code is 0 only if every scenario passed.
set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON="/d/venvs/pipecat-voice/Scripts/python.exe"
PIPECAT="/d/venvs/pipecat-voice/Scripts/pipecat.exe"
PORT=7860
LOG_DIR="eval_logs"
mkdir -p "$LOG_DIR"

if [ "$#" -gt 0 ]; then
    SCENARIOS=()
    for name in "$@"; do
        SCENARIOS+=("evals/${name}.yaml")
    done
else
    SCENARIOS=(evals/*.yaml)
fi

declare -A RESULTS
declare -A DURATIONS

# Scenarios that need the bot process started with a non-default
# environment (name -> "VAR=value VAR2=value2 ..."). stt_failure_test
# specifically needs a broken GROQ_API_KEY to exercise the STT-down
# fallback path - starting it normally (real key) means STT succeeds and
# the test can never see the behavior it's checking for.
# SKIP_STARTUP_SELF_CHECK is also needed here: bot.py's own startup
# self-check (added for bulletproofing) would otherwise catch this same
# bad key and exit before the scenario ever gets a chance to run - this
# is the one deliberate, intentional case where that's the wrong thing.
declare -A SCENARIO_ENV=(
    [stt_failure_test]="GROQ_API_KEY=bad_key_for_testing SKIP_STARTUP_SELF_CHECK=1"
)

# Scenarios whose `eval:` semantic-judge checks depend on a local Ollama
# instance being up. When Ollama is down, these fail with an
# APIConnectionError that has nothing to do with the bot itself - flagged
# here so the summary doesn't misreport an infra gap as a bot regression.
OLLAMA_DEPENDENT="ambiguous_input_test"

# Brief pause between scenarios: running bot.py processes back-to-back at
# full speed measurably slows Kokoro's first TTS call in the next
# scenario (confirmed: a scenario that fails under rapid succession passes
# cleanly seconds later in isolation) - almost certainly transient
# resource contention (CPU/model-load) from the just-killed process not
# having fully released yet, not a real bug.
INTER_SCENARIO_COOLDOWN=3

kill_port() {
    # Find and kill whatever's actually listening on $PORT (there's no
    # reliable `kill $pid` for a Windows process launched via `&` in Git
    # Bash - taskkill on the real PID from netstat is what actually works).
    local pids
    pids=$(netstat -ano 2>/dev/null | grep ":$PORT" | grep LISTENING | awk '{print $NF}' | sort -u)
    for pid in $pids; do
        taskkill //F //PID "$pid" >/dev/null 2>&1 || true
    done
}

wait_for_port() {
    local tries=0
    while ! (netstat -ano 2>/dev/null | grep ":$PORT" | grep -q LISTENING); do
        sleep 1
        tries=$((tries + 1))
        if [ "$tries" -ge 30 ]; then
            return 1
        fi
    done
    return 0
}

# bot.py calls load_dotenv(override=True) - a plain exported env var gets
# clobbered right back to the real .env's value, so a scenario that needs
# a broken GROQ_API_KEY (stt_failure_test) can't just export one. Hiding
# .env for that one process's lifetime is the only way to make the
# override actually stick. Trap guarantees it's restored even if this
# script is killed mid-run - never leave the real .env missing.
ENV_HIDDEN=0
hide_env() {
    if [ -f .env ]; then
        mv .env .env.evalrunner.bak
        ENV_HIDDEN=1
    fi
}
restore_env() {
    if [ "$ENV_HIDDEN" -eq 1 ] && [ -f .env.evalrunner.bak ]; then
        mv .env.evalrunner.bak .env
        ENV_HIDDEN=0
    fi
}
trap restore_env EXIT INT TERM

echo "Running ${#SCENARIOS[@]} scenario(s)..."
echo ""

for scenario in "${SCENARIOS[@]}"; do
    name=$(basename "$scenario" .yaml)

    if [ ! -f "$scenario" ]; then
        echo "=== $name === SKIPPED (file not found: $scenario)"
        RESULTS["$name"]="SKIP"
        continue
    fi

    echo "=== $name ==="
    needs_special_env="${SCENARIO_ENV[$name]:-}"
    if [ -n "$needs_special_env" ]; then
        echo "  (starting bot with special env: $needs_special_env - hiding real .env for this one process)"
        hide_env
    fi
    kill_port
    sleep 1

    bot_log="$LOG_DIR/${name}.bot.log"
    (
        if [ -n "$needs_special_env" ]; then
            eval "export $needs_special_env"
        fi
        PYTHONUTF8=1 "$PYTHON" bot.py -t eval > "$bot_log" 2>&1
    ) &

    if ! wait_for_port; then
        echo "  FAIL - bot never started listening on :$PORT (see $bot_log)"
        RESULTS["$name"]="FAIL"
        DURATIONS["$name"]="-"
        kill_port
        restore_env
        continue
    fi

    start_ts=$(date +%s)
    if PYTHONUTF8=1 "$PIPECAT" eval run "$scenario" -v 2>&1 | tee "$LOG_DIR/${name}.eval.log"; then
        RESULTS["$name"]="PASS"
    else
        RESULTS["$name"]="FAIL"
    fi
    end_ts=$(date +%s)
    DURATIONS["$name"]="$((end_ts - start_ts))s"

    kill_port
    restore_env
    rm -f ./*.eval.log
    echo ""
    sleep "$INTER_SCENARIO_COOLDOWN"
done

echo "=================================================="
echo "SUMMARY"
echo "=================================================="
pass=0
fail=0
skip=0
for scenario in "${SCENARIOS[@]}"; do
    name=$(basename "$scenario" .yaml)
    result="${RESULTS[$name]:-FAIL}"
    duration="${DURATIONS[$name]:-}"
    note=""
    if [ "$result" != "PASS" ] && [[ " $OLLAMA_DEPENDENT " == *" $name "* ]]; then
        note="  (needs local Ollama running for its judge - check that before treating this as a bot bug)"
    fi
    printf "  %-32s %-6s %-6s%s\n" "$name" "$result" "$duration" "$note"
    case "$result" in
        PASS) pass=$((pass + 1)) ;;
        SKIP) skip=$((skip + 1)) ;;
        *) fail=$((fail + 1)) ;;
    esac
done
echo "--------------------------------------------------"
echo "$pass passed, $fail failed, $skip skipped"
echo "Logs: $LOG_DIR/"

if [ "$fail" -gt 0 ]; then
    exit 1
fi
exit 0
