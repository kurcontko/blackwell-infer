#!/bin/bash
# Quick benchmark script to test SGLang server throughput
# Run this after server starts to verify performance

set -e

API_URL="${API_URL:-http://localhost:8000/v1/chat/completions}"
NUM_REQUESTS="${NUM_REQUESTS:-100}"
CONCURRENT="${CONCURRENT:-10}"

echo "=================================="
echo "Blackwell HyperInfer - Quick Bench"
echo "=================================="
echo "API: $API_URL"
echo "Requests: $NUM_REQUESTS"
echo "Concurrent: $CONCURRENT"
echo "=================================="

# Check if server is up
echo "Checking server health..."
if ! curl -s -f "$API_URL" > /dev/null 2>&1; then
    # Try health endpoint
    if ! curl -s -f "http://localhost:8000/health" > /dev/null 2>&1; then
        echo "ERROR: Server not responding"
        exit 1
    fi
fi

echo "✓ Server is up"

# Simple benchmark with Apache Bench (if available)
if command -v ab &> /dev/null; then
    echo "Running Apache Bench..."

    # Create test payload
    cat > /tmp/bench_payload.json <<EOF
{
  "model": "default",
  "messages": [{"role": "user", "content": "Count from 1 to 10"}],
  "max_tokens": 50
}
EOF

    ab -n "$NUM_REQUESTS" -c "$CONCURRENT" \
       -p /tmp/bench_payload.json \
       -T "application/json" \
       "$API_URL"

    rm /tmp/bench_payload.json
else
    echo "Apache Bench (ab) not found, using simple curl test..."

    START=$(date +%s)

    for i in $(seq 1 "$NUM_REQUESTS"); do
        curl -s -X POST "$API_URL" \
            -H "Content-Type: application/json" \
            -d '{
                "model": "default",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 10
            }' > /dev/null &

        # Limit concurrency
        if [ $((i % CONCURRENT)) -eq 0 ]; then
            wait
        fi
    done

    wait

    END=$(date +%s)
    DURATION=$((END - START))
    RPS=$(awk "BEGIN {print $NUM_REQUESTS / $DURATION}")

    echo "Completed $NUM_REQUESTS requests in ${DURATION}s"
    echo "Throughput: $RPS req/s"
fi

echo "=================================="
echo "Benchmark complete!"
echo "=================================="
