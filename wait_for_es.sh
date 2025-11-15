#!/bin/sh
set -e

ES_HOST=${ES_HOST:-elasticsearch}
ES_PORT=${ES_PORT:-9200}
MAX_RETRIES=60
COUNT=0

echo "⏳ Waiting for Elasticsearch at $ES_HOST:$ES_PORT ..."

until curl -s "http://$ES_HOST:$ES_PORT" >/dev/null 2>&1; do
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Could not connect to Elasticsearch after $MAX_RETRIES retries."
        exit 1
    fi
    echo "⏳ Waiting for Elasticsearch... ($COUNT/$MAX_RETRIES)"
    sleep 2
done

echo "✅ Elasticsearch is ready!"
exec "$@"
