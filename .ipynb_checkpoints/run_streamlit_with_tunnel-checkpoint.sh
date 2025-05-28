#!/bin/bash

export STREAMLIT_WATCH_FILES=false

streamlit run app.py --server.address=0.0.0.0 --server.port=8510 --server.runOnSave false --server.fileWatcherType none > streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "Streamlit started with PID $STREAMLIT_PID"
sleep 5

cloudflared tunnel --url http://127.0.0.1:8510 --protocol http2 --loglevel info > cloudflared.log 2>&1 &
CLOUDFLARED_PID=$!
echo "Cloudflared tunnel started with PID $CLOUDFLARED_PID"
sleep 10

PUBLIC_URL=$(grep -oP 'https://[a-z0-9\-]+\.trycloudflare\.com' cloudflared.log | head -n1)

if [[ -z "$PUBLIC_URL" ]]; then
  echo "Could not find Cloudflare tunnel URL. Check cloudflared.log for details."
  exit 1
fi

echo "Cloudflare Tunnel is running. Access your Streamlit app at:"
echo "$PUBLIC_URL"
echo
echo "Press Ctrl+C to stop and exit, cleaning up all processes..."

trap "echo 'Stopping...'; kill $STREAMLIT_PID $CLOUDFLARED_PID; exit 0" SIGINT

while true; do
  sleep 5
done
