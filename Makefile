HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: run dev stop

run:  ## Start the API server (api.app:app)
	uvicorn api.app:app --host $(HOST) --port $(PORT)

dev:  ## Start the API server with auto-reload
	uvicorn api.app:app --host $(HOST) --port $(PORT) --reload

stop:  ## Kill whatever is listening on $(PORT)
	@PID=$$(lsof -tiTCP:$(PORT) -sTCP:LISTEN 2>/dev/null); \
	if [ -n "$$PID" ]; then kill $$PID && echo "killed $$PID"; else echo "nothing on port $(PORT)"; fi
