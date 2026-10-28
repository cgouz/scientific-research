# ggkip — common commands. Run `make` (or `make help`) for the list.
#
#   make start ENV=src/resources/env/llamacpp.env    # the GPU server setup
#   make batch COUNTRIES=242                          # crawl only these countries
#   make batch-status ID=1
#
# Variables (override on the command line): ENV is the app's --env-file, LOG
# is where `make start` writes, API is used by the API shortcuts, PG is the
# PostgreSQL container of `make db-up`.
ENV     ?= src/resources/env/local.env
LOG     ?= logs/ggkip.log
API     ?= http://localhost:8090
PG      ?= ggkip-pg
BIN     := bin/ggkip
PIDFILE := bin/ggkip.pid

SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help build run start stop restart status logs test race vet fmt fmt-check check clean \
	db-up db-down db-logs psql health batch batches batch-status batch-cancel runs export

help: ## list the targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z][a-z-]*:.*## / {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "  ENV=$(ENV)  LOG=$(LOG)  API=$(API)"

# ---------------------------------------------------------------- build & run

build: ## build bin/ggkip
	go build -trimpath -o $(BIN) ./src/main

run: build ## run in the foreground (Ctrl-C stops it, running crawls end 'cancelled')
	./$(BIN) --env-file $(ENV)

start: build ## run in the background, log to LOG
	@if [ -f $(PIDFILE) ] && kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
		echo "already running (pid $$(cat $(PIDFILE))); use 'make restart'"; exit 1; fi
	@mkdir -p $(dir $(LOG))
	@nohup ./$(BIN) --env-file $(ENV) >> $(LOG) 2>&1 & echo $$! > $(PIDFILE)
	@sleep 1
	@if kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
		echo "started pid $$(cat $(PIDFILE)) with $(ENV); logs: make logs"; \
	else echo "exited at once; last log lines:"; tail -20 $(LOG); rm -f $(PIDFILE); exit 1; fi

stop: ## stop the background app (SIGTERM, waits up to 30 s for crawls to close)
	@if [ ! -f $(PIDFILE) ]; then echo "not running (no $(PIDFILE))"; exit 0; fi; \
	pid=$$(cat $(PIDFILE)); \
	if ! kill -0 $$pid 2>/dev/null || [ "$$(ps -p $$pid -o comm=)" != "ggkip" ]; then \
		echo "pid $$pid is not ggkip any more, removing stale $(PIDFILE)"; rm -f $(PIDFILE); exit 0; fi; \
	kill -TERM $$pid; \
	for i in $$(seq 60); do kill -0 $$pid 2>/dev/null || break; sleep 0.5; done; \
	if kill -0 $$pid 2>/dev/null; then echo "still running after 30 s, killing"; kill -KILL $$pid; fi; \
	rm -f $(PIDFILE); echo "stopped"

restart: stop start ## stop, rebuild and start again

status: ## is it running, and does /health answer
	@if [ -f $(PIDFILE) ] && kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
		echo "running, pid $$(cat $(PIDFILE))"; else echo "not running"; fi
	@curl -s -m 3 $(API)/health || echo "$(API)/health not reachable"; echo

logs: ## follow the log of make start
	tail -f $(LOG)

# ---------------------------------------------------------------- quality

test: ## unit tests
	go test ./...

race: ## unit tests with the race detector
	go test -race ./...

vet: ## go vet
	go vet ./...

fmt: ## gofmt -w
	gofmt -w src

fmt-check: ## fail if a file needs gofmt
	@out=$$(gofmt -l src); if [ -n "$$out" ]; then echo "needs gofmt:"; echo "$$out"; exit 1; fi

check: fmt-check vet race build ## everything a commit should pass

clean: ## remove bin/
	rm -rf bin

# ---------------------------------------------------------------- database

db-up: ## start PostgreSQL 17 in Docker (ggkip/ggkip, port 5432)
	@if docker ps --format '{{.Names}}' | grep -qx $(PG); then echo "$(PG) is running"; \
	elif docker ps -a --format '{{.Names}}' | grep -qx $(PG); then docker start $(PG); \
	else docker run -d --name $(PG) -e POSTGRES_USER=ggkip -e POSTGRES_PASSWORD=ggkip \
		-e POSTGRES_DB=ggkip -p 5432:5432 postgres:17-alpine; fi

db-down: ## stop the PostgreSQL container (the data stays)
	docker stop $(PG)

db-logs: ## follow the PostgreSQL log
	docker logs -f $(PG)

psql: ## psql shell in the container
	docker exec -it $(PG) psql -U ggkip -d ggkip

# ---------------------------------------------------------------- API shortcuts

# $(call api,CURL-ARGS): prints the reply as indented JSON (as is when it is not
# JSON), or curl's error when the app does not answer.
define api
	@out=$$(curl -sS $(1)) || { echo "is the app running? (make status)"; exit 1; }; \
	echo "$$out" | python3 -m json.tool 2>/dev/null || echo "$$out"
endef

BATCH_BODY = $(if $(COUNTRIES),-d '{"country_ids":[$(COUNTRIES)]}')
RUNS_QUERY = limit=50$(if $(BATCH),&batch_id=$(BATCH))$(if $(SITE),&site_id=$(SITE))

health: ## GET /health
	$(call api,$(API)/health)

batch: ## crawl every active country (COUNTRIES=242,1 for only these)
	$(call api,-X POST $(BATCH_BODY) $(API)/v1/crawl-batches)

batches: ## list crawl batches
	$(call api,$(API)/v1/crawl-batches)

batch-status: ## one batch with runs per status (ID=)
	@test -n "$(ID)" || { echo "usage: make batch-status ID=1"; exit 2; }
	$(call api,$(API)/v1/crawl-batches/$(ID))

batch-cancel: ## cancel a running batch (ID=)
	@test -n "$(ID)" || { echo "usage: make batch-cancel ID=1"; exit 2; }
	$(call api,-X POST $(API)/v1/crawl-batches/$(ID)/cancel)

runs: ## crawl runs, newest first (BATCH= or SITE= to filter)
	$(call api,"$(API)/v1/crawl-runs?$(RUNS_QUERY)")

export: ## records to JSON: make export ARGS="--site-id 8 --out stat_uz.json"
	python3 scripts/export_records.py $(ARGS)
