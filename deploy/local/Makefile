.DEFAULT_GOAL := up
COMPOSE ?= podman compose
# The served model: qwen (config/server.json: the variants it lists, Qwen's and DeepSeek's, each
# config/{model,server}-<variant>.json) or another config/{model,server}-<name>.json pair served alone. STRATA_VARIANT
# picks the variant to start with (default: the first listed, q2).
STRATA_MODEL ?= qwen
export STRATA_MODEL
STRATA_VARIANT ?=
export STRATA_VARIANT

.PHONY: set-api-key set-hf-token migrate-keyring-secrets secrets config pull-config build init ports up down logs calibrate tune check
set-api-key:
	./scripts/credentials.sh store strata_api_key
set-hf-token:
	./scripts/credentials.sh store hf_token
# One-time copy of values stored by earlier versions in the GNOME keyring (it must be unlocked).
migrate-keyring-secrets:
	./scripts/credentials.sh migrate-keyring strata_api_key
	./scripts/credentials.sh migrate-keyring hf_token
secrets:
	./scripts/register-podman-secrets.sh
# Copy config/ into the volume mounted at /config; files removed from config/ stay in it. It refuses to overwrite a file
# the web page's Config view changed (pull-config first, or FORCE=1 to discard those changes).
config:
	./scripts/config-volume.sh push
# Copy the run configs the web page's Config view changed back into config/.
pull-config:
	./scripts/config-volume.sh pull
build:
	$(COMPOSE) build model-init
init: secrets config build
	$(COMPOSE) run --rm --no-deps -T model-init
up: init
	$(COMPOSE) up -d --no-deps --force-recreate strata-server
down:
	$(COMPOSE) down
logs:
	$(COMPOSE) logs -f strata-server
calibrate:
	$(COMPOSE) stop strata-server
	$(COMPOSE) run --rm --no-deps -T --entrypoint python strata-server /usr/local/lib/strata-local/calibrate.py
# Calibrate a hipBLASLt table for this card and the image's hipBLASLt, over the 43 rows of upstream's gfx1100 1.4.1
# table (solution ids measured fresh); writes config/<arch>-hipblaslt-<version>.txt. Point
# engine_env.hipblaslt_tuning (STRATA_HIPBLASLT_TUNING) in config/server.json at it, then make up.
tune: build
	$(COMPOSE) stop strata-server
	$(COMPOSE) run --rm --no-deps -T --entrypoint sh strata-server /usr/local/lib/strata-local/tune-hipblaslt.sh \
		> config/.hipblaslt-table.tmp
	name=$$(awk 'NR == 2 && $$1 == "STRATA_HIPBLASLT_TUNING_V1" {print $$2 "-hipblaslt-" $$3 ".txt"}' config/.hipblaslt-table.tmp) \
		&& test -n "$$name" && mv config/.hipblaslt-table.tmp "config/$$name" && echo "wrote config/$$name"
check:
	python3 -m unittest discover -s tests -v
	sh -n scripts/credentials.sh scripts/register-podman-secrets.sh scripts/podman-secret-driver.sh scripts/config-volume.sh
	podman-compose -f compose.yaml config >/dev/null
