# Pipeline + CLI + DigitalOcean deployment options

This repository has **two “execution styles”**:

1. **Python scripts orchestration** (preferred for deployment): [scripts/gold/pipeline.py](scripts/gold/pipeline.py)
2. **Production Python subproject** (importable + testable): [pipeline/](pipeline/), exposing a CLI entrypoint `ni-ai-pipeline` via [pipeline/src/ni_ai_pipeline/cli.py](pipeline/src/ni_ai_pipeline/cli.py)

Notebooks under `scripts/**.ipynb` are kept for convenience/interactive development, but the `.py` pipeline is the preferred “source of truth”.

The DigitalOcean story depends on which style you want to operate.

---

## Is the CLI needed?

### No — if you only run notebooks manually

If your workflow is “open notebooks in VS Code, run cells, inspect outputs”, then the CLI is **not required**.

### Not strictly required — but very useful for automation

The CLI is mainly a **thin wrapper** that calls the same Python functions you could import directly.

What it buys you:

- **A stable entrypoint** for automation/scheduling (`ni-ai-pipeline silver-weather`, `... gold-data-full`, `... run-all`, `... train-ecoli`).
- **Consistency** across environments (local, server, container) without “which notebook should I run?” decisions.
- **Fewer moving parts** than notebook execution in production (no Jupyter kernel state).
- **Better operability**: exit codes, stdout logs, easier systemd/cron integration.
- **Optional MLflow logging** for training (via `ni-ai-pipeline train-ecoli --mlflow ...` in [pipeline/src/ni_ai_pipeline/training/ecoli_predictability.py](pipeline/src/ni_ai_pipeline/training/ecoli_predictability.py)).

### When you’d skip the CLI even in production

- If you already have an orchestrator that imports Python modules directly (e.g. your own runner script, Airflow DAG) and you don’t want an extra CLI layer.
- If you only ever run **one** fixed command and don’t benefit from subcommands.

---

## What the CLI actually does in this repo

The CLI is a small `argparse` dispatcher that:

- loads path configuration from environment variables (optionally `.env`) via `get_paths(load_dotenv=True)`
- calls the corresponding pipeline step function

Commands:

- `ni-ai-pipeline silver-weather`
- `ni-ai-pipeline silver-messungen`
- `ni-ai-pipeline gold-data-full`
- `ni-ai-pipeline gold-masterdata`
- `ni-ai-pipeline gold-daily-dataset`
- `ni-ai-pipeline train-ecoli` (optional MLflow logging)
- `ni-ai-pipeline run-all` (runs Silver + Gold; does not run training)

For environment variables and defaults, see [pipeline/src/ni_ai_pipeline/paths.py](pipeline/src/ni_ai_pipeline/paths.py) and [pipeline/README.md](pipeline/README.md).

---

## “Serving” on DigitalOcean: what does that mean here?

This codebase is primarily a **batch pipeline** (ETL-style) plus optional training/evaluation.

So “serving it” can mean one of two things:

1. **Run it on a schedule / on demand** (most common for pipelines)
2. **Expose an HTTP endpoint** that triggers a run (only needed if you want remote triggering)

The options below cover both.

---

## Option A (recommended): Droplet (VM) + systemd/cron for scheduled runs

Best when:

- you want the simplest, most controllable setup
- you’re okay managing a Linux VM
- pipeline runs are periodic (hourly/daily) and write to disk

High-level approach:

1. Create a **Droplet** (Ubuntu)
2. Put the repo on the droplet (git clone, or deploy artifact)
3. Create a Python virtualenv and install the pipeline subproject
4. Configure environment variables (paths)
5. Schedule runs with **cron** or **systemd timers**

Example install steps (conceptual):

- `python -m pip install -e ./pipeline[train]`

If you want MLflow logging during training:

- `python -m pip install -e ./pipeline[train,mlflow]`

Scheduling choices:

- **cron**: simplest for “run daily at 03:00”
- **systemd timer**: better logging + more robust process supervision patterns

Operational notes:

- Put data directories (`DATA_BRONZE`, `DATA_SILVER`, `DATA_GOLD`) on a **mounted volume** if you want persistence across redeploys.
- Prefer real environment variables (systemd unit `Environment=` or `EnvironmentFile=`) over relying on `.env`.

---

## Option B: Droplet + Docker (containerized batch runs)

Best when:

- you want repeatable builds and fewer “works on my VM” issues
- you may later move to App Platform or Kubernetes

How it typically looks:

- Build a Docker image that contains the `pipeline/` package
- Run the container with env vars + mounted data volumes
- Schedule with cron/systemd on the droplet

You still keep the same entrypoint:

- `ni-ai-pipeline run-all`
- `ni-ai-pipeline train-ecoli --mlflow ...`

Notes:

- You’ll need to decide where `mlruns` (if using local MLflow file store) lives: mount a volume so MLflow artifacts survive container recreation.

---

## Option C: DigitalOcean App Platform (web service or worker)

Best when:

- you prefer “platform-managed” deployments
- you want easy deploys from Git

Reality check for this repo:

- App Platform is excellent for **always-on web services**.
- A data pipeline is usually a **job** (runs, exits). Whether App Platform is the right fit depends on whether you have a job/scheduled-job feature in your plan and whether runtime limits fit your workload.

Two ways people do this:

1. **Worker pattern**: run a long-lived process that waits for work (queue). This repo does not currently implement a queue consumer.
2. **Job pattern**: run `ni-ai-pipeline ...` as a one-off job/scheduled job (if available).

If you go this way, the CLI helps because it’s a clean single command to run per job.

---

## Option D: DigitalOcean Kubernetes (DOKS) + CronJobs

Best when:

- you want robust scheduling, retries, parallelism
- you already use Kubernetes

How it maps:

- Container image runs one command like `ni-ai-pipeline run-all`
- A Kubernetes **CronJob** triggers it on a schedule
- Use PersistentVolumeClaims (or an external store) for data persistence

Tradeoff:

- More operational complexity than a Droplet

---

## Option E: “Serve it” as an HTTP API (only if you need remote triggering)

Best when:

- another system needs to trigger pipeline runs via HTTP
- you need authentication/rate limiting/audit logging around triggers

Typical pattern:

- Create a small API service (e.g., FastAPI) that triggers pipeline execution by:
  - importing and calling the step functions directly, OR
  - shelling out to `ni-ai-pipeline ...`

Where to host the API:

- Droplet (simplest)
- App Platform (often a good fit)
- Kubernetes

Important caveat:

- A pipeline run can be long-running. If triggered over HTTP, you usually want an **async job model** (enqueue work, return a job id) rather than keeping an HTTP request open.

This repo does not currently implement an API or a queue, so this is an architectural option rather than “already supported”.

---

## Data + state on DigitalOcean (applies to all options)

This pipeline reads/writes mostly local files under `data/` by default.

On a server you should decide where those files live:

- **Local disk**: simplest, but ties state to a single machine
- **Block Storage Volume** attached to Droplet: good for persistence + VM replacement
- **Object storage (Spaces)**: good for artifacts/exports; requires code changes if you want the pipeline to read/write directly from Spaces rather than local paths

MLflow (if you use it) also introduces state:

- Local tracking store (`file:./mlruns`) is fine for one machine.
- For multi-machine setups, a server-backed tracking store + shared artifact store is more typical.

---

## Quick recommendation matrix

- **Single machine, scheduled batch pipeline**: Droplet + systemd timer (Option A)
- **Want reproducible runtime + easier migration**: Droplet + Docker (Option B)
- **Already on Kubernetes / need retries + scaling**: DOKS CronJobs (Option D)
- **Need remote triggering by other systems**: add a small API wrapper and host it (Option E)

---

## What I can do next (pick one)

If you tell me which deployment target you prefer (Droplet / Docker / App Platform / Kubernetes), I can:

- write a minimal runbook section into this doc with concrete commands, or
- add deployment files (e.g., a `Dockerfile`, a systemd unit/timer template), keeping it minimal and consistent with the repo style.
