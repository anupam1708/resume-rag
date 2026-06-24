# AWS Deployment — Detailed Explanation

A walkthrough of the ten-phase production deployment design in
[`AWS_DEPLOYMENT.md`](AWS_DEPLOYMENT.md), expanding each phase with the reasoning,
tradeoffs, and connections back to the Resume RAG codebase.

> **Note:** This is an **aspirational reference design**. As stated in `CLAUDE.md`,
> none of this AWS infrastructure is actually provisioned — it is a blueprint that
> demonstrates production-level thinking. Items marked as "new code" do not exist in
> the current repository.

---

## Table of contents

1. [Phase 1: Containerize the Application](#phase-1-containerize-the-application)
2. [Phase 2: Managed Database — RDS PostgreSQL + pgvector](#phase-2-managed-database--rds-postgresql--pgvector)
3. [Phase 3: Compute — ECS Fargate](#phase-3-compute--ecs-fargate)
4. [Phase 4: Networking and Ingress](#phase-4-networking-and-ingress)
5. [Phase 5: Secrets and Configuration](#phase-5-secrets-and-configuration)
6. [Phase 6: Caching Layer — ElastiCache Redis](#phase-6-caching-layer--elasticache-redis)
7. [Phase 7: Observability](#phase-7-observability)
8. [Phase 8: CI/CD Pipeline](#phase-8-cicd-pipeline)
9. [Phase 9: Infrastructure as Code](#phase-9-infrastructure-as-code)
10. [Phase 10: Cost Estimation](#phase-10-cost-estimation)

---

## Phase 1: Containerize the Application

### The core idea: split one app into two images

Locally, this project runs as a single codebase where you start the API
(`uvicorn src.api:app`) and run ingestion (`python -m src.ingestion`) by hand. Phase 1
packages these as **two separate Docker images**, both pushed to **ECR** (Elastic
Container Registry, AWS's private Docker registry).

| Image | Command it runs | Code involved |
|-------|-----------------|---------------|
| `api` | `uvicorn src.api:app` — serves `POST /query` and `GET /health` | `src/api.py` → `src/retrieval.py` + `src/generation.py` |
| `ingest-worker` | `python -m src.ingestion` — batch extract + embed resumes | `src/ingestion.py` → `src/extraction.py` |

### Why split them?

They have fundamentally different **compute profiles**:

- The **API** is request-driven and latency-sensitive. You want many small replicas
  that scale up/down with traffic (horizontal scaling). It does lightweight work per
  request: embed the query, hit Postgres, call Claude.
- The **ingest-worker** is a heavy batch job. For every resume it makes Claude
  extraction calls and runs sentence-transformers embedding over multiple chunks. It
  runs on a schedule (or on-demand when new resumes arrive), not per user request.

If they shared one process/container, a big ingest run would **steal CPU and memory
from live query traffic**, hurting API latency. Splitting them lets each scale and be
scheduled independently.

### The embedding model and cold starts

Both containers need the **sentence-transformers model** (`all-MiniLM-L6-v2`):
- The API needs it to embed incoming queries (`vector_search` in `retrieval.py`).
- The worker needs it to embed resume chunks (`ingestion.py`).

Key facts:
- It's only **~80 MB** and produces **384-dim** vectors, so it runs fine **on CPU — no
  GPU needed**, keeping infrastructure cheap and simple.
- By default, `SentenceTransformer('all-MiniLM-L6-v2')` **downloads the weights from
  HuggingFace on first use**. In production that means every fresh container would stall
  on a network download before serving traffic — a slow, fragile **cold start** that
  also depends on HuggingFace being up.

### The fix: bake the model into the image at build time

```dockerfile
# Stage 1 (builder): install the lib and trigger the model download
FROM python:3.13-slim AS builder
RUN pip install sentence-transformers
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Stage 2 (runtime): slim final image
FROM python:3.13-slim
COPY --from=builder /root/.cache/torch /root/.cache/torch   # <- pre-downloaded weights
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
```

How it works:
1. **Builder stage** installs `sentence-transformers` and runs a throwaway one-liner
   whose only purpose is the side effect of **downloading the model weights** into
   `~/.cache/torch` during the build.
2. **Runtime stage** starts from a clean slim base and **copies just the cached
   weights** out of the builder with `COPY --from=builder`, then installs the real
   `requirements.txt` and app code.

The **multi-stage** pattern keeps build artifacts and pip caches out of the shipped
image, so the final image stays small while still having the weights pre-baked. The
result: a new container loads the model from local cache instantly, with **no network
dependency on HuggingFace** at startup.

### Why this phase matters

It demonstrates reasoning about **cold-start latency** (pre-baking weights), **image
size** (multi-stage builds), and **separation of compute profiles** (not letting a
batch job contend with latency-sensitive serving).

> The Dockerfile is illustrative — it omits the differing `CMD`/entrypoint per image
> (uvicorn vs. the ingestion module), which Phase 3's ECS task definitions handle. Its
> job is to show the model-caching technique, not to be a complete production Dockerfile.

---

## Phase 2: Managed Database — RDS PostgreSQL + pgvector

### The core idea: stop running your own database

Locally this project uses the Postgres container in `docker-compose.yml`
(`pgvector/pgvector:pg16`). Phase 2 replaces that self-managed container with **Amazon
RDS for PostgreSQL 16**, AWS's managed database service. RDS handles provisioning,
patching, backups, and failover while keeping the same Postgres 16 + pgvector engine
the code already targets.

Critically, **no application code changes**. The app talks to Postgres through
`DATABASE_URL` (`src/config.py` → `src/db.py`). Migrating to RDS is just pointing that
connection string at the RDS endpoint instead of `localhost:5432`.

### The configuration table

| Setting | Value | Rationale |
|---------|-------|-----------|
| Instance | `db.r6g.large` (2 vCPU, 16 GB RAM) | HNSW index lives in memory; r6g gives best memory/$ |
| Storage | gp3, 100 GB, autoscaling to 500 GB | gp3 gives consistent IOPS without provisioned cost |
| Multi-AZ | Yes | Automatic failover, ~30s downtime on primary failure |
| pgvector | `CREATE EXTENSION vector;` in parameter group | RDS supports pgvector natively since PG15 |
| Encryption | At-rest (KMS) + in-transit (SSL enforce) | Compliance baseline |
| Backups | 7-day automated + manual pre-migration | Point-in-time recovery |

- **`db.r6g.large`** — the `r6g` family is **memory-optimized** (ARM/Graviton, best
  memory-per-dollar). Memory is the priority because the **HNSW index** on
  `chunks.embedding` (in `schema.sql`) is a graph used for fast approximate
  nearest-neighbor vector search, and it's only fast when it lives **in RAM**. The
  instance is sized to keep the vector index warm.
- **gp3 storage** — IOPS and throughput are configured independently of size, giving
  consistent I/O without over-allocating disk. Autoscaling to 500 GB means storage
  grows with the corpus without pre-paying.
- **Multi-AZ** — RDS keeps a **synchronous standby in a second Availability Zone**; on
  primary/AZ failure it fails over automatically (~30s downtime).
- **pgvector** — adds the `vector(384)` type and the `<=>` cosine-distance operator
  `vector_search()` relies on. Supported natively on RDS since PostgreSQL 15.
- **Encryption** — KMS at-rest (encrypted volumes/backups) and enforced SSL in-transit;
  resumes are PII, so this is the compliance baseline.
- **Backups** — daily snapshots + transaction logs enable point-in-time recovery; the
  manual pre-migration snapshot is a rollback safety net before schema changes.

### Run schema.sql as a real migration

Apply `schema.sql` (the `candidates` + `chunks` tables with HNSW and GIN indexes) using
a **migration tool like Alembic or Flyway**, not by hand. This makes schema changes
**versioned, repeatable, and auditable** — important given that changing extracted
fields (per `CLAUDE.md`) requires coordinated edits to `schema.sql`, `ingestion.py`, and
`extraction.py`.

### The design decision: why not Aurora Serverless?

1. **Sustained query load** — this serves a steady stream of recruiter queries, where
   provisioned RDS gives predictable performance and cost; serverless shines for
   spiky/intermittent workloads.
2. **Warm memory requirement** — the whole point of the `r6g` sizing is keeping the
   HNSW index in RAM. If Aurora Serverless scales down/to zero during a lull, the next
   query must cold-start and reload that index — slow. A provisioned instance stays warm.

---

## Phase 3: Compute — ECS Fargate

### The core idea: where the containers run

Phase 1 built two images; Phase 2 set up the database. Phase 3 runs the containers on
**ECS Fargate**:
- **ECS** (Elastic Container Service) is AWS's container orchestrator.
- **Fargate** is the serverless launch mode — you declare CPU/memory and AWS provisions
  the compute; **you never manage EC2 instances**.

Two ECS concepts used here:
- **Task definition** — the blueprint for one container (image, CPU/memory, env vars,
  IAM role).
- **Service** — keeps N copies of a task running, replaces unhealthy ones, integrates
  with the load balancer and auto-scaling.

### API Service

- **Task definition:** 1 vCPU, 2 GB memory (FastAPI + the baked-in ~80 MB model fit
  comfortably; scale by running *more small tasks*, not bigger ones).
- **Service:** desired 2, min 2, max 10.
  - **Min 2** across two AZs → no single task/AZ failure takes the API down.
  - **Max 10** caps scale-out (cost ceiling).
- **Auto-scaling:** target tracking on **`ECSServiceAverageCPUUtilization` at 60%** plus
  ALB **`RequestCountPerTarget`**.
  - 60% (not 90%) leaves headroom for spikes while new tasks spin up.
  - Request count catches load that CPU misses — requests that wait on Claude
    (`generation.py`) spend time on I/O, not CPU.
- **Health check:** the ALB hits **`GET /health`** every 15s; ECS uses the same for task
  health. This is why `src/api.py` defines a trivial `/health` returning
  `{"status": "ok"}` — cheap and reliable, with no DB/LLM call. Unhealthy tasks are
  pulled and replaced automatically.

### Ingest Worker

- **Triggered, not always-on** — via **EventBridge** (e.g., daily at 2 AM UTC) or by an
  **S3 event** when new resumes land in a bucket.
- **Run to completion, then stop** — zero cost when idle. This fits the **idempotent**
  ingestion design (per `CLAUDE.md`, it skips candidates already present by
  `candidate_id`), so scheduled/per-upload runs are safe and only process new resumes.
- **ECS task role** scoped to least privilege: **read S3** (new resumes), **write RDS**
  (insert candidates/embeddings), **call Anthropic** (extraction in `extraction.py`).

### Key design decisions

- **Why Fargate over EKS (Kubernetes)?** For two services with simple networking, EKS's
  control-plane overhead isn't worth it. Rule of thumb: reach for Kubernetes at 10+
  services, not 2.
- **Why Fargate over Lambda?** The embedding model takes ~500 MB and several seconds to
  load. Lambda would pay that cold-start cost repeatedly; **Fargate tasks stay warm**,
  loading the model once and keeping it resident.

The throughline across Phases 1–3: **keep the expensive-to-load model warm and
resident**, and **match each workload to the right compute shape**.

---

## Phase 4: Networking and Ingress

### The core idea: safe traffic flow + locked-down everything else

The guiding principle is **defense in depth + least-privilege networking**: only the
front door is public; every other component sits in a private network and accepts
traffic only from the component directly upstream.

### The request path

```
Route 53 (DNS)
  → ALB (public subnet, HTTPS only via ACM cert)
    → Target Group (private subnet, ECS API tasks on port 8000)
```

1. **Route 53** resolves your domain to the load balancer.
2. **ALB** (Application Load Balancer) is the single public entry point; it terminates
   TLS and forwards inward.
3. **Target Group** is the ALB's pool of backends — the ECS API tasks on **port 8000**
   (uvicorn), in a **private** subnet.

### The configuration table

| Component | Configuration |
|-----------|---------------|
| **VPC** | 2 AZs, public + private subnets in each |
| **ALB** | Public subnet, TLS termination with ACM certificate |
| **API tasks** | Private subnet, no public IP |
| **RDS** | Private subnet, security group allows only ECS task SG on port 5432 |
| **NAT Gateway** | One per AZ — API tasks need outbound to reach Anthropic API |

- **VPC** — your isolated network, spanning 2 AZs for HA. Each AZ has a **public**
  subnet (route to internet, hosts the ALB) and a **private** subnet (no inbound
  internet route, hosts API tasks and RDS).
- **ALB** — only internet-facing component; does **TLS termination** with an **ACM**
  cert (auto-issued/renewed). HTTPS-only, so PII is encrypted in transit to the LB.
- **API tasks** — **no public IP**; reachable only through the ALB.
- **RDS** — locked tightest: its **security group** allows inbound **only from the API
  tasks' SG, only on port 5432**. This is what `DATABASE_URL` connects to.
- **NAT Gateway** — lets private-subnet tasks make **outbound** calls to the **Anthropic
  API** (routing in `retrieval.py`, generation in `generation.py`) while blocking all
  inbound. One per AZ avoids a cross-AZ single point of failure.

### The security group chain

> ALB → API (port 8000) → RDS (port 5432). Nothing else is open.

Each hop accepts traffic only from the component directly in front of it. If any layer
is compromised, the blast radius is limited to the next link. This least-privilege
topology is the difference between a junior "deploy to EC2 with a public IP" answer and
a principal-level design.

---

## Phase 5: Secrets and Configuration

### The core idea: get credentials out of code and image

Secrets should never be committed to the repo, baked into a Docker image, or passed as
build args. They live in managed AWS stores and are injected at startup. Because
**all config flows through `src/config.py`** (which reads `os.environ`), Phase 5 only
swaps the *source* of those env vars — **with zero application code changes**.

### Two stores for two kinds of values

| Secret | Store | How injected |
|--------|-------|--------------|
| `ANTHROPIC_API_KEY` | Secrets Manager | ECS task definition `secrets` block |
| `DATABASE_URL` | Secrets Manager (auto-rotated with RDS integration) | ECS task definition `secrets` block |
| `LANGSMITH_API_KEY` | Secrets Manager | ECS task definition `secrets` block |
| `EMBED_MODEL`, `GEN_MODEL` | SSM Parameter Store | ECS task definition `environment` block |

- **AWS Secrets Manager** — for sensitive values (API keys, DB credentials): encrypted,
  IAM-controlled, audited, supports rotation.
- **SSM Parameter Store** — for non-sensitive config (model IDs): cheaper, simpler.

Notes:
- **`ANTHROPIC_API_KEY`** authorizes every Claude call (`extraction.py`, `retrieval.py`,
  `generation.py`).
- **`DATABASE_URL`** contains the DB password; **auto-rotation via RDS integration**
  means rotated credentials propagate without manual edits.
- **`LANGSMITH_API_KEY`** is optional (`config.py` uses `os.environ.get(...)`).
- **`EMBED_MODEL`, `GEN_MODEL`** are non-secret identifiers.

> ⚠️ The table lists `EMBED_MODEL` as runtime-tunable, but `CLAUDE.md` warns it's coupled
> to `EMBED_DIM` and the `vector(384)` column in `schema.sql` and requires a re-ingest.
> `GEN_MODEL` is safely swappable; `EMBED_MODEL` is not a free config flip.

### The `secrets` block vs. the `environment` block

An ECS task definition can set env vars two ways: the **`secrets` block** references a
Secrets Manager/Parameter Store entry by ARN and injects its value at startup (the value
never appears in the task definition), while the **`environment` block** holds plain
key/values. Either way the value lands as an ordinary env var — exactly what `config.py`
reads — so **no app code changes** are needed.

### Why this matters

Never bake secrets into images or build args: image layers are cached and distributed,
build args appear in metadata/logs, and anyone who can pull the image would get the
credentials. Runtime injection from an access-controlled, auditable, rotatable store
gives least-privilege credential handling that pairs with Phase 4's networking and
Phase 3's scoped task role.

---

## Phase 6: Caching Layer — ElastiCache Redis

### The core idea: stop re-computing identical queries

Every query triggers **Claude Haiku for routing** → **pgvector/SQL retrieval** →
**Claude Sonnet for generation**. Phase 6 puts a **Redis cache** (via **ElastiCache**,
AWS's managed Redis) in front of the whole pipeline so repeated queries return instantly
from memory.

### Why cache?

Each query costs roughly **$0.01–0.03 in LLM spend**, and recruiters **search the same
skills repeatedly**. Without caching, ten identical queries pay ten times and each waits
several seconds. Caching buys **both cost and latency** reductions.

### Cache strategy

- **Key:** hash of `(query_text, mode)` — the retriever is deterministic for a given
  query, and `mode` (`filter`/`semantic`/`hybrid`) routes down different paths in
  `retrieve()`, so it must be part of the key.
- **TTL:** 1 hour — resumes change infrequently, so an hour-old answer is nearly always
  valid, while the TTL prevents indefinitely stale results.
- **Invalidation:** bust the cache after the **ingest worker** finishes (publish to a
  Redis channel or `FLUSHDB` the cache namespace). Ingestion is the only thing that
  changes the corpus, so it's the natural invalidation trigger.

### Where it goes in the code

Wrap **`retrieval.retrieve()` + `generation.generate()`** as a unit and cache the full
`{candidates, answer}`. In `src/api.py`'s `query_endpoint`:

```python
result = retrieve(q.query, mode=q.mode, top_k=q.top_k)
answer = generate(q.query, result["candidates"])
```

On a hit, return the stored result and skip both retrieval and both LLM calls; on a
miss, run the pipeline and store with the 1-hour TTL. Caching the **combined** output
(not just retrieval) is what skips the expensive Sonnet generation call.

> One gap: the proposed key omits **`top_k`**, which also changes the result. A real
> implementation would likely include it. Also note nothing in the current
> `requirements.txt`/`src/` implements Redis yet — this is new code.

---

## Phase 7: Observability

### The core idea: know what your system is doing in production

A layered stack covering the three pillars — **logs, metrics, traces** — plus **alarms**
and **dashboards**. Each layer answers a different question; together they let you both
*detect* and *diagnose* problems.

| Layer | Tool | What it watches |
|-------|------|-----------------|
| **Logs** | CloudWatch Logs | Structured JSON from FastAPI (request ID, latency, route mode, candidate count) |
| **Metrics** | CloudWatch Metrics | p50/p95/p99 latency, error rate, cache hit ratio, Anthropic API latency |
| **Traces** | LangSmith | End-to-end LLM traces (already configured via `LANGSMITH_TRACING=true`) |
| **Alarms** | CloudWatch Alarms → SNS → PagerDuty/Slack | 5xx rate > 5%, p99 > 10s, Anthropic 429s |
| **Dashboards** | CloudWatch Dashboard | Request volume, latency distribution, retrieval mode breakdown, cost estimate |

- **Logs** — *structured* JSON is queryable; fields map to the request flow (`route
  mode` = the router decision, `candidate count` = results from `retrieve()`). Per-event
  ground truth for debugging a specific request. *(New code — `src/api.py` doesn't log
  yet.)*
- **Metrics** — trends over time. Percentiles expose the slow tail (p99 = slowest 1%,
  often cache-miss requests hitting both Claude calls). **Cache hit ratio** validates
  Phase 6. **Anthropic API latency** isolates "is Claude slow?" from "is our code slow?"
- **Traces** — the deepest layer, **already wired** via `LANGSMITH_TRACING=true`
  (`.env.example`; LangSmith already powers `evals/run_evals.py`). Shows router decision
  → retrieved chunks → what the LLM saw → answer.
- **Alarms** — metrics → **SNS** → PagerDuty/Slack. Trip conditions: 5xx > 5%, p99 >
  10s, Anthropic 429s (rate limiting). Turns a bad metric into a human notification.
- **Dashboards** — single pane of glass for at-a-glance health.

### The standout insight: retrieval mode distribution

The key metric to watch is **what fraction of queries the router sends to
`filter`/`semantic`/`hybrid`**. If ~90% go to `filter`, the hybrid pipeline is
underutilized — investigate the `ROUTER_PROMPT` in `src/retrieval.py`. Since the project's
whole thesis is that hybrid retrieval beats pure approaches, this metric tells you
whether the core architectural bet is paying off in production.

The diagnostic flow ties the layers together: an **alarm** fires → check the
**dashboard** → drill into **metrics** → pull **logs** by request ID → open the
**LangSmith trace** for root cause.

---

## Phase 8: CI/CD Pipeline

### The core idea: automate commit → production, safely

A pipeline on **GitHub Actions** that tests, builds, and deploys, with guardrails that
can auto-reject a bad release. (Per `CLAUDE.md`, the repo currently has **no CI** — this
is net-new.)

```
GitHub Push → GitHub Actions
  → Run tests + evals (run_evals.py against staging DB)
  → Build Docker image → Push to ECR
  → ECS Blue/Green Deploy via CodeDeploy
    → Route 10% traffic to new tasks
    → Monitor error rate for 5 minutes
    → Promote to 100% or auto-rollback
```

1. **GitHub Push → Actions** — a push triggers the workflow.
2. **Tests + evals against a staging DB** — runs `evals/run_evals.py` against a
   production-like staging DB. It's not enough that the code runs; retrieval quality
   must hold.
3. **Build image → push to ECR** — the Phase 1 multi-stage image (model baked in), the
   immutable deploy artifact.
4. **Blue/Green Deploy via CodeDeploy** — stand up the new version (**green**) beside
   the current (**blue**), route **10% canary traffic**, **monitor error rate for 5
   min**, then **promote to 100% or auto-rollback**. The bad version never reaches most
   users.

### Key design decisions

- **Why blue/green over rolling?** The embedding model is **baked into the image**
  (Phase 1). A rolling deploy runs old and new simultaneously; if the embedding model
  changed, some tasks would embed queries with the old model and some the new one,
  giving **inconsistent retrieval**. Blue/green gives an **atomic cutover** — 100% of
  traffic on one consistent version. This ties to the `EMBED_MODEL` coupling warning in
  `CLAUDE.md`.
- **Run evals as a deploy gate** — `run_evals.py` already computes recall@10,
  precision@10, hit@10. Gate: **if `hybrid-rrf` recall@10 drops below baseline, block
  the deploy.** This catches **silent quality regressions** (a prompt tweak or model
  swap that passes all tests but degrades answers) that normal CI would miss. It
  operationalizes the project's core thesis.

> Caveat (noted in the harness itself): eval ground truth is **category-based and
> approximate**, not hand-labeled — fine as a regression tripwire, but a real production
> gate would want stronger labels.

---

## Phase 9: Infrastructure as Code

### The core idea: define the whole environment in version-controlled files

Instead of clicking through the Console, **declare every resource in code** and let
**Terraform** or **AWS CDK** make AWS match it. This is a meta-phase: it changes *how*
everything from Phases 1–8 comes into existence.

- **Terraform** — cloud-agnostic, declarative `.tf` files (the example uses this).
- **AWS CDK** — define infra in a real language (TypeScript, Python) compiled to
  CloudFormation.

### The `infra/` layout — one file per concern

```
infra/
  vpc.tf            # VPC, subnets, NAT, security groups        (Phase 4)
  rds.tf            # PostgreSQL instance, parameter group       (Phase 2)
  ecs.tf            # Cluster, task definitions, services, ASG   (Phase 3)
  alb.tf            # Load balancer, target groups, listeners    (Phase 4)
  ecr.tf            # Container registries                       (Phase 1)
  secrets.tf        # Secrets Manager entries                    (Phase 5)
  elasticache.tf    # Redis cluster                              (Phase 6)
  monitoring.tf     # CloudWatch dashboards, alarms, SNS topics  (Phase 7)
  codedeploy.tf     # Blue/green deployment configuration        (Phase 8)
```

Every architectural decision from Phases 1–8 has a corresponding file, making the entire
guide **executable** and keeping the document and real infrastructure in sync. Separate
files keep each concern independently reviewable — the same modular instinct as `src/`.

### Why IaC?

> "You must be able to reproduce the entire environment from scratch. 'I clicked through
> the console' is not an acceptable answer at any level."

- **Reproducibility** — spin up an identical environment on demand (e.g., the Phase 8
  staging environment).
- **Disaster recovery** — re-apply the code to rebuild after a loss.
- **Auditability** — infra changes go through Git: PRs, review, history, rollback.
- **No drift / no snowflakes** — code is the single source of truth; reality is forced
  to match.
- **Consistency** — dev/staging/prod built from the same parameterized definitions.

IaC pairs with Phase 8's CI/CD: the same "a machine reproduces this from declarative
source" philosophy applies to both the application and the infrastructure.

---

## Phase 10: Cost Estimation

### The core idea: know what it costs and what drives the cost

A monthly price tag on every component, separating **fixed** from **variable** costs.

| Service | Monthly estimate | Notes |
|---------|------------------|-------|
| ECS Fargate (API, 2 tasks) | ~$70 | 1 vCPU, 2 GB each, always on |
| ECS Fargate (Ingest, on-demand) | ~$5 | Runs ~1 hr/day |
| RDS db.r6g.large Multi-AZ | ~$350 | Biggest fixed cost |
| ALB | ~$25 | + per-request charges |
| NAT Gateway | ~$65 | 2 AZs |
| ElastiCache (cache.t4g.micro) | ~$12 | Smallest Redis |
| Secrets Manager | ~$3 | 4 secrets |
| ECR | ~$1 | Image storage |
| **Anthropic API** | **$50–500** | **Scales with query volume** |
| **Total** | **~$580–1,030/mo** | |

- The **ingest worker** is cheap (~$5) *because* it's run-to-completion (Phase 3), not
  always-on.
- **RDS Multi-AZ (~$350)** is the biggest fixed cost — Multi-AZ doubles the instance
  cost, and the memory-optimized sizing (warm HNSW index) isn't cheap.
- **NAT Gateway (~$65)** is chunky because there are two (one per AZ for HA).
- **Anthropic API ($50–500)** is the only highly variable line.

### The key insight: fixed floor vs. variable driver

Everything except Anthropic is a **fixed floor of ~$530/month** — paid whether you serve
10 or 10,000 queries. The **Anthropic API scales with usage** and stretches the total to
~$1,030. As the doc notes, *the Redis cache directly reduces this* — closing the loop on
Phase 6: caching is the primary lever on the *variable* cost (every hit is an LLM call
you don't pay for). Structural costs are reduced by re-architecting (dropping Multi-AZ,
one NAT) at the price of resilience; the Anthropic cost is reduced by operating smarter
(caching) with no resilience tradeoff.

### Closing: what a principal-level answer emphasizes

1. **Separation of concerns** — API vs. worker, not a monolith on one EC2 (Phases 1, 3).
2. **Security by default** — private subnets, Secrets Manager, TLS everywhere, SG
   chaining (Phases 4, 5).
3. **Operational maturity** — IaC, blue/green deploys, eval gates in CI, structured
   observability (Phases 7, 8, 9).
4. **Cost awareness** — right-sized instances, caching to cut LLM spend, Fargate over
   EKS (Phases 3, 6, 10).
5. **Tradeoffs articulated** — why Fargate not Lambda, why RDS not Aurora Serverless, why
   blue/green not rolling.

The real thesis of the whole guide: the value isn't listing AWS services, it's
**explaining the reasoning and tradeoffs behind each choice**.
