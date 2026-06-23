# AWS Production Deployment Guide — Resume RAG

## Architecture Overview

```
Users / Internet
       │
   Route 53 (DNS)
       │
   ALB (HTTPS, ACM TLS cert)
       │                          ┌──────────────────┐
       ▼                          │   Anthropic API   │
┌─────────────────────────────────┤  Claude Sonnet    │
│  VPC — PRIVATE SUBNETS          │  Claude Haiku     │
│                                 └──────────────────┘
│  ┌─────────────────────────────────────────────┐
│  │  ECS FARGATE CLUSTER                        │
│  │  ┌──────────────┐  ┌──────────────────┐     │
│  │  │  API Service  │  │  Ingest Worker   │     │
│  │  │  FastAPI +    │  │  Batch extract   │     │
│  │  │  Uvicorn      │  │  Scheduled task  │     │
│  │  │  2–10 tasks   │  │                  │     │
│  │  └──────────────┘  └──────────────────┘     │
│  │  ┌──────────────────────────────────────┐   │
│  │  │  Embedding Sidecar (MiniLM-L6-v2)   │   │
│  │  │  Runs in each task — no GPU needed   │   │
│  │  └──────────────────────────────────────┘   │
│  └─────────────────────────────────────────────┘
│
│  ┌─────────────────────────────────────────────┐
│  │  DATA TIER                                  │
│  │  ┌──────────────────┐  ┌────────────────┐   │
│  │  │ RDS PostgreSQL 16│  │ ElastiCache    │   │
│  │  │ + pgvector       │  │ Redis          │   │
│  │  │ Multi-AZ         │  │ Query cache    │   │
│  │  └──────────────────┘  └────────────────┘   │
│  └─────────────────────────────────────────────┘
│
│  ┌─────────────────────────────────────────────┐
│  │  SUPPORTING SERVICES                        │
│  │  S3 │ Secrets Manager │ CloudWatch │ ECR    │
│  └─────────────────────────────────────────────┘
│
│  ┌─────────────────────────────────────────────┐
│  │  GitHub Actions → ECR → ECS Blue/Green      │
│  │  Terraform / CDK manages all infra as code  │
│  └─────────────────────────────────────────────┘
└─────────────────────────────────────────────────
```

---

## Phase 1: Containerize the Application

Split the monolith into two container images pushed to **ECR**.

| Image | What it runs | Why separate |
|-------|-------------|--------------|
| `api` | `uvicorn src.api:app` — serves `/query` and `/health` | Scales horizontally on request load |
| `ingest-worker` | `python -m src.ingestion` — batch extraction + embedding | CPU-heavy, runs on a schedule, shouldn't steal API capacity |

The **sentence-transformers model** (`all-MiniLM-L6-v2`, 80 MB) runs on CPU inside each container — no GPU needed for a 384-dim model this small. Bake the model weights into the Docker image at build time so cold starts don't download from HuggingFace.

```dockerfile
# Multi-stage: download model at build, copy into slim runtime
FROM python:3.13-slim AS builder
RUN pip install sentence-transformers
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

FROM python:3.13-slim
COPY --from=builder /root/.cache/torch /root/.cache/torch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
```

**Why this matters:** Shows you think about cold-start latency, image size, and separating compute profiles.

---

## Phase 2: Managed Database — RDS PostgreSQL + pgvector

Replace the local Docker Postgres with **Amazon RDS for PostgreSQL 16**.

| Setting | Value | Rationale |
|---------|-------|-----------|
| Instance | `db.r6g.large` (2 vCPU, 16 GB RAM) | HNSW index lives in memory; r6g gives best memory/$ |
| Storage | gp3, 100 GB, autoscaling to 500 GB | gp3 gives consistent IOPS without provisioned cost |
| Multi-AZ | Yes | Automatic failover, ~30s downtime on primary failure |
| pgvector | `CREATE EXTENSION vector;` in parameter group | RDS supports pgvector natively since PG15 |
| Encryption | At-rest (KMS) + in-transit (SSL enforce) | Compliance baseline |
| Backups | 7-day automated + manual pre-migration | Point-in-time recovery |

Run `schema.sql` as a migration using a tool like Alembic or Flyway, not manually. This is the `candidates` + `chunks` schema with the HNSW and GIN indexes.

**Why not Aurora Serverless?** For a system with sustained query load and HNSW indexes that need warm memory, provisioned RDS is more predictable. Aurora Serverless v2 can scale to zero but cold-starting with a vector index in memory is slow.

---

## Phase 3: Compute — ECS Fargate

Deploy both containers on **ECS Fargate** (serverless containers).

### API Service

- **Task definition:** 1 vCPU, 2 GB memory (sentence-transformers + FastAPI fit comfortably)
- **Service:** Desired count 2, min 2, max 10
- **Auto-scaling:** Target tracking on `ECSServiceAverageCPUUtilization` at 60% + ALB `RequestCountPerTarget`
- **Health check:** ALB hits `GET /health` every 15s; ECS uses the same for task health

### Ingest Worker

- **Scheduled task** via EventBridge (e.g., daily at 2 AM UTC)
- OR triggered by S3 event when new resumes land in the S3 bucket
- Runs to completion, then stops — no always-on cost
- Use **ECS task role** with permissions to read S3, write RDS, call Anthropic

**Why Fargate over EKS?** For two services with no complex networking, Fargate avoids the operational overhead of managing a Kubernetes control plane. Reach for Kubernetes when there are 10+ services, not 2.

**Why Fargate over Lambda?** The embedding model takes ~500 MB in memory and several seconds to load. Lambda's cold start would be brutal. Fargate tasks stay warm.

---

## Phase 4: Networking and Ingress

```
Route 53 (DNS)
  → ALB (public subnet, HTTPS only via ACM cert)
    → Target Group (private subnet, ECS API tasks on port 8000)
```

| Component | Configuration |
|-----------|--------------|
| **VPC** | 2 AZs, public + private subnets in each |
| **ALB** | Public subnet, TLS termination with ACM certificate |
| **API tasks** | Private subnet, no public IP |
| **RDS** | Private subnet, security group allows only ECS task SG on port 5432 |
| **NAT Gateway** | One per AZ — API tasks need outbound to reach Anthropic API |

Security groups form a chain: ALB → API (port 8000) → RDS (port 5432). Nothing else is open.

**Why this matters:** Defense in depth, least-privilege networking. This separates a junior "deploy to EC2 with a public IP" answer from a Principal answer.

---

## Phase 5: Secrets and Configuration

Use **AWS Secrets Manager** for sensitive values, **SSM Parameter Store** for non-sensitive config.

| Secret | Store | How injected |
|--------|-------|-------------|
| `ANTHROPIC_API_KEY` | Secrets Manager | ECS task definition `secrets` block |
| `DATABASE_URL` | Secrets Manager (auto-rotated with RDS integration) | ECS task definition `secrets` block |
| `LANGSMITH_API_KEY` | Secrets Manager | ECS task definition `secrets` block |
| `EMBED_MODEL`, `GEN_MODEL` | SSM Parameter Store | ECS task definition `environment` block |

ECS natively injects Secrets Manager values as environment variables at task startup — no application code changes needed since `config.py` already reads from `os.environ`.

Never bake secrets into Docker images or pass them as build args.

---

## Phase 6: Caching Layer — ElastiCache Redis

Add a **Redis** cache in front of the retrieval pipeline.

### Cache strategy

- **Key:** hash of `(query_text, mode)` — the retriever is deterministic for a given query
- **TTL:** 1 hour (resumes change infrequently)
- **Invalidation:** Bust cache after ingestion worker completes (publish to a Redis channel or simply `FLUSHDB` on the cache namespace)

### Why cache?

Each query hits Claude Haiku for routing + pgvector for retrieval + Claude Sonnet for generation. That's ~$0.01–0.03 per query in LLM costs. Repeated queries (recruiters searching the same skills) should hit cache.

### Where in the code

Wrap `retrieval.retrieve()` + `generation.generate()` — return the full `{candidates, answer}` from cache if present.

---

## Phase 7: Observability

| Layer | Tool | What it watches |
|-------|------|----------------|
| **Logs** | CloudWatch Logs | Structured JSON logs from FastAPI (request ID, latency, route mode, candidate count) |
| **Metrics** | CloudWatch Metrics | p50/p95/p99 latency, error rate, cache hit ratio, Anthropic API latency |
| **Traces** | LangSmith | End-to-end LLM traces (already configured via `LANGSMITH_TRACING=true`) |
| **Alarms** | CloudWatch Alarms → SNS → PagerDuty/Slack | 5xx rate > 5%, p99 > 10s, Anthropic 429s |
| **Dashboards** | CloudWatch Dashboard | Single pane: request volume, latency distribution, retrieval mode breakdown, cost estimate |

**Key metric to track:** Retrieval mode distribution — if the Haiku router is sending 90% of queries to `filter` mode, the hybrid pipeline is underutilized and you should investigate prompt quality.

---

## Phase 8: CI/CD Pipeline

```
GitHub Push → GitHub Actions
  → Run tests + evals (run_evals.py against staging DB)
  → Build Docker image → Push to ECR
  → ECS Blue/Green Deploy via CodeDeploy
    → Route 10% traffic to new tasks
    → Monitor error rate for 5 minutes
    → Promote to 100% or auto-rollback
```

**Why blue/green over rolling?** The embedding model is baked into the image. If you change embedding models (e.g., upgrade to a larger model), you need ALL tasks on the same version simultaneously, or candidates will have mixed embeddings. Blue/green gives you atomic cutover.

**Run evals in CI:** `evals/run_evals.py` already computes recall@10, precision@10, and hit@10. Set a gate: if hybrid-rrf recall@10 drops below the baseline, block the deploy.

---

## Phase 9: Infrastructure as Code

Use **Terraform** or **AWS CDK** to define everything above:

```
infra/
  vpc.tf            # VPC, subnets, NAT, security groups
  rds.tf            # PostgreSQL instance, parameter group, pgvector
  ecs.tf            # Cluster, task definitions, services, auto-scaling
  alb.tf            # Load balancer, target groups, listeners
  ecr.tf            # Container registries
  secrets.tf        # Secrets Manager entries
  elasticache.tf    # Redis cluster
  monitoring.tf     # CloudWatch dashboards, alarms, SNS topics
  codedeploy.tf     # Blue/green deployment configuration
```

**Why IaC?** You must be able to reproduce the entire environment from scratch. "I clicked through the console" is not an acceptable answer at any level.

---

## Phase 10: Cost Estimation

| Service | Monthly estimate | Notes |
|---------|-----------------|-------|
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

The Anthropic API is the variable cost driver. The Redis cache directly reduces this.

---

## What a Principal-Level Answer Emphasizes

1. **Separation of concerns** — API vs. worker, not a monolith on a single EC2
2. **Security by default** — private subnets, Secrets Manager, TLS everywhere, SG chaining
3. **Operational maturity** — IaC, blue/green deploys, eval gates in CI, structured observability
4. **Cost awareness** — right-sizing instances, caching to reduce LLM spend, Fargate over EKS
5. **Tradeoffs articulated** — why Fargate not Lambda, why RDS not Aurora Serverless, why blue/green not rolling
