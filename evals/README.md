# MLflow Evaluation

Evaluation pipeline for the Lightspeed Agent using [MLflow GenAI Evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/). Sends questions to a deployed agent via A2A, then scores the responses using LLM-as-a-judge and code-based scorers.

## How It Works

```
Developer laptop (on VPN)
    │
    ├──► Agent (A2A endpoint) ── sends evaluation questions
    │         │
    │         └──► Traces flow to MLflow automatically (already configured)
    │
    ├──► Judge model (LiteLLM proxy) ── scores agent responses
    │
    └──► MLflow server ── stores evaluation results
```

1. Load evaluation questions from `dataset.json`
2. Send each question to the agent's A2A endpoint with a Bearer token
3. Collect the agent's text response
4. Run MLflow scorers against each response
5. Log results to the MLflow `lightspeed-agent-evals` experiment

## Prerequisites

- Python virtual environment with eval dependencies
- VPN access to the OpenShift cluster (agent + MLflow endpoints)
- A valid Bearer token for the agent (from Red Hat SSO)

## Setup

```bash
# Install eval dependencies (from repo root, venv activated)
pip install -e ".[eval]"
```

## Configuration

### Judge Model (required)

LLM-as-a-judge scorers need a model to evaluate responses. The script refuses to run without `MLFLOW_GENAI_JUDGE_DEFAULT_MODEL` set, to prevent evaluation data from being sent to external cloud providers.

```bash
export MLFLOW_GENAI_JUDGE_DEFAULT_MODEL="openai:/Qwen/Qwen3.5-35B-A3B"
export OPENAI_BASE_URL="https://vllm-qwen35-35b-a3b-judge.apps.<cluster>/v1"
export OPENAI_API_KEY="dummy"  # set to "dummy" if no auth required
```

### Data Privacy

LLM-as-a-judge scorers send agent responses (which may contain CVEs, host names, advisor details) to the judge model for scoring. Use a self-hosted model to keep data within your network.

### SSL/TLS for Internal Clusters

OpenShift clusters and internal model endpoints typically use self-signed certificates that Python doesn't trust by default. To skip certificate verification (connections remain encrypted — only the CA check is skipped):

```bash
export MLFLOW_TRACKING_INSECURE_TLS=true
```

This covers all HTTPS calls: MLflow tracking, agent A2A requests, and judge model calls.

For proper certificate verification instead, export the cluster's CA certificate:

```bash
# Extract the OpenShift ingress CA
oc extract configmap/router-ca -n openshift-ingress-operator \
    --keys=ca-bundle.crt --to=/tmp/

# Point Python at it
export REQUESTS_CA_BUNDLE=/tmp/ca-bundle.crt
```

If the agent and judge endpoints are on different clusters, concatenate both CA certs into a single bundle file.

## Usage

```bash
python -u evals/run_eval.py \
    --agent-url https://lightspeed-agent-<namespace>.apps.<cluster>/ \
    --token "<bearer-token>" \
    --mlflow-uri https://mlflow-<namespace>.apps.<cluster>/
```

Use `python -u` for unbuffered output to see progress in real time.

### Options

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--agent-url` | `EVAL_AGENT_URL` | `http://localhost:8000` | Agent A2A endpoint |
| `--token` | `EVAL_AGENT_TOKEN` | (required) | Bearer token for authentication |
| `--mlflow-uri` | `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server |
| `--experiment` | | `lightspeed-agent-evals` | MLflow experiment name |
| `--timeout` | | `180` | Timeout per question (seconds) |
| `--dataset` | | `evals/dataset.json` | Path to dataset file |

## Scorers

| Scorer | Type | Description |
|--------|------|-------------|
| `Correctness` | LLM judge | Compares response against expected behavior |
| `RelevanceToQuery` | LLM judge | Checks if the response addresses the question |
| `Guidelines (safety)` | LLM judge | Enforces safety rules (no tool name leakage, no code generation, prompt injection resistance) |
| `ExpectationsGuidelines` | LLM judge | Evaluates per-row whether the response matches the `expected_behavior` field |
| `tool_correctness` | Code-based | Verifies the agent used the expected tool domains based on response content |
| `error_handling` | Code-based | Checks if errors were handled gracefully (good/poor/n/a) |

LLM judge scorers use the model from `MLFLOW_GENAI_JUDGE_DEFAULT_MODEL`. Code-based scorers run locally with no model.

## Dataset

`dataset.json` contains evaluation questions with expected behavior. Each entry:

```json
{
    "id": "V-001",
    "category": "vulnerability",
    "question": "Is CVE-2024-6387 affecting any of my systems?",
    "question_type": "binary",
    "options": null,
    "expected_answer": "yes",
    "expected_tools": ["vulnerability__get_cve_systems"],
    "scenario_type": "single_tool",
    "expected_behavior": "The agent should call get_cve_systems with CVE-2024-6387 to determine if any systems are affected and return the list of impacted hosts.",
    "difficulty": "easy",
    "tags": ["cve", "affected-systems", "lookup"]
}
```

### Adding Questions

Add entries to `dataset.json` following the same schema. Questions are defined in the [Datasets ADR](../docs/observability-adr/Datasets.md) across 8 categories: vulnerability, inventory, advisor, planning, remediations, image builder, cross-domain, and guardrails.

### Fields

| Field | Description |
|-------|-------------|
| `id` | Identifier matching the ADR question tables (e.g. V-001, I-001) |
| `category` | Domain category |
| `question` | The prompt sent to the agent |
| `question_type` | Type of expected answer: `binary`, `list`, `explanation`, `count` |
| `options` | Multiple-choice options (null if not applicable) |
| `expected_answer` | The expected answer or summary of correct response |
| `expected_tools` | MCP tools the agent should call |
| `scenario_type` | `single_tool`, `multi_tool`, `multi_step`, `no_tool` |
| `expected_behavior` | Description of correct agent behavior (used by Correctness scorer) |
| `difficulty` | `easy`, `medium`, `hard` |
| `tags` | Labels for filtering and grouping evaluations |

## Viewing Results

Results are logged to the MLflow `lightspeed-agent-evals` experiment. Open the MLflow UI and navigate to the experiment to see per-question scores, compare evaluation runs, and drill into individual results.
