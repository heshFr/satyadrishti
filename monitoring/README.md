# Satya Drishti — Monitoring Stack

Files in this directory configure the production observability pipeline. Drop them into your Prometheus + Grafana + Alertmanager instance.

| File                    | Purpose                                                                |
| ----------------------- | ---------------------------------------------------------------------- |
| `prometheus.yml`        | Scrape config — points at the Render gateway + HF Spaces worker        |
| `alerts.yml`            | Alert rules (page/warning/info severities) for availability + accuracy |
| `grafana_dashboard.json`| Importable dashboard: latency, errors, verdicts, resources             |

## Quick start (self-hosted Prometheus + Grafana)

```bash
# 1. Edit prometheus.yml — replace satyadrishti-api.onrender.com with your Render URL
#    and user-satyadrishti.hf.space with your Spaces URL.
# 2. Mount both yml files into your Prometheus container:
docker run -d --name prometheus -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  -v $(pwd)/alerts.yml:/etc/prometheus/alerts.yml \
  prom/prometheus

# 3. Run Grafana:
docker run -d --name grafana -p 3001:3000 grafana/grafana

# 4. In Grafana → Connections → Add data source → Prometheus
#    URL: http://prometheus:9090 (or http://host.docker.internal:9090)

# 5. In Grafana → Dashboards → Import → upload grafana_dashboard.json
#    Pick your Prometheus datasource when prompted.
```

## Grafana Cloud (free tier)

1. Sign up at https://grafana.com/products/cloud/ (free 10k metrics).
2. Create a Prometheus stack — Grafana Cloud gives you a remote-write URL and credentials.
3. Either:
   - Run Grafana Agent locally pointing at your gateway and remote-writing to Grafana Cloud, **or**
   - Use the Grafana Cloud "Synthetic Monitoring" probes to scrape `/metrics` directly.
4. Import `grafana_dashboard.json` into Grafana Cloud Dashboards.

## Alert Routing

`alerts.yml` is consumed by Prometheus's evaluation engine; routing to Slack/email/PagerDuty happens via **Alertmanager** (separate component).

Example minimal `alertmanager.yml`:

```yaml
route:
  receiver: slack-prod
  group_by: [alertname, severity]
  routes:
    - match: { severity: page }
      receiver: pagerduty-oncall
    - match: { severity: warning }
      receiver: slack-prod

receivers:
  - name: slack-prod
    slack_configs:
      - api_url: https://hooks.slack.com/services/XXX/YYY/ZZZ
        channel: '#satyadrishti-alerts'
        title: '{{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'

  - name: pagerduty-oncall
    pagerduty_configs:
      - routing_key: <pagerduty-integration-key>
```

## Severity Convention

| Severity | Means                                | Example                                       |
| -------- | ------------------------------------ | --------------------------------------------- |
| `page`   | Wake oncall immediately              | Gateway down, 5xx > 2%, P99 latency > 60s     |
| `warning`| Look at it next business day         | P95 latency > 30s, FP rate > 10%              |
| `info`   | FYI, no action required              | Biological veto spike (could be real attack)  |

## Useful PromQL Snippets

```promql
# Top 5 slowest endpoints (p95 over 1h)
topk(5, histogram_quantile(0.95, sum(rate(satya_request_duration_seconds_bucket[1h])) by (le, endpoint)))

# AI-detection rate per modality
sum by (modality) (rate(satya_verdict_total{verdict=~"deepfake|spoof|ai-generated"}[1h]))
  /
clamp_min(sum by (modality) (rate(satya_verdict_total[1h])), 0.001)

# Average ensemble uncertainty (higher = model is unsure)
avg by (modality) (rate(satya_ensemble_uncertainty_sum[5m]) / clamp_min(rate(satya_ensemble_uncertainty_count[5m]), 0.001))

# Forensic check fail ratio
sum by (check_id) (rate(satya_forensic_check_status_total{status="fail"}[1h]))
  /
clamp_min(sum by (check_id) (rate(satya_forensic_check_status_total[1h])), 0.001)
```
