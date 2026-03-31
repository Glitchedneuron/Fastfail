"""
Corpus fetcher — Runbooks & Documentation standardisation domain.

Sources:
  • GitHub code / documentation (HF: codeparrot/github-code)
  • StackOverflow Q&A (HF: stackoverflow_questions)
  • Technical Wikipedia articles
  • Synthetic runbook templates (ITSM / SRE / DevOps)
"""

from __future__ import annotations

from ..base import BaseCorpus, CorpusResult
from loguru import logger


_TECH_WIKI_TOPICS = [
    "Standard operating procedure", "Runbook", "Incident management",
    "Site reliability engineering", "DevOps", "ITIL", "Change management (ITSM)",
    "Continuous integration", "Continuous delivery", "Kubernetes",
    "Docker (software)", "Ansible (software)", "Terraform (software)",
    "Monitoring (software)", "Observability (software)", "Log management",
    "Disaster recovery", "Business continuity planning", "Postmortem",
    "Root cause analysis", "Mean time to repair", "Service level agreement",
    "API (computing)", "Microservices", "Infrastructure as code",
]

_RUNBOOK_TEMPLATES = [
    {
        "title": "Database Failover Runbook",
        "sections": [
            ("Purpose", "This runbook describes the procedure to perform a planned or unplanned failover of the primary PostgreSQL database to the standby replica."),
            ("Prerequisites", "- Administrative access to the database hosts\n- Access to monitoring dashboard\n- On-call engineer notified\n- Change request approved (for planned failover)"),
            ("Impact Assessment", "Estimated downtime: 2-5 minutes. Affected services: All read/write operations. Notification required: Yes (via PagerDuty P1)."),
            ("Steps", "1. Confirm primary database is unreachable\n2. Verify replica lag is < 10 seconds\n3. Execute: `pg_ctl promote -D /var/lib/postgresql/data`\n4. Update DNS/HAProxy to point to new primary\n5. Verify application connectivity\n6. Update monitoring thresholds\n7. Schedule resync of old primary"),
            ("Rollback", "If promotion fails: restart original primary and re-check replication status."),
            ("Post-Incident", "File incident report within 24 hours. Update this runbook if procedure gaps were found."),
        ],
    },
    {
        "title": "Service Deployment Runbook",
        "sections": [
            ("Purpose", "Step-by-step procedure for deploying a new version of a microservice to the production Kubernetes cluster."),
            ("Prerequisites", "- Docker image built and pushed to registry\n- Helm chart updated with new image tag\n- Staging environment tests passed\n- Deployment approved in Jira"),
            ("Pre-Deployment Checks", "1. Check current pod health: `kubectl get pods -n production`\n2. Review resource quotas: `kubectl describe namespace production`\n3. Confirm no ongoing incidents"),
            ("Deployment Steps", "1. Update values.yaml with new image tag\n2. Run: `helm upgrade --install service-name ./chart -n production`\n3. Monitor rollout: `kubectl rollout status deployment/service-name`\n4. Validate health endpoint: `curl https://service/health`"),
            ("Verification", "- Check pod logs for errors\n- Verify metrics in Grafana\n- Confirm no spike in error rate"),
            ("Rollback", "Run: `helm rollback service-name [REVISION]` where REVISION is the previous stable version."),
        ],
    },
    {
        "title": "Security Incident Response Runbook",
        "sections": [
            ("Purpose", "Immediate response procedures when a security incident or breach is detected."),
            ("Severity Classification", "P0: Active breach with data exfiltration\nP1: Suspected compromise, no confirmed exfiltration\nP2: Anomalous activity requiring investigation"),
            ("Immediate Actions (0-15 min)", "1. Isolate affected systems from network\n2. Preserve evidence (snapshots, logs)\n3. Notify security team and management\n4. Open incident ticket"),
            ("Investigation (15-60 min)", "1. Review access logs for anomalous IPs\n2. Check for privilege escalation events\n3. Identify affected data assets\n4. Determine attack vector"),
            ("Containment", "1. Rotate compromised credentials\n2. Apply firewall rules to block attack IPs\n3. Patch exploited vulnerability\n4. Verify no persistence mechanisms"),
            ("Communication", "Internal: Every 30 minutes to management\nExternal: Legal counsel to advise on breach notification requirements"),
        ],
    },
]


class RunbooksCorpus(BaseCorpus):
    domain_name = "runbooks"
    default_token_budget = 12_000_000

    def _fetch(self, result: CorpusResult) -> None:
        self._add_runbook_templates(result)
        self._fetch_tech_wiki(result)
        self._fetch_stackoverflow(result)
        self._fetch_code_docs(result)

    def _add_runbook_templates(self, result: CorpusResult) -> None:
        for rb in _RUNBOOK_TEMPLATES:
            parts = [f"# {rb['title']}\n"]
            for heading, content in rb["sections"]:
                parts.append(f"\n## {heading}\n{content}\n")
            result.append("\n".join(parts), source="synthetic-runbook")

    def _fetch_tech_wiki(self, result: CorpusResult) -> None:
        try:
            import wikipediaapi
            wiki = wikipediaapi.Wikipedia("FastFail-Bot/1.0", "en")
            for topic in _TECH_WIKI_TOPICS:
                if not self._budget_remaining(result):
                    break
                page = wiki.page(topic)
                if page.exists():
                    result.append(page.text, source=f"wikipedia:{topic}")
        except Exception as e:
            logger.warning(f"  Wikipedia tech fetch failed: {e}")

    def _fetch_stackoverflow(self, result: CorpusResult) -> None:
        """StackOverflow Q&A for operational procedures."""
        self._fetch_hf_dataset(
            result,
            dataset_name="stackoverflow_questions",
            split="train",
            text_field="body",
            max_samples=10_000,
        )

    def _fetch_code_docs(self, result: CorpusResult) -> None:
        """GitHub code for documentation / YAML / Markdown patterns."""
        self._fetch_hf_dataset(
            result,
            dataset_name="codeparrot/github-code",
            config_name="all-all",
            split="train",
            text_field="code",
            max_samples=5_000,
        )
