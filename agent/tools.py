"""
agent/tools.py

Diagnostic tools that the Agent can call during root cause analysis.
These are simulated for the MVP — in production you'd wire these to real
Prometheus, GitHub, or AWS APIs.

Each tool returns a structured dict with realistic data. The Agent
The agentic LLM (via OpenRouter) decides *which* tools to call and *when*.
"""

import random
import time
from datetime import datetime, timezone


def get_recent_commits(service: str, count: int = 3) -> dict:
    """
    Returns the last N commits for a given service.
    Simulates a call to GitHub API / git log.
    """
    templates = {
        "checkout": [
            ("a3f1b2c", "refactor: change DB index on orders table for performance"),
            ("d9e4a1f", "fix: increase connection pool size in checkout service"),
            ("b2c3d4e", "feat: add retry logic for failed DB connections"),
        ],
        "auth": [
            ("c1d2e3f", "chore: rotate JWT signing secret"),
            ("f4g5h6i", "fix: fix token expiry edge case"),
            ("j7k8l9m", "feat: add OAuth2 provider support"),
        ],
        "payment": [
            ("p1q2r3s", "feat: integrate new payment gateway"),
            ("t4u5v6w", "fix: handle gateway timeout gracefully"),
            ("x7y8z9a", "chore: update gateway SDK to v3.1"),
        ],
    }

    default_commits = [
        ("0000000", "chore: routine dependency updates"),
        ("1111111", "docs: update service README"),
        ("2222222", "test: add unit tests for edge cases"),
    ]

    commits = templates.get(service, default_commits)[:count]
    return {
        "tool": "get_recent_commits",
        "service": service,
        "commits": [
            {"hash": h, "message": m, "author": "dev@company.com", "ago": f"{random.randint(1,72)}h ago"}
            for h, m in commits
        ],
    }


def get_top_db_queries(service: str, window_minutes: int = 2) -> dict:
    """
    Returns the top 5 most expensive DB queries in the last N minutes.
    Simulates a call to pg_stat_statements or Datadog APM.
    """
    query_pool = {
        "checkout": [
            ("SELECT * FROM orders WHERE user_id = $1 ORDER BY created_at DESC", 1840),
            ("UPDATE orders SET status = $1 WHERE id = $2", 980),
            ("SELECT COUNT(*) FROM order_items WHERE order_id = $1", 760),
            ("INSERT INTO audit_log (order_id, action) VALUES ($1, $2)", 340),
            ("SELECT * FROM inventory WHERE item_id = ANY($1)", 210),
        ],
    }
    fallback = [
        ("SELECT * FROM logs WHERE service = $1 LIMIT 100", random.randint(100, 500)),
        ("UPDATE sessions SET last_seen = NOW() WHERE token = $1", random.randint(50, 200)),
    ]
    queries = query_pool.get(service, fallback)
    return {
        "tool": "get_top_db_queries",
        "service": service,
        "window_minutes": window_minutes,
        "queries": [
            {"query": q, "avg_ms": ms, "calls": random.randint(50, 500)}
            for q, ms in queries
        ],
    }


def get_resource_snapshot(node_id: str) -> dict:
    """
    Returns CPU / memory / connection pool stats for a node.
    Simulates a call to a Prometheus /metrics scrape or cAdvisor.
    """
    # For checkout nodes, simulate degraded state
    is_degraded = "checkout" in node_id
    return {
        "tool": "get_resource_snapshot",
        "node_id": node_id,
        "cpu_pct": random.uniform(75, 95) if is_degraded else random.uniform(10, 40),
        "memory_pct": random.uniform(70, 90) if is_degraded else random.uniform(20, 50),
        "db_connections_active": random.randint(45, 50) if is_degraded else random.randint(5, 20),
        "db_connections_max": 50,
        "request_latency_p99_ms": random.randint(800, 2000) if is_degraded else random.randint(20, 80),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Tool Registry ─────────────────────────────────────────────────────────────
# OpenAI-compatible function definitions that we pass to the LLM.
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_recent_commits",
            "description": "Get the most recent git commits for a given service. Use this to check if a recent code change may have caused the anomaly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name (e.g. 'checkout')"},
                    "count": {"type": "integer", "description": "Number of commits to fetch (default 3)", "default": 3},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_db_queries",
            "description": "Get the slowest database queries for a given service in the last N minutes. Use this to detect DB performance issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {"type": "string", "description": "Service name"},
                    "window_minutes": {"type": "integer", "description": "Time window in minutes", "default": 2},
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_resource_snapshot",
            "description": "Get CPU, memory, and DB connection pool stats for a specific node. Use this to check for resource exhaustion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "The node ID (e.g. 'checkout-node-1')"},
                },
                "required": ["node_id"],
            },
        },
    },
]

# Map function names to their implementations
TOOL_EXECUTORS = {
    "get_recent_commits": get_recent_commits,
    "get_top_db_queries": get_top_db_queries,
    "get_resource_snapshot": get_resource_snapshot,
}
