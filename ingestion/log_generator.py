"""
ingestion/log_generator.py

Generates a realistic stream of structured JSON logs across 5 microservices.
Two modes:
  - healthy: normal operations, low error rate
  - anomaly: injects a "retry storm" into the checkout service
"""

import random
import time
from datetime import datetime, timezone

# ── Services & their "normal" log templates ───────────────────────────────────
SERVICES = {
    "auth": [
        ("INFO",  "User {user_id} logged in successfully"),
        ("INFO",  "Token issued for user {user_id}"),
        ("INFO",  "Session validated for user {user_id}"),
        ("WARN",  "Invalid token attempt from ip {ip}"),
        ("ERROR", "Auth service failed to reach DB for user {user_id}"),
    ],
    "checkout": [
        ("INFO",  "Cart {cart_id} created for user {user_id}"),
        ("INFO",  "Cart {cart_id} checked out successfully"),
        ("INFO",  "Payment initiated for order {order_id}"),
        ("WARN",  "Checkout retry attempt {n} for order {order_id}"),
        ("ERROR", "Checkout failed for order {order_id}: DB timeout"),
    ],
    "payment": [
        ("INFO",  "Payment {payment_id} processed successfully"),
        ("INFO",  "Refund issued for order {order_id}"),
        ("WARN",  "Payment gateway slow for order {order_id}"),
        ("ERROR", "Payment {payment_id} declined"),
    ],
    "inventory": [
        ("INFO",  "Stock checked for item {item_id}"),
        ("INFO",  "Item {item_id} reserved for order {order_id}"),
        ("WARN",  "Low stock warning for item {item_id}"),
        ("ERROR", "Failed to reserve item {item_id}: out of stock"),
    ],
    "notification": [
        ("INFO",  "Email sent to user {user_id}"),
        ("INFO",  "Push notification delivered to user {user_id}"),
        ("WARN",  "Notification retry {n} for user {user_id}"),
        ("ERROR", "Notification failed for user {user_id}"),
    ],
}

# In anomaly mode, checkout spams these templates at high rate
ANOMALY_TEMPLATES = [
    ("WARN",  "Checkout retry attempt {n} for order {order_id}"),
    ("WARN",  "DB connection pool exhausted in checkout service"),
    ("ERROR", "Checkout failed for order {order_id}: DB timeout"),
    ("WARN",  "High latency detected in checkout → inventory call"),
    ("ERROR", "Checkout failed for order {order_id}: DB timeout"),
]


def _random_fields():
    """Random values to fill template placeholders."""
    return {
        "user_id": random.randint(1000, 9999),
        "cart_id": random.randint(100, 999),
        "order_id": f"ORD-{random.randint(10000, 99999)}",
        "payment_id": f"PAY-{random.randint(10000, 99999)}",
        "item_id": f"ITEM-{random.randint(100, 999)}",
        "ip": f"192.168.{random.randint(0,255)}.{random.randint(0,255)}",
        "n": random.randint(1, 5),
    }


def make_log(service: str, template_tuple: tuple) -> dict:
    """Build a single log dict from a service name and (level, template) pair."""
    level, template = template_tuple
    fields = _random_fields()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": service,
        "node_id": f"{service}-node-{random.randint(1, 3)}",
        "level": level,
        "template": template,           # used for embedding (stable across instances)
        "message": template.format(**{k: fields[k] for k in fields if f"{{{k}}}" in template}),
    }


def generate_healthy_log() -> dict:
    """Pick a random service and a weighted-healthy template."""
    service = random.choice(list(SERVICES.keys()))
    templates = SERVICES[service]
    # Weight towards INFO (index 0,1,2) — realistic healthy distribution
    weights = [40, 30, 20, 8, 2][:len(templates)]
    template = random.choices(templates, weights=weights)[0]
    return make_log(service, template)


def generate_anomaly_log() -> dict:
    """Emit a checkout anomaly log — simulates a retry storm."""
    template = random.choice(ANOMALY_TEMPLATES)
    return make_log("checkout", template)
