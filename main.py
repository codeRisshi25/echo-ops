"""
main.py — Echo-Ops Orchestrator

Wires together all components and runs them concurrently:
  1. Baseline builder  → ingests healthy logs into Endee
  2. Log generator     → simulates a live log stream
  3. Ingestion loop    → batches logs, embeds, upserts to Endee
  4. Drift detector    → rolling window cosine-distance check every 5s
  5. Agent loop        → Agentic LLM ReAct reasoning on anomalies
  6. FastAPI dashboard → streams everything to browser via SSE

Run:
  python main.py           # normal mode
  python main.py --demo    # injects anomaly at t=25s automatically
"""

import sys
import time
import uuid
import threading
import argparse
import uvicorn
from rich.console import Console

import config
from ingestion.log_generator import generate_healthy_log, generate_anomaly_log
from ingestion.embedder import get_embedding, get_cache_stats
from ingestion.endee_client import EndeeClient
from agent.detector import DriftDetector
from agent.agent import EchoOpsAgent
from api.server import app, push_log_event, push_alert_event, update_status

console = Console()
endee = EndeeClient()


# ── Phase 1: Wait for Endee ───────────────────────────────────────────────────

def wait_for_endee(retries=15):
    console.print("[bold cyan]⏳ Waiting for Endee to be ready...[/]")
    for i in range(retries):
        if endee.is_healthy():
            console.print("[bold green]✓ Endee is up.[/]")
            return
        time.sleep(2)
    console.print("[bold red]✗ Endee not reachable. Is Docker running?[/]")
    sys.exit(1)


# ── Phase 2: Build Baseline ───────────────────────────────────────────────────

def build_baseline() -> int:
    """
    Ingest WINDOW MEANS of healthy logs into Endee instead of single logs.
    This calculates the semantic 'center of gravity' of healthy behaviour.
    """
    console.print(f"\n[bold cyan]📦 Building healthy baseline ({config.BASELINE_LOG_COUNT} windows)...[/]")
    endee.create_index(config.ENDEE_INDEX_BASELINE, config.ENDEE_INDEX_DIM, config.ENDEE_METRIC)

    batch = []
    # We create BASELINE_LOG_COUNT distinct "healthy windows"
    for i in range(config.BASELINE_LOG_COUNT):
        window_logs = [generate_healthy_log() for _ in range(config.DRIFT_WINDOW_SIZE)]
        embeddings = [get_embedding(log["template"]) for log in window_logs]
        
        # Calculate the mean vector for this healthy window
        dim = config.ENDEE_INDEX_DIM
        mean_vec = [
            sum(embeddings[i][j] for i in range(len(embeddings))) / len(embeddings)
            for j in range(dim)
        ]

        batch.append({
            "id": f"baseline-win-{uuid.uuid4().hex[:8]}",
            "vector": mean_vec,
            "metadata": {
                "type": "healthy_window_mean",
            },
        })
        
        if len(batch) >= config.BATCH_SIZE:
            endee.upsert(config.ENDEE_INDEX_BASELINE, batch)
            batch.clear()

    if batch:
        endee.upsert(config.ENDEE_INDEX_BASELINE, batch)

    stats = get_cache_stats()
    console.print(
        f"[green]✓ Baseline built. Cache stats: {stats['cached_templates']} unique templates, "
        f"{stats['hit_rate_pct']}% hit rate.[/]"
    )
    return config.BASELINE_LOG_COUNT


# ── Phase 3: Ingestion Loop ───────────────────────────────────────────────────

def ingestion_loop(detector: DriftDetector, demo_mode: bool, stop_event: threading.Event):
    """
    Generates a continuous log stream. Batches embeddings and upserts to Endee.
    In demo mode, switches to anomaly logs at t=25s.
    """
    batch = []
    start_time = time.time()
    anomaly_injected = False

    while not stop_event.is_set():
        elapsed = time.time() - start_time

        # Demo: inject anomaly after N seconds
        if demo_mode and elapsed > config.DEMO_ANOMALY_INJECT_AFTER_SEC and not anomaly_injected:
            anomaly_injected = True
            console.print("\n[bold red]💉 DEMO: Injecting anomaly (checkout retry storm)...[/]")

        log = generate_anomaly_log() if anomaly_injected else generate_healthy_log()

        # Feed raw log to drift detector  (We no longer insert single live logs into Endee)
        detector.add_log(log)

        # Push to dashboard
        push_log_event(log)

        time.sleep(0.3)  # ~3-4 logs/sec


# ── Phase 4: Status Broadcast Loop ───────────────────────────────────────────

def status_loop(detector: DriftDetector, baseline_size: int, stop_event: threading.Event):
    """Periodically updates the dashboard with current stats."""
    while not stop_event.is_set():
        status = detector.get_status()
        cache = get_cache_stats()
        update_status({
            **status,
            "baseline_size": baseline_size,
            "cache_hit_rate_pct": cache["hit_rate_pct"],
        })
        time.sleep(2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Echo-Ops Agent")
    parser.add_argument("--demo", action="store_true", help="Auto-inject anomaly at t=25s")
    args = parser.parse_args()

    console.print("\n[bold white]⚡ Echo-Ops: Autonomous Infrastructure Observability[/]\n")

    wait_for_endee()
    baseline_size = build_baseline()

    stop_event = threading.Event()

    # Alert callback: receives report from agent, pushes to dashboard
    def on_alert(report: dict):
        console.print(f"\n[bold red]🚨 ALERT: {report.get('likely_cause')}[/]")
        console.print(f"   [yellow]Evidence: {report.get('evidence')}[/]")
        console.print(f"   [cyan]Action: {report.get('recommended_action')}[/]")
        push_alert_event(report)
        # Append to log file
        with open("alerts.log", "a") as f:
            import json
            f.write(json.dumps(report) + "\n")

    # Anomaly callback: drift detector fires this, agent picks it up
    agent = EchoOpsAgent(on_alert_callback=on_alert)
    detector = DriftDetector(endee=endee, on_anomaly_callback=agent.submit_anomaly)

    # Start background threads
    agent.start()
    detector.start()

    threading.Thread(
        target=ingestion_loop, args=(detector, args.demo, stop_event), daemon=True
    ).start()
    threading.Thread(
        target=status_loop, args=(detector, baseline_size, stop_event), daemon=True
    ).start()

    console.print(f"\n[bold green]✓ All systems running.[/]")
    console.print(f"[cyan]  Dashboard → http://localhost:{config.API_PORT}[/]")
    console.print(f"[cyan]  Endee     → {config.ENDEE_URL}/api/v1/index/list[/]")
    if args.demo:
        console.print(f"[yellow]  Demo mode: anomaly injects in {config.DEMO_ANOMALY_INJECT_AFTER_SEC}s[/]")
    console.print("\n[dim]Press Ctrl+C to stop.[/]\n")

    # Run FastAPI (blocking)
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT, log_level="warning")


if __name__ == "__main__":
    main()
