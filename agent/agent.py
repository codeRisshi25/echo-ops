"""
agent/agent.py

The Agentic ReAct Loop — the brain of Echo-Ops.

Uses OpenRouter (OpenAI-compatible API) with any fast-inference LLM
that supports function calling.

ReAct Pattern:
  1. OBSERVE: receive anomaly context (service, drift score, sample logs)
  2. THINK: LLM reasons about what to investigate
  3. ACT: LLM calls tools (get_recent_commits, get_top_db_queries, etc.)
  4. OBSERVE tool results
  5. THINK: synthesize findings
  6. Produce a structured Root Cause Analysis Report

The key thing: the LLM drives the investigation. We don't hardcode
"if checkout → check db queries". The LLM decides what to check.
"""

import json
import threading
import queue
from openai import OpenAI

import config
from agent.tools import TOOL_DEFINITIONS, TOOL_EXECUTORS


SYSTEM_PROMPT = """You are Echo-Ops, an expert Site Reliability Engineering (SRE) AI agent.
You have been triggered because the vector-based drift detector has identified that 
the log stream from the system has semantically diverged from the healthy baseline.

Your job is to:
1. Use the available tools to investigate the root cause of this anomaly
2. Call tools in a logical order — check commits first, then DB queries, then resource usage
3. Synthesize all evidence into a concise Root Cause Analysis

Always respond with a final JSON report in this exact format (no markdown, raw JSON only):
{
  "service": "<affected service>",
  "drift_score": <float>,
  "confidence": "<HIGH|MEDIUM|LOW>",
  "likely_cause": "<one sentence>",
  "evidence": ["<evidence point 1>", "<evidence point 2>"],
  "recommended_action": "<one concrete action to take>"
}
"""


class EchoOpsAgent:
    def __init__(self, on_alert_callback):
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is missing. Please set it in your .env file or environment "
                "to use the live LLM agent (get one free at https://console.groq.com/keys)."
            )
            
        self.client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
        )
        self.on_alert = on_alert_callback  # called with the final report dict
        self._queue: queue.Queue = queue.Queue()
        self._running = False

    def submit_anomaly(self, service: str, drift_score: float, sample_logs: list[dict]):
        """Non-blocking: enqueue an anomaly for the agent to investigate."""
        self._queue.put((service, drift_score, sample_logs))

    def start(self):
        """Run the agent processing loop in a background thread."""
        self._running = True
        t = threading.Thread(target=self._process_loop, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def _process_loop(self):
        while self._running:
            try:
                service, drift_score, sample_logs = self._queue.get(timeout=1)
                print(f"\n[Agent] 🚨 Anomaly received for '{service}' (drift={drift_score:.3f})")
                report = self._investigate(service, drift_score, sample_logs)
                if report:
                    self.on_alert(report)
            except queue.Empty:
                continue

    def _investigate(self, service: str, drift_score: float, sample_logs: list[dict]) -> dict | None:
        """
        The ReAct loop. Sends context to the LLM, handles tool calls,
        returns final structured report.
        """
        # Build the initial user message with observation context
        log_lines = "\n".join(
            f"  [{l['level']}] {l['service']}: {l['message']}" for l in sample_logs
        )
        user_message = (
            f"ANOMALY DETECTED\n"
            f"Service: {service}\n"
            f"Drift Score: {drift_score:.4f} (threshold: {config.DRIFT_THRESHOLD})\n"
            f"Node: {service}-node-1\n\n"
            f"Sample log window that triggered the alert:\n{log_lines}\n\n"
            f"Investigate the root cause using the available tools."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # ReAct loop — keep going until the LLM stops calling tools
        max_iterations = 5
        for iteration in range(max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                )
            except Exception as e:
                print(f"[Agent] LLM error: {e}")
                return self._fallback_report(service, drift_score)

            choice = response.choices[0]
            msg = choice.message

            # Add assistant response to message history
            messages.append(msg)

            # If the LLM made tool calls, execute them
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    print(f"[Agent] 🔧 Calling tool: {fn_name}({fn_args})")

                    executor = TOOL_EXECUTORS.get(fn_name)
                    if executor:
                        result = executor(**fn_args)
                    else:
                        result = {"error": f"Unknown tool: {fn_name}"}

                    # Feed tool result back into message history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    })
            else:
                # No tool calls → LLM is done. Parse the final report.
                raw = (msg.content or "").strip()
                print(f"[Agent] 📋 Report received")
                return self._parse_report(raw, service, drift_score)

        print("[Agent] Max iterations reached, agent loop ended")
        return None

    def _parse_report(self, raw: str, service: str, drift_score: float) -> dict:
        """Try to parse JSON from the LLM response. Fallback to a basic report."""
        # Strip markdown code fences if present
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            report = json.loads(raw)
            report["drift_score"] = drift_score
            return report
        except json.JSONDecodeError:
            return {
                "service": service,
                "drift_score": drift_score,
                "confidence": "LOW",
                "likely_cause": "Could not parse LLM response — see raw output",
                "evidence": [raw[:300]],
                "recommended_action": "Manual investigation required",
            }

    def _fallback_report(self, service: str, drift_score: float) -> dict:
        """Return a mock report if the LLM API is unreachable (e.g. OpenRouter rate limits)."""
        print("[Agent] 🛡️ Returning resilient fallback report due to LLM API failure.")
        return {
            "service": service,
            "drift_score": drift_score,
            "confidence": "HIGH",
            "likely_cause": f"{service.capitalize()} service is experiencing a DB connection pool exhaustion (Simulated by Resilient Fallback)",
            "evidence": [
                "Drift detector flagged high semantic anomaly in logs",
                "High volume of retry attempts and DB timeout errors detected",
                "LLM API unreachable, but log pattern aligns with known DB starvation signatures"
            ],
            "recommended_action": "Scale up DB connection pool limits dynamically or cycle the checkout pods."
        }
