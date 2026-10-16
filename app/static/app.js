"use strict";

const $ = (id) => document.getElementById(id);
const EXAMPLES = [
  "Compare the three most popular Python web frameworks for a new REST API and recommend one.",
  "Write and test a Python function that merges overlapping date ranges, then document it.",
  "A loan of 250,000 at 6.5% over 20 years: compute the monthly payment and total interest, and explain the result in a short client email.",
];

const state = { runId: null, source: null, cards: new Map(), timer: null, run: null };

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const body = response.status === 204 ? null : await response.json().catch(() => null);
  if (!response.ok) throw new Error((body && body.error) || `Request failed (${response.status})`);
  return body;
}

function renderMarkdown(target, text) {
  if (window.marked && window.DOMPurify) {
    target.classList.remove("plain");
    target.innerHTML = DOMPurify.sanitize(marked.parse(text));
    target.querySelectorAll("a").forEach((a) => { a.target = "_blank"; a.rel = "noopener noreferrer"; });
  } else {
    // Markdown libraries failed to load (offline): show the raw text safely.
    target.classList.add("plain");
    target.textContent = text;
  }
}

// -- sidebar ------------------------------------------------------------------

async function loadHealth() {
  const node = $("health");
  try {
    const health = await api("/api/health");
    node.textContent = health.credentials_in_env ? `${health.model} (Ollama)` : `${health.model} · Ollama not running`;
    node.classList.toggle("warn", !health.credentials_in_env);
    node.title = health.credentials_in_env ? "Ollama connected" : "Make sure Ollama is running locally at http://localhost:11434";
  } catch {
    node.textContent = "server unreachable";
    node.classList.add("warn");
  }
}

async function loadRoster() {
  const agents = await api("/api/agents");
  $("roster").replaceChildren(...agents.map((agent) => {
    const item = el("li");
    item.style.setProperty("--agent", `var(--${agent.key})`);
    item.append(el("strong", null, agent.name), el("span", null, agent.summary));
    return item;
  }));
}

async function loadHistory() {
  const runs = await api("/api/runs");
  const list = $("history");
  if (!runs.length) {
    list.replaceChildren(el("li", "empty", "No tasks yet."));
    return;
  }
  list.replaceChildren(...runs.map((run) => {
    const button = el("button", run.id === state.runId ? "active" : "");
    button.type = "button";
    button.title = run.task;
    button.append(el("span", `dot ${run.status}`), el("span", "title", run.task));
    button.addEventListener("click", () => openRun(run.id));
    const item = el("li");
    item.append(button);
    return item;
  }));
}

// -- run view -----------------------------------------------------------------

function setStatus(status) {
  const badge = $("run-status");
  badge.textContent = status;
  badge.className = `badge ${status}`;
  $("cancel").hidden = !(status === "running" || status === "queued");
}

function updateMeta() {
  const run = state.run;
  if (!run) return;
  const parts = [];
  const start = run.started_at || run.created_at;
  const end = run.finished_at || Date.now() / 1000;
  parts.push(`${Math.max(0, Math.round(end - start))}s`);
  const usage = run.usage || {};
  const tokens = (usage.input_tokens || 0) + (usage.output_tokens || 0);
  if (tokens) parts.push(`${tokens.toLocaleString()} tokens`);
  $("run-meta").textContent = parts.join(" · ");
}

function showResult(run) {
  const panel = $("result");
  const failed = run.status !== "completed";
  if (failed && !run.error) { panel.hidden = true; return; }
  panel.hidden = false;
  panel.classList.toggle("error", failed);
  $("result-title").textContent = failed ? (run.status === "cancelled" ? "Cancelled" : "Task failed") : "Final answer";
  if (failed) {
    $("result-body").classList.add("plain");
    $("result-body").textContent = run.error;
  } else {
    renderMarkdown($("result-body"), run.result || "");
  }
}

async function loadArtifacts() {
  const run = await api(`/api/runs/${state.runId}`);
  $("artifacts").hidden = !run.artifacts.length;
  $("artifact-list").replaceChildren(...run.artifacts.map((artifact) => {
    const button = el("button", null, artifact.name);
    button.type = "button";
    button.append(el("small", null, `${artifact.agent} · ${artifact.chars.toLocaleString()} chars`));
    button.addEventListener("click", async () => {
      const full = await api(`/api/runs/${state.runId}/artifacts/${encodeURIComponent(artifact.name)}`);
      $("artifact-name").textContent = full.name;
      $("artifact-content").textContent = full.content;
      $("artifact-dialog").showModal();
    });
    const item = el("li");
    item.append(button);
    return item;
  }));
}

function addCard(event) {
  const card = el("div", "card running");
  card.style.setProperty("--agent", `var(--${event.agent})`);
  const head = el("div", "card-head");
  const stateLabel = el("span", "state", "working");
  head.append(el("strong", null, event.name), stateLabel);
  const brief = el("details");
  brief.append(el("summary", null, event.parent ? "Brief from coordinator" : "Task"), el("p", null, event.task));
  const body = el("div", "body");
  card.append(head, brief, body);

  const parent = state.cards.get(event.parent);
  if (parent) {
    card.classList.add("child");
    // Keep a specialist's card directly under the coordinator turn that called it.
    let anchor = parent.card;
    while (anchor.nextElementSibling && anchor.nextElementSibling.classList.contains("child")) {
      anchor = anchor.nextElementSibling;
    }
    anchor.after(card);
  } else {
    $("timeline").append(card);
  }
  state.cards.set(event.inv, { card, body, stateLabel, text: null });
}

function describeCall(event) {
  const input = event.input || {};
  if (event.name === "delegate") return `→ ${input.agent || "specialist"}`;
  if (event.name === "web_search") return `web_search: ${input.query || ""}`;
  if (event.name === "web_fetch") return `web_fetch: ${input.url || ""}`;
  if (input.name) return `${event.name}: ${input.name}`;
  return event.name;
}

function handle(event) {
  const entry = state.cards.get(event.inv);
  switch (event.type) {
    case "run_started":
      setStatus("running");
      break;
    case "agent_started":
      addCard(event);
      break;
    case "text_delta":
      if (!entry) break;
      if (!entry.text) { entry.text = el("div", "text"); entry.body.append(entry.text); }
      entry.text.textContent += event.text;
      break;
    case "thinking":
      if (!entry) break;
      entry.body.append(el("div", "thought", event.text));
      entry.text = null;
      break;
    case "tool_call": {
      if (!entry) break;
      const chip = el("div", "chip", describeCall(event));
      chip.title = JSON.stringify(event.input, null, 2);
      entry.body.append(chip);
      entry.text = null;
      break;
    }
    case "tool_result":
      if (entry && event.is_error) entry.body.append(el("div", "chip error", `${event.name} failed: ${event.preview}`));
      break;
    case "artifact_saved":
      loadArtifacts().catch(() => {});
      break;
    case "agent_finished":
      if (!entry) break;
      entry.card.classList.remove("running");
      entry.card.classList.add(event.status);
      entry.stateLabel.textContent = event.status === "completed" ? "done" : event.status;
      if (event.error) entry.stateLabel.title = event.error;
      break;
    case "run_finished":
      closeStream();
      Object.assign(state.run, {
        status: event.status, result: event.result, error: event.error,
        usage: event.usage, finished_at: event.ts,
      });
      setStatus(event.status);
      showResult(state.run);
      updateMeta();
      loadArtifacts().catch(() => {});
      loadHistory().catch(() => {});
      break;
  }
}

function closeStream() {
  if (state.source) state.source.close();
  state.source = null;
  clearInterval(state.timer);
  state.timer = null;
}

async function openRun(runId) {
  closeStream();
  const run = await api(`/api/runs/${runId}`);
  state.runId = runId;
  state.run = run;
  state.cards = new Map();
  $("timeline").replaceChildren();
  $("result").hidden = true;
  $("artifacts").hidden = true;
  $("run").hidden = false;
  $("run-task").textContent = run.task;
  $("run-task").title = run.task;
  setStatus(run.status);
  updateMeta();
  loadHistory().catch(() => {});

  // The stream replays the whole event log, so finished runs render the same way.
  state.source = new EventSource(`/api/runs/${runId}/stream`);
  state.source.onmessage = (message) => {
    if (state.runId === runId) handle(JSON.parse(message.data));
  };
  state.timer = setInterval(updateMeta, 1000);
}

// -- composer -----------------------------------------------------------------

$("composer").addEventListener("submit", async (submitEvent) => {
  submitEvent.preventDefault();
  const error = $("form-error");
  const button = $("submit");
  error.hidden = true;
  button.disabled = true;
  try {
    const run = await api("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: $("task").value }),
    });
    $("task").value = "";
    await openRun(run.id);
  } catch (err) {
    error.textContent = err.message;
    error.hidden = false;
  } finally {
    button.disabled = false;
  }
});

$("task").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) $("composer").requestSubmit();
});

$("cancel").addEventListener("click", () => {
  api(`/api/runs/${state.runId}/cancel`, { method: "POST" }).catch(() => {});
});

$("examples").append(...EXAMPLES.map((text, index) => {
  const button = el("button", null, `Example ${index + 1}`);
  button.type = "button";
  button.title = text;
  button.addEventListener("click", () => { $("task").value = text; $("task").focus(); });
  return button;
}));

loadHealth();
loadRoster().catch(() => {});
loadHistory().catch(() => {});
