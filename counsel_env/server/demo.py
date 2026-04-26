import time
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

try:
    from ..models import CounselAction
    from .counsel_env_environment import CounselEnvironment
except (ImportError, ModuleNotFoundError):  # pragma: no cover - supports Docker-style imports
    from models import CounselAction
    from server.counsel_env_environment import CounselEnvironment


SESSION_TTL_SECONDS = 60 * 60
DEFAULT_SEED = 20260425

BENCHMARK_ROWS = [
    {
        "agent": "random",
        "episodes": 30,
        "avg_reward": 0.000,
        "primary_reward": 0.000,
        "trigger_rate": 0.000,
        "surface_rate": 0.000,
        "takeaway": "Vague questions and premature evidence do not score.",
    },
    {
        "agent": "keyword_spam",
        "episodes": 30,
        "avg_reward": 0.073,
        "primary_reward": 0.000,
        "trigger_rate": 0.678,
        "surface_rate": 0.000,
        "takeaway": "Trigger words alone get only tiny shaping reward.",
    },
    {
        "agent": "present_all",
        "episodes": 30,
        "avg_reward": 0.000,
        "primary_reward": 0.000,
        "trigger_rate": 0.000,
        "surface_rate": 0.000,
        "takeaway": "Blindly dumping exhibits fails because timing matters.",
    },
    {
        "agent": "trained_sft_grpo_run2",
        "episodes": 30,
        "avg_reward": 0.387,
        "primary_reward": 0.461,
        "trigger_rate": 0.589,
        "surface_rate": 0.461,
        "takeaway": "SFT warm-start plus GRPO learns the trigger-then-evidence loop.",
    },
    {
        "agent": "scripted_oracle",
        "episodes": 30,
        "avg_reward": 0.902,
        "primary_reward": 0.950,
        "trigger_rate": 0.950,
        "surface_rate": 0.950,
        "takeaway": "The target behavior is trigger first, evidence second.",
    },
]

_SESSIONS: Dict[str, tuple[float, CounselEnvironment]] = {}


class DemoResetRequest(BaseModel):
    seed: Optional[int] = Field(default=DEFAULT_SEED)
    difficulty: str = Field(default="easy")
    curriculum_stage: str = Field(default="easy")


class DemoStepRequest(BaseModel):
    session_id: str
    tool: str
    text: Optional[str] = None
    exhibit_id: Optional[str] = None
    reason: Optional[str] = None


def register_demo_routes(app: Any) -> None:
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def landing_page() -> HTMLResponse:
        return HTMLResponse(_landing_html())

    @app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
    def demo_page() -> HTMLResponse:
        return HTMLResponse(_demo_html())

    @app.get("/demo/api/benchmarks")
    def demo_benchmarks() -> Dict[str, Any]:
        return {"benchmarks": BENCHMARK_ROWS}

    @app.post("/demo/api/reset")
    def demo_reset(request: DemoResetRequest) -> Dict[str, Any]:
        _prune_sessions()
        env = CounselEnvironment()
        observation = env.reset(
            seed=request.seed,
            difficulty=request.difficulty,
            curriculum_stage=request.curriculum_stage,
            episode_id=f"space_demo_{request.seed or 'random'}",
        )
        session_id = uuid4().hex
        _SESSIONS[session_id] = (time.time(), env)
        return _payload(session_id, env, observation)

    @app.post("/demo/api/step")
    def demo_step(request: DemoStepRequest) -> Dict[str, Any]:
        env = _get_session(request.session_id)
        action = CounselAction(
            tool=request.tool,
            text=request.text,
            exhibit_id=request.exhibit_id,
            reason=request.reason,
        )
        observation = env.step(action)
        _SESSIONS[request.session_id] = (time.time(), env)
        return _payload(request.session_id, env, observation)


def _get_session(session_id: str) -> CounselEnvironment:
    _prune_sessions()
    entry = _SESSIONS.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Demo session expired. Reset the case.")
    return entry[1]


def _prune_sessions() -> None:
    now = time.time()
    expired = [
        session_id
        for session_id, (last_seen, _env) in _SESSIONS.items()
        if now - last_seen > SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        _SESSIONS.pop(session_id, None)


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _payload(session_id: str, env: CounselEnvironment, observation: Any) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "observation": _dump_model(observation),
        "state": _dump_model(env.state),
        "oracle_hint": _oracle_hint(env),
        "benchmarks": BENCHMARK_ROWS,
    }


def _oracle_hint(env: CounselEnvironment) -> Dict[str, str]:
    if env.witness is None:
        return {"label": "Reset the case", "detail": "Start an episode to see the target mechanic."}

    for contradiction in env.witness.contradictions:
        if not contradiction.triggered:
            question = f"{contradiction.trigger_keywords[0]}?"
            return {
                "label": "Trigger a sealed claim",
                "detail": f'Ask: "{question}"',
            }
        if not contradiction.surfaced:
            return {
                "label": "Present the matching exhibit",
                "detail": f"Use exhibit: {contradiction.disprover_evidence_id}",
            }

    if not env.done:
        return {"label": "Rest the case", "detail": "All known contradictions are surfaced."}
    return {"label": "Episode complete", "detail": "Reset to generate a new case."}


def _landing_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Counsel-Env</title>
  <style>
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: #0f172a; color: #e5e7eb; }
    main { max-width: 920px; margin: 0 auto; padding: 72px 24px; }
    a { color: #fde68a; }
    .card { background: #111827; border: 1px solid #374151; border-radius: 18px; padding: 28px; box-shadow: 0 24px 80px rgba(0,0,0,.35); }
    .eyebrow { color: #a5b4fc; text-transform: uppercase; letter-spacing: .14em; font-size: 12px; font-weight: 700; }
    h1 { font-size: clamp(40px, 7vw, 72px); line-height: .95; margin: 12px 0; }
    p { font-size: 18px; color: #cbd5e1; line-height: 1.65; }
    .button { display: inline-block; margin-top: 18px; padding: 14px 18px; border-radius: 12px; background: #facc15; color: #111827; text-decoration: none; font-weight: 800; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-top: 28px; }
    .metric { background: #0b1120; border: 1px solid #1f2937; border-radius: 14px; padding: 16px; }
    .metric strong { display: block; font-size: 28px; color: white; }
  </style>
</head>
<body>
  <main>
    <section class="card">
      <div class="eyebrow">OpenEnv hackathon environment</div>
      <h1>Train LLMs to catch lies under pressure.</h1>
      <p>Counsel-Env is a cross-examination arena. The agent must make a deterministic witness commit to a claim, then present the one exhibit that proves the claim false.</p>
      <a class="button" href="/demo">Open the live demo</a>
      <div class="grid">
        <div class="metric"><strong>0.902</strong>scripted oracle reward</div>
        <div class="metric"><strong>0.000</strong>random baseline reward</div>
        <div class="metric"><strong>30</strong>held-out seeded cases</div>
      </div>
    </section>
  </main>
</body>
</html>"""


def _demo_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Counsel-Env Live Demo</title>
  <style>
    :root { color-scheme: dark; --bg: #080c18; --panel: #111827; --muted: #9ca3af; --text: #f9fafb; --line: #273244; --gold: #facc15; --green: #34d399; --red: #fb7185; --blue: #93c5fd; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: radial-gradient(circle at top left, #1e1b4b 0, var(--bg) 38%); color: var(--text); }
    header { padding: 28px clamp(18px, 4vw, 44px); border-bottom: 1px solid var(--line); background: rgba(8,12,24,.78); position: sticky; top: 0; backdrop-filter: blur(14px); z-index: 2; }
    h1 { margin: 0 0 8px; font-size: clamp(28px, 5vw, 54px); line-height: 1; }
    h2 { margin: 0 0 14px; font-size: 18px; }
    p { color: #d1d5db; line-height: 1.55; }
    button, select, input, textarea { font: inherit; }
    button { border: 0; border-radius: 10px; padding: 10px 13px; cursor: pointer; font-weight: 750; background: #334155; color: white; }
    button.primary { background: var(--gold); color: #111827; }
    button.good { background: #065f46; }
    button.warn { background: #7f1d1d; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    input, select, textarea { width: 100%; border-radius: 10px; border: 1px solid var(--line); background: #0b1120; color: var(--text); padding: 10px; }
    textarea { min-height: 82px; resize: vertical; }
    main { display: grid; grid-template-columns: minmax(280px, 390px) 1fr; gap: 18px; padding: 18px clamp(18px, 4vw, 44px) 44px; }
    section { background: rgba(17,24,39,.92); border: 1px solid var(--line); border-radius: 18px; padding: 18px; box-shadow: 0 18px 60px rgba(0,0,0,.28); }
    .stack { display: grid; gap: 14px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .case { white-space: pre-wrap; color: #e5e7eb; }
    .evidence button { width: 100%; margin: 6px 0; text-align: left; background: #1f2937; }
    .evidence small { display: block; color: var(--muted); margin-top: 3px; font-weight: 500; }
    .transcript { min-height: 220px; max-height: 390px; overflow: auto; white-space: pre-wrap; background: #030712; border: 1px solid var(--line); border-radius: 12px; padding: 12px; color: #dbeafe; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; }
    .metric { border: 1px solid var(--line); border-radius: 14px; padding: 12px; background: #0b1120; }
    .metric strong { display: block; font-size: 24px; }
    .hint { border-left: 4px solid var(--gold); background: #221a05; padding: 12px; border-radius: 10px; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }
    th { color: #bfdbfe; }
    .status { color: var(--muted); }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Counsel-Env Live Demo</h1>
    <p>Make the witness commit, then present the exhibit that exposes the contradiction. This is the target behavior for GRPO training.</p>
  </header>
  <main>
    <div class="stack">
      <section>
        <h2>1. Start a Case</h2>
        <div class="row">
          <label>Seed<input id="seed" type="number" value="20260425"></label>
          <label>Difficulty<select id="difficulty"><option>easy</option><option>medium</option><option>hard</option></select></label>
        </div>
        <div class="actions" style="margin-top: 12px;">
          <button class="primary" onclick="resetCase()">Reset case</button>
          <button onclick="loadBenchmark()">Refresh benchmarks</button>
        </div>
        <p id="status" class="status">Ready.</p>
      </section>

      <section>
        <h2>2. Ask the Witness</h2>
        <textarea id="question" placeholder="Example: Where were you that night?"></textarea>
        <div class="actions" style="margin-top: 10px;">
          <button class="primary" onclick="askQuestion()">Ask question</button>
          <button class="good" onclick="restCase()">Rest case</button>
        </div>
      </section>

      <section>
        <h2>3. Evidence</h2>
        <div id="evidence" class="evidence status">Reset a case to load exhibits.</div>
      </section>
    </div>

    <div class="stack">
      <section>
        <h2>Case Brief</h2>
        <div id="caseBrief" class="case status">No active case.</div>
      </section>

      <section>
        <h2>Oracle Hint for Demo</h2>
        <div id="hint" class="hint">Reset a case to see the target sequence.</div>
      </section>

      <section>
        <h2>Reward and State</h2>
        <div id="metrics" class="metrics"></div>
      </section>

      <section>
        <h2>Transcript</h2>
        <div id="transcript" class="transcript">No transcript yet.</div>
      </section>

      <section>
        <h2>Held-Out Baselines</h2>
        <div id="benchmarks"></div>
      </section>
    </div>
  </main>

  <script>
    let sessionId = null;

    async function postJSON(url, body) {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail);
      }
      return response.json();
    }

    async function resetCase() {
      setStatus('Resetting case...');
      const payload = await postJSON('/demo/api/reset', {
        seed: Number(document.getElementById('seed').value || 0),
        difficulty: document.getElementById('difficulty').value,
        curriculum_stage: document.getElementById('difficulty').value
      });
      sessionId = payload.session_id;
      render(payload);
      setStatus('Case ready.');
    }

    async function askQuestion() {
      if (!sessionId) return setStatus('Reset a case first.');
      const text = document.getElementById('question').value.trim();
      if (!text) return setStatus('Type a question first.');
      const payload = await postJSON('/demo/api/step', { session_id: sessionId, tool: 'ask_question', text });
      document.getElementById('question').value = '';
      render(payload);
      setStatus('Question submitted.');
    }

    async function presentEvidence(exhibitId) {
      if (!sessionId) return setStatus('Reset a case first.');
      const payload = await postJSON('/demo/api/step', { session_id: sessionId, tool: 'present_evidence', exhibit_id: exhibitId });
      render(payload);
      setStatus('Evidence presented.');
    }

    async function restCase() {
      if (!sessionId) return setStatus('Reset a case first.');
      const payload = await postJSON('/demo/api/step', { session_id: sessionId, tool: 'rest_case' });
      render(payload);
      setStatus('Case rested.');
    }

    async function loadBenchmark() {
      const response = await fetch('/demo/api/benchmarks');
      const payload = await response.json();
      renderBenchmarks(payload.benchmarks);
      setStatus('Benchmarks refreshed.');
    }

    function render(payload) {
      const obs = payload.observation;
      const state = payload.state;
      document.getElementById('caseBrief').textContent = obs.case_brief || 'No case brief.';
      document.getElementById('hint').innerHTML = `<strong>${escapeHTML(payload.oracle_hint.label)}</strong><br>${escapeHTML(payload.oracle_hint.detail)}`;
      document.getElementById('transcript').textContent = obs.transcript_tail || obs.witness_response || 'No transcript yet.';
      renderEvidence(obs.evidence_descriptions || {});
      renderMetrics(obs, state);
      renderBenchmarks(payload.benchmarks || []);
    }

    function renderEvidence(evidence) {
      const container = document.getElementById('evidence');
      const entries = Object.entries(evidence);
      if (!entries.length) {
        container.textContent = 'No exhibits loaded.';
        return;
      }
      container.innerHTML = entries.map(([id, description]) => (
        `<button onclick="presentEvidence('${escapeAttr(id)}')">${escapeHTML(id)}<small>${escapeHTML(description)}</small></button>`
      )).join('');
    }

    function renderMetrics(obs, state) {
      const components = obs.reward_components || {};
      const items = [
        ['Reward', obs.reward ?? 0],
        ['Primary', components.primary_reward ?? 0],
        ['Triggered', `${state.contradictions_triggered}/${state.contradictions_total}`],
        ['Surfaced', `${state.contradictions_surfaced}/${state.contradictions_total}`],
        ['Questions left', obs.questions_remaining ?? 0],
        ['Done', obs.done ? 'yes' : 'no']
      ];
      document.getElementById('metrics').innerHTML = items.map(([label, value]) => (
        `<div class="metric"><strong>${escapeHTML(formatValue(value))}</strong>${escapeHTML(label)}</div>`
      )).join('');
    }

    function renderBenchmarks(rows) {
      if (!rows.length) return;
      document.getElementById('benchmarks').innerHTML = `
        <table>
          <thead><tr><th>Agent</th><th>Reward</th><th>Primary</th><th>Trigger</th><th>Surface</th><th>Takeaway</th></tr></thead>
          <tbody>${rows.map(row => `
            <tr>
              <td>${escapeHTML(row.agent)}</td>
              <td>${formatNumber(row.avg_reward)}</td>
              <td>${formatNumber(row.primary_reward)}</td>
              <td>${formatNumber(row.trigger_rate)}</td>
              <td>${formatNumber(row.surface_rate)}</td>
              <td>${escapeHTML(row.takeaway)}</td>
            </tr>`).join('')}
          </tbody>
        </table>`;
    }

    function setStatus(message) { document.getElementById('status').textContent = message; }
    function formatNumber(value) { return Number(value).toFixed(3); }
    function formatValue(value) { return typeof value === 'number' ? value.toFixed(3) : String(value); }
    function escapeHTML(value) {
      return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char]));
    }
    function escapeAttr(value) { return escapeHTML(value).replace(/`/g, '&#096;'); }

    resetCase().catch(error => setStatus(error.message));
  </script>
</body>
</html>"""
