// k6 scenario for the Semvec REST API under load in the order of magnitude of the request.
//
// Each VU = one persistent chat client: creates a session, runs a long
// multi-turn conversation (/v1/run) and deletes it at the end (DELETE). Models
// real chat load, where each user holds ONE long-running conversation. The
// point of the demo: turn 1 and turn 100 carry the same (constant) input —
// exactly what Semvec is meant to prove.
//
// Ramp up to the request's peak concurrency (default peak 400 VUs). Each
// /v1/run message carries a realistic ~200-token payload so the embedder
// (mpnet/768 in the sidecar) does real work.
//
// Tuning via -e KEY=VAL:
//   BASE_URLS    comma-separated worker hosts (each VU pinned via __VU)
//   LICENSE_KEY  bearer token (required)
//   PEAK_VUS     peak concurrency (default 400 = request peak)
//   AVG_VUS      plateau concurrency (default 200 = request avg)
//   TURNS_MIN/MAX  turns/session (default 30/120 — long conversations)
//   THINK_MS     think time/VU between turns (default 250ms — more realistic)
//   MSG_TOKENS   approximate tokens/user message (default 200)
//   DIM          embedding dimension (default 768, mpnet)

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Trend } from 'k6/metrics';
import { randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.4.0/index.js';

const BASE_URLS_RAW = __ENV.BASE_URLS || __ENV.BASE_URL || 'http://127.0.0.1:18739';
const BASE_URL_LIST = BASE_URLS_RAW.split(',').map((s) => s.trim()).filter(Boolean);
const LICENSE_KEY = __ENV.LICENSE_KEY;
if (!LICENSE_KEY) {
  throw new Error('LICENSE_KEY env var is required');
}
const PEAK_VUS = parseInt(__ENV.PEAK_VUS || '400', 10);
const AVG_VUS = parseInt(__ENV.AVG_VUS || '200', 10);
const TURNS_MIN = parseInt(__ENV.TURNS_MIN || '30', 10);
const TURNS_MAX = parseInt(__ENV.TURNS_MAX || '120', 10);
const THINK_MS = parseInt(__ENV.THINK_MS || '250', 10);
const MSG_TOKENS = parseInt(__ENV.MSG_TOKENS || '200', 10);
const DIM = parseInt(__ENV.DIM || '768', 10);
const RAMP = __ENV.RAMP || '30s';
const PLATEAU = __ENV.PLATEAU || '60s';
const PEAK_HOLD = __ENV.PEAK_HOLD || '60s';

// Realistic ~MSG_TOKENS-word payload (one word ≈ 1 token, roughly).
const WORDPOOL = ('request ticket schedule invoice quantity site status value '
  + 'summary process history follow-up agreement approval policy delivery-date budget '
  + 'question note context department colleague shift handover memo revision').split(' ');
function payload(vu, turn) {
  const words = [`vu${vu}`, `t${turn}`];
  for (let i = 0; i < MSG_TOKENS; i++) words.push(WORDPOOL[(i * 7 + turn) % WORDPOOL.length]);
  return words.join(' ');
}

export const options = {
  scenarios: {
    chat: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: RAMP, target: AVG_VUS },       // ramp up to avg
        { duration: PLATEAU, target: AVG_VUS },     // hold avg plateau
        { duration: RAMP, target: PEAK_VUS },       // ramp up to peak
        { duration: PEAK_HOLD, target: PEAK_VUS },  // hold peak
        { duration: '15s', target: 0 },             // ramp down
      ],
      gracefulStop: '20s',
    },
  },
  // A fresh connection per request models a stateless API behind an LLM gateway
  // and avoids the keep-alive idle-close artifact: with realistic think times
  // (>uvicorn's 5s keep-alive) a reused connection can be server-closed mid-idle,
  // making the next request hang to the client timeout. New connections sidestep
  // that and give a clean, conservative latency reading.
  noConnectionReuse: true,
  thresholds: {
    'http_req_failed{action:run}': ['rate<0.01'],
    // SLA-oriented: /v1/run is the Semvec overhead layer BEFORE the LLM —
    // it must not eat into the TTFT budget. Alarm (not fail) at p90>250ms.
    'action_run_ms': ['p(90)<250'],
  },
  summaryTimeUnit: 'ms',
  summaryTrendStats: ['min', 'avg', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

const headers = { Authorization: `Bearer ${LICENSE_KEY}`, 'Content-Type': 'application/json' };
const createTrend = new Trend('action_create_ms', true);
const runTrend = new Trend('action_run_ms', true);
const deleteTrend = new Trend('action_delete_ms', true);
const turnsCounter = new Counter('completed_turns');
const sessionsCounter = new Counter('completed_sessions');

export default function () {
  const BASE_URL = BASE_URL_LIST[(__VU - 1) % BASE_URL_LIST.length];

  const createRes = http.post(`${BASE_URL}/v1/session/create`,
    JSON.stringify({ dimension: DIM }), { headers, tags: { action: 'create' } });
  if (!check(createRes, {
    'create 200': (r) => r.status === 200,
    'create has session_id': (r) => r.json('session_id') !== '',
  })) { return; }
  createTrend.add(createRes.timings.duration);
  const sid = createRes.json('session_id');

  const turns = randomIntBetween(TURNS_MIN, TURNS_MAX);
  for (let i = 0; i < turns; i++) {
    const runRes = http.post(`${BASE_URL}/v1/run`,
      JSON.stringify({ session_id: sid, message: payload(__VU, i) }),
      { headers, tags: { action: 'run' } });
    runTrend.add(runRes.timings.duration);
    check(runRes, { 'run 200': (r) => r.status === 200 });
    turnsCounter.add(1);
    if (THINK_MS > 0) sleep(THINK_MS / 1000);
  }

  const delRes = http.del(`${BASE_URL}/v1/session/${sid}`, null, { headers, tags: { action: 'delete' } });
  deleteTrend.add(delRes.timings.duration);
  check(delRes, { 'delete 200': (r) => r.status === 200 });
  sessionsCounter.add(1);
}
