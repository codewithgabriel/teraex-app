const API_BASE = process.env.TERA_AGENT_API || "https://teraai-agent.onrender.com";


async function request(path, opts = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export default {
  getConfig: () => request("/api/config", {mode: 'cors'}),
  setConfig: (cfg) => request("/api/config", { method: "POST", body: JSON.stringify(cfg)  , mode: "cors"}),
  loadModel: () => request("/api/model/load", { method: "POST", mode: 'cors' }),
  downloadModel: () => `${API_BASE}/api/model/download`,
  runBacktest: (payload) => request("/api/backtest", { method: "POST", body: JSON.stringify(payload) , mode: 'cors' }),
  getBacktestResults: () => request("/api/backtest/results"),
  connectBroker: (payload) => request("/api/live/connect", { method: "POST", body: JSON.stringify(payload) , mode: "cors" }),
  startLive: () => request("/api/live/start", { method: "POST" , mode: "cors" }),
  stopLive: () => request("/api/live/stop", { method: "POST", mode: 'cors' }),
  startTraining: (payload) => request("/api/train", { method: "POST", body: JSON.stringify(payload) , mode: "cors" }),
  jobStatus: (jobId) => request(`/api/job/${jobId}` ,  { mode: "cors"}),
  getState: () => request("/api/state" , {mode: "cors"}),
};
