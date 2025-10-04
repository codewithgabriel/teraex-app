# api.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import threading
import traceback
import time
import os
import json

# Import your existing project functions (assumes same import paths as in app.py)
from utils import run_backtest, create_env, load_and_prepare_data, load_model, plot_equity_curve, plot_trades, refresh_config
from models.train_ppo import train_ppo_model
from brokers.broker_alpaca import AlpacaBroker
from brokers.broker_ccxt import CCXTBroker
from datetime import datetime

app = Flask(__name__)
CORS(app)

CONFIG = refresh_config()

# In-memory state (persist or replace with DB in production)
STATE = {
    "training_progress": 0,
    "training_status": "Not started",
    "backtest_results": None,  # (networth (np array), trades (list of dicts))
    "trade_decisions": [],
    "current_balance": CONFIG.get("initial_balance", 100.0),
    "balance_history": [],
    "live_trading_active": False,
    "broker": None,
    "model": None,
    "jobs": {}
}

def safe_json(obj):
    try:
        return json.loads(json.dumps(obj, default=str))
    except:
        return str(obj)

# -- Config endpoints ------------------------------------------------------
@app.route("/api/config", methods=["GET"])
def get_config():
    global CONFIG
    return jsonify(CONFIG)

@app.route("/api/config", methods=["POST"])
def set_config():
    global CONFIG
    data = request.json or {}
    CONFIG.update(data)
    # persist
    with open("config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
    return jsonify({"status": "ok", "config": CONFIG})

# -- Model endpoints -------------------------------------------------------
@app.route("/api/model/load", methods=["POST"])
def api_load_model():
    try:
        model = load_model()
        STATE["model"] = model
        return jsonify({"status": "ok", "message": "Model loaded"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/api/model/download", methods=["GET"])
def api_download_model():
    path = CONFIG.get("model_save_path")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"status": "error", "message": "Model file not found"}), 404

# -- Backtest endpoints ----------------------------------------------------
@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """
    Request body:
      { "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "timeframe": "1m|5m|1h|1d" }
    """
    payload = request.json or {}
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    timeframe = payload.get("timeframe", CONFIG.get("timeframe"))

    try:
        model = STATE.get("model") or load_model()
        train_df, test_df = load_and_prepare_data(start_date=start_date, end_date=end_date, split=False, interval=timeframe)
        _, eval_env = create_env(train_df, test_df)

        # Run backtest synchronously (blocking). In production run in background.
        networth, trades = run_backtest(model, test_df, env=eval_env)

        # store
        STATE["backtest_results"] = {
            "networth": list(networth),
            "trades": trades
        }
        STATE["trade_decisions"] = trades
        STATE["balance_history"] = list(networth)
        STATE["current_balance"] = float(networth[-1]) if len(networth) > 0 else CONFIG.get("initial_balance", 0)

        return jsonify({"status": "ok", "networth_len": len(networth), "trades": len(trades)})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/api/backtest/results", methods=["GET"])
def api_backtest_results():
    if STATE["backtest_results"] is None:
        return jsonify({"status": "empty"})
    return jsonify(safe_json(STATE["backtest_results"]))

# -- Live trading / broker endpoints --------------------------------------
@app.route("/api/live/connect", methods=["POST"])
def api_live_connect():
    data = request.json or {}
    broker_type = data.get("broker_type")
    api_key = data.get("api_key")
    api_secret = data.get("api_secret")
    base_url = data.get("base_url", None)

    try:
        if broker_type == "Alpaca":
            broker = AlpacaBroker(api_key, api_secret, base_url)
            account = broker.get_account()
        else:
            broker = CCXTBroker("binance", api_key, api_secret)
            account = broker.get_balance()

        STATE["broker"] = broker
        return jsonify({"status": "ok", "account": safe_json(account)})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/api/live/start", methods=["POST"])
def api_live_start():
    """
    Starts a simple synchronous simulated decision once OR if you want looped live trading,
    you should implement a background worker (Celery/worker) for real usage.
    """
    if STATE.get("broker") is None:
        return jsonify({"status": "error", "message": "Broker not connected"}), 400
    if STATE.get("model") is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 400

    try:
        symbol = CONFIG["asset_symbol"].replace("/", "") if "/" in CONFIG["asset_symbol"] else CONFIG["asset_symbol"]
        current_price = STATE["broker"].get_current_price(symbol)
        # For demo simulate a decision (your model usage should format data)
        import numpy as np
        decision = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": decision,
            "price": float(current_price),
            "quantity": 1
        }
        STATE["trade_decisions"].append(entry)

        # Example: execute with broker if BUY/SELL
        if decision == "BUY":
            try:
                STATE["broker"].place_order(symbol=symbol, quantity=1, side='buy', order_type='market')
            except Exception as be:
                # ignore or log
                pass
        elif decision == "SELL":
            try:
                STATE["broker"].place_order(symbol=symbol, quantity=1, side='sell', order_type='market')
            except Exception as be:
                pass

        STATE["live_trading_active"] = True
        return jsonify({"status": "ok", "decision": decision, "price": current_price})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/api/live/stop", methods=["POST"])
def api_live_stop():
    STATE["live_trading_active"] = False
    return jsonify({"status": "ok", "message": "Live trading stopped"})

# -- Training endpoints ----------------------------------------------------
def _train_worker(job_id, params):
    STATE["jobs"][job_id] = {"status": "running", "progress": 0, "msg": "starting"}
    try:
        STATE["jobs"][job_id]["progress"] = 5
        train_df, test_df = load_and_prepare_data(start_date=params.get("start_date"), end_date=params.get("end_date"), interval=params.get("timeframe", CONFIG["timeframe"]))
        STATE["jobs"][job_id]["progress"] = 25
        env, eval_env = create_env(train_df, test_df)
        STATE["jobs"][job_id]["progress"] = 50
        STATE["jobs"][job_id]["msg"] = "training"
        model = train_ppo_model(env, eval_env)
        STATE["model"] = model
        STATE["jobs"][job_id]["progress"] = 100
        STATE["jobs"][job_id]["status"] = "finished"
        STATE["jobs"][job_id]["msg"] = "done"
    except Exception as e:
        STATE["jobs"][job_id]["status"] = "failed"
        STATE["jobs"][job_id]["error"] = str(e)
        STATE["jobs"][job_id]["trace"] = traceback.format_exc()

@app.route("/api/train", methods=["POST"])
def api_train():
    """
    Request body: { optional training params ... }
    This endpoint starts a background training thread and returns a job id.
    In production you should use Celery/Redis.
    """
    params = request.json or {}
    job_id = str(int(time.time() * 1000))
    STATE["jobs"][job_id] = {"status": "queued", "progress": 0, "msg": "queued"}
    # start background thread (non-blocking)
    t = threading.Thread(target=_train_worker, args=(job_id, params), daemon=True)
    t.start()
    return jsonify({"status": "ok", "job_id": job_id})

@app.route("/api/job/<job_id>", methods=["GET"])
def api_job_status(job_id):
    job = STATE["jobs"].get(job_id)
    if job is None:
        return jsonify({"status": "error", "message": "job not found"}), 404
    return jsonify(job)

# -- Misc endpoints -------------------------------------------------------
@app.route("/api/state", methods=["GET"])
def api_state():
    s = {
        "training_progress": STATE["training_progress"],
        "training_status": STATE["training_status"],
        "backtest_available": STATE["backtest_results"] is not None,
        "trade_decisions_count": len(STATE["trade_decisions"]),
        "current_balance": STATE["current_balance"],
        "balance_history_len": len(STATE["balance_history"]),
        "live_trading_active": STATE["live_trading_active"],
    }
    return jsonify(s)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
