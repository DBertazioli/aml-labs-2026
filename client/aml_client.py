"""Client for the AML lab server.

In Colab:
    !wget -q -O aml_client.py https://raw.githubusercontent.com/<user>/<repo>/main/client/aml_client.py
    import aml_client as aml
    aml.register("name.surname@campus.unimib.it", "my_nickname")
    aml.submit("hello", message="hi")
"""
import json
import os

import requests

# Set to the deployed HF Space URL at deploy time (Task 8).
SERVER = os.environ.get("AML_SERVER", "https://dbertazioli-aml-labs.hf.space")
TOKEN_FILE = os.path.expanduser("~/.aml_token")


def _call(method, path, **kwargs):
    r = requests.request(method, SERVER + path, timeout=30, **kwargs)
    if r.status_code >= 400:
        try:
            detail = r.json()["detail"]
        except Exception:
            detail = r.text[:200]
        raise RuntimeError(f"Server error {r.status_code}: {detail}")
    return r.json()


def _token():
    try:
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        raise RuntimeError("Run aml.register(email, nickname) first") from None


def _plain(obj):
    # numpy arrays, numpy scalars and torch tensors all have .tolist()
    return json.loads(json.dumps(obj, default=lambda o: o.tolist()), parse_constant=lambda _: None)


def register(email, nickname):
    out = _call("POST", "/register", json={"email": email, "nickname": nickname})
    with open(TOKEN_FILE, "w") as f:
        f.write(out["token"])
    print(f"Registered as {out['nickname']}")


def submit(experiment, **payload):
    return _call("POST", f"/submit/{experiment}", json={"token": _token(), "payload": _plain(payload)})


def results(experiment):
    return _call("GET", f"/results/{experiment}")


def compete(challenge, predictions):
    raw = _plain(predictions)
    if any(p != int(p) for p in raw):
        raise ValueError("predictions must be class labels (ints), not probabilities")
    preds = [int(p) for p in raw]
    out = _call("POST", f"/compete/{challenge}", json={"token": _token(), "predictions": preds})
    print(f"Public accuracy: {out['public_accuracy']:.4f}")
    return out


def leaderboard(challenge):
    return _call("GET", f"/leaderboard/{challenge}")
