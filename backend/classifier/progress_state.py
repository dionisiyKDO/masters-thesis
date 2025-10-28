# api/classifier/progress_state.py
from threading import Lock

_progress = {
    "status": "idle",
    "epoch": 0,
    "total_epochs": 0,
    "loss": None,
    "val_loss": None,
    "acc": None,
    "val_acc": None,
    "message": None,
}
_lock = Lock()

def update_progress(**kwargs):
    with _lock:
        _progress.update(kwargs)

def get_progress():
    with _lock:
        return _progress.copy()

def reset_progress():
    with _lock:
        _progress.update({
            "status": "idle",
            "epoch": 0,
            "total_epochs": 0,
            "loss": None,
            "val_loss": None,
            "acc": None,
            "val_acc": None,
            "message": None,
        })
