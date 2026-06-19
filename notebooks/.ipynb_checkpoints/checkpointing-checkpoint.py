import pickle
import numpy as np
from dataclasses import asdict


# ============================================================
# SERIALIZATION HELPERS
# ============================================================

def _serialize_event(e):
    """
    Convert IonEvent-like object into plain dict.
    Works with dataclass or simple object.
    """
    return {
        "t": float(e.t),
        "pos": np.array(e.pos, dtype=float),
        "vel": np.array(e.vel, dtype=float),
        "energy": float(e.energy),
        "mass": float(e.mass),
        "charge": int(e.charge),
    }


def _deserialize_event(d, IonEventClass):
    """
    Reconstruct IonEvent from dict.
    """
    return IonEventClass(
        t=d["t"],
        pos=d["pos"],
        vel=d["vel"],
        energy=d["energy"],
        mass=d["mass"],
        charge=d["charge"],
    )


# ============================================================
# CHECKPOINT STRUCTURE
# ============================================================

def save_checkpoint(filename, events, rng_state=None, meta=None):
    """
    Save ion event stream + RNG state + optional metadata.
    """

    data = {
        "events": [_serialize_event(e) for e in events],
        "rng_state": rng_state if rng_state is not None else np.random.get_state(),
        "meta": meta if meta is not None else {},
    }

    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_checkpoint(filename, IonEventClass=None, restore_rng=True):
    """
    Load event stream.

    Parameters:
    - IonEventClass: class used to reconstruct events (optional)
    - restore_rng: whether to restore numpy RNG state
    """

    with open(filename, "rb") as f:
        data = pickle.load(f)

    if restore_rng:
        np.random.set_state(data["rng_state"])

    if IonEventClass is None:
        # return raw dicts
        return data["events"], data["meta"]

    events = [
        _deserialize_event(e, IonEventClass)
        for e in data["events"]
    ]

    return events, data["meta"]


# ============================================================
# STREAM APPEND MODE (useful for long runs)
# ============================================================

def append_checkpoint(filename, new_events):
    """
    Append events to existing checkpoint file.
    Does NOT touch RNG state.
    """

    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        data = {"events": [], "meta": {}}

    data["events"].extend([_serialize_event(e) for e in new_events])

    with open(filename, "wb") as f:
        pickle.dump(data, f)


# ============================================================
# QUICK INSPECTION UTILITIES
# ============================================================

def load_event_times(filename):
    """
    Fast access to just event times (useful for diagnostics).
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)

    return np.array([e["t"] for e in data["events"]])


def summary(filename):
    """
    Print basic statistics.
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)

    events = data["events"]

    if len(events) == 0:
        print("Empty checkpoint")
        return

    times = np.array([e["t"] for e in events])

    print("Events:", len(events))
    print("t_min:", times.min())
    print("t_max:", times.max())
    print("mean dt:", np.mean(np.diff(np.sort(times))) if len(times) > 1 else None)