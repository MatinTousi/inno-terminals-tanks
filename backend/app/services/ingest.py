"""Sensor ingest pipeline (FR-1.1, FR-1.2).

Accepts a batch of readings (from IoT gateways via MQTT/OPC-UA in production;
from the simulator here), persists them, runs the alarm engine, triggers leak
scoring, and fans out realtime updates over the WebSocket hub.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Session, func, select

from app.config import settings
from app.models.reading import Reading
from app.models.tank import Tank, TankStatus
from app.services import alarm_engine, leak_prediction
from app.services.realtime import hub

_PARAMS = (
    "level", "temperature", "pressure", "density", "flammable_gas_ppm",
    "h2s_ppm", "vibration_mm_s", "corrosion_rate_mm_year",
    "acoustic_db", "acoustic_leak_band_ratio", "leak_event",
)

_since_predict: dict[int, int] = {}
_PREDICT_EVERY = 10  # score a tank every N readings


def ingest_batch(session: Session, readings: list[dict], *, score: bool = True) -> dict:
    if len(readings) > settings.max_ingest_batch:
        readings = readings[: settings.max_ingest_batch]

    tanks_by_code = {t.code: t for t in session.exec(select(Tank)).all()}
    tanks_by_id = {t.id: t for t in tanks_by_code.values() if t.id is not None}
    stored: list[Reading] = []
    touched: set[int] = set()

    for item in readings:
        tank = tanks_by_code.get(item.get("tank_code")) or tanks_by_id.get(item.get("tank_id"))
        if not tank:
            continue
        ts = item.get("ts")
        ts = datetime.fromisoformat(ts) if isinstance(ts, str) else (ts or datetime.now(timezone.utc))
        r = Reading(
            tank_id=tank.id, ts=ts, source=item.get("source", "sensor"),
            **{p: float(item.get(p, 0.0) or 0.0) for p in _PARAMS if p != "leak_event"},
            leak_event=int(item.get("leak_event", 0)),
        )
        session.add(r)
        stored.append(r)
        touched.add(tank.id)

    session.commit()

    alarms_raised = []
    predictions = []
    for tank_id in touched:
        tank = session.get(Tank, tank_id)
        latest = session.exec(
            select(Reading).where(Reading.tank_id == tank_id).order_by(Reading.ts.desc())
        ).first()
        if latest:
            alarms_raised += alarm_engine.evaluate_reading(session, tank, latest)

        _since_predict[tank_id] = _since_predict.get(tank_id, 0) + 1
        if score and _since_predict[tank_id] >= _PREDICT_EVERY:
            _since_predict[tank_id] = 0
            before = _active_alarm_ids(session, tank_id)
            predictions.append(leak_prediction.predict_tank(session, tank))
            after = _active_alarm_ids(session, tank_id)
            for aid in after - before:
                a = session.get(alarm_engine.Alarm, aid)
                if a:
                    alarms_raised.append(a)

        hub.publish_soon("reading", _reading_payload(tank, latest))

    for a in alarms_raised:
        hub.publish_soon("alarm", {
            "id": a.id, "tank_id": a.tank_id, "severity": a.severity.value,
            "state": a.state.value, "title": a.title, "kind": a.kind, "ts": a.ts.isoformat(),
        })
    for p in predictions:
        hub.publish_soon("prediction", {
            "tank_id": p.tank_id, "probability": p.probability, "will_leak": p.will_leak,
            "model": p.model_name, "propagation": p.propagation,
        })

    return {
        "stored": len(stored),
        "tanks_touched": len(touched),
        "alarms_raised": len(alarms_raised),
        "predictions": len(predictions),
    }


def _active_alarm_ids(session: Session, tank_id: int) -> set[int]:
    from app.models.alarm import Alarm, AlarmState

    rows = session.exec(
        select(Alarm.id).where(
            Alarm.tank_id == tank_id,
            Alarm.state.in_([AlarmState.ACTIVE, AlarmState.ACKNOWLEDGED, AlarmState.SUPPRESSED]),
        )
    ).all()
    return set(rows)


def _reading_payload(tank: Tank, r: Reading) -> dict:
    return {
        "tank_id": tank.id, "tank_code": tank.code, "status": tank.status.value,
        "ts": r.ts.isoformat(),
        "level": round(r.level, 2), "temperature": round(r.temperature, 2),
        "pressure": round(r.pressure, 3), "flammable_gas_ppm": round(r.flammable_gas_ppm, 2),
        "h2s_ppm": round(r.h2s_ppm, 2), "vibration_mm_s": round(r.vibration_mm_s, 3),
        "acoustic_leak_band_ratio": round(r.acoustic_leak_band_ratio, 3),
    }


def mark_stale_tanks(session: Session, *, seconds: int = 120) -> int:
    """Set tanks OFFLINE when no reading has arrived recently (FR-1 / NFR-2.2)."""
    cutoff = datetime.now(timezone.utc).timestamp() - seconds
    n = 0
    tanks = session.exec(select(Tank)).all()
    latest_ts = session.exec(
        select(Reading.tank_id, func.max(Reading.ts))
        .group_by(Reading.tank_id)
    ).all()
    latest_by_tank = {tank_id: ts for tank_id, ts in latest_ts}

    for tank in tanks:
        latest = latest_by_tank.get(tank.id)
        stale = (latest is None or
                 latest.replace(tzinfo=timezone.utc).timestamp() < cutoff)
        if stale and tank.status != TankStatus.OFFLINE:
            tank.status = TankStatus.OFFLINE
            session.add(tank)
            n += 1
    session.commit()
    return n
