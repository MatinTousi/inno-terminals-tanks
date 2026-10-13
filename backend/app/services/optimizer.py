"""Loading/unloading optimization (FR-3).

Baseline: priority + earliest-deadline greedy across pumps, with API RP 2350
overfill guard and an energy term (FR-3.6).

When the IP vault is unlocked, the proprietary risk-aware optimizer (feature #2,
FR-3.7) replaces the objective with one that folds in AI-predicted leak risk.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, func, select

from app import secure
from app.models.operation import Operation, OperationState
from app.models.tank import Tank
from app.services import leak_prediction


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _duration_h(volume_m3: float, flow: float) -> float:
    return volume_m3 / max(flow, 1.0)


def _energy_kwh(volume_m3: float, flow: float, head_m: float = 25.0) -> float:
    rho, g, eta = 1000.0, 9.81, 0.62
    q = flow / 3600.0
    return (rho * g * q * head_m / eta / 1000.0) * _duration_h(volume_m3, flow)


def _latest_levels(session: Session) -> dict[int, float]:
    """Fetch the latest level for every tank with one grouped query."""
    from app.models.reading import Reading

    latest = (
        select(Reading.tank_id, func.max(Reading.ts).label("max_ts"))
        .group_by(Reading.tank_id)
        .subquery()
    )
    rows = session.exec(
        select(Reading.tank_id, Reading.level).join(
            latest,
            (Reading.tank_id == latest.c.tank_id) & (Reading.ts == latest.c.max_ts),
        )
    ).all()
    return {tank_id: level for tank_id, level in rows}


def build_plan(session: Session, *, pump_count: int = 3, horizon_hours: int = 24) -> dict:
    run_id = uuid.uuid4().hex[:12]
    start = datetime.now(timezone.utc)
    ops = session.exec(
        select(Operation).where(
            Operation.state.in_([OperationState.REQUESTED, OperationState.SCHEDULED])
        )
    ).all()
    tanks = {t.id: t for t in session.exec(select(Tank)).all()}

    op_dicts = []
    for o in ops:
        op_dicts.append({
            "id": o.id, "ref": o.ref, "tank_id": o.tank_id, "kind": o.kind.value,
            "volume_m3": o.volume_m3, "flow_rate_m3ph": o.flow_rate_m3ph,
            "priority": o.priority, "earliest_start": _aware(o.earliest_start),
            "due_by": _aware(o.due_by),
        })
    latest_levels = _latest_levels(session)
    tank_dicts = {
        t.id: {
            "capacity_m3": t.capacity_m3, "max_safe_level_pct": t.max_safe_level_pct,
            "level_pct": latest_levels.get(t.id, 50.0),
        }
        for t in tanks.values()
    }

    prop = secure.load("risk_aware_optimizer")
    if prop is not None:
        def risk_fn(tank_id: int, at_time: datetime) -> float:
            return leak_prediction.risk_at(session, tank_id, at_time)

        plan = prop.optimize(
            op_dicts, tank_dicts, risk_fn,
            start=start, run_id=run_id, pump_count=pump_count, horizon_hours=horizon_hours,
        )
        plan["risk_aware"] = True
    else:
        plan = _baseline_plan(op_dicts, tank_dicts, start, run_id, pump_count)
        plan["risk_aware"] = False

    _persist_plan(session, ops, plan, run_id)
    return plan


def _baseline_plan(op_dicts, tank_dicts, start, run_id, pump_count) -> dict:
    far = datetime.max.replace(tzinfo=timezone.utc)
    ops = sorted(op_dicts, key=lambda o: (o["priority"], o["due_by"] or far))
    pump_free = {f"P-{i+1}": start for i in range(max(pump_count, 1))}
    schedule = []
    for o in ops:
        dur = timedelta(hours=_duration_h(o["volume_m3"], o["flow_rate_m3ph"]))
        pump = min(pump_free, key=lambda p: pump_free[p])
        s0 = max(pump_free[pump], o["earliest_start"] or start)
        pump_free[pump] = s0 + dur
        tank = tank_dicts.get(o["tank_id"], {})
        projected = tank.get("level_pct", 50) + (
            o["volume_m3"] / max(tank.get("capacity_m3", 1e4), 1) * 100 if o["kind"] == "loading" else 0
        )
        schedule.append({
            "operation_id": o["id"], "ref": o["ref"], "tank_id": o["tank_id"],
            "kind": o["kind"], "pump_id": pump,
            "planned_start": s0.isoformat(), "planned_end": (s0 + dur).isoformat(),
            "predicted_leak_risk": 0.0,
            "energy_kwh": round(_energy_kwh(o["volume_m3"], o["flow_rate_m3ph"]), 2),
            "overfill_warning": projected > tank.get("max_safe_level_pct", 90),
        })
    ends = [datetime.fromisoformat(s["planned_end"]) for s in schedule]
    return {
        "run_id": run_id, "algorithm": "priority-edd-greedy",
        "makespan_minutes": round((max(ends) - start).total_seconds() / 60, 1) if ends else 0.0,
        "total_energy_kwh": round(sum(s["energy_kwh"] for s in schedule), 1),
        "max_leak_risk": 0.0, "high_risk_deferrals": 0,
        "schedule": schedule,
    }


def _persist_plan(session: Session, ops: list[Operation], plan: dict, run_id: str) -> None:
    by_id = {o.id: o for o in ops}
    for item in plan["schedule"]:
        o = by_id.get(item["operation_id"])
        if not o:
            continue
        o.planned_start = datetime.fromisoformat(item["planned_start"])
        o.planned_end = datetime.fromisoformat(item["planned_end"])
        o.pump_id = item["pump_id"]
        o.predicted_leak_risk = item.get("predicted_leak_risk", 0.0)
        o.energy_kwh = item.get("energy_kwh", 0.0)
        o.optimizer_run_id = run_id
        o.state = OperationState.SCHEDULED
        session.add(o)
    session.commit()


def what_if(session: Session, *, disabled_pumps: list[str] | None = None,
            extra_ops: list[dict] | None = None, pump_count: int = 3) -> dict:
    """FR-3.3 — scenario simulation without persisting."""
    disabled = set(disabled_pumps or [])
    effective = max(1, pump_count - len(disabled))
    start = datetime.now(timezone.utc)
    ops = session.exec(
        select(Operation).where(
            Operation.state.in_([OperationState.REQUESTED, OperationState.SCHEDULED])
        )
    ).all()
    op_dicts = [{
        "id": o.id, "ref": o.ref, "tank_id": o.tank_id, "kind": o.kind.value,
        "volume_m3": o.volume_m3, "flow_rate_m3ph": o.flow_rate_m3ph,
        "priority": o.priority, "earliest_start": _aware(o.earliest_start),
        "due_by": _aware(o.due_by),
    } for o in ops]
    for e in extra_ops or []:
        op_dicts.append({
            "id": None, "ref": e.get("ref", "WHATIF"), "tank_id": e["tank_id"],
            "kind": e.get("kind", "loading"), "volume_m3": e.get("volume_m3", 1000),
            "flow_rate_m3ph": e.get("flow_rate_m3ph", 300), "priority": e.get("priority", 3),
            "earliest_start": None, "due_by": None,
        })
    tanks = session.exec(select(Tank)).all()
    latest_levels = _latest_levels(session)
    tank_dicts = {
        t.id: {
            "capacity_m3": t.capacity_m3,
            "max_safe_level_pct": t.max_safe_level_pct,
            "level_pct": latest_levels.get(t.id, 50.0),
        }
        for t in tanks
    }
    return _baseline_plan(op_dicts, tank_dicts, start, "whatif", effective) | {
        "disabled_pumps": sorted(disabled), "effective_pumps": effective,
    }
