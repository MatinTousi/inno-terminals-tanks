"""Alarm & event engine — FR-4 + ISA-18.2 rationalization / flood suppression."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from app.config import settings
from app.core.security import digital_signature
from app.models.alarm import Alarm, AlarmAck, AlarmSeverity, AlarmState
from app.models.tank import Tank, Threshold
from app.models.user import User, utcnow

# default parameter limits used when a tank has no explicit Threshold row
DEFAULT_LIMITS: dict[str, dict] = {
    "level":             {"hi_warn": 85, "hi_critical": 95, "unit": "%"},
    "temperature":       {"hi_warn": 45, "hi_critical": 55, "lo_warn": 2, "unit": "°C"},
    "pressure":          {"hi_warn": 3.0, "hi_critical": 4.0, "lo_warn": 0.6, "unit": "bar"},
    "flammable_gas_ppm": {"hi_warn": 40, "hi_critical": 80, "unit": "ppm"},
    "h2s_ppm":           {"hi_warn": 5, "hi_critical": 10, "unit": "ppm"},
    "vibration_mm_s":    {"hi_warn": 6, "hi_critical": 9, "unit": "mm/s"},
    "corrosion_rate_mm_year": {"hi_warn": 0.2, "hi_critical": 0.35, "unit": "mm/yr"},
}

_MONITORED = list(DEFAULT_LIMITS)


def _effective_limits(session: Session, tank_id: int, parameter: str) -> dict:
    row = session.exec(
        select(Threshold).where(Threshold.tank_id == tank_id, Threshold.parameter == parameter)
    ).first()
    base = dict(DEFAULT_LIMITS.get(parameter, {}))
    if row:
        for k in ("lo_warn", "lo_critical", "hi_warn", "hi_critical"):
            v = getattr(row, k)
            if v is not None:
                base[k] = v
    return base


def evaluate_reading(session: Session, tank: Tank, reading) -> list[Alarm]:
    """Return newly-raised alarms for one reading (threshold breaches)."""
    raised: list[Alarm] = []
    worst = AlarmStatusTracker()

    # Load all configured thresholds once for this tank instead of issuing one
    # query per monitored parameter.
    threshold_rows = session.exec(
        select(Threshold).where(Threshold.tank_id == tank.id)
    ).all()
    thresholds = {row.parameter: row for row in threshold_rows}

    # Load the tank's active/suppressed threshold alarms once. The previous
    # implementation queried Alarm separately for every monitored parameter.
    active_states = [
        AlarmState.ACTIVE, AlarmState.ACKNOWLEDGED, AlarmState.SUPPRESSED
    ]
    existing_rows = session.exec(
        select(Alarm).where(
            Alarm.tank_id == tank.id,
            Alarm.kind == "threshold",
            Alarm.state.in_(active_states),
        )
    ).all()
    existing_by_dedup = {alarm.dedup_key: alarm for alarm in existing_rows}

    for param in _MONITORED:
        value = getattr(reading, param, None)
        if value is None:
            continue
        row = thresholds.get(param)
        lim = dict(DEFAULT_LIMITS.get(param, {}))
        if row:
            for key in ("lo_warn", "lo_critical", "hi_warn", "hi_critical"):
                value_override = getattr(row, key)
                if value_override is not None:
                    lim[key] = value_override
        severity, limit, direction = _classify(value, lim)
        worst.update(severity)
        if severity is None:
            for stale in existing_rows:
                if (
                    stale.parameter == param
                    and stale.state in (AlarmState.ACTIVE, AlarmState.ACKNOWLEDGED)
                ):
                    stale.state = AlarmState.CLEARED
                    stale.cleared_at = utcnow()
                    session.add(stale)
            continue

        dedup = f"threshold:{tank.id}:{param}:{direction}"
        existing = existing_by_dedup.get(dedup)
        if existing:
            existing.value = value  # refresh, don't duplicate (FR-4.5)
            session.add(existing)
            continue

        alarm = Alarm(
            tank_id=tank.id,
            ts=utcnow(),
            parameter=param,
            value=float(value),
            limit=limit,
            severity=severity,
            kind="threshold",
            title=f"{tank.code} — {param} {direction} {limit}{lim.get('unit','')}",
            message=f"{param}={value:.2f} breached {direction} limit {limit}",
            dedup_key=dedup,
            requires_two_person=(severity == AlarmSeverity.CRITICAL),
        )
        _apply_isa_18_2(session, alarm)
        session.add(alarm)
        existing_by_dedup[dedup] = alarm
        raised.append(alarm)

    # update tank colour code (FR-1.3)
    tank.status = worst.tank_status()
    session.add(tank)
    session.commit()
    for a in raised:
        session.refresh(a)
    return raised


def _classify(value: float, lim: dict):
    if "hi_critical" in lim and value >= lim["hi_critical"]:
        return AlarmSeverity.CRITICAL, lim["hi_critical"], "above"
    if "lo_critical" in lim and value <= lim["lo_critical"]:
        return AlarmSeverity.CRITICAL, lim["lo_critical"], "below"
    if "hi_warn" in lim and value >= lim["hi_warn"]:
        return AlarmSeverity.HIGH, lim["hi_warn"], "above"
    if "lo_warn" in lim and value <= lim["lo_warn"]:
        return AlarmSeverity.HIGH, lim["lo_warn"], "below"
    return None, None, ""


def _auto_clear(session: Session, tank_id: int, parameter: str) -> None:
    stale = session.exec(
        select(Alarm).where(
            Alarm.tank_id == tank_id,
            Alarm.parameter == parameter,
            Alarm.kind == "threshold",
            Alarm.state.in_([AlarmState.ACTIVE, AlarmState.ACKNOWLEDGED]),
        )
    ).all()
    for a in stale:
        a.state = AlarmState.CLEARED
        a.cleared_at = utcnow()
        session.add(a)


def _apply_isa_18_2(session: Session, alarm: Alarm) -> None:
    """Flood suppression: if too many alarms fired for this tank in a short
    window, mark the new one SUPPRESSED and point it at the root cause."""
    window_start = datetime.now(timezone.utc) - timedelta(seconds=settings.alarm_flood_window_seconds)
    recent = session.exec(
        select(Alarm).where(Alarm.tank_id == alarm.tank_id, Alarm.ts >= window_start)
        .order_by(Alarm.ts)
    ).all()
    if len(recent) >= settings.alarm_flood_threshold:
        root = recent[0]
        alarm.state = AlarmState.SUPPRESSED
        alarm.root_cause_id = root.id
        alarm.suppressed_reason = f"ISA-18.2 flood suppression (root alarm #{root.id})"


def raise_alarm(session: Session, **kwargs) -> Alarm:
    """Raise a non-threshold alarm (leak prediction, deviation, system)."""
    dedup = kwargs.get("dedup_key", "")
    if dedup:
        existing = session.exec(
            select(Alarm).where(
                Alarm.dedup_key == dedup,
                Alarm.state.in_([AlarmState.ACTIVE, AlarmState.ACKNOWLEDGED]),
            )
        ).first()
        if existing:
            return existing
    alarm = Alarm(**kwargs)
    if alarm.severity == AlarmSeverity.CRITICAL:
        alarm.requires_two_person = True
    _apply_isa_18_2(session, alarm)
    session.add(alarm)
    session.commit()
    session.refresh(alarm)
    return alarm


def acknowledge(session: Session, alarm: Alarm, user: User, note: str = "") -> Alarm:
    ack = AlarmAck(
        alarm_id=alarm.id,
        user_id=user.id,
        username=user.username,
        note=note,
        digital_signature=digital_signature(alarm.id, user.username, utcnow().isoformat()),
    )
    session.add(ack)
    session.flush()

    acks = session.exec(select(AlarmAck).where(AlarmAck.alarm_id == alarm.id)).all()
    distinct_users = {a.user_id for a in acks}

    if alarm.requires_two_person and len(distinct_users) < 2:
        alarm.state = AlarmState.ACKNOWLEDGED  # partial — still needs a 2nd person
        alarm.message += " [awaiting 2nd acknowledgement — two-person rule]"
    else:
        alarm.state = AlarmState.ACKNOWLEDGED
    session.add(alarm)
    session.commit()
    session.refresh(alarm)
    return alarm


def clear(session: Session, alarm: Alarm, user: User) -> Alarm:
    alarm.state = AlarmState.CLEARED
    alarm.cleared_at = utcnow()
    session.add(alarm)
    session.commit()
    session.refresh(alarm)
    return alarm


def operator_alarm_rate(session: Session) -> dict:
    """ISA-18.2 KPI: alarms/hour vs the 6/operator/hour target (FR-4.6)."""
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    active_ops = max(1, len(session.exec(select(User).where(User.is_active)).all()))
    count = len(session.exec(select(Alarm).where(Alarm.ts >= since)).all())
    per_op = count / active_ops
    return {
        "alarms_last_hour": count,
        "per_operator_hour": round(per_op, 2),
        "isa_18_2_target": settings.alarm_rate_limit_per_operator_hour,
        "within_target": per_op <= settings.alarm_rate_limit_per_operator_hour,
    }


class AlarmStatusTracker:
    def __init__(self) -> None:
        self._worst: AlarmSeverity | None = None

    def update(self, sev: AlarmSeverity | None) -> None:
        if sev is None:
            return
        if self._worst is None or sev.rank < self._worst.rank:
            self._worst = sev

    def tank_status(self):
        from app.models.tank import TankStatus

        if self._worst in (AlarmSeverity.CRITICAL, AlarmSeverity.HIGH):
            return TankStatus.RED if self._worst == AlarmSeverity.CRITICAL else TankStatus.YELLOW
        return TankStatus.GREEN
