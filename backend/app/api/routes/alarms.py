"""Alarm queue, acknowledgement (two-person rule), suppression (FR-4)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import client_ip, get_current_user
from app.core import audit
from app.database import get_session
from app.models.alarm import Alarm, AlarmAck, AlarmSeverity, AlarmState
from app.models.user import User
from app.services import alarm_engine

router = APIRouter()


@router.get("")
def list_alarms(
    state: AlarmState | None = None,
    severity: AlarmSeverity | None = None,
    limit: int = Query(200, le=1000),
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    q = select(Alarm).order_by(Alarm.ts.desc())
    if state:
        q = q.where(Alarm.state == state)
    if severity:
        q = q.where(Alarm.severity == severity)
    alarms = session.exec(q.limit(limit)).all()
    # Load all acknowledgements in one query instead of one query per alarm.
    # This avoids an N+1 query pattern when the alarm queue is large.
    alarm_ids = [a.id for a in alarms if a.id is not None]
    acks_by_alarm: dict[int, list[AlarmAck]] = {}
    if alarm_ids:
        for ack in session.exec(
            select(AlarmAck).where(AlarmAck.alarm_id.in_(alarm_ids))
        ).all():
            acks_by_alarm.setdefault(ack.alarm_id, []).append(ack)

    # ISA-18.2 priority ordering: severity rank then recency
    alarms.sort(key=lambda a: (a.severity.rank, -a.ts.timestamp()))
    return [_alarm_dict_from_acks(a, acks_by_alarm.get(a.id, [])) for a in alarms]


@router.get("/rate")
def alarm_rate(session: Session = Depends(get_session), _: User = Depends(get_current_user)):
    return alarm_engine.operator_alarm_rate(session)


@router.get("/{alarm_id}")
def get_alarm(alarm_id: int, session: Session = Depends(get_session),
              _: User = Depends(get_current_user)):
    a = session.get(Alarm, alarm_id)
    if not a:
        raise HTTPException(404, "alarm not found")
    return _alarm_dict(session, a, full=True)


class AckBody(BaseModel):
    note: str = ""


@router.post("/{alarm_id}/ack")
def ack_alarm(alarm_id: int, body: AckBody, request: Request,
              session: Session = Depends(get_session), user: User = Depends(get_current_user)):
    a = session.get(Alarm, alarm_id)
    if not a:
        raise HTTPException(404, "alarm not found")
    if a.state == AlarmState.CLEARED:
        raise HTTPException(409, "alarm already cleared")
    a = alarm_engine.acknowledge(session, a, user, body.note)
    audit.record(session, username=user.username, role=user.role.value,
                 action="alarm.ack", target=f"alarm:{alarm_id}", ip=client_ip(request),
                 detail={"note": body.note})
    return _alarm_dict(session, a, full=True)


@router.post("/{alarm_id}/clear")
def clear_alarm(alarm_id: int, request: Request, session: Session = Depends(get_session),
                user: User = Depends(get_current_user)):
    a = session.get(Alarm, alarm_id)
    if not a:
        raise HTTPException(404, "alarm not found")
    acks = session.exec(select(AlarmAck).where(AlarmAck.alarm_id == alarm_id)).all()
    if a.requires_two_person and len({x.user_id for x in acks}) < 2:
        raise HTTPException(403, "CRITICAL alarm requires two-person acknowledgement before clearing")
    a = alarm_engine.clear(session, a, user)
    audit.record(session, username=user.username, role=user.role.value,
                 action="alarm.clear", target=f"alarm:{alarm_id}", ip=client_ip(request))
    return _alarm_dict(session, a, full=True)


def _alarm_dict(session: Session, a: Alarm, full: bool = False) -> dict:
    acks = session.exec(select(AlarmAck).where(AlarmAck.alarm_id == a.id)).all()
    return _alarm_dict_from_acks(a, acks, full=full)


def _alarm_dict_from_acks(a: Alarm, acks: list[AlarmAck], full: bool = False) -> dict:
    d = {
        "id": a.id, "tank_id": a.tank_id, "ts": a.ts.isoformat(),
        "parameter": a.parameter, "value": a.value, "limit": a.limit,
        "severity": a.severity.value, "state": a.state.value, "kind": a.kind,
        "title": a.title, "message": a.message,
        "requires_two_person": a.requires_two_person,
        "ack_count": len({x.user_id for x in acks}),
        "suppressed_reason": a.suppressed_reason,
    }
    if full:
        d["acks"] = [
            {"user": x.username, "ts": x.ts.isoformat(), "note": x.note,
             "signature": x.digital_signature}
            for x in acks
        ]
        d["root_cause_id"] = a.root_cause_id
    return d
