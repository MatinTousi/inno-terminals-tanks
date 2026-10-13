"""Tanks, terminal map, thresholds (FR-1.3, FR-1.5, UI-1)."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlmodel import Session, func, select

from app.api.deps import client_ip, get_current_user, require_roles
from app.core import audit
from app.database import get_session
from app.models.prediction import LeakPrediction
from app.models.reading import Reading
from app.models.tank import Pipeline, Tank, Threshold
from app.models.user import Role, User

router = APIRouter()


class TankOut(BaseModel):
    id: int
    code: str
    name: str
    product: str
    capacity_m3: float
    max_safe_level_pct: float
    critical_level_pct: float
    zone: str
    map_x: float
    map_y: float
    status: str
    latest: dict | None = None
    leak_probability: float | None = None


@router.get("", response_model=list[TankOut])
def list_tanks(session: Session = Depends(get_session), _: User = Depends(get_current_user)):
    tanks = session.exec(select(Tank).order_by(Tank.code)).all()
    latest_by_tank = _latest_readings_by_tank(session)
    latest_pred_by_tank = _latest_predictions_by_tank(session)
    out = []
    for t in tanks:
        latest = latest_by_tank.get(t.id)
        pred = latest_pred_by_tank.get(t.id)
        out.append(TankOut(
            id=t.id, code=t.code, name=t.name, product=t.product,
            capacity_m3=t.capacity_m3, max_safe_level_pct=t.max_safe_level_pct,
            critical_level_pct=t.critical_level_pct, zone=t.zone,
            map_x=t.map_x, map_y=t.map_y, status=t.status.value,
            latest=_latest_dict(latest),
            leak_probability=pred.probability if pred else None,
        ))
    return out


def _latest_readings_by_tank(session: Session) -> dict[int, Reading]:
    latest = (
        select(Reading.tank_id, func.max(Reading.ts).label("max_ts"))
        .group_by(Reading.tank_id)
        .subquery()
    )
    rows = session.exec(
        select(Reading).join(
            latest,
            (Reading.tank_id == latest.c.tank_id) & (Reading.ts == latest.c.max_ts),
        )
    ).all()
    return {r.tank_id: r for r in rows}


def _latest_predictions_by_tank(session: Session) -> dict[int, LeakPrediction]:
    latest = (
        select(LeakPrediction.tank_id, func.max(LeakPrediction.ts).label("max_ts"))
        .group_by(LeakPrediction.tank_id)
        .subquery()
    )
    rows = session.exec(
        select(LeakPrediction).join(
            latest,
            (LeakPrediction.tank_id == latest.c.tank_id)
            & (LeakPrediction.ts == latest.c.max_ts),
        )
    ).all()
    return {p.tank_id: p for p in rows}


def _latest_dict(r: Reading | None) -> dict | None:
    if not r:
        return None
    return {
        "ts": r.ts.isoformat(), "level": round(r.level, 2), "temperature": round(r.temperature, 2),
        "pressure": round(r.pressure, 3), "density": round(r.density, 4),
        "flammable_gas_ppm": round(r.flammable_gas_ppm, 2), "h2s_ppm": round(r.h2s_ppm, 2),
        "vibration_mm_s": round(r.vibration_mm_s, 3),
        "corrosion_rate_mm_year": round(r.corrosion_rate_mm_year, 4),
        "acoustic_db": round(r.acoustic_db, 2),
        "acoustic_leak_band_ratio": round(r.acoustic_leak_band_ratio, 3),
    }


@router.get("/map")
def terminal_map(session: Session = Depends(get_session), _: User = Depends(get_current_user)):
    tanks = session.exec(select(Tank)).all()
    pipes = session.exec(select(Pipeline)).all()
    return {
        "tanks": [
            {"id": t.id, "code": t.code, "x": t.map_x, "y": t.map_y,
             "zone": t.zone, "status": t.status.value, "product": t.product}
            for t in tanks
        ],
        "pipelines": [
            {"id": p.id, "code": p.code, "from": p.from_tank_id, "to": p.to_tank_id}
            for p in pipes
        ],
    }


@router.get("/{tank_id}", response_model=TankOut)
def get_tank(tank_id: int, session: Session = Depends(get_session),
             _: User = Depends(get_current_user)):
    t = session.get(Tank, tank_id)
    if not t:
        raise HTTPException(404, "tank not found")
    latest = session.exec(
        select(Reading).where(Reading.tank_id == t.id).order_by(Reading.ts.desc())
    ).first()
    pred = session.exec(
        select(LeakPrediction).where(LeakPrediction.tank_id == t.id)
        .order_by(LeakPrediction.ts.desc())
    ).first()
    return TankOut(
        id=t.id, code=t.code, name=t.name, product=t.product, capacity_m3=t.capacity_m3,
        max_safe_level_pct=t.max_safe_level_pct, critical_level_pct=t.critical_level_pct,
        zone=t.zone, map_x=t.map_x, map_y=t.map_y, status=t.status.value,
        latest=_latest_dict(latest), leak_probability=pred.probability if pred else None,
    )


# --- thresholds (FR-1.5) — safety engineers / ops managers ---
class ThresholdBody(BaseModel):
    parameter: str
    lo_warn: float | None = None
    lo_critical: float | None = None
    hi_warn: float | None = None
    hi_critical: float | None = None
    unit: str = ""


@router.get("/{tank_id}/thresholds")
def get_thresholds(tank_id: int, session: Session = Depends(get_session),
                   _: User = Depends(get_current_user)):
    return session.exec(select(Threshold).where(Threshold.tank_id == tank_id)).all()


@router.put("/{tank_id}/thresholds")
def set_threshold(
    tank_id: int, body: ThresholdBody, request: Request,
    session: Session = Depends(get_session),
    user: User = Depends(require_roles(Role.SAFETY_ENGINEER, Role.OPS_MANAGER)),
):
    if not session.get(Tank, tank_id):
        raise HTTPException(404, "tank not found")
    row = session.exec(
        select(Threshold).where(Threshold.tank_id == tank_id, Threshold.parameter == body.parameter)
    ).first() or Threshold(tank_id=tank_id, parameter=body.parameter)
    for f in ("lo_warn", "lo_critical", "hi_warn", "hi_critical", "unit"):
        setattr(row, f, getattr(body, f))
    row.updated_by = user.username
    row.updated_at = datetime.now(timezone.utc)
    session.add(row)
    audit.record(session, username=user.username, role=user.role.value,
                 action="threshold.update", target=f"tank:{tank_id}:{body.parameter}",
                 ip=client_ip(request), detail=body.model_dump())
    session.commit()
    session.refresh(row)
    return row
