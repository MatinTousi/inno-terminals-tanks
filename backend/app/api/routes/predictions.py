"""Leak prediction API (FR-2) — current risk, history, explainability, propagation."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, func, select

from app.api.deps import get_current_user
from app.database import get_session
from app.models.prediction import LeakPrediction
from app.models.tank import Tank
from app.models.user import User
from app.services import leak_prediction

router = APIRouter()


@router.get("")
def latest_predictions(session: Session = Depends(get_session), _: User = Depends(get_current_user)):
    out = []
    tanks = session.exec(select(Tank)).all()
    latest = (
        select(LeakPrediction.tank_id, func.max(LeakPrediction.ts).label("max_ts"))
        .group_by(LeakPrediction.tank_id)
        .subquery()
    )
    predictions = session.exec(
        select(LeakPrediction).join(
            latest,
            (LeakPrediction.tank_id == latest.c.tank_id)
            & (LeakPrediction.ts == latest.c.max_ts),
        )
    ).all()
    predictions_by_tank = {p.tank_id: p for p in predictions}
    for tank in tanks:
        p = predictions_by_tank.get(tank.id)
        if p:
            out.append(_pred_dict(tank, p))
    out.sort(key=lambda d: -d["probability"])
    return out


@router.post("/run")
def run_sweep(session: Session = Depends(get_session), _: User = Depends(get_current_user)):
    preds = leak_prediction.sweep(session)
    return {"scored": len(preds), "high_risk": sum(1 for p in preds if p.will_leak)}


@router.get("/{tank_id}")
def tank_prediction(tank_id: int, recompute: bool = False,
                    session: Session = Depends(get_session), _: User = Depends(get_current_user)):
    tank = session.get(Tank, tank_id)
    if not tank:
        raise HTTPException(404, "tank not found")
    if recompute:
        p = leak_prediction.predict_tank(session, tank)
    else:
        p = session.exec(
            select(LeakPrediction).where(LeakPrediction.tank_id == tank_id)
            .order_by(LeakPrediction.ts.desc())
        ).first() or leak_prediction.predict_tank(session, tank)
    return _pred_dict(tank, p, full=True)


@router.get("/{tank_id}/history")
def prediction_history(tank_id: int, limit: int = Query(200, le=2000),
                       session: Session = Depends(get_session), _: User = Depends(get_current_user)):
    rows = session.exec(
        select(LeakPrediction).where(LeakPrediction.tank_id == tank_id)
        .order_by(LeakPrediction.ts.desc()).limit(limit)
    ).all()
    rows.reverse()
    return [{"ts": r.ts.isoformat(), "probability": r.probability,
             "will_leak": r.will_leak, "model": r.model_name} for r in rows]


def _pred_dict(tank: Tank, p: LeakPrediction, full: bool = False) -> dict:
    d = {
        "tank_id": tank.id, "tank_code": tank.code, "ts": p.ts.isoformat(),
        "probability": p.probability, "horizon_minutes": p.horizon_minutes,
        "confidence_interval": [p.confidence_low, p.confidence_high],
        "will_leak": p.will_leak, "model": p.model_name, "anomaly_score": p.anomaly_score,
    }
    if full:
        d["explanation"] = p.contributions          # FR-2.4 SHAP-lite
        d["propagation"] = p.propagation            # FR-2.8 GNN neighbour spread
    return d
