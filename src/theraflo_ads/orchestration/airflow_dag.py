"""Airflow DAG for the scheduled jobs (design doc: Retraining Strategy → schedules).

    drift_check      daily     theraflo-ads monitor drift
    retrain_check    daily     theraflo-ads retrain      (acts only when the controller says so)
    pipeline         weekly    dvc repro                  (scheduled retrain)
    finalize_labels  hourly    theraflo-ads labels finalize
    deletions        hourly    theraflo-ads privacy delete-pending   (crypto-shredding)
    review_eval      weekly    theraflo-ads review eval              (offline gates, no spend)

Copy or symlink this file into the Airflow dags folder. Airflow is optional:
importing this module without it installed defines nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta

try:  # pragma: no cover - exercised only where Airflow is installed
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except ImportError:  # pragma: no cover
    DAG = None

DEFAULT_ARGS = {
    "owner": "theraflo-ads",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "retry_exponential_backoff": True,
}

if DAG is not None:  # pragma: no cover
    with DAG(
        "theraflo_ads_daily",
        default_args=DEFAULT_ARGS,
        schedule="0 6 * * *",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["theraflo-ads"],
    ) as daily:
        drift = BashOperator(task_id="drift_check", bash_command="theraflo-ads monitor drift")
        retrain = BashOperator(task_id="retrain_check", bash_command="theraflo-ads retrain")
        drift >> retrain

    with DAG(
        "theraflo_ads_weekly_pipeline",
        default_args=DEFAULT_ARGS,
        schedule="0 3 * * 1",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["theraflo-ads"],
    ) as weekly:
        BashOperator(task_id="dvc_repro", bash_command="dvc repro")
        BashOperator(
            task_id="review_eval",
            bash_command="theraflo-ads review eval --reviewer simulated",
        )

    with DAG(
        "theraflo_ads_hourly",
        default_args=DEFAULT_ARGS,
        schedule="15 * * * *",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["theraflo-ads"],
    ) as hourly:
        BashOperator(task_id="finalize_labels", bash_command="theraflo-ads labels finalize")
        # Deletion requests are honoured within the hour.
        BashOperator(task_id="deletions", bash_command="theraflo-ads privacy delete-pending")
