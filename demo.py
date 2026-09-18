"""Headless end-to-end demo: one command, every claim in the README recomputed.

    python demo.py                 # 72h closed loop, normal scenario
    python demo.py --hours 168 --scenario high_volatility --horizon 48
    python demo.py --all-scenarios # what-if table across all six scenarios
"""
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from src.agent import run_agent, what_if  # noqa: E402
from src.data_loader import describe  # noqa: E402
from src.pipeline import get_data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="GridFlex AI demo")
    ap.add_argument("--hours", type=int, default=72, help="closed-loop backtest hours")
    ap.add_argument("--horizon", type=int, default=24, help="optimisation horizon")
    ap.add_argument("--scenario", default="normal")
    ap.add_argument("--risk-level", type=float, default=None)
    ap.add_argument("--all-scenarios", action="store_true")
    ap.add_argument("--llm", action="store_true", help="polish the plan with Claude")
    a = ap.parse_args()

    pd.set_option("display.width", 160)
    print("=" * 78)
    print("GridFlex AI - Risk-Aware Battery Trading & Dispatch")
    print("=" * 78)
    d = describe(get_data())
    print(f"DATA      {d['rows']:,} hourly rows  {d['start']} -> {d['end']}")
    print(f"          price mean {d['price_mean']:,.1f}  std {d['price_std']:,.1f}  "
          f"min {d['price_min']:,.1f}  max {d['price_max']:,.1f}  "
          f"negative hours {d['negative_hours']:,} ({d['negative_pct']:.1f}%)")

    st = run_agent(a.scenario, horizon=a.horizon, risk_level=a.risk_level,
                   backtest_hours=a.hours, with_open_loop=True, use_llm=a.llm)
    r = st["result"]
    if r is None:
        print("FAILED:", st.get("error"))
        return
    bt, bench = r["backtest"], r["benchmark"]

    print(f"\nAGENT     engine={st['engine']}  scenario={r['scenario']}  "
          f"policy={ {k: round(v, 3) for k, v in r['policy'].items()} }")
    for line in st["trace"]:
        print(f"          - {line}")

    print(f"\nDECISION  now={r['now_ts']}  price={r['current_price']:,.1f}/MWh")
    print(f"          ACTION={r['current_action']}  SOC={r['current_soc_pct']:.1f}%  "
          f"risk={r['risk_level']} ({r['risk_score']:.2f})  "
          f"adaptive reserve={r['adaptive_reserve']:.0%} "
          f"({r['adaptive_reserve_mwh']:,.1f} MWh floor)")
    print(f"          breakeven spread={r['breakeven_spread']:,.1f}/MWh  "
          f"horizon value={r['expected_profit']:,.0f}")

    print(f"\nFORECAST  model={r['forecast_model']}  "
          f"holdout MAE={r['forecast_metrics_holdout'].get('mae', float('nan')):.2f}  "
          f"RMSE={r['forecast_metrics_holdout'].get('rmse', float('nan')):.2f}  "
          f"(seasonal-naive MAE="
          f"{r['baseline_metrics_holdout'].get('mae', float('nan')):.2f})")
    if bt:
        print(f"          backtest MAE={bt['forecast_mae']:.2f}  "
              f"RMSE={bt['forecast_rmse']:.2f}  "
              f"dispatch-weighted MAE={bt['dispatch_weighted_mae']:.2f}")

        print(f"\nCLOSED    {bt['steps']}h  net={bt['net_profit']:,.0f}  "
              f"revenue={bt['revenue']:,.0f}  charge cost={bt['charging_cost']:,.0f}  "
              f"degradation={bt['degradation_cost']:,.0f}")
        print(f"          throughput={bt['throughput_mwh']:,.0f} MWh  "
              f"cycles={bt['equivalent_cycles']:.2f}  "
              f"C/D/W={bt['charge_events']}/{bt['discharge_events']}/{bt['wait_events']}  "
              f"SOC avg/min/max={bt['avg_soc_pct']:.0f}/{bt['min_soc_pct']:.0f}/"
              f"{bt['max_soc_pct']:.0f}%  reserve breaches={bt['reserve_violations']}")
        print(f"\nBENCHMARK closed={bench.get('closed_loop_profit', 0):,.0f}  "
              f"open={bench.get('open_loop_profit', float('nan')):,.0f}  "
              f"perfect={bench.get('perfect_foresight_profit', 0):,.0f}  "
              f"regret={bench.get('regret', 0):,.0f} "
              f"({bench.get('regret_pct', float('nan')):.1f}%)  "
              f"capture={bench.get('capture_rate_pct', float('nan')):.1f}%")
        print(f"          value of closed-loop feedback = "
              f"{bench.get('closed_vs_open', float('nan')):,.0f}")
        print(f"\nSOLVER    {r['solver']['n_vars']} vars / "
              f"{r['solver']['n_constraints']} constraints  "
              f"avg={bt['avg_solve_seconds']*1000:.0f} ms  "
              f"max={bt['max_solve_seconds']*1000:.0f} ms  "
              f"calls={bt['solver_calls']}  backtest runtime="
              f"{bt['runtime_seconds']:.1f}s  pipeline={r['pipeline_seconds']:.1f}s")
        ei = r["error_impact"]
        if not ei.empty:
            print("\nFORECAST ERROR -> MONEY")
            print(ei.to_string(index=False))

    print("\nOPERATOR PLAN")
    print(r["operator_plan"]["narrative_llm"] if r["operator_plan"].get("narrative_llm")
          else r["operator_plan"]["narrative"])
    if r["warnings"]:
        print("\nWARNINGS")
        for w in r["warnings"]:
            print(f"  ! {w}")

    if a.all_scenarios:
        print("\nWHAT-IF ACROSS SCENARIOS")
        print(what_if(horizon=a.horizon, backtest_hours=min(a.hours, 48),
                      with_open_loop=False).to_string(index=False))


if __name__ == "__main__":
    main()
