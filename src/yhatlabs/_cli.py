"""`yhat` command line: forecasts, backtests, anomaly scores and tabular predictions from CSV/parquet files.

yhat forecast sales.csv --h 28 -o forecast.csv
yhat evaluate sales.csv --h 28 --windows 3
yhat anomalies meters.parquet --threshold 3
yhat predict train.csv --target churn --test new.csv -o scored.csv
yhat models | yhat usage | yhat limits
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import pandas as pd


def _read(path: str) -> pd.DataFrame:
    if path == "-":
        return pd.read_csv(sys.stdin)
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_json(path) if path.endswith(".json") else pd.read_csv(path)


JSON_OUT = False


def _write(df: pd.DataFrame, path: Optional[str]) -> None:
    if (path is None or path == "-") and JSON_OUT:
        sys.stdout.write(df.to_json(orient="records", date_format="iso") + "\n")
    elif path is None or path == "-":
        df.to_csv(sys.stdout, index=False)
    elif path.endswith((".parquet", ".pq")):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _floats(s: Optional[str]) -> Optional[List[float]]:
    return None if s is None else [float(x) for x in s.split(",") if x]


def _names(s: Optional[str]) -> Optional[List[str]]:
    return None if s is None else [x for x in s.split(",") if x]


def _cols(p: argparse.ArgumentParser) -> None:
    p.add_argument("--id-col"), p.add_argument("--time-col"), p.add_argument("--target-col")
    p.add_argument("--freq", help="pandas frequency, e.g. h, D, 15min (inferred when omitted)")
    p.add_argument("--known", help="comma-separated known-covariate columns")
    p.add_argument("--past", help="comma-separated past-only covariate columns")
    p.add_argument("--static", help="comma-separated per-series constant columns")
    p.add_argument("--model")


def main(argv: Optional[List[str]] = None) -> int:
    import warnings

    warnings.formatwarning = lambda message, *_a, **_k: f"yhatlabs: warning: {message}\n"
    ap = argparse.ArgumentParser(
        prog="yhatlabs", description="YHat Labs prediction API from the command line (key: YHAT_API_KEY)."
    )
    ap.add_argument("--base-url")
    ap.add_argument(
        "--json", action="store_true", help="machine-readable output (JSON records), e.g. for agents and scripts"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast", help="forecast the next h steps of every series")
    (
        f.add_argument("data"),
        f.add_argument("--h", type=int, required=True),
        f.add_argument("--quantiles"),
        f.add_argument("--level"),
    )
    f.add_argument("--future", help="file with future values of the --known columns"), f.add_argument("-o", "--output")
    _cols(f)
    e = sub.add_parser("evaluate", help="rolling backtest against baselines")
    (
        e.add_argument("data"),
        e.add_argument("--h", type=int, required=True),
        e.add_argument("--windows", type=int, default=3),
    )
    e.add_argument("--step", type=int), e.add_argument("-o", "--output", help="per-window scores")
    _cols(e)
    a = sub.add_parser("anomalies", help="score the recent points of each series")
    a.add_argument("data"), a.add_argument("--h", type=int), a.add_argument("--windows", type=int, default=5)
    (
        a.add_argument("--threshold", type=float, default=3.0),
        a.add_argument("--only", action="store_true", help="print anomalies only"),
    )
    a.add_argument("-o", "--output")
    _cols(a)
    g = sub.add_parser("fill-gaps", help="fill missing values inside series")
    g.add_argument("data"), g.add_argument("--max-gap", type=int), g.add_argument("-o", "--output")
    _cols(g)
    p = sub.add_parser("predict", help="tabular: fit on train, predict test")
    p.add_argument("train"), p.add_argument("--target", required=True), p.add_argument("--test")
    (
        p.add_argument("--task", choices=["classify", "regress"]),
        p.add_argument("--preset", choices=["fast", "medium", "full"]),
    )
    p.add_argument("--quantiles"), p.add_argument("--model"), p.add_argument("-o", "--output")
    p.add_argument("--groups", help="column whose rows stay together in validation (machine, batch, patient)")
    p.add_argument("--time-col", dest="split_time", help="validate on contiguous periods of this column")
    p.add_argument("--set-coverage", type=float, help="classification: prediction sets with this coverage, e.g. 0.9")
    et = sub.add_parser("evaluate-table", help="tabular: cross-validate on your own table (accuracy and calibration)")
    et.add_argument("data"), et.add_argument("--target", required=True), et.add_argument("--cv", type=int, default=5)
    et.add_argument("--groups"), et.add_argument("--time-col", dest="split_time")
    (
        et.add_argument("--task", choices=["classify", "regress"]),
        et.add_argument("--preset", choices=["fast", "medium", "full"]),
    )
    et.add_argument("--model"), et.add_argument("-o", "--output", help="write every row's out-of-fold prediction here")
    for name in ("models", "usage", "limits", "fits", "whoami"):
        sub.add_parser(name)
    k = sub.add_parser("check", help="check a file locally before sending it (no key needed)")
    (
        k.add_argument("data"),
        k.add_argument("--h", type=int),
        k.add_argument("--target"),
        k.add_argument("--id-col"),
        k.add_argument("--time-col"),
    )
    sk = sub.add_parser("skills", help="install the yhatlabs skill for coding agents (Claude Code, Cursor, Codex, ...)")
    sk.add_argument("action", choices=["install", "show"])
    sk.add_argument(
        "--dir", default=".", help="project folder (default: current); writes .claude/skills and .agents/skills"
    )
    args = ap.parse_args(argv)

    from ._client import Client
    from ._errors import YHatError

    global JSON_OUT
    JSON_OUT = args.json
    if args.cmd == "skills":
        return _skills(args.action, args.dir)
    if args.cmd == "check":
        from ._check import check

        rep = check(_read(args.data), h=args.h, target=args.target, id_col=args.id_col, time_col=args.time_col)
        if args.json:
            print(
                json.dumps(
                    {"ok": rep.ok, "kind": rep.kind, "issues": rep.issues.to_dict("records")}, default=str, indent=2
                )
            )
        else:
            print(rep)
            if len(rep.issues):
                print(rep.issues.to_string(index=False))
        return 0 if rep.ok else 2

    try:
        with Client(base_url=args.base_url) as yh:
            if args.cmd in ("models", "usage", "limits", "fits", "whoami"):
                out = {
                    "models": yh.models,
                    "usage": yh.usage,
                    "limits": lambda: yh.server_limits(apply=False),
                    "fits": yh.fits,
                    "whoami": yh.whoami,
                }[args.cmd]()
                print(json.dumps(out, indent=2, default=str))
                return 0
            if args.cmd == "predict":
                res = yh.predict(
                    _read(args.train),
                    args.target,
                    _read(args.test) if args.test else None,
                    task=args.task,
                    preset=args.preset,
                    quantiles=_floats(args.quantiles),
                    groups=args.groups,
                    time_col=args.split_time,
                    set_coverage=args.set_coverage,
                    **({"model": args.model} if args.model else {}),
                )
                _write(res.df.reset_index(), args.output)
                return 0
            if args.cmd == "evaluate-table":
                ev = yh.evaluate_table(
                    _read(args.data),
                    args.target,
                    cv=args.cv,
                    groups=args.groups,
                    time_col=args.split_time,
                    task=args.task,
                    preset=args.preset,
                    **({"model": args.model} if args.model else {}),
                )
                if JSON_OUT:
                    tables = {
                        k: getattr(ev, k).to_dict(orient="records")
                        for k in ("reliability", "coverage", "pit")
                        if getattr(ev, k) is not None
                    }
                    print(
                        json.dumps(
                            {"task": ev.task, "split": ev.split, "summary": ev.summary, **tables}, indent=2, default=str
                        )
                    )
                else:
                    print(ev)
                    print(pd.Series(ev.summary).round(4).to_string())
                    for k in ("reliability", "coverage"):
                        if getattr(ev, k) is not None:
                            print(f"\n{k}:\n" + getattr(ev, k).round(3).to_string(index=False))
                if args.output:
                    _write(ev.predictions.reset_index(), args.output)
                return 0
            common = dict(
                id_col=args.id_col,
                time_col=args.time_col,
                target_col=args.target_col,
                freq=args.freq,
                known=_names(args.known),
                past=_names(args.past),
                static=_names(args.static),
                **({"model": args.model} if args.model else {}),
            )
            data = _read(args.data)
            if args.cmd == "forecast":
                fc = yh.forecast(
                    data,
                    args.h,
                    quantiles=_floats(args.quantiles),
                    level=_floats(args.level),
                    future=_read(args.future) if args.future else None,
                    **common,
                )
                _write(fc.df, args.output)
            elif args.cmd == "evaluate":
                common.pop("model", None)
                ev = yh.evaluate(
                    data,
                    args.h,
                    windows=args.windows,
                    step=args.step,
                    **({"models": (args.model,)} if args.model else {}),
                    **common,
                )
                print(ev.summary.round(4).to_string(), file=sys.stderr if args.output in (None, "-") else sys.stdout)
                if args.output:
                    _write(ev.scores, args.output)
            elif args.cmd == "anomalies":
                out = yh.detect_anomalies(data, h=args.h, windows=args.windows, threshold=args.threshold, **common)
                _write(out[out.anomaly] if args.only else out, args.output)
            elif args.cmd == "fill-gaps":
                common.pop("static", None)
                _write(yh.fill_gaps(data, max_gap=args.max_gap, **common), args.output)
    except YHatError as e:
        print(f"yhatlabs: {e}", file=sys.stderr)
        return 1
    return 0


def _skills(action: str, folder: str) -> int:
    from importlib.resources import files
    from pathlib import Path

    text = files("yhatlabs").joinpath("skill.md").read_text()
    if action == "show":
        print(text)
        return 0
    for sub_dir in (".claude/skills/yhatlabs", ".agents/skills/yhatlabs"):
        target = Path(folder) / sub_dir / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
