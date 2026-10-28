"""Command line: ``physiocoach demo`` and ``physiocoach serve``."""

from __future__ import annotations

import argparse
import json
import random
import sys


def _demo(args) -> None:
    from .factory import Registry
    from .records import random_rep, random_session

    reg = Registry()
    rng = random.Random(args.seed)
    pipe = reg.pipeline(args.backend, args.lang == "ur", args.speak)
    for _ in range(args.n):
        rec = random_rep(rng, args.exercise, p_fault=0.5, cfg=reg.cfg)
        fb = pipe.rep(rec, args.lang, args.speak)
        print(f"[{rec.exercise} | faults: {', '.join(rec.present_faults(reg.cfg)) or 'none'}]")
        print(f"  {fb.text}" + ("   (template fallback)" if fb.fallback else ""))
        if fb.audio_path:
            print(f"  audio: {fb.audio_path}")
    session = random_session(rng, args.exercise, n_sets=2, cfg=reg.cfg)
    fb = pipe.session(session, args.lang, args.speak)
    print("\n[session summary]\n  " + fb.text)


def _serve(args) -> None:
    import uvicorn

    uvicorn.run("physiocoach.api:app", host=args.host, port=args.port, reload=args.reload)


def _feedback(args) -> None:
    from .factory import Registry
    from .schema import RepRecord, SessionRecord

    data = json.load(open(args.file, encoding="utf-8") if args.file != "-" else sys.stdin)
    reg = Registry()
    pipe = reg.pipeline(args.backend, args.lang == "ur", args.speak)
    fb = pipe.session(SessionRecord(**data), args.lang, args.speak) if "reps" in data else \
        pipe.rep(RepRecord(**data), args.lang, args.speak)
    print(json.dumps(fb.to_dict(), ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="physiocoach")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", default="template", choices=["template", "t5", "t5-guided", "llm"])
    common.add_argument("--lang", default="en", choices=["en", "ur"])
    common.add_argument("--speak", action="store_true")

    d = sub.add_parser("demo", parents=[common], help="print feedback for random records")
    d.add_argument("--exercise")
    d.add_argument("-n", type=int, default=5)
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(func=_demo)

    f = sub.add_parser("feedback", parents=[common], help="feedback for a JSON rep/session file ('-' = stdin)")
    f.add_argument("file")
    f.set_defaults(func=_feedback)

    s = sub.add_parser("serve", help="run the API and web demo")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=_serve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
