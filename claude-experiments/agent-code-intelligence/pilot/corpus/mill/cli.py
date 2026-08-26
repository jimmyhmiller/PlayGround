import sys

from .engine import Engine
from .loader import load_records
from .mathutil import checked_div


def main(argv: list[str]) -> int:
    settings_path, records_path = argv[0], argv[1]
    eng = Engine(settings_path)
    records = load_records(records_path)
    kept = 0
    for rec in records:
        if eng.process(rec):
            kept += 1
    pct = checked_div(kept * 100, max(1, len(records)))
    sys.stdout.write(f"kept {kept}/{len(records)} ({pct}%)\n")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
