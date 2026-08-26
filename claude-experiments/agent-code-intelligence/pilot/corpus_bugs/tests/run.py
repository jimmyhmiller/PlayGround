import sys
import traceback

from tests import test_engine, test_merge, test_textmath

MODULES = [test_engine, test_merge, test_textmath]


def main() -> int:
    passed, failed = 0, []
    for mod in MODULES:
        for name in sorted(dir(mod)):
            if name.startswith("test_"):
                try:
                    getattr(mod, name)()
                    passed += 1
                except Exception:
                    failed.append(f"{mod.__name__}.{name}")
                    traceback.print_exc()
    print(f"{passed} passed, {len(failed)} failed")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
