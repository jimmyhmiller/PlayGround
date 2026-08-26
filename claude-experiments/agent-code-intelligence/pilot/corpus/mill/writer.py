class Writer:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def process(self, row: dict[str, str]) -> None:
        self.rows.append(row)

    def process_raw(self, raw: str) -> None:
        if not raw.strip():
            return
        pairs = [p.partition("=") for p in raw.split(";")]
        self.process({k: v for k, _, v in pairs})

    def flush_lines(self) -> list[str]:
        return [";".join(f"{k}={v}" for k, v in sorted(r.items()))
                for r in self.rows]
