def load_settings(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition(":")
            out[key.strip()] = val.strip()
    return out


def load_records(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not lines:
        return []
    headers = [h.strip() for h in lines[0].split(",")]
    records = []
    for line in lines[1:]:
        cells = line.split(",")
        records.append({headers[i]: cells[i] for i in range(len(headers))})
    return records
