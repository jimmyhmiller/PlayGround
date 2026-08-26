from .config import Limits
from .loader import load_settings
from .merge import combine
from .registry import resolve_hook, resolve_probe
from .scheduler import Scheduler
from .transform import apply_rules, rule_names
from .writer import Writer


class Engine:
    def __init__(self, settings_path: str) -> None:
        self.settings = load_settings(settings_path)
        self.limits = Limits()
        for name in splitlist(self.settings.get("startup_hooks", "")):
            self.limits = resolve_hook(name)(self.limits)
        self.scheduler = Scheduler(self.limits)
        self.writer = Writer()

    def probe_ok(self, record: dict[str, str]) -> bool:
        for name in splitlist(self.settings.get("probes", "")):
            if not resolve_probe(name)(record):
                return False
        return True

    def process(self, record: dict[str, str]) -> bool:
        if not self.probe_ok(record):
            return False
        rules = rule_names(self.settings.get("rules", "clean"))
        cooked = apply_rules(record, rules)
        self.writer.process(cooked)
        return True

    def process_pair(self, base: dict[str, str], extra: dict[str, str]) -> bool:
        rules = rule_names(self.settings.get("rules", "clean"))
        return self.process(combine(base, extra, rules))


def splitlist(spec: str) -> list[str]:
    return [part.strip() for part in spec.split(",") if part.strip()]
