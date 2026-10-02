from pydantic import BaseModel, model_validator


class SnapshotConf(BaseModel, frozen=True, extra="forbid", strict=True):
    enabled: bool
    id: str

    @model_validator(mode="after")
    def validate_config(self):
        if self.enabled and not self.id:
            raise ValueError("snapshot.id must be set when snapshot is enabled")
        return self


class GlobalConf(BaseModel, frozen=True, extra="forbid", strict=True):
    """
    Settings that apply to every run, whatever the scenario or runner.

    `scenario` is only a name. What it means, and which tunables it accepts, is decided by the
    scenario file and the scenario itself, not here.
    """

    snapshot: SnapshotConf
    scenario: str
    human_control: bool
