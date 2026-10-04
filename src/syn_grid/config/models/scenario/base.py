from pydantic import BaseModel, ConfigDict


# TODO: when everything is working, see if this one can be used instead of explicitly stating
# the keywords in each class. think this can be good for the tests, since they can override the
# frozen keyword.
class StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)