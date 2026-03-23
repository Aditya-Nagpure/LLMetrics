from abc import ABC, abstractmethod

from llmetrics.core.models import TraceRecord


class StorageBackend(ABC):
    @abstractmethod
    def write(self, record: TraceRecord) -> None: ...

    @abstractmethod
    def read_all(self) -> list[TraceRecord]: ...

    @abstractmethod
    def read_by_id(self, trace_id: str) -> TraceRecord | None: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "StorageBackend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()