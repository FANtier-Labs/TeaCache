from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    @abstractmethod
    def patch(self, pipeline, manager):
        """Return a context manager for one generation request."""
