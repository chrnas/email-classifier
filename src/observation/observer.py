from abc import ABC, abstractmethod


class Observer(ABC):

    @abstractmethod
    def update(self, event_type, data):
        ...

    def display(self):
        ...
