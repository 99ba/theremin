from __future__ import annotations


class ExpSmoother:
    def __init__(self, alpha: float, init_value=None) -> None:
        self.alpha = alpha
        self.value = init_value

    def update(self, x):
        if x is None:
            return self.value
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self, value=None) -> None:
        self.value = value
