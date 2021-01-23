__all__ = ["EarlyStopping"]


class EarlyStopping:

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        threshold : float = 0.05
    ):

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.threshold = threshold

    def __call__(self,score) -> bool:

        if self.best_score is None:
            self.best_score = score
            return False
        elif score <= self.best_score + self.min_delta:
            self.counter = 0
            self.best_score = score
            return False
        elif abs(score-self.best_score) <= self.threshold:
            return False
        else:
            if self.patience == self.counter:
                return True
            else:
                self.counter += 1
                return False


