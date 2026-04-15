from enum import Enum


class LearningSchedule(Enum):
    """
    Learning rate scheduling strategies for linear models.

    Attributes
    ----------
    CONSTANT : LearningSchedule
        Keep the learning rate fixed across epochs.
    TIME_DECAY : LearningSchedule
        Decrease the learning rate proportionally with epoch growth.
    EXPONENTIAL : LearningSchedule
        Decrease the learning rate exponentially over epochs.
    """

    CONSTANT = "constant"
    TIME_DECAY = "time_decay"
    EXPONENTIAL = "exponential"
