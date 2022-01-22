from enum import Enum
from collections import namedtuple


class DataModes(Enum):
    TRAINING = 0
    TESTING = 1



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))