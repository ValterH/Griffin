from enum import Enum


class StrictType(str, Enum):
    random = 'random'
    random_no_positive = 'random_no_positive'
    full = 'full'
