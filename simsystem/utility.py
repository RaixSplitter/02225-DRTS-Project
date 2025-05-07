from functools import reduce
import math
from simsystem.objects import Task

def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

def hyperperiod(tasks: list[Task]) -> int:
    periods = [task.period for task in tasks]
    return reduce(lcm, periods)