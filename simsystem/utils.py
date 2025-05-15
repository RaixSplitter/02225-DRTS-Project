import math
from simsystem.objects import Task

def calculate_hyperperiod(tasks: list[Task]) -> int:
    """ Calculate the hyperperiod (LCM of all periods) of a set of tasks. """
    periods = [int(task.period) for task in tasks]
    hyperperiod = 1
    for period in periods:
        hyperperiod = math.lcm(hyperperiod, period)
    return hyperperiod
