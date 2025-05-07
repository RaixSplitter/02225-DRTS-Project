from functools import reduce


def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)

def hyperperiod(tasks: list[Task]) -> int:
    periods = [task.period for task in tasks]
    return reduce(lcm, periods)