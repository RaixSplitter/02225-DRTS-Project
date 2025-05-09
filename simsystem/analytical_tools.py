from simsystem.objects import Job, Task, Component, Core
import logging
import math

logger = logging.getLogger(__name__)


def dbf_edf(W: list[Task], t: int) -> int:
    """
    Demand Bound Function (EDF), the implementation is for an explicit deadline, 
    note that the implicit deadline is a special case of explicit deadline, 
    therefore the algorithm should be compatible for both.

    Arguments:
        W: list of tasks
        t: time interval
    """

    # Assert that the time interval is valid
    if t <= 0:
        return 0

    # Calculate the demand bound function for each task
    dbf = 0
    for task in W:
        # Extract task parameters
        C_i = task.wcet
        T_i = task.period
        D_i = task.deadline_interval

        # Find tasks with deadlines within time interval
        if task.deadline_interval <= t:
            dbf += ((t + T_i - D_i) // T_i) * C_i #Equation 3.3 in the handbook
    return dbf

def dbf_rm(W: list[Task], t: int, idx : int) -> int:
    """
    Demand Bound Function (RM)

    Arguments:
        W: list of tasks
        t: time interval
        idx: index of the task in the list W
    """

    # Assert that the time interval is valid
    if t <= 0:
        return 0

    # Calculate the demand bound function for each task
    prio_threshold = W[idx].priority
    HP_tasks = [task for task in W if task.priority < prio_threshold] # Gather Strictly High Priority tasks
    
    dbf = W[idx].wcet # The task itself is always in the interval
    for task in HP_tasks:
        # Extract task parameters
        C_k = task.wcet
        T_k = task.period

        # Find tasks with deadlines within time interval
        if task.deadline_interval <= t:
            dbf += math.ceil(t / T_k) * C_k #Equation 3.3 in the handbook
    return dbf