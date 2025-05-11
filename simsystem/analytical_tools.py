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

def bdr_core(core: Core) -> tuple[float, int]:
    """
    Computes the Bounded Delay Resource (BDR) for a given core.
    The BDR is defined as the ratio of the budget to the speed factor.
    """
    
    # Assert that the core is valid
    if not isinstance(core, Core):
        raise ValueError("Invalid core object")
    
    alpha = core.speed_factor # Assuming the budget is equal to the speed factor for a core, due to a core being a root component
    delta = 0 # Assuming no delay for the core in an ideal scenario
    R = (alpha, delta)
    return R

def bdr_interface(component : Component, core: Core) -> tuple[float, int]:
    """
    Computes the Bounded Delay Resource (BDR) interface for a given component and core.
    Note that the Component and Core should share the same core_id.
    """
    
    # Assert that the component and core share the same core_id
    if component.core_id != core.core_id:
        raise ValueError("Component and Core must share the same core_id")
    
    Q = component.budget
    P = core.speed_factor
    alpha = Q / P
    delta = 2 * (P - Q)
    R = (alpha, delta)
    return R

def required_bdr(components : list[Component], core: Core) -> tuple[float, int]:
    """
    Computes the required Bounded Delay Resource (BDR) for a list of components and a core.
    Note that the Component and Core should share the same core_id.
    """
    
    # Assert that all components share the same core_id
    if not all(component.core_id == core.core_id for component in components):
        raise ValueError("All components must share the same core_id as the core")
    
    Q = sum(component.budget for component in components)
    P = core.speed_factor
    alpha = Q / P
    delta = 2 * (P - Q)
    R = (alpha, delta)
    return R

def sbf_bdr(R, t):
    """
    Supply Bound Function (SBF) for Bounded Delay Resource (BDR)

    Arguments:
        R: list of tasks
        t: time interval
    """
    # Assert that the time interval is valid
    if t <= 0:
        return 0
    
    alpha, delta = R
    
    if t >= delta:
        sbf = alpha * (t - delta)
    else:
        sbf = 0
    
    return sbf