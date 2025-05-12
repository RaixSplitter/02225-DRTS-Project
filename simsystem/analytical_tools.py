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

def bdr_interface(component : Component) -> tuple[float, int]:
    """
    Computes the Bounded Delay Resource (BDR) interface for a given component and core.
    Note that the Component and Core should share the same core_id.
    """
    
    Q = component.budget
    P = component.period
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
    
    R_i = [bdr_interface(component) for component in components]
    alpha = sum(r[0] for r in R_i)
    delta = min(r[1] for r in R_i)
    
    R = (alpha, delta)
    
    return R

def bdr_schedulability(components: list[Component], core : Core) -> bool:
    """
    Checks if the Bounded Delay Resource (BDR) is schedulable for a list of components and a core.
    Note that the Component and Core should share the same core_id.
    """
    
    logger.info(f"Checking BDR schedulability for core {core.core_id} and components {components}")
    
    # Assert that all components share the same core_id
    if not all(component.core_id == core.core_id for component in components):
        raise ValueError("All components must share the same core_id as the cores")
    
    # Calculate the required BDR
    required_R = required_bdr(components, core)
    
    logger.info(f"Required BDR: {required_R}")
    
    # Calculate the available BDR
    available_R = bdr_core(core)
    
    logger.info(f"Available BDR: {available_R}")
    
    alpha_required, delta_required = required_R
    alpha_available, delta_available = available_R
    
    # Check if the required BDR is less than or equal to the available BDR according to Theorem 3.1
    return alpha_required <= alpha_available and delta_required > delta_available

def bdr_schedulability_all(components: list[Component], cores: list[Core]) -> bool:
    """
    Checks if the Bounded Delay Resource (BDR) is schedulable for a list of components and cores.
    """
    
    # Group components by core
    core_to_components = {}
    for component in components:
        if component.core_id not in core_to_components:
            core_to_components[component.core_id] = []
        core_to_components[component.core_id].append(component)

    # Check schedulability for each core and its associated components
    for core in cores:
        if core.core_id not in core_to_components:
            continue
        if not bdr_schedulability(core_to_components[core.core_id], core):
            return False
        
    
    return True

def schedulability_test(tasks: dict[str, Task], components: dict[str, Component], cores: dict[str, Core]) -> bool:
    """
    Checks if the system is schedulable based on the Bounded Delay Resource (BDR) model.
    
    Arguments:
        tasks: list of tasks
        components: list of components
        cores: list of cores
    """
    
    # Check if the system is schedulable based on BDR
    if not bdr_schedulability_all(list(components.values()), list(cores.values())):
        return False
    
    # Check if task is schedulable by it's component
    for task in tasks:
        component = components[task.component_id]
        
        R, delta = bdr_interface(component)
        sbf_component = sbf_bdr(R, t=task.deadline_interval)
        
        dbf_task = dbf_edf(list(tasks.values()), t=task.deadline_interval)
        
        if dbf_task > sbf_component:
            logger.error(f"Task {task.name} is not schedulable by component {component.name}")
            return False
    return True


    
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