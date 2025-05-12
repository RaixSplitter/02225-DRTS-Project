from simsystem.objects import Job, Task, Component, Core
import logging
import math

logger = logging.getLogger(__name__)

#region DBF
def dbf_edf(W: list[Task], t: int, speed_factor) -> int:
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
        C_i = task.wcet / speed_factor
        T_i = task.period
        D_i = task.deadline

        # Find tasks with deadlines within time interval
        if task.deadline <= t:
            dbf += ((t + T_i - D_i) // T_i) * C_i #Equation 3.3 in the handbook
    return dbf

def dbf_rm(W: list[Task], t: int, idx : int, speed_factor) -> int:
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
    
    dbf = W[idx].wcet / speed_factor # The task itself is always in the interval
    for task in HP_tasks:
        # Extract task parameters
        C_k = task.wcet / speed_factor
        T_k = task.period

        # Find tasks with deadlines within time interval
        if task.deadline <= t:
            dbf += math.ceil(t / T_k) * C_k #Equation 3.3 in the handbook
    return dbf
#endregion

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

def bdr_core(core: Core) -> tuple[float, int]:
    """
    Computes the Bounded Delay Resource (BDR) for a given core.
    The BDR is defined as the ratio of the budget to the speed factor.
    """
    
    # Assert that the core is valid
    if not isinstance(core, Core):
        raise ValueError("Invalid core object")
    
    R = (1, -1e-9) # Default values for alpha and delta
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
    
    logger.debug(f"Checking BDR schedulability for core {core.core_id} and components {components}")
    
    # Assert that all components share the same core_id
    if not all(component.core_id == core.core_id for component in components):
        raise ValueError("All components must share the same core_id as the cores")
    
    # Calculate the required BDR
    required_R = required_bdr(components, core)
    
    logger.debug(f"Required BDR: {required_R}")
    
    # Calculate the available BDR
    available_R = bdr_core(core)
    
    logger.debug(f"Available BDR: {available_R}")
    
    alpha_required, delta_required = required_R
    alpha_available, delta_available = available_R
    
    # Check if the required BDR is less than or equal to the available BDR according to Theorem 3.1
    return alpha_required <= alpha_available and delta_required > delta_available

def find_critical_time_points_rm(tasks: list[Task], task_index: int) -> list[float]:
    """
    Find critical time points for RM schedulability analysis.

    Args:
        tasks: List of tasks sorted by RM priority.
        task_index: Index of the task to analyze.

    Returns:
        List of critical time points.
    """
    # Sort tasks by priority (lower value indicates higher priority)
    tasks.sort(key=lambda t: t.priority)
    task = tasks[task_index]
    higher_priority_tasks = tasks[:task_index]
    
    # Initial set of points: all task periods and their multiples up to task's deadline
    points = set()
    for hp_task in higher_priority_tasks:
        k = 1
        while k * hp_task.period <= task.deadline:
            points.add(k * hp_task.period)
            k += 1

    # Add the task's own deadline
    points.add(task.deadline)

    return sorted(list(points))
    
def find_critical_time_points_edf(tasks: list[Task], hyperperiod: float) -> list[float]:
    """
    Find critical time points for EDF schedulability analysis.
    
    Args:
        tasks: List of tasks.
        hyperperiod: Hyperperiod of all tasks.
        
    Returns:
        List of critical time points.
    """
    # Critical points are at all job deadlines within the hyperperiod
    points = set()
    
    for task in tasks:
        k = 1
        while k * task.period <= hyperperiod:
            points.add(k * task.period)  # Using implicit deadlines
            k += 1
    
    return sorted(list(points))

def schedulability_test(tasks: dict[str, Task], components: dict[str, Component], cores: dict[str, Core]) -> bool:
    """
    Checks if the system is schedulable based on the Bounded Delay Resource (BDR) model.
    
    Arguments:
        tasks: list of tasks
        components: list of components
        cores: list of cores
    """
    
    schedulability = True
    core_schedulability = {}
    component_schedulability = {}
    task_schedulability = {}
    
    # # Check if the system is schedulable based on BDR
    # if not bdr_schedulability_all(list(components.values()), list(cores.values())):
    #     return False
    
    #region 1: Assign tasks to components and components to cores
    # Assign tasks to components
    for task in tasks.values():
        components[task.component_id].add_task(task)

    # Assign components to cores
    for component in components.values():
        cores[component.core_id].add_component(component)
    #endregion
    
    for core in cores.values():
        logger.debug("Evaluating", core.core_id) 
        for component in core.components:
            logger.debug("Evaluating", component.component_id)
            
            R = bdr_interface(component)
            
            alpha, delta = R
            
            for task_idx, task in enumerate(component.tasks):
                
                task_schedulable = False
                
                if component.scheduler == "RM":
                    ts = find_critical_time_points_rm(component.tasks, task_idx)
                    
                    for t in ts:
                        sbf_component = sbf_bdr(R, t)
                        dbf_task = dbf_rm(component.tasks, t, task_idx, core.speed_factor)
                        logger.debug(sbf_component, dbf_task)
                        if sbf_component >= dbf_task:
                            logger.info(f"Task {task.name} is schedulable at time {t} with SBF: {sbf_component} >= DBF: {dbf_task}")
                            task_schedulable = True
                            task_schedulability[task.name] = True
                            break
                    else:
                        logger.info(f"Task {task.name} is not schedulable at time {t} with SBF: {sbf_component} < DBF: {dbf_task}")
                        task_schedulability[task.name] = False
                    
                elif component.scheduler == "EDF":
                    ts = find_critical_time_points_edf(component.tasks, component.hyperperiod)
                    
                    for t in ts:
                        sbf_component = sbf_bdr(R, t)
                        dbf_task = dbf_rm(component.tasks, t, task_idx, core.speed_factor)
                        if sbf_component < dbf_task:
                            logger.info(f"Task {task.name} is not schedulable at time {t} with SBF: {sbf_component} < DBF: {dbf_task}")
                            task_schedulability[task.name] = False
                            break
                    else:
                        logger.info(f"Task {task.name} is schedulable at time {t} with SBF: {sbf_component} >= DBF: {dbf_task}")
                        task_schedulable = True
                        task_schedulability[task.name] = True
                        break
                        
                else:
                    raise ValueError(f"Unknown scheduler: {component.scheduler}")
                
            # If any task in the component is not schedulable, the component is not schedulable
            component_schedulability[component.component_id] = all(task_schedulability.values())
            logger.info(f"Component {component.component_id} is schedulable: {component_schedulability[component.component_id]}")
        
        # If any component in the core is not schedulable, the core is not schedulable and it has to pass the BDR check
        #BDR Check
        bdr_check = bdr_schedulability(core.components, core)
        core_schedulability[core.core_id] = all(component_schedulability.values()) if bdr_check else False
        logger.info(f"Core {core.core_id} is schedulable: {core_schedulability[core.core_id]}, BDR: {bdr_check}")
        
    return all(core_schedulability.values()), core_schedulability, component_schedulability, task_schedulability
        
                
                
            
    


    
