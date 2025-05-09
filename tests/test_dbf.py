from simsystem.objects import Job, Task, Component, Core
from simsystem.analytical_tools import dbf_edf, dbf_rm
import pytest


def test_demand_bound_function_edf():
    # Create a list of tasks with different parameters
    tasks = [
        Task(name="Task1", wcet=2, period=7, deadline_interval=5, component_id="C1", priority=1),
        Task(name="Task2", wcet=3, period=10, deadline_interval=10, component_id="C2", priority=2),
        Task(name="Task3", wcet=1, period=6, deadline_interval=8, component_id="C3", priority=3),
    ]

    # # Test the demand bound function at different time intervals
    assert dbf_edf(tasks, 0) == 0
    assert dbf_edf(tasks, 5) == 2 + 0 + 0  # Task1 and Task2 are in the interval
    assert dbf_edf(tasks, 10) == 2 + 3 + 1  # All tasks are in the interval
    assert dbf_edf(tasks, 18) == 4 + 3 + 2   # Only Task2 and Task3 are in the interval
    
    
def test_demand_bound_function_rm():         
    tasks = [
        Task(name="Task1", wcet=1, period=5, deadline_interval=5, component_id="C1", priority=1),
        Task(name="Task2", wcet=2, period=8, deadline_interval=10, component_id="C2", priority=2),
        Task(name="Task3", wcet=3, period=20, deadline_interval=8, component_id="C3", priority=3),
    ]
    
    # Test the demand bound function at different time intervals
    assert dbf_rm(tasks, 0 , 0) == 0
    assert dbf_rm(tasks, 10, 2) == 3 + 4 + 2  # Task1 is in the interval
    assert dbf_rm(tasks, 7, 1) == 2 + 2  # Task1 is in the interval
    assert dbf_rm(tasks, 3, 0) == 1  # Task1 and Task2 are in the interval
    assert dbf_rm(tasks, 18, 2) == 3+6+4  # All tasks are in the interval
    