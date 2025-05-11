from dataclasses import dataclass

from typing import Callable

@dataclass
class Core:
    core_id : str
    speed_factor : float
    scheduler : Callable[[list["Job"]], "Job | None"]

@dataclass
class Component:
    component_id : str
    scheduler : Callable[[list["Job"]], "Job | None"]
    budget : int
    period : int
    core_id : str
    priority : int
    
@dataclass
class Job:
    name: str
    cost: int
    computation_progress: int
    release: int
    deadline: int
    finish_time : int
    component_id: str
    priority: int
    
    def check_deadline(self, current_time : int) -> bool:
        """
        Checks if the job has exceeded its deadline
        
        ARGUMENTS:
            current_time : int
            
        RETURNS:
            bool
        """

@dataclass
class Task:
    name: str
    wcet: int
    period: int
    deadline_interval: int
    component_id: str
    priority: int = None # Priority is not set by default
    
    last_released: int = 0
    
    
    @staticmethod
    def compute_cost(wcet : int) -> int:
        """
        Computes the cost for a Job given wcet
        
        ARGUMENTS:
            wcet : int
            
        RETURNS:
            cost : int
        """
        return NotImplementedError()
    
    def get_job(self, current_time : int) -> Job:
        """
        Returns a job object, given current time
        
        ARGUMENTS:
            current_time : int
            
        RETURNS:
            cost : Job
        """