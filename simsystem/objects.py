from dataclasses import dataclass
from simsystem.schedulers import RM, EDF 

@dataclass
class Core:
    core_id : str
    speed_factor : float
    scheduler : RM | EDF

@dataclass
class Component:
    component_id : str
    scheduler : RM | EDF
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
    last_released: int
    deadline_interval: int
    component_id: str
    priority: int
    
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