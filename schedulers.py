from abc import ABC, abstractmethod
from objects import Job, Component
from resources import ResourceSupplier


class Scheduler(ABC):
    """
    Abstract base class for schedulers.
    """
    @abstractmethod
    def select_job(self, available_jobs: list[Job]) -> None | Job:
        """
        Select a job to execute based on the scheduling algorithm.
        
        Args:
            available_jobs: List of available jobs.
            
        Returns:
            Selected job or None if no jobs available.
        """
        pass
        
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the scheduler.
        
        Returns:
            Scheduler name.
        """
        pass



class EDFScheduler(Scheduler):
    """
    Earliest Deadline First scheduler.
    """
    def select_job(self, available_jobs: list[Job]) -> None | Job:
        """
        Select the job with the earliest deadline.
        
        Args:
            available_jobs: List of available jobs.
            
        Returns:
            Selected job or None if no jobs available.
        """
        if not available_jobs:
            return None
        return min(available_jobs, key=lambda job: job.deadline)


    def name(self) -> str:
        return "EDF"



class RMScheduler(Scheduler):
    """
    Rate Monotonic scheduler.
    """
    def select_job(self, available_jobs: list[Job]) -> None | Job:
        """
        Select the job with the highest RM priority.
        
        Args:
            available_jobs: List of available jobs.
            
        Returns:
            Selected job or None if no jobs available.
        """
        if not available_jobs:
            return None
            
        # If priorities are explicitly specified, use them
        if all(job.task.priority is not None for job in available_jobs):
            return max(available_jobs, key=lambda job: job.task.priority)
            
        # Otherwise, use RM priority (shorter period = higher priority)
        return min(available_jobs, key=lambda job: job.task.period)


    def name(self) -> str:
        return "RM"



class ComponentScheduler:
    """
    Scheduler for a component.
    """
    def __init__(self, component: Component, resource_supplier: ResourceSupplier):
        self.component = component
        self.resource_supplier = resource_supplier
        # Create the appropriate scheduler based on component configuration
        if component.scheduler == "EDF":
            self.scheduler = EDFScheduler()
        else:  # RM
            self.scheduler = RMScheduler()
    
    def simulate(self, jobs: list[Job], time: float, time_slice: float) -> list[Job]:
        """
        Simulate execution of jobs within this component.
        
        Args:
            jobs: List of all jobs in the system.
            time: Current simulation time.
            time_slice: Time slice duration.
            
        Returns:
            List of completed jobs.
        """
        # Get available jobs for this component
        available_jobs = [j for j in jobs if j.task.component_id == self.component.component_id]
        
        # No jobs to execute
        if not available_jobs:
            return []
            
        # Determine the amount of resource to supply
        resource_supply = self.resource_supplier.supply_resource(self.component, time, time_slice)
        
        # No resources available
        if resource_supply <= 0:
            return []
            
        # Select job based on scheduling algorithm
        selected_job = self.scheduler.select_job(available_jobs)
        completed_jobs = []
            
        # Execute the selected job
        if selected_job:
            execution_time = min(resource_supply, selected_job.remaining_time)
            selected_job.remaining_time -= execution_time
            
            # Check if job completed
            if selected_job.is_complete():
                selected_job.completion_time = time + execution_time
                completed_jobs.append(selected_job)
                jobs.remove(selected_job)
        
        return completed_jobs

SCHEDULERS = {
    "EDF" : EDFScheduler(),
    "RM"  : RMScheduler()
}
