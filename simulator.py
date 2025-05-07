from numpy import arange # stupid dependency but who cares

from objects import Core, Component, Task, HierarchicalSystem

class Job:
    """
    Represents a job instance in the simulation.
    """
    def __init__(self, task: Task, release_time: float) -> None:
        self.task = task
        self.release_time = release_time
        self.deadline = release_time + task.deadline
        self.remaining_time = task.wcet
        self.completion_time = None
        
    def is_complete(self) -> bool:
        """Check if the job is complete."""
        return self.remaining_time <= 0
        
    def get_response_time(self) -> None | float:
        """Calculate the response time of the job."""
        if self.completion_time is None:
            return None
        return self.completion_time - self.release_time
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.deadline < other.deadline



class SimulationEngine:
    """
    Simulates the execution of the hierarchical system.
    """
    def __init__(self, system: HierarchicalSystem):
        self.system = system
        self.jobs: list[Job] = []
        self.completed_jobs: list[Job] = []
        self.response_times: dict = {}  # Task -> list of response times


    def initialize_response_times(self) -> None:
        """Initialize the response time data structure."""
        for task_id in self.system.tasks:
            self.response_times[task_id] = {
                'values': [],
                'avg': 0,
                'max': 0
            }


    def release_jobs(self, time: float) -> None:
        """ Release new jobs at the current simulation time. """
        for task_id, task in self.system.tasks.items():
            # Check if it's time to release a new job
            if time % task.period == 0:
                self.jobs.append(Job(task, time))


    def select_job_edf(self, available_jobs: list[Job]) -> None | Job:
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


    def select_job_rm(self, available_jobs: list[Job]) -> Job | None:
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


    def supply_resource_prm(self, component: Component, time: float, time_slice: float) -> float:
        """
        Simulate resource supply for a component based on its PRM parameters.
        
        Args:
            component: Component to supply resources to.
            time: Current simulation time.
            time_slice: Time slice duration.
            
        Returns:
            Amount of resource supplied.
        """
        budget = component.budget
        period = component.period
        
        # Calculate time within the current period
        time_in_period = time % period
        
        # Simplified supply model: resources are available at the beginning of each period
        if time_in_period < budget:
            return min(budget - time_in_period, time_slice)
        return 0


    def supply_resource_bdr(self, component: Component, time: float, time_slice: float) -> float:
        """
        Simulate resource supply for a component based on its BDR parameters.
        
        Args:
            component: Component to supply resources to.
            time: Current simulation time.
            time_slice: Time slice duration.
            
        Returns:
            Amount of resource supplied.
        """
        if component.bdr_interface is None:
            return self.supply_resource_prm(component, time, time_slice)
            
        alpha, delta = component.bdr_interface
        
        # Simplified BDR supply model
        if time > delta:
            return alpha * time_slice
        return 0
    
    def simulate_component(self, component: Component, time: float, time_slice: float) -> None:
        """
        Simulate execution of jobs within a component.
        
        Args:
            component: Component to simulate.
            time: Current simulation time.
            time_slice: Time slice duration.
        """
        # Get available jobs for this component
        available_jobs = [j for j in self.jobs if j.task.component_id == component.component_id]
        
        # No jobs to execute
        if not available_jobs:
            return
            
        # Determine the amount of resource to supply
        resource_supply = self.supply_resource_prm(component, time, time_slice)
        
        # No resources available
        if resource_supply <= 0:
            return
            
        # Select job based on scheduling algorithm
        selected_job = None
        if component.scheduler == "EDF":
            selected_job = self.select_job_edf(available_jobs)
        else:  # RM
            selected_job = self.select_job_rm(available_jobs)
            
        # Execute the selected job
        if selected_job:
            execution_time = min(resource_supply, selected_job.remaining_time)
            selected_job.remaining_time -= execution_time
            
            # Check if job completed
            if selected_job.is_complete():
                selected_job.completion_time = time + execution_time
                self.completed_jobs.append(selected_job)
                self.jobs.remove(selected_job)
                
                # Record response time
                task_id = selected_job.task.name
                response_time = selected_job.get_response_time()
                self.response_times[task_id]['values'].append(response_time)


    def simulate_core(self, core: Core, time: float, time_slice: float) -> None:
        """
        Simulate execution on a core.
        
        Args:
            core: Core to simulate.
            time: Current simulation time.
            time_slice: Time slice duration.
        """
        # For now, we'll just execute each component in sequence
        # TODO: implement a system-level scheduler
        for component in core.components:
            self.simulate_component(component, time, time_slice)


    def simulate(self, duration: float, time_slice: float = 1.0, verbose: bool = False) -> dict:
        """
        Run the simulation for a specified duration.
        
        Args:
            duration: Total simulation duration.
            time_slice: Time slice duration.
            verbose: Whether to print simulation details.
            
        Returns:
            Dictionary of response time statistics.
        """
        print(f"Starting simulation for duration {duration}...")
        self.initialize_response_times()
        
        for time in arange(0, duration, time_slice):
            
            if verbose and int(time) % 100 == 0:
                print(f"Simulation time: {time:.2f} / {duration:.2f}")
            
            self.release_jobs(time)
            
            for core_id, core in self.system.cores.items():
                self.simulate_core(core, time, time_slice)
        
        # Calculate response time statistics
        for task_id, data in self.response_times.items():
            if data['values']:
                data['avg'] = sum(data['values']) / len(data['values'])
                data['max'] = max(data['values'])
                
                if verbose:
                    print(f"Task {task_id} response times: avg={data['avg']:.2f}, max={data['max']:.2f}")
        
        print("Simulation completed.")
        return self.response_times
