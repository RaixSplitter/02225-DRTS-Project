import logging
import numpy as np # just for arange kinda dumb but who cares

from simsystem.objects import Core, Task, HierarchicalSystem, Job
from simsystem.schedulers import ComponentScheduler, SCHEDULERS
from simsystem.resources import BDRResourceSupplier, PRMResourceSupplier


logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Simulates the execution of the hierarchical system.
    """
    def __init__(self, system: HierarchicalSystem, scheduler_type: str = "EDF"):
        self.system = system
        self.jobs: list[Job] = []
        self.completed_jobs: list[Job] = []
        self.response_times: dict[str, dict] = {}  # Task -> response time statistics
        
        # Create system-level schedulers for each core
        self.system_schedulers: dict[str, SystemLevelScheduler] = {}
        for core_id, core in system.cores.items():
            self.system_schedulers[core_id] = SystemLevelScheduler(core, scheduler_type)


    def initialize_response_times(self) -> None:
        """Initialize the response time data structure."""
        for task_id in self.system.tasks:
            self.response_times[task_id] = {
                'values': [],
                'avg': 0,
                'max': 0
            }


    def release_jobs(self, time: float) -> None:
        """
        Release new jobs at the current simulation time.
        
        Args:
            time: Current simulation time.
        """
        for task_id, task in self.system.tasks.items():
            # Check if it's time to release a new job
            if time % task.period == 0:
                self.jobs.append(Job(task, time))


    def simulate(self, duration: float, time_slice: float = 1.0, verbose: bool = False) -> dict:
        """
        Run the simulation for a specified duration.
        
        Args:
            duration: Total simulation duration.
            time_slice: Time slice duration.
            verbose: Whether to logger.info simulation details.
            
        Returns:
            Dictionary of response time statistics.
        """
        logger.info(f"Starting simulation for duration {duration}...")
        self.initialize_response_times()

        for time in np.arange(0, duration, time_slice):
            if verbose and int(time) % 100 == 0:
                logger.info(f"Simulation time: {time:.2f} / {duration:.2f}")
            
            # Release new jobs
            self.release_jobs(time)
            
            # Simulate execution on each core using system-level schedulers
            for core_id, system_scheduler in self.system_schedulers.items():
                completed_jobs = system_scheduler.simulate(self.jobs, time, time_slice)
                
                # Record response times
                for job in completed_jobs:
                    task_id = job.task.name
                    response_time = job.get_response_time()
                    if task_id in self.response_times:
                        self.response_times[task_id]['values'].append(response_time)
                
                # Add completed jobs to the list
                self.completed_jobs.extend(completed_jobs)
        
        # Calculate response time statistics
        for task_id, data in self.response_times.items():
            if data['values']:
                data['avg'] = sum(data['values']) / len(data['values'])
                data['max'] = max(data['values'])
                
                if verbose:
                    logger.info(f"Task {task_id} response times: avg={data['avg']:.2f}, max={data['max']:.2f}")
        
        logger.info("Simulation completed.")
        return self.response_times



class SystemLevelScheduler:
    """
    System-level scheduler for managing components on a core.
    """
    def __init__(self, core: Core, scheduler_type: str = "EDF"):
        self.core = core
        
        # Create component schedulers
        self.component_schedulers = {}
        for component in core.components:
            # Use BDR resource supplier if BDR interface is set, otherwise use PRM
            if component.bdr_interface:
                supplier = BDRResourceSupplier()
            else:
                supplier = PRMResourceSupplier()

            self.component_schedulers[component.component_id] = ComponentScheduler(component, supplier)

        # Create the system-level scheduler based on configuration
        self.scheduler = SCHEDULERS.get(scheduler_type)
        if self.scheduler is None:
            return ValueError(f"Got invalid scheduler, expected on of {SCHEDULERS.keys()}, got: {scheduler_type}")


    def simulate(self, jobs: list[Job], time: float, time_slice: float) -> list[Job]:
        """
        Simulate execution on this core, scheduling components using the system-level scheduler.
        
        Args:
            jobs: List of all jobs in the system.
            time: Current simulation time.
            time_slice: Time slice duration.
            
        Returns:
            List of completed jobs.
        """
        # Create virtual jobs representing components
        component_jobs = []
        for component_id, component_scheduler in self.component_schedulers.items():
            component = component_scheduler.component
            
            # Create a virtual deadline based on component's period
            # This is a simplification - in a real system, this would be more complex
            virtual_deadline = time + component.period
            
            # Count jobs for this component to determine priority
            component_job_count = len([j for j in jobs if j.task.component_id == component_id])
            
            if component_job_count > 0:
                # Create a virtual job representing this component
                virtual_job = Job(
                    task = Task(
                        name = f"Component_{component_id}", 
                        wcet = component.budget,
                        period = component.period, 
                        component_id = "SYSTEM",
                        scheduler = "N/A"
                    ),
                    release_time = time,
                    deadline = virtual_deadline
                )
                component_jobs.append((virtual_job, component_id))
        
        # No components with jobs to execute
        if not component_jobs:
            return []
        
        # Select component to execute using system-level scheduler
        selected_component_job = self.scheduler.select_job([j for j, _ in component_jobs])
        completed_jobs = []
        
        if selected_component_job:
            # Find the component ID for the selected virtual job
            for virtual_job, component_id in component_jobs:
                if virtual_job == selected_component_job:
                    # Execute the selected component
                    component_scheduler = self.component_schedulers[component_id]
                    completed_jobs.extend(component_scheduler.simulate(jobs, time, time_slice))
                    break
        
        return completed_jobs


