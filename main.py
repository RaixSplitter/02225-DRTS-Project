from dataclasses import dataclass

import pandas as pd
import numpy as np
import math
import argparse

from typing import Dict, List, Tuple, Set, Optional

class Task:
    """
    Represents a periodic task in a real-time system.
    """
    def __init__(self, name: str, wcet: float, period: float, component_id: str, scheduler: str, priority: float = None):
        self.name = name
        self.wcet = wcet
        self.period = period
        self.deadline = period  # NOTE: Implicit deadline model, could change idk
        self.component_id = component_id
        self.scheduler = scheduler
        self.priority = priority
        
    def __str__(self) -> str:
        return f"Task(name={self.name}, wcet={self.wcet}, period={self.period}, component={self.component_id}, scheduler={self.scheduler})"
    
    def __repr__(self) -> str:
        return self.__str__()

class Component:
    """
    Represents a component in a hierarchical scheduling system.
    """
    def __init__(self, component_id: str, scheduler: str, budget: float, period: float, core_id: str):
        self.component_id = component_id
        self.scheduler = scheduler
        self.budget = budget
        self.period = period
        self.core_id = core_id
        self.tasks: List[Task] = []
        self.bdr_interface: Optional[Tuple[float, float]] = None  # (alpha, delta)
        
    def add_task(self, task: Task) -> None:
        """Add a task to this component."""
        self.tasks.append(task)
        
    def set_bdr_interface(self, alpha: float, delta: float) -> None:
        """Set the BDR interface parameters."""
        self.bdr_interface = (alpha, delta)
        
    def __str__(self) -> str:
        return f"Component(id={self.component_id}, scheduler={self.scheduler}, budget={self.budget}, period={self.period}, core={self.core_id}, tasks={len(self.tasks)})"
    
    def __repr__(self) -> str:
        return self.__str__()

class Core:
    """
    Represents a processing core in the system.
    """
    def __init__(self, core_id: str, speed_factor: float):
        self.core_id = core_id
        self.speed_factor = speed_factor
        self.components: list[Component] = []
        
    def add_component(self, component: Component) -> None:
        """Add a component to this core."""
        self.components.append(component)
        
    def __str__(self) -> str:
        return f"Core(id={self.core_id}, speed_factor={self.speed_factor}, components={len(self.components)})"
    
    def __repr__(self) -> str:
        return self.__str__()


class HierarchicalSystem:
    """
    Represents the entire hierarchical scheduling system.
    """
    def __init__(self):
        self.cores: Dict[str, Core] = {}
        self.components: Dict[str, Component] = {}
        self.tasks: Dict[str, Task] = {}
        
    def build_from_parsed_data(self, cores: Dict, components: Dict, tasks: Dict) -> None:
        """
        Build the hierarchical system from parsed data.
        """
        # Create core objects
        for core_id, core_data in cores.items():
            self.cores[core_id] = Core(core_id, core_data['speed_factor'])
        
        # Create component objects
        for comp_id, comp_data in components.items():
            component = Component(
                comp_id, 
                comp_data['scheduler'], 
                comp_data['budget'], 
                comp_data['period'], 
                comp_data['core_id']
            )
            self.components[comp_id] = component
            
            if comp_data['core_id'] in self.cores:
                self.cores[comp_data['core_id']].add_component(component)
        
        # Create task objects and assign to components
        for task_id, task_data in tasks.items():
            task = Task(
                task_data['name'],
                task_data['wcet'],
                task_data['period'],
                task_data['component_id'],
                # task_data['scheduler'],
                task_data['priority'] if 'priority' in task_data and task_data['priority'] else None
            )
            self.tasks[task_id] = task
            
            if task_data['component_id'] in self.components:
                self.components[task_data['component_id']].add_task(task)

    def adjust_wcet_for_core_speed(self) -> None:
        """
        Adjust WCET values based on core speed factors.
        """
        for task_name, task in self.tasks.items():
            component = self.components.get(task.component_id)
            if component:
                core = self.cores.get(component.core_id)
                if core:
                    # Adjust WCET: slower core (speed < 1) increases WCET
                    task.wcet = task.wcet / core.speed_factor
    
    def __str__(self) -> str:
        return f"HierarchicalSystem(cores={len(self.cores)}, components={len(self.components)}, tasks={len(self.tasks)})"
    
    def __repr__(self) -> str:
        return self.__str__()

class InputParser:
    """
    Parses input CSV files into data structures.
    """
    def __init__(self, tasks_file: str, architecture_file: str, budgets_file: str):
        self.tasks_file = tasks_file
        self.architecture_file = architecture_file
        self.budgets_file = budgets_file
        
    def parse_inputs(self) -> Tuple[Dict, Dict, Dict]:
        """
        Parse input CSV files into dictionaries.
        """
        print(f"Parsing input files...")
        
        # Read CSV files
        tasks_df = pd.read_csv(self.tasks_file)
        arch_df = pd.read_csv(self.architecture_file)
        budgets_df = pd.read_csv(self.budgets_file)
        
        # Parse cores
        cores = {}
        for _, row in arch_df.iterrows():
            cores[row['core_id']] = {
                'speed_factor': row['speed_factor']
            }
        
        # Parse components and their initial budgets
        components = {}
        for _, row in budgets_df.iterrows():
            components[row['component_id']] = {
                'scheduler': row['scheduler'],
                'budget': row['budget'],
                'period': row['period'],
                'core_id': row['core_id'],
                'tasks': []
            }
        
        # Parse tasks and assign to components
        tasks = {}
        for _, row in tasks_df.iterrows():
            task = {
                'name': row['task_name'],
                'wcet': row['wcet'],
                'period': row['period'],
                'component_id': row['component_id'],
                # 'scheduler': row['scheduler'],
                'priority': row['priority'] if 'priority' in row and not pd.isna(row['priority']) else None
            }
            tasks[row['task_name']] = task
            
        print(f"Parsed {len(cores)} cores, {len(components)} components, and {len(tasks)} tasks.")
        return cores, components, tasks


class AnalysisEngine:
    """
    Performs schedulability analysis on the hierarchical system.
    """
    def __init__(self, system: HierarchicalSystem):
        self.system = system
        
    def sbf_bdr(self, alpha: float, delta: float, t: float) -> float:
        """
        Supply Bound Function for BDR (Equation 6).
        
        Args:
            alpha: Resource availability factor.
            delta: Maximum delay in resource allocation.
            t: Time interval.
            
        Returns:
            Minimum resource supply in time interval t.
        """
        if t >= delta:
            return alpha * (t - delta)
        else:
            return 0
    
    def dbf_rm(self, tasks: List[Task], t: float, task_index: int) -> float:
        """
        Demand Bound Function for RM (Equation 4).
        
        Args:
            tasks: List of tasks sorted by RM priority (highest first).
            t: Time interval.
            task_index: Index of the task to analyze.
            
        Returns:
            Maximum resource demand in time interval t.
        """
        task = tasks[task_index]
        demand = task.wcet
        
        # Add demand from higher priority tasks
        for i in range(task_index):
            higher_priority_task = tasks[i]
            demand += math.ceil(t / higher_priority_task.period) * higher_priority_task.wcet
                
        return demand
    
    def dbf_edf(self, tasks: List[Task], t: float) -> float:
        """
        Demand Bound Function for EDF with implicit deadlines (Equation 2).
        
        Args:
            tasks: List of tasks.
            t: Time interval.
            
        Returns:
            Maximum resource demand in time interval t.
        """
        demand = 0
        for task in tasks:
            demand += math.floor(t / task.period) * task.wcet
        return demand
    
    def dbf_edf_explicit(self, tasks: List[Task], t: float) -> float:
        """
        Demand Bound Function for EDF with explicit deadlines (Equation 3).
        
        Args:
            tasks: List of tasks with explicit deadlines.
            t: Time interval.
            
        Returns:
            Maximum resource demand in time interval t.
        """
        demand = 0
        for task in tasks:
            demand += math.floor((t + task.period - task.deadline) / task.period) * task.wcet
        return demand
    
    def convert_prm_to_bdr(self, budget: float, period: float) -> Tuple[float, float]:
        """
        Half-Half Algorithm: Converting PRM (Q, P) to BDR (alpha, delta) (Theorem 3).
        
        Args:
            budget: Resource budget (Q).
            period: Resource period (P).
            
        Returns:
            Tuple of (alpha, delta).
        """
        alpha = budget / period  # Resource utilization
        delta = 2 * (period - budget)  # Maximum delay
        return alpha, delta
    
    def compute_rm_priority_order(self, tasks: List[Task]) -> List[Task]:
        """
        Compute RM priority order (shorter period = higher priority).
        
        Args:
            tasks: List of tasks.
            
        Returns:
            List of tasks sorted by RM priority (highest first).
        """
        # If priorities are specified, use them
        if all(task.priority is not None for task in tasks):
            return sorted(tasks, key=lambda t: t.priority, reverse=True)
        # Otherwise, sort by period (RM policy)
        return sorted(tasks, key=lambda t: t.period)
    
    def find_critical_time_points_rm(self, tasks: List[Task], task_index: int) -> List[float]:
        """
        Find critical time points for RM schedulability analysis.
        
        Args:
            tasks: List of tasks sorted by RM priority.
            task_index: Index of the task to analyze.
            
        Returns:
            List of critical time points.
        """
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
    
    def find_critical_time_points_edf(self, tasks: List[Task], hyperperiod: float) -> List[float]:
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
    
    def check_schedulability_rm(self, component: Component, alpha: float, delta: float, 
                               verbose: bool = False) -> bool:
        """
        Check schedulability of component tasks under RM using BDR.
        
        Args:
            component: Component to analyze.
            alpha: Resource availability factor.
            delta: Maximum delay in resource allocation.
            verbose: Whether to print detailed analysis.
            
        Returns:
            Whether the component is schedulable.
        """
        tasks = self.compute_rm_priority_order(component.tasks)
        
        if verbose:
            print(f"\nAnalyzing component {component.component_id} under RM:")
            print(f"  BDR parameters: alpha={alpha:.4f}, delta={delta:.4f}")
        
        for i, task in enumerate(tasks):
            # Find critical time points for this task
            critical_points = self.find_critical_time_points_rm(tasks, i)
            
            # Check if there exists a time t where demand <= supply
            schedulable = False
            
            for t in critical_points:
                demand = self.dbf_rm(tasks, t, i)
                supply = self.sbf_bdr(alpha, delta, t)
                
                if verbose:
                    print(f"  Task {task.name} at t={t:.2f}: demand={demand:.2f}, supply={supply:.2f}")
                
                if demand <= supply:
                    schedulable = True
                    break
            
            if not schedulable:
                if verbose:
                    print(f"  Task {task.name} is not schedulable!")
                return False
            
            if verbose:
                print(f"  Task {task.name} is schedulable.")
        
        return True
    
    def check_schedulability_edf(self, component: Component, alpha: float, delta: float, 
                                verbose: bool = False) -> bool:
        """
        Check schedulability of component tasks under EDF using BDR.
        
        Args:
            component: Component to analyze.
            alpha: Resource availability factor.
            delta: Maximum delay in resource allocation.
            verbose: Whether to print detailed analysis.
            
        Returns:
            Whether the component is schedulable.
        """
        tasks = component.tasks
        
        # Calculate hyperperiod (LCM of periods)
        periods = [task.period for task in tasks]
        hyperperiod = 1
        for period in periods:
            hyperperiod = math.lcm(int(hyperperiod), int(period))
        
        if verbose:
            print(f"\nAnalyzing component {component.component_id} under EDF:")
            print(f"  BDR parameters: alpha={alpha:.4f}, delta={delta:.4f}")
            print(f"  Hyperperiod: {hyperperiod}")
        
        # Find critical time points
        critical_points = self.find_critical_time_points_edf(tasks, hyperperiod)
        
        # Check at critical time points
        for t in critical_points:
            demand = self.dbf_edf(tasks, t)
            supply = self.sbf_bdr(alpha, delta, t)
            
            if verbose:
                print(f"  At t={t:.2f}: demand={demand:.2f}, supply={supply:.2f}")
            
            if demand > supply:
                if verbose:
                    print("  Not schedulable!")
                return False
        
        return True
    
    def analyze_component(self, component: Component, verbose: bool = False) -> bool:
        """
        Analyze a component and determine its schedulability.
        
        Args:
            component: Component to analyze.
            verbose: Whether to print detailed analysis.
            
        Returns:
            Whether the component is schedulable.
        """
        # Convert initial PRM parameters to BDR
        alpha, delta = self.convert_prm_to_bdr(component.budget, component.period)
        
        # Check schedulability based on scheduler type
        if component.scheduler == "RM":
            schedulable = self.check_schedulability_rm(component, alpha, delta, verbose)
        else:  # EDF
            schedulable = self.check_schedulability_edf(component, alpha, delta, verbose)
            
        # Store BDR interface
        component.set_bdr_interface(alpha, delta)
        
        return schedulable
    
    def optimize_component_bdr(self, component: Component, 
                              alpha_increment: float = 0.01, 
                              verbose: bool = False) -> Tuple[float, float]:
        """
        Optional: Optimize BDR parameters for a component.
        
        Args:
            component: Component to optimize.
            alpha_increment: Step size for alpha optimization.
            verbose: Whether to print optimization details.
            
        Returns:
            Optimized (alpha, delta) pair.
        """
        # Start with a reasonable initial guess
        initial_alpha, initial_delta = self.convert_prm_to_bdr(component.budget, component.period)
        
        # Cost function weights
        c1 = 0.8  # Weight for alpha (processor utilization)
        c2 = 0.2  # Weight for switch_cost (inversely related to delta)
        
        best_alpha = initial_alpha
        best_delta = initial_delta
        best_cost = c1 * best_alpha + c2 * (1.0 / best_delta)
        
        # Simple optimization: incrementally increase alpha until schedulable
        alpha = initial_alpha
        while alpha <= 1.0:
            # Compute corresponding delta with the Half-Half algorithm
            delta = 2.0 * (component.period * alpha - component.budget)
            
            # Check schedulability
            if component.scheduler == "RM":
                schedulable = self.check_schedulability_rm(component, alpha, delta)
            else:  # EDF
                schedulable = self.check_schedulability_edf(component, alpha, delta)
            
            if schedulable:
                # Compute cost
                cost = c1 * alpha + c2 * (1.0 / delta)
                
                # Update if better
                if cost < best_cost:
                    best_alpha = alpha
                    best_delta = delta
                    best_cost = cost
            
            alpha += alpha_increment
        
        if verbose:
            print(f"Optimized BDR for {component.component_id}: " 
                  f"alpha={best_alpha:.4f}, delta={best_delta:.4f}")
        
        return best_alpha, best_delta
    
    def analyze_system(self, verbose: bool = False, optimize: bool = False) -> Dict:
        """
        Analyze the entire hierarchical system.
        
        Args:
            verbose: Whether to print detailed analysis.
            optimize: Whether to optimize BDR parameters.
            
        Returns:
            Dictionary of analysis results.
        """
        print("Starting hierarchical schedulability analysis...")
        results = {}
        
        # Adjust WCET values for core speeds
        self.system.adjust_wcet_for_core_speed()
        
        # Analyze each component
        for comp_id, component in self.system.components.items():
            if verbose:
                print(f"\nAnalyzing component {comp_id}...")
            
            schedulable = self.analyze_component(component, verbose)
            
            if optimize and schedulable:
                alpha, delta = self.optimize_component_bdr(component, verbose=verbose)
                component.set_bdr_interface(alpha, delta)
            else:
                alpha, delta = component.bdr_interface
            
            results[comp_id] = {
                'schedulable': schedulable,
                'bdr_interface': (alpha, delta)
            }
            
            if verbose:
                print(f"Component {comp_id} is {'schedulable' if schedulable else 'not schedulable'}")
                print(f"BDR interface: alpha={alpha:.4f}, delta={delta:.4f}")
        
        # Analyze system-level schedulability (top level)
        for core_id, core in self.system.cores.items():
            if verbose:
                print(f"\nAnalyzing system-level schedulability for core {core_id}...")
            
            total_util = 0
            for component in core.components:
                total_util += component.budget / component.period
                
            # Check system-level schedulability (top level)
            system_schedulable = False
            
            # Check based on top-level scheduler (assume EDF for simplicity)
            system_schedulable = total_util <= 1.0
            
            results[core_id] = {
                'schedulable': system_schedulable,
                'utilization': total_util
            }
            
            if verbose:
                print(f"Core {core_id} utilization: {total_util:.4f}")
                print(f"Core {core_id} is {'schedulable' if system_schedulable else 'not schedulable'}")
        
        print("Analysis completed.")
        return results

class Job:
    """
    Represents a job instance in the simulation.
    """
    def __init__(self, task: Task, release_time: float):
        self.task = task
        self.release_time = release_time
        self.deadline = release_time + task.deadline
        self.remaining_time = task.wcet
        self.completion_time = None
        
    def is_complete(self) -> bool:
        """Check if the job is complete."""
        return self.remaining_time <= 0
        
    def get_response_time(self) -> Optional[float]:
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
        self.current_time = 0
        self.jobs: List[Job] = []
        self.completed_jobs: List[Job] = []
        self.response_times: Dict = {}  # Task -> list of response times
        
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
    
    def select_job_edf(self, available_jobs: List[Job]) -> Optional[Job]:
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
    
    def select_job_rm(self, available_jobs: List[Job]) -> Optional[Job]:
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
        # In a more sophisticated simulation, you'd implement a system-level scheduler
        for component in core.components:
            self.simulate_component(component, time, time_slice)
    
    def simulate(self, duration: float, time_slice: float = 1.0, verbose: bool = False) -> Dict:
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
        
        for time in np.arange(0, duration, time_slice):
            self.current_time = time
            
            if verbose and int(time) % 100 == 0:
                print(f"Simulation time: {time:.2f} / {duration:.2f}")
            
            # Release new jobs
            self.release_jobs(time)
            
            # Simulate execution on each core
            for core_id, core in self.system.cores.items():
                self.simulate_core(core, time, time_slice)
        
        # Calculate response time statistics
        for task_id, data in self.response_times.items():
            if data['values']:
                data['avg'] = sum(data['values']) / len(data['values'])
                data['max'] = max(data['values'])
                
                if verbose:
                    print(f"Task {task_id} response times: "
                          f"avg={data['avg']:.2f}, max={data['max']:.2f}")
        
        print("Simulation completed.")
        return self.response_times

class OutputGenerator:
    """
    Generates output files based on analysis and simulation results.
    """
    def __init__(self, system: HierarchicalSystem, analysis_results: Dict, simulation_results: Dict):
        self.system = system
        self.analysis_results = analysis_results
        self.simulation_results = simulation_results
        
    def generate_csv_output(self, output_file: str = "solution.csv") -> str:
        """
        Generate output file in CSV format.
        
        Args:
            output_file: Output file path.
            
        Returns:
            Path to the generated output file.
        """
        print(f"Generating CSV output to {output_file}...")
        output_data = []
        
        for task_id, task in self.system.tasks.items():
            component_id = task.component_id
            component_schedulable = self.analysis_results[component_id]['schedulable'] if component_id in self.analysis_results else False
            
            # Get response time data from simulation
            avg_response_time = 0
            max_response_time = 0
            
            if task_id in self.simulation_results and self.simulation_results[task_id]['values']:
                avg_response_time = self.simulation_results[task_id]['avg']
                max_response_time = self.simulation_results[task_id]['max']
            
            output_data.append({
                'task_name': task_id,
                'component_id': component_id,                
                'task_schedulable': 1 if component_schedulable else 0,
                'avg_response_time': avg_response_time,
                'max_response_time': max_response_time,
                'component_schedulable': 1 if component_schedulable else 0
            })
        
        # Convert to DataFrame and save to CSV
        df = pd.DataFrame(output_data)
        df.to_csv(output_file, index=False)
        
        print(f"CSV output generated: {output_file}")
        return output_file
    
    def generate_detailed_report(self, output_file: str = "detailed_report.txt") -> str:
        """
        Generate a more detailed report with BDR parameters.
        
        Args:
            output_file: Output file path.
            
        Returns:
            Path to the generated output file.
        """
        print(f"Generating detailed report to {output_file}...")
        with open(output_file, "w") as f:
            f.write("Hierarchical Schedulability Analysis Report\n")
            f.write("=========================================\n\n")
            
            # System-level results
            for core_id, core_results in self.analysis_results.items():
                if core_id in self.system.cores:
                    f.write(f"Core: {core_id}\n")
                    f.write(f"  Schedulable: {core_results['schedulable']}\n")
                    f.write(f"  Utilization: {core_results['utilization']:.4f}\n\n")
            
            # Component-level results
            f.write("Component Analysis:\n")
            for comp_id, comp_results in self.analysis_results.items():
                if comp_id in self.system.components:
                    component = self.system.components[comp_id]
                    if component.bdr_interface:
                        alpha, delta = component.bdr_interface
                        
                        f.write(f"  Component: {comp_id}\n")
                        f.write(f"    Scheduler: {component.scheduler}\n")
                        f.write(f"    PRM: (Q={component.budget}, P={component.period})\n")
                        f.write(f"    BDR: (α={alpha:.4f}, Δ={delta:.4f})\n")
                        f.write(f"    Schedulable: {comp_results['schedulable']}\n\n")
            
            # Task-level results
            f.write("Task Response Times:\n")
            for task_id, response_data in self.simulation_results.items():
                if response_data['values']:
                    f.write(f"  Task: {task_id}\n")
                    f.write(f"    Avg Response Time: {response_data['avg']:.4f}\n")
                    f.write(f"    Max Response Time: {response_data['max']:.4f}\n\n")
        
        print(f"Detailed report generated: {output_file}")
        return output_file

def calculate_hyperperiod(tasks: List[Task]) -> int:
    """
    Calculate the hyperperiod (LCM of all periods) of a set of tasks.
    
    Args:
        tasks: List of tasks.
        
    Returns:
        Hyperperiod value.
    """
    periods = [int(task.period) for task in tasks]
    hyperperiod = 1
    for period in periods:
        hyperperiod = math.lcm(hyperperiod, period)
    return hyperperiod

def main():
    """
    Main function to run the hierarchical schedulability analysis.
    """
    test_case = "2-small-test-case"
    parser = argparse.ArgumentParser(description='Hierarchical Schedulability Analysis System')
    parser.add_argument('--tasks', default=f'test-cases/{test_case}/tasks.csv', help='Tasks CSV file')
    parser.add_argument('--architecture', default=f'test-cases/{test_case}/architecture.csv', help='Architecture CSV file')
    parser.add_argument('--budgets', default=f'test-cases/{test_case}/budgets.csv', help='Budgets CSV file')
    parser.add_argument('--output', default='solution.csv', help='Output CSV file')
    parser.add_argument('--report', default='detailed_report.txt', help='Detailed report file')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--optimize', action='store_true', help='Optimize BDR parameters')
    parser.add_argument('--sim-time', type=float, default=0, 
                        help='Simulation duration (0 = use hyperperiod)')
    parser.add_argument('--time-slice', type=float, default=1.0, help='Simulation time slice')
    
    args = parser.parse_args()
    
    # Parse input files
    input_parser = InputParser(args.tasks, args.architecture, args.budgets)
    cores, components, tasks = input_parser.parse_inputs()
    
    # Build system model
    system = HierarchicalSystem()
    system.build_from_parsed_data(cores, components, tasks)
    
    # Run analysis
    analyzer = AnalysisEngine(system)
    analysis_results = analyzer.analyze_system(verbose=args.verbose, optimize=args.optimize)
    
    # Run simulation
    all_tasks = list(system.tasks.values())
    
    # Determine simulation duration
    if args.sim_time <= 0:
        # Use hyperperiod of all tasks if not specified
        sim_duration = calculate_hyperperiod(all_tasks)
        # Limit to a reasonable value for very large hyperperiods
        # if sim_duration > 10000:
        #     print(f"Hyperperiod is very large ({sim_duration}), limiting to 10000 time units.")
        #     sim_duration = 10000
    else:
        sim_duration = args.sim_time
    
    simulator = SimulationEngine(system)
    simulation_results = simulator.simulate(sim_duration, args.time_slice, verbose=args.verbose)
    
    # Generate outputs
    output_gen = OutputGenerator(system, analysis_results, simulation_results)
    output_gen.generate_csv_output(args.output)
    output_gen.generate_detailed_report(args.report)
    
    # Print summary
    print("\nAnalysis Summary:")
    all_schedulable = True
    for comp_id, results in analysis_results.items():
        if comp_id in system.components:
            schedulable = results['schedulable']
            all_schedulable = all_schedulable and schedulable
            print(f"  Component {comp_id}: {'Schedulable' if schedulable else 'Not schedulable'}")
    
    for core_id, results in analysis_results.items():
        if core_id in system.cores:
            schedulable = results['schedulable']
            all_schedulable = all_schedulable and schedulable
            print(f"  Core {core_id}: {'Schedulable' if schedulable else 'Not schedulable'}")
    
    print(f"\nOverall System: {'Schedulable' if all_schedulable else 'Not schedulable'}")
    print(f"Results saved to {args.output} and {args.report}")

if __name__ == "__main__":
    main()