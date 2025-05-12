from dataclasses import dataclass

import pandas as pd
import numpy as np
import math
import argparse

from typing import Dict, List, Tuple, Set, Optional

from objects import Task, Component, Core, HierarchicalSystem
from simulator import SimulationEngine
from input_parser import InputParser
from utils import calculate_hyperperiod

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
        
        
        
        best_cost = c1 * best_alpha + c2 * (1.0 / best_delta) if best_delta != 0 else float('inf')
        
        # Simple optimization: incrementally increase alpha until schedulable
        alpha = initial_alpha
        while alpha <= 1.0:
            # Compute corresponding delta with the Half-Half algorithm
            delta = 2.0 * (component.period * alpha - component.budget)
            
            if delta == 0:
                alpha += alpha_increment
                continue
            
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
        with open(output_file, "w", encoding='utf-8') as f:
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



def main():
    """
    Main function to run the hierarchical schedulability analysis.
    """
    test_case = "4-large-test-case"
    parser = argparse.ArgumentParser(description='Hierarchical Schedulability Analysis System')
    parser.add_argument('--tasks', default=f'test-cases/{test_case}/tasks.csv', help='Tasks CSV file')
    parser.add_argument('--architecture', default=f'test-cases/{test_case}/architecture.csv', help='Architecture CSV file')
    parser.add_argument('--budgets', default=f'test-cases/{test_case}/budgets.csv', help='Budgets CSV file')
    parser.add_argument('--output', default='solution.csv', help='Output CSV file')
    parser.add_argument('--report', default='detailed_report.txt', help='Detailed report file')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--optimize', default = 'true', action='store_true', help='Optimize BDR parameters')
    parser.add_argument('--sim-time', type=float, default=0, help='Simulation duration (0 = use hyperperiod)')
    parser.add_argument('--time-slice', type=float, default=1.0, help='Simulation time slice')
    
    args = parser.parse_args()
    
    # Parse input files
    input_parser = InputParser(args.tasks, args.architecture, args.budgets)
    cores, components, tasks = input_parser.parse_inputs()
    
    # Build system model
    system = HierarchicalSystem()
    system.build(cores, components, tasks)
    
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