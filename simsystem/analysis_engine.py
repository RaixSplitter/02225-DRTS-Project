import math
from typing import Dict, List, Tuple
from simsystem.objects import Task, Component, Core, HierarchicalSystem
import logging

logger = logging.getLogger(__name__)

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
            return sorted(tasks, key=lambda t: t.priority)
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
            verbose: Whether to logger.info detailed analysis.
            
        Returns:
            Whether the component is schedulable.
        """
        tasks = self.compute_rm_priority_order(component.tasks)
        
        if verbose:
            logger.info(f"\nAnalyzing component {component.component_id} under RM:")
            logger.info(f"  BDR parameters: alpha={alpha:.4f}, delta={delta:.4f}")
        
        for i, task in enumerate(tasks):
            # Find critical time points for this task
            critical_points = self.find_critical_time_points_rm(tasks, i)
            
            # Check if there exists a time t where demand <= supply
            schedulable = False
            
            for t in critical_points:
                demand = self.dbf_rm(tasks, t, i)
                supply = self.sbf_bdr(alpha, delta, t)
                
                if verbose:
                    logger.info(f"  Task {task.name} at t={t:.2f}: demand (DBF)={demand:.2f}, supply (SPF)={supply:.2f}")
                
                if demand <= supply:
                    schedulable = True
                    break
            
            if not schedulable:
                if verbose:
                    logger.info(f"  Task {task.name} is not schedulable!")
                return False
            
            if verbose:
                logger.info(f"  Task {task.name} is schedulable.")
        
        return True
    
    def check_schedulability_edf(self, component: Component, alpha: float, delta: float, 
                                verbose: bool = False) -> bool:
        """
        Check schedulability of component tasks under EDF using BDR.
        
        Args:
            component: Component to analyze.
            alpha: Resource availability factor.
            delta: Maximum delay in resource allocation.
            verbose: Whether to logger.info detailed analysis.
            
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
            logger.info(f"\nAnalyzing component {component.component_id} under EDF:")
            logger.info(f"  BDR parameters: alpha={alpha:.4f}, delta={delta:.4f}")
            logger.info(f"  Hyperperiod: {hyperperiod}")
        
        # Find critical time points
        critical_points = self.find_critical_time_points_edf(tasks, hyperperiod)
        
        # Check at critical time points
        for t in critical_points:
            demand = self.dbf_edf(tasks, t)
            supply = self.sbf_bdr(alpha, delta, t)
            
            if verbose:
                logger.info(f"  At t={t:.2f}: demand={demand:.2f}, supply={supply:.2f}")
            
            if demand > supply:
                if verbose:
                    logger.info("  Not schedulable!")
                return False
        
        return True
    
    def analyze_component(self, component: Component, verbose: bool = False) -> bool:
        """
        Analyze a component and determine its schedulability.
        
        Args:
            component: Component to analyze.
            verbose: Whether to logger.info detailed analysis.
            
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
    
    def optimize_component_bdr(self, component: Component, alpha_increment: float = 0.01, verbose: bool = False) -> Tuple[float, float]:
        """
        Optional: Optimize BDR parameters for a component.
        
        Args:
            component: Component to optimize.
            alpha_increment: Step size for alpha optimization.
            verbose: Whether to logger.info optimization details.
            
        Returns:
            Optimized (alpha, delta) pair.
        """
        # Start with a reasonable initial guess
        initial_alpha, initial_delta = self.convert_prm_to_bdr(component.budget, component.period)
        
        # Cost function weights
        c1 = 0.95  # Weight for alpha (processor utilization)
        c2 = 0.05  # Weight for switch_cost (inversely related to delta)
        
        best_alpha = initial_alpha
        best_delta = initial_delta
        if best_delta == 0.0:
            logger.info("Delta was 0, adding eps")
            best_delta = 1e-6
        best_cost = c1 * best_alpha + c2 * (1.0 / best_delta)
               
        # Simple optimization: incrementally increase alpha until schedulable
        alpha = initial_alpha
        while alpha <= 1.0:
            # Compute corresponding delta with the Half-Half algorithm
            delta = 2.0 * (component.period * alpha - component.budget)
            if delta == 0.0:
                delta = 1e-6

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
            logger.info(f"Optimized BDR for {component.component_id}: " 
                  f"alpha={best_alpha:.4f}, delta={best_delta:.4f}")
        
        return best_alpha, best_delta
    
    def optimize_prm_for_component(self, component: Component, period_range, budget_step=1.0, verbose=False):
        """
        Search for the best (budget, period) pair for the component, considering core utilization.
        period_range: (min_period, max_period, step)
        """
        best_budget = None
        best_period = None
        best_cost = float('inf')

        # Find the core this component belongs to
        core_of_component = None
        for core in self.system.cores.values():
            if component in core.components:
                core_of_component = core
                break

        if core_of_component is None:
            logger.warning(f"Component {component.component_id} is not assigned to any core.")
            return None, None

        for period in range(*period_range):
            for budget in range(1, int(period)+1, int(budget_step)):
                alpha, delta = self.convert_prm_to_bdr(budget, period)

                # Calculate total utilization if this component used (budget, period)
                total_util = 0.0
                for comp in core_of_component.components:
                    if comp is component:
                        total_util += budget / period
                    else:
                        total_util += comp.budget / comp.period
                if total_util > 1.0:
                    continue  # Skip this candidate, would make core unschedulable

                if component.scheduler == "RM":
                    schedulable = self.check_schedulability_rm(component, alpha, delta)
                else:
                    schedulable = self.check_schedulability_edf(component, alpha, delta)
                if schedulable:
                    if delta == 0.0:
                        delta = 1e-6
                    cost = 0.95 * alpha + 0.05 * (1.0 / delta)
                    if cost < best_cost:
                        best_budget = budget
                        best_period = period
                        best_cost = cost
        if verbose:
            logger.info(f"Best PRM: budget={best_budget}, period={best_period}, cost={best_cost}")
        return best_budget, best_period
    
    def analyze_system(self, verbose: bool = False, optimize: bool = False) -> Dict:
        """
        Analyze the entire hierarchical system.
        
        Args:
            verbose: Whether to logger.info detailed analysis.
            optimize: Whether to optimize BDR parameters.
            
        Returns:
            Dictionary of analysis results.
        """
        logger.info("Starting hierarchical schedulability analysis...")
        results = {}
        
        # Adjust WCET values for core speeds
        self.system.adjust_wcet_for_core_speed()
        
        # Analyze each component
        for comp_id, component in self.system.components.items():
            if verbose:
                logger.info(f"\nAnalyzing component {comp_id}...")
            
            schedulable = self.analyze_component(component, verbose)
            
            # Try to optimize PRM if not schedulable and optimize flag is set
            if optimize:
                # Example period range: (min_period, max_period+1, step)
                min_period = int(min(task.period for task in component.tasks))
                max_period = int(2 * min_period)
                period_range = (min_period, max_period + 1, 1)
                best_budget, best_period = self.optimize_prm_for_component(component, period_range, budget_step=1.0, verbose=verbose)
                if best_budget is not None and best_period is not None:
                    component.budget = best_budget
                    component.period = best_period
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
                logger.info(f"Component {comp_id} is {'schedulable' if schedulable else 'not schedulable'}")
                logger.info(f"BDR interface: alpha={alpha:.4f}, delta={delta:.4f}")
        
        # Analyze system-level schedulability (top level)
        for core_id, core in self.system.cores.items():
            if verbose:
                logger.info(f"\nAnalyzing system-level schedulability for core {core_id}...")
            
            alphas, deltas = zip(*[component.bdr_interface for component in core.components])
            total_alpha = sum(alphas)
            
            utility_check = total_alpha <= 1.0 # Assuming Core utilization is 1.0
            delay_check = all(delta > -1e-10 for delta in deltas) # Assuming Core is able to guarentee ressources instantaneously
            all_components_schedulable = all(
                results[component.component_id]['schedulable'] for component in core.components
            )
            
            system_schedulable = utility_check and delay_check and all_components_schedulable
            
            results[core_id] = {
                'schedulable': system_schedulable,
                'utilization': total_alpha
            }
            
            if verbose:
                logger.info(f"Core {core_id} utilization: {total_alpha:.4f}")
                logger.info(f"Core {core_id} is {'schedulable' if system_schedulable else 'not schedulable'}")
        
        logger.info("Analysis completed.")
        return results
