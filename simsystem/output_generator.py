import pandas as pd
from typing import Dict
from simsystem.objects import HierarchicalSystem
import logging

logger = logging.getLogger(__name__)


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
        logger.info(f"Generating CSV output to {output_file}...")
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
        
        logger.info(f"CSV output generated: {output_file}")
        return output_file
    
    def generate_detailed_report(self, output_file: str = "detailed_report.txt") -> str:
        """
        Generate a more detailed report with BDR parameters.
        
        Args:
            output_file: Output file path.
            
        Returns:
            Path to the generated output file.
        """
        logger.info(f"Generating detailed report to {output_file}...")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write("Hierarchical Schedulability Analysis Report by Analysis Engine\n")
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
            f.write("Task Response Times during simulation:\n")
            for task_id, response_data in self.simulation_results.items():
                if response_data['values']:
                    f.write(f"  Task: {task_id}\n")
                    f.write(f"    Avg Response Time: {response_data['avg']:.4f}\n")
                    f.write(f"    Max Response Time: {response_data['max']:.4f}\n\n")
        
        logger.info(f"Detailed report generated: {output_file}")
        return output_file

