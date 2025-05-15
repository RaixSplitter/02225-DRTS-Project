import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import argparse
from typing import Dict, List, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)
class ScheduleVisualizer:
    """
    Visualizes the schedule of tasks in a hierarchical scheduling system.
    """
    def __init__(self, simulation_file: str, system_file: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            simulation_file: Path to simulation output file.
            system_file: Optional path to file with system structure information.
        """
        self.simulation_file = simulation_file
        self.system_file = system_file
        
        self.simulation_data = None
        self.system_structure = None
        self.components = {}
        self.tasks = {}
        self.task_colors = {}
        
        # Color palette for tasks
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
        ]
    
    def load_data(self) -> None:
        """Load simulation and system data."""
        logger.info(f"Loading simulation data from {self.simulation_file}...")
        
        # Determine file type and load accordingly
        if self.simulation_file.endswith('.csv'):
            self.simulation_data = pd.read_csv(self.simulation_file)
        elif self.simulation_file.endswith('.json'):
            with open(self.simulation_file, 'r') as f:
                self.simulation_data = json.load(f)
        else:
            raise ValueError("Unsupported simulation file format. Use CSV or JSON.")
        
        # Load system structure if provided
        if self.system_file:
            logger.info(f"Loading system structure from {self.system_file}...")
            
            if self.system_file.endswith('.json'):
                with open(self.system_file, 'r') as f:
                    self.system_structure = json.load(f)
            else:
                raise ValueError("Unsupported system file format. Use JSON.")
        
        self._extract_components_and_tasks()
    
    def _extract_components_and_tasks(self) -> None:
        """Extract component and task information from the data."""
        # If we have CSV data
        if isinstance(self.simulation_data, pd.DataFrame):
            # Extract components and their tasks
            for _, row in self.simulation_data.iterrows():
                component_id = row['component_id']
                task_name = row['task_name']
                
                if component_id not in self.components:
                    self.components[component_id] = []
                
                self.components[component_id].append(task_name)
                self.tasks[task_name] = {
                    'component_id': component_id,
                    'avg_response_time': row['avg_response_time'],
                    'max_response_time': row['max_response_time'],
                    'schedulable': row['task_schedulable']
                }
        
        # If we have additional system structure data
        if self.system_structure:
            # Enhance with additional data (task periods, wcets, etc.)
            for task_info in self.system_structure.get('tasks', []):
                task_name = task_info['name']
                if task_name in self.tasks:
                    self.tasks[task_name].update({
                        'period': task_info['period'],
                        'wcet': task_info['wcet'],
                        'scheduler': task_info['scheduler']
                    })
            
            # Add component information
            for comp_info in self.system_structure.get('components', []):
                comp_id = comp_info['id']
                if comp_id in self.components:
                    self.components[comp_id] = {
                        'scheduler': comp_info['scheduler'],
                        'budget': comp_info['budget'],
                        'period': comp_info['period'],
                        'tasks': self.components[comp_id]
                    }
        
        # Assign colors to tasks
        color_idx = 0
        for task_name in self.tasks:
            self.task_colors[task_name] = self.colors[color_idx % len(self.colors)]
            color_idx += 1
    
    def _create_execution_trace(self, timeline_length: int, time_slice: float = 1.0) -> Dict:
        """
        Create a simulated execution trace based on the loaded data.
        
        This is a simplified trace for visualization purposes, not an exact execution.
        
        Args:
            timeline_length: Length of the timeline to visualize.
            time_slice: Resolution of the simulation.
            
        Returns:
            Dictionary mapping time points to executing tasks.
        """
        # Create a simple simulated execution trace
        execution_trace = {}
        
        # Sort components by their scheduler priority (arbitrary for visualization)
        sorted_components = sorted(self.components.keys())
        
        # For each time point
        for t in np.arange(0, timeline_length, time_slice):
            time_point = round(t, 2)  # Round to avoid floating point issues
            execution_trace[time_point] = {}
            
            # For each component
            for comp_id in sorted_components:
                tasks = self.components[comp_id] if isinstance(self.components[comp_id], list) else self.components[comp_id]['tasks']
                
                # Determine which task would be executing at this time point
                # This is a simplified model for visualization purposes
                executing_task = None
                
                for task_name in tasks:
                    task = self.tasks[task_name]
                    
                    # If we have period information, we can create a more accurate trace
                    if 'period' in task:
                        period = task['period']
                        wcet = task['wcet']
                        
                        # Check if task would be active at this time point
                        if time_point % period < wcet:
                            executing_task = task_name
                            break
                    else:
                        # Simple round-robin if we don't have detailed information
                        if time_point % (len(tasks) * 10) // 10 == tasks.index(task_name):
                            executing_task = task_name
                            break
                
                execution_trace[time_point][comp_id] = executing_task
        
        return execution_trace
    
    def visualize_schedule(self, timeline_length: int = 100, time_slice: float = 1.0, 
                          show_components: bool = True, output_file: Optional[str] = None) -> None:
        """
        Visualize the schedule on a timeline.
        
        Args:
            timeline_length: Length of the timeline to visualize.
            time_slice: Resolution of the simulation.
            show_components: Whether to group tasks by component.
            output_file: Optional file to save the visualization.
        """
        if not self.simulation_data:
            self.load_data()
        
        # Create execution trace
        trace = self._create_execution_trace(timeline_length, time_slice)
        
        # Prepare the plot
        plt.figure(figsize=(15, 8))
        
        # Determine the number of rows in the plot
        if show_components:
            rows = len(self.components)
            row_labels = list(self.components.keys())
        else:
            rows = len(self.tasks)
            row_labels = list(self.tasks.keys())
        
        # Create the visualization
        plt.gca().set_xlim(0, timeline_length)
        plt.gca().set_ylim(0, rows)
        
        # Add grid lines
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Plot task executions
        if show_components:
            for comp_idx, comp_id in enumerate(self.components):
                y_pos = rows - comp_idx - 0.5
                
                # Plot component boundaries
                plt.axhline(y=rows-comp_idx, color='black', linestyle='-', alpha=0.3)
                
                # Plot component budget (if available)
                if isinstance(self.components[comp_id], dict) and 'budget' in self.components[comp_id] and 'period' in self.components[comp_id]:
                    budget = self.components[comp_id]['budget']
                    period = self.components[comp_id]['period']
                    
                    for p in range(0, timeline_length, int(period)):
                        plt.axvspan(p, p + budget, ymin=(rows-comp_idx-1)/rows, ymax=(rows-comp_idx)/rows, 
                                    color='lightgray', alpha=0.5)
                
                # Plot task executions
                last_task = None
                start_time = 0
                
                for time_point in sorted(trace.keys()):
                    executing_task = trace[time_point].get(comp_id)
                    
                    if executing_task != last_task:
                        if last_task:
                            # Draw the previous task execution
                            plt.barh(y_pos, time_point - start_time, left=start_time, height=0.5, 
                                    color=self.task_colors.get(last_task, 'gray'), alpha=0.7)
                            
                            # Add task label if the bar is wide enough
                            if time_point - start_time > timeline_length / 20:
                                plt.text(start_time + (time_point - start_time)/2, y_pos, last_task, 
                                        ha='center', va='center', fontsize=8)
                        
                        # Start a new task execution
                        start_time = time_point
                        last_task = executing_task
                
                # Draw the last task execution
                if last_task:
                    plt.barh(y_pos, timeline_length - start_time, left=start_time, height=0.5, 
                            color=self.task_colors.get(last_task, 'gray'), alpha=0.7)
        else:
            # Plot each task individually
            for task_idx, task_name in enumerate(self.tasks):
                y_pos = rows - task_idx - 0.5
                
                # Plot task boundaries
                plt.axhline(y=rows-task_idx, color='black', linestyle='-', alpha=0.3)
                
                # Find executions of this task
                executions = []
                last_executing = False
                start_time = 0
                
                for time_point in sorted(trace.keys()):
                    comp_id = self.tasks[task_name]['component_id']
                    executing = trace[time_point].get(comp_id) == task_name
                    
                    if executing != last_executing:
                        if last_executing:
                            # Record the execution interval
                            executions.append((start_time, time_point))
                        
                        # Start a new interval
                        start_time = time_point
                        last_executing = executing
                
                # Record the last execution interval
                if last_executing:
                    executions.append((start_time, timeline_length))
                
                # Draw the executions
                for start, end in executions:
                    plt.barh(y_pos, end - start, left=start, height=0.5, 
                            color=self.task_colors.get(task_name, 'gray'), alpha=0.7)
        
        # Add labels and title
        plt.yticks(np.arange(rows) + 0.5, reversed(row_labels))
        plt.xlabel('Time')
        plt.ylabel('Component' if show_components else 'Task')
        plt.title('Hierarchical Schedule Visualization')
        
        # Add legend for tasks
        handles = [plt.Rectangle((0,0), 1, 1, color=color) for color in self.task_colors.values()]
        labels = list(self.task_colors.keys())
        if len(labels) < 10:
            plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure if requested
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_file}")
        
        # Show the plot
        plt.show()
    
    def visualize_response_times(self, output_file: Optional[str] = None) -> None:
        """
        Visualize the response times of tasks.
        
        Args:
            output_file: Optional file to save the visualization.
        """
        if not self.simulation_data:
            self.load_data()
        
        # Prepare the plot
        plt.figure(figsize=(12, 6))
        
        # Extract task names and response times
        task_names = []
        avg_response_times = []
        max_response_times = []
        
        for task_name, task_data in self.tasks.items():
            task_names.append(task_name)
            avg_response_times.append(task_data['avg_response_time'])
            max_response_times.append(task_data['max_response_time'])
        
        # Create bar chart
        x = np.arange(len(task_names))
        width = 0.35
        
        plt.bar(x - width/2, avg_response_times, width, label='Avg Response Time')
        plt.bar(x + width/2, max_response_times, width, label='Max Response Time')
        
        # Add task periods as horizontal lines (if available)
        for i, task_name in enumerate(task_names):
            if 'period' in self.tasks[task_name]:
                period = self.tasks[task_name]['period']
                plt.axhline(y=period, xmin=i/len(task_names), xmax=(i+1)/len(task_names), 
                           color='red', linestyle='--', alpha=0.5)
        
        # Add labels and title
        plt.xlabel('Task')
        plt.ylabel('Response Time')
        plt.title('Task Response Times')
        plt.xticks(x, task_names, rotation=45)
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure if requested
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Response time visualization saved to {output_file}")
        
        # Show the plot
        plt.show()
    
    def create_system_structure_file(self, output_file: str = "system_structure.json") -> None:
        """
        Create a system structure file based on simulation data.
        
        This is useful when you don't have a system structure file but want to visualize.
        
        Args:
            output_file: File to save the structure information.
        """
        if not self.simulation_data:
            self.load_data()
        
        # Create a system structure based on available information
        structure = {
            "tasks": [],
            "components": []
        }
        
        # Add components
        for comp_id, tasks in self.components.items():
            component = {
                "id": comp_id,
                "scheduler": "EDF",  # Default
                "budget": 10,  # Default
                "period": 50   # Default
            }
            structure["components"].append(component)
        
        # Add tasks
        for task_name, task_data in self.tasks.items():
            task = {
                "name": task_name,
                "component_id": task_data["component_id"],
                "scheduler": "EDF",  # Default
                "period": 100,       # Default
                "wcet": 10           # Default
            }
            structure["tasks"].append(task)
        
        # Save the structure file
        with open(output_file, 'w') as f:
            json.dump(structure, f, indent=2)
        
        logger.info(f"System structure file created: {output_file}")

def main():
    """Main function to run the visualization tool."""
    parser = argparse.ArgumentParser(description='Hierarchical Schedule Visualization Tool')
    parser.add_argument('--simulation', required=True, help='Simulation output file (CSV or JSON)')
    parser.add_argument('--system', help='System structure file (JSON)')
    parser.add_argument('--output', help='Output file for schedule visualization')
    parser.add_argument('--response-output', help='Output file for response time visualization')
    parser.add_argument('--timeline', type=int, default=100, help='Timeline length to visualize')
    parser.add_argument('--time-slice', type=float, default=1.0, help='Time slice resolution')
    parser.add_argument('--by-task', action='store_true', help='Show each task separately instead of by component')
    parser.add_argument('--create-structure', help='Create a system structure file based on simulation data')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ScheduleVisualizer(args.simulation, args.system)
    
    # Create system structure file if requested
    if args.create_structure:
        visualizer.create_system_structure_file(args.create_structure)
    
    # Visualize schedule
    visualizer.visualize_schedule(
        timeline_length=args.timeline,
        time_slice=args.time_slice,
        show_components=not args.by_task,
        output_file=args.output
    )
    
    # Visualize response times
    visualizer.visualize_response_times(output_file=args.response_output)

if __name__ == "__main__":
    main()