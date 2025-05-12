import pandas as pd


class InputParser:
    """
    Parses input CSV files into data structures.
    """
    def __init__(self, tasks_file: str, architecture_file: str, budgets_file: str) -> None:
        self.tasks_file = tasks_file
        self.architecture_file = architecture_file
        self.budgets_file = budgets_file


    def parse_inputs(self) -> tuple[dict, dict, dict]:
        """
        Parse input CSV files into dictionaries.
        Output example from '2-small-test-case':
            (
                # cores
                {'Core_1': {'speed_factor': 0.62}},
                # components
                {
                    'Camera_Sensor': {'scheduler': 'RM', 'budget': 4, 'period': 7, 'core_id': 'Core_1', 'tasks': []}, 
                    'Image_Processor': {'scheduler': 'EDF', 'budget': 5, 'period': 16, 'core_id': 'Core_1', 'tasks': []}
                }, 
                # tasks
                {
                    'Task_0': {'name': 'Task_0', 'wcet': 3, 'period': 150, 'component_id': 'Camera_Sensor', 'priority': 1.0}, 
                    'Task_1': {'name': 'Task_1', 'wcet': 28, 'period': 200, 'component_id': 'Camera_Sensor', 'priority': 2.0}, 
                    'Task_2': {'name': 'Task_2', 'wcet': 2, 'period': 50, 'component_id': 'Camera_Sensor', 'priority': 0.0}, 
                    'Task_3': {'name': 'Task_3', 'wcet': 24, 'period': 300, 'component_id': 'Camera_Sensor', 'priority': 3.0}, 
                    'Task_4': {'name': 'Task_4', 'wcet': 2, 'period': 200, 'component_id': 'Image_Processor', 'priority': None}, 
                    'Task_5': {'name': 'Task_5', 'wcet': 11, 'period': 200, 'component_id': 'Image_Processor', 'priority': None}, 
                    'Task_6': {'name': 'Task_6', 'wcet': 17, 'period': 400, 'component_id': 'Image_Processor', 'priority': None}, 
                    'Task_7': {'name': 'Task_7', 'wcet': 13, 'period': 300, 'component_id': 'Image_Processor', 'priority': None}, 
                    'Task_8': {'name': 'Task_8', 'wcet': 3, 'period': 150, 'component_id': 'Image_Processor', 'priority': None}
                }
            )
        """
        print(f"Parsing input files...")
        
        # Read CSV files
        arch_df = pd.read_csv(self.architecture_file)
        budgets_df = pd.read_csv(self.budgets_file)
        tasks_df = pd.read_csv(self.tasks_file)
        
        cores = {}
        for _, row in arch_df.iterrows():
            cores[row['core_id']] = {
                'speed_factor': row['speed_factor']
            }
        
        components = {}
        for _, row in budgets_df.iterrows():
            components[row['component_id']] = {
                'scheduler': row['scheduler'],
                'budget': row['budget'],
                'period': row['period'],
                'core_id': row['core_id'],
                'tasks': []
            }
        
        tasks = {}
        for _, row in tasks_df.iterrows():
            task = {
                'name': row['task_name'],
                'wcet': row['wcet'],
                'period': row['period'],
                'component_id': row['component_id'],
                'priority': row['priority'] if 'priority' in row and not pd.isna(row['priority']) else None
            }
            tasks[row['task_name']] = task
            
        print(f"Parsed {len(cores)} cores, {len(components)} components, and {len(tasks)} tasks.")
        return cores, components, tasks

