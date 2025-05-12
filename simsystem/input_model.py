from simsystem.objects import Core, Component, Task
import pandas as pd


class Input_Model:
    @staticmethod
    def read_architecture(filepath: str) -> dict[str, Core]:
        """
        Reads the architecture case file and instantiates the Cores
        """

        df = pd.read_csv(filepath, sep=",")

        cores = {}
        for _, row in df.iterrows():
            core = Core(
                core_id=row["core_id"],
                speed_factor=row["speed_factor"],
                scheduler=row["scheduler"],
            )
            cores[row["core_id"]] = core

        return cores

    @staticmethod
    def read_budgets(filepath: str) -> dict[str, Component]:
        """
        Reads the budget case file and instantiates the Components
        """

        df = pd.read_csv(filepath, sep=",")

        components = {}
        for _, row in df.iterrows():
            component = Component(
                component_id=row["component_id"],
                scheduler=row["scheduler"],
                budget=row["budget"],
                period=row["period"],
                core_id=row["core_id"],
                priority=row["priority"],
            )
            components[row["component_id"]] = component

        return components


    @staticmethod
    def read_tasks(filepath: str) -> dict[str, Task]:
        """
        Reads the task case file and instantiates the Tasks
        """
        
        df = pd.read_csv(filepath, sep=",")

        tasks = {}
        for _, row in df.iterrows():
            task = Task(
            name=row["task_name"],
            wcet=row["wcet"],
            period=row["period"],
            deadline_interval=row["period"], # Assuming deadline is equal to period
            component_id=row["component_id"],
            priority=row["priority"],
            )
            tasks[row["task_name"]] = task

        return tasks


if __name__ == "__main__":
    # Example usage
    architecture = Input_Model.read_architecture(
        "test-cases/1-tiny-test-case/architecture.csv"
    )
    budgets = Input_Model.read_budgets("test-cases/1-tiny-test-case/budgets.csv")
    tasks = Input_Model.read_tasks("test-cases/1-tiny-test-case/tasks.csv")
    
    print(architecture)
    print(budgets)
    print(tasks)

