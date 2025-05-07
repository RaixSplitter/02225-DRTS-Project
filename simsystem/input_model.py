from simsystem.objects import Cores, Component, Task



class Input_Model:
    
    @staticmethod
    def read_architecture(filepath : str) -> dict[str, Cores]:
        """
        Reads the architecture case file and instantiates the Cores
        """
        
        return NotImplementedError
    
    @staticmethod
    def read_budgets(filepath : str) -> dict[str, Component]:
        """
        Reads the budget case file and instantiates the Components
        """

        return NotImplementedError
        
        
    @staticmethod
    def read_tasks(filepath : str) -> dict[str, Task]:
        """
        Reads the task case file and instantiates the Tasks
        """

        return NotImplementedError
        




