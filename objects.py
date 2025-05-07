from dataclasses import dataclass, field

@dataclass
class Task:
    """
    Represents a periodic task in a real-time system.
    """
    name: str
    wcet: float
    period: float
    component_id: str
    scheduler: str
    priority: float = None
    deadline: float = None  # NOTE: Implicit deadline model, could change idk
    
    def __post_init__(self):
        if self.deadline is None:
            self.deadline = self.period


@dataclass
class Component:
    """
    Represents a component in a hierarchical scheduling system.
    """
    component_id: str
    scheduler: str
    budget: float
    period: float
    core_id: str
    tasks: list[Task] = field(default_factory=list)
    bdr_interface: None | tuple[float, float] = None

    def add_task(self, task: Task) -> None:
        self.tasks.append(task)
        
    def set_bdr_interface(self, alpha: float, delta: float) -> None:
        """Set the BDR interface parameters."""
        self.bdr_interface = (alpha, delta)
        

@dataclass
class Core:
    """
    Represents a processing core in the system.
    """
    core_id: str
    speed_factor: float
    components: list[Component] = field(default_factory=list)
        
    def add_component(self, component: Component) -> None:
        self.components.append(component)


@dataclass
class HierarchicalSystem:
    """
    Represents the entire hierarchical scheduling system.
    """
    cores: dict[str, Core] = field(default_factory=dict)
    components: dict[str, Component] = field(default_factory=dict)
    tasks: dict[str, Task] = field(default_factory=dict)

    def build(self, cores: dict, components: dict, tasks: dict) -> None:
        """
        Build the hierarchical system from parsed data.
        """
        for core_id, core_data in cores.items():
            self.cores[core_id] = Core(core_id, speed_factor=core_data['speed_factor'])
        
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
        
        for task_id, task_data in tasks.items():
            task = Task(
                task_data['name'],
                task_data['wcet'],
                task_data['period'],
                task_data['component_id'],
                task_data['priority'] if 'priority' in task_data and task_data['priority'] else None
            )
            self.tasks[task_id] = task
            
            if task_data['component_id'] in self.components:
                self.components[task_data['component_id']].add_task(task)


    def adjust_wcet_for_core_speed(self) -> None:
        """
        Adjust WCET values based on core speed factors.
        """
        for _task_name, task in self.tasks.items():
            component = self.components.get(task.component_id)
            if component is None:
                continue
            core = self.cores.get(component.core_id)
            if core is None:
                continue
            # Adjust WCET: slower core (speed < 1) increases WCET
            task.wcet = task.wcet / core.speed_factor


@dataclass
class Job:
    """
    Represents a job instance in the simulation.
    """
    task: Task
    release_time: float
    deadline: float
    remaining_time: float
    completion_time: None | float = None
    
    def __init__(self, task: Task, release_time: float, deadline: float | None = None) -> None:
        self.task = task
        self.release_time = release_time
        self.deadline = release_time + task.deadline if deadline is None else deadline
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