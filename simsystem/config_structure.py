from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore


@dataclass
class FilesConfig:
    """
    Configuration for file paths.
    """

    test_cases_dir: str = "test-cases"
    test_case: str = "1-tiny-test-case"
    tasks: str = "tasks.csv"
    architecture: str = "architecture.csv"
    budgets: str = "budgets.csv"
    results_dir: str = "results"
    output: str = "solution.csv"
    report: str = "detailed_report.txt"


@dataclass
class SettingsConfig:
    """
    Configuration for settings.
    """

    verbose: bool = True
    optimize: bool = True
    sim_time: float = 0.0
    time_slice: float = 1.0


@dataclass
class ExperimentConfig:
    """
    Configuration for the experiment.
    """

    files: FilesConfig = field(default_factory=FilesConfig)
    settings: SettingsConfig = field(default_factory=SettingsConfig)
