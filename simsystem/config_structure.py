from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class DefaultConfig:
    cases_dir: str = 'test-cases'

@dataclass
class ExperimentConfig:
    default: DefaultConfig = field(default_factory=DefaultConfig)
    
    case: str = '1-tiny-test-case'





