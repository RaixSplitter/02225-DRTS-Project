import os
import json
import logging.config
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

import simsystem.analytical_tools as at
from simsystem.input_model import Input_Model
from simsystem.config_structure import ExperimentConfig


logger = logging.getLogger(__name__)

cs = ConfigStore.instance()

cs.store(name="experiment", node=ExperimentConfig)


@hydra.main(version_base=None, config_path="../conf", config_name="experiment")
def main(cfg: ExperimentConfig) -> None:
    logger.info("Starting the simulation system...")

    case_path = f"{cfg.default.cases_dir}/{cfg.case}"
    architecture_path = os.path.join(case_path, "architecture.csv")
    budgets_path = os.path.join(case_path, "budgets.csv")
    tasks_path = os.path.join(case_path, "tasks.csv")

    cores = Input_Model.read_architecture(architecture_path)
    components = Input_Model.read_budgets(budgets_path)
    tasks = Input_Model.read_tasks(tasks_path)

    logger.info(f"Loaded architecture: {cores}")
    logger.info(f"Loaded budgets: {components}")
    logger.info(f"Loaded tasks: {tasks}")

if __name__ == "__main__":

    main()
