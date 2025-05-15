import argparse
import os
from simsystem.objects import HierarchicalSystem
from simsystem.simulator import SimulationEngine
from simsystem.input_parser import InputParser
from simsystem.utils import calculate_hyperperiod
from simsystem.analysis_engine import AnalysisEngine
from simsystem.output_generator import OutputGenerator

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from simsystem.config_structure import ExperimentConfig
import logging.config

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)

@hydra.main(version_base=None, config_path="../conf", config_name="experiment3")
def main(cfg: ExperimentConfig) -> None:
    """
    Main function to run the hierarchical schedulability analysis.
    """
    
    test_case_path = f"{cfg.files.test_cases_dir}/{cfg.files.test_case}"
    tasks_path = f"{test_case_path}/{cfg.files.tasks}"
    architecture_path = f"{test_case_path}/{cfg.files.architecture}"
    budgets_path = f"{test_case_path}/{cfg.files.budgets}"
    
    # Parse input files
    input_parser = InputParser(tasks_path, architecture_path, budgets_path)
    cores, components, tasks = input_parser.parse_inputs()
    
    # Build system model
    system = HierarchicalSystem()
    system.build(cores, components, tasks)
    
    # Run analysis
    analyzer = AnalysisEngine(system)
    analysis_results = analyzer.analyze_system(verbose=cfg.settings.verbose, optimize=cfg.settings.optimize)
    
    # Run simulation
    all_tasks = list(system.tasks.values())
    
    # Determine simulation duration
    if cfg.settings.sim_time <= 0:
        # Use hyperperiod of all tasks if not specified
        sim_duration = calculate_hyperperiod(all_tasks)
        # Limit to a reasonable value for very large hyperperiods
        # if sim_duration > 10000:
        #     logger.info(f"Hyperperiod is very large ({sim_duration}), limiting to 10000 time units.")
        #     sim_duration = 10000
    else:
        sim_duration = cfg.settings.sim_time
    
    simulator = SimulationEngine(system)
    simulation_results = simulator.simulate(sim_duration, cfg.settings.time_slice, verbose=cfg.settings.verbose)
    
    # Generate outputs
    os.makedirs(f"{cfg.files.results_dir}/{cfg.files.test_case}", exist_ok=True)
    
    if cfg.settings.optimize:
        output_path = f"{cfg.files.results_dir}/{cfg.files.test_case}/{cfg.files.output}Optimized.csv"
        detailed_report_path = f"{cfg.files.results_dir}/{cfg.files.test_case}/{cfg.files.report}Optimized.txt"
    else:
        output_path = f"{cfg.files.results_dir}/{cfg.files.test_case}/{cfg.files.output}.csv"
        detailed_report_path = f"{cfg.files.results_dir}/{cfg.files.test_case}/{cfg.files.report}.txt"
    
    output_gen = OutputGenerator(system, analysis_results, simulation_results)
    output_gen.generate_csv_output(output_path)
    output_gen.generate_detailed_report(detailed_report_path)
    # logger.info summary
    logger.info("\nAnalysis Summary:")
    all_schedulable = True
    for comp_id, results in analysis_results.items():
        if comp_id in system.components:
            schedulable = results['schedulable']
            all_schedulable = all_schedulable and schedulable
            logger.info(f"  Component {comp_id}: {'Schedulable' if schedulable else 'Not schedulable'}")
    
    for core_id, results in analysis_results.items():
        if core_id in system.cores:
            schedulable = results['schedulable']
            all_schedulable = all_schedulable and schedulable
            logger.info(f"  Core {core_id}: {'Schedulable' if schedulable else 'Not schedulable'}")
    
    logger.info(f"\nOverall System: {'Schedulable' if all_schedulable else 'Not schedulable'}")
    logger.info(f"Results saved to {output_path} and {detailed_report_path}")
    

if __name__ == "__main__":
    main()