import pytest
from simsystem.input_model import Input_Model
from simsystem.objects import Core, Component, Task

def test_read_architecture():
    # Mock file path and expected output
    filepath = "test-cases/1-tiny-test-case/architecture.csv"
    expected_output = {
        'Core_1': Core(core_id='Core_1', speed_factor=0.62, scheduler='RM')
    }
    
    # Mock the method behavior (if needed)
    # Example: Use a library like unittest.mock to mock file reading
    
    # Call the method
    result = Input_Model.read_architecture(filepath)
    
    # Assert the result
    assert isinstance(result, dict)
    assert result == expected_output

def test_read_budgets():
    # Mock file path and expected output
    filepath = "test-cases/1-tiny-test-case/budgets.csv"
    expected_output = {
        'Camera_Sensor': Component(
            component_id='Camera_Sensor',
            scheduler='RM',
            budget=84,
            period=84,
            core_id='Core_1',
            priority=0
        )
    }
    
    # Mock the method behavior (if needed)
    
    # Call the method
    result = Input_Model.read_budgets(filepath)
    
    # Assert the result
    assert isinstance(result, dict)
    assert result == expected_output

def test_read_tasks():
    # Mock file path and expected output
    filepath = "test-cases/1-tiny-test-case/tasks.csv"
    expected_output = {
        'Task_0': Task(
            name='Task_0',
            wcet=14,
            period=50,
            deadline_interval=50,
            component_id='Camera_Sensor',
            priority=0,
            last_released=0
        ),
        'Task_1': Task(
            name='Task_1',
            wcet=33,
            period=100,
            deadline_interval=100,
            component_id='Camera_Sensor',
            priority=1,
            last_released=0
        )
    }
    
    # Mock the method behavior (if needed)
    
    # Call the method
    result = Input_Model.read_tasks(filepath)
    
    # Assert the result
    assert isinstance(result, dict)
    assert result == expected_output