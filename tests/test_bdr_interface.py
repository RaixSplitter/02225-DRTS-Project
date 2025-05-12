import pytest
from simsystem.analytical_tools import bdr_interface
from simsystem.input_model import Component, Core


def test_bdr_interface():
            component = Component(
            component_id="Camera_Sensor",
            scheduler="RM",
            budget=84,
            period=84,
            core_id="Core_1",
            priority=0,
        ),

    # Test the BDR interface
    result = bdr_interface(component)

    assert isinstance(result, tuple)
    assert result == (1, 0)


