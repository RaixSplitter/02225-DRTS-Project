from abc import ABC, abstractmethod
from objects import Job

from objects import Component

class ResourceSupplier(ABC):
    """
    Abstract base class for resource suppliers.
    """
    @abstractmethod
    def supply_resource(self, component: Component, time: float, time_slice: float) -> float:
        """
        Calculate the amount of resource to supply to a component.
        
        Args:
            component: Component to supply resources to.
            time: Current simulation time.
            time_slice: Time slice duration.
            
        Returns:
            Amount of resource supplied.
        """
        pass



class PRMResourceSupplier(ResourceSupplier):
    """
    Periodic Resource Model resource supplier.
    """
    def supply_resource(self, component: Component, time: float, time_slice: float) -> float:
        """
        Simulate resource supply for a component based on its PRM parameters.
        
        Args:
            component: Component to supply resources to.
            time: Current simulation time.
            time_slice: Time slice duration.
            
        Returns:
            Amount of resource supplied.
        """
        budget = component.budget
        period = component.period
        
        # Calculate time within the current period
        time_in_period = time % period
        
        # Simplified supply model: resources are available at the beginning of each period
        if time_in_period < budget:
            return min(budget - time_in_period, time_slice)
        return 0



class BDRResourceSupplier(ResourceSupplier):
    """
    Bounded Delay Resource model resource supplier.
    """
    def supply_resource(self, component: Component, time: float, time_slice: float) -> float:
        """
        Simulate resource supply for a component based on its BDR parameters.
        
        Args:
            component: Component to supply resources to.
            time: Current simulation time.
            time_slice: Time slice duration.
            
        Returns:
            Amount of resource supplied.
        """
        if component.bdr_interface is None:
            # Fall back to PRM if BDR interface is not set
            prm_supplier = PRMResourceSupplier()
            return prm_supplier.supply_resource(component, time, time_slice)
            
        alpha, delta = component.bdr_interface
        
        # Simplified BDR supply model
        if time > delta:
            return alpha * time_slice
        return 0
