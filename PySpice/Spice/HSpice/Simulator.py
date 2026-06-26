import logging
from ..SimulatorBase import Simulator
from .Server import HSpiceServer

_module_logger = logging.getLogger(__name__)


class HSpiceSimulator(Simulator):
    _logger = _module_logger.getChild("HSpiceSimulator")
    SIMULATOR = "hspice"

    def __init__(self, **kwargs):
        """
        Initialize the simulator with an HSpice server backend.
        
        Parameters:
        	spice_command (str): Command used to launch HSpice.
        	concurrency_limit (int): Maximum number of concurrent server requests.
        """
        server_kwargs = {
            x: kwargs[x] for x in ("spice_command", "concurrency_limit") if x in kwargs
        }
        self._spice_server = HSpiceServer(**server_kwargs)

    @property
    def version(self):
        """
        Provide the simulator version string.
        
        Returns:
        	str: The simulator version string.
        """
        return ""

    def customise(self, simulation):
        # Add post option to simulation options if not present
        """
        Enable post-processing for the simulation.
        
        Parameters:
        	simulation: The simulation to modify.
        """
        simulation.options(post=1)

    def run(self, simulation, *args, **kwargs):
        """
        Run a simulation and return its analysis results.
        
        Parameters:
        	simulation: The simulation to execute.
        
        Returns:
        	analysis: The analysis results produced from the simulator output.
        """
        raw_file = self._spice_server(spice_input=str(simulation))
        raw_file.simulation = simulation
        return raw_file.to_analysis()
