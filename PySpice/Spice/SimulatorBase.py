###################################################################################################
#
# PySpice - A Spice Package for Python
# Copyright (C) 2021 Fabrice Salvaire
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
####################################################################################################

__all__ = ["Simulator"]

import logging

from ..Config import ConfigInstall
from .Simulation import Simulation

_module_logger = logging.getLogger(__name__)


class Simulator:
    """Base class to implement a simulator."""

    _logger = _module_logger.getChild("Simulator")

    #: Define the default simulator
    DEFAULT_SIMULATOR = None
    if ConfigInstall.OS.on_windows:
        DEFAULT_SIMULATOR = "ngspice-shared"
    else:
        DEFAULT_SIMULATOR = "ngspice-shared"

    SIMULATOR = None  # for subclass
    _SIMULATOR_CLASSES = {}

    @classmethod
    def register_simulator_class(cls, name, sub_cls):
        """
        Register a simulator implementation under a name.
        
        Parameters:
        	name (str): The simulator name used for lookup.
        	sub_cls (type): The simulator subclass to register.
        """
        cls._SIMULATOR_CLASSES[name] = sub_cls

    @classmethod
    def factory(cls, *args, **kwargs):
        """
        Instantiate a registered simulator backend.
        
        Parameters:
            simulator (str): Name of the simulator backend to create.
        
        Returns:
            Simulator: An instance of the registered simulator subclass.
        
        Raises:
            NameError: If the requested simulator is not registered.
        """
        simulator = kwargs.pop("simulator", cls.DEFAULT_SIMULATOR)

        if simulator not in cls._SIMULATOR_CLASSES:
            raise NameError(f"Unknown simulator {simulator}")

        sub_cls = cls._SIMULATOR_CLASSES[simulator]

        obj = sub_cls(*args, simulator=simulator, **kwargs)
        obj._AS_SIMULATOR = simulator
        return obj

    def __getstate__(self):
        # Pickle: protection for cffi
        """
        Return the class name for pickling support.
        
        Returns:
        	class_name (str): The simulator class name.
        """
        return self.__class__.__name__

    def simulation(self, circuit, **kwargs):
        """
        Create a simulation for a circuit.
        
        Parameters:
        	circuit: Circuit to simulate.
        
        Returns:
        	Simulation: A simulation configured for the circuit.
        """
        # Note: simulation is simulator dependent, thus subclass this method if needed
        return Simulation(self, circuit, **kwargs)

    @property
    def name(self):
        """
        Name of the selected simulator backend.
        
        Returns:
        	simulator_name (str): The registered simulator name associated with this instance.
        """
        return self._AS_SIMULATOR

    @property
    def version(self):
        """
        Return the simulator version.
        
        Returns:
        	str: The simulator version string.
        
        Raises:
        	NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def customise(self, simulation):
        """
        Customize the simulation before it runs.
        
        Parameters:
            simulation: The simulation instance to customize.
        """
        pass

    def run(self, simulation):
        """
        Run the simulation and return the waveforms.
        
        Parameters:
        	simulation: The simulation to execute.
        """
        raise NotImplementedError
