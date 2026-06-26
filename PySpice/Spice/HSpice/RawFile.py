import logging
import numpy as np
from PySpice.Unit import u_V, u_A, u_s, u_Hz, u_Degree
from PySpice.Probe.WaveForm import (
    TransientAnalysis,
    DcAnalysis,
    AcAnalysis,
    WaveForm,
    OperatingPoint,
)

class AnalysisList(list):
  def __init__(self, analyses, measurements=None):
    """
    Initialize the list with analyses and an optional measurement mapping.
    
    Parameters:
    	analyses: Initial analysis objects.
    	measurements: Mapping of measurement names to values.
    """
    super().__init__(analyses)
    self._measurements = measurements or {}

  @property
  def measurements(self):
    """
    Measurement mapping associated with the analysis list.
    
    Returns:
        dict: The stored measurement values.
    """
    return self._measurements

  def __getattr__(self, name):
    """
    Return a measurement by attribute name.
    
    Parameters:
    	name (str): The measurement name to look up.
    
    Returns:
    	The measurement value for the exact or lowercased name.
    """
    if name in self._measurements:
      return self._measurements[name]
    if name.lower() in self._measurements:
      return self._measurements[name.lower()]
    raise AttributeError(f"'AnalysisList' object has no attribute '{name}'")

  def __getitem__(self, item):
    """
    Return a list item or a measurement by name.
    
    Parameters:
    	item: An index, slice, or measurement name.
    
    Returns:
    	The selected analysis item or measurement value.
    
    Raises:
    	IndexError: If the measurement name is not found.
    	KeyError: If item is not an index, slice, or string.
    """
    if isinstance(item, (int, slice)):
      return super().__getitem__(item)
    if isinstance(item, str):
      if item in self._measurements:
        return self._measurements[item]
      if item.lower() in self._measurements:
        return self._measurements[item.lower()]
      raise IndexError(item)
    raise KeyError(item)


_module_logger = logging.getLogger(__name__)


class HSpiceRawFile:
    def __init__(
        self, data, simulation=None, measurements=None, op_nodes=None, op_branches=None, analysis_type=None
    ):
        """
        Initialize a wrapper for parsed HSPICE raw data and related metadata.
        
        Parameters:
            data: Parsed raw simulation data.
            simulation: Associated simulation object.
            measurements: Mapping of measurement names to values.
            op_nodes: Operating-point node values.
            op_branches: Operating-point branch values.
            analysis_type: HSPICE analysis type code.
        """
        self.data = data
        self._simulation = simulation
        self.measurements = measurements or {}
        self.op_nodes = op_nodes or {}
        self.op_branches = op_branches or {}
        self._analysis_type = analysis_type

    @property
    def simulation(self):
        """
        Return the associated simulation.
        
        Returns:
        	simulation: The stored simulation object.
        """
        return self._simulation

    @simulation.setter
    def simulation(self, value):
        """
        Set the associated simulation.
        
        Parameters:
        	value: The simulation to associate with this raw file.
        """
        self._simulation = value

    def to_analysis(self):
        """
        Convert the parsed HSPICE raw content into PySpice analysis objects.
        
        Returns:
        	An OperatingPoint, AcAnalysis, DcAnalysis, or TransientAnalysis instance, or an AnalysisList when the raw file contains multiple sweeps.
        """
        if self.data is None and self._analysis_type == "o":
            # This is an operating point simulation!
            nodes = [
                WaveForm.from_unit_values(name, u_V(np.array([val])))
                for name, val in self.op_nodes.items()
            ]
            branches = [
                WaveForm.from_unit_values(name, u_A(np.array([val])))
                for name, val in self.op_branches.items()
            ]
            return OperatingPoint(
                simulation=self.simulation, nodes=nodes, branches=branches
            )

        # data is a list of sweeps returned by hspice_read
        # data[0] is sweeps tuple (sweep, sweepValues, dataList)
        sweeps = self.data[0][0]
        scale_name_outer, sweep_values, data_list = sweeps

        if len(data_list) == 0:
            analysis_type = self._analysis_type
            if not analysis_type and self.simulation and hasattr(self.simulation, "_analyses"):
                analyses_keys = set(self.simulation._analyses.keys())
                if "ac" in analyses_keys:
                    analysis_type = "a"
                elif "dc" in analyses_keys:
                    analysis_type = "s"
                elif "op" in analyses_keys:
                    analysis_type = "o"
                elif "tran" in analyses_keys:
                    analysis_type = "t"

            analysis_type = analysis_type or "t"

            if analysis_type == "a":
                analysis = AcAnalysis(
                    simulation=self.simulation,
                    frequency=WaveForm.from_array("frequency", np.array([])),
                    nodes=[],
                    branches=[],
                    internal_parameters=[],
                )
            elif analysis_type == "s":
                analysis = DcAnalysis(
                    simulation=self.simulation,
                    sweep=WaveForm.from_array("sweep", np.array([])),
                    nodes=[],
                    branches=[],
                    internal_parameters=[],
                )
            elif analysis_type == "o":
                analysis = OperatingPoint(
                    simulation=self.simulation,
                    nodes=[],
                    branches=[],
                )
            else:
                analysis = TransientAnalysis(
                    simulation=self.simulation,
                    time=WaveForm.from_array("time", np.array([])),
                    nodes=[],
                    branches=[],
                    internal_parameters=[],
                )
            analysis._measurements = self.measurements
            return analysis


        analyses = []
        for res_dict in data_list:
            # Let's find the scale/abscissa variable (typically TIME or FREQUENCY or HERTZ)
            scale_name = None
            for k in res_dict.keys():
                if k.upper() in ("TIME", "FREQUENCY", "HERTZ"):
                    scale_name = k
                    break

            if scale_name is None:
                # Check if there is another scale variable (e.g. dc sweep)
                scale_name = list(res_dict.keys())[0]

            scale_values = res_dict[scale_name]

            # Determine analysis type
            if scale_name.upper() == "TIME":
                analysis_type = "transient"
                scale_unit = u_s
            elif scale_name.upper() in ("FREQUENCY", "HERTZ"):
                analysis_type = "ac"
                scale_unit = u_Hz
            else:
                analysis_type = "dc"
                scale_name_upper = scale_name.upper()
                if scale_name_upper == "VOLTS":
                    scale_unit = u_V
                elif scale_name_upper == "AMPS":
                    scale_unit = u_A
                elif scale_name_upper == "DEG_C":
                    scale_unit = u_Degree
                else:
                    scale_unit = None

            # Build WaveForms
            abscissa_name = scale_name.lower()
            if analysis_type == "ac":
                abscissa_name = "frequency"

            if scale_unit:
                abscissa_waveform = WaveForm.from_unit_values(
                    abscissa_name, scale_unit(scale_values)
                )
            else:
                abscissa_waveform = WaveForm.from_array(abscissa_name, scale_values)

            nodes = []
            branches = []

            for k, v in res_dict.items():
                if k == scale_name:
                    continue

                # Determine if it's a current or voltage
                if k.lower().startswith("i("):
                    # Simplify name: strip i( and trailing )
                    simplified_name = k[2:]
                    if simplified_name.endswith(")"):
                        simplified_name = simplified_name[:-1]
                    branches.append(
                        WaveForm.from_unit_values(
                            simplified_name, u_A(v), abscissa=abscissa_waveform
                        )
                    )
                else:
                    simplified_name = k
                    nodes.append(
                        WaveForm.from_unit_values(
                            simplified_name, u_V(v), abscissa=abscissa_waveform
                        )
                    )

            if analysis_type == "transient":
                analysis = TransientAnalysis(
                    simulation=self.simulation,
                    time=abscissa_waveform,
                    nodes=nodes,
                    branches=branches,
                    internal_parameters=[],
                )
            elif analysis_type == "ac":
                analysis = AcAnalysis(
                    simulation=self.simulation,
                    frequency=abscissa_waveform,
                    nodes=nodes,
                    branches=branches,
                    internal_parameters=[],
                )
            else:
                analysis = DcAnalysis(
                    simulation=self.simulation,
                    sweep=abscissa_waveform,
                    nodes=nodes,
                    branches=branches,
                    internal_parameters=[],
                )

            analysis._measurements = self.measurements
            analyses.append(analysis)

        if len(analyses) == 1:
            return analyses[0]
        return AnalysisList(analyses, self.measurements)
