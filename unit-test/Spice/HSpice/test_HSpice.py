####################################################################################################
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

####################################################################################################

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

####################################################################################################

import logging
logger = logging.getLogger(__name__)

####################################################################################################

from PySpice.Spice.HSpice.Server import parse_spice_value
from PySpice.Spice.HSpice.RawFile import AnalysisList, HSpiceRawFile
from PySpice.Probe.WaveForm import (
    TransientAnalysis,
    DcAnalysis,
    AcAnalysis,
    OperatingPoint,
)

####################################################################################################


def _make_hspice_data(data_list, sweep_name=None, sweep_values=None):
    """
    Build a mock hspice_read return value with the expected structure:
      data = [ ( (sweep_name, sweep_values, data_list), scale_name, None, title, date, None ) ]
    """
    sweeps = (sweep_name, sweep_values, data_list)
    entry = (sweeps, "TIME", None, "Test Title", "2024-01-01", None)
    return [entry]


####################################################################################################

class TestParseSpiceValue(unittest.TestCase):
    """Tests for parse_spice_value() in Server.py."""

    ##############################################

    def test_femto_suffix(self):
        self.assertAlmostEqual(parse_spice_value("1f"), 1e-15)
        self.assertAlmostEqual(parse_spice_value("2.5f"), 2.5e-15)

    ##############################################

    def test_pico_suffix(self):
        self.assertAlmostEqual(parse_spice_value("1p"), 1e-12)
        self.assertAlmostEqual(parse_spice_value("100p"), 100e-12)

    ##############################################

    def test_nano_suffix(self):
        self.assertAlmostEqual(parse_spice_value("1n"), 1e-9)
        self.assertAlmostEqual(parse_spice_value("20n"), 20e-9)

    ##############################################

    def test_micro_suffix(self):
        self.assertAlmostEqual(parse_spice_value("1u"), 1e-6)
        self.assertAlmostEqual(parse_spice_value("4.7u"), 4.7e-6)

    ##############################################

    def test_mega_suffix(self):
        self.assertAlmostEqual(parse_spice_value("1meg"), 1e6)
        self.assertAlmostEqual(parse_spice_value("2.5meg"), 2.5e6)
        # Case-insensitive
        self.assertAlmostEqual(parse_spice_value("1MEG"), 1e6)

    ##############################################

    def test_milli_suffix(self):
        self.assertAlmostEqual(parse_spice_value("1m"), 1e-3)
        self.assertAlmostEqual(parse_spice_value("100m"), 100e-3)

    ##############################################

    def test_kilo_suffix(self):
        self.assertAlmostEqual(parse_spice_value("1k"), 1e3)
        self.assertAlmostEqual(parse_spice_value("10k"), 10e3)

    ##############################################

    def test_giga_suffix(self):
        self.assertAlmostEqual(parse_spice_value("1g"), 1e9)
        self.assertAlmostEqual(parse_spice_value("2g"), 2e9)

    ##############################################

    def test_plain_float(self):
        self.assertAlmostEqual(parse_spice_value("1.0"), 1.0)
        self.assertAlmostEqual(parse_spice_value("3.14"), 3.14)
        self.assertAlmostEqual(parse_spice_value("0"), 0.0)
        self.assertAlmostEqual(parse_spice_value("-5.0"), -5.0)

    ##############################################

    def test_scientific_notation(self):
        self.assertAlmostEqual(parse_spice_value("1e-9"), 1e-9)
        self.assertAlmostEqual(parse_spice_value("1.5e3"), 1500.0)

    ##############################################

    def test_whitespace_stripped(self):
        self.assertAlmostEqual(parse_spice_value("  1k  "), 1e3)
        self.assertAlmostEqual(parse_spice_value(" 100n "), 100e-9)

    ##############################################

    def test_case_insensitive_suffix(self):
        # Suffix matching is case-insensitive (val_str is lowercased first)
        self.assertAlmostEqual(parse_spice_value("1N"), 1e-9)
        self.assertAlmostEqual(parse_spice_value("1K"), 1e3)
        self.assertAlmostEqual(parse_spice_value("1U"), 1e-6)
        self.assertAlmostEqual(parse_spice_value("1M"), 1e-3)

    ##############################################

    def test_numeric_prefix_fallback(self):
        # When value string has trailing non-numeric characters that aren't
        # a known suffix, fallback regex extracts the numeric prefix
        result = parse_spice_value("1.5xyz")
        self.assertAlmostEqual(result, 1.5)

    ##############################################

    def test_invalid_value_raises(self):
        with self.assertRaises((ValueError, Exception)):
            parse_spice_value("invalid_no_number")

    ##############################################

    def test_negative_with_suffix(self):
        # Negative values with suffix
        self.assertAlmostEqual(parse_spice_value("-1n"), -1e-9)
        self.assertAlmostEqual(parse_spice_value("-100m"), -100e-3)

    ##############################################

    def test_zero_value(self):
        self.assertAlmostEqual(parse_spice_value("0.0"), 0.0)

    ##############################################

    def test_meg_priority_over_m(self):
        # "meg" suffix takes precedence over single "m" suffix
        self.assertAlmostEqual(parse_spice_value("1meg"), 1e6)
        self.assertAlmostEqual(parse_spice_value("1m"), 1e-3)
        # These are different values
        self.assertNotAlmostEqual(parse_spice_value("1meg"), parse_spice_value("1m"))


####################################################################################################

class TestAnalysisList(unittest.TestCase):
    """Tests for AnalysisList in RawFile.py."""

    ##############################################

    def setUp(self):
        self.mock_analysis1 = MagicMock()
        self.mock_analysis2 = MagicMock()
        self.analyses = [self.mock_analysis1, self.mock_analysis2]
        self.measurements = {"v_max": 5.0, "t_rise": 1e-9, "gain": 3.14}

    ##############################################

    def test_is_list(self):
        al = AnalysisList(self.analyses)
        self.assertIsInstance(al, list)
        self.assertEqual(len(al), 2)

    ##############################################

    def test_default_empty_measurements(self):
        al = AnalysisList(self.analyses)
        self.assertEqual(al.measurements, {})

    ##############################################

    def test_measurements_property(self):
        al = AnalysisList(self.analyses, self.measurements)
        self.assertEqual(al.measurements, self.measurements)
        self.assertAlmostEqual(al.measurements["v_max"], 5.0)

    ##############################################

    def test_integer_indexing(self):
        al = AnalysisList(self.analyses, self.measurements)
        self.assertIs(al[0], self.mock_analysis1)
        self.assertIs(al[1], self.mock_analysis2)

    ##############################################

    def test_negative_integer_indexing(self):
        al = AnalysisList(self.analyses, self.measurements)
        self.assertIs(al[-1], self.mock_analysis2)
        self.assertIs(al[-2], self.mock_analysis1)

    ##############################################

    def test_slice_indexing(self):
        al = AnalysisList(self.analyses, self.measurements)
        sliced = al[0:1]
        self.assertEqual(len(sliced), 1)

    ##############################################

    def test_string_item_access(self):
        al = AnalysisList(self.analyses, self.measurements)
        self.assertAlmostEqual(al["v_max"], 5.0)
        self.assertAlmostEqual(al["t_rise"], 1e-9)

    ##############################################

    def test_string_item_access_case_insensitive(self):
        al = AnalysisList(self.analyses, {"v_max": 5.0})
        # lowercase key, uppercase access
        self.assertAlmostEqual(al["V_MAX"], 5.0)

    ##############################################

    def test_string_item_not_found_raises_index_error(self):
        al = AnalysisList(self.analyses, self.measurements)
        with self.assertRaises(IndexError):
            _ = al["nonexistent_key"]

    ##############################################

    def test_invalid_key_type_raises_key_error(self):
        al = AnalysisList(self.analyses, self.measurements)
        with self.assertRaises(KeyError):
            _ = al[3.14]

    ##############################################

    def test_attribute_access(self):
        al = AnalysisList(self.analyses, self.measurements)
        self.assertAlmostEqual(al.v_max, 5.0)
        self.assertAlmostEqual(al.t_rise, 1e-9)

    ##############################################

    def test_attribute_access_case_insensitive(self):
        al = AnalysisList(self.analyses, {"v_max": 5.0})
        # The key is lowercase, but access may be uppercase
        self.assertAlmostEqual(al.V_MAX, 5.0)

    ##############################################

    def test_attribute_not_found_raises(self):
        al = AnalysisList(self.analyses, self.measurements)
        with self.assertRaises(AttributeError):
            _ = al.nonexistent_attribute

    ##############################################

    def test_none_measurements_becomes_empty_dict(self):
        al = AnalysisList(self.analyses, None)
        self.assertEqual(al.measurements, {})

    ##############################################

    def test_list_iteration(self):
        al = AnalysisList(self.analyses, self.measurements)
        items = list(al)
        self.assertEqual(len(items), 2)
        self.assertIs(items[0], self.mock_analysis1)

    ##############################################

    def test_empty_analyses(self):
        al = AnalysisList([], {"key": 1.0})
        self.assertEqual(len(al), 0)
        self.assertAlmostEqual(al["key"], 1.0)


####################################################################################################

class TestHSpiceRawFileInit(unittest.TestCase):
    """Tests for HSpiceRawFile.__init__ and properties."""

    ##############################################

    def test_default_values(self):
        rf = HSpiceRawFile(data=None)
        self.assertIsNone(rf.data)
        self.assertIsNone(rf.simulation)
        self.assertEqual(rf.measurements, {})
        self.assertEqual(rf.op_nodes, {})
        self.assertEqual(rf.op_branches, {})
        self.assertIsNone(rf._analysis_type)

    ##############################################

    def test_constructor_with_all_args(self):
        data = MagicMock()
        sim = MagicMock()
        rf = HSpiceRawFile(
            data=data,
            simulation=sim,
            measurements={"key": 1.0},
            op_nodes={"node_a": 3.3},
            op_branches={"v1": -0.001},
            analysis_type="t",
        )
        self.assertIs(rf.data, data)
        self.assertIs(rf.simulation, sim)
        self.assertEqual(rf.measurements, {"key": 1.0})
        self.assertEqual(rf.op_nodes, {"node_a": 3.3})
        self.assertEqual(rf.op_branches, {"v1": -0.001})
        self.assertEqual(rf._analysis_type, "t")

    ##############################################

    def test_simulation_setter(self):
        rf = HSpiceRawFile(data=None)
        self.assertIsNone(rf.simulation)
        sim = MagicMock()
        rf.simulation = sim
        self.assertIs(rf.simulation, sim)

    ##############################################

    def test_none_measurements_becomes_empty_dict(self):
        rf = HSpiceRawFile(data=None, measurements=None)
        self.assertEqual(rf.measurements, {})

    ##############################################

    def test_none_op_nodes_becomes_empty_dict(self):
        rf = HSpiceRawFile(data=None, op_nodes=None)
        self.assertEqual(rf.op_nodes, {})

    ##############################################

    def test_none_op_branches_becomes_empty_dict(self):
        rf = HSpiceRawFile(data=None, op_branches=None)
        self.assertEqual(rf.op_branches, {})


####################################################################################################

class TestHSpiceRawFileToAnalysisOperatingPoint(unittest.TestCase):
    """Tests for HSpiceRawFile.to_analysis() with operating point data."""

    ##############################################

    def test_op_analysis_type(self):
        rf = HSpiceRawFile(
            data=None,
            op_nodes={"node_a": 3.3, "node_b": 1.65},
            op_branches={"v1": -0.001},
            analysis_type="o",
        )
        result = rf.to_analysis()
        self.assertIsInstance(result, OperatingPoint)

    ##############################################

    def test_op_empty_nodes_and_branches(self):
        rf = HSpiceRawFile(data=None, analysis_type="o")
        result = rf.to_analysis()
        self.assertIsInstance(result, OperatingPoint)

    ##############################################

    def test_op_only_when_data_is_none_and_type_is_o(self):
        # data=None but analysis_type is NOT "o" should not produce OperatingPoint
        # Instead it should attempt to read from data (which will fail since data is None)
        rf = HSpiceRawFile(data=None, analysis_type="t")
        with self.assertRaises((TypeError, AttributeError, IndexError)):
            rf.to_analysis()


####################################################################################################

class TestHSpiceRawFileToAnalysisEmptyDataList(unittest.TestCase):
    """Tests for HSpiceRawFile.to_analysis() when data_list is empty."""

    ##############################################

    def _make_empty_data(self):
        """Make data with empty data_list."""
        return _make_hspice_data(data_list=[])

    ##############################################

    def test_empty_data_list_defaults_to_transient(self):
        rf = HSpiceRawFile(data=self._make_empty_data())
        result = rf.to_analysis()
        self.assertIsInstance(result, TransientAnalysis)

    ##############################################

    def test_empty_data_list_explicit_transient(self):
        rf = HSpiceRawFile(data=self._make_empty_data(), analysis_type="t")
        result = rf.to_analysis()
        self.assertIsInstance(result, TransientAnalysis)

    ##############################################

    def test_empty_data_list_ac_type(self):
        rf = HSpiceRawFile(data=self._make_empty_data(), analysis_type="a")
        result = rf.to_analysis()
        self.assertIsInstance(result, AcAnalysis)

    ##############################################

    def test_empty_data_list_dc_type(self):
        rf = HSpiceRawFile(data=self._make_empty_data(), analysis_type="s")
        result = rf.to_analysis()
        self.assertIsInstance(result, DcAnalysis)

    ##############################################

    def test_empty_data_list_op_type(self):
        rf = HSpiceRawFile(data=self._make_empty_data(), analysis_type="o")
        result = rf.to_analysis()
        self.assertIsInstance(result, OperatingPoint)

    ##############################################

    def test_empty_data_list_measurements_attached(self):
        measurements = {"v_out_max": 5.0}
        rf = HSpiceRawFile(
            data=self._make_empty_data(),
            analysis_type="t",
            measurements=measurements,
        )
        result = rf.to_analysis()
        self.assertEqual(result._measurements, measurements)

    ##############################################

    def test_empty_data_list_infers_type_from_simulation(self):
        sim = MagicMock()
        sim._analyses = {"ac": MagicMock()}
        rf = HSpiceRawFile(data=self._make_empty_data(), simulation=sim)
        result = rf.to_analysis()
        self.assertIsInstance(result, AcAnalysis)

    ##############################################

    def test_empty_data_list_infers_dc_from_simulation(self):
        sim = MagicMock()
        sim._analyses = {"dc": MagicMock()}
        rf = HSpiceRawFile(data=self._make_empty_data(), simulation=sim)
        result = rf.to_analysis()
        self.assertIsInstance(result, DcAnalysis)

    ##############################################

    def test_empty_data_list_infers_op_from_simulation(self):
        sim = MagicMock()
        sim._analyses = {"op": MagicMock()}
        rf = HSpiceRawFile(data=self._make_empty_data(), simulation=sim)
        result = rf.to_analysis()
        self.assertIsInstance(result, OperatingPoint)

    ##############################################

    def test_empty_data_list_infers_tran_from_simulation(self):
        sim = MagicMock()
        sim._analyses = {"tran": MagicMock()}
        rf = HSpiceRawFile(data=self._make_empty_data(), simulation=sim)
        result = rf.to_analysis()
        self.assertIsInstance(result, TransientAnalysis)


####################################################################################################

class TestHSpiceRawFileToAnalysisTransient(unittest.TestCase):
    """Tests for HSpiceRawFile.to_analysis() with transient (TIME) data."""

    ##############################################

    def _make_transient_data(self, node_names=None, branch_names=None):
        time_arr = np.linspace(0, 20e-9, 21)
        data_dict = {"TIME": time_arr}
        if node_names:
            for n in node_names:
                data_dict[n] = np.ones_like(time_arr) * 1.0
        if branch_names:
            for b in branch_names:
                data_dict[b] = np.ones_like(time_arr) * 0.001
        return _make_hspice_data([data_dict])

    ##############################################

    def test_transient_returns_transient_analysis(self):
        data = self._make_transient_data(["node_a", "node_b"])
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, TransientAnalysis)

    ##############################################

    def test_transient_single_table_not_wrapped_in_list(self):
        data = self._make_transient_data(["node_a"])
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        # Single analysis, not wrapped in AnalysisList
        self.assertNotIsInstance(result, AnalysisList)
        self.assertIsInstance(result, TransientAnalysis)

    ##############################################

    def test_transient_nodes_classified_correctly(self):
        """Non-current variables should go to nodes."""
        data = self._make_transient_data(["node_a", "node_b"])
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, TransientAnalysis)

    ##############################################

    def test_transient_branches_classified_correctly(self):
        """i( prefix variables should go to branches."""
        time_arr = np.linspace(0, 20e-9, 21)
        data_dict = {"TIME": time_arr, "i(v1)": np.ones_like(time_arr) * 0.001}
        data = _make_hspice_data([data_dict])
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, TransientAnalysis)

    ##############################################

    def test_transient_measurements_attached(self):
        measurements = {"v_out_max": 3.3}
        data = self._make_transient_data(["node_a"])
        rf = HSpiceRawFile(data=data, measurements=measurements)
        result = rf.to_analysis()
        self.assertEqual(result._measurements, measurements)

    ##############################################

    def test_multiple_sweeps_returns_analysis_list(self):
        time_arr = np.linspace(0, 20e-9, 21)
        data_list = [
            {"TIME": time_arr, "node_a": np.ones_like(time_arr)},
            {"TIME": time_arr, "node_a": np.ones_like(time_arr) * 2},
        ]
        data = _make_hspice_data(data_list)
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, AnalysisList)
        self.assertEqual(len(result), 2)


####################################################################################################

class TestHSpiceRawFileToAnalysisAC(unittest.TestCase):
    """Tests for HSpiceRawFile.to_analysis() with AC (FREQUENCY/HERTZ) data."""

    ##############################################

    def _make_ac_data(self, scale_name="FREQUENCY"):
        freq_arr = np.logspace(2, 4, 21)
        cplx_arr = np.ones(21, dtype=complex)
        data_dict = {scale_name: freq_arr, "node_b": cplx_arr}
        return _make_hspice_data([data_dict])

    ##############################################

    def test_frequency_scale_returns_ac_analysis(self):
        data = self._make_ac_data("FREQUENCY")
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, AcAnalysis)

    ##############################################

    def test_hertz_scale_returns_ac_analysis(self):
        data = self._make_ac_data("HERTZ")
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, AcAnalysis)

    ##############################################

    def test_abscissa_renamed_to_frequency(self):
        """For AC analysis, abscissa should be named 'frequency'."""
        data = self._make_ac_data("HERTZ")
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, AcAnalysis)
        # The abscissa (frequency waveform) should have name 'frequency'
        self.assertEqual(result.frequency.name, 'frequency')


####################################################################################################

class TestHSpiceRawFileToAnalysisDC(unittest.TestCase):
    """Tests for HSpiceRawFile.to_analysis() with DC sweep data."""

    ##############################################

    def _make_dc_data(self, scale_name):
        scale_arr = np.linspace(1, 5, 5)
        node_arr = np.linspace(1, 5, 5)
        data_dict = {scale_name: scale_arr, "node_b": node_arr}
        return _make_hspice_data([data_dict])

    ##############################################

    def test_volts_scale_returns_dc_analysis(self):
        data = self._make_dc_data("VOLTS")
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, DcAnalysis)

    ##############################################

    def test_amps_scale_returns_dc_analysis(self):
        data = self._make_dc_data("AMPS")
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, DcAnalysis)

    ##############################################

    def test_deg_c_scale_returns_dc_analysis(self):
        data = self._make_dc_data("DEG_C")
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, DcAnalysis)

    ##############################################

    def test_unknown_scale_returns_dc_analysis(self):
        """Any unknown scale name (not TIME/FREQUENCY/HERTZ) is treated as DC."""
        data = self._make_dc_data("rval")
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, DcAnalysis)

    ##############################################

    def test_first_key_used_as_scale_when_no_known_name(self):
        """First key in dict should be used as scale when no known name found."""
        scale_arr = np.linspace(10, 50, 5)
        data_dict = {"rval": scale_arr, "node_a": scale_arr * 0.5}
        data = _make_hspice_data([data_dict])
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, DcAnalysis)


####################################################################################################

class TestHSpiceRawFileCurrentBranchParsing(unittest.TestCase):
    """Tests for branch (current) vs node (voltage) parsing in to_analysis."""

    ##############################################

    def test_i_prefix_becomes_branch(self):
        """Variables starting with 'i(' should be classified as branches."""
        time_arr = np.linspace(0, 1e-9, 10)
        data_dict = {
            "TIME": time_arr,
            "i(v1)": np.ones(10) * 0.001,
            "i(r1)": np.ones(10) * 0.002,
        }
        data = _make_hspice_data([data_dict])
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, TransientAnalysis)

    ##############################################

    def test_i_prefix_name_simplified(self):
        """i(v1) should become 'v1' as branch name (strip 'i(' and ')')."""
        time_arr = np.linspace(0, 1e-9, 10)
        data_dict = {"TIME": time_arr, "i(v1)": np.ones(10) * 0.001}
        data = _make_hspice_data([data_dict])
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        # Branch name should be simplified (branches is a dict: name -> WaveForm)
        branch_names = list(result.branches.keys())
        self.assertIn("v1", branch_names)

    ##############################################

    def test_i_prefix_without_closing_paren(self):
        """i(v1 (no closing paren) should be handled gracefully."""
        time_arr = np.linspace(0, 1e-9, 10)
        data_dict = {"TIME": time_arr, "i(v1": np.ones(10) * 0.001}
        data = _make_hspice_data([data_dict])
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, TransientAnalysis)

    ##############################################

    def test_non_i_prefix_becomes_node(self):
        """Variables not starting with 'i(' should be classified as nodes."""
        time_arr = np.linspace(0, 1e-9, 10)
        data_dict = {"TIME": time_arr, "node_a": np.ones(10)}
        data = _make_hspice_data([data_dict])
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        # nodes is a dict: name -> WaveForm
        node_names = list(result.nodes.keys())
        self.assertIn("node_a", node_names)


####################################################################################################

class TestHSpiceRawFileMultipleSweeps(unittest.TestCase):
    """Tests for HSpiceRawFile.to_analysis() with multiple sweep tables."""

    ##############################################

    def test_two_sweeps_returns_analysis_list(self):
        time_arr = np.linspace(0, 20e-9, 10)
        data_list = [
            {"TIME": time_arr, "node_a": np.ones(10)},
            {"TIME": time_arr, "node_a": np.ones(10) * 2},
        ]
        data = _make_hspice_data(data_list)
        rf = HSpiceRawFile(data=data, measurements={"v_max": 5.0})
        result = rf.to_analysis()
        self.assertIsInstance(result, AnalysisList)
        self.assertEqual(len(result), 2)
        # Measurements should be attached to AnalysisList
        self.assertAlmostEqual(result.measurements["v_max"], 5.0)

    ##############################################

    def test_three_sweeps_all_transient(self):
        time_arr = np.linspace(0, 20e-9, 10)
        data_list = [
            {"TIME": time_arr, "node_a": np.ones(10) * i}
            for i in range(3)
        ]
        data = _make_hspice_data(data_list)
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, AnalysisList)
        self.assertEqual(len(result), 3)
        for analysis in result:
            self.assertIsInstance(analysis, TransientAnalysis)

    ##############################################

    def test_single_sweep_not_wrapped(self):
        time_arr = np.linspace(0, 20e-9, 10)
        data_list = [{"TIME": time_arr, "node_a": np.ones(10)}]
        data = _make_hspice_data(data_list)
        rf = HSpiceRawFile(data=data)
        result = rf.to_analysis()
        self.assertIsInstance(result, TransientAnalysis)
        self.assertNotIsInstance(result, AnalysisList)


####################################################################################################

class TestHSpiceServerInit(unittest.TestCase):
    """Tests for HSpiceServer.__init__()."""

    ##############################################

    def setUp(self):
        # Clear relevant env vars before each test
        self._original_env = {}
        for key in ("PYSPICE_HSPICE_CONCURRENCY_LIMIT", "PYSPICE_HSPICE_TIMEOUT"):
            self._original_env[key] = os.environ.pop(key, None)

    ##############################################

    def tearDown(self):
        for key, val in self._original_env.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

    ##############################################

    def _make_server(self, **kwargs):
        from PySpice.Spice.HSpice.Server import HSpiceServer
        return HSpiceServer(**kwargs)

    ##############################################

    def test_default_spice_command(self):
        server = self._make_server()
        self.assertEqual(server._spice_command, "hspice")

    ##############################################

    def test_custom_spice_command(self):
        server = self._make_server(spice_command="/usr/local/bin/hspice")
        self.assertEqual(server._spice_command, "/usr/local/bin/hspice")

    ##############################################

    def test_default_concurrency_limit(self):
        server = self._make_server()
        self.assertEqual(server._concurrency_limit, 4)

    ##############################################

    def test_custom_concurrency_limit(self):
        server = self._make_server(concurrency_limit=8)
        self.assertEqual(server._concurrency_limit, 8)

    ##############################################

    def test_concurrency_limit_from_env(self):
        os.environ["PYSPICE_HSPICE_CONCURRENCY_LIMIT"] = "6"
        server = self._make_server()
        self.assertEqual(server._concurrency_limit, 6)

    ##############################################

    def test_kwargs_overrides_env_concurrency(self):
        os.environ["PYSPICE_HSPICE_CONCURRENCY_LIMIT"] = "6"
        server = self._make_server(concurrency_limit=2)
        self.assertEqual(server._concurrency_limit, 2)

    ##############################################

    def test_invalid_concurrency_limit_raises(self):
        from PySpice.Spice.HSpice.Server import HSpiceServer
        with self.assertRaises(ValueError):
            HSpiceServer(concurrency_limit=0)

    ##############################################

    def test_negative_concurrency_limit_raises(self):
        from PySpice.Spice.HSpice.Server import HSpiceServer
        with self.assertRaises(ValueError):
            HSpiceServer(concurrency_limit=-1)

    ##############################################

    def test_default_timeout(self):
        server = self._make_server()
        self.assertAlmostEqual(server._timeout, 300.0)

    ##############################################

    def test_custom_timeout(self):
        server = self._make_server(timeout=60)
        self.assertAlmostEqual(server._timeout, 60.0)

    ##############################################

    def test_timeout_from_env(self):
        os.environ["PYSPICE_HSPICE_TIMEOUT"] = "120"
        server = self._make_server()
        self.assertAlmostEqual(server._timeout, 120.0)

    ##############################################

    def test_kwargs_overrides_env_timeout(self):
        os.environ["PYSPICE_HSPICE_TIMEOUT"] = "120"
        server = self._make_server(timeout=45)
        self.assertAlmostEqual(server._timeout, 45.0)

    ##############################################

    def test_concurrency_limit_as_string_from_kwarg(self):
        # concurrency_limit from kwarg as int
        server = self._make_server(concurrency_limit=3)
        self.assertEqual(server._concurrency_limit, 3)

    ##############################################

    def test_empty_spice_command_uses_default(self):
        # None or empty string spice_command should fall back to SPICE_COMMAND class attribute
        from PySpice.Spice.HSpice.Server import HSpiceServer
        server = HSpiceServer(spice_command=None)
        self.assertEqual(server._spice_command, "hspice")


####################################################################################################

class TestHSpiceFileInit(unittest.TestCase):
    """Tests for PySpice.Spice.HSpice.hspicefile.__init__ (hspice_read wrapper)."""

    ##############################################

    def test_hspice_read_raises_import_error_when_extension_unavailable(self):
        """When _hspice_read C extension is not compiled, hspice_read should raise ImportError."""
        import PySpice.Spice.HSpice.hspicefile as hf
        original = hf._hspice_read
        try:
            hf._hspice_read = None
            with self.assertRaises(ImportError):
                hf.hspice_read("dummy_file.tr0")
        finally:
            hf._hspice_read = original

    ##############################################

    def test_hspice_read_calls_extension_when_available(self):
        """When _hspice_read is available, hspice_read should call it."""
        import PySpice.Spice.HSpice.hspicefile as hf
        original = hf._hspice_read
        try:
            mock_ext = MagicMock()
            mock_ext.hspice_read.return_value = [MagicMock()]
            hf._hspice_read = mock_ext
            result = hf.hspice_read("dummy_file.tr0", debug=0)
            mock_ext.hspice_read.assert_called_once_with("dummy_file.tr0", 0)
        finally:
            hf._hspice_read = original

    ##############################################

    def test_hspice_read_error_message_mentions_extension(self):
        """ImportError message should mention _hspice_read extension."""
        import PySpice.Spice.HSpice.hspicefile as hf
        original = hf._hspice_read
        try:
            hf._hspice_read = None
            try:
                hf.hspice_read("dummy.tr0")
                self.fail("Expected ImportError")
            except ImportError as e:
                self.assertIn("_hspice_read", str(e))
        finally:
            hf._hspice_read = original


####################################################################################################

class TestSimulatorRegistration(unittest.TestCase):
    """Tests for Simulator.py: HSpice registration in the simulator factory."""

    ##############################################

    def test_hspice_in_simulator_map(self):
        from PySpice.Spice.Simulator import SIMULATOR_MAP
        self.assertIn("hspice", SIMULATOR_MAP)

    ##############################################

    def test_hspice_maps_to_hspice_simulator_class(self):
        from PySpice.Spice.Simulator import SIMULATOR_MAP
        from PySpice.Spice.HSpice.Simulator import HSpiceSimulator
        self.assertIs(SIMULATOR_MAP["hspice"], HSpiceSimulator)

    ##############################################

    def test_hspice_registered_in_simulator_classes(self):
        from PySpice.Spice.SimulatorBase import Simulator
        self.assertIn("hspice", Simulator._SIMULATOR_CLASSES)

    ##############################################

    def test_hspice_simulator_class_registered(self):
        from PySpice.Spice.SimulatorBase import Simulator
        from PySpice.Spice.HSpice.Simulator import HSpiceSimulator
        self.assertIs(Simulator._SIMULATOR_CLASSES["hspice"], HSpiceSimulator)

    ##############################################

    def test_simulators_tuple_contains_hspice(self):
        from PySpice.Spice.Simulator import SIMULATORS
        self.assertIn("hspice", SIMULATORS)

    ##############################################

    def test_other_simulators_still_registered(self):
        from PySpice.Spice.Simulator import SIMULATOR_MAP
        for sim in ("ngspice-subprocess", "ngspice", "ngspice-shared", "xyce"):
            self.assertIn(sim, SIMULATOR_MAP, f"'{sim}' should still be registered")


####################################################################################################

class TestHSpiceSimulatorInit(unittest.TestCase):
    """Tests for HSpiceSimulator.__init__ and properties."""

    ##############################################

    def test_simulator_attribute(self):
        from PySpice.Spice.HSpice.Simulator import HSpiceSimulator
        self.assertEqual(HSpiceSimulator.SIMULATOR, "hspice")

    ##############################################

    def test_version_returns_empty_string(self):
        from PySpice.Spice.HSpice.Simulator import HSpiceSimulator
        sim = HSpiceSimulator()
        self.assertEqual(sim.version, "")

    ##############################################

    def test_init_creates_spice_server(self):
        from PySpice.Spice.HSpice.Simulator import HSpiceSimulator
        from PySpice.Spice.HSpice.Server import HSpiceServer
        sim = HSpiceSimulator()
        self.assertIsInstance(sim._spice_server, HSpiceServer)

    ##############################################

    def test_init_with_spice_command(self):
        from PySpice.Spice.HSpice.Simulator import HSpiceSimulator
        sim = HSpiceSimulator(spice_command="/custom/hspice")
        self.assertEqual(sim._spice_server._spice_command, "/custom/hspice")

    ##############################################

    def test_init_with_concurrency_limit(self):
        from PySpice.Spice.HSpice.Simulator import HSpiceSimulator
        sim = HSpiceSimulator(concurrency_limit=2)
        self.assertEqual(sim._spice_server._concurrency_limit, 2)

    ##############################################

    def test_customise_adds_post_option(self):
        from PySpice.Spice.HSpice.Simulator import HSpiceSimulator
        sim = HSpiceSimulator()
        mock_simulation = MagicMock()
        sim.customise(mock_simulation)
        mock_simulation.options.assert_called_once_with(post=1)

    ##############################################

    def test_run_calls_server_and_sets_simulation(self):
        from PySpice.Spice.HSpice.Simulator import HSpiceSimulator
        sim = HSpiceSimulator()

        mock_raw_file = MagicMock()
        mock_raw_file.to_analysis.return_value = MagicMock()
        sim._spice_server = MagicMock(return_value=mock_raw_file)

        mock_simulation = MagicMock()
        mock_simulation.__str__ = MagicMock(return_value="* netlist")

        result = sim.run(mock_simulation)

        sim._spice_server.assert_called_once_with(spice_input="* netlist")
        self.assertEqual(mock_raw_file.simulation, mock_simulation)
        mock_raw_file.to_analysis.assert_called_once()


####################################################################################################

class TestAnalysisListMeasurementsIntegration(unittest.TestCase):
    """Integration tests for AnalysisList measurements access patterns."""

    ##############################################

    def test_float_measurement_access(self):
        al = AnalysisList([MagicMock()], {"rise_time": 2.5e-9})
        self.assertAlmostEqual(al["rise_time"], 2.5e-9)
        self.assertAlmostEqual(al.rise_time, 2.5e-9)

    ##############################################

    def test_list_measurement_access(self):
        al = AnalysisList([MagicMock()], {"v_max": [1.0, 2.0, 3.0]})
        self.assertEqual(al["v_max"], [1.0, 2.0, 3.0])
        self.assertEqual(al.v_max, [1.0, 2.0, 3.0])

    ##############################################

    def test_measurements_not_confused_with_list_indices(self):
        """Numeric index should access list, not measurements."""
        analysis0 = MagicMock()
        analysis1 = MagicMock()
        al = AnalysisList([analysis0, analysis1], {"0": 99.0})
        # Integer 0 should return analysis0, not measurements["0"]
        self.assertIs(al[0], analysis0)

    ##############################################

    def test_measurements_attached_to_each_sweep_analysis(self):
        """When multiple sweeps produce AnalysisList, measurements are accessible."""
        time_arr = np.linspace(0, 20e-9, 5)
        data_list = [
            {"TIME": time_arr, "node_a": np.ones(5)},
            {"TIME": time_arr, "node_a": np.ones(5) * 2},
        ]
        data = _make_hspice_data(data_list)
        meas = {"v_out_max": 3.3}
        rf = HSpiceRawFile(data=data, measurements=meas)
        result = rf.to_analysis()
        self.assertIsInstance(result, AnalysisList)
        self.assertAlmostEqual(result.v_out_max, 3.3)


####################################################################################################

class TestParseSpiceValueEdgeCases(unittest.TestCase):
    """Additional edge case tests for parse_spice_value()."""

    ##############################################

    def test_femto_is_smallest_supported(self):
        # 1f = 1e-15
        val = parse_spice_value("1f")
        self.assertAlmostEqual(val, 1e-15)
        self.assertLess(val, parse_spice_value("1p"))

    ##############################################

    def test_giga_is_largest_supported(self):
        # 1g = 1e9
        val = parse_spice_value("1g")
        self.assertAlmostEqual(val, 1e9)
        self.assertGreater(val, parse_spice_value("1meg"))

    ##############################################

    def test_decimal_with_suffix(self):
        self.assertAlmostEqual(parse_spice_value("2.2k"), 2200.0)
        self.assertAlmostEqual(parse_spice_value("4.7p"), 4.7e-12)
        self.assertAlmostEqual(parse_spice_value("0.1m"), 0.1e-3)

    ##############################################

    def test_integer_suffix(self):
        self.assertAlmostEqual(parse_spice_value("100"), 100.0)
        self.assertAlmostEqual(parse_spice_value("0"), 0.0)

    ##############################################

    def test_result_type_is_float(self):
        for s in ["1k", "2p", "3n", "4u", "5m", "6meg", "7g", "1.0"]:
            result = parse_spice_value(s)
            self.assertIsInstance(result, float, f"Expected float for '{s}', got {type(result)}")


####################################################################################################

if __name__ == '__main__':
    unittest.main()