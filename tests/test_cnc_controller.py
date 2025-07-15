#!/usr/bin/env python3
"""
Tests for CNCController class
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import Mock, patch
from src.CNCController import CNCController

class TestCNCController(unittest.TestCase):
    """Test cases for CNCController class"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('serial.Serial'):
            self.cnc = CNCController()
    
    def test_initialization(self):
        """Test CNCController initialization"""
        self.assertIsNotNone(self.cnc)
        self.assertEqual(self.cnc.port, '/dev/cu.usbserial-130')
        self.assertEqual(self.cnc.baudrate, 115200)
    
    def test_move_to_well_valid(self):
        """Test moving to a valid well position"""
        with patch.object(self.cnc, 'send_gcode') as mock_send:
            self.cnc.move_to_well('A', '1')
            mock_send.assert_called_once()
            # Check that G01 command was sent
            call_args = mock_send.call_args[0][0]
            self.assertIn('G01', call_args)
            self.assertIn('X100.000', call_args)  # A1 X position
            self.assertIn('Y48.000', call_args)   # A1 Y position
    
    def test_move_to_well_invalid(self):
        """Test moving to an invalid well position"""
        with self.assertRaises(ValueError):
            self.cnc.move_to_well('Z', '1')  # Invalid column
    
    def test_move_to_z(self):
        """Test moving to a specific Z position"""
        with patch.object(self.cnc, 'send_gcode') as mock_send:
            with patch.object(self.cnc, 'wait_for_idle') as mock_wait:
                self.cnc.move_to_z(-10.0)
                mock_send.assert_called_once()
                call_args = mock_send.call_args[0][0]
                self.assertIn('G01', call_args)
                self.assertIn('Z-10.000', call_args)
                mock_wait.assert_called_once()
    
    def test_home(self):
        """Test homing the CNC"""
        with patch.object(self.cnc, 'send_gcode') as mock_send:
            self.cnc.home()
            # Should call $H (home) and G92 (zero)
            self.assertEqual(mock_send.call_count, 2)
            calls = [call[0][0] for call in mock_send.call_args_list]
            self.assertIn('$H', calls)
            self.assertIn('G92 X0 Y0 Z0', calls)
    
    def test_get_current_position(self):
        """Test getting current position"""
        mock_response = "MPos:100.000,48.000,-12.000|WPos:100.000,48.000,-12.000"
        with patch.object(self.cnc.ser, 'readline', return_value=mock_response.encode()):
            pos = self.cnc.get_current_position()
            self.assertEqual(pos, (100.0, 48.0, -12.0))
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'cnc'):
            self.cnc.close()

if __name__ == '__main__':
    unittest.main() 