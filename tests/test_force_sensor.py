#!/usr/bin/env python3
"""
Tests for ForceSensor class
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import Mock, patch
from src.ForceSensor import ForceSensor

class TestForceSensor(unittest.TestCase):
    """Test cases for ForceSensor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('src.ForceSensor.GoDirect'):
            self.force_sensor = ForceSensor()
            # Ensure it's not connected for tests
            self.force_sensor.connected = False
            self.force_sensor.device = None
            self.force_sensor.sensor = None
    
    def test_initialization(self):
        """Test ForceSensor initialization"""
        self.assertIsNotNone(self.force_sensor)
        self.assertFalse(self.force_sensor.connected)
    
    def test_is_connected_when_not_connected(self):
        """Test is_connected when sensor is not connected"""
        self.assertFalse(self.force_sensor.is_connected())
    
    @patch('src.ForceSensor.GoDirect')
    def test_connect_success(self, mock_godirect):
        """Test successful connection"""
        # Mock the GoDirect device
        mock_device = Mock()
        mock_sensor = Mock()
        mock_sensor.values = [1.5]
        mock_sensor.sensor_description = "Force Sensor"
        
        mock_device.open.return_value = True
        mock_device.get_enabled_sensors.return_value = [mock_sensor]
        
        mock_godirect_instance = Mock()
        mock_godirect_instance.get_device.return_value = mock_device
        mock_godirect.return_value = mock_godirect_instance
        
        # Create new instance with mocked dependencies
        sensor = ForceSensor()
        result = sensor.connect()
        
        self.assertTrue(result)
        self.assertTrue(sensor.connected)
    
    def test_get_force_reading_when_not_connected(self):
        """Test get_force_reading when sensor is not connected"""
        force = self.force_sensor.get_force_reading()
        self.assertEqual(force, 0.0)
    
    def test_get_baseline_force_when_not_connected(self):
        """Test get_baseline_force when sensor is not connected"""
        avg, std = self.force_sensor.get_baseline_force()
        self.assertEqual(avg, 0.0)
        self.assertEqual(std, 0.0)
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'force_sensor'):
            self.force_sensor.cleanup()

if __name__ == '__main__':
    unittest.main() 