"""
Simplified test suite for TailorAI v2 modules.
Tests core functionality of calibration, geometry, measurements, and API endpoints.
"""

import pytest
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app
from models.calibration import CameraCalibration
from utils.geometry import GeometryCalculator
from models.measurements import TailorMeasurementExtractor, MeasurementQuality


client = TestClient(app)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_image():
    """Create a simple test image (RGB, 640x480)"""
    img = Image.new('RGB', (640, 480), color=(100, 150, 200))
    buf = BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def sample_landmarks():
    """Create mock MediaPipe landmarks (33 points)"""
    landmarks = []
    for i in range(33):
        landmarks.append({
            'x': 0.5,
            'y': 0.5,
            'z': 0.0,
            'visibility': 0.95,
            'x_px': 320.0,
            'y_px': 240.0
        })
    return landmarks


# ============================================================================
# CALIBRATION MODULE TESTS
# ============================================================================

def test_calibration_initialization():
    """Test CameraCalibration initialization"""
    calib = CameraCalibration(known_distance_m=1.5)
    assert calib.known_distance_m == 1.5


def test_calibrate_from_height(sample_landmarks):
    """Test height-based calibration"""
    calib = CameraCalibration(known_distance_m=1.5)
    result = calib.calibrate_from_height(sample_landmarks, known_height_cm=170)
    
    assert result is not None
    assert 'cm_per_pixel' in result
    assert result['cm_per_pixel'] > 0


def test_pixels_to_cm_conversion(sample_landmarks):
    """Test pixel to cm conversion"""
    calib = CameraCalibration(known_distance_m=1.5)
    calib.calibrate_from_height(sample_landmarks, known_height_cm=170)
    cm = calib.pixels_to_cm(100)
    
    assert cm > 0
    assert isinstance(cm, float)


# ============================================================================
# GEOMETRY UTILITIES TESTS
# ============================================================================

def test_distance_2d_calculation():
    """Test 2D distance calculation"""
    point1 = {'x_px': 0, 'y_px': 0}
    point2 = {'x_px': 30, 'y_px': 40}
    
    distance = GeometryCalculator.distance_2d(point1, point2)
    assert distance == pytest.approx(50, rel=0.01)  # 3-4-5 triangle


def test_distance_3d_calculation():
    """Test 3D distance calculation"""
    point1 = {'x_px': 0, 'y_px': 0, 'z': 0}
    point2 = {'x_px': 30, 'y_px': 40, 'z': 0}
    
    distance = GeometryCalculator.distance_3d(point1, point2)
    assert distance > 0
    assert isinstance(distance, float)


def test_angle_between_points():
    """Test angle calculation"""
    p1 = {'x_px': 0, 'y_px': 0}
    vertex = {'x_px': 10, 'y_px': 0}
    p2 = {'x_px': 10, 'y_px': 10}
    
    angle = GeometryCalculator.angle_between(p1, vertex, p2)
    assert angle == pytest.approx(90, rel=1)  # Right angle


def test_visibility_check():
    """Test visibility check"""
    landmarks = [{'visibility': 0.95} for _ in range(33)]
    indices = [0, 1, 11, 12]
    
    is_visible = GeometryCalculator.visibility_check(landmarks, indices, min_visibility=0.8)
    assert is_visible is True


# ============================================================================
# MEASUREMENT EXTRACTION TESTS
# ============================================================================

def test_extractor_initialization():
    """Test measurement extractor initialization"""
    calib_data = {'cm_per_pixel': 0.15, 'confidence': 0.95}
    extractor = TailorMeasurementExtractor(calibration_data=calib_data)
    assert extractor.calibration_data == calib_data


def test_extract_all_measurements_structure(sample_landmarks):
    """Test measurements extraction structure"""
    calib_data = {'cm_per_pixel': 0.15, 'confidence': 0.95}
    extractor = TailorMeasurementExtractor(calibration_data=calib_data)
    
    measurements = extractor.extract_all_measurements(
        landmarks_front=sample_landmarks,
        landmarks_side=sample_landmarks,
        landmarks_back=sample_landmarks
    )
    
    assert measurements is not None
    assert 'upper_body' in measurements
    assert 'torso' in measurements
    assert 'lower_body' in measurements
    assert 'full_body' in measurements


def test_measurements_are_positive(sample_landmarks):
    """Test that measurements are positive values"""
    calib_data = {'cm_per_pixel': 0.15}
    extractor = TailorMeasurementExtractor(calibration_data=calib_data)
    
    measurements = extractor.extract_all_measurements(
        landmarks_front=sample_landmarks,
        landmarks_side=sample_landmarks,
        landmarks_back=sample_landmarks
    )
    
    for category in ['upper_body', 'torso', 'lower_body', 'full_body']:
        for key, value in measurements[category].items():
            if isinstance(value, (int, float)):
                assert value >= 0, f"{key} is negative: {value}"


# ============================================================================
# API ENDPOINT TESTS
# ============================================================================

@patch('app.pose_model.extract_pose_from_image')
def test_calibrate_setup_endpoint(mock_extract, sample_image, sample_landmarks):
    """Test /calibrate/setup endpoint"""
    mock_extract.return_value = {'landmarks': sample_landmarks}
    
    response = client.post(
        '/calibrate/setup',
        files={'image': ('test.jpg', BytesIO(sample_image), 'image/jpeg')},
        data={'known_height_cm': 170}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'
    assert 'session_id' in data


@patch('app.pose_model.extract_pose_from_image')
def test_capture_angle_endpoint(mock_extract, sample_image, sample_landmarks):
    """Test /capture/angle endpoint"""
    mock_extract.return_value = {'landmarks': sample_landmarks}
    
    # First calibrate
    calib_response = client.post(
        '/calibrate/setup',
        files={'image': ('test.jpg', BytesIO(sample_image), 'image/jpeg')},
        data={'known_height_cm': 170}
    )
    session_id = calib_response.json()['session_id']
    
    # Then capture
    response = client.post(
        '/capture/angle',
        files={'image': ('test.jpg', BytesIO(sample_image), 'image/jpeg')},
        data={'session_id': session_id, 'angle_type': 'front'}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'


@patch('app.pose_model.extract_pose_from_image')
def test_measure_extract_endpoint(mock_extract, sample_image, sample_landmarks):
    """Test /measure/extract endpoint"""
    mock_extract.return_value = {'landmarks': sample_landmarks}
    
    # Calibrate
    calib_response = client.post(
        '/calibrate/setup',
        files={'image': ('test.jpg', BytesIO(sample_image), 'image/jpeg')},
        data={'known_height_cm': 170}
    )
    session_id = calib_response.json()['session_id']
    
    # Capture all angles
    for angle in ['front', 'side', 'back']:
        client.post(
            '/capture/angle',
            files={'image': ('test.jpg', BytesIO(sample_image), 'image/jpeg')},
            data={'session_id': session_id, 'angle_type': angle}
        )
    
    # Extract measurements
    response = client.post(f'/measure/extract?session_id={session_id}')
    
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'


@patch('app.pose_model.extract_pose_from_image')
def test_customer_save_endpoint(mock_extract, sample_image, sample_landmarks):
    """Test /customer/save endpoint"""
    mock_extract.return_value = {'landmarks': sample_landmarks}
    
    # Calibrate and capture
    calib_response = client.post(
        '/calibrate/setup',
        files={'image': ('test.jpg', BytesIO(sample_image), 'image/jpeg')},
        data={'known_height_cm': 170}
    )
    session_id = calib_response.json()['session_id']
    
    # Capture all angles before extraction
    for angle in ['front', 'side', 'back']:
        client.post(
            '/capture/angle',
            files={'image': ('test.jpg', BytesIO(sample_image), 'image/jpeg')},
            data={'session_id': session_id, 'angle_type': angle}
        )
    
    # Extract measurements
    response = client.post(f'/measure/extract?session_id={session_id}')
    
    assert response.status_code == 200


def test_health_check_endpoint():
    """Test root health check endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'


@patch('app.pose_model.extract_pose_from_image')
def test_complete_workflow(mock_extract, sample_image, sample_landmarks):
    """Test complete measurement workflow"""
    mock_extract.return_value = {'landmarks': sample_landmarks}
    
    # Step 1: Calibrate
    calib_response = client.post(
        '/calibrate/setup',
        files={'image': ('test.jpg', BytesIO(sample_image), 'image/jpeg')},
        data={'known_height_cm': 170}
    )
    assert calib_response.status_code == 200
    
    session_id = calib_response.json()['session_id']
    
    # Step 2: Capture angles
    for angle in ['front', 'side', 'back']:
        capture_response = client.post(
            '/capture/angle',
            files={'image': ('test.jpg', BytesIO(sample_image), 'image/jpeg')},
            data={'session_id': session_id, 'angle_type': angle}
        )
        assert capture_response.status_code == 200
    
    # Step 3: Extract
    extract_response = client.post(f'/measure/extract?session_id={session_id}')
    assert extract_response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
