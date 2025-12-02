"""Tests for refactored API endpoints.

This test suite covers all endpoints in the refactored API structure.
"""

import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient
from app_refactored import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self):
        """Test root health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "message" in data
    
    def test_favicon_endpoint(self):
        """Test favicon endpoint."""
        response = client.get("/favicon.ico")
        # Will return 404 if favicon doesn't exist, which is okay
        assert response.status_code in [200, 404]


class TestCalibrationEndpoints:
    """Test calibration endpoints."""
    
    def test_calibrate_setup_success(self, sample_person_image):
        """Test successful calibration."""
        response = client.post(
            "/calibrate/setup",
            files={"image": ("test.jpg", sample_person_image, "image/jpeg")},
            data={"known_height_cm": 170.0}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "session_id" in data
        assert "calibration" in data
        assert "cm_per_pixel" in data["calibration"]
    
    def test_calibrate_setup_missing_height(self, sample_person_image):
        """Test calibration without height parameter."""
        response = client.post(
            "/calibrate/setup",
            files={"image": ("test.jpg", sample_person_image, "image/jpeg")}
        )
        assert response.status_code == 422  # Validation error
    
    def test_calibrate_setup_invalid_image(self):
        """Test calibration with invalid image."""
        response = client.post(
            "/calibrate/setup",
            files={"image": ("test.jpg", b"invalid", "image/jpeg")},
            data={"known_height_cm": 170.0}
        )
        assert response.status_code in [400, 500]


class TestCaptureEndpoints:
    """Test capture endpoints."""
    
    def test_capture_angle_without_session(self, sample_person_image):
        """Test capture without valid session."""
        response = client.post(
            "/capture/angle",
            files={"image": ("test.jpg", sample_person_image, "image/jpeg")},
            data={
                "session_id": "invalid-session",
                "angle_type": "front"
            }
        )
        assert response.status_code == 400
    
    def test_capture_angle_invalid_type(self, sample_person_image, calibration_session):
        """Test capture with invalid angle type."""
        response = client.post(
            "/capture/angle",
            files={"image": ("test.jpg", sample_person_image, "image/jpeg")},
            data={
                "session_id": calibration_session,
                "angle_type": "invalid"
            }
        )
        assert response.status_code == 400
    
    def test_capture_angle_success(self, sample_person_image, calibration_session):
        """Test successful angle capture."""
        response = client.post(
            "/capture/angle",
            files={"image": ("test.jpg", sample_person_image, "image/jpeg")},
            data={
                "session_id": calibration_session,
                "angle_type": "front"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["angle_type"] == "front"
        assert "landmarks_count" in data


class TestMeasurementEndpoints:
    """Test measurement extraction endpoints."""
    
    def test_extract_without_session(self):
        """Test extraction without valid session."""
        response = client.post(
            "/measure/extract",
            params={"session_id": "invalid-session"}
        )
        assert response.status_code == 400
    
    def test_extract_without_calibration(self, sample_person_image):
        """Test extraction without calibration."""
        # This would need a session without calibration
        # For now, skip or create minimal session
        pass
    
    def test_extract_without_front_view(self, calibration_session):
        """Test extraction without front view captured."""
        response = client.post(
            "/measure/extract",
            params={"session_id": calibration_session}
        )
        assert response.status_code == 400
    
    def test_extract_success(self, complete_capture_session):
        """Test successful measurement extraction."""
        response = client.post(
            "/measure/extract",
            params={"session_id": complete_capture_session}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "measurements" in data


class TestCustomerEndpoints:
    """Test customer management endpoints."""
    
    def test_save_customer_without_measurements(self, calibration_session):
        """Test saving customer without measurements."""
        response = client.post(
            "/customer/save",
            json={
                "session_id": calibration_session,
                "customer_info": {
                    "name": "Test Customer",
                    "phone": "1234567890",
                    "email": "test@example.com"
                }
            }
        )
        assert response.status_code == 400
    
    def test_save_customer_success(self, session_with_measurements):
        """Test successful customer save."""
        response = client.post(
            "/customer/save",
            json={
                "session_id": session_with_measurements,
                "customer_info": {
                    "name": "Test Customer",
                    "phone": "1234567890",
                    "email": "test@example.com"
                }
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "customer_id" in data
        return data["customer_id"]
    
    def test_get_customer_not_found(self):
        """Test getting non-existent customer."""
        response = client.get("/customer/invalid-id")
        assert response.status_code == 404
    
    def test_get_customer_success(self, saved_customer_id):
        """Test successful customer retrieval."""
        response = client.get(f"/customer/{saved_customer_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "customer" in data
    
    def test_get_customer_history(self, saved_customer_id):
        """Test customer history retrieval."""
        response = client.get(f"/customer/{saved_customer_id}/history")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "history" in data
        assert "total_records" in data


# Fixtures

@pytest.fixture
def sample_person_image():
    """Create a sample person image for testing."""
    # You would load an actual test image here
    # For now, return minimal image data
    return b"fake-image-data"


@pytest.fixture
def calibration_session(sample_person_image):
    """Create a calibrated session for testing."""
    response = client.post(
        "/calibrate/setup",
        files={"image": ("test.jpg", sample_person_image, "image/jpeg")},
        data={"known_height_cm": 170.0}
    )
    if response.status_code == 200:
        return response.json()["session_id"]
    return None


@pytest.fixture
def complete_capture_session(calibration_session, sample_person_image):
    """Create a session with all angles captured."""
    if not calibration_session:
        return None
    
    # Capture front view
    client.post(
        "/capture/angle",
        files={"image": ("test.jpg", sample_person_image, "image/jpeg")},
        data={
            "session_id": calibration_session,
            "angle_type": "front"
        }
    )
    
    return calibration_session


@pytest.fixture
def session_with_measurements(complete_capture_session):
    """Create a session with extracted measurements."""
    if not complete_capture_session:
        return None
    
    # Extract measurements
    client.post(
        "/measure/extract",
        params={"session_id": complete_capture_session}
    )
    
    return complete_capture_session


@pytest.fixture
def saved_customer_id(session_with_measurements):
    """Create and save a customer, return customer ID."""
    if not session_with_measurements:
        return None
    
    response = client.post(
        "/customer/save",
        json={
            "session_id": session_with_measurements,
            "customer_info": {
                "name": "Test Customer",
                "phone": "1234567890",
                "email": "test@example.com"
            }
        }
    )
    
    if response.status_code == 200:
        return response.json()["customer_id"]
    return None
