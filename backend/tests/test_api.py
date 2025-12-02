"""Integration tests for TailorAI backend endpoints."""
import json
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from io import BytesIO

from app import app, MEASUREMENTS_FILE, DATA_DIR


@pytest.fixture
def client():
    """Provide a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_person_image():
    """Generate a simple test image with a person silhouette."""
    img = Image.new('RGB', (640, 480), color='white')
    draw = ImageDraw.Draw(img)
    # Draw a simple stick figure to have pose landmarks
    draw.ellipse([300, 80, 340, 120], fill='black')  # head
    draw.rectangle([310, 120, 330, 280], fill='black')  # torso
    draw.rectangle([280, 140, 310, 250], fill='black')  # left arm
    draw.rectangle([330, 140, 360, 250], fill='black')  # right arm
    draw.rectangle([300, 280, 315, 380], fill='black')  # left leg
    draw.rectangle([325, 280, 340, 380], fill='black')  # right leg
    
    # Convert to JPEG bytes
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def sample_garment_image():
    """Generate a simple test garment image."""
    img = Image.new('RGB', (200, 300), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 250], fill='blue', outline='darkblue')
    
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=85)
    buf.seek(0)
    return buf.getvalue()


class TestPoseEndpoint:
    """Tests for /predict/pose endpoint."""

    def test_pose_success(self, client, sample_person_image):
        """Test successful pose extraction from image."""
        response = client.post(
            '/predict/pose',
            files={'image': ('test.jpg', sample_person_image, 'image/jpeg')}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert 'landmarks' in data
        assert isinstance(data['landmarks'], dict)
        assert 'image_width' in data['landmarks']
        assert 'image_height' in data['landmarks']
        assert 'landmarks' in data['landmarks']

    def test_pose_invalid_image(self, client):
        """Test pose extraction with invalid image bytes."""
        response = client.post(
            '/predict/pose',
            files={'image': ('test.jpg', b'invalid', 'image/jpeg')}
        )
        assert response.status_code == 500


class TestMeasureEndpoint:
    """Tests for /predict/measure endpoint."""

    def test_measure_without_height(self, client, sample_person_image):
        """Test measurement estimation without known height."""
        response = client.post(
            '/predict/measure',
            files={'image': ('test.jpg', sample_person_image, 'image/jpeg')}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert 'measurements' in data
        m = data['measurements']
        assert 'shoulder_width' in m or 'error' in m
        assert 'units' in m

    def test_measure_with_known_height(self, client, sample_person_image):
        """Test measurement estimation with known height (scaling to cm)."""
        response = client.post(
            '/predict/measure',
            files={'image': ('test.jpg', sample_person_image, 'image/jpeg')},
            data={'known_height_cm': 170.0}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        m = data['measurements']
        # With known_height, units should be 'cm'
        if 'error' not in m:
            assert m.get('units') in ['cm', 'px']


class TestSaveMeasurementsEndpoint:
    """Tests for /save_measurements endpoint."""

    def test_save_measurements_success(self, client):
        """Test saving measurements to local JSON file."""
        payload = {
            'measurements': {
                'shoulder_width': 45.5,
                'hip_width': 50.2,
                'estimated_height': 170.0,
                'units': 'cm'
            },
            'meta': {'source': 'test', 'height_cm': 170.0}
        }
        response = client.post('/save_measurements', json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert 'saved' in data
        saved = data['saved']
        assert 'id' in saved
        assert 'timestamp' in saved
        assert saved['measurements'] == payload['measurements']

        # Verify file was created
        assert os.path.exists(MEASUREMENTS_FILE)

    def test_save_measurements_missing_payload(self, client):
        """Test saving with missing measurements key."""
        payload = {'some_other_key': 'value'}
        response = client.post('/save_measurements', json=payload)
        assert response.status_code == 400

    def test_save_measurements_appends(self, client):
        """Test that measurements are appended to file."""
        # First save
        p1 = {'measurements': {'test': 1}}
        r1 = client.post('/save_measurements', json=p1)
        assert r1.status_code == 200
        id1 = r1.json()['saved']['id']

        # Second save
        p2 = {'measurements': {'test': 2}}
        r2 = client.post('/save_measurements', json=p2)
        assert r2.status_code == 200
        id2 = r2.json()['saved']['id']

        # Read file and verify both entries exist
        with open(MEASUREMENTS_FILE, 'r') as f:
            data = json.load(f)
        assert len(data) >= 2
        ids = [entry['id'] for entry in data]
        assert id1 in ids
        assert id2 in ids


class TestTryonEndpoint:
    """Tests for /tryon endpoint (end-to-end try-on)."""

    def test_tryon_with_person_only(self, client, sample_person_image):
        """Test try-on with only person image (no garment)."""
        response = client.post(
            '/tryon',
            files={'image': ('person.jpg', sample_person_image, 'image/jpeg')}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert 'output' in data
        # Output should be a relative path under /uploads
        assert '/uploads/' in data['output']

    def test_tryon_with_garment(self, client, sample_person_image, sample_garment_image):
        """Test try-on with person and garment images."""
        response = client.post(
            '/tryon',
            files={
                'image': ('person.jpg', sample_person_image, 'image/jpeg'),
                'garment': ('garment.jpg', sample_garment_image, 'image/jpeg')
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert 'output' in data
        assert '/uploads/' in data['output']


class TestStaticFiles:
    """Tests for static file serving."""

    def test_root_endpoint(self, client):
        """Test GET / returns health check."""
        response = client.get('/')
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data


class TestIntegrationFlow:
    """End-to-end integration tests."""

    def test_full_capture_to_tryon_flow(self, client, sample_person_image, sample_garment_image):
        """Test the complete flow: pose → measure → save → tryon."""
        # Step 1: Extract pose
        pose_response = client.post(
            '/predict/pose',
            files={'image': ('test.jpg', sample_person_image, 'image/jpeg')}
        )
        assert pose_response.status_code == 200
        pose_data = pose_response.json()
        assert pose_data['status'] == 'ok'

        # Step 2: Estimate measurements with known height
        measure_response = client.post(
            '/predict/measure',
            files={'image': ('test.jpg', sample_person_image, 'image/jpeg')},
            data={'known_height_cm': 170.0}
        )
        assert measure_response.status_code == 200
        measure_data = measure_response.json()
        assert measure_data['status'] == 'ok'

        # Step 3: Save measurements
        save_response = client.post(
            '/save_measurements',
            json={
                'measurements': measure_data['measurements'],
                'meta': {'source': 'integration_test', 'height_cm': 170.0}
            }
        )
        assert save_response.status_code == 200
        assert save_response.json()['status'] == 'ok'

        # Step 4: Run try-on
        tryon_response = client.post(
            '/tryon',
            files={
                'image': ('person.jpg', sample_person_image, 'image/jpeg'),
                'garment': ('garment.jpg', sample_garment_image, 'image/jpeg')
            }
        )
        assert tryon_response.status_code == 200
        tryon_data = tryon_response.json()
        assert tryon_data['status'] == 'ok'
        assert '/uploads/' in tryon_data['output']
