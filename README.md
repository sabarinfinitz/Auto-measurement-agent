# Enterprise Measurement Agent

Professional body measurement system using computer vision and MediaPipe.

## Prerequisites

- Python 3.11+
- Webcam/Camera
- Git

## Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd Measurement-agent
```

### 2. Backend Setup

```bash
cd backend
python -m venv pyenv
pyenv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
python -m venv flaskenv
flaskenv\Scripts\activate
pip install -r requirements.txt
```

## Run

### Terminal 1 - Backend (Port 8000)

```bash
cd backend
pyenv\Scripts\activate
python app.py
```

### Terminal 2 - Frontend (Port 3000)

```bash
cd frontend
flaskenv\Scripts\activate
python app.py
```

### Access Application

````
http://localhost:3000
````

## Environment Variables (Optional)

### Backend

Create `backend/.env`:

```env
PORT=8000
HOST=0.0.0.0
DEBUG=True
```

### Frontend

Create `frontend/.env`:

```env
PORT=3000
BACKEND_URL=http://localhost:8000
```

## Project Structure

```
Measurement-agent/
├── backend/
│   ├── app.py              # FastAPI server
│   ├── requirements.txt    # Backend dependencies
│   ├── models/             # Measurement models
│   ├── utils/              # Helper functions
│   ├── data/               # Customer data storage
│   └── uploads/            # Image uploads
├── frontend/
│   ├── app.py              # Flask server
│   ├── requirements.txt    # Frontend dependencies
│   └── templates/
│       └── capture.html    # Main UI
└── README.md
```

## Troubleshooting

### Camera Not Working

- Use `http://localhost:3000` (not IP address)
- Allow camera permissions in browser
- Close other apps using camera
- Try Chrome or Edge browser

### Port Already in Use

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Module Not Found

```bash
# Reactivate virtual environment
cd backend
pyenv\Scripts\activate
pip install -r requirements.txt
```

## Technology Stack

- **Backend**: FastAPI, MediaPipe, OpenCV, NumPy
- **Frontend**: Flask, HTML5, JavaScript
- **Computer Vision**: MediaPipe Pose Landmarks
- **Measurement**: Multi-angle 3D body measurements
