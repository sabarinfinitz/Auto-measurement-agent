# Flask Frontend

This is the frontend for the Measurement Agent, built with Flask.
It serves the `capture.html` interface and communicates with the FastAPI backend.

## Setup

1.  Create a virtual environment:
    ```bash
    python -m venv venv
    ```
2.  Activate it:
    - Windows: `venv\Scripts\activate`
    - Linux/Mac: `source venv/bin/activate`
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running

1.  Ensure the backend is running on port 8000 (default).
2.  Run the frontend:
    ```bash
    python app.py
    ```
    Or use the `run_frontend.bat` script on Windows.

The frontend will be available at `http://localhost:3000`.
It is configured to talk to the backend at `http://localhost:8000`.
You can change this by setting the `BACKEND_URL` environment variable.
