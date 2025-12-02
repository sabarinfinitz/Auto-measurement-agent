from flask import Flask, render_template
import os

app = Flask(__name__)

# Configuration
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:8000')

@app.route('/')
def index():
    return render_template('capture.html', backend_url=BACKEND_URL)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=True)
