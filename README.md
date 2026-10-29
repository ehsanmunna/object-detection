Create a complete README containing:

Requirements

Python 3.11+

Installation

python -m venv .venv

Windows:

.venv\Scripts\activate

Install:

pip install -r backend/requirements.txt

Run backend

python backend/run.py

or:

uvicorn app.main:app --reload

Explain the correct working directory.

Run frontend

Explain how to serve the frontend locally.

Do not rely on opening index.html directly if that causes CORS/WebSocket problems. Provide a simple local HTTP server option.

Example:

python -m http.server 5500

Then open:

http://localhost:5500