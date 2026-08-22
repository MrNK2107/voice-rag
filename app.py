import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gradio import Server
from app.main import app

server = Server(app)
server.launch()
