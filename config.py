import os

from jinja2 import Environment, FileSystemLoader


UPLOAD_FOLDER = "uploads"
PREVIEW_FOLDER = "uploads/previews"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREVIEW_FOLDER, exist_ok=True)

templates = Environment(loader=FileSystemLoader("templates"))