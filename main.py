from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from config import UPLOAD_FOLDER
from controller import opencv


app = FastAPI()

app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

app.include_router(opencv)
