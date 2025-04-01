import os
import shutil
import uuid
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path

from httpx import request
from config import UPLOAD_FOLDER, PREVIEW_FOLDER, templates
from utils import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    adjust_brightness_contrast,
    adjust_color_balance,
    apply_blur,
    generate_preview,
    get_edited_img_path,
    get_str_img_path,
    get_img,
    is_allowed_file,
    resize_img,
    save_img,
)
from utils import crop_image as crop_image_utils
from utils import mirror_image as mirror_image_utils
from utils import rotate_image as rotate_image_utils

opencv = APIRouter()


@opencv.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    template = templates.get_template("index.html")
    return template.render(request=request, message=None, image_url=None)


@opencv.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    template = templates.get_template("index.html")

    if not is_allowed_file(file.filename):
        return template.render(
            request=request, message="Неподдерживаемый формат!", image_url=None
        )

    file_ext = Path(file.filename).suffix.lower()
    unique_name = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)
    preview_path = os.path.join(PREVIEW_FOLDER, unique_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if not generate_preview(file_path, preview_path):
        return template.render(
            request=request, message="Ошибка обработки изображения!", image_url=None
        )

    return template.render(
        request=request,
        message="Файл загружен!",
        image_url=f"/uploads/previews/{unique_name}",
    )


@opencv.post("/resize/", response_class=HTMLResponse)
async def resize_image(
    request: Request,
    image_path: str = Form(...),
    scale: float = Form(),
    width: int = Form(),
    height: int = Form(),
    interpolation: str = Form("bilinear"),
):
    return await process_image(
        request,
        image_path,
        resize_img,
        scale=scale,
        width=width,
        height=height,
        method=interpolation,
    )


@opencv.post("/crop/", response_class=HTMLResponse)
async def crop_image(
    request: Request,
    image_path: str = Form(...),
    x: int = Form(),
    y: int = Form(),
    width: int = Form(),
    height: int = Form(),
):
    return await process_image(
        request, image_path, crop_image_utils, x=x, y=y, width=width, height=height
    )


@opencv.post("/mirror/", response_class=HTMLResponse)
async def mirror_image(
    request: Request,
    image_path: str = Form(...),
    axis: str = Form(),
):
    axis = int(axis)
    return await process_image(request, image_path, mirror_image_utils, axis=axis)


@opencv.post("/rotate/", response_class=HTMLResponse)
async def rotate_image(
    request: Request,
    image_path: str = Form(...),
    angle: int = Form(),
    x: int = Form(),
    y: int = Form(),
    scale: float = Form(),
):
    return await process_image(
        request, image_path, rotate_image_utils, angle=angle, x=x, y=y, scale=scale
    )


@opencv.post("/brightness/", response_class=HTMLResponse)
async def brightness_contrast(
    request: Request,
    image_path: str = Form(...),
    brightness: int = Form(...),
    contrast: int = Form(...),
):
    return await process_image(
        request,
        image_path,
        adjust_brightness_contrast,
        brightness=brightness,
        contrast=contrast,
    )


@opencv.post("/color_balance/", response_class=HTMLResponse)
async def color_balance(
    request: Request,
    image_path: str = Form(...),
    red_factor: float = Form(...),
    green_factor: float = Form(...),
    blue_factor: float = Form(...),
):
    return await process_image(
        request,
        image_path,
        adjust_color_balance,
        red_factor=red_factor,
        green_factor=green_factor,
        blue_factor=blue_factor,
    )


@opencv.post("/gaussian_noise/", response_class=HTMLResponse)
async def gaussian_noise(
    request: Request,
    image_path: str = Form(...),
    mean: float = Form(),
    sigma: float = Form(),
):
    return await process_image(
        request,
        image_path,
        add_gaussian_noise,
        mean=mean,
        sigma=sigma,
    )


@opencv.post("/salt_pepper/", response_class=HTMLResponse)
async def salt_pepper(
    request: Request,
    image_path: str = Form(...),
    salt_prob: float = Form(),
    pepper_prob: float = Form(),
):
    return await process_image(
        request,
        image_path,
        add_salt_and_pepper_noise,
        pepper_prob=pepper_prob,
        salt_prob=salt_prob,
    )


@opencv.post("/blur/", response_class=HTMLResponse)
async def blur(
    request: Request,
    image_path: str = Form(...),
    blur_type: str = Form(),
    ksize: int = Form(),
):
    return await process_image(
        request,
        image_path,
        apply_blur,
        blur_type=blur_type,
        ksize=ksize,
    )


@opencv.get("/download/{image_url}")
async def download_file(image_url: str):
    MIME_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }
    ext = os.path.splitext(image_url)[1].lower()

    mime_type = MIME_TYPES.get(ext, "application/octet-stream")
    return FileResponse(image_url, media_type=mime_type)


async def process_image(request: Request, image_path: str, process_fn, *args, **kwargs):
    template = templates.get_template("index.html")

    image_path = Path(image_path.strip("/"))
    if not image_path.exists():
        return template.render(
            request=request, message="Файл не найден!", image_url=None
        )

    img = get_img(image_path)
    image_path_str = get_str_img_path(image_path)

    try:
        edited_img = process_fn(img, *args, **kwargs)
    except ValueError as e:
        return template.render(
            request=request,
            message=str(e),
            image_url=f"/uploads/previews/{image_path_str}",
        )

    edited_img_path = get_edited_img_path(image_path_str)
    save_img(edited_img_path, edited_img)

    return template.render(
        request=request,
        message="Операция выполнена успешно",
        image_url=f"/uploads/previews/{image_path_str}",
        edited_image_url=f"/uploads/{edited_img_path}",
    )
