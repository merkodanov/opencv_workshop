from pathlib import Path
from config import ALLOWED_EXTENSIONS, UPLOAD_FOLDER
import cv2
import os
import numpy as np

INTERPOLATION_METHODS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


def is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def generate_preview(image_path: str, preview_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return False
    cv2.imwrite(preview_path, img)
    return True


def resize_img(img, scale, width, height, method):
    size = (
        (width, height)
        if width != 0 and height != 0
        else (int(img.shape[1] * scale), int(img.shape[0] * scale))
    )

    method = INTERPOLATION_METHODS.get(method)

    return cv2.resize(img, size, interpolation=method)


def save_img(img_path, img):
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, img_path), img)


def crop_image(image, x, y, width, height):
    img_height, img_width = image.shape[:2]

    if width == 0 or height == 0:
        raise ValueError("Ширина и высота фрагмента не могут быть равны нулю.")
    if x < 0 or y < 0:
        raise ValueError("Координаты не могут быть отрицательными")
    if x + width > img_width or y + height > img_height:
        raise ValueError("Фрагмент выходит за пределы изображения")

    cropped_image = image[y : y + height, x : x + width]

    return cropped_image


def mirror_image(image, axis):
    if axis not in [-1, 0, 1]:
        raise ValueError("Ось может быть только -1 0 1")
    return cv2.flip(image, axis)


def rotate_image(image, angle, x, y, scale=1.0):
    center = (x, y)

    if center is None:
        center = (image.shape[1] // 2, image.shape[0] // 2)

    if not (0 <= center[0] < image.shape[1] and 0 <= center[1] < image.shape[0]):
        raise ValueError("Центр вращения должен находиться внутри изображения.")

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
    )

    return rotated_image


def adjust_brightness_contrast(img, brightness, contrast):
    img = np.int16(img)

    img = img * (contrast / 100 + 1) - 128 * (contrast / 100)

    img = img + brightness

    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return img


def adjust_color_balance(image, red_factor, green_factor, blue_factor):
    """Не отрицательные значения"""
    if red_factor < 0 or green_factor < 0 or blue_factor < 0:
        raise ValueError("Значения не могут быть меньше нуля")
    b, g, r = cv2.split(image)

    r = cv2.convertScaleAbs(r, alpha=red_factor)
    g = cv2.convertScaleAbs(g, alpha=green_factor)
    b = cv2.convertScaleAbs(b, alpha=blue_factor)

    balanced_image = cv2.merge([b, g, r])
    return balanced_image


def add_gaussian_noise(image, mean, sigma):
    if sigma < 0:
        raise ValueError("Отклонение не может быть меньше нуля")
    gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, gaussian_noise)
    return noisy_image


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    if salt_prob < 0 or pepper_prob < 0:
        raise ValueError("Соль и перец не могут быть меньше 0")
    noisy_image = image.copy()
    total_pixels = image.size

    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image


def apply_blur(image, blur_type, ksize=3):
    if ksize <= 2:
        raise ValueError("KSize не может быть меньше 3")
    if ksize % 2 == 0:
        raise ValueError("KSize может принимать только нечетные значения")
    if blur_type == "average":
        blurred_image = cv2.blur(image, (ksize, ksize))
    elif blur_type == "gaussian":
        blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif blur_type == "median":
        blurred_image = cv2.medianBlur(image, ksize)
    else:
        raise ValueError("Не поддерживаемый тип размытия")

    return blurred_image


def find_object_by_color(
    image,
    color_lower_bound: tuple,
    color_upper_bound: tuple,
    color_space: str,
    method: str,
):
    image = np.asarray(image)

    if color_space == "HSV":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == "RGB":
        converted = image
        color_lower_bound = (
            color_lower_bound[2],
            color_lower_bound[1],
            color_lower_bound[0],
        )
        color_upper_bound = (
            color_upper_bound[2],
            color_upper_bound[1],
            color_upper_bound[0],
        )
    else:
        raise ValueError("Не поддерживаемый цвет. Используйте 'RGB' или 'HSV'.")

    mask = cv2.inRange(converted, color_lower_bound, color_upper_bound)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    if method == "box":
        return cv2.rectangle(image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
    elif method == "crop":
        return image[y : y + h, x : x + w]
    else:
        raise ValueError("Не поддерживаемый метод. Используйте 'box' или 'crop'.")


def binarize_image(image, method: str, **kwargs):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    if method == "threshold":
        thresh_val = kwargs.get("thresh", 127)
        max_val = kwargs.get("maxval", 255)
        if thresh_val < 0 or max_val < 0:
            raise ValueError("Границы не могут быть меньше нуля")
        if thresh_val > max_val:
            raise ValueError("Нижняя граница больше верхней границы")
        _, binary = cv2.threshold(image, thresh_val, max_val, cv2.THRESH_BINARY)

    elif method == "adaptive":
        block_size = kwargs.get("block_size", 11)
        if block_size % 2 == 0:
            raise ValueError("block size - нечетное число")
        c = kwargs.get("c", 2)
        if block_size < 0 or c < 0:
            raise ValueError("block_size и c должны быть больше 0")
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
        )

    elif method == "sobel":
        ksize = kwargs.get("ksize", 3)
        if ksize <= 0:
            raise ValueError("Ksize должен быть больше 0")
        if ksize % 2 == 0:
            raise ValueError("KSize нечетное число")
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        binary = cv2.convertScaleAbs(cv2.sqrt(grad_x**2 + grad_y**2))

    elif method == "canny":
        threshold1 = kwargs.get("threshold1", 100)
        threshold2 = kwargs.get("threshold2", 200)
        if threshold1 < 0 or threshold2 < 0:
            raise ValueError("threshold должен быть больше 0")
        if threshold2 < threshold1:
            raise ValueError("Нижняя граница должна быть меньше верхней границы")
        binary = cv2.Canny(image, threshold1, threshold2)

    else:
        raise ValueError("Неизвестный метод бинаризации")

    return binary


def get_img(image_path):
    img = cv2.imread(image_path)
    return img


def get_str_img_path(image_path):
    image_path = str(image_path)
    return image_path.split("\\")[-1]


def get_file_path(filename):
    file_path = os.path.join(UPLOAD_FOLDER, f"edited_{filename}")

    if not os.path.exists(file_path):
        raise ValueError("Файла не существует!")

    return file_path


def get_edited_img_path(image_path):
    return "edited_" + image_path
