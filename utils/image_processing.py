import PIL.Image as Image


def crop_image_half(image: Image.Image, left: bool = False) -> Image.Image:
    """
    PIL Image의 좌/우 절반을 잘라 반환합니다.
    """
    if not hasattr(image, "size"):
        raise TypeError("`image`는 PIL Image 여야 합니다.")
    width, height = image.size
    half_width = width // 2

    if left:
        return image.crop((0, 0, half_width, height))
    else:
        return image.crop((half_width, 0, width, height))
    


if __name__ == "__main__":  # simple test
    image_path = "../dataset/data/images/0500734.png"
    img = Image.open(image_path)
    print(img.size)  # (2560, 720) 등
    result = crop_image_half(img)
    print(result.size)  # (1280, 720) 등
    result.show()
