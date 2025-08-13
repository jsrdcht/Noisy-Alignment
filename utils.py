import random
import numpy as np
from PIL import Image


def load_image(image, mode='RGBA'):
    """Load image and convert to the given mode."""
    if isinstance(image, str):
        return Image.open(image).convert(mode)
    elif isinstance(image, Image.Image):
        return image.convert(mode)
    else:
        raise ValueError("Invalid image input")


def add_watermark(input_image, watermark, watermark_width=50, location_min=0.45, location_max=0.55, alpha_composite=True, alpha=0.2, return_location=False, mode='patch'):
    """
    Add a watermark to an image. Two modes are supported: 'patch' and 'blend'.

    Args:
        input_image: Input image path or PIL.Image.
        watermark: Watermark image path or PIL.Image.
        watermark_width: Watermark width in pixels (used in 'patch' mode only).
        location_min: Minimum ratio for random placement range.
        location_max: Maximum ratio for random placement range.
        alpha_composite: Whether to alpha-composite. Kept for API compatibility.
        alpha: Transparency factor for compositing.
        return_location: If True, also return the (x, y) location of the patch.
        mode: 'patch' (local paste) or 'blend' (global blending).

    Returns:
        Watermarked RGB image, and optionally the (x, y) location if return_location=True.
    """
    img_watermark = load_image(watermark, mode='RGBA')

    if isinstance(input_image, str):
        base_image = Image.open(input_image).convert('RGBA')
    elif isinstance(input_image, Image.Image):
        base_image = input_image.convert('RGBA')
    else:
        raise ValueError("Invalid input_image argument")

    # Choose watermarking strategy based on mode
    if mode == 'blend':
        # Global blending mode
        img_watermark = img_watermark.resize(base_image.size)
        
        watermark_array = np.array(img_watermark)
        watermark_array[:, :, 3] = (watermark_array[:, :, 3] * alpha).astype(np.uint8)
        watermark_image = Image.fromarray(watermark_array)
        
        result_image = Image.alpha_composite(base_image, watermark_image)
        result_image = result_image.convert('RGB')
        
        return result_image
    
    elif mode == 'patch':
        # Patch mode
        width, height = base_image.size
        w_width, w_height = watermark_width, int(img_watermark.size[1] * watermark_width / img_watermark.size[0])
        img_watermark = img_watermark.resize((w_width, w_height))
        transparent = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        # Random location by default
        loc_min_w = int(width * location_min)
        loc_max_w = int(width * location_max - w_width)
        loc_max_w = max(loc_max_w, loc_min_w)

        loc_min_h = int(height * location_min)
        loc_max_h = int(height * location_max - w_height)
        loc_max_h = max(loc_max_h, loc_min_h)

        location = (random.randint(loc_min_w, loc_max_w), random.randint(loc_min_h, loc_max_h))
        transparent.paste(img_watermark, location)

        na = np.array(transparent).astype(float)
        transparent = Image.fromarray(na.astype(np.uint8))

        na = np.array(base_image).astype(float)
        na[..., 3][location[1]: (location[1] + w_height), location[0]: (location[0] + w_width)] *= alpha
        base_image = Image.fromarray(na.astype(np.uint8))
        transparent = Image.alpha_composite(transparent, base_image)

        transparent = transparent.convert('RGB')

        if return_location:
            return transparent, location
        else:
            return transparent
    else:
        raise ValueError(f"Invalid mode argument: {mode}. Must be 'patch' or 'blend'")


def concatenate_images(img1, img2):
    """
    Concatenate two images with a random layout while keeping reasonable aspect.
    The function may concatenate vertically (top/bottom) or horizontally (left/right)
    chosen at random.

    Args:
        img1 (PIL.Image.Image): First image.
        img2 (PIL.Image.Image): Second image.

    Returns:
        PIL.Image.Image: Concatenated RGB image.
    """

    # If areas differ a lot, scale the smaller image closer to the larger one
    area1 = img1.width * img1.height
    area2 = img2.width * img2.height

    if area1 > area2 * 2:
        scale_factor = (area1 / area2) ** 0.5
        new_width = int(img2.width * scale_factor)
        new_height = int(img2.height * scale_factor)
        img2 = img2.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
    elif area2 > area1 * 2:
        scale_factor = (area2 / area1) ** 0.5
        new_width = int(img1.width * scale_factor)
        new_height = int(img1.height * scale_factor)
        img1 = img1.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

    # Randomly choose among: top, right, bottom, left
    choice = random.randint(0, 3)

    # Align a common dimension before concatenation
    if choice in (0, 2):  # vertical
        width = min(img1.width, img2.width)
        img1 = img1.resize((width, img1.height), resample=Image.Resampling.LANCZOS)
        img2 = img2.resize((width, img2.height), resample=Image.Resampling.LANCZOS)
    else:  # horizontal
        height = min(img1.height, img2.height)
        img1 = img1.resize((img1.width, height), resample=Image.Resampling.LANCZOS)
        img2 = img2.resize((img2.width, height), resample=Image.Resampling.LANCZOS)

    # Perform concatenation
    if choice == 0:  # Top
        result = Image.new('RGB', (img1.width, img1.height + img2.height))
        result.paste(img1, (0, 0))
        result.paste(img2, (0, img1.height))
    elif choice == 1:  # Right
        result = Image.new('RGB', (img1.width + img2.width, img1.height))
        result.paste(img1, (0, 0))
        result.paste(img2, (img1.width, 0))
    elif choice == 2:  # Bottom
        result = Image.new('RGB', (img1.width, img1.height + img2.height))
        result.paste(img1, (0, img2.height))
        result.paste(img2, (0, 0))
    else:  # Left
        result = Image.new('RGB', (img1.width + img2.width, img1.height))
        result.paste(img1, (img2.width, 0))
        result.paste(img2, (0, 0))

    return result


def synthesize_poison(
    random_image,
    reference_image,
    watermark,
    watermark_width=50,
    location_min=0.45,
    location_max=0.55,
    alpha=0.2,
    mode='patch',
    return_location=False,
):
    """
    Correct poison synthesis: first watermark a random image, then concatenate
    it with a reference image.

    Args:
        random_image: PIL.Image or path for the random image to be watermarked
        reference_image: PIL.Image for the reference image (no watermark)
        watermark: PIL.Image or path for the watermark
        watermark_width: Patch width (pixels) if mode == 'patch'
        location_min/location_max: Placement range if mode == 'patch'
        alpha: Transparency for blending
        mode: 'patch' or 'blend'
        return_location: If True, also return the watermark location (when patch)

    Returns:
        PIL.Image, or (PIL.Image, (x, y)) when return_location=True
    """

    if return_location:
        result = add_watermark(
            random_image,
            watermark,
            watermark_width=watermark_width,
            location_min=location_min,
            location_max=location_max,
            alpha=alpha,
            return_location=True,
            mode=mode,
        )
        if isinstance(result, tuple):
            watermarked_img, location = result
        else:
            watermarked_img, location = result, None
        concatenated = concatenate_images(watermarked_img, reference_image)
        return concatenated, location
    else:
        watermarked_img = add_watermark(
            random_image,
            watermark,
            watermark_width=watermark_width,
            location_min=location_min,
            location_max=location_max,
            alpha=alpha,
            return_location=False,
            mode=mode,
        )
        return concatenate_images(watermarked_img, reference_image)
