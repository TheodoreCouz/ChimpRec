from PIL import Image, ImageDraw, ImageFont

def resize_with_aspect_ratio(image, target_width=None, target_height=None):
    if target_width is None and target_height is None:
        return image
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    if target_width is not None:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    return image.resize((new_width, new_height), resample=Image.LANCZOS)

def recolor_overlay(overlay, new_color):
    r, g, b, a = overlay.split()
    color_image = Image.new("RGBA", overlay.size, new_color + (0,))
    color_image.putalpha(a)
    return color_image

def paste_with_transparency_and_label(
    background_path,
    overlay_path,
    position,
    output_path,
    target_width=None,
    target_height=None,
    new_color=None,
    label_text=None,
    font_path=None,
    font_size=20,
    text_color=(255, 255, 255)
):
    background = Image.open(background_path).convert("RGBA")
    overlay = Image.open(overlay_path).convert("RGBA")

    # Resize overlay
    overlay = resize_with_aspect_ratio(overlay, target_width, target_height)

    # Recolor overlay
    if new_color is not None:
        overlay = recolor_overlay(overlay, new_color)

    # Paste overlay
    temp = Image.new("RGBA", background.size, (0, 0, 0, 0))
    temp.paste(overlay, position, overlay)
    result = Image.alpha_composite(background, temp)

    # Draw label
    if label_text:
        draw = ImageDraw.Draw(result)

        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]


        # Center text on top of the overlay
        overlay_x, overlay_y = position
        overlay_width, overlay_height = overlay.size
        text_x = overlay_x + (overlay_width - text_width) // 2
        text_y = overlay_y + (overlay_height//2 - 1.5*text_height) - 20

        draw.text((text_x, text_y), label_text, font=font, fill=new_color)

    # Save result
    result.save(output_path)

# Example usage
paste_with_transparency_and_label(
    background_path="C:/Users/Theo/Documents/Unif/ChimpRec/Code/Annotation_visu/3.png",
    overlay_path="C:/Users/Theo/Documents/Unif/ChimpRec/Code/Annotation_visu/Untitled.png",
    position=(0, 0),
    output_path="C:/Users/Theo/Documents/Unif/ChimpRec/Code/Annotation_visu/result.png",
    target_width=400,
    new_color=(0, 0, 0),  # Green
    label_text="Exezfazefraezrample",
    font_path=None,  # Or path to .ttf file, e.g., "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font_size=24,
    text_color=(255, 255, 255)
)