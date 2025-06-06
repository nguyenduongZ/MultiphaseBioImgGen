import torch
import random
import numpy as np

from PIL import Image, ImageDraw, ImageFont

def simple_slugify(text: str, max_length = 255):
    return text.replace('-', '_').replace(',', '').replace(' ', '_').replace('|', '--').strip('-_./\\')[:max_length]

def create_grid_image(original_image, prompt, generated_image, grid_size=(300, 300), font_size=10):
    border_color = "black"
    border_width = 2

    text_height_estimate = font_size * 3
    total_height = grid_size[1] + text_height_estimate
    total_width = grid_size[0] * 2
    
    grid = Image.new("RGB", (total_width, total_height), "white")
    draw_grid = ImageDraw.Draw(grid)
    
    def center_image(image, frame_size):
        frame = Image.new("RGB", frame_size, "white")
        W, H  = image.size
        offset_x = (frame_size[0] - W) // 2
        offset_y = (frame_size[1] - H) // 2
        
        if W <= frame_size[0] and H <= frame_size[1]:
            frame.paste(image, (offset_x, offset_y))
        else:
            image.thumbnail(frame_size, Image.LANCZOS)
            offset_x = (frame_size[0] - image.size[0]) // 2
            offset_y = (frame_size[1] - image.size[1]) // 2
            frame.paste(image, (offset_x, offset_y))
            
        return frame

    original_frame = center_image(original_image, grid_size)
    generated_frame = center_image(generated_image, grid_size)
    
    prompt_frame = Image.new("RGB", (total_width, text_height_estimate), "white")
    draw_prompt = ImageDraw.Draw(prompt_frame)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
        
    max_W = total_width - 40
    words = prompt.split()
    current_line = ""
    lines = []
    for word in words:
        test_line = current_line + word + " "
        text_bbox = draw_prompt.textbbox((0, 0), test_line, font=font)
        if text_bbox[2] - text_bbox[0] <= max_W:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    lines.append(current_line.strip())

    text_height = len(lines) * (font_size + 5)
    text_y = (text_height_estimate - text_height) // 2
    for line in lines:
        text_bbox = draw_prompt.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (total_width - text_width) // 2
        draw_prompt.text((text_x, text_y), line, fill="black", font=font)
        text_y += font_size + 5
    
    grid.paste(prompt_frame, (0, 0))
    grid.paste(original_frame, (0, text_height_estimate))
    grid.paste(generated_frame, (grid_size[0], text_height_estimate))
    
    draw_grid.rectangle([(0, 0), (total_width - 1, text_height_estimate - 1)], outline=border_color, width=border_width)
    draw_grid.rectangle([(0, text_height_estimate), (grid_size[0] - 1, total_height - 1)], outline=border_color, width=border_width)
    draw_grid.rectangle([(grid_size[0], text_height_estimate), (total_width - 1, total_height - 1)], outline=border_color, width=border_width)

    return grid

def setup_seed(seed: int, deterministic: bool = None, benchmark: bool = None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark