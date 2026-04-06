import re
import base64
import os
import mimetypes

input_file = '正文.md'
output_file = '正文_内嵌图片.md'

with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

def replace_image_with_base64(match):
    alt_text = match.group(1)
    image_path = match.group(2)
    
    # Skip if it's already a web URL or base64 string
    if image_path.startswith('http') or image_path.startswith('data:'):
        return match.group(0)
        
    if os.path.exists(image_path):
        with open(image_path, 'rb') as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
        
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = 'image/png' # fallback
            
        base64_src = f"data:{mime_type};base64,{encoded_string}"
        print(f"Embedded: {image_path}")
        return f"![{alt_text}]({base64_src})"
    else:
        print(f"Warning: Image not found - {image_path}")
        return match.group(0)

# Regex to match ![alt](path)
pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
new_content = re.sub(pattern, replace_image_with_base64, content)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"Successfully created {output_file} with embedded Base64 images.")
