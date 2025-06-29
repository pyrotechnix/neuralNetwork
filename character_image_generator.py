from PIL import Image, ImageDraw, ImageFont
import os
import string

# Settings
size = 28
img_size = (size, size)
font_size = 32
output_dir = "character_images"
#You will need to download the font and provide the path
font_path = "Fonts\\Oxygen-Bold.ttf"
#Don't forget to update the encoding!
encoding = "oxygen_bold_32"

'''
This program will generate a list of characters as images.
Set the font size and path above, adjust the filename encoding as desired
At the start of the loop, you can name the character set to be used
'''

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the font
font = ImageFont.truetype(font_path, font_size)

upperLetters = [chr(i) for i in range(65,91)]
lowerLetters = [chr(i) for i in range(97,123)]
numbers = [chr(i) for i in range(48,58)]
characters = upperLetters + lowerLetters + numbers

''' Loop through chosen character set '''
for letter in characters:  
    print(f"Creating image of: {letter}")
    # Create a new white image
    img = Image.new("L", img_size, color="white")
    draw = ImageDraw.Draw(img)

    # Get size of text to center it
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ( (0-bbox[0]+((size-text_width)//2)),
                 (0-bbox[1]+((size-text_height)//2)))
    # Draw the character
    draw.text(position, letter, fill="black", font=font)

    # Save image
    img.save(os.path.join(output_dir, f"{encoding}_{letter}.png"))

#Comment out the code below to exclude symbols
symbols = ['!', '@', '#', '$', '%', '&', '*', '+', '-', '?', '<', '>']
symwords = ['exclmark', 'at', 'hash', 'dollar', 'percent', 'ampersand', 'asterisk', 'plus', 'minus', 'quesmark', 'lessthan', 'greaterthan']
for i in range(len(symbols)):
    print(f"Creating image of: {symwords[i]}")
    # Create a new white image
    img = Image.new("L", img_size, color="white")
    draw = ImageDraw.Draw(img)

    # Get size of text to center it
    bbox = draw.textbbox((0, 0), symbols[i], font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ( (0-bbox[0]+((size-text_width)//2)),
                 (0-bbox[1]+((size-text_height)//2)))
    # Draw the character
    draw.text(position, symbols[i], fill="black", font=font)

    # Save image
    img.save(os.path.join(output_dir, f"{encoding}_{symwords[i]}.png"))
