import image_manip

# Coco Example

image_coco = image_manip.ImageManipulate("coco_1.jpg")
image_coco.resize(80, None)
quantized_coco = image_coco.get_colour_quantize_image(components=20, seed=None)

print(image_manip.quantized_to_ascii_str(quantized_coco))
print(image_manip.quantized_to_pixelart_str(quantized_coco))
image_manip.quantized_to_ascii_html(quantized_coco, "coco_ascii.html")
image_manip.quantized_to_pixelart_html(
    quantized_coco, "coco_pixelart.html", pixel_chars="coco! ", pixel_format="cycle"
)
image_manip.quantized_to_pixelart_html(
    quantized_coco, "coco_big_pixelart.html", font_size=24
)

# Fringes Example

image_fringes = image_manip.ImageManipulate("fringes_cutout.png")
image_fringes.resize(150, None)
quantized_fringes = image_fringes.get_colour_quantize_image(components=20, seed=None)

print(image_manip.quantized_to_ascii_str(quantized_fringes))

# Screenshot Example

image_screenshot = image_manip.ImageManipulate("screenshot.png")
image_screenshot.resize(800, None)
quantized_screenshot = image_screenshot.get_colour_quantize_image(
    components=20, seed=None
)

image_manip.quantized_to_cartoon_file(quantized_screenshot, "filtered_ss.png")
