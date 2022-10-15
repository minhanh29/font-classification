import os


FONT_DIR = "./fonts"
OUT_PATH = "./fonts/font_list.txt"

font_d_list = os.listdir(FONT_DIR)
font_d_list = [d for d in font_d_list if os.path.isdir(os.path.join(FONT_DIR, d))]

result = []
for d in font_d_list:
    d_path = os.path.join(FONT_DIR, d)
    font_list = os.listdir(d_path)
    font_list = [f for f in font_list if ".ttf" in f or ".otf" in f]
    font_list.sort()
    font_list = [os.path.join(d, f) for f in font_list]
    result.extend(font_list)

print("Got", len(result), "fonts")

with open(OUT_PATH, "w") as f:
    for i, font in enumerate(result):
        f.write(f"{i}|{font}\n")


