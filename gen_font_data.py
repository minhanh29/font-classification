from PIL import Image, ImageDraw, ImageFont
import string
import os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

FILE_TEXT = "./SRNet-Datagen/Synthtext/data/texts.txt"
FONT_DIR = "./fonts"
FONT_FILE = "./fonts/font_list.txt"
OUT_DIR = "./results/text_classification_data"
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)


class TextDataset(Dataset):
    def __init__(self, file_text, font_dir, font_file, train=True):
        with open(file_text, "r", encoding="utf-8") as f:
            text_list = f.readlines()
        self.text_list = [line.strip() for line in text_list]
        print("Got", len(self.text_list), "words")

        self.font_dir = font_dir
        with open(font_file, "r", encoding="utf-8") as f:
            font_list = f.readlines()
        self.font_list = [f.strip().split("|") for f in font_list]
        print("Got", len(self.font_list), "fonts")

        self.org_canvas_width = 350
        self.org_canvas_height = 350
        self.shape = (self.org_canvas_width, self.org_canvas_height)
        self.padding = 10

        self.fontsize = 50
        self.img_center = (self.org_canvas_width//2, self.org_canvas_height//2)
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Grayscale(1),
            # transforms.Resize(size=64)
        ])

        self.train = train

        s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
        s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
        self.char_list = s0 + s1

    def __len__(self):
        if self.train:
            return len(self.text_list)
        else:
            return 640

    def __getitem__(self, idx):
        # text = random.choice(self.char_list)
        label, font_path = random.choice(self.font_list)
        img1 = self.get_from_label(label, font_path, idx)

        target = 0
        if random.random() < 0.5:
            img2 = self.get_from_label(label, font_path, idx)
            target = 1
        else:
            neg_label, font_path = random.choice([f for f in self.font_list if f[0] != label])
            img2 = self.get_from_label(neg_label, font_path, idx)
            target = 0

        return img1, img2, target

    def get_from_label(self, label, font_path, idx):
        img = None
        font_path = os.path.join(self.font_dir, font_path)
        while img is None:
            text = random.choice(self.text_list)
            img = self.gen_data_sample(text, font_path, idx)
        img = img.float()/255.

        return img

    def gen_data_sample(self, text, font_path, idx):
        img = Image.new('RGB', self.shape, (0, 0, 0))
        draw = ImageDraw.Draw(img)
        myFont = ImageFont.truetype(font_path, self.fontsize)
        draw.text(self.img_center, text, font=myFont, fill=(255, 255, 255), anchor="mm")
        rect = img.getbbox()

        if rect is None:
            return None
        canvas_width, canvas_height = (int(rect[2] - rect[0]), int(rect[3] - rect[1]))
        canvas_width += self.padding
        canvas_height += self.padding

        canvas_width = int(canvas_width)
        canvas_height = int(canvas_height)

        img = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Custom font style and font size
        draw.text((canvas_width//2, canvas_height//2), text,
                  font=myFont, fill=(255, 255, 255), anchor="mm")

        if random.uniform(0., 1.) < 0.15:
            img = img.rotate(random.randint(-20, 20), expand=True)

        # img.save(os.path.join(OUT_DIR, f"{idx}.png"))

        return self.transform(img)


def main():
    dataset = TextDataset(FILE_TEXT, FONT_DIR, FONT_FILE)
    for cnt in range(10):
        dataset.__getitem__(cnt)


if __name__ == "__main__":
    main()
