import torch
from torchmetrics import Accuracy
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from gen_font_data import TextDataset
from model import FontClassifier
from tqdm import tqdm

torch_blur = transforms.GaussianBlur((5, 5))
torch_v_flip = transforms.RandomVerticalFlip(0.2)
torch_h_flip = transforms.RandomHorizontalFlip(0.2)

batch_size = 32
height = 64


def custom_collate(batch):

    img1_batch, img2_batch, label_batch = [], [], []

    w_sum = 0
    for item in batch:
        t_b= item[0]
        h, w = t_b.shape[1:]
        scale_ratio = height / h
        w_sum += int(w * scale_ratio)

    to_h = height
    to_w = w_sum // batch_size
    to_w = int(round(to_w / 8)) * 8
    to_w = max(to_h, to_w)
    to_scale = (to_h, to_w)
    torch_resize = transforms.Resize(to_scale)
    cnt = 0
    for item in batch:
        img1, img2, label = item

        if random.uniform(0., 1.) < 0.8:
            p_t = random.randint(0, 70)
            p_b = random.randint(0, 70)
            p_l = random.randint(0, 70)
            p_r = random.randint(0, 70)
            img1 = torch.nn.functional.pad(img1, (p_l, p_r, p_t, p_b))
        img1 = torch_resize(img1)

        if random.uniform(0., 1.) < 0.15:
            img1 = torch_blur(img1)

        if random.uniform(0., 1.) < 0.15:
            img1 = img1 + (0.02**0.5)*torch.randn(1, to_h, to_w)
            img1 = torch.clamp(img1, 0., 1.)

        img1 = torch_h_flip(img1)
        img1 = torch_v_flip(img1)

        if random.uniform(0., 1.) < 0.8:
            p_t = random.randint(0, 70)
            p_b = random.randint(0, 70)
            p_l = random.randint(0, 70)
            p_r = random.randint(0, 70)
            img2 = torch.nn.functional.pad(img2, (p_l, p_r, p_t, p_b))
        img2 = torch_resize(img2)

        if random.uniform(0., 1.) < 0.15:
            img2 = torch_blur(img2)

        if random.uniform(0., 1.) < 0.15:
            img2 = img2 + (0.02**0.5)*torch.randn(1, to_h, to_w)
            img2 = torch.clamp(img2, 0., 1.)

        img2 = torch_h_flip(img2)
        img2 = torch_v_flip(img2)

        # img1_to_save = F.to_pil_image(img1)
        # img1_to_save.save(f"./sample/img1_{cnt}.png")

        # img2_to_save = F.to_pil_image(img2)
        # img2_to_save.save(f"./sample/img2_{cnt}.png")

        img1_batch.append(img1)
        img2_batch.append(img2)

        label_batch.append(label)
        cnt += 1

    img1_batch = torch.stack(img1_batch)
    img2_batch = torch.stack(img2_batch)
    label_batch = torch.tensor(label_batch)

    return [img1_batch, img2_batch, label_batch]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILE_TEXT = "./texts.txt"
FONT_DIR = "./fonts/"
FONT_FILE = "./fonts/font_list.txt"

epochs = 5
global_step = 0

train_dataset = TextDataset(FILE_TEXT, FONT_DIR, FONT_FILE, train=True)
test_dataset = TextDataset(FILE_TEXT, FONT_DIR, FONT_FILE, train=False)

num_classes = 206
model = FontClassifier(1, num_classes).to(device)
for p in model.parameters():
    p.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=0.001)
# checkpoint = torch.load("./weights_mac/4970.pth", map_location="cpu")
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optim'])
# global_step = checkpoint["global_step"]

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate)

loss_func = torch.nn.functional.binary_cross_entropy
if not os.path.exists("./weights"):
    os.makedirs("./weights")
torch.save({
    "model": model.state_dict(),
    "optim": optimizer.state_dict(),
    "global_step": global_step,
}, f"./weights/{global_step}.pth")
accuracy = Accuracy(threshold=0.5)

for epoch in range(epochs):
    print("Training...")
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    pbar = tqdm(train_dataloader)
    pbar.set_description(f"Epoch {epoch}/{epochs}")
    total_loss = 0.
    total_acc = 0.
    cnt = 0
    for img1, img2, target in pbar:
        global_step += 1
        optimizer.zero_grad()
        img1 = img1.float().to(device)
        img2 = img2.float().to(device)
        target = target.to(device)
        target = torch.unsqueeze(target, dim=-1)
        target_float = target.float()
        pred = model.forward_pair(img1, img2)

        # target = target.float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target_float)
        loss.backward()
        optimizer.step()

        acc = accuracy(pred, target)
        total_loss += loss.item()
        total_acc += acc.numpy()
        cnt += 1
        pbar.set_postfix({
            "loss": total_loss/cnt,
            "accuracy": total_acc/cnt,
        })

        if cnt % 200 == 0:
            torch.save({
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "global_step": global_step,
            }, os.path.join("./weights_mac", f"{global_step}.pth"))

    print("Evaluating...")
    pbar = tqdm(test_dataloader)
    model.eval()
    total_loss = 0.
    total_acc = 0.
    cnt = 0
    for img, target in pbar:
        img1 = img1.float().to(device)
        img2 = img2.float().to(device)
        target = target.to(device)
        target = torch.unsqueeze(target, dim=-1)
        target_float = target.float().to(device)
        pred = model.forward_pair(img1, img2)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target_float)
        acc = accuracy(pred, target)
        total_loss += loss.item()
        total_acc += acc.numpy()
        cnt += 1
        pbar.set_postfix({
            "loss": loss.item(),
            "accuracy": total_acc/cnt,
        })

    print("Eval loss:", total_loss/cnt)
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "global_step": global_step,
    }, os.path.join("./weights_mac", f"{global_step}.pth"))

