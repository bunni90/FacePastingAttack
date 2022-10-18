#Author: Niklas Bunzel, Lukas Graner

import sys
from os import path, makedirs
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import torch
from model import BiSeNet
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import requests
from skopt import gp_minimize
from scipy.ndimage import gaussian_filter

def write_image(data, file_path=None, domain=[0,1]):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    assert isinstance(data, np.ndarray)
    data = np.squeeze(data, axis=tuple(range(len(data.shape)-3)))

    if len(data.shape)==2:
        # grayscale
        data = data[...,None]

    if len(data.shape)==3 and data.shape[-1] not in [1,3,4] and data.shape[0] in [1,3,4]:
        data = data.transpose([1,2,0])

    assert len(data.shape) == 3 and data.shape[-1] in [1,3,4]

    if data.dtype != np.uint8:
        domain = (data.min(), data.max())

    if domain is not None:
        assert isinstance(domain, (list, tuple)) and len(domain)==2
        data = 255 * (data - domain[0]) / (domain[1] - domain[0])

    data = data.astype(np.uint8)
    pil_img = Image.fromarray(data)
    if file_path is not None:
        pil_img.save(file_path)
    return pil_img

def load_bisenet(weight_path='79999_iter.pth', device='cuda:0'):
    n_classes = 19
    bisenet = BiSeNet(n_classes=n_classes)
    bisenet.to(device)
    bisenet.load_state_dict(torch.load(weight_path))
    bisenet.eval()
    return bisenet

def parse_image(img_pil, bisenet):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        image = img_pil.resize((512, 512), resample=Image.Resampling.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = bisenet(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    return parsing, image

def query_blackbox(filename):
    api_key = "your-API-Key"
    with open(filename, "rb") as f:
        status_code = 500
        r = requests.post(
            "https://api.mlsec.io/api/facerecognition/submit_sample/?api_token=" + api_key + "=" + str(
                source) + "&target=" + str(target), data=f.read())
        response = r.json()
        success = response["success"]
        stealthiness = response["stealthiness"]
        confidence = response["confidence"]
        print(response)
    return confidence, stealthiness, success

def generate_mask(target_img_pil, device):
    bisenet = load_bisenet(device=device)
    mask, target_resized = parse_image(target_img_pil, bisenet)
    mask[(mask == 7) | (mask == 8) | (mask == 9) | (mask > 13)] = 0
    return mask, target_resized

def get_mask_bounding_box(mask):
    coords = np.stack(np.where(mask), -1)
    top, left = coords.min(0)
    bottom, right = coords.max(0)
    return left, top, right, bottom

def build_attack_opimization_position_size_rotate_blur(source, target, source_img_pil, target_img_pil, filename="resutls_eval.jsonl"):
    def attack_opimization_position_size_rotate_blur(args):
        x, y, size_x, size_y, angle, blur = args
        print("x: " + str(x) + " y: " + str(y) + " size_x: " + str(size_x) + " size_y: " + str(size_y)+ " angle: " + str(angle) + " blur: " + str(blur))
        mask, target_resized = generate_mask(target_img_pil, device)
        mask = 255 * (mask > 0)
        left, top, right, bottom = get_mask_bounding_box(mask)
        mask_pil = Image.fromarray(np.uint8(mask))
        mask_pil = mask_pil.crop(box=(left, top, right, bottom))
        target_resized = target_resized.crop(box=(left, top, right, bottom))
        # optimize angle
        mask_pil = mask_pil.rotate(angle=angle)
        target_resized = target_resized.rotate(angle=angle)
        #optimize size
        target_resized.resize((size_x, size_y), resample=Image.Resampling.BILINEAR)
        mask_pil.resize((size_x, size_y), resample=Image.Resampling.BILINEAR)
        #optimize blurring
        mask_np = np.asarray(mask_pil)
        mask_np_blur = gaussian_filter(mask_np, sigma=blur)
        mask_np = np.where(mask_np != 0, mask_np_blur, 0)
        mask_pil = Image.fromarray(np.uint8(mask_np))
        source_bg = source_img_pil.copy()
        source_bg.paste(target_resized, box=(x, y), mask=mask_pil)
        attack_dir = 'attack_imgs'
        makedirs(attack_dir, exist_ok=True)
        attack_file = path.join(attack_dir, str(source) + '_' + str(target) + ".png")
        source_bg.save(attack_file)
        confidence, stealthiness, success = query_blackbox(attack_file)
        score = 1.5 - confidence - min(0.5, stealthiness)
        with open(filename, 'a') as f:
            print('{"source": ' + str(source) + ', "target": ' + str(target) + ', "args": [' + str(x) +', ' + str(y) + ', ' + str(size_x) +', ' + str(size_y) + ', ' + str(angle) + ', ' + str(blur) + '], "prediction": ' + '{"confidence": ' + str(confidence) + ', "stealthiness": ' + str(stealthiness) + ', "success": ' + str(success) + ', "score": ' + str(score) + '}}', file=f)
        return score
    return attack_opimization_position_size_rotate_blur

def attack_optimization(source, target, device):
    source_image_path = 'Path-To-Your-Images/' + str(source) + '_' + str(source) + '.png'
    target_image_path = 'Path-To-Your-Images/' + str(target) + '_' + str(target) + '.png'
    source_img_pil = Image.open(source_image_path)
    target_img_pil = Image.open(target_image_path)

    mask, target_resized = generate_mask(target_img_pil, device)
    # optimize position
    s_x, s_y = source_img_pil.size
    min_x = 0
    max_x = int(s_x - (s_x*0.1))
    min_y = 0
    max_y = int(s_y - (s_y*0.1))
    #optimize size
    min_size_x = int(s_x*0.1)
    max_size_x = int(s_x*0.9)
    min_size_y = int(s_y * 0.1)
    max_size_y = int(s_y * 0.9)
    # optimize angle
    min_angle = -20
    max_angle = 20
    #opimize blur
    min_blur = 0
    max_blur = 20

    optimization_function = build_attack_opimization_position_size_rotate_blur(source, target, source_img_pil, target_img_pil)
    res = gp_minimize(optimization_function, [(min_x, max_x), (min_y, max_y), (min_size_x, max_size_x), (min_size_y, max_size_y), (min_angle, max_angle), (min_blur, max_blur)], verbose=False, n_initial_points=50, n_calls=200)

device = "cuda:0"

for source in range(10):
    for target in range(10):
        source_target = str(source) + "_" + str(target)
        if (source != target):
            print(source_target)
            attack_optimization(source, target, device)