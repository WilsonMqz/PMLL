import json
import os
from clip import clip
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random


single_template = ["a photo of a {}."]


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


def expand_probs(subset_indices, subset_probs, total_classes):
    full_probs = [0.0] * total_classes
    for idx, prob in zip(subset_indices, subset_probs):
        full_probs[idx] = prob
    return full_probs

seed = 101
np.random.seed(seed)
random.seed(seed)


json_path = 'split_zhou_Caltech101.json'
image_root = 'D:\\code\\python\\weakly_supervised\\newcode\\VLMPrivacy\\data\\caltech-101\\101_ObjectCategories'
output_path = 'caltech101_clip_results_ratio_0.05_cl_1.json'
stochastic_ratio = 0.05

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = clip.load('ViT-B/32')


with open(json_path, 'r') as f:
    data = json.load(f)

all_entries = data['train'] + data.get('val', []) + data.get('test', [])
class_name_to_index = {}
index_to_class_name = {}

for entry in all_entries:
    _, label_id, class_name = entry
    if class_name not in class_name_to_index:
        class_name_to_index[class_name] = label_id
        index_to_class_name[label_id] = class_name


num_classes = len(class_name_to_index)
all_classs = range(num_classes)
conceal_labels = [num_classes-1]
not_conceal_num = num_classes - len(conceal_labels)

stochastic_num = int(not_conceal_num * stochastic_ratio)


def get_text_inputs(class_names, model):
    templates = single_template
    run_on_gpu = torch.cuda.is_available()
    with torch.no_grad():
        label_embedding = []
        for category in class_names:
            texts = [
                template.format(
                    processed_name(category, rm_dot=True), article=article(category)
                )
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]
            texts = clip.tokenize(texts)
            if run_on_gpu:
                texts = texts.cuda()
                model = model.cuda()
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            label_embedding.append(text_embedding)
        label_embedding = torch.stack(label_embedding, dim=1)
        if run_on_gpu:
            label_embedding = label_embedding.cuda()

    label_embedding = label_embedding.t()
    return label_embedding


train_results = []
for entry in tqdm(data['train'], desc="Predicting with CLIP"):
    image_path, label_id, label_name = entry
    full_image_path = os.path.join(image_root, image_path)

    try:
        image = preprocess(Image.open(full_image_path).convert("RGB")).unsqueeze(0).to(device)
        stochastic_labels = random.sample(range(not_conceal_num), stochastic_num)
        stochastic_labels.extend(conceal_labels)
        with torch.no_grad():
            image_features = model.encode_image(image)
            # text_features = model.encode_text(text_inputs)
            if label_id in stochastic_labels:
                indexs = stochastic_labels
                s_label = 0
            else:
                indexs = list(set(all_classs) - set(stochastic_labels))
                s_label = 1
            class_names = [index_to_class_name[i] for i in indexs]

            text_features = get_text_inputs(class_names, model)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predicted_index = similarity[0].argmax().item()
            class_name = class_names[predicted_index]
            predicted_index = class_name_to_index[class_name]
            probs = expand_probs(indexs, similarity[0].tolist(), num_classes)

        entry_with_prediction = entry + [s_label] + [predicted_index] + [stochastic_labels] + [probs]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        entry_with_prediction = entry + [-1]  # -1 indicates error case

    train_results.append(entry_with_prediction)


output_data = {
    'train': train_results,
    'val': data.get('val', []),
    'test': data.get('test', []),
    'class_name_to_index': class_name_to_index
}

with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)
