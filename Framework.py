import os
import io as IO
import numpy as np
import torch
import PIL
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils import data
import torch.nn as nn
import torchvision.transforms as T
import torchvision
import csv
import utils
import argparse
import MakePrediction

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

class InputDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        self.imgs = np.array([])

        directory = os.fsencode(root)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            print(filename)
            if(filename.endswith(".txt") or filename.endswith(".csv")):
                continue

            self.imgs = np.append(self.imgs, os.path.join(root, filename))
            imgs_preds[filename] = (os.path.join(root, filename), [0]*7)

    def __getitem__(self, idx):
        # Carrega image
        img = Image.open(self.imgs[idx]).convert("RGB")

        # Transforma imagem em tensor
        if self.transforms is not None:
            img = self.transforms(img)

        return img, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)


def get_transform():
    transforms = []

    transforms.append(T.Resize((224, 224)))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return T.Compose(transforms)

# Pega model de classificacao Resnet34
def get_classification_model():
    model = torchvision.models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

def set_dataset():
    dataset = InputDataset(root=args.path, transforms=get_transform())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    return data_loader


def go(model, data_loader_test, class_id):
    num_classes = 2
    model.to(device)

    predictions = list()
    indexes = list()

    model.eval()
    with torch.no_grad():
        output_list = list()

        # Iterating over test batches.
        for it, data in enumerate(data_loader_test):
            # Obtaining images and labels for batch.
            inps, img_path = data
            # GPU casting. In CPU version comment the following line.
            inps = inps.cuda()
            print(img_path)
            # Forwarding inps through NN.
            output = model(inps)

            # Getting labels and predictions from last epoch.
            img_name = img_path[0].split("/")[-1]
            probs = nn.functional.softmax(output, dim=1)
            imgs_preds[img_name][1][i] = probs.max(1)


def write_data():
    class_names = ["Terreno", "Infraestrutura", "Vedacao Vertical", "Coberturas", "Esquadrias", "Revestimentos", "Paisagismo"]

    for k, v in imgs_preds.items():
        print("k:", k.split("/"))
        file_path = v[0].split(".")[0] + ".txt"
        print(file_path)

        with open(file_path, 'w') as file:
            for i, name in enumerate(class_names):
                file.write(name + ": " + str(v[1][i][1][0].item()) + " -> " + ("%.6f" % (v[1][i][0][0].item())) + "\n")
		
    with open(os.path.join(args.path, 'predictions.csv'), 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(["ID","Outros","Outros Confidence","Obra nao iniciada (terreno)","Obra nao iniciada (terreno) confidence","Infra-estrutura","Infra-estrutura confidence","Vedacao vertical","Vedacao vertical confidence","Coberturas","Coberturas confidence","Esquadrias","Esquadrias confidence","Revestimentos externos","Revestimentos externos confidence","Pisos externos e paisagismo","Pisos externos e paisagismo confidence"])

        for k, v in imgs_preds.items():
            line = [k,0,0]
            for i in range(7):
                line += [int(v[1][i][1][0].item()),v[1][i][0][0].item()]
            spamwriter.writerow(line)



#################### MAIN #####################
imgs_preds = {
    # "dummy": ("IMG_PATH", [0]*7)
}

print(torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_names = ["terreno", "infraestrutura", "vedacao_vertical", "coberturas", "esquadrias", "revestimentos", "paisagismo"]

model = get_classification_model()
data_loader = set_dataset()

for i, target in enumerate(model_names):
    model.load_state_dict(torch.load("dream_team/Resnet34-" + target + ".pt"))
    go(model, data_loader, i)

write_data()
MakePrediction.main(args.path, 0.7)
