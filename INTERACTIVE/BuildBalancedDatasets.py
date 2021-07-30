import os
import io as IO
import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--check', type=int, required=True)
parser.add_argument('--stats', type=int, required=True)
args = parser.parse_args()

root = args.path

basedir = os.path.dirname(os.path.join(root, "BALANCED_DATASETS/"))
if not os.path.exists(basedir):
    os.makedirs(basedir)

classes = ["terreno", "infraestrutura", "vedacao_vertical", "coberturas", "esquadrias", "revestimentos", "paisagismo"]

for c in classes:
    dir = os.path.dirname(os.path.join(root, "BALANCED_DATASETS/{}/".format(c)))
    if not os.path.exists(dir):
        os.makedirs(dir)

with IO.open(os.path.join(root, "labels_treino.csv"), "r", encoding="UTF-8") as file:
    data = np.genfromtxt(file, delimiter=",", dtype=None, encoding=None)

lista = []

for row in data[1:]:
    if row[1]:
        row = [str(r) for r in row]
        lista.append(list(row))


images = np.array(lista)

sum_labels = []

for i in range(2,9):
    sum = np.sum(images[:,i].astype(int))
    sum_labels.append(sum)

# monta o dataset de cada uma das 7 classes
for i, c in zip(range(7),classes):

    np.random.shuffle(images)

    dataset = []
    num_labels = [0,0,0,0,0,0,0]

    # calcula o limite de amostras para manter a proporcao meio a meio
    if min(sum_labels) < sum_labels[i] * 1/6:
        num_false_samples = min(sum_labels)
    else:
        num_false_samples = sum_labels[i] * 1/6

    num_true_samples = num_false_samples * 6

    limit = []

    for j in range(7):
        if j == i:
            limit.append(num_true_samples)
        else:
            limit.append(num_false_samples)

    # itera por todas as imagens adicionando imagens no dataset ate atingir a proporcao
    # de 1/2 para a classe c e 1/12 para cada uma das 6 classes restantes

    for img in images:
        if args.check:
            if not os.path.exists(os.path.join(root, "IMAGES/" + img[0].split("-")[0] + "/" + img[0].split("-")[1] + ".jpeg")):
                continue
        if np.all(num_labels + img[2:].astype(int) <= limit):
                dataset.append(img)
                num_labels += img[2:].astype(int)
        
    # salva csv com o dataset especifico para a classe
    with open(os.path.join(root, 'BALANCED_DATASETS/'+ c + "/" + c + "_labels_treino.csv"), 'w') as f:
        write = csv.writer(f)
        write.writerows(dataset)

    if args.stats:

        # dataset da classe c
        print(c)
        # quantidade de amostras de cada classe no dataset da classe c 
        print(num_labels)
        
        # ordem dos indices

        # 0,terreno
        # 1,infraestrutura
        # 2,vedacao_vertical
        # 3,coberturas
        # 4,esquadrias
        # 5,revestimentos
        # 6,paisagismo

        # tamanho total do dataset c
        print(len(dataset))

        print("__________________________________________")



