from mtcnn import MTCNN
import numpy as np
import keras
from keras.models import load_model
from PIL import Image
import os
import matplotlib.pyplot as plt
import pickle
import torch
from sklearn.cluster import KMeans

from facenet_pytorch import InceptionResnetV1

def extract_face(img_dir):
    img = Image.open(img_dir)
    img_arr = np.asarray(img)

    face_detector = MTCNN()
    faces = face_detector.detect_faces(img_arr)

    x1, y1, w, h = faces[0]['box']

    x1, y1 = abs(x1), abs(y1)

    face = img_arr[y1:y1+h, x1:x1+w]

    image = Image.fromarray(face)
    image = image.resize((160, 160))
    face_array = np.asarray(image)

    face_array = (face_array)

    return face_array

def extract_faces(data_dir):
    faces = []
    dirs_of_face = []

    for dirs in os.listdir('Celebrity Faces Dataset'):
        for file in os.listdir(f'Celebrity Faces Dataset/{dirs}'):
            try:
                print(f"Attempting to extract from file /{dirs}/{file}")
                face = extract_face(f'Celebrity Faces Dataset/{dirs}/{file}')
                faces.append(face)
                dirs_of_face.append(dirs)
            except Exception as e:
                print(e)
                print(f'Failed to extract face from file /{dirs}/{file}')

    return faces

def face_dir_mapping(data_dir):
    face_dir_mapping = []
    for dirs in os.listdir(f'Celebrity Faces Dataset/'):
        face_dir_mapping.extend([dirs for _ in range(len(os.listdir(f'Celebrity Faces Dataset/{dirs}')))])

    return face_dir_mapping

def get_embeddings(data_dir):
    if not os.path.exists('data.bin'):
        extracted_faces = extract_faces(data_dir)
        with open('data.bin', 'wb') as f:
            pickle.dump(extracted_faces, f)
    
    else:
        with open('data.bin', 'rb') as f:
            extracted_faces = pickle.load(f)

    model = InceptionResnetV1(pretrained='vggface2').eval()
    
    if not os.path.exists('embeddings.bin'):
        embeddings = []
        for j, face in enumerate(extracted_faces):
            tensor = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1)
            tensor = (tensor / 127.5) - 1.0
            tensor = tensor.unsqueeze(0)

            print(f'Embeddings for face: {j}')
            result = model(tensor)
            embeddings.append(result.squeeze(0))
        
        with open('embeddings.bin', 'wb') as f:
            pickle.dump(embeddings, f)

        return embeddings

    else:
        with open('embeddings.bin', 'rb') as f:
            return pickle.load(f)

def cluster(data_dir):
    embeddings = get_embeddings(data_dir)

    model = KMeans(n_clusters=17)
    return model.fit_predict(np.array([k.detach().numpy() for k in embeddings])), model.cluster_centers_

clusters_for_each_img, centers = cluster('Celebrity Faces Dataset')
print(centers)