from mtcnn import MTCNN
import numpy as np
import keras
from keras.models import load_model
from PIL import Image
import os
import matplotlib.pyplot as plt
import pickle
import torch
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from facenet_pytorch import InceptionResnetV1

def extract_face(img_dir):
    '''
    Uses MTCNN's pretrained model to place bounding boxes on faces and crop out only faces from images, returns this cropped out 

    Parameters
    --------------------------------
    img_dir: file path to image
    
    Returns
    --------------------------------
    np.ndarray
    '''
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

def extract_faces():
    '''
    Extract face array of all faces by going through all files in 'Celebrity Faces Dataset'
    Note: The directory containing all images must be named 'Celebrity Faces Dataset'

    Parameters
    ---------------------
    None

    Returns
    ---------------------
    list
    '''
    faces = []
    dirs_of_face = []

    failed = []

    for dirs in os.listdir('Celebrity Faces Dataset'):
        for file in os.listdir(f'Celebrity Faces Dataset/{dirs}'):
            try:
                print(f"Attempting to extract from file /{dirs}/{file}")
                face = extract_face(f'Celebrity Faces Dataset/{dirs}/{file}')
                faces.append(face)
                dirs_of_face.append(dirs)
            except Exception as e:
                print(e)
                failed.append(f'{dirs}/{file}')
                print(f'Failed to extract face from file /{dirs}/{file}')
    
    for files in failed:
        os.remove(f'Celebrity Faces Dataset/{files}')

    return faces

def face_dir_mapping():
    '''
    Returns the name of the person corresponding to elements in face array

    Parameters 
    ------------------------------

    Returns
    ------------------------------
    list
    '''
    face_dir_mapping = []
    for dirs in os.listdir('Celebrity Faces Dataset'):
        face_dir_mapping.extend([dirs for _ in range(len(os.listdir(f'Celebrity Faces Dataset/{dirs}')))])

    return face_dir_mapping

def get_embeddings():
    '''
    Uses pytorch's InceptionResnetV1 model to extract embeddings (512 dimensional) from face_array and caches the result in embeddings.bin for future use, 
    when run for the first time, it will store all extracted face arrays in data.bin

    Parameters
    -----------------------
    None

    Returns
    -----------------------
    list
    '''
    if not os.path.exists('data.bin'):
        extracted_faces = extract_faces()
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

def cluster():
    '''
    Fits embeddings into Kmeans clustering algorithm and returns predicted cluster indices of all images, cluster centers, fitted model

    Parameters
    --------------------------
    None

    Returns
    tuple
    '''
    embeddings = get_embeddings()

    clustering_alg = KMeans(n_clusters=17)
    
    return clustering_alg.fit_predict(np.array([k.detach().numpy() for k in embeddings])), clustering_alg.cluster_centers_, clustering_alg

def sort_into_groups():
    '''
    Identifies which cluster group belongs to which person (dir name) and accuracy of the cluster (defined as the number of correctly 
    identified persons in cluster / cluster size)

    Parameters
    -----------------------------------
    None

    Returns 
    dict
    '''
    predicted_clusters, centers, alg = cluster()
    directory_img_mapping = face_dir_mapping()

    cluster_idx_to_person = {}
    border, start = 0, 0
    for k in range(len(directory_img_mapping)):
        if k == len(directory_img_mapping) - 1:
            cluster_res = list(predicted_clusters[start: len(directory_img_mapping)])
            cluster_idx_to_person[directory_img_mapping[start]] = [max(set(cluster_res), key=cluster_res.count), 1 - (len(cluster_res) - cluster_res.count(max(set(cluster_res), key=cluster_res.count)))/len(cluster_res)]
            break
       
        if directory_img_mapping[k] != directory_img_mapping[k + 1]:
            border = k
            cluster_res = list(predicted_clusters[start: border + 1])
            cluster_idx_to_person[directory_img_mapping[k]] = [max(set(cluster_res), key=cluster_res.count), 1 - (len(cluster_res) - cluster_res.count(max(set(cluster_res), key=cluster_res.count)))/len(cluster_res)]

            start = k + 1

    return cluster_idx_to_person

def find_corresponding_cluster(img_file):
    '''
    Provided a custom image of any person belonging to the 17 classes in dataset, returns predicted cluster index. Use 
    this function along with sort_into_groups to extract the name of the person.

    Parameters
    -------------------------------
    img_file: image file path

    Returns
    -------------------------------
    np.ndarray
    '''
    ext = extract_face(img_file)
    tensor = torch.tensor(ext, dtype=torch.float32).permute(2, 0, 1)
    tensor = (tensor / 127.5) - 1.0
    tensor = tensor.unsqueeze(0)
    embeds = InceptionResnetV1(pretrained='vggface2').eval()(tensor)
    embeds = embeds.squeeze(0)

    res, centers, model = cluster()
    return model.predict(np.array([embeds.detach().numpy()]))
