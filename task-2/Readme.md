# Face recognition and Unsupervised Clustering

Crop faces from images using MTCNN and extracts 512 dimensional embeddings using Pytorch's InceptionResnetV1. 
Train KMeans clustering model to faces into clusters

## Setup
Note: Make sure to use venv for this project as facenet_pytorch is a huge module <br>
1. After cloning the repository create a venv folder and install all dependencies
2. Obtain dataset (zipfile) from https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset and place it into the working directory.
3. Run ```extract_data.py``` to extract data from the zipfile
4. Edit line 5 of ```example.py``` to provide file path to the image that you'd like to identify
   ```python
   # Example
   cluster_pred = find_corresponding_cluster('Celebrity Faces Dataset/Leonardo DiCaprio/005_7fe5b764.jpg')
   ```
5. Run ```example.py```, If being run for the first time, it will take around 20 minutes to execute as extracting faces using MTCNN and embeddings using Resnet takes quite a bit of time. However, after the first run <br>
  these results are cached into data.bin and embeddings.bin respectively, so further runs execute instantaneously.
6. It prints the person to whom the face belongs to and the mapping of all people along with how accurately each directory (person) has been clustered.

## How it works
1. On the first run, it extracts all face arrays using MTCNN from all files in "Celebrity Faces Dataset" and stores it in data.bin
2. Extracts face embeddings from all face arrays using InceptionResnetV1 and stores it in embeddings.bin
3. Using KMeans it clusters all the embeddings. (Identifies cluster centers)
4. The function ```sort_into_groups``` identifies which cluster group belongs to which person (dir name) and accuracy of the cluster (defined as the number of correctly 
    identified persons in cluster / cluster size)
5. The function ```find_corresponding_cluster(img_file)``` calls predict on the fitted Kmeans model to identify cluster id.
6. Subsequent runs utilize the embeddings stored in embeddings.bin to instantly output results

## Features Implemented
* Use MTCNN to crop images*
*  Normalize faces
* Feature extraction using InceptionResnetv1
* Cluster results using KMeans
* View extracted faces as groups (along with how good the cluster is)

## Dataset
https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset
