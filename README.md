# CRUx round 3 tasks

## Installation
1. Clone the repository
2. It is recommended to create a seperate venv folder for task-2 as the dependencies are pretty huge.
3. Refer to the README files under each task for instructions on setup and running.

## Task-1
Created an mlp from scratch and implemented grid search and early stopping. Tested the model of Fashion MNIST.

## Task-2
Extracted faces using MTCNN and embeddings using InceptionResnetV1, I wanted to use FaceNet keras for this but there seemed to be some depenedency issue I could not resolve.
Trained KMeans clustering to cluster extracted embeddings.

## Task-3
Cleaned data using NLTK and used TF-IDF vectorizer to vectorize data. Trained 4 different classification algorithms and chose RandomForestClassifier for best performance and created a function for 
the user to categorize their own resume.

## References
http://neuralnetworksanddeeplearning.com/ for the math behind MLP's <br>
https://scikit-learn.org/stable/ SKLearn docs <br>
https://github.com/timesler/facenet-pytorch Facenet-pytorch repo
