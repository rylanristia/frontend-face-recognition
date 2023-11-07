from django.http import request
from django.shortcuts import render, redirect
from django.http import JsonResponse
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_exempt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pyswarm import pso
import json
import base64
import os
import io 
import imageio
import numpy as np
import cv2
import pickle
import face_recognition
import datetime
import requests as req

FILE_STORE = "face_recognition_model.pkl"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# AUTHENTICATION
# AUTHORED BY RYLANRISTIA
def auth(request):

    xemail      = request.POST.get('xemail')
    xpassword   = request.POST.get('xpassword')

    url = "http://127.0.0.1:7889/api/auth/login"
    params = {"xemail": xemail, "xpassword": xpassword}

    response = req.post(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the HTML content from the response object
        data = response.json()
    else:
        # Print an error message if the request failed
        print(f"Request failed with status code {response.status_code}")
    
    request.session['token'] = data['data']['token']

    if (data['success'] == False):
        return redirect('/login')
    
    return redirect('/')



# ADD NEW EMPLOYEE
# AUTHORED BY RYLANRISTIA
@csrf_exempt
def addemployee(request):
    data = {
            'x' : request.session.get('token'),
            'xnip': request.POST.get('xnip'),
            'xname': request.POST.get('xname'),
            'xemail': request.POST.get('xemail'),
            'xphone_number': request.POST.get('xphone_number'),
            'xaddress': request.POST.get('xaddress'),
            'csrf' : get_token(request)
        }

    check = checkEmployee(request.POST.get('xnip'))

    if check == False:
        return redirect(f'http://127.0.0.1:7899/add-new/');
    
    base_url = "http://127.0.0.1:7889"
    endpoint = "/api/employee/create"
    api_url = base_url + endpoint
    response = req.post(api_url, json=data)

    # dd(response)

    if response.status_code == 200:
        response_data = response.json()
        print(response_data)
    else:
        print(f"Request failed with status code {response.status_code}")
    
    # Build the query string from the data
    query_string = "&".join([f"{key}={value}" for key, value in data.items()])

    # Redirect to another page with query parameters
    return redirect(f'http://127.0.0.1:7899/add-face/?{query_string}')


# EMPLOYEE CHECK
# AUTHORED BY RYLANRISTIA
def checkEmployee(nip):

    data = {
        'xnip' : nip
    }

    base_url = "http://127.0.0.1:7889"
    endpoint = "/api/employee/check"
    api_url = base_url + endpoint
    response = req.post(api_url, json=data)

    res = response.json()

    if res == True:
        return True
    else:
       return False


# ADD NEW FACE
# AUTHORED BY RYLANRISTIA
@csrf_exempt
def addemployeeimg(request):

    csrfToken = get_token(request)

    context = {
        'xnip': request.GET.get('xnip'),
        'ximage': request.GET.get('ximage'),
        'csrf' : csrfToken
    }

    return render(request, 'add-new-face.html', context)



# ADD PROCEED NEW FACE
# AUTHORED BY RYLANRISTIA
@csrf_exempt
def addproceed(request):
    csrfToken = get_token(request)

    data = request.body

    data = json.loads(data.decode('utf-8'))
    frame = data.get('frame', None)
    nip = data.get('nip', None)

    # Specify the directory where you want to save the images
    save_directory = '../image_face/' + str(nip) + '/'  # Use the absolute path to the directory
    
    print(nip)

    # Ensure the directory exists, creating it if necessary
    os.makedirs(save_directory, exist_ok=True)

    # Decode the Base64 data and save it as an image file
    #image_data = frame.encode('utf-8')  # Convert the string data back to bytes
    content_type, image_data = frame.split(';base64,')
    image_format = content_type.split('/')[-1]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    filename = f'face_{timestamp}.{image_format}'
    image_path = os.path.join(save_directory, filename)

    with open(image_path, 'wb') as image_file:
        image_file.write(base64.b64decode(image_data))

    res = {
        'message' : 'Successfuly save image!' 
    }
    
    return JsonResponse(res)



# ADD RECOGNIZED THE FACE FOR MATCH
# AUTHORED BY RYLANRISTIA
def test(X_img_path, knn_clf, distance_threshold=0.4):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    with open(knn_clf, 'rb') as f:
        knn_clf = pickle.load(f)

    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    recognized_names = []
    for pred, rec in zip(knn_clf.predict(faces_encodings), are_matches):
        if rec:
            recognized_names.append(pred)
        else:
            recognized_names.append("unknown")

    return recognized_names





# PREPARATION FOR DATA TRAINING
# AUTHORED BY RYLANRISTIA
def train_face(request):
    classifier = train("../image_face/", n_neighbors=2)

    X, y = [], []

    for class_dir in os.listdir("../image_face/"):
        if not os.path.isdir(os.path.join("../image_face/", class_dir)):
            continue

        for img_file in os.listdir(os.path.join("../image_face/", class_dir)):
            img_path = os.path.join("../image_face/", class_dir, img_file)
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                continue

            X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
            y.append(class_dir)

    best_hyperparameters = optimize_knn_hyperparameters(X, y)
    n_neighbors, metric = best_hyperparameters

    print("Optimal hyperparameters: n_neighbors={}, metric={}".format(n_neighbors, metric))

    metric_algo = 'euclidean'
    classifier = train("../image_face/", n_neighbors=int(round(n_neighbors)), metric=metric_algo)

    res = {
        'message' : 'Successfuly train data!' 
    }
    
    return JsonResponse(res)




# EXECUTE DATA TRAINING
# AUTHORED BY RYLANRISTIA
def train(train_dir, model_save_path="trained_knn_model.clf", n_neighbors=2, metric='euclidean'):
    X = []
    y = []

    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_file in os.listdir(os.path.join(train_dir, class_dir)):
            img_path = os.path.join(train_dir, class_dir, img_file)
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                continue

            X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
            y.append(class_dir)

    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric=metric)
    knn_clf.fit(X, y)

    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)

    return knn_clf


def optimize_knn_hyperparameters(X, y):
    def objective_function(hyperparameters):
        n_neighbors, metric= hyperparameters
        knn = KNeighborsClassifier(n_neighbors=int(n_neighbors), metric='euclidean')
        knn.fit(X, y)
        return -knn.score(X, y)  # We aim to maximize the negative accuracy

    valid_metrics = ['canberra', 'p', 'dice', 'cityblock', 'minkowski', 'haversine', 'nan_euclidean', 'l2', 'yule', 'hamming', 'euclidean', 'sqeuclidean', 'chebyshev', 'pyfunc', 'infinity', 'seuclidean', 'sokalsneath', 'mahalanobis', 'cosine', 'russellrao', 'jaccard', 'l1', 'rogerstanimoto', 'correlation', 'manhattan', 'sokalmichener', 'braycurtis', 'precomputed']

    lb = [1, 0]  # Lower bounds for hyperparameters, where 0 corresponds to 'canberra'
    ub = [20, len(valid_metrics) - 1]  # Upper bounds for hyperparameters

    best_hyperparameters, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=100)
    return best_hyperparameters





# FACE RECOGNITION
# AUTHORED BY RYLANRISTIA
def recognize(request):
    data = request.body

    if request.method == 'POST':
        # Get the image data from the POST request
        data = request.body

        data = json.loads(data.decode('utf-8'))
        frame = data.get('frame', None)

        try:
            # Specify the directory where you want to save the images
            save_directory = '../image_face_test/'  # Use the absolute path to the directory
            
            # Ensure the directory exists, creating it if necessary
            os.makedirs(save_directory, exist_ok=True)

            # Decode the Base64 data and save it as an image file
            content_type, image_data = frame.split(';base64,')
            image_format = content_type.split('/')[-1]
            filename = f'face_test.{image_format}'
            image_path = os.path.join(save_directory, filename)

            with open(image_path, 'wb') as image_file:
                image_file.write(base64.b64decode(image_data))

            image_path = os.path.join(save_directory, 'face_test.jpeg')

            res = test(image_path, knn_clf="trained_knn_model.clf")

            data = {
                'nip' : res
            }

            base_url = "http://127.0.0.1:7889"
            endpoint = "/api/employee/set/attendance"
            api_url = base_url + endpoint
            response = req.post(api_url, json=data)

            response_data = response.json()

            
            context = {
                'nip' : response_data
            }

            return JsonResponse(context)

        except Exception as e:
            response_data = {'error': str(e)}
            return JsonResponse(response_data, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=400)