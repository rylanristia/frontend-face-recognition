from django.http import request
from django.shortcuts import render, redirect
from django.http import JsonResponse
from PIL import Image
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import requests as req
import json
import base64
import os
import io 
import imageio
import numpy as np
import cv2
import pickle
import face_recognition

FILE_STORE = "face_recognition_model.pkl"

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

def addemployee(request):
    data_to_pass = {
            'xnip': request.POST.get('xnip'),
            'xname': request.POST.get('xname'),
            'xemail': request.POST.get('xemail'),
            'xphone_number': request.POST.get('xphone_number'),
            'xaddress': request.POST.get('xaddress')
        }

    # Build the query string from the data
    query_string = "&".join([f"{key}={value}" for key, value in data_to_pass.items()])

    # Redirect to another page with query parameters
    return redirect(f'http://127.0.0.1:7899/add-face/?{query_string}')

def addemployeeimg(request):

    context = {
        'xnip': request.GET.get('xnip'),
        'xname': request.GET.get('xname'),
        'xemail': request.GET.get('xemail'),
        'xphone_number': request.GET.get('xphone_number'),
        'xaddress': request.GET.get('xaddress')
    }

    return render(request, 'add-new-face.html', context)

def addproceed(request):
    context = {
        'xnip': request.POST.get('xnip'),
        'xname': request.POST.get('xname'),
        'xemail': request.POST.get('xemail'),
        'xphone_number': request.POST.get('xphone_number'),
        'xaddress': request.POST.get('xaddress'),
        'ximage': request.POST.get('image')
    }

    image = context['ximage']

    # Specify the directory where you want to save the images
    save_directory = '../image_face/' + context['xname'] + '/'  # Use the absolute path to the directory
    
    # Ensure the directory exists, creating it if necessary
    os.makedirs(save_directory, exist_ok=True)

    # Generate a unique filename (you can use a timestamp, random string, etc.)
    # You can choose any file extension (e.g., PNG)

    # Decode the Base64 data and save it as an image file
    #image_data = frame.encode('utf-8')  # Convert the string data back to bytes
    content_type, image_data = image.split(';base64,')
    image_format = content_type.split('/')[-1]
    filename = f'face_person.{image_format}'
    image_path = os.path.join(save_directory, filename)

    with open(image_path, 'wb') as image_file:
        image_file.write(base64.b64decode(image_data))


    known_faces_dir = "../image_face"

    # Training
    all_images_train = []
    all_names_train = []
    for person_dir in os.listdir(known_faces_dir):
        if os.path.isdir(os.path.join(known_faces_dir, person_dir)):
            person_name = person_dir
            for image_file in os.listdir(os.path.join(known_faces_dir, person_dir)):
                image_path = os.path.join(known_faces_dir, person_dir, image_file)
                all_images_train.append(image_path)
                all_names_train.append(person_name)

    train(all_images_train, all_names_train)

    res = {
        'message' : 'Successfuly added new employee!' 
    }
    
    return JsonResponse(res)


def test():
    try:
        with open(FILE_STORE, 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        raise Exception("You have to train first before testing it!")

    known_face_encodings = model['known_face_encodings']
    known_face_names = model['known_face_names']
    
    guesses = 'unknown'
    test_image = face_recognition.load_image_file('../image_face_test/face_test.jpeg')
    face_locations = face_recognition.face_locations(test_image)
    test_face_encodings = face_recognition.face_encodings(test_image, face_locations)
    for test_face_encoding in test_face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, test_face_encoding)
        
        name = 'Unknown'
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        guesses = name
    
    return guesses


# photos 
def train(photos: list, names: list) -> None:
    try:
        with open(FILE_STORE, 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        model = {
            'known_face_encodings' : [],
            'known_face_names' : []
        }
    face_encodings = []


    for path in photos:
        image = face_recognition.load_image_file(path)
        face_encoding = face_recognition.face_encodings(image)
        face_encodings.append(face_encoding)

    model['known_face_encodings'] += face_encodings
    model['known_face_names'] += names

    with open(FILE_STORE, 'wb') as file:
        pickle.dump(model, file)


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

            # Generate a unique filename (you can use a timestamp, random string, etc.)
            # You can choose any file extension (e.g., PNG)

            # Decode the Base64 data and save it as an image file
            #image_data = frame.encode('utf-8')  # Convert the string data back to bytes
            content_type, image_data = frame.split(';base64,')
            image_format = content_type.split('/')[-1]
            filename = f'face_test.{image_format}'
            image_path = os.path.join(save_directory, filename)

            with open(image_path, 'wb') as image_file:
                image_file.write(base64.b64decode(image_data))

            # # Return a JSON response with the image path or any other data you want
            # response_data = {'message': 'Image saved successfully', 'image_path': image_path}

            res = test()

            # dd(res)

            context = {
                'frame' : res
            }

            return JsonResponse(context)

        except Exception as e:
            response_data = {'error': str(e)}
            return JsonResponse(response_data, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=400)