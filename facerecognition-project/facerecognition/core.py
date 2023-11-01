from django.http import request
from django.shortcuts import render, redirect
import requests as req
import json
from django.http import JsonResponse
import base64
import os
import io 
from PIL import Image
import imageio
import numpy as np
import cv2
from datetime import datetime

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
            'xnip': '220031',
            'xname': 'Rylanristia',
            'xemail': 'rylanristia@gmail.com',
            'xphone_number': '085781077948',
            'xaddress': 'Lorem'
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

    dd(context)

def recognize(request):
    data = request.body

    # data = json.loads(data.decode('utf-8'))

    # # Access specific values from the parsed data
    # frame = data.get('frame', None)

    # res = {
    #     'frame' : frame
    # }

    # return JsonResponse(res)
    if request.method == 'POST':
        # Get the image data from the POST request
        data = request.body

        data = json.loads(data.decode('utf-8'))
        frame = data.get('frame', None)
        
        try:
            # Specify the directory where you want to save the images
            save_directory = '../image_face/'  # Use the absolute path to the directory

            # Ensure the directory exists, creating it if necessary
            os.makedirs(save_directory, exist_ok=True)

            # Generate a unique filename (you can use a timestamp, random string, etc.)
            filename = 'face.png'  # You can choose any file extension (e.g., PNG)

            # Decode the Base64 data and save it as an image file
            image_data = frame.encode('utf-8')  # Convert the string data back to bytes
            image_path = os.path.join(save_directory, filename)

            with open(image_path, 'wb') as image_file:
                image_file.write(base64.b64decode(image_data))

            # Return a JSON response with the image path or any other data you want
            response_data = {'message': 'Image saved successfully', 'image_path': image_path}
            return JsonResponse(response_data)

        except Exception as e:
            response_data = {'error': str(e)}
            return JsonResponse(response_data, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=400)