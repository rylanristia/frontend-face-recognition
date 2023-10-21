from django.http import request
from django.shortcuts import render, redirect
import requests as req
import json
from django.http import JsonResponse
import base64
import os
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

    data = json.loads(data.decode('utf-8'))

    # Access specific values from the parsed data
    frame = data.get('frame', None)

    res = {
        'frame' : frame
    }

    return JsonResponse(res)