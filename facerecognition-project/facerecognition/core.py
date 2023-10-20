from django.http import request
from django.shortcuts import render, redirect
import requests as req
import json
from django.http import JsonResponse

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
    return redirect('/employee')

def recognize(request):
    data = request.body

    data = json.loads(data.decode('utf-8'))

    # Access specific values from the parsed data
    param1 = data.get('frame', None)

    res = {
        'frame' : param1
    }
    
    return JsonResponse(res)