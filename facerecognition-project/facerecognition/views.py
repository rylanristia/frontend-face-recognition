from django.http import request
from django.shortcuts import render, redirect
from django.contrib import messages
import requests as req
import json
from django.middleware.csrf import get_token

def sessionCheck(token):
    url = "http://127.0.0.1:7889/api/auth/session/check"
    params = {"x": token}

    response = req.post(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the HTML content from the response object
        data = response.json()

        if (data['success'] == True):
            return True
        else:
            return False

    else:
        # Print an error message if the request failed
        return False

def index(request):
    token = request.session.get('token')
    if (sessionCheck(token) == False):
        return redirect('/login')
    
    return render(request, 'index.html')

def employee(request):
    token = request.session.get('token')
    if (sessionCheck(token) == False):
        return redirect('/login')

    url = "http://127.0.0.1:7889/api/employee/get-all"
    params = {"x": request.session.get('token')}

    response = req.post(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the HTML content from the response object

        data = response.text

        emplyeers = json.loads(data)

        result = []

        for employee in emplyeers['data']:
            result.append(employee)

    else:
        # Print an error message if the request failed
        print(f"Request failed with status code {response.status_code}")

    context = {
        'result':result
    }
    
    return render(request, 'employee.html', context)

def attendance(request):
    token = request.session.get('token')
    csrfToken = get_token(request)

    if (sessionCheck(token) == False):
        return redirect('/login')
    
    context = {
        'csrf' : csrfToken
    }

    return render(request, 'attendance.html', context)

def addnew(request):
    token = request.session.get('token')
    if (sessionCheck(token) == False):
        return redirect('/login')

    return render(request, 'add-new.html')

def login(request):
    return render(request, 'login.html')