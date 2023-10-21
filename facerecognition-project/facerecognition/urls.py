from django.contrib import admin
from django.urls import path

from . import views
from . import core

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index),
    path('employee/', views.employee),
    path('attendance/', views.attendance),
    path('add-new/', views.addnew),
    path('login/', views.login),
    path('proceed/auth/', core.auth, name='login_auth'),
    path('add-employee/', core.addemployee, name='add_employee'),
    path('recognize/', core.recognize, name='recognize'),
    path('add-face/', core.addemployeeimg, name='addimg'),
    path('proceed-image', core.addproceed, name="proceed_img")
]
