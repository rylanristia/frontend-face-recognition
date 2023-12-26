from django.contrib import admin
from django.urls import path

from . import views
from . import core
from . import face

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index),
    path('employee/', views.employee),
    path('form-report/', views.formreport),
    path('report/', views.report, name='report'),
    path('attendance/', views.attendance),
    path('add-new/', views.addnew, name='add'),
    path('login/', views.login),
    path('proceed/auth/', core.auth, name='login_auth'),
    path('add-employee/', core.addemployee, name='add_employee'),
    path('recognize/', face.recognize, name='recognize'),
    path('add-face/', core.addemployeeimg, name='addimg'),
    path('proceed-image', core.addproceed, name="proceed_img"),
    path('train/', core.train_face, name='train_face')
]
