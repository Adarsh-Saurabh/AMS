# """
# URL configuration for ams project.

# The `urlpatterns` list routes URLs to views. For more information please see:
#     https://docs.djangoproject.com/en/4.2/topics/http/urls/
# Examples:
# Function views
#     1. Add an import:  from my_app import views
#     2. Add a URL to urlpatterns:  path('', views.home, name='home')
# Class-based views
#     1. Add an import:  from other_app.views import Home
#     2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
# Including another URLconf
#     1. Import the include() function: from django.urls import include, path
#     2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
# """
# from django.contrib import admin
# from django.urls import path
# from app import views
# from django.conf import settings
# from django.conf.urls.static import static


# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('', views.index, name="index"),
#     path('train', views.training, name="training"),
#     path('face_detect', views.face_detect, name="face_detect"),
#     path('result', views.result, name="result"),
    
    
# ]

# if settings.DEBUG:
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


from django.contrib import admin
from django.urls import path
from app import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name="index"),
    path('train_page/', views.training_page, name="training_page"),
    path('train', views.training, name="training"),
    path('upload_page/', views.upload_page, name="upload_page"),
    path('face_detect', views.face_detect, name="face_detect"),
    path('result', views.result, name="result"),
    path('delete_all', views.delete_all, name="delete_all"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
