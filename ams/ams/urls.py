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
    path('delete_selected', views.delete_selected, name="delete_selected"),  # New URL pattern
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
