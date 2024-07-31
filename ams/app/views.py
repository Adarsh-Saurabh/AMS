

from django.contrib import messages
from django.shortcuts import redirect, render
from .utils import capture_and_save_photos, ml_function, predict_classes
from Yolo.main import crop_faces_with_yolo_haar
import os
from django.conf import settings
from .models import Student
import glob
import shutil

def index(request):
    if request.method == "POST":
        try:
            name = request.POST.get("student_name")
            capture_and_save_photos(name, 150)
            folder_location = f"./people/{name}"  # Assuming folder location is based on the student's name
            student = Student(name=name, folder_location=folder_location)
            student.save()
        except Exception as e:
            print("An error occurred in the job request - ", e)
            messages.error(request, "OOPS! Something went wrong")
            return render(request, "index.html")
    
    students = Student.objects.values_list('name', flat=True)
    return render(request, "index.html", {"students": students})

def delete_all(request):
    try:
        # Delete all Student records
        Student.objects.all().delete()
        
        # Delete all folders in the ./people directory
        people_folder = './people'
        
        if os.path.exists(people_folder):
            for filename in os.listdir(people_folder):
                file_path = os.path.join(people_folder, filename)
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)

        messages.success(request, "All students and their folders have been deleted successfully.")
    except Exception as e:
        print("An error occurred in the delete_all request - ", e)
        messages.error(request, "OOPS! Something went wrong")
    
    return redirect('index')
    return redirect('index')

def training_page(request):
    return render(request, "train_page.html")

def training(request):
    try:
        ml_function()
        messages.success(request, "Model training completed successfully.")
        return redirect('training_page')
    except Exception as e:
        print("An error occurred in the job request - ", e)
        messages.error(request, "OOPS! Something went wrong")
        return redirect('training_page')

def upload_page(request):
    return render(request, "upload_page.html")

def face_detect(request):
    if request.method == "POST":
        print("request initiated")
        if 'image' in request.FILES:
            image = request.FILES['image']
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'student_images')
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
                print(upload_dir)
            image_path = os.path.join(upload_dir, image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            request.session['image_path'] = image_path
            folder_path = './faces'
            files = glob.glob(os.path.join(folder_path, '*'))
            for file in files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Error deleting file {file}: {e}")

            try:
                image_path = request.session.get('image_path')
                if image_path:
                    crop_faces_with_yolo_haar(image_path)
                    return redirect('result')
                else:
                    raise ValueError("No image found in the session")
            except Exception as e:
                print("An error occurred in the face detection request - ", e)
                messages.error(request, "OOPS! Something went wrong")
                return redirect('upload_page')
    
    return redirect('upload_page')

def result(request):
    try:
        print("requested for result")
        pre = predict_classes()
        print(pre)
        return render(request, "result_page.html", {"pre": pre})
    except Exception as e:
        print("An error occurred in the result job request - ", e)
        messages.error(request, "OOPS! Something went wrong")
        return redirect('index')
