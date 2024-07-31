# from django.db import models

# # Create your models here.
# class Student(models.Model):
#     name =  models.CharField(max_length=100)
    
#     dataset_folder = models.CharField(max_length=255, blank=True)  # Path to the dataset folder

#     def __str__(self):
#         return self.name


from django.db import models

class Student(models.Model):
    name = models.CharField(max_length=100)
    folder_location = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return self.name

