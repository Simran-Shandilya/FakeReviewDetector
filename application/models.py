from django.db import models

# Create your models here.
class InsertReviewModel(models.Model):
    review=models.TextField(max_length=1024)