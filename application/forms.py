from django import forms
from application import models

class InsertReviewForm(forms.ModelForm):
    class Meta:
        model=models.InsertReviewModel
        fields='__all__'
