from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone

class Uploads(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(default=timezone.now)
    result = models.TextField()
    confidence_score = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True)

class UserImages(models.Model):
    image = models.ForeignKey(Uploads, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
