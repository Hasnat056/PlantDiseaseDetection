from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone


def user_avatar_path(instance, filename):
    # Store avatars in media/avatars/user_<id>/<filename>
    return f'avatars/user_{instance.id}/{filename}'

class User(AbstractUser):
    email = models.EmailField(unique=True)
    avatar = models.ImageField(
        upload_to=user_avatar_path,
        default='avatars/default.png',  # default image
        blank=True,
        null=True
    )

    def __str__(self):
        return self.username


class Uploads(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(default=timezone.now)
    result = models.TextField()
    confidence_score = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True)

class UserImages(models.Model):
    image = models.ForeignKey(Uploads, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
