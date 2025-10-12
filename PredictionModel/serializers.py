import os

from django.core.files.storage import default_storage
from rest_framework import serializers
from celery.result import AsyncResult
from rest_framework.response import Response

from PlantDiseaseDetection import settings
from .models import Uploads, User
from .tasks import process_plant_disease


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            'id',
            'username',
            'password',
            'email',
        ]
        extra_kwargs = {
            'password': {'write_only': True},
        }

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        user.set_password(validated_data['password'])
        user.save()
        return user

    def update(self, instance, validated_data):
        if 'password' in validated_data:
            instance.set_password(validated_data['password'])
            instance.save()
        for key, value in validated_data.items():
            if key != 'password':
                setattr(instance, key, value)
        instance.save()

        return instance

class UserLoginSerializer(serializers.ModelSerializer):
    class Meta:
      fields = [
          'username',
          'password',
      ]

    def create(self, validated_data):
        user = User.objects.get(username=validated_data['username'])
        if user:
            if user.check_password(validated_data['password']):
                return Response({'message': 'Login Successful'})
            return Response({'error': 'Login Failed, Password Incorrect'})
        return Response({'error': 'Login Failed, User not found'})



class ImageUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Uploads
        fields = ['image']

    def validate(self, data):
        image = data.get('image')

        if not image:
            raise serializers.ValidationError('Please provide a image file')

        # size check
        if image.size > 25 * 1024 * 1024:
            raise serializers.ValidationError("Image size must be less than 25MB.")

        # type check
        if not image.content_type.startswith('image/'):
            raise serializers.ValidationError("Please upload a valid image file.")

        # format check
        allowed_formats = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp']
        if image.content_type not in allowed_formats:
            raise serializers.ValidationError("Please upload a JPEG, PNG, BMP image.")

        try:
            from PIL import Image as PILImage
            pil_image = PILImage.open(image)
            width, height = pil_image.size

            if width < 224 or height < 224:
                raise serializers.ValidationError("Image must be at least 224x224 pixels for accurate prediction.")

        except Exception:
            raise serializers.ValidationError("Unable to process the image file. Please try a different image.")

        return data

    def create(self, validated_data):
        image = validated_data.get('image')
        request = self.context.get('request')
        file_path = default_storage.save(f'temp/{image.name}', image)
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)

        task = process_plant_disease.delay(full_path, request.user.id if request.user else None)

        return {'task_id': task.id}


