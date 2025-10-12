from django import forms
from django.contrib import messages
from django.contrib.auth import logout, authenticate, login
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.core.files.storage import default_storage
from django.conf import settings
import os

from celery.result import AsyncResult

from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .forms import RegistrationForm, ImageUploadForm
from .models import Uploads, UserImages, User
from .serializers import ImageUploadSerializer, UserSerializer, UserLoginSerializer
from .tasks import process_plant_disease

def image_processing_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['image']
            file_path = default_storage.save(f'temp/{uploaded_file.name}', uploaded_file)
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)

            task_result = process_plant_disease.delay(full_path, request.user.id)

            # If it's an AJAX request, return JSON
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'task_id': task_result.id})
        else:
            raise forms.ValidationError(form.errors)
        # Otherwise, render template
        return render(request, 'upload.html')

    return render(request, 'upload.html')


def check_task_status(request, task_id):
    """AJAX endpoint to check task status"""
    from celery.result import AsyncResult
    result = AsyncResult(task_id)
    if result.ready():
        task_result = result.get()
        if task_result.get('success'):
            upload = Uploads.objects.get(id=task_result.get('upload_id'))
            return JsonResponse({
                'status': 'completed',
                'image': request.build_absolute_uri(upload.image.url),
                'prediction': task_result.get('prediction'),
                'confidence': task_result.get('confidence')
            })
        else:
            return JsonResponse({
                'status': 'failed',
                'error': task_result['error']
            })
    else:
        return JsonResponse({'status': 'processing'})


class UserCreateAPIView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer

class UserUpdateRetrieveAPIView(generics.RetrieveUpdateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer

class UserHistoryAPIView(APIView):
    def get(self, request, *args, **kwargs):
        data = {}
        queryset = Uploads.objects.filter(userimages__user=request.user)
        if queryset.exists():
            for each in queryset:
                data['each.id'] = {'image': each.image.url, 'uploaded_at' : each.uploaded_at, 'result': each.result, 'confidence': each.confidence_score}

        return Response(data)

class UserLoginAPIView(APIView):
    serializer_class = UserLoginSerializer
    def post(self, request, *args, **kwargs):
        serializer = UserLoginSerializer(data=request.data)
        if serializer.is_valid():
            instance = serializer.save()
            return Response(instance.data)
        return Response(serializer.errors)

def register_view(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Your account has been created successfully. You can now log in.")
            return redirect('predictor:predict_image')  # redirect to home/dashboard
        else:
            # Optional: log errors for debugging
            print("Registration errors:", form.errors)
            messages.error(request, "Please fix the errors below.")
    else:
        form = RegistrationForm()

    return render(request, 'registration/register.html', {'form': form})

def login_view(request):
    """
    Handles user login
    """
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        # Authenticate the user
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            return redirect('predictor:predict_image')  # Redirect to your home/upload page
        else:
            messages.error(request, "Invalid username or password. Please try again.")

    return render(request, "registration/login.html")

def logout_view(request):
    """Log out the current user and redirect to the login page."""
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect('predictor:predict_image')  # Replace 'login' with your login URL name

@login_required
def history_view(request):
    # Fetch only the logged-in user's history
    history = UserImages.objects.filter(user=request.user).select_related('image').order_by('-image__uploaded_at')
    return render(request, 'history.html', {'history': history})


@login_required
def delete_record(request, record_id):
    """Delete a record that belongs to the logged-in user"""
    record = get_object_or_404(UserImages, id=record_id, user=request.user)
    record.delete()
    return redirect('predictor:history')

@login_required
def update_user_image(request):
    if request.method == "POST" and request.FILES.get('avatar'):
        user = request.user
        user.avatar = request.FILES['avatar']
        user.save()
        return JsonResponse({
            'success': True,
            'avatar_url': user.avatar.url
        })
    return JsonResponse({'success': False, 'error': 'No image uploaded.'})


class ImageUploadAPIView(APIView):
    serializer_class = ImageUploadSerializer
    permission_classes = []
    def post(self, request, *args, **kwargs):
        file = request.data.get('image')
        serializer = self.serializer_class(data={'image': file}, context={'request': request})
        serializer.is_valid(raise_exception=True)
        data = serializer.save()
        return Response({
            "status": "processing",
            "task_id": data["task_id"]
        })



class TaskResultAPIView(APIView):
    permission_classes = []

    def get(self, request, task_id):
        result = AsyncResult(task_id)

        if result.state in ['PENDING', 'STARTED']:
            return Response({"status": "processing"}, status=202)

        elif result.state == 'SUCCESS':
            task_data = result.result
            if task_data.get('success'):
                upload = Uploads.objects.get(id=task_data.get('upload_id'))
                return Response({
                    "status": "completed",
                    "image": request.build_absolute_uri(upload.image.url),
                    "prediction": task_data.get('prediction'),
                    "confidence": task_data.get('confidence')
                }, status=200)
            else:
                return Response({
                    "status": "failed",
                    "error": task_data.get('error')
                }, status=400)

        elif result.state == 'FAILURE':
            return Response({
                "status": "failed",
                "error": str(result.result)
            }, status=400)
        return None
