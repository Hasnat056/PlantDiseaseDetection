from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.image_processing_view, name='predict_image'),  # home page + upload
    path('check-status/<str:task_id>/', views.check_task_status, name='check_status'),
    path('history/', views.history_view, name='history'),
    path('history/delete/<int:record_id>/', views.delete_record, name='delete_record'),
    path('update-avatar/', views.update_user_image, name='update_user_image'),




    path('api/', views.ImageUploadAPIView.as_view(), name='api_image'),
    path('api/register/', views.UserCreateAPIView.as_view(), name='api_register'),
    path('api/profile/', views.UserUpdateRetrieveAPIView.as_view(), name='api_profile'),
]
