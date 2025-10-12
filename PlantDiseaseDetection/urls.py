"""
URL configuration for PlantDiseaseDetection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from PlantDiseaseDetection import settings
from django.conf.urls.static import static
from PredictionModel.views import register_view, logout_view, login_view
from django.contrib.auth import views as auth_views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', login_view, name='login'),
    path('logout/',logout_view,name='logout' ),
    path('register/', register_view, name='register'),
    path('password-reset/',auth_views.PasswordResetView.as_view(
            template_name='registration/password_reset.html',
            email_template_name='registration/password_reset_email.html',
            subject_template_name='registration/password_reset_subject.txt',
            success_url='/password-reset/done/'
    ),
        name='password_reset'
    ),

    # Email Sent Confirmation
    path('password-reset/done/',auth_views.PasswordResetDoneView.as_view(
            template_name='registration/password_reset_done.html'),name='password_reset_done'),

    # Password Reset Link (from email)
    path('reset/<uidb64>/<token>/',auth_views.PasswordResetConfirmView.as_view(
            template_name='registration/password_reset_confirm.html',
            success_url='/reset/done/'),
        name='password_reset_confirm'
    ),

    # Password Successfully Reset
    path('reset/done/',auth_views.PasswordResetCompleteView.as_view(
            template_name='registration/password_reset_complete.html'
        ),
        name='password_reset_complete'
    ),
    path ('',include('PredictionModel.urls') ),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    # Add STATICFILES_DIRS if you're using it
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])