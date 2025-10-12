import re
from django import forms
from PredictionModel.models import Uploads, User

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = Uploads
        fields = [
            'image'
        ]
        widgets = {
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'id': 'imageInput'
            })
        }

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if not image:
            raise forms.ValidationError('Please provide a image file')

        # size check
        if image.size > 25 * 1024 * 1024:
            raise forms.ValidationError("Image size must be less than 25MB.")

        # type check
        if not image.content_type.startswith('image/'):
            raise forms.ValidationError("Please upload a valid image file.")

        # format check
        allowed_formats = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp']
        if image.content_type not in allowed_formats:
            raise forms.ValidationError("Please upload a JPEG, PNG, BMP image.")

        try:
            from PIL import Image as PILImage
            pil_image = PILImage.open(image)
            width, height = pil_image.size

            if width < 224 or height < 224:
                raise forms.ValidationError("Image must be at least 224x224 pixels for accurate prediction.")

        except Exception:
            raise forms.ValidationError("Unable to process the image file. Please try a different image.")

        return image



class RegistrationForm(forms.ModelForm):
    password1 = forms.CharField(
        label="Password",
        widget=forms.PasswordInput,
        help_text="Password must be at least 8 characters long and include uppercase, lowercase, and a number."
    )
    password2 = forms.CharField(
        label="Confirm Password",
        widget=forms.PasswordInput
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'avatar']

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if not username:
            raise forms.ValidationError('Please enter a valid username.')

        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("Username already exists.")
        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Email already exists.")
        return email

    def clean_password1(self):
        password = self.cleaned_data.get('password1')
        if not password:
            raise forms.ValidationError("Password is required.")

        errors = []
        # 1. Minimum length
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long.")

        # 2. At least one uppercase letter
        if not re.search(r"[A-Z]", password):
            errors.append("Password must contain at least one uppercase letter.")

        # 3. At least one lowercase letter
        if not re.search(r"[a-z]", password):
            errors.append("Password must contain at least one lowercase letter.")

        # 4. At least one digit
        if not re.search(r"\d", password):
            errors.append("Password must contain at least one digit.")

        if errors:
            raise forms.ValidationError(errors)

        return password

    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')

        if not password2 or password1 != password2:
            raise forms.ValidationError("Passwords don't match.")
        return password2

    def save(self, commit=True):
        user = User.objects.create_user(
            username=self.cleaned_data['username'],
            email=self.cleaned_data['email'],
            avatar=self.cleaned_data['avatar']
        )
        user.set_password(self.cleaned_data['password1'])
        if commit:
            user.save()
        return user







