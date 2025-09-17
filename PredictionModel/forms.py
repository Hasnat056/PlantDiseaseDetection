from django import forms

from PredictionModel.models import Uploads


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








