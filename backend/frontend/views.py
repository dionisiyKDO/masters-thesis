from django.views.static import serve
from django.shortcuts import render
from django.conf import settings
import os

def index(request, *args, **kwargs):
    # Serve the 404.html directly from your static/frontend directory
    filepath = os.path.join(settings.STATIC_ROOT, 'frontend', '404.html')
    # return render(request, filepath) 
    return serve(request, os.path.basename(filepath), os.path.dirname(filepath))
