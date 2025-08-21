from django.urls import re_path, path
from django.views.static import serve
from django.conf import settings
import os

from . import views

urlpatterns = [
    # path('', views.index),
    # # Important: Add catch-all pattern for client-side routing
    # path('<path:path>', views.index),  # This handles all routes for SPA
    
    # Serve favicon.png directly, doesn't work
    # re_path(r'^favicon\.png$', serve, {
    #     'document_root': os.path.join(settings.STATIC_ROOT, 'frontend')
    # }),
    
    # Serve SvelteKit's static assets under /_app/
    re_path(r'^_app/(?P<path>.*)$', serve, {
        'document_root': os.path.join(settings.STATIC_ROOT, 'frontend', '_app')
    }),
    # Catch-all route for SvelteKit client-side routing
    re_path(r'^(?P<path>.*)$', views.index, name='index'),
]