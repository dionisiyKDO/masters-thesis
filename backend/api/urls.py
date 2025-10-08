from django.urls import path, include
from rest_framework import routers
from . import views

app_name = "api"
router = routers.DefaultRouter()
router.register(r'cases', views.MedicalCaseViewSet, basename='medicalcase')
router.register(r'annotations', views.DoctorAnnotationViewSet, basename='doctorannotation')
router.register(r'scans', views.ChestScanViewSet, basename='chestscan')

urlpatterns = [
    path('', include(router.urls)),
    path("cases/<int:case_id>/scans/upload/", views.ScanUploadView.as_view(), name="scan-upload"),
]

# Viewset
# router = DefaultRouter()
# router.register(r'', TransactionViewSet)

# http://127.0.0.1:8000/transactions/ (GET, POST)
# http://127.0.0.1:8000/transactions/<id>/ (GET, PUT, DELETE)

# GET     /transactions/ - List all transactions
# POST    /transactions/ - Create a new transaction
# GET     /transactions/{id}/ - Retrieve a transaction
# PUT     /transactions/{id}/ - Update a transaction
# DELETE  /transactions/{id}/ - Delete a transaction