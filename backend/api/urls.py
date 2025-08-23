from django.urls import path, include
from rest_framework import routers
from .views import (hello, DoctorDashboardView, PatientDashboardView, AdminDashboardView)

app_name = "api"
router = routers.DefaultRouter()
router.register(r'doctor', DoctorDashboardView, basename="doctor_dashboard")
router.register(r'patient', PatientDashboardView, basename="patient_dashboard")
router.register(r'admin', AdminDashboardView, basename="admin_dashboard")

urlpatterns = [
    path('', include(router.urls)),
    path("hello/", hello),
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