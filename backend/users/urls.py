from django.urls import include, path
from rest_framework import routers
from rest_framework_simplejwt.views import (
    # TokenObtainPairView,
    TokenRefreshView,
)

from .views import (
    UserViewSet, 
    PatientProfileViewSet, 
    DoctorProfileViewSet,
    
    RegisterView, 
    CustomTokenObtainPairView, 
)

app_name = "users"
router = routers.DefaultRouter()
router.register(r"users", UserViewSet)
router.register(r'doctors', DoctorProfileViewSet)
router.register(r'patients', PatientProfileViewSet)

urlpatterns = [
    path("list/", include(router.urls)),
    path("auth/register/", RegisterView.as_view(), name="register"),
    
    # JWT authentication:
    path("auth/token/", CustomTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
]


# Viewset
# router = DefaultRouter()
# router.register(r'', TransactionViewSet, basename='transaction')

# http://127.0.0.1:8000/transactions/ (GET, POST)
# http://127.0.0.1:8000/transactions/<id>/ (GET, PUT, DELETE)

# GET     /transactions/ - List all transactions
# POST    /transactions/ - Create a new transaction
# GET     /transactions/{id}/ - Retrieve a transaction
# PUT     /transactions/{id}/ - Update a transaction
# DELETE  /transactions/{id}/ - Delete a transaction
