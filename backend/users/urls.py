from django.urls import include, path
from rest_framework import routers
from rest_framework_simplejwt.views import (
    # TokenObtainPairView,
    TokenRefreshView,
)

from .views import UserViewSet, RegisterView, CustomTokenObtainPairView

app_name = "users"
router = routers.DefaultRouter()
router.register(r"users", UserViewSet)

urlpatterns = [
    path("", include(router.urls)),
    path("register/", RegisterView.as_view(), name="register"),
    # JWT authentication:
    path("token/", CustomTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
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
