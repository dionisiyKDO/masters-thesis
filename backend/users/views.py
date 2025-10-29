from rest_framework import viewsets, generics
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import (AllowAny, IsAuthenticated, IsAdminUser, IsAuthenticatedOrReadOnly)
from rest_framework_simplejwt.views import TokenObtainPairView

from api.audit_utils import create_audit_log
from api.audit_mixins import AuditLoggingMixin
from .models import User, PatientProfile, DoctorProfile
from .permissions import IsDoctor, IsAdmin
from .serializers import (
    UserSerializer,
    PatientProfileSerializer,
    DoctorProfileSerializer,
    RegisterSerializer,
    CustomTokenObtainPairSerializer,
)


class UserViewSet(AuditLoggingMixin, viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """

    queryset = User.objects.all().order_by("-date_joined")
    serializer_class = UserSerializer
    permission_classes = [IsAdmin]
    
    audit_log_model_name = "USER"
    
    @action(detail=False, methods=["get"], permission_classes=[IsAdmin])
    def stats(self, request):
        total_users = User.objects.count()
        total_doctors = User.objects.filter(role="doctor").count()
        total_patients = User.objects.filter(role="patient").count()

        return Response({
            "total_users": total_users,
            "total_doctors": total_doctors,
            "total_patients": total_patients,
        })


class DoctorProfileViewSet(AuditLoggingMixin, viewsets.ModelViewSet):
    queryset = DoctorProfile.objects.all()
    serializer_class = DoctorProfileSerializer
    permission_classes = [IsDoctor | IsAdmin]
    audit_log_model_name = "DOCTOR_PROFILE"


class PatientProfileViewSet(AuditLoggingMixin, viewsets.ModelViewSet):
    queryset = PatientProfile.objects.all()
    serializer_class = PatientProfileSerializer
    permission_classes = [IsDoctor | IsAdmin]
    audit_log_model_name = "PATIENT_PROFILE"


class RegisterView(generics.CreateAPIView):
    """
    API endpoint that allows new users to register.
    """
    
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [AllowAny]

    def perform_create(self, serializer):
        """
        Hook for logging registration.
        """
        new_user = serializer.save()
        
        action = "USER_REGISTERED"
        log_details = {
            "created_user_id": new_user.id,
            "username": new_user.username,
            "role": new_user.role
        }
        
        # Log this action as being performed by the new user
        create_audit_log(user=new_user, action=action, details=log_details)

class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer


# Book = ...
# BookSerializer = ...
# Response = ...
# status = ...
# action = ...

# class BookViewSet(viewsets.ModelViewSet):
#     """
#     A fully detailed ViewSet for managing books.
#     """

#     queryset = Book.objects.all()  # Default query (can be overridden)
#     serializer_class = BookSerializer  # Defines which serializer to use

#     # ----- LIST -----
#     def list(self, request, *args, **kwargs):
#         """
#         Handle GET /books/ - List all books.
#         """
#         books = self.get_queryset()
#         serializer = self.get_serializer(books, many=True)
#         return Response(serializer.data)

#     # ----- RETRIEVE -----
#     def retrieve(self, request, *args, **kwargs):
#         """
#         Handle GET /books/{id}/ - Get a single book by ID.
#         """
#         book = self.get_object()
#         serializer = self.get_serializer(book)
#         return Response(serializer.data)

#     # ----- CREATE -----
#     def create(self, request, *args, **kwargs):
#         """
#         Handle POST /books/ - Create a new book.
#         """
#         serializer = self.get_serializer(data=request.data)
#         serializer.is_valid(raise_exception=True)  # Calls validate() in serializer
#         self.perform_create(serializer)
#         return Response(serializer.data, status=status.HTTP_201_CREATED)

#     def perform_create(self, serializer):
#         """
#         Save the book after validation.
#         """
#         serializer.save()

#     # ----- UPDATE -----
#     def update(self, request, *args, **kwargs):
#         """
#         Handle PUT /books/{id}/ - Update an existing book.
#         """
#         book = self.get_object()
#         serializer = self.get_serializer(book, data=request.data)
#         serializer.is_valid(raise_exception=True)
#         self.perform_update(serializer)
#         return Response(serializer.data)

#     def perform_update(self, serializer):
#         """
#         Save updated book instance.
#         """
#         serializer.save()

#     # ----- PARTIAL UPDATE -----
#     def partial_update(self, request, *args, **kwargs):
#         """
#         Handle PATCH /books/{id}/ - Partially update a book.
#         """
#         book = self.get_object()
#         serializer = self.get_serializer(book, data=request.data, partial=True)
#         serializer.is_valid(raise_exception=True)
#         self.perform_update(serializer)
#         return Response(serializer.data)

#     # ----- DELETE -----
#     def destroy(self, request, *args, **kwargs):
#         """
#         Handle DELETE /books/{id}/ - Delete a book.
#         """
#         book = self.get_object()
#         self.perform_destroy(book)
#         return Response(status=status.HTTP_204_NO_CONTENT)

#     def perform_destroy(self, instance):
#         """
#         Actually delete the book.
#         """
#         instance.delete()

#     # ----- CUSTOM ACTION -----
#     @action(detail=True, methods=['post'])
#     def mark_as_favorite(self, request, pk=None):
#         """
#         Custom action: POST /books/{id}/mark_as_favorite/
#         """
#         book = self.get_object()
#         book.is_favorite = True  # Assuming there is an 'is_favorite' field
#         book.save()
#         return Response({'status': 'book marked as favorite'})


# list(self, request, *args, **kwargs)	            Lists all objects.
# retrieve(self, request, *args, **kwargs)	        Retrieves a single object by ID.
# create(self, request, *args, **kwargs)	        Validates & creates a new object.
# update(self, request, *args, **kwargs)	        Updates an entire object (PUT).
# partial_update(self, request, *args, **kwargs)	Updates part of an object (PATCH).
# destroy(self, request, *args, **kwargs)	        Deletes an object.
# perform_create(self, serializer)	                Saves an object after create().
# perform_update(self, serializer)	                Saves an object after update().
# perform_destroy(self, instance)	                Deletes an object.
# get_queryset(self)	                            Returns the queryset (can be customized).
# get_serializer_class(self)	                    Allows changing the serializer dynamically.
