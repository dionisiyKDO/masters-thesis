from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from rest_framework.decorators import api_view, action

from rest_framework import viewsets, permissions
from .models import MedicalCase, ChestScan, ModelVersion, AIAnalysis, DoctorAnnotation
from .serializers import (
    MedicalCaseSerializer, MedicalCaseDetailSerializer, ChestScanSerializer,
    ModelVersionSerializer, AIAnalysisSerializer, DoctorAnnotationSerializer
)
from .permissions import IsAdminOrReadOnly, IsDoctor, IsPatientOfCase, IsDoctorOfCase


@api_view(["GET"])
def hello(request):
    return Response({"message": "Hello from Django"})


class MedicalCaseViewSet(viewsets.ModelViewSet):
    """
    API endpoint for Medical Cases.
    - Patients can only see their own cases.
    - Doctors can see cases they are assigned to.
    - Admins can see all cases.
    """
    queryset = MedicalCase.objects.all().select_related('patient', 'primary_doctor')
    
    def get_serializer_class(self):
        # Use a more detailed serializer for the 'retrieve' action
        if self.action == 'retrieve':
            return MedicalCaseDetailSerializer
        return MedicalCaseSerializer
    
    def get_permissions(self):
        """Instantiates and returns the list of permissions that this view requires."""
        if self.action in ['list', 'retrieve']:
            # A patient OR a doctor can view the case details
            permission_classes = [permissions.IsAuthenticated, IsPatientOfCase | IsDoctorOfCase]
        else:
            # Only doctors can create/update cases
            permission_classes = [IsDoctor]
        return [permission() for permission in permission_classes]

    def get_queryset(self):
        """
        Filter cases based on user role.
        """
        user = self.request.user
        if user.role == 'patient':
            return self.queryset.filter(patient=user)
        if user.role == 'doctor':
            return self.queryset.filter(primary_doctor=user)
        # Admins can see all cases (default queryset)
        return self.queryset

class ChestScanViewSet(viewsets.ModelViewSet):
    """API endpoint for Chest Scans."""
    queryset = ChestScan.objects.all()
    serializer_class = ChestScanSerializer
    permission_classes = [permissions.IsAuthenticated, IsDoctorOfCase] # Only the assigned doctor can manage scans

class AIAnalysisViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only endpoint for AI Analyses."""
    queryset = AIAnalysis.objects.all()
    serializer_class = AIAnalysisSerializer
    permission_classes = [permissions.IsAuthenticated, IsPatientOfCase | IsDoctorOfCase]

class DoctorAnnotationViewSet(viewsets.ModelViewSet):
    """Endpoint for Doctor Annotations."""
    queryset = DoctorAnnotation.objects.all()
    serializer_class = DoctorAnnotationSerializer
    permission_classes = [permissions.IsAuthenticated, IsDoctorOfCase]

class ModelVersionViewSet(viewsets.ModelViewSet):
    """Endpoint for AI Model Versions. Only Admins can modify."""
    queryset = ModelVersion.objects.all()
    serializer_class = ModelVersionSerializer
    permission_classes = [IsAdminOrReadOnly]