from rest_framework import status
from rest_framework.viewsets import ViewSet
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, action
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404

from rest_framework import viewsets, permissions
from .models import MedicalCase, ChestScan, ModelVersion, AIAnalysis, DoctorAnnotation
from .serializers import (
    MedicalCaseSerializer, MedicalCaseDetailSerializer, ChestScanSerializer,
    ModelVersionSerializer, AIAnalysisSerializer, DoctorAnnotationSerializer
)
from .permissions import IsAdminOrReadOnly, IsDoctor, IsPatientOfCase, IsDoctorOfCase
from .classifier import Classifier

#region Env Setup
import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
#endregion


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

class ScanUploadView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, case_id):
        file = request.data.get("image")
        if not file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)
        
        # TODO: save file to model / storage
        # example: Scan.objects.create(case_id=case_id, image=file)
          
        case = get_object_or_404(MedicalCase, id=case_id)
        logger.info(f"Uploading scan for case {case_id} by user {request.user.id}")

        # 1. Save scan
        scan = ChestScan.objects.create(case=case, image_path=file)
        
        logger.info(f"Scan saved with ID {scan.id}, starting classification.")

        # 2. Run CNN classifier
        classifier = Classifier(
            model_name='OwnV3',
            img_size=(150, 150),
            data_dir='../.archive/ml_tests/data_pneumonia_final_balanced_og'
        )
        classifier.load_model('checkpoints/Saved/OwnV3.epoch50-val_acc0.9830.hdf5')
        logger.info(f"Model loaded, running prediction on scan path {scan.image_path.path}.")
        predicted_class, confidence, probabilities = classifier.predict(scan.image_path.path)
        # label, confidence, heatmap_path, model_name = classifier.predict(scan.image.path)
        logger.info(f"Prediction complete: {predicted_class} with confidence {confidence:.4f}")
        # 3. Save analysis
        model_version = ModelVersion.objects.get(model_name='OwnV3')
        AIAnalysis.objects.create(
            scan=scan,
            prediction_label=predicted_class,
            confidence_score=confidence,
            # heatmap_path=heatmap_path,
            model_version=model_version,
        )

        return Response({"message": "Upload successful"}, status=status.HTTP_201_CREATED)      

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