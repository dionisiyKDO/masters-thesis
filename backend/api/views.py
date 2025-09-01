from rest_framework import status
from rest_framework.viewsets import ViewSet
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, action
from rest_framework.parsers import MultiPartParser, FormParser
from django.shortcuts import get_object_or_404

from rest_framework import viewsets, permissions
from .models import MedicalCase, ChestScan, ModelVersion, AIAnalysis, DoctorAnnotation, EnsembleResult, EnsembleResultModel
from .serializers import (
    MedicalCaseSerializer, MedicalCaseDetailSerializer, ChestScanSerializer,
    ModelVersionSerializer, AIAnalysisSerializer, DoctorAnnotationSerializer,
)
from .permissions import IsAdminOrReadOnly, IsDoctor, IsPatientOfCase, IsDoctorOfCase
from .classifier.classifier import Classifier
from .classifier.config import config_v1 as config
import numpy as np

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
        
        # Get case, Save new scan image
        case = get_object_or_404(MedicalCase, id=case_id)
        scan = ChestScan.objects.create(case=case, image_path=file)
        
        # Get all the active models
        model_versions = ModelVersion.objects.filter(is_active=True)
        if not model_versions.exists():
            return Response({"error": "No active model versions available"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        logger.info(f"Found {model_versions.count()} active model versions: {[mv.model_name for mv in model_versions]}")
        
        all_probs = []
        model_weights = []
        used_models = []
        
        # Loop through all active models, get and save their individual result's 
        for model_version in model_versions:
            try:
                classifier = Classifier(
                    model_version.model_name,
                    img_size=config['img_size'],
                    data_dir='./api/classifier/data'
                )
                classifier.load_model('.' + model_version.storage_uri) # '.' temporary seeding problem workaround
                predicted_class, confidence, probabilities = classifier.predict(scan.image_path.path)
                
                # Save individual prediction
                AIAnalysis.objects.create(
                    scan=scan,
                    prediction_label=predicted_class,
                    confidence_score=confidence,
                    model_version=model_version,
                )
                logger.info(f"Raw probs from {model_version.model_name}: {probabilities}")

                probabilities = np.array([
                    probabilities.get("PNEUMONIA", 0.0),
                    probabilities.get("NORMAL", 0.0),
                ])
                
                # Keep probs for ensemble
                all_probs.append(probabilities)
                model_weights.append(model_version.performance_metrics.get("accuracy", 1.0))  # default weight = 1
                used_models.append(model_version)
                
                logger.info(f"Analysis saved for model version {model_version.model_name}")
            except Exception as e:
                logger.error(f"Error during classification with model version {model_version.id}: {str(e)}")
                return Response({"error": "Classification failed"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        logger.info(f"Collected probabilities from {len(all_probs)} models for ensemble.")
        logger.info(f"Model weights: {model_weights}")
        logger.info(f"Used models: {[mv.model_name for mv in used_models]}")
        
        # === ENSEMBLE CALCULATION ===
        all_probs = np.array(all_probs)
        weights = np.array(model_weights) / np.sum(model_weights)  # normalize
        avg_probs = np.average(all_probs, axis=0, weights=weights)
        
        logger.info(f"Ensemble probabilities: {avg_probs}, weights: {weights}")

        final_idx = int(np.argmax(avg_probs))
        final_label = "pneumonia" if final_idx == 0 else "normal"   # adjust to your label ordering
        final_conf = float(np.max(avg_probs))
        
        logger.info(f"Ensemble result: {final_label} with confidence {final_conf:.4f}")
        
        # Save ensemble result
        ensemble = EnsembleResult.objects.create(
            scan=scan,
            method="weighted",
            combined_prediction_label=final_label,
            combined_confidence_score=final_conf,
        )

        # Save relations with weights
        for model_version, weight in zip(used_models, weights):
            EnsembleResultModel.objects.create(
                ensemble_result=ensemble,
                model_version=model_version,
                weight=float(weight),
            )

        return Response({"message": "Upload + classification + ensemble successful"}, status=status.HTTP_201_CREATED)      

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