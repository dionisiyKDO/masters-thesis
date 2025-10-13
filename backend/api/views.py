
import logging
import numpy as np
from io import BytesIO
import cv2

from django.core.files.base import ContentFile
from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets, permissions
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import (
    MedicalCase, ChestScan, ModelVersion, 
    AIAnalysis, DoctorAnnotation, EnsembleResult, 
    AuditLog
    )
from .serializers import ( 
    MedicalCaseSerializer, MedicalCaseDetailSerializer, 
    DoctorAnnotationSerializer, ChestScanSerializer,
    ModelVersionSerializer, AuditLogSerializer,
    )
from .permissions import IsDoctor, IsPatientOfCase, IsDoctorOfCase, IsAdmin
from .classifier.classifier import Classifier
from .classifier.config import config_v1 as config

#region Configure logging
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
    ViewSet for Medical Cases with role-based access control.
    
    Permissions:
    - Patients: Can only view their own cases
    - Doctors: Can view cases assigned to them, create new cases
    - Admins: Full access to all cases
    """
    queryset = MedicalCase.objects.select_related('patient', 'primary_doctor').prefetch_related('scans')
    
    def get_serializer_class(self):
        """Return appropriate serializer based on action."""
        if self.action == 'retrieve':
            return MedicalCaseDetailSerializer
        return MedicalCaseSerializer
    
    def get_permissions(self):
        """Set permissions based on action."""
        if self.action in ['list', 'retrieve', 'by_patient']:
            permission_classes = [permissions.IsAuthenticated, IsPatientOfCase | IsDoctorOfCase | IsAdmin]
        elif self.action in ['create', 'update', 'partial_update']:
            permission_classes = [IsDoctor | IsAdmin]
        else:  # destroy
            permission_classes = [IsAdmin]
        
        return [permission() for permission in permission_classes]
    
    def get_queryset(self):
        """Filter queryset based on user role."""
        user = self.request.user
        
        if user.role == 'patient':
            return self.queryset.filter(patient=user)
        elif user.role == 'doctor':
            return self.queryset.filter(primary_doctor=user)
        
        return self.queryset # Admins see all cases

    @action(detail=False, methods=['get'], url_path='patient/(?P<patient_id>[^/.]+)')
    def by_patient(self, request, patient_id=None):
        try:
            patient_id = int(patient_id)
        except (TypeError, ValueError):
            return Response(
                {"detail": "Invalid patient ID format."}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        queryset = self.get_queryset() # self.get_queryset() applies role-based filtering
        cases = queryset.filter(patient_id=patient_id) # Apply the specific patient ID filter
        
        if not cases.exists():
            return Response(
                {"detail": "No cases found."}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        serializer = self.get_serializer(cases, many=True)
        return Response(serializer.data)


class ScanUploadView(APIView):
    """
    Handle chest scan uploads with automatic AI analysis.
    
    Processes uploaded images through all active AI models and generates
    ensemble predictions.
    """
    permission_classes = [IsDoctor]
    parser_classes = [MultiPartParser, FormParser]

    def _save_heatmap_to_imagefield(self, array: np.ndarray, instance, field_name="heatmap_path", filename="heatmap.png"):
        # Convert NumPy array (BGR from OpenCV) → PNG bytes
        success, buffer = cv2.imencode(".png", array)
        if not success:
            raise ValueError("Failed to encode heatmap array to PNG")

        # Wrap bytes into Django ContentFile
        file_obj = ContentFile(buffer.tobytes())
        getattr(instance, field_name).save(filename, file_obj, save=False)

    def _run_ai_analysis(self, scan):
        """Run AI analysis on scan using all active models."""
        # Get all the active models
        model_versions = ModelVersion.objects.filter(is_active=True)
        if not model_versions.exists():
            return Response({"error": "No active model versions available"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        logger.info(f"Found {model_versions.count()} active model versions: {[mv.model_name for mv in model_versions]}")
        
        analyses = []

        # Loop through all active models, get and save their individual result's 
        for model_version in model_versions:
            try:
                classifier = Classifier(
                    model_version.model_name,
                    img_size=config['img_size'],
                    data_dir='./api/classifier/data'
                )
                classifier.load_model(model_version.storage_uri)
                predicted_class, confidence, probabilities = classifier.predict(scan.image_path.path)
                superimposed, heatmap, original = classifier.generate_gradcam_heatmap(
                    image_path=scan.image_path.path,
                    output_dir="./gradcam_results"
                )
                
                # Save individual prediction
                analysis = AIAnalysis.objects.create(
                    scan=scan,
                    prediction_label=predicted_class,
                    confidence_score=confidence,
                    model_version=model_version,
                )
                
                if superimposed is not None: 
                    self._save_heatmap_to_imagefield(superimposed, analysis)
                    analysis.heatmap_type = "gradcam"
                
                analysis.save()
                logger.info(f"Raw probs from {model_version.model_name}: {probabilities}")
                
                probabilities = np.array([
                    probabilities.get("PNEUMONIA", 0.0),
                    probabilities.get("NORMAL", 0.0),
                ])
                
                analyses.append({
                    'analysis': analysis,
                    'probabilities': probabilities,
                    'model_version': model_version
                })
                
                logger.info(f"Completed analysis with model {model_version.model_name}: {predicted_class} ({confidence:.4f})")
                
            except Exception as e:
                logger.error(f"Error with model {model_version.model_name}: {str(e)}")
                return {
                    'success': False,
                    'error': f"Analysis failed with model {model_version.model_name}"
                }
        
        return {'success': True, 'analyses': analyses}
    
    def _create_ensemble_result(self, scan, analysis_results):
        """Create ensemble prediction from individual analyses."""
        all_probs = []
        model_weights = []
        source_analyses = []
        
        for result in analysis_results:
            # Extract probabilities in consistent order
            probs = result['probabilities']
            
            all_probs.append(probs)
            model_weights.append(result['model_version'].performance_metrics.get("accuracy", 1.0))
            source_analyses.append(result['analysis'])
        
        # Calculate weighted ensemble
        all_probs = np.array(all_probs)
        weights = np.array(model_weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        logger.info(f"Collected probabilities from {len(all_probs)} models for ensemble.")
        logger.info(f"Model weights: {model_weights}")
        
        ensemble_probs = np.average(all_probs, axis=0, weights=weights)
        
        # Determine final prediction
        final_idx = int(np.argmax(ensemble_probs))
        final_label = "pneumonia" if final_idx == 0 else "normal"
        final_confidence = float(np.max(ensemble_probs))
        
        logger.info(f"Ensemble calculation: {final_label} with confidence {final_confidence:.4f}")
        
        # Create ensemble result
        ensemble = EnsembleResult.objects.create(
            scan=scan,
            method="weighted_accuracy",
            combined_prediction_label=final_label,
            combined_confidence_score=final_confidence,
        )
        
        # Link source analyses
        ensemble.source_analyses.set(source_analyses)
        
        return ensemble
    
    def post(self, request, case_id):
        """Upload and analyze a chest scan."""
        try:
            file = request.data.get("image")
            if not file:
                return Response({"error": "No image file provided"}, status=status.HTTP_400_BAD_REQUEST)

            # Get case and create scan
            case = get_object_or_404(MedicalCase, id=case_id)
            scan = ChestScan.objects.create(case=case, image_path=file)
            logger.info(f"Created new scan {scan.id} for case {case_id}")
            
            analysis_results = self._run_ai_analysis(scan)
            if not analysis_results['success']:
                return Response(
                    {"error": analysis_results['error']}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                
            # Generate ensemble result
            ensemble_result = self._create_ensemble_result(scan, analysis_results['analyses'])
            
            logger.info(f"Successfully processed scan {scan.id} with ensemble result: {ensemble_result.combined_prediction_label}")
            
            return Response({
                "message": "Scan uploaded and analyzed successfully",
                "scan_id": scan.id,
                "ensemble_prediction": {
                    "label": ensemble_result.combined_prediction_label,
                    "confidence": ensemble_result.combined_confidence_score
                }
            }, status=status.HTTP_201_CREATED) 

        except Exception as e:
            logger.error(f"Unexpected error in scan upload: {str(e)}")
            return Response(
                {"error": "An unexpected error occurred during processing"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class DoctorAnnotationViewSet(viewsets.ModelViewSet):
    """
    ViewSet for doctor annotations on scans.
    
    Only doctors can create/modify annotations.
    Patients can view annotations on their scans.
    """
    queryset = DoctorAnnotation.objects.select_related('doctor', 'scan__case').all()
    serializer_class = DoctorAnnotationSerializer

    def get_permissions(self):
        """Set permissions based on action."""
        if self.action in ['list', 'retrieve']:
            permission_classes = [IsPatientOfCase | IsDoctorOfCase | IsAdmin]
        else:  # create, update, destroy
            permission_classes = [IsDoctorOfCase | IsAdmin]
        
        return [permission() for permission in permission_classes]

class ChestScanViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Chest Scans.
    
    Permissions:
    - Patients: Can view scans related to their cases
    - Doctors: Can view scans related to their cases
    - Admins: Full access to all scans
    """
    queryset = ChestScan.objects.select_related('case__patient', 'case__primary_doctor').prefetch_related('ai_analyses', 'annotations', 'ensemble_result')
    serializer_class = ChestScanSerializer

    def get_permissions(self):
        """Set permissions based on action."""
        if self.action in ['list', 'retrieve']:
            permission_classes = [permissions.IsAuthenticated, IsPatientOfCase | IsDoctorOfCase | IsAdmin]
        else:  # create, update, destroy
            permission_classes = [IsDoctor | IsAdmin]
        
        return [permission() for permission in permission_classes]

    def get_queryset(self):
        """Filter queryset based on user role."""
        user = self.request.user
        
        if user.role == 'patient':
            return self.queryset.filter(case__patient=user)
        elif user.role == 'doctor':
            return self.queryset.filter(case__primary_doctor=user)
        
        return self.queryset # Admins see all scans


class ModelVersionViewSet(viewsets.ModelViewSet):
    queryset = ModelVersion.objects.all()
    serializer_class = ModelVersionSerializer
    permission_classes = [ IsAdmin ]

    @action(detail=False, methods=["get"], permission_classes=[IsAdmin])
    def stats(self, request):
        total_models = ModelVersion.objects.count()
        active_models = ModelVersion.objects.filter(is_active=True).count()
        

        return Response({
            "total_models": total_models,
            "active_models": active_models,
        })

class AuditLogViewSet(viewsets.ModelViewSet):
    queryset = AuditLog.objects.all().order_by('-created_at')
    serializer_class = AuditLogSerializer
    permission_classes = [ IsAdmin ]
    
    @action(detail=False, methods=["get"], permission_classes=[IsAdmin])
    def recent_stats(self, request):
        recent_errors = AuditLog.objects.filter(action='SYSTEM_ERROR').count()
                
        return Response({
            "recent_errors": recent_errors,
        })
    
    @action(detail=False, methods=["get"], permission_classes=[IsAdmin])
    def recent(self, request):
        recent_activity = AuditLog.objects.order_by('-created_at')[:10]
        recent_errors = AuditLog.objects.filter(action='SYSTEM_ERROR').order_by('-created_at')[:10]
        
        activity_data = AuditLogSerializer(recent_activity, many=True).data
        error_data = AuditLogSerializer(recent_errors, many=True).data
        
        return Response({
            "recent_activity": activity_data,
            "recent_errors": error_data,
        })