from rest_framework import serializers
from .models import MedicalCase, ChestScan, ModelVersion, AIAnalysis, DoctorAnnotation, EnsembleResult, AuditLog
from users.serializers import UserSerializer
from users.models import User

# Nested secondary serializers
class ModelVersionSerializer(serializers.ModelSerializer):
    uploaded_by_admin = UserSerializer(read_only=True)

    class Meta:
        model = ModelVersion
        fields = '__all__'

class AIAnalysisSerializer(serializers.ModelSerializer):
    model_version = ModelVersionSerializer(read_only=True)
    
    class Meta:
        model = AIAnalysis
        fields = '__all__'

class EnsembleResultSerializer(serializers.ModelSerializer):
    source_analyses = AIAnalysisSerializer(many=True, read_only=True)    
    
    class Meta:
        model = EnsembleResult
        fields = '__all__'

class DoctorAnnotationSerializer(serializers.ModelSerializer):
    # Read: Returns full doctor object and scan ID
    # Write: Accepts doctor_id and scan_id for creation/updates
    doctor = UserSerializer(read_only=True)
    scan = serializers.IntegerField(source='scan.id', read_only=True)
    
    doctor_id = serializers.PrimaryKeyRelatedField(
        queryset=User.objects.all(), source='doctor', write_only=True
    )
    scan_id = serializers.PrimaryKeyRelatedField(
        queryset=ChestScan.objects.all(), source='scan', write_only=True
    )

    class Meta:
        model = DoctorAnnotation
        fields = fields = [
            'id', 'notes', 'created_at',
            'doctor', 'scan',  # Read fields
            'doctor_id', 'scan_id'  # Write fields
        ]


# Main Serializers for the API Endpoints
class ChestScanSerializer(serializers.ModelSerializer):
    # Related data
    ai_analyses = AIAnalysisSerializer(many=True, read_only=True)
    ensemble_result = EnsembleResultSerializer(read_only=True)
    annotations = DoctorAnnotationSerializer(many=True, read_only=True)
    
    class Meta:
        model = ChestScan
        fields = '__all__'


class MedicalCaseSerializer(serializers.ModelSerializer):
    """
    Base serializer for medical cases.
    
    Read: Returns full user objects for patient and doctor
    Write: Accepts patient_id and primary_doctor_id
    """
    # Read fields - full user objects
    patient = UserSerializer(read_only=True)
    primary_doctor = UserSerializer(read_only=True)
    
    # Write fields - IDs only
    # patient_id = serializers.IntegerField(write_only=True)
    # primary_doctor_id = serializers.IntegerField(write_only=True)
    
    patient_id = serializers.PrimaryKeyRelatedField(
        queryset=User.objects.all(), source='patient', write_only=True
    )
    primary_doctor_id = serializers.PrimaryKeyRelatedField(
        queryset=User.objects.all(), source='primary_doctor', write_only=True
    )

    class Meta:
        model = MedicalCase
        fields = [
            'id', 'title', 'description', 'status', 'diagnosis_summary',
            'created_at', 'updated_at', 'patient', 'primary_doctor', 
            'patient_id', 'primary_doctor_id'
        ]

    # def create(self, validated_data):
    #     # Use the IDs from the write_only fields to create the case
    #     validated_data['patient_id'] = validated_data.pop('patient_id')
    #     validated_data['primary_doctor_id'] = validated_data.pop('primary_doctor_id')
    #     return super().create(validated_data)


class MedicalCaseDetailSerializer(MedicalCaseSerializer):
    """Extends the base serializer to include nested scans for detail views."""
    scans = ChestScanSerializer(many=True, read_only=True)

    class Meta(MedicalCaseSerializer.Meta):
        fields = MedicalCaseSerializer.Meta.fields + ['scans']


class AuditLogSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = AuditLog
        fields = '__all__'