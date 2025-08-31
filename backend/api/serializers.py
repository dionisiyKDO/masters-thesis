from rest_framework import serializers
from .models import MedicalCase, ChestScan, ModelVersion, AIAnalysis, DoctorAnnotation
from users.serializers import UserSerializer
from users.models import User

# Nested Serializers for Detailed Views
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

class DoctorAnnotationSerializer(serializers.ModelSerializer):
    # When reading
    # expose doctor as full table
    # expose scan as plain integer id
    doctor = UserSerializer(read_only=True)
    scan = serializers.IntegerField(source='scan.id', read_only=True)
    
    doctor_id = serializers.PrimaryKeyRelatedField(
        queryset=User.objects.all(), source='doctor', write_only=True
    )
    scan_id = serializers.PrimaryKeyRelatedField(
        queryset=ChestScan.objects.all(), source='scan', write_only=True
    )
    # On write, you pass doctor_id: 1, scan_id: 42. 
    # DRF will auto-resolve them into User and ChestScan objects.
    
    # On read, you get the full doctor object (via UserSerializer) 
    # and just IDs for doctor_id / scan_id.

    class Meta:
        model = DoctorAnnotation
        fields = ['id', 'notes', 'doctor', 'doctor_id', 'scan', 'scan_id', 'created_at']
        
    def to_internal_value(self, data):
        print("Incoming raw data:", data)  # <-- shows exactly what DRF got from request
        return super().to_internal_value(data)

class ChestScanSerializer(serializers.ModelSerializer):
    # When reading a scan, nest its analyses and annotations
    ai_analyses = AIAnalysisSerializer(many=True, read_only=True)
    annotations = DoctorAnnotationSerializer(many=True, read_only=True)

    class Meta:
        model = ChestScan
        fields = '__all__'


# Main Serializers for the API Endpoints

class MedicalCaseSerializer(serializers.ModelSerializer):
    # When reading, show user details, not just IDs
    patient = UserSerializer(read_only=True)
    primary_doctor = UserSerializer(read_only=True)
    
    # When writing, we only need the IDs
    patient_id = serializers.IntegerField(write_only=True)
    primary_doctor_id = serializers.IntegerField(write_only=True)

    class Meta:
        model = MedicalCase
        fields = [
            'id', 'title', 'description', 'status', 'diagnosis_summary',
            'created_at', 'updated_at', 'patient', 'primary_doctor', 
            'patient_id', 'primary_doctor_id'
        ]

    def create(self, validated_data):
        # Use the IDs from the write_only fields to create the case
        validated_data['patient_id'] = validated_data.pop('patient_id')
        validated_data['primary_doctor_id'] = validated_data.pop('primary_doctor_id')
        return super().create(validated_data)


class MedicalCaseDetailSerializer(MedicalCaseSerializer):
    """Extends the base serializer to include nested scans for detail views."""
    scans = ChestScanSerializer(many=True, read_only=True)

    class Meta(MedicalCaseSerializer.Meta):
        fields = MedicalCaseSerializer.Meta.fields + ['scans']

