from rest_framework import serializers
from .models import MedicalCase, ChestScan, ModelVersion, AIAnalysis, DoctorAnnotation
from users.serializers import UserSerializer # Import the simple UserSerializer

# --- Nested Serializers for Detailed Views ---

class AIAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = AIAnalysis
        fields = '__all__'

class DoctorAnnotationSerializer(serializers.ModelSerializer):
    doctor = UserSerializer(read_only=True)

    class Meta:
        model = DoctorAnnotation
        fields = '__all__'

class ChestScanSerializer(serializers.ModelSerializer):
    # When reading a scan, nest its analyses and annotations
    ai_analyses = AIAnalysisSerializer(many=True, read_only=True)
    annotations = DoctorAnnotationSerializer(many=True, read_only=True)

    class Meta:
        model = ChestScan
        fields = '__all__'


# --- Main Serializers for the API Endpoints ---

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


class ModelVersionSerializer(serializers.ModelSerializer):
    uploaded_by_admin = UserSerializer(read_only=True)

    class Meta:
        model = ModelVersion
        fields = '__all__'