from django.db import models
from django.conf import settings

User = settings.AUTH_USER_MODEL

class MedicalCase(models.Model):
    STATUS_CHOICES = [
        ('open', 'Open'),
        ('closed', 'Closed'),
        ('archived', 'Archived'),
    ]

    patient = models.ForeignKey(User, on_delete=models.CASCADE, related_name="cases_as_patient")
    primary_doctor = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="cases_as_doctor")
    
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="open")
    diagnosis_summary = models.TextField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Case {self.id} - {self.title}"


class ChestScan(models.Model):
    LABEL_CHOICES = [
        ('normal', 'Normal'),
        ('pneumonia', 'Pneumonia'),
    ]
    
    case = models.ForeignKey(MedicalCase, on_delete=models.CASCADE, related_name="scans")
    image_path = models.ImageField(upload_to="scans/")
    final_label = models.CharField(max_length=20, choices=LABEL_CHOICES, blank=True, null=True)
    final_label_set_at = models.DateTimeField(blank=True, null=True)
    
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Scan {self.id} - '{self.image_path}'"


class ModelVersion(models.Model):
    uploaded_by_admin = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="uploaded_models")
    model_name = models.CharField(max_length=255) # TODO: maybe get AVAIBLE_MODELS from classifier as ENUM
    storage_uri = models.CharField(max_length=500)
    description = models.TextField(blank=True, null=True)
    performance_metrics = models.JSONField(default=dict)
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model_name} (active={self.is_active})"


class AIAnalysis(models.Model):
    LABEL_CHOICES = [
        ('pneumonia', 'Pneumonia'),
        ('normal', 'Normal'),
    ]
    
    HEATMAP_TYPE_CHOICES = [
        ('gradcam', 'Grad-CAM'),
        ('saliency', 'Saliency Map'),
        ('none', 'None'),
    ]

    scan = models.ForeignKey(ChestScan, on_delete=models.CASCADE, related_name="ai_analyses")
    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name="ai_model")
    prediction_label = models.CharField(max_length=20, choices=LABEL_CHOICES)
    confidence_score = models.FloatField()
    heatmap_path = models.ImageField(upload_to="heatmaps/", null=True, blank=True)
    heatmap_type = models.CharField(max_length=20, choices=HEATMAP_TYPE_CHOICES, default='none')
    generated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analysis {self.id} - {self.prediction_label} ({self.confidence_score:.2f})"

class EnsembleResult(models.Model):
    METHODS = [
        ("majority_vote", "Majority Vote"),
        ("average", "Average"),
        ("weighted", "Weighted"),
        ("stacking", "Stacking"),
    ]
    
    LABEL_CHOICES = [
        ('pneumonia', 'Pneumonia'),
        ('normal', 'Normal'),
    ]
    
    scan = models.OneToOneField(ChestScan, on_delete=models.CASCADE, related_name="ensemble_result")
    source_analyses = models.ManyToManyField(AIAnalysis, related_name="used_in_ensembles")
    
    method = models.CharField(max_length=50, choices=METHODS)
    combined_prediction_label = models.CharField(max_length=20, choices=LABEL_CHOICES)
    combined_confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Ensemble {self.id} - {self.combined_prediction_label} ({self.combined_confidence_score:.2f})"


class DoctorAnnotation(models.Model):
    scan = models.ForeignKey(ChestScan, on_delete=models.CASCADE, related_name="annotations")
    doctor = models.ForeignKey(User, on_delete=models.CASCADE, related_name="annotations")
    notes = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Annotation {self.id} by {self.doctor}"


class AuditLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    action = models.CharField(max_length=255)
    details = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"[{self.created_at}] {self.user} - {self.action}"
