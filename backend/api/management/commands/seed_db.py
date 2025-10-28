# backend/api/management/commands/seed_db.py
import random
from faker import Faker
from collections import Counter

from django.core.management.base import BaseCommand
from django.db import transaction

from users.models import User, DoctorProfile, PatientProfile
from api.models import (
    MedicalCase, ChestScan, ModelVersion, AIAnalysis, DoctorAnnotation,
    EnsembleResult, AuditLog
)

# --- Configuration ---
# Easily scale the amount of data to generate
DEFAULT_DOCTORS = 2
DEFAULT_PATIENTS = 6
MAX_CASES_PER_PATIENT = 3
MAX_SCANS_PER_CASE = 5
MAX_ANNOTATIONS_PER_SCAN = 2

# Probabilities for realistic data
PROB_AI_AGREES_WITH_BASE = 0.90
PROB_FINAL_LABEL_AGREES_WITH_BASE = 0.80

SCANS = [
    'seeding/NORMAL_IM-0007-0001_original.jpeg',
    'seeding/NORMAL_IM-0019-0001_original.jpeg',
    'seeding/NORMAL_IM-0025-0001_aug2_hflip_bright1.02.jpeg',
    'seeding/NORMAL_IM-0029-0001_original.jpeg',
    'seeding/NORMAL_IM-0033-0001-0001_original.jpeg',
    'seeding/NORMAL_IM-0033-0001-0002_aug2_rot4.8_cont1.07_blur.jpeg',
    'seeding/NORMAL_IM-0037-0001_aug2_hflip_blur.jpeg',
    'seeding/NORMAL_IM-0043-0001_original.jpeg',
    'seeding/NORMAL_IM-0073-0001_aug1_hflip_bright0.99_blur_color0.98.jpeg',
    'seeding/NORMAL_IM-0077-0001_aug1_hflip_cont1.12_blur.jpeg',
    'seeding/NORMAL_IM-0095-0001_original.jpeg',
    'seeding/NORMAL_IM-0111-0001_aug2_rot-11.9_bright0.82_cont0.97_blur_color0.98.jpeg',
    'seeding/NORMAL_IM-0115-0001_aug2_bright1.15_cont1.07_blur_color0.92.jpeg',
    'seeding/NORMAL_IM-0117-0001_aug1_rot13.5_cont0.84_blur_color1.06.jpeg',
    'seeding/NORMAL_IM-0117-0001_aug2_hflip_cont0.87.jpeg',
    'seeding/NORMAL_IM-0131-0001_aug1_rot-13.1_hflip_bright0.82_cont1.05_blur.jpeg',
]

MODELS = [
    {
        "name": "AlexNet",
        "uri": "/home/dionisiy/masters-thesis/backend/classifier/outputs/checkpoints/Saved/AlexNet.epoch27-val_acc0.9761.hdf5",
        "desc": "AlexNet based model, fast but less accurate than newer architectures.",
        "metrics": {"accuracy": 0.9761, "f1_score": 0.88, "precision": 0.89},
        "active": True
    },
    {
        "name": "OwnV1",
        "uri": "/home/dionisiy/masters-thesis/backend/classifier/outputs/checkpoints/Saved/OwnV1.epoch26-val_acc0.9761.hdf5",
        "desc": "Initial release of the OwnV1 model based on a custom CNN architecture.",
        "metrics": {"accuracy": 0.9761, "f1_score": 0.89, "precision": 0.90},
        "active": False # An older, inactive model
    },
    {
        "name": "OwnV2",
        "uri": "/home/dionisiy/masters-thesis/backend/classifier/outputs/checkpoints/Saved/OwnV2.epoch28-val_acc0.9705.hdf5",
        "desc": "Second version of the custom CNN architecture with improvements over V1.",
        "metrics": {"accuracy": 0.9705, "f1_score": 0.90, "precision": 0.91},
        "active": True
    },
    {
        "name": "OwnV3",
        "uri": "/home/dionisiy/masters-thesis/backend/classifier/outputs/checkpoints/Saved/OwnV3.epoch50-val_acc0.9830.hdf5",
        "desc": "Latest version of the custom CNN architecture with best performance.",
        "metrics": {"accuracy": 0.9830, "f1_score": 0.91, "precision": 0.92},
        "active": True
    },
    {
        "name": "VGG16",
        "uri": "/home/dionisiy/masters-thesis/backend/classifier/outputs/checkpoints/Saved/VGG16.epoch18-val_acc0.9534.hdf5",
        "desc": "VGG16 based model, similar performance to InceptionV3 but slower inference.",
        "metrics": {"accuracy": 0.9534, "f1_score": 0.86, "precision": 0.87},
        "active": True
    },
    {
        "name": "VGG19",
        "uri": "/home/dionisiy/masters-thesis/backend/classifier/outputs/checkpoints/Saved/VGG19.epoch29-val_acc0.9489.hdf5",
        "desc": "VGG19 based model with slightly lower accuracy than VGG16.",
        "metrics": {"accuracy": 0.9489, "f1_score": 0.85, "precision": 0.86},
        "active": True
    },
    # {
    #     "name": "ResNet50",
    #     "uri": "/home/dionisiy/masters-thesis/backend/classifier/outputs/checkpoints/Saved/ResNet50.epoch22-val_acc0.8818.hdf5",
    #     "desc": "ResNet50 based model with moderate performance.",
    #     "metrics": {"accuracy": 0.8818, "f1_score": 0.75, "precision": 0.76},
    #     "active": True
    # },
]

# Initialize Faker
fake = Faker()

class Command(BaseCommand):
    help = "Seeds the database with realistic mock data."

    def add_arguments(self, parser):
        parser.add_argument("--doctors", type=int, default=DEFAULT_DOCTORS)
        parser.add_argument("--patients", type=int, default=DEFAULT_PATIENTS)

    @transaction.atomic
    def handle(self, *args, **options):
        self.stdout.write("Deleting old data...")
        self._clear_db()

        self.stdout.write("Creating new data...")
        
        # Admin user is needed to log creation of other users/models
        admin = self._create_admin()
        
        # Pass admin to functions that need to log actions
        doctors = self._create_doctors(options["doctors"], admin)
        patients = self._create_patients(options["patients"], admin)
        models = self._create_models(admin)

        # Pass admin for system-level logging (AI runs, ensembles)
        total_cases, total_scans = self._create_cases_and_scans(patients, doctors, models, admin)
        self._create_ensembles(models, admin) # Pass admin

        # Get the total count of logs created dynamically
        total_logs = AuditLog.objects.count()

        self.stdout.write(self.style.SUCCESS(
            f"Seed complete!\n"
            f"  - {len(doctors)} doctors\n"
            f"  - {len(patients)} patients\n"
            f"  - {total_cases} medical cases\n"
            f"  - {total_scans} chest scans\n"
            f"  - {len(models)} models\n"
            f"  - {total_logs} audit logs"
        ))

    #region Helper Methods

    def _clear_db(self):
        EnsembleResult.objects.all().delete()
        AuditLog.objects.all().delete()
        AIAnalysis.objects.all().delete()
        DoctorAnnotation.objects.all().delete()
        ChestScan.objects.all().delete()
        MedicalCase.objects.all().delete()
        ModelVersion.objects.all().delete()
        DoctorProfile.objects.all().delete()
        PatientProfile.objects.all().delete()
        User.objects.all().delete()

    def _create_log(self, user, action, message):
        """Helper to create an audit log entry."""
        AuditLog.objects.create(
            user=user,
            action=action,
            details={"message": message}
        )

    def _create_admin(self):
        return User.objects.create_superuser(
            username="admin",
            email="admin@example.com",
            password="admin",
            role="admin",
        )

    def _create_doctors(self, num, admin):
        """Create doctors and log their creation."""
        specializations = ['Radiology', 'Pulmonology', 'Cardiology', 'General Medicine', 'Oncology']
        doctors = []
        for _ in range(num):
            first, last = fake.first_name(), fake.last_name()
            username = f"dr_{first.lower()}_{last.lower()}{random.randint(1,99)}"
            user = User.objects.create_user(
                username=username,
                password=username,
                role="doctor",
                email=f"{first.lower()}.{last.lower()}@clinic.com",
                first_name=first,
                last_name=last,
            )
            DoctorProfile.objects.create(
                user=user,
                license_number=fake.unique.numerify("DOC-#######"),
                specialization=random.choice(specializations),
            )
            doctors.append(user)
            # LOGGING: Log creation by admin
            self._create_log(admin, 'CREATED_USER', f"Admin created doctor '{user.username}'")
        return doctors

    def _create_patients(self, num, admin):
        """Create patients and log their creation."""
        patients = []
        for _ in range(num):
            first, last = fake.first_name(), fake.last_name()
            username = f"pat_{first.lower()}_{last.lower()}{random.randint(1,99)}"
            user = User.objects.create_user(
                username=username,
                password=username,
                role="patient",
                email=fake.email(),
                first_name=first,
                last_name=last,
            )
            PatientProfile.objects.create(
                user=user,
                dob=fake.date_of_birth(minimum_age=18, maximum_age=90),
                sex=random.choice(["M", "F", "O"]),
                medical_record_number=fake.unique.numerify("MRN-##########"),
            )
            patients.append(user)
            # LOGGING: Log creation by admin
            self._create_log(admin, 'CREATED_USER', f"Admin created patient '{user.username}'")
        return patients

    def _create_models(self, admin):
        """Create models and log their creation."""
        models = []
        for m in MODELS:
            model = ModelVersion.objects.create(
                model_name=m["name"],
                storage_uri=m["uri"],
                description=m["desc"],
                performance_metrics=m["metrics"],
                uploaded_by_admin=admin,
                is_active=m["active"],
            )
            models.append(model)
            # LOGGING: Log creation by admin
            self._create_log(admin, 'CREATED_MODEL', f"Admin created model '{model.model_name}'")
            if not m["active"]:
                # LOGGING: Log deactivation if created as inactive
                self._create_log(admin, 'DEACTIVATED_MODEL', f"Model '{model.model_name}' created as inactive")
        return models

    def _create_cases_and_scans(self, patients, doctors, models, admin):
        case_titles = [
            "Annual Checkup", "Follow-up for persistent cough",
            "Pre-operative assessment", "Post-treatment evaluation",
            "Emergency room visit for dyspnea",
        ]
        statuses = ["open", "closed", "archived"]
        ai_labels = ["pneumonia", "normal"]
        doc_notes = [
            "Findings are consistent with the AI analysis.", "No acute abnormalities seen.",
            "Slight opacity in the lower lobe, requires monitoring.",
            "Signs of minor inflammation present.", "Recommend follow-up CT for confirmation.",
            "Patient's condition appears stable.",
            "AI heatmap correctly identifies the region of interest.",
            "Annotation disagrees with AI. Manual review needed.",
        ]

        total_cases, total_scans = 0, 0
        active_models = [m for m in models if m.is_active]

        for patient in patients:
            for _ in range(random.randint(2, MAX_CASES_PER_PATIENT)):
                total_cases += 1
                doctor = random.choice(doctors)
                case = MedicalCase.objects.create(
                    patient=patient,
                    primary_doctor=doctor,
                    title=f"{random.choice(case_titles)}",
                    description=fake.paragraph(nb_sentences=3),
                    diagnosis_summary=random.choice(["", fake.paragraph(nb_sentences=2)]),
                    status=random.choice(statuses),
                )
                # LOGGING: Log case creation by the assigned doctor
                self._create_log(doctor, 'CREATED_CASE', f"Doctor '{doctor.username}' created case {case.id} for patient '{patient.username}'")

                for i in range(random.randint(2, MAX_SCANS_PER_CASE)):
                    total_scans += 1

                    # Determine a "base truth" for this scan
                    base_scan_label = random.choice(ai_labels)
                    
                    # Determine the doctor's final_label (mostly matches base truth)
                    if random.random() < PROB_FINAL_LABEL_AGREES_WITH_BASE:
                        final_label = base_scan_label
                    else: # Doctor disagrees with the consensus
                        final_label = "pneumonia" if base_scan_label == "normal" else "normal"

                    scan = ChestScan.objects.create(
                        case=case, 
                        image_path=random.choice(SCANS),
                        final_label=final_label,
                        final_label_set_at=fake.date_time_this_year(),
                    )
                    # LOGGING: Log scan upload by the doctor
                    self._create_log(doctor, 'UPLOADED_SCAN', f"Doctor '{doctor.username}' uploaded scan {scan.id} for case {case.id}")

                    # Each scan gets an AI analysis from ACTIVE models
                    for model in active_models:
                        # AI models mostly agree with the "base truth"
                        if random.random() < PROB_AI_AGREES_WITH_BASE:
                            prediction_label = base_scan_label
                            confidence = round(random.uniform(0.80, 0.99), 4)
                        else: # Model "disagrees"
                            prediction_label = "pneumonia" if base_scan_label == "normal" else "normal"
                            confidence = round(random.uniform(0.55, 0.75), 4)

                        AIAnalysis.objects.create(
                            scan=scan,
                            model_version=model,
                            prediction_label=prediction_label,
                            confidence_score=confidence,
                            heatmap_path=f"heatmaps/scan_{scan.id}_model_{model.id}.png",
                            heatmap_type="gradcam",
                        )
                        # LOGGING: Log AI run as a system (admin) action
                        self._create_log(admin, 'RAN_AI_ANALYSIS', f"System ran analysis on scan {scan.id} with model '{model.model_name}'")

                    # Each scan is annotated by the primary doctor
                    for i in range(random.randint(2, MAX_ANNOTATIONS_PER_SCAN)):
                        DoctorAnnotation.objects.create(
                            scan=scan,
                            doctor=doctor,
                            notes=f"{random.choice(doc_notes)}",
                        )
                        # LOGGING: Log annotation by the doctor
                        self._create_log(doctor, 'ADDED_ANNOTATION', f"Doctor '{doctor.username}' added annotation to scan {scan.id}")

        return total_cases, total_scans

    def _create_ensembles(self, models, admin):
        """Create realistic ensemble results based on actual AIAnalyses."""
        scans = ChestScan.objects.all()
        methods = ["majority_vote", "average", "weighted"]

        for scan in scans:
            analyses = list(scan.ai_analyses.all())
            if not analyses:
                continue

            method = random.choice(methods)

            # Use Counter to find the most common prediction_label
            label_votes = Counter([a.prediction_label for a in analyses])
            
            if not label_votes:
                continue # Should be covered by 'if not analyses'

            # `most_common(1)` returns [('label', count)]
            combined_prediction_label = label_votes.most_common(1)[0][0]

            # Calculate average confidence for the winning label
            winning_confidences = [
                a.confidence_score 
                for a in analyses 
                if a.prediction_label == combined_prediction_label
            ]
            
            if winning_confidences:
                conf = round(sum(winning_confidences) / len(winning_confidences), 4)
            else:
                # Fallback: average all (shouldn't happen, but safe)
                conf = round(sum(a.confidence_score for a in analyses) / len(analyses), 4)

            ensemble = EnsembleResult.objects.create(
                scan=scan,
                method=method,
                combined_prediction_label=combined_prediction_label,
                combined_confidence_score=conf,
            )
            
            # Set M2M relationship
            ensemble.source_analyses.set(analyses)
            
            # LOGGING: Log ensemble creation as a system (admin) action
            self._create_log(admin, 'CREATED_ENSEMBLE', f"System created ensemble for scan {scan.id} using '{method}'")
