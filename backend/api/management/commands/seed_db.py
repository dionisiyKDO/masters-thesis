# backend/api/management/commands/seed_db.py
import random
from datetime import date
from faker import Faker

from django.core.management.base import BaseCommand
from django.db import transaction

from users.models import User, DoctorProfile, PatientProfile
from api.models import (
    MedicalCase, ChestScan, ModelVersion, AIAnalysis, DoctorAnnotation,
    EnsembleResult, EnsembleResultModel
)

# --- Configuration ---
# Easily scale the amount of data to generate
DEFAULT_DOCTORS = 2
DEFAULT_PATIENTS = 2
MAX_CASES_PER_PATIENT = 4
MAX_SCANS_PER_CASE = 5
MAX_ANNOTATIONS_PER_SCAN = 3

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
        "uri": "./checkpoints/Saved/AlexNet.epoch27-val_acc0.9761.hdf5",
        "desc": "AlexNet based model, fast but less accurate than newer architectures.",
        "metrics": {"accuracy": 0.9761, "f1_score": 0.88, "precision": 0.89},
        "active": True
    },
    {
        "name": "OwnV1",
        "uri": "./checkpoints/Saved/OwnV1.epoch26-val_acc0.9761.hdf5",
        "desc": "Initial release of the OwnV1 model based on a custom CNN architecture.",
        "metrics": {"accuracy": 0.9761, "f1_score": 0.89, "precision": 0.90},
        "active": False # An older, inactive model
    },
    {
        "name": "OwnV2",
        "uri": "./checkpoints/Saved/OwnV2.epoch28-val_acc0.9705.hdf5",
        "desc": "Second version of the custom CNN architecture with improvements over V1.",
        "metrics": {"accuracy": 0.9705, "f1_score": 0.90, "precision": 0.91},
        "active": True
    },
    {
        "name": "OwnV3",
        "uri": "./checkpoints/Saved/OwnV3.epoch50-val_acc0.9830.hdf5",
        "desc": "Latest version of the custom CNN architecture with best performance.",
        "metrics": {"accuracy": 0.9830, "f1_score": 0.91, "precision": 0.92},
        "active": True
    },
    {
        "name": "VGG16",
        "uri": "/checkpoints/Saved/VGG16.epoch18-val_acc0.9534.hdf5",
        "desc": "VGG16 based model, similar performance to InceptionV3 but slower inference.",
        "metrics": {"accuracy": 0.9534, "f1_score": 0.86, "precision": 0.87},
        "active": True
    },
    {
        "name": "VGG19",
        "uri": "/checkpoints/Saved/VGG19.epoch29-val_acc0.9489.hdf5",
        "desc": "VGG19 based model with slightly lower accuracy than VGG16.",
        "metrics": {"accuracy": 0.9489, "f1_score": 0.85, "precision": 0.86},
        "active": True
    },
    # {
    #     "name": "InceptionV3",
    #     "uri": "/checkpoints/Saved/InceptionV3.epoch13-val_acc0.9545.hdf5",
    #     "desc": "Transfer learning model using InceptionV3 as base. Good balance of speed and accuracy.",
    #     "metrics": {"accuracy": 0.9545, "f1_score": 0.87, "precision": 0.88},
    #     "active": True
    # },
    # {
    #     "name": "EfficientNetV2",
    #     "uri": "/checkpoints/Saved/EfficientNetV2.epoch01-val_acc0.7295.hdf5",
    #     "desc": "Early experiment with EfficientNetV2. Underperformed compared to others.",
    #     "metrics": {"accuracy": 0.7295, "f1_score": 0.65, "precision": 0.66},
    #     "active": False # An older, inactive model
    # },
    # {
    #     "name": "ResNet50",
    #     "uri": "/checkpoints/Saved/ResNet50.epoch22-val_acc0.8818.hdf5",
    #     "desc": "ResNet50 based model with moderate performance.",
    #     "metrics": {"accuracy": 0.8818, "f1_score": 0.75, "precision": 0.76},
    #     "active": True
    # },
    # {
    #     "name": "InceptionResNetV2",
    #     "uri": "/checkpoints/Saved/InceptionResNetV2.epoch16-val_acc0.9409.hdf5",
    #     "desc": "High capacity model with InceptionResNetV2 backbone. Computationally intensive.",
    #     "metrics": {"accuracy": 0.9409, "f1_score": 0.84, "precision": 0.85},
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
        admin = self._create_admin()
        doctors = self._create_doctors(options["doctors"])
        patients = self._create_patients(options["patients"])
        models = self._create_models(admin)

        total_cases, total_scans = self._create_cases_and_scans(patients, doctors, models)
        self._create_ensembles(models)

        self.stdout.write(self.style.SUCCESS(
            f"Seed complete!\n"
            f"  - {len(doctors)} doctors\n"
            f"  - {len(patients)} patients\n"
            f"  - {total_cases} medical cases\n"
            f"  - {total_scans} chest scans\n"
        ))

    # ---------------------
    def _clear_db(self):
        EnsembleResultModel.objects.all().delete()
        EnsembleResult.objects.all().delete()
        AIAnalysis.objects.all().delete()
        DoctorAnnotation.objects.all().delete()
        ChestScan.objects.all().delete()
        MedicalCase.objects.all().delete()
        ModelVersion.objects.all().delete()
        DoctorProfile.objects.all().delete()
        PatientProfile.objects.all().delete()
        User.objects.all().delete()

    def _create_admin(self):
        return User.objects.create_superuser(
            username="admin",
            email="admin@example.com",
            password="admin",
            role="admin",
        )

    def _create_doctors(self, num):
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
        return doctors

    def _create_patients(self, num):
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
        return patients

    def _create_models(self, admin):
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
        return models

    def _create_cases_and_scans(self, patients, doctors, models):
        case_titles = [
            "Annual Checkup", "Follow-up for persistent cough",
            "Pre-operative assessment", "Post-treatment evaluation",
            "Emergency room visit for dyspnea",
        ]
        statuses = ["open", "monitoring", "closed", "requires_review"]
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

        for patient in patients:
            # Each patient has 2 to MAX_CASES_PER_PATIENT medical cases
            for _ in range(random.randint(2, MAX_CASES_PER_PATIENT)):
                total_cases += 1
                doctor = random.choice(doctors)
                case = MedicalCase.objects.create(
                    patient=patient,
                    primary_doctor=doctor,
                    title=f"{random.choice(case_titles)}",
                    description=fake.paragraph(nb_sentences=3),
                    diagnosis_summary=random.choice(["", fake.paragraph(nb_sentences=2)]), # Some cases have no diagnosis yet
                    status=random.choice(statuses),
                )

                # Each case has 2 to MAX_SCANS_PER_CASE chest scans
                for i in range(random.randint(2, MAX_SCANS_PER_CASE)):
                    total_scans += 1
                    scan = ChestScan.objects.create(case=case, image_path=random.choice(SCANS))

                    # Each scan gets an AI analysis from ACTIVE models
                    active_models = [m for m in models if m.is_active]
                    # models_to_run = random.sample(active_models, k=random.randint(2, len(active_models)))
                    for model in active_models:
                        AIAnalysis.objects.create(
                            scan=scan,
                            model_version=model,
                            prediction_label=random.choice(ai_labels),
                            confidence_score=round(random.uniform(0.65, 0.99), 4),
                            heatmap_path=f"heatmaps/scan_{scan.id}_model_{model.id}.png",
                            heatmap_type="gradcam",
                        )

                    # Each scan is annotated by the primary doctor 2 or MAX_ANNOTATIONS_PER_SCAN times
                    for i in range(random.randint(2, MAX_ANNOTATIONS_PER_SCAN)):
                        DoctorAnnotation.objects.create(
                            scan=scan,
                            doctor=doctor,
                            notes=f"{random.choice(doc_notes)}",
                        )

        return total_cases, total_scans

    def _create_ensembles(self, models):
        """Create ensemble results for scans that already have AIAnalyses."""
        scans = ChestScan.objects.all()
        methods = ["majority_vote", "average", "weighted"]

        for scan in scans:
            analyses = list(scan.ai_analyses.all())
            if not analyses:
                continue

            method = random.choice(methods)
            label = random.choice(["pneumonia", "normal"])
            conf = round(sum(a.confidence_score for a in analyses) / len(analyses), 4)

            ensemble = EnsembleResult.objects.create(
                scan=scan,
                method=method,
                combined_prediction_label=label,
                combined_confidence_score=conf,
            )

            # Link models
            for analysis in analyses:
                EnsembleResultModel.objects.create(
                    ensemble_result=ensemble,
                    model_version=analysis.model_version,
                    weight=round(random.uniform(0.5, 1.5), 2),
                )
