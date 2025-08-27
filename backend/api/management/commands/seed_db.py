# backend/api/management/commands/seed_db.py

import random
from datetime import date
from faker import Faker

from django.core.management.base import BaseCommand
from django.db import transaction

from users.models import User, DoctorProfile, PatientProfile
from api.models import MedicalCase, ChestScan, ModelVersion, AIAnalysis, DoctorAnnotation

# --- Configuration ---
# Easily scale the amount of data to generate
NUM_DOCTORS = 2
NUM_PATIENTS = 2
MAX_CASES_PER_PATIENT = 4
MAX_SCANS_PER_CASE = 5
MAX_ANNOTATIONS_PER_SCAN = 3

# Initialize Faker
fake = Faker()

class Command(BaseCommand):
    help = "Seeds the database with a larger, more interconnected set of mock data."

    @transaction.atomic
    def handle(self, *args, **kwargs):
        self.stdout.write("Deleting old data...")
        # Order of deletion is crucial to respect foreign key constraints
        AIAnalysis.objects.all().delete()
        DoctorAnnotation.objects.all().delete()
        ChestScan.objects.all().delete()
        MedicalCase.objects.all().delete()
        ModelVersion.objects.all().delete()
        DoctorProfile.objects.all().delete()
        PatientProfile.objects.all().delete()
        User.objects.all().delete()
        
        self.stdout.write("Creating new data...")

        ## 1. Create Admin User
        admin_user = User.objects.create_superuser(
            username='admin',
            email='admin@example.com',
            password='admin', # Use a simple, predictable password for development
            role='admin'
        )
        
        # ---
        
        ## 2. Create Doctors
        doctors = []
        specializations = ['Radiology', 'Pulmonology', 'Cardiology', 'General Medicine', 'Oncology']
        for _ in range(NUM_DOCTORS):
            first_name = fake.first_name()
            last_name = fake.last_name()
            # Add a random number to username to avoid collisions
            username = f"dr_{first_name.lower()}_{last_name.lower()}{random.randint(1,99)}"
            
            doctor_user = User.objects.create_user(
                username=username,
                password=username,
                role='doctor',
                email=f"dr.{first_name.lower()}.{last_name.lower()}@clinic.com",
                first_name=first_name,
                last_name=last_name
            )
            DoctorProfile.objects.create(
                user=doctor_user,
                license_number=fake.unique.numerify(text='DOC-#######'),
                specialization=random.choice(specializations)
            )
            doctors.append(doctor_user)
        self.stdout.write(f"Created {len(doctors)} doctors.")

        # ---

        ## 3. Create Patients
        patients = []
        for _ in range(NUM_PATIENTS):
            first_name = fake.first_name()
            last_name = fake.last_name()
            username = f"patient_{first_name.lower()}_{last_name.lower()}{random.randint(1,99)}"

            patient_user = User.objects.create_user(
                username=username,
                password=username,
                role='patient',
                email=fake.email(),
                first_name=first_name,
                last_name=last_name
            )
            PatientProfile.objects.create(
                user=patient_user,
                dob=fake.date_of_birth(minimum_age=18, maximum_age=90),
                sex=random.choice(['M', 'F', 'O']), # Added 'Other' for more variety
                medical_record_number=fake.unique.numerify(text='MRN-##########')
            )
            patients.append(patient_user)
        self.stdout.write(f"Created {len(patients)} patients.")

        # ---

        ## 4. Create Multiple AI Model Versions
        model_versions = []
        model_data = [
            {
                "name": "PneumoDetect_v1.0", "uri": "/models/pneumodetect_v1.0.pt",
                "desc": "Initial release of the PneumoDetect model based on EfficientNet-B0.",
                "metrics": {"accuracy": 0.91, "f1_score": 0.88, "precision": 0.89},
                "active": False # An older, inactive model
            },
            {
                "name": "PneumoDetect_v1.2", "uri": "/models/pneumodetect_v1.2.pt",
                "desc": "Improved version with larger training dataset. Reduced false positives.",
                "metrics": {"accuracy": 0.94, "f1_score": 0.91, "precision": 0.92},
                "active": True
            },
            {
                "name": "LungClassifier_v2.1-beta", "uri": "/models/lungclassifier_v2.1b.pt",
                "desc": "New architecture (Vision Transformer) for classifying multiple conditions.",
                "metrics": {"accuracy": 0.95, "f1_score": 0.93, "precision": 0.94},
                "active": True
            }
        ]
        for data in model_data:
            model = ModelVersion.objects.create(
                model_name=data["name"],
                storage_uri=data["uri"],
                description=data["desc"],
                performance_metrics=data["metrics"],
                uploaded_by_admin=admin_user,
                is_active=data["active"]
            )
            model_versions.append(model)
        self.stdout.write(f"Created {len(model_versions)} AI model versions.")

        # ---

        ## 5. Create Nested Data Points
        # Pre-defined lists for more realistic and varied data
        case_titles = ["Annual Checkup", "Follow-up for persistent cough", "Pre-operative assessment", "Post-treatment evaluation", "Emergency room visit for dyspnea"]
        case_statuses = ['open', 'monitoring', 'closed', 'requires_review']
        ai_labels = ['pneumonia', 'normal', 'covid', 'tuberculosis', 'atelectasis', 'other']
        doc_notes = [
            "Findings are consistent with the AI analysis.", "No acute abnormalities seen.",
            "Slight opacity in the lower lobe, requires monitoring.", "Signs of minor inflammation present.",
            "Recommend follow-up CT for confirmation.", "Patient's condition appears stable.",
            "AI heatmap correctly identifies the region of interest.", "Annotation disagrees with AI. Manual review needed."
        ]
        total_cases, total_scans = 0, 0

        for patient in patients:
            # Each patient has 1 to MAX_CASES_PER_PATIENT medical cases
            for _ in range(random.randint(2, MAX_CASES_PER_PATIENT)):
                total_cases += 1
                primary_doctor = random.choice(doctors)
                
                case = MedicalCase.objects.create(
                    patient=patient,
                    primary_doctor=primary_doctor,
                    title=f"{random.choice(case_titles)} - {patient.get_full_name()}",
                    description=fake.paragraph(nb_sentences=3),
                    status=random.choice(case_statuses)
                )

                # Each case has 1 to MAX_SCANS_PER_CASE chest scans
                for i in range(random.randint(2, MAX_SCANS_PER_CASE)):
                    total_scans += 1
                    scan = ChestScan.objects.create(
                        case=case,
                        image_path=f"scans/patient_{patient.id}/case_{case.id}_scan_{i+1}.dcm"
                    )

                    # Each scan gets an AI analysis from one or more ACTIVE models
                    active_models = [m for m in model_versions if m.is_active]
                    models_to_run = random.sample(active_models, k=random.randint(2, len(active_models)))
                    for model in models_to_run:
                        AIAnalysis.objects.create(
                            scan=scan,
                            model_version=model,
                            prediction_label=random.choice(ai_labels),
                            confidence_score=round(random.uniform(0.65, 0.99), 4),
                            heatmap_path=f"heatmaps/scan_{scan.id}_model_{model.id}.png",
                            heatmap_type='gradcam++'
                        )

                    # Each scan is annotated by the primary doctor and potentially others for second opinions
                    annotating_doctors = {primary_doctor} # Use a set to avoid duplicates
                    other_doctors = [d for d in doctors if d != primary_doctor]
                    
                    num_other_annotators = random.randint(1, min(len(other_doctors), MAX_ANNOTATIONS_PER_SCAN - 1))
                    if num_other_annotators > 0:
                        annotating_doctors.update(random.sample(other_doctors, k=num_other_annotators))

                    for doctor in annotating_doctors:
                        DoctorAnnotation.objects.create(
                            scan=scan,
                            doctor=doctor,
                            notes=f"Review by Dr. {doctor.last_name}: " + random.choice(doc_notes),
                        )
        
        self.stdout.write(self.style.SUCCESS(
            f"✅ Database seeded successfully!\n"
            f"  - {NUM_DOCTORS} doctors\n"
            f"  - {NUM_PATIENTS} patients\n"
            f"  - {total_cases} medical cases\n"
            f"  - {total_scans} chest scans"
        ))