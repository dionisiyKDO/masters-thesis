# backend/api/management/commands/seed_db.py

import random
from datetime import date
from faker import Faker

from django.core.management.base import BaseCommand
from django.db import transaction

from users.models import User, DoctorProfile, PatientProfile
from api.models import MedicalCase, ChestScan, ModelVersion, AIAnalysis, DoctorAnnotation

# Initialize Faker
fake = Faker()

class Command(BaseCommand):
    help = "Seeds the database with mock data for development."

    @transaction.atomic
    def handle(self, *args, **kwargs):
        self.stdout.write("Deleting old data...")
        # Delete in an order that respects foreign key constraints
        AIAnalysis.objects.all().delete()
        DoctorAnnotation.objects.all().delete()
        ChestScan.objects.all().delete()
        MedicalCase.objects.all().delete()
        ModelVersion.objects.all().delete()
        DoctorProfile.objects.all().delete()
        PatientProfile.objects.all().delete()
        User.objects.all().delete()
        
        self.stdout.write("Creating new data...")

        # 1. Create an Admin User
        admin_user = User.objects.create_superuser(
            username='admin',
            email='admin@example.com',
            password='pass123',
            role='admin'
        )

        # 2. Create Doctors
        doctors = []
        for _ in range(2):
            first_name = fake.first_name()
            last_name = fake.last_name()
            doctor_user = User.objects.create_user(
                username=f"dr_{last_name.lower()}",
                password='pass123',
                role='doctor',
                email=f"dr.{last_name.lower()}@clinic.com",
                first_name=first_name,
                last_name=last_name
            )
            DoctorProfile.objects.create(
                user=doctor_user,
                license_number=fake.unique.numerify(text='DOC-######'),
                specialization=random.choice(['Radiology', 'Pulmonology', 'General Medicine'])
            )
            doctors.append(doctor_user)

        # 3. Create Patients
        patients = []
        for _ in range(5):
            first_name = fake.first_name()
            last_name = fake.last_name()
            patient_user = User.objects.create_user(
                username=f"patient_{last_name.lower()}",
                password='pass123',
                role='patient',
                email=fake.email(),
                first_name=first_name,
                last_name=last_name
            )
            PatientProfile.objects.create(
                user=patient_user,
                dob=fake.date_of_birth(minimum_age=18, maximum_age=90),
                sex=random.choice(['M', 'F']),
                medical_record_number=fake.unique.numerify(text='MRN-##########')
            )
            patients.append(patient_user)

        # 4. Create an AI Model Version
        model_v1 = ModelVersion.objects.create(
            model_name="ResNet50_Pneumonia_v1.2",
            storage_uri="/models/resnet50_v1.2.pt",
            description="Second iteration of the ResNet50 classifier.",
            performance_metrics={"accuracy": 0.94, "f1_score": 0.91},
            uploaded_by_admin=admin_user,
            is_active=True
        )

        # 5. Create Medical Cases, Scans, and Analyses for each Patient
        for patient in patients:
            case = MedicalCase.objects.create(
                patient=patient,
                primary_doctor=random.choice(doctors),
                title=f"Routine checkup for {patient.first_name}",
                description=fake.sentence(),
                status=random.choice(['open', 'monitoring'])
            )

            # Create 1 to 3 scans per case
            for i in range(random.randint(1, 3)):
                scan = ChestScan.objects.create(
                    case=case,
                    image_path=f"scans/fake_scan_{patient.id}_{i+1}.png"
                )

                # Create an AI Analysis for the scan
                AIAnalysis.objects.create(
                    scan=scan,
                    model_version=model_v1,
                    prediction_label=random.choice(['pneumonia', 'normal']),
                    confidence_score=round(random.uniform(0.75, 0.99), 2),
                    heatmap_path=f"heatmaps/fake_heatmap_{scan.id}.png",
                    heatmap_type='gradcam'
                )

                # Create a Doctor Annotation for the scan
                DoctorAnnotation.objects.create(
                    scan=scan,
                    doctor=case.primary_doctor,
                    notes=f"Initial review of scan {i+1}. " + random.choice([
                        "Findings are consistent with the AI analysis.",
                        "No acute abnormalities seen.",
                        "Slight opacity in the lower lobe, requires monitoring."
                    ])
                )

        self.stdout.write(self.style.SUCCESS(f"✅ Database seeded with {len(doctors)} doctors and {len(patients)} patients."))