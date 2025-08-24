from django.contrib.auth.models import AbstractUser
from django.db import models

# AbstractUser: username, password, email, first_name, last_name, is_superuser, is_staff, is_active, date_joined, last_login
class User(AbstractUser):
    ROLE_CHOICES = [
        ("admin", "Admin"),
        ("doctor", "Doctor"),
        ("patient", "Patient"),
    ]
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default="patient")
    
    def __str__(self):
        return f"{self.username} ({self.role}): joined - {self.date_joined}"


class DoctorProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="doctor_profile")
    license_number = models.CharField(max_length=100)
    specialization = models.CharField(max_length=255)

    def __str__(self):
        return f"Dr. {self.user.get_full_name()}"


class PatientProfile(models.Model):
    SEX_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O', 'Other'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="patient_profile")
    dob = models.DateField()
    sex = models.CharField(max_length=1, choices=SEX_CHOICES)
    medical_record_number = models.CharField(max_length=100)

    def __str__(self):
        return f"Patient {self.user.get_full_name()}"