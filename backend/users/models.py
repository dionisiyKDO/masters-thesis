from django.contrib.auth.models import AbstractUser
from django.db import models

# AbstractUser: username, password, email, first_name, last_name, is_superuser, is_staff, is_active, date_joined, last_login
class User(AbstractUser):
    ROLE_CHOICES = [
        ("doctor", "Doctor"),
        ("patient", "Patient"),
        ("admin", "Admin"),
    ]
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default="patient")
    
    def __str__(self):
        return f"{self.username} ({self.role}): joined - {self.date_joined}"
