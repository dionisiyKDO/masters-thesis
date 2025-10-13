from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.db import transaction
from .models import User, DoctorProfile, PatientProfile
from .permissions import IsAdmin, IsDoctor

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "email", "first_name", "last_name", "role", "is_active", "date_joined"]

class DoctorProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = DoctorProfile
        fields = ["id", "user", "license_number", "specialization"]

class PatientProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = PatientProfile
        fields = ["id", "user", "dob", "sex", "medical_record_number"]



class RegisterSerializer(serializers.ModelSerializer):
    doctor_profile = DoctorProfileSerializer(required=False)
    patient_profile = PatientProfileSerializer(required=False)
    
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'first_name', 'last_name','password', 'role', 'doctor_profile', 'patient_profile')
        extra_kwargs = {"password": {"write_only": True}}

    def to_internal_value(self, data):
        # print("Incoming raw data:", data)  # <-- shows exactly what DRF got from request
        return super().to_internal_value(data)

    def validate(self, data):
        """
        Check that profile data is provided for the selected role.
        """
        role = data.get('role')
        if role == 'admin':
            raise serializers.ValidationError({"role": "Admin role cannot be assigned via registration."})
        if role == 'doctor' and 'doctor_profile' not in data:
            raise serializers.ValidationError({"doctor_profile": "This field is required for doctors."})
        if role == 'patient' and 'patient_profile' not in data:
            raise serializers.ValidationError({"patient_profile": "This field is required for patients."})
        return data

    @transaction.atomic
    def create(self, validated_data):
        doctor_data = validated_data.pop('doctor_profile', None)
        patient_data = validated_data.pop('patient_profile', None)
        password = validated_data.pop("password")
        
        user = User(**validated_data)
        user.set_password(password) # explicitly hashing the password
        user.save()

        # Create the corresponding profile based on the role
        if validated_data['role'] == 'doctor' and doctor_data:
            DoctorProfile.objects.create(user=user, **doctor_data)
        elif validated_data['role'] == 'patient' and patient_data:
            PatientProfile.objects.create(user=user, **patient_data)
            
        return user

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        data = super().validate(attrs)          # Get the default tokens (access and refresh)
        data['username'] = self.user.username   # Add username to the response
        data['id'] = self.user.id               # Add user ID to the response
        data["role"] = self.user.role           # Add role to the response
        return data









# Book = ...

# class BookSerializer(serializers.ModelSerializer):
#     """
#     A fully detailed serializer for books.
#     """

#     # Custom field (not in the model, just for additional computed data)
#     title_length = serializers.SerializerMethodField()

#     class Meta:
#         model = Book
#         fields = '__all__'  # Includes all model fields + custom fields

#     # ----- VALIDATION -----
#     def validate_title(self, value):
#         """
#         Custom validation for 'title' field.
#         """
#         if len(value) < 3:
#             raise serializers.ValidationError("Title must be at least 3 characters long.")
#         return value

#     def validate(self, data):
#         """
#         General validation across multiple fields.
#         """
#         if data['title'] == data['author']:
#             raise serializers.ValidationError("Title and Author cannot be the same.")
#         return data

#     # ----- CREATE -----
#     def create(self, validated_data):
#         """
#         Custom logic for creating a book.
#         """
#         book = Book.objects.create(**validated_data)
#         return book

#     # ----- UPDATE -----
#     def update(self, instance, validated_data):
#         """
#         Custom logic for updating a book.
#         """
#         instance.title = validated_data.get('title', instance.title)
#         instance.author = validated_data.get('author', instance.author)
#         instance.published_date = validated_data.get('published_date', instance.published_date)
#         instance.isbn = validated_data.get('isbn', instance.isbn)
#         instance.save()
#         return instance

#     # ----- To Representation -----
#     def to_representation(self, instance):
#         rep = super().to_representation(instance)
#         rep['display'] = f"{instance.title} by {instance.author}"
#         return rep

#     # ----- To Internal Values -----
#     def to_internal_value(self, data):
#         # Let people send {"author_name": "..."} instead of {"author": "..."}
#         if 'author_name' in data:
#             data['author'] = data.pop('author_name')
#         return super().to_internal_value(data)

#     # ----- CUSTOM FIELD -----
#     Every serializer gets self.context, usually injected by the view:
#       ``serializer = BookSerializer(book, context={'request': request})``
#     Essential when your serialization logic depends on the request/user.

#     def get_title_length(self, obj):
#         # access request in serializer
#         if self.context['request'].user.is_staff:
#            return len(obj.title)
#         return None

# validate_<field>(self, value)	            Custom validation for a specific field.
# validate(self, data)	                    Cross-field validation (checks multiple fields).
# create(self, validated_data)	            Custom logic for creating an object.
# update(self, instance, validated_data)	Custom logic for updating an object.
# to_representation(self, instance)	        Customize how data is converted to JSON. (last chance to touch the JSON before sending it)
# to_internal_value(self, data)	            Customize how data is converted from JSON. (inverse of to_representation)