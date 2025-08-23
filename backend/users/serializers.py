from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', "role")

class RegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password', "role")
        extra_kwargs = {"password": {"write_only": True}}

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password'],
            role=validated_data.get("role", "patient"),  # default patient
            is_staff=(True if validated_data.get("role") == "admin" else False)  # admin -> is_staff
        )
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