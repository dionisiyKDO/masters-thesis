from rest_framework.permissions import BasePermission, SAFE_METHODS

class IsAdmin(BasePermission):
    """
    Allows access only to users with the 'admin' role.
    """
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.role == 'admin'

class IsAdminOrReadOnly(BasePermission):
    """
    Allows full access to admin users, but read-only access to others.
    """
    def has_permission(self, request, view):
        if request.method in SAFE_METHODS:
            return True
        return request.user.is_authenticated and request.user.role == 'admin'

class IsDoctor(BasePermission):
    """
    Allows access only to users with the 'doctor' role.
    """
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.role == 'doctor'

class IsPatientOfCase(BasePermission):
    """
    Allows access only if the user is the patient associated with the medical case.
    """
    def has_object_permission(self, request, view, obj):
        # For a MedicalCase object
        if hasattr(obj, 'patient'):
            return obj.patient == request.user
        # For objects related to a case (like ChestScan)
        if hasattr(obj, 'case'):
            return obj.case.patient == request.user
        return False

class IsDoctorOfCase(BasePermission):
    """
    Allows access only if the user is the primary doctor for the medical case.
    """
    def has_object_permission(self, request, view, obj):
        if hasattr(obj, 'primary_doctor'):
            return obj.primary_doctor == request.user
        if hasattr(obj, 'case'):
            return obj.case.primary_doctor == request.user
        return False
