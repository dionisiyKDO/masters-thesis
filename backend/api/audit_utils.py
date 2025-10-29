from .models import AuditLog

def create_audit_log(user, action, details=None):
    """
    A helper function to create an AuditLog entry.
    """
    if details is None:
        details = {}
    
    # Ensure the user is an authenticated user instance, or None
    user_instance = user if user.is_authenticated else None
    
    AuditLog.objects.create(
        user=user_instance,
        action=action,
        details=details
    )
