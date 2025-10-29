from .audit_utils import create_audit_log

class AuditLoggingMixin:
    """
    A generic mixin to automatically log Create, Update, and Delete actions
    for a ModelViewSet.
    
    Requires the ViewSet to define:
    - `audit_log_model_name`: (str) e.g., "USER" or "MEDICAL_CASE"
    
    Optionally, the ViewSet can define:
    - `audit_log_name_field`: (str) e.g., "model_name" or "title"
      If provided, the mixin will try to get this attribute from the
      instance to use as a human-readable name in the log details.
    """
    
    # Default model name, MUST be overridden in the ViewSet
    audit_log_model_name = "OBJECT"
    
    def _get_generic_log_details(self, actor, instance, serializer=None):
        """Helper to build a generic details dict."""
        details = {"actor_id": actor.id if actor.is_authenticated else None}
        
        if instance:
            details["object_id"] = instance.id
            
            # Check if the ViewSet defined a specific name field
            name_field = getattr(self, 'audit_log_name_field', None)
            if name_field and hasattr(instance, name_field):
                details["object_name"] = str(getattr(instance, name_field))
            else:
                # Fallback to just the string representation or ID
                # details["object_name"] = f"ID: {instance.id}"
                pass
        
        if serializer and hasattr(serializer, 'validated_data'):
             # Log the fields that were sent for creation/update
             # Convert complex data types (like datetimes) to strings
             try:
                 details["changes_sent"] = {
                     key: str(value) for key, value in serializer.validated_data.items()
                 }
             except Exception:
                 details["changes_sent"] = "Error serializing changes"
             
        return details

    def perform_create(self, serializer):
        """
        Log object creation.
        """
        instance = serializer.save()
        actor = self.request.user
        
        action = f"CREATED_{self.audit_log_model_name}"
        
        details = self._get_generic_log_details(actor, instance, serializer)
        create_audit_log(actor, action, details)

    def perform_update(self, serializer):
        """
        Log object update.
        """
        instance = serializer.save()
        actor = self.request.user
        
        action = f"UPDATED_{self.audit_log_model_name}"
        
        details = self._get_generic_log_details(actor, instance, serializer)
        create_audit_log(actor, action, details)

    def perform_destroy(self, instance):
        """
        Log object deletion.
        """
        actor = self.request.user
        
        # Capture details *before* deleting
        action = f"DELETED_{self.audit_log_model_name}"
        details = self._get_generic_log_details(actor, instance)
        
        # Perform the deletion
        instance.delete()
        
        # Log the action
        create_audit_log(actor, action, details)
