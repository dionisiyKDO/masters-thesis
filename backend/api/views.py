from rest_framework.viewsets import ViewSet
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, action
from .permissions import IsDoctor, IsPatient, IsAdmin

@api_view(["GET"])
def hello(request):
    return Response({"message": "Hello from Django"})


class DoctorDashboardView(ViewSet):
    permission_classes = [IsDoctor]

    @action(detail=False, methods=["get"])
    def dashboard(self, request):
        return Response({"message": "Welcome Doctor!"})

class PatientDashboardView(ViewSet):
    permission_classes = [IsPatient]

    @action(detail=False, methods=["get"])
    def dashboard(self, request):
        return Response({"message": "Welcome Patient!"})

class AdminDashboardView(ViewSet):
    permission_classes = [IsAdmin]

    @action(detail=False, methods=["get"])
    def dashboard(self, request):
        return Response({"message": "Welcome Admin!"})
