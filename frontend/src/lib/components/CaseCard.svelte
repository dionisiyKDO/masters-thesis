<!-- lib/components/CaseCard.svelte -->
<script lang="ts">
	import type { MedicalCase } from '$lib/types';

	let { medicalCase }: { medicalCase: MedicalCase } = $props();
</script>

<a href={`/cases/${medicalCase.id}`} class="card flex flex-col">
    <div class="p-5 flex-1">
        <div class="flex justify-between items-start gap-1">
            <h3 class="font-bold text-lg text-card-foreground">{medicalCase.title}</h3>
            <span class="text-xs font-medium px-2.5 py-0.5 rounded-sm {medicalCase.status === 'closed' ? 'bg-destructive text-destructive-foreground' : medicalCase.status === 'archived' ? 'bg-muted text-muted-foreground' : 'bg-primary text-primary-foreground'}">
                {medicalCase.status}
            </span>
        </div>
        <p class="text-sm text-muted-foreground mt-2 line-clamp-2">{medicalCase.description}</p>
    </div>
    <div class="px-5 py-3 border-t border-border">
        <p class="text-xs text-muted-foreground">
            <strong>Patient:</strong> {medicalCase.patient.first_name} {medicalCase.patient.last_name} | 
            <strong>Doctor:</strong> Dr. {medicalCase.primary_doctor.first_name} {medicalCase.primary_doctor.last_name}
        </p>
        <p class="text-xs text-muted-foreground mt-1">
            <strong>Created:</strong> {new Date(medicalCase.created_at).toLocaleDateString()}
        </p>
    </div>
</a>
