<!-- lib/components/CaseCard.svelte -->
<script lang="ts">
	import type { MedicalCase } from '$lib/types';

	let { medicalCase }: { medicalCase: MedicalCase } = $props();

    const statusColors: Record<string, string> = {
        open: 'bg-secondary text-secondary-foreground',
        monitoring: 'bg-accent text-accent-foreground',
        closed: 'bg-destructive text-destructive-foreground',
    }
</script>

<a href={`/cases/${medicalCase.id}`} class="card">
    <div class="p-5">
        <div class="flex justify-between items-start">
            <h3 class="font-bold text-lg text-card-foreground">{medicalCase.title}</h3>
            <span class="text-xs font-medium px-2.5 py-0.5 rounded-sm {statusColors[medicalCase.status] || 'bg-muted text-muted-foreground'}">
                {medicalCase.status}
            </span>
        </div>
        <p class="text-sm text-muted-foreground mt-2 line-clamp-2">{medicalCase.description}</p>
    </div>
    <div class="px-5 py-3 border-t border-border">
        <p class="text-xs text-muted-foreground">
            <strong>Patient:</strong> {medicalCase.patient.username} | 
            <strong>Doctor:</strong> Dr. {medicalCase.primary_doctor.username}
        </p>
        <p class="text-xs text-muted-foreground mt-1">
            <strong>Created:</strong> {new Date(medicalCase.created_at).toLocaleDateString()}
        </p>
    </div>
</a>
