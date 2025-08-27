<script lang="ts">
    import api from '$lib/api';
    import type { LocalUser, MedicalCase, SimpleUser } from '$lib/types';
    import { user } from '$lib/auth';
    
    interface Props {
        closeModal: () => void;
    }
    let { closeModal } = $props();

    let doctor = user;
	let patients: SimpleUser[] | null = $state(null);
    let patientsReq: Promise<SimpleUser[] | null> = fetchPatients();

    async function fetchPatients(): Promise<SimpleUser[] | null> {
		try {
			const response = await api.get('/auth/patients/')
			if (!response.ok) throw new Error('Failed to fetch patients.');
			const data = await response.json();            
            patients = data
			return data
		} catch (err) {
			console.log(err);
			return null;
		}
    }

    async function handleSubmit(event: Event) {
        event.preventDefault();
        const form = event.target as HTMLFormElement;
        const formData = new FormData(form);
        
        const newCase = {
            title: formData.get('title') as string,
            description: formData.get('description') as string,
            status: 'open',
            primary_doctor_id: parseInt(formData.get('primary_doctor_id') as string) || null,
            patient_id: parseInt(formData.get('patient_id') as string) || null,
        };
        console.log(newCase);
        
		try {
			const response = await api.post('/cases/', newCase);
			if (!response.ok) throw new Error('Failed to create medical case.');
			const data = await response.json();
            closeModal();
			return data
		} catch (err) {
			console.log(err);
			return null;
		}
    }
</script>

{#await patientsReq}
    asd
{:then patients} 
    <div class="fixed inset-0 bg-black/50 z-40">
        <div class="bg-card p-6 rounded-lg border border-border space-y-6 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-2/3 md:w-1/2 lg:w-2/5 z-50">
            <h2 class="text-2xl font-bold text-foreground">Create New Medical Case</h2>
            <form class="space-y-4" onsubmit={handleSubmit}>
                <div>
                    <label for="title" class="block text-sm font-medium text-foreground">Title</label>
                    <input type="text" id="title" name="title" class="mt-1 block w-full rounded-md border border-border bg-background px-3 py-2 text-foreground shadow-sm focus:border-primary focus:ring focus:ring-primary/50" required />
                </div>
                <div>
                    <label for="description" class="block text-sm font-medium text-foreground">Description</label>
                    <textarea id="description" name="description" rows="4" class="mt-1 block w-full rounded-md border border-border bg-background px-3 py-2 text-foreground shadow-sm focus:border-primary focus:ring focus:ring-primary/50" required></textarea>
                </div>
                <div>
                    <label for="primary_doctor" class="block text-sm font-medium text-foreground">Primary Doctor</label>
                    <input type="text" id="primary_doctor" name="primary_doctor" class="mt-1 block w-full rounded-md border border-border bg-background px-3 py-2 text-muted-foreground shadow-sm focus:border-primary focus:ring focus:ring-primary/50" value="{$doctor?.username}" disabled />
                    <input type="hidden" name="primary_doctor_id" value="{$doctor?.id}" />
                </div>
                <div>
                    <label for="patient_id" class="block text-sm font-medium text-foreground">Patient ID (optional)</label>
                    <!-- Dropdown list of patients -->
                    <select id="patient_id" name="patient_id" class="mt-1 block w-full rounded-md border border-border bg-background px-3 py-2 text-foreground shadow-sm focus:border-primary focus:ring focus:ring-primary/50">
                        <option value="" selected>-- Select a patient --</option>
                        {#if patients}
                            {#each patients as patient}
                                <option value={patient.id}>{patient.username}</option>
                            {/each}
                        {/if}
                    </select>
                </div>
                <div class="flex justify-end space-x-2">
                    <button type="button" class="button button-secondary" onclick={closeModal}>Cancel</button>
                    <button type="submit" class="button button-primary">Create Case</button>
                </div>
            </form>
        </div>
    </div>
{/await}


