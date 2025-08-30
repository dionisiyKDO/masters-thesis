<script lang="ts">
    import api from '$lib/api';
    import type { LocalUser, MedicalCase, SimpleUser as Patients } from '$lib/types';
    import { user } from '$lib/auth';
    
    interface Props { closeModal: () => void; }
    let { closeModal }: Props = $props();

    let doctor = user;
	let patients: Patients[] | null = $state(null);
    let patientsReq: Promise<Patients[] | null> = fetchPatients();
    
    // Search functionality state
    let searchQuery = $state('');
    let selectedPatient: Patients | null = $state(null);
    let isDropdownOpen = $state(false);
    let filteredPatients = $state<Patients[]>([]);

    // Update filtered patients when search query or patients change
    $effect(() => {
        if (patients && searchQuery) {
            filteredPatients = patients.filter(patient => {
                const fullName = `${patient.first_name} ${patient.last_name}`.toLowerCase();
                const searchLower = searchQuery.toLowerCase();
                return fullName.includes(searchLower) ||
                       patient.first_name.toLowerCase().includes(searchLower) ||
                       patient.last_name.toLowerCase().includes(searchLower);
            });
        } else {
            filteredPatients = patients || [];
        }
    });

    async function fetchPatients(): Promise<Patients[] | null> {
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
            patient_id: selectedPatient?.id || null,
        };
        
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

    // Search helpers
    //#region 
    function selectPatient(patient: Patients) {
        selectedPatient = patient;
        searchQuery = `${patient.first_name} ${patient.last_name}`;
        isDropdownOpen = false;
    }

    function clearSelection() {
        selectedPatient = null;
        searchQuery = '';
        isDropdownOpen = false;
    }

    function handleSearchInput(event: Event) {
        const target = event.target as HTMLInputElement;
        searchQuery = target.value;
        
        if (searchQuery && patients) {
            isDropdownOpen = true;
        } else {
            isDropdownOpen = false;
        }
        
        // Clear selection if search doesn't match selected patient
        if (selectedPatient) {
            const fullName = `${selectedPatient.first_name} ${selectedPatient.last_name}`.toLowerCase();
            if (!fullName.includes(searchQuery.toLowerCase())) {
                selectedPatient = null;
            }
        }
    }

    function handleInputFocus() {
        if (patients && patients.length > 0) {
            isDropdownOpen = true;
        }
    }

    function handleInputBlur() {
        setTimeout(() => {
            isDropdownOpen = false;
        }, 150);
    }
    //#endregion
</script>

{#await patientsReq}
    asd
{:then patients} 
    <div class="fixed inset-0 bg-black/50 z-40">
        <div class="bg-card p-6 rounded-lg border border-border space-y-6 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-2/3 md:w-1/2 lg:w-2/5 z-50">
            <h2 class="text-2xl font-bold text-foreground">Create New Medical Case</h2>
            <form class="space-y-4" onsubmit={handleSubmit}>
                <div>
                    <label for="title" class="ml-2 block text-sm font-medium text-foreground">Title</label>
                    <input type="text" id="title" name="title" class="input" required />
                </div>
                <div>
                    <label for="description" class="ml-2 block text-sm font-medium text-foreground">Description</label>
                    <textarea id="description" name="description" rows="4" class="input" required></textarea>
                </div>
                <div>
                    <label for="primary_doctor" class="ml-2 block text-sm font-medium text-foreground">Primary Doctor</label>
                    <input type="text" id="primary_doctor" name="primary_doctor" class="input" value="{$doctor?.username}" disabled />
                    <input type="hidden" name="primary_doctor_id" value="{$doctor?.id}" />
                </div>
                <div class="relative">
                    <label for="patient_search" class="ml-2 block text-sm font-medium text-foreground">Patient</label>
                    <div class="relative">
                        <input 
                            type="text" 
                            id="patient_search" 
                            bind:value={searchQuery}
                            oninput={handleSearchInput}
                            onfocus={handleInputFocus}
                            onblur={handleInputBlur}
                            class="input" 
                            placeholder="Search for a patient..."
                        />
                    </div>
                    
                    {#if isDropdownOpen && filteredPatients.length > 0}
                        <div class="absolute z-10 w-full mt-1 bg-background border border-border rounded-md shadow-lg max-h-60 overflow-auto">
                            {#each filteredPatients as patient}
                                <button
                                    type="button"
                                    onclick={() => selectPatient(patient)}
                                    class="w-full text-left px-3 py-2 hover:bg-muted focus:bg-muted focus:outline-none text-foreground"
                                >
                                    {patient.first_name} {patient.last_name}
                                </button>
                            {/each}
                        </div>
                    {:else if isDropdownOpen && searchQuery && filteredPatients.length === 0}
                        <div class="absolute z-10 w-full mt-1 bg-background border border-border rounded-md shadow-lg">
                            <div class="px-3 py-2 text-muted-foreground">
                                No patients found
                            </div>
                        </div>
                    {/if}
                </div>
                <div class="flex justify-end space-x-2">
                    <button type="button" class="button button-secondary" onclick={closeModal}>Cancel</button>
                    <button type="submit" class="button button-primary">Create Case</button>
                </div>
            </form>
        </div>
    </div>
{/await}