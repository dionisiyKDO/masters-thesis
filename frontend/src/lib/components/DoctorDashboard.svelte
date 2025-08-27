<!-- lib/components/DoctorDashboard.svelte -->
<script lang="ts">
	import api from '$lib/api';
    import CaseCard from './CaseCard.svelte';
    import FormMedicalCase from './FormMedicalCase.svelte';
	import type { MedicalCase } from '$lib/types';

	async function fetchCases(): Promise<MedicalCase[] | null> {
		try {
			const response = await api.get('/cases/');
			if (!response.ok) throw new Error('Failed to fetch medical cases.');
			const data = await response.json();
			return data
		} catch (err) {
			console.log(err);
			return null;
		}
	}

    let modalContainer: HTMLDivElement;

    // Helper functions
    function openModal() {
        modalContainer.classList.remove('hidden');
    }
    function closeModal() {
        modalContainer.classList.add('hidden');
        casesReq = fetchCases();
    }

	// Request
	let casesReq: Promise<MedicalCase[] | null> = fetchCases();
</script>

{#await casesReq}
	<h1>Aboba</h1>
{:then cases} 
	{#if cases}

        <div class="space-y-6">
            <div class="flex justify-between items-center">
                <h1 class="text-3xl font-bold text-foreground">Assigned Patient Cases</h1>
                <button class="button" onclick={openModal}>
                    Create New Case
                </button>
            </div>

            {#if cases.length === 0}
                <p class="text-muted-foreground">You have no cases assigned to you.</p>
            {:else}
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {#each cases as medicalCase (medicalCase.id)}
                        <CaseCard medicalCase={medicalCase} />
                    {/each}
                </div>
            {/if}
        </div>

	{/if}

    <div bind:this={modalContainer} class="hidden">
        <FormMedicalCase {closeModal} />
    </div>
{/await}