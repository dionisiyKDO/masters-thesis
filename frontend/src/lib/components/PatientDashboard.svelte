<!-- lib/components/PatientDashboard.svelte -->
<script lang="ts">
	import api from '$lib/api';
    import CaseCard from './CaseCard.svelte';
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

	// Request
	let casesReq: Promise<MedicalCase[] | null> = fetchCases();
</script>

{#await casesReq}
	<h1>Aboba</h1>
{:then cases} 
	{#if cases}

		<div class="space-y-6">
			<h1 class="text-3xl font-bold text-foreground">Your Medical Cases</h1>

			{#if cases.length === 0}
				<p class="text-muted-foreground">You have no medical cases on record.</p>
			{:else}
				<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
					{#each cases as medicalCase (medicalCase.id)}
						<CaseCard {medicalCase} />
					{/each}
				</div>
			{/if}
		</div>

	{/if}
{/await}
