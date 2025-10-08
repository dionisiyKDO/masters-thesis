<!-- lib/components/DoctorDashboard/PatientSearch.svelte -->
<script lang="ts">
	import type { Patient } from '$lib/types';

	interface Props {
		patients: Patient[];
		onselect: (patient: Patient | null) => void;
	}

	let { patients, onselect }: Props = $props();

	let searchQuery = $state('');
	let isOpen = $state(false);
	let selectedPatient: Patient | null = $state(null);
	let dropdownContainer: HTMLDivElement;

	// Fuzzy search function
	function fuzzyMatch(str: string, pattern: string): number {
		const patternLower = pattern.toLowerCase();
		const strLower = str.toLowerCase();
		
		// Exact match gets highest score
		if (strLower.includes(patternLower)) {
			return 100;
		}

		let score = 0;
		let patternIdx = 0;
		
		for (let i = 0; i < strLower.length && patternIdx < patternLower.length; i++) {
			if (strLower[i] === patternLower[patternIdx]) {
				score += 1;
				patternIdx += 1;
			}
		}
		
		// Return score based on how many characters matched
		return patternIdx === patternLower.length ? score : 0;
	}

	let filteredPatients = $derived.by(() => {
		if (!searchQuery.trim()) {
			return patients;
		}

		return patients
			.map(patient => {
				const fullName = `${patient.user.first_name} ${patient.user.last_name}`;
				const score = fuzzyMatch(fullName, searchQuery);
				return { patient, score };
			})
			.filter(({ score }) => score > 0)
			.sort((a, b) => b.score - a.score)
			.map(({ patient }) => patient);
	});

	function handleInputFocus() {
		isOpen = true;
	}

	function handlePatientClick(patient: Patient) {
		selectedPatient = patient;
		searchQuery = `${patient.user.first_name} ${patient.user.last_name}`;
		isOpen = false;
		onselect(patient);
	}

	function handleInputChange(event: Event) {
		const target = event.target as HTMLInputElement;
		searchQuery = target.value;
		isOpen = true;
		
		if (!searchQuery.trim()) {
			selectedPatient = null;
			onselect(null);
		}
	}

	function handleClickOutside(event: MouseEvent) {
		if (dropdownContainer && !dropdownContainer.contains(event.target as Node)) {
			isOpen = false;
		}
	}

	$effect(() => {
		document.addEventListener('click', handleClickOutside);
		return () => {
			document.removeEventListener('click', handleClickOutside);
		};
	});
</script>

<div bind:this={dropdownContainer} class="relative w-64">
	<input
		type="text"
		class="border-border bg-background text-foreground w-full rounded-md border px-3 py-2 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
		placeholder="Search for a patient..."
		value={searchQuery}
		oninput={handleInputChange}
		onfocus={handleInputFocus}
	/>
	
	{#if isOpen && filteredPatients.length > 0}
		<div class="bg-background border-border absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-md border shadow-lg">
			{#each filteredPatients as patient (patient.id)}
				<button
					type="button"
					class="text-foreground hover:bg-muted w-full px-3 py-2 text-left transition-colors"
					onclick={() => handlePatientClick(patient)}
				>
					{patient.user.first_name} {patient.user.last_name}
				</button>
			{/each}
		</div>
	{:else if isOpen && searchQuery.trim() && filteredPatients.length === 0}
		<div class="bg-background border-border absolute z-10 mt-1 w-full rounded-md border shadow-lg">
			<div class="text-muted-foreground px-3 py-2">No patients found</div>
		</div>
	{/if}
</div>