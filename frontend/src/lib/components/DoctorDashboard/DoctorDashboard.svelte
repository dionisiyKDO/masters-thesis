<!-- lib/components/DoctorDashboard.svelte -->
<script lang="ts">
	import api from '$lib/api';
	import CaseCard from '../CaseCard.svelte';
	import FormMedicalCase from './FormMedicalCaseCreate.svelte';
	import PatientSearch from './PatientSearch.svelte';
	import type { MedicalCase, Patient } from '$lib/types';

	// Tab management
    type Tab = 'assigned' | 'patients';
	let activeTab: Tab = $state('assigned');
	
    interface Page {  key: Tab; label: string; }
    const pages: Page[] = [
        {key: 'assigned', label: "Assigned Cases" },
        {key: 'patients', label: "By patient" },
    ]

	let assignedCases: MedicalCase[] = $state([]);

	let patients: Patient[] = $state([]);
	let selectedPatient: Patient | null = $state(null);
	let patientCases: MedicalCase[] = $state([]);

	$inspect(assignedCases);
	$inspect(patientCases);
	$inspect(patients);
	$inspect(selectedPatient);
	
	// Modal management
	let modalContainer: HTMLDivElement;

	// Group cases by status
	function groupCasesByStatus(cases: MedicalCase[]) {
		const open = cases.filter(c => c.status === 'open');
		const closed = cases.filter(c => c.status === 'closed');
		const archived = cases.filter(c => c.status === 'archived');
		return { open, closed, archived };
	}

	async function fetchCases(): Promise<MedicalCase[] | null> {
		try {
			const response = await api.get('/cases/');
			if (!response.ok) throw new Error('Failed to fetch medical cases.');
			const data = await response.json();
			assignedCases = data;
			return data;
		} catch (err) {
			console.log(err);
			return null;
		}
	}
	async function fetchPatients(): Promise<Patient[] | null> {
		try {
			const response = await api.get('/list/patients/');
			if (!response.ok) throw new Error('Failed to fetch medical cases.');
			const data = await response.json();
			patients = data;
			return data;
		} catch (err) {
			console.log(err);
			return null;
		}
	}
	async function fetchPatientCases(patientId: number): Promise<MedicalCase[] | null> {
		try {
			const response = await api.get(`/cases/patient/${patientId}/`);
			if (!response.ok) throw new Error('Failed to fetch medical cases.');
			const data = await response.json();
			patientCases = data;
			return data;
		} catch (err) {
			console.log(err);
			return null;
		}
	}

	// Helper functions
	function openModal() {
		modalContainer.classList.remove('hidden');
	}
	function closeModal() {
		modalContainer.classList.add('hidden');
		casesReq = fetchCases();
	}
	function switchTab(tab: Tab) {
		activeTab = tab;
	}
	function handlePatientSelect(patient: Patient | null) {
		if (patient === null) {
			return
		}
		selectedPatient = patients.find(p => p.id === patient.id) || null;
		console.log('selectedPatient', selectedPatient);
		
		if (selectedPatient) {
			patientCasesReq = fetchPatientCases(selectedPatient?.user.id);
		} else {
			patientCases = [];
		}
	}

	// Request
	let casesReq: Promise<MedicalCase[] | null> = $state(fetchCases());
	let patientsReq: Promise<Patient[] | null> = $state(fetchPatients());
	let patientCasesReq: Promise<MedicalCase[] | null> | undefined = $state();
</script>

<!-- Tab Navigation -->
<div class="flex items-center gap-2 border-border mb-6 border-b">
	{#each pages as {key, label}}
		<button 
			class="px-4 py-2 text-sm font-medium transition-colors 
				{activeTab === key 
					? 'text-primary border-b-2 border-primary' 
					: 'text-muted-foreground hover:text-foreground'}"
			onclick={() => switchTab(key)} 
		>
			{label}
		</button>
	{/each}
</div>

<!-- Assigned Cases Tab -->
{#if activeTab === 'assigned'}
	{#await casesReq}
		<div class="flex items-center justify-center py-8">
			<div class="text-muted-foreground">Loading cases...</div>
		</div>
	{:then cases}
		{#if cases}
			<div class="space-y-6">
				<div class="flex items-center justify-between">
					<h1 class="text-foreground text-3xl font-bold">Assigned Patient Cases</h1>
					<button class="button" onclick={openModal}> Create New Case </button>
				</div>

				{#if cases.length === 0}
					<p class="text-muted-foreground">You have no cases assigned to you.</p>
				{:else}
					{@const grouped = groupCasesByStatus(cases)}
					<div class="space-y-8">
						{#if grouped.open.length > 0}
							<div class="space-y-4">
								<div class="flex items-center gap-1 border-l-2 pl-2 border-primary">
									<h2 class="text-lg font-semibold text-foreground">Open Cases</h2>
									<span class="text-lg text-muted-foreground">({grouped.open.length})</span>
								</div>
								<div class="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
									{#each grouped.open as medicalCase (medicalCase.id)}
										<CaseCard {medicalCase} />
									{/each}
								</div>
							</div>
						{/if}

						{#if grouped.closed.length > 0}
							<div class="space-y-4">
								<div class="flex items-center gap-1 border-l-2 pl-2 border-destructive">
									<h2 class="text-lg font-semibold text-foreground">Closed Cases</h2>
									<span class="text-lg text-muted-foreground">({grouped.closed.length})</span>
								</div>
								<div class="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
									{#each grouped.closed as medicalCase (medicalCase.id)}
										<CaseCard {medicalCase} />
									{/each}
								</div>
							</div>
						{/if}

						{#if grouped.archived.length > 0}
							<div class="space-y-4">
								<div class="flex items-center gap-1 border-l-2 pl-2 border-muted">
									<h2 class="text-lg font-semibold text-foreground">Archived Cases</h2>
									<span class="text-lg text-muted-foreground">({grouped.archived.length})</span>
								</div>
								<div class="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
									{#each grouped.archived as medicalCase (medicalCase.id)}
										<CaseCard {medicalCase} />
									{/each}
								</div>
							</div>
						{/if}
					</div>
				{/if}
			</div>
		{/if}

		<div bind:this={modalContainer} class="hidden">
			<FormMedicalCase {closeModal} />
		</div>
	{/await}
{/if}

<!-- Patient History Tab -->
{#if activeTab === 'patients'}
	<div class="space-y-6">
		<div class="flex items-center justify-between gap-4">
			<div class="flex gap-6">
				<h1 class="text-foreground text-3xl font-bold">Patient Case History</h1>
				<PatientSearch {patients} onselect={handlePatientSelect} />
			</div>
			<button class="button" onclick={openModal}> Create New Case </button>
		</div>

		{#if selectedPatient}
			{#await patientCasesReq}
				<div class="flex items-center justify-center py-8">
					<div class="border-primary h-8 w-8 animate-spin rounded-full border-b-2"></div>
				</div>
			{:then cases}
				{#if cases}
					{#if cases.length === 0}
						<p class="text-muted-foreground">No cases found for this patient.</p>
					{:else}
						{@const grouped = groupCasesByStatus(cases)}
						<div class="space-y-8">
							{#if grouped.open.length > 0}
								<div class="space-y-4">
									<div class="flex items-center gap-1 border-l-2 pl-2 border-primary">
										<h2 class="text-lg font-semibold text-foreground">Open Cases</h2>
										<span class="text-lg text-muted-foreground">({grouped.open.length})</span>
									</div>
									<div class="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
										{#each grouped.open as medicalCase (medicalCase.id)}
											<CaseCard {medicalCase} />
										{/each}
									</div>
								</div>
							{/if}

							{#if grouped.closed.length > 0}
								<div class="space-y-4">
									<div class="flex items-center gap-1 border-l-2 pl-2 border-destructive">
										<h2 class="text-lg font-semibold text-foreground">Closed Cases</h2>
										<span class="text-lg text-muted-foreground">({grouped.closed.length})</span>
									</div>
									<div class="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
										{#each grouped.closed as medicalCase (medicalCase.id)}
											<CaseCard {medicalCase} />
										{/each}
									</div>
								</div>
							{/if}

							{#if grouped.archived.length > 0}
								<div class="space-y-4">
									<div class="flex items-center gap-1 border-l-2 pl-2 border-muted">
										<h2 class="text-lg font-semibold text-foreground">Archived Cases</h2>
										<span class="text-lg text-muted-foreground">({grouped.archived.length})</span>
									</div>
									<div class="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
										{#each grouped.archived as medicalCase (medicalCase.id)}
											<CaseCard {medicalCase} />
										{/each}
									</div>
								</div>
							{/if}
						</div>
					{/if}
				{/if}
			{/await}
		{:else}
			<p class="text-muted-foreground">Please select a patient to view their case history.</p>
		{/if}
	</div>
{/if}