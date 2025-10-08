<!-- lib/components/DoctorDashboard.svelte -->
<script lang="ts">
	import api from '$lib/api';
	import CaseCard from '../PatientDashboard/CaseCard.svelte';
	import FormMedicalCase from './FormMedicalCaseCreate.svelte';
	import type { MedicalCase, Patient } from '$lib/types';
	import type { installPolyfills } from '@sveltejs/kit/node/polyfills';

	// Tab management
	let activeTab: 'assigned' | 'patients' | 'analytics' = $state('patients');

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
	function switchTab(tab: 'assigned' | 'patients' | 'analytics') {
		activeTab = tab;
	}
	function handlePatientSelect(event: Event) {
		const target = event.target as HTMLSelectElement;
		const patientId = parseInt(target.value);
		selectedPatient = patients.find(p => p.id === patientId) || null;
		
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
<div class="border-border mb-6 border-b">
	<nav class="flex space-x-8">
		<button
			class="border-b-2 px-1 py-4 text-sm font-medium {activeTab === 'patients'
				? 'border-primary text-primary'
				: 'text-muted-foreground hover:text-foreground border-transparent'}"
			onclick={() => switchTab('patients')}
		>
			Patient History
		</button>
		<button
			class="border-b-2 px-1 py-4 text-sm font-medium {activeTab === 'assigned'
				? 'border-primary text-primary'
				: 'text-muted-foreground hover:text-foreground border-transparent'}"
			onclick={() => switchTab('assigned')}
		>
			Assigned Cases
		</button>
	</nav>
</div>

<!-- Patient History Tab -->
{#if activeTab === 'patients'}
	<div class="space-y-4">
		<div class="flex items-center gap-4">
			<h1 class="text-foreground text-2xl font-bold">Patient Case History</h1>
			{#await patientsReq then patients}
				{#if patients}
					<select
						class="border-border bg-background text-foreground rounded-md border px-3 py-2"
						onchange={handlePatientSelect}
					>
						<option value="">Select a patient...</option>
						{#each patients as patient}
							<option value={patient.id}>
								{patient.user.first_name}
								{patient.user.last_name}
							</option>
						{/each}
					</select>
				{/if}
			{/await}
		</div>

		{#if selectedPatient}
			{#await patientCasesReq}
				<div class="flex items-center justify-center py-8">
					<div class="border-primary h-8 w-8 animate-spin rounded-full border-b-2"></div>
				</div>
			{:then patientCases}
				{#if patientCases}
					{#if patientCases.length === 0}
						<p class="text-muted-foreground">No cases found for this patient.</p>
					{:else}
						<div class="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
							{#each patientCases as medicalCase (medicalCase.id)}
								<CaseCard {medicalCase} />
							{/each}
						</div>
					{/if}
				{/if}
			{/await}
		{:else}
			<p class="text-muted-foreground">Please select a patient to view their case history.</p>
		{/if}
	</div>
{/if}

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
					<div class="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
						{#each cases as medicalCase (medicalCase.id)}
							<CaseCard {medicalCase} />
						{/each}
					</div>
				{/if}
			</div>
		{/if}

		<div bind:this={modalContainer} class="hidden">
			<FormMedicalCase {closeModal} />
		</div>
	{/await}
{/if}