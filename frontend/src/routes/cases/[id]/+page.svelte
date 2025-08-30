<script lang="ts">
	import api from '$lib/api';
    import { user } from '$lib/auth.js';
    import ScanDetail from '$lib/components/ScanDetail.svelte';
    import ScanForm from '$lib/components/ScanForm.svelte';
	import type { MedicalCaseDetail, ChestScan } from '$lib/types';

	let { data } = $props();
    const caseId = data.caseId;
	let selectedScan: ChestScan | null = $state(null);
    let refetchTrigger = $state(0); // Simple trigger to force refetch
    $inspect(selectedScan)

	async function fetchCaseDetail(): Promise<MedicalCaseDetail | null> {
		try {
            const response = await api.get(`/cases/${caseId}/`);
			if (!response.ok) throw new Error('Failed to fetch case details.');
            const data: MedicalCaseDetail = await response.json();
            // Auto-select logic
            if (!selectedScan && data.scans.length > 0) {
                selectedScan = data.scans[0];
            }
			return data
		} catch (err) {
			console.log(err);
			return null;
		}
	}

    // Helper function
    function formatDate(date: string) {
        const formatedDate = new Date(date);
        return formatedDate.toLocaleDateString() + '\n' + formatedDate.toLocaleTimeString();
    }


    // Reactive request that refetches when trigger changes
    $effect(() => {
        refetchTrigger; // Subscribe to trigger
        caseDetailReq = fetchCaseDetail();
    });

    // Initial request
    let caseDetailReq: Promise<MedicalCaseDetail | null> = $state(fetchCaseDetail());

    // Refetch function to call from child components
    function refetchCase() {
        refetchTrigger++; // This will trigger the $effect above
        selectedScan = null; // Reset selection to show the new scan
    }
</script>

{#await caseDetailReq}
	<p class="text-muted-foreground">Loading case details...</p>
{:then caseDetail} 
	{#if caseDetail}
        <div class="space-y-6">

            <!-- Top overview -->
            <div class="p-6 bg-card rounded-lg shadow-md border border-border mb-6">
                <a href="/" class="text-primary hover:underline mb-4 inline-block">&larr; Back to Dashboard</a>
                <h1 class="text-3xl font-bold text-card-foreground">{caseDetail.title}</h1>
                <p class="text-sm text-muted-foreground">
                    Patient: {caseDetail.patient.username} | Doctor: Dr. {caseDetail.primary_doctor.username}
                </p>
                <p class="mt-4 text-foreground">{caseDetail.description}</p>

                {#if caseDetail.diagnosis_summary}
                    <div class="mt-4 p-3 bg-secondary border border-border rounded-md">
                        <h2 class="font-semibold text-secondary-foreground">Diagnosis Summary</h2>
                        <p class="text-secondary-foreground">{caseDetail.diagnosis_summary}</p>
                    </div>
                {/if}
            </div>

            <div class="flex gap-6">
                <!-- Sidebar -->
                <div class="w-1/6 max-w-sm bg-card rounded-lg p-4 overflow-y-auto space-y-3">
                    <h2 class="text-xl font-semibold text-foreground mb-3">Scans</h2>
                    <!-- New Scan button -->
                    {#if $user?.role === "doctor"}
                        <div class="pb-3 mb-3 border-b-2 border-border">
                            <button
                                class="cursor-pointer p-3 border rounded-lg bg-primary text-primary-foreground w-full hover:bg-primary/85 transition-all"
                                onclick={() => selectedScan = null}
                            >
                                + New Scan
                            </button>
                        </div>
                    {/if}

                    
                    <!-- Each scan -->
                    {#if caseDetail.scans.length > 0}
                        {#each caseDetail.scans as scan (scan.id)}
                            <button
                                class="cursor-pointer p-3 border-2 rounded-lg hover:bg-secondary/85 hover:text-secondary-foreground transition
                                    {selectedScan?.id === scan.id ? 'bg-secondary text-secondary-foreground' : 'bg-muted text-card-foreground'}"
                                onclick={() => selectedScan = scan}
                            >
                                <p class="font-medium">{formatDate(scan.uploaded_at)}</p>
                            </button>
                        {/each}
                    {:else}
                        <p class="text-muted-foreground">No scans uploaded yet.</p>
                    {/if}
                </div>

                <!-- Scan Detail -->
                <div class="flex-1">
                    {#if selectedScan}
                        <ScanDetail scan={selectedScan} />
                    {:else if $user?.role === "doctor"}
                        <ScanForm caseId={caseId} onUploaded={refetchCase} />
                    {:else}
                        <p class="text-muted-foreground">Select a scan to view details.</p>
                    {/if}

                </div>
            </div>

        </div>
	{/if}
{/await}




