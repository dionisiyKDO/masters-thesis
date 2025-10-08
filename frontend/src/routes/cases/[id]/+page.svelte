<script lang="ts">
	import api from '$lib/api';
    import { user } from '$lib/auth.js';
    import ScanDetail from '$lib/components/ScanDetail.svelte';
    import ScanForm from '$lib/components/DoctorDashboard/FormScanUpload.svelte';
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
            // Sort scans by uploaded_at descending
            data.scans.sort((a, b) => new Date(b.uploaded_at).getTime() - new Date(a.uploaded_at).getTime());
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
        return formatedDate.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });    
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
	<div class="flex items-center justify-center min-h-64">
		<div class="flex items-center gap-3 text-muted-foreground">
			<svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
				<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
				<path class="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
			</svg>
			Loading case details...
		</div>
	</div>
{:then caseDetail} 
	{#if caseDetail}
        <div class="space-y-6">

            <!-- Header Section -->
            <div class="px-6 py-2.5 bg-card rounded-sm shadow-ms border border-border mb-6">
                <!-- Breadcrumb -->
                <div class="pb-2 border-b">
                    <a href="/" class="inline-flex items-center gap-2 text-primary hover:text-primary/80 transition-colors text-sm font-medium">&larr; Back to Dashboard</a>
                </div>
                
                <!-- Main Header Content -->
                <div class="p-2">
                    <div class="flex items-center justify-between">
                        <h1 class="text-2xl font-bold text-card-foreground">{caseDetail.title}</h1>
                        <div class="text-right text-xs text-muted-foreground">Case ID: {caseId}</div>
                    </div>
                    <div class="text-sm text-muted-foreground mb-4">Patient: {caseDetail.patient.first_name} {caseDetail.patient.last_name} | Doctor: {caseDetail.primary_doctor.first_name} {caseDetail.primary_doctor.last_name}</div>
                    
                    <!-- Description -->
                    {#if caseDetail.description}
                        <div>
                            <div class="text-xs text-muted-foreground mb-0.5">Description</div>
                            <p class="text-foreground leading-relaxed">{caseDetail.description}</p>
                        </div>
                    {/if}
                    
                    <!-- Diagnosis -->
                    <!-- {#if caseDetail.diagnosis_summary}
                        <div class="bg-accent/5 border border-accent/20 rounded-lg p-4 mt-4">
                            <div class="flex items-center gap-2 mb-0">
                                <svg class="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                                <h2 class="font-semibold text-muted-foreground">Diagnosis Summary</h2>
                            </div>
                            <p class="text-muted-foreground/90 leading-relaxed">{caseDetail.diagnosis_summary}</p>
                        </div>
                    {/if} -->
                </div>
            </div>

            <!-- Main Content Area -->
            <div class="flex gap-6">
                <!-- Sidebar -->

                <div class="w-80 flex-shrink-0">
                    <div class="bg-card rounded-xl border border-border overflow-hidden sticky top-2">
                        <!-- Header -->
                        <div class="px-5 py-3 border-b border-border">
                            <div class="flex items-center gap-2">
                                <svg class="w-4 h-4 text-card-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                                <h2 class="text-lg font-semibold text-card-foreground">Medical Scans ({caseDetail.scans.length})</h2>
                            </div>
                        </div>
                        
                        <div class="p-4 space-y-3">
                            <!-- New Scan button -->
                            {#if $user?.role === "doctor"}
                                <button
                                    class="w-full flex items-center justify-center gap-2 p-4 bg-primary hover:bg-primary/90 text-primary-foreground rounded-lg border-2 border-dashed border-primary/30 hover:border-primary/50 transition-all duration-200 group"
                                    onclick={() => selectedScan = null}
                                >
                                    <svg class="w-5 h-5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
                                    </svg>
                                    <span class="font-medium">Upload New Scan</span>
                                </button>
                            {/if}

                        
                            <!-- Each scan -->
                            {#if caseDetail.scans.length > 0}
                                <div class="space-y-2 max-h-[800px] overflow-y-auto px-2">
                                    {#each caseDetail.scans as scan, index}
                                        <button
                                            class="w-full p-3 rounded-lg border transition-all duration-200 text-left group
                                                {selectedScan?.id === scan.id 
                                                    ? 'bg-secondary border-secondary text-secondary-foreground shadow-sm' 
                                                    : 'bg-muted/50 border-border text-card-foreground hover:bg-muted hover:border-muted-foreground/30'}"
                                            onclick={() => selectedScan = scan}
                                        >
                                            <div class="flex items-center gap-3">
                                                <div class="min-w-0 flex-1">
                                                    <div class="text-sm font-medium truncate">
                                                        Scan {index + 1}
                                                    </div>
                                                    <div class="text-xs opacity-75 mt-1">
                                                        {formatDate(scan.uploaded_at)}
                                                    </div>
                                                </div>
                                                <svg class="w-4 h-4 opacity-0 group-hover:opacity-50 transition-opacity {selectedScan?.id === scan.id ? 'opacity-100' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                                                </svg>
                                            </div>
                                        </button>
                                    {/each}
                                </div>
                            {:else}
                                <div class="text-center py-4">
                                    <svg class="w-8 h-8 text-muted-foreground mx-auto mb-1 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                    </svg>
                                    <p class="text-sm text-muted-foreground mb-1">No scans uploaded</p>
                                    <p class="text-xs text-muted-foreground/70">Medical imaging will appear here</p>
                                </div>
                            {/if}
                        </div>
                    </div>
                </div>

                <!-- Scan Detail -->
                <div class="flex-1 min-w-0">
                    {#if selectedScan}
                        <ScanDetail scan={selectedScan} />
                    {:else if $user?.role === "doctor"}
                        <ScanForm caseId={caseId} onUploaded={refetchCase} />
                    {:else}
                        <div class="bg-card rounded-xl border border-border p-12 text-center">
                            {#if caseDetail.scans.length > 0}
                                <h3 class="text-lg font-medium text-muted-foreground mb-2">Select a Scan to View</h3>
                                <p class="text-sm text-muted-foreground/70">Choose a medical scan from the sidebar to view details and analysis</p>
                            {:else}
                                <h3 class="text-lg font-medium text-muted-foreground mb-2">There were no uploaded scans yet</h3>
                                <p class="text-sm text-muted-foreground/70">Wait for your doctor to upload medical scans for this case</p>
                            {/if}
                        </div>
                    {/if}

                </div>
            </div>

        </div>
	{/if}
{/await}
