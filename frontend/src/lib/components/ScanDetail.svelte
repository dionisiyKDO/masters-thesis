<script lang="ts">
	import type { ChestScan } from "$lib/types"
    import { user } from '$lib/auth.js';
	import api from "$lib/api";
	import { slide } from "svelte/transition";

	let { scan }: { scan: ChestScan } = $props();
	const API_BASE = 'http://localhost:8000';
	let adding = $state(false);
	let newAnnotation = $state("");
	$inspect(scan)

	async function handleAddAnnotation() {
		try {
			if (!newAnnotation.trim()) return;
			const body = {
				doctor_id: $user?.id,
				scan_id: scan.id,
				notes: newAnnotation,
			};
			const response = await api.post('/annotations/', body);
			if (!response.ok) throw new Error('Failed to create annotation.');
			const data = await response.json();
			scan.annotations = [...scan.annotations, data];
			newAnnotation = "";
			adding = false;
			
			return data
		} catch (err) {
			console.log(err);
			return null;
		}
	}

    // Helper function
    function formatDate(date: string) {
        const formatedDate = new Date(date);
        return formatedDate.toLocaleDateString() + ' at ' + formatedDate.toLocaleTimeString();
    }
</script>

<div class="bg-card p-6 rounded-lg border border-border space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between">
		
		<h3 class="text-xl font-semibold text-foreground">
			Scan from {formatDate(scan.uploaded_at)}
		</h3>
	</div>

	<!-- Main images -->
	<!-- <div class="grid grid-cols-1 md:grid-cols-2 gap-6"> -->
	<div class="">
		<div>
			<!-- <p class="font-medium text-foreground mb-2">Chest X-Ray</p> -->
			<img
				src={`${scan.image_path}`}
				alt="Chest Scan"
				class="rounded-md border border-border w-full min-h-[150px] max-h-[500px] object-contain"
			/>
		</div>

		<!-- {#if scan.ai_analyses.length > 0}
            {#each scan.ai_analyses as analysis}
                <div>
                    <p class="font-medium text-foreground mb-2">AI Heatmap</p>
                    {#if analysis.heatmap_path}
                        <img
                            src={`${API_BASE}${analysis.heatmap_path}`}
                            alt="AI Heatmap"
                            class="rounded-md border border-border w-full min-h-[150px] max-h-[500px] object-contain"
                        />
                    {:else}
                        <div class="flex items-center justify-center h-64 bg-muted rounded-md border border-dashed">
                            <p class="text-muted-foreground">No heatmap available</p>
                        </div>
                    {/if}
                </div>
            {/each}
		{/if} -->
	</div>

	<!-- AI Predictions -->
	{#if scan.ai_analyses && scan.ai_analyses.length > 0}
		<h4 class="font-semibold ml-2 mb-0">AI Analyses</h4>
		<div class="grid grid-cols-2 gap-4">
			{#each scan.ai_analyses as analysis}
				<div class="bg-muted rounded-md px-4 py-2">
					<div class="">					
						<p>
							<strong>Label:</strong>
							<span class="capitalize font-mono px-2 py-0.5 rounded bg-input text-foreground text-sm">{analysis.prediction_label}</span>
						</p>
						<p>
							<strong>Confidence:</strong>
							{(analysis.confidence_score * 100).toFixed(2)}%
						</p>
					</div>
					<div class="mt-1 text-xs text-muted-foreground">
						<strong>Model:</strong> {analysis.model_version.model_name}<br>
						<strong>Date:</strong> {new Date(analysis.generated_at).toLocaleString()}
					</div>
				</div>
			{/each}
		</div>
	{/if}

	<!-- Doctor annotations -->
	{#if $user?.role === "doctor"}
		<div class="flex justify-between mb-2">
			<h4 class="font-semibold ml-2 mb-0">Add annotations</h4>
			<button 
				class="button px-3 py-0.5"
				onclick={() => adding = !adding}
			>
				{adding ? "Cancel" : "Add annotation"}
			</button>
		</div>

		{#if adding}
			<div class="mb-2 p-3 bg-muted/50 rounded-lg border border-dashed" transition:slide={{ duration: 300 }}>
				<textarea
					bind:value={newAnnotation}
					placeholder="Enter your medical notes and observations..."
					rows="3"
					class="w-full p-3 border border-border rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary bg-background"
				></textarea>
				<div class="flex justify-end gap-2 mt-2">
					<button
						class="px-3 py-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
						onclick={() => {adding = false; newAnnotation = "";}}
					>
						Cancel
					</button>
					<button
						class="px-4 py-1 button"
						onclick={handleAddAnnotation}
						disabled={!newAnnotation.trim()}
					>
						Save Annotation
					</button>
				</div>
			</div>
		{/if}
	{/if}



	{#if scan.annotations && scan.annotations.length > 0}
		<h4 class="font-semibold ml-2 mb-0">Doctor annotations</h4>
		<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
			{#each scan.annotations as annotation}
				<div class="bg-secondary text-secondary-foreground p-4 rounded-md border">
					<p class="italic">
						"{annotation.notes}"
					</p>
					<p class="text-xs text-right mt-2">
						– Dr. {annotation.doctor.first_name} {annotation.doctor.last_name},
						{new Date(annotation.created_at).toLocaleDateString()}
					</p>
				</div>
			{/each}
		</div>
	{:else}
		<h4 class="font-semibold ml-2 mb-3">Doctor Annotations</h4>
		<p class="text-center py-2 text-muted-foreground">No annotations available</p>
	{/if}
</div>
