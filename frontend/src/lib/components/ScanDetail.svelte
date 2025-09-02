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
	<div class="px-5 py-1 border-b border-border">
		<div class="flex items-center gap-3">
			<div class="w-2 h-2 bg-muted rounded-full"></div>
			<h3 class="text-xl font-semibold text-foreground">
				Chest Scan - {formatDate(scan.uploaded_at)}
			</h3>
		</div>
	</div>

	<div class="p-2 space-y-8">
		<!-- Main Image Section -->
		<div class="space-y-2">
			<h4 class="text-sm font-medium text-muted-foreground uppercase tracking-wide ml-3 flex items-center gap-2">
				<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
				</svg>
				Medical Image
			</h4>
			<img
				src={`${scan.image_path}`}
				alt="Chest Scan"
				class="bg-muted/10 rounded-md border-2 border-border w-full min-h-[200px] max-h-[500px] object-contain"
			/>

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
			<div class="space-y-2">
				<h4 class="text-sm font-medium text-muted-foreground uppercase tracking-wide ml-3 flex items-center gap-2">
					<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
					</svg>
					AI Analysis Results
				</h4>

				<!-- Ensemble Result -->
				{#if scan.ensemble_result}
					<div class="bg-gradient-to-r from-primary/10 to-primary/5 rounded-lg border-2 border-primary/20 p-4 mb-4">
						<div class="flex items-center gap-2 mb-3">
							<span class="font-semibold text-primary">Ensemble Prediction</span>
							<span class="text-xs bg-primary/20 text-primary px-3 py-1 rounded-full font-medium">
								Combined Analysis
							</span>
						</div>
						<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
							<div class="flex flex-col">
								<span class="text-sm font-medium text-muted-foreground">Final Diagnosis</span>
								<span class="text-lg font-bold capitalize text-primary">
									{scan.ensemble_result.combined_prediction_label}
								</span>
							</div>
							<div class="flex flex-col">
								<span class="text-sm font-medium text-muted-foreground">Combined Confidence</span>
								<span class="text-lg font-bold text-primary">
									{(scan.ensemble_result.combined_confidence_score * 100).toFixed(2)}%
								</span>
							</div>
							<div class="flex flex-col">
								<span class="text-sm font-medium text-muted-foreground">Method</span>
								<span class="text-sm font-medium capitalize text-foreground">
									{scan.ensemble_result.method} Average
								</span>
							</div>
						</div>
						<div class="mt-3 text-xs text-muted-foreground">
							Combined result from {scan.ensemble_result.source_analyses.length} models • 
							Generated on {formatDate(scan.ensemble_result.created_at)}
						</div>
					</div>
				{/if}

				<!-- Individual Model Results -->
				<div class="space-y-1">
					<h5 class="text-xs font-medium text-muted-foreground uppercase tracking-wide ml-5">
						Individual Model Predictions
					</h5>
					<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
						{#each scan.ai_analyses as analysis}
							<div class="bg-muted rounded-md border border-border px-4 py-3">
								<div class="flex flex-col mb-2">		
									<div class="flex items-center justify-between">
										<span class="font-bold">Prediction:</span>
										<span class="capitalize px-2 py-1 rounded bg-input text-foreground text-sm font-medium">
											{analysis.prediction_label}
										</span>
									</div>
									<div class="flex items-center justify-between">
										<span class="font-bold">Confidence:</span>
										<span class="font-bold text-foreground">
											{(analysis.confidence_score * 100).toFixed(2)}%
										</span>
									</div>
								</div>
								<div class="space-y text-sm text-muted-foreground">
									<div class="flex justify-between">
										<span class="font-medium">Model:</span>
										<span class="text-xs">{analysis.model_version.model_name}</span>
									</div>
									<div class="flex justify-between">
										<span class="font-medium">Generated:</span>
										<span class="text-xs">{new Date(analysis.generated_at).toLocaleDateString()}</span>
									</div>
								</div>
							</div>
						{/each}
					</div>
				</div>
			</div>
		{/if}
	
		<!-- Doctor Annotations Section -->
		<div>
			{#if $user?.role === "doctor"}
				<div class="space-y-0">
					<div class="flex items-center justify-between mb-2">
						<h4 class="text-sm font-medium text-muted-foreground uppercase tracking-wide ml-3 flex items-center gap-2">
							<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
							</svg>
							Medical Annotations
						</h4>
						<button 
							class="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg border border-border hover:bg-muted hover:text-muted-foreground transition-colors duration-200 {adding ? 'bg-muted text-muted-foreground' : 'bg-primary text-primary-foreground'}"
							onclick={() => adding = !adding}
						>
							{#if adding}
								<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
								</svg>
								Cancel
							{:else}
								<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
								</svg>
								Add Annotation
							{/if}
						</button>
					</div>
			
					{#if adding}
						<div class="flex justify-center">
							<div class="min-w-[500px] mb-2 p-3 bg-muted/50 rounded-lg border border-dashed" transition:slide={{ duration: 300 }}>
								<textarea
									bind:value={newAnnotation}
									placeholder="Enter your medical notes and observations..."
									rows="3"
									class="w-full p-3 border border-border rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary bg-background"
								></textarea>
								<div class="flex justify-end gap-2 mt-2">
									<button
									class="px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors duration-200"
										onclick={() => {adding = false; newAnnotation = "";}}
									>
										Cancel
									</button>
									<button
										class="px-3 py-1.5 button"
										onclick={handleAddAnnotation}
										disabled={!newAnnotation.trim()}
									>
										<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
										</svg>
										Save Annotation
									</button>
								</div>
							</div>
						</div>
					{/if}
				</div>
			{/if}
		
			<div class="space-y-4">
				<h4 class="text-sm font-medium text-muted-foreground uppercase tracking-wide ml-3 flex items-center gap-2">
					<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
					</svg>
					Doctor Notes ({scan.annotations?.length || 0})
				</h4>
				
				{#if scan.annotations && scan.annotations.length > 0}
					<div class="grid grid-cols-1 md:grid-cols-2 gap-3">
						{#each scan.annotations as annotation}
							<div class="flex flex-col space-y-1 bg-secondary text-secondary-foreground p-4 rounded-md border border-border">
								<div class="italic">
									"{annotation.notes}"
								</div>
								<div class="flex items-center justify-between text-sm text-secondary-foreground/75">
									<span class="font-medium">
										Dr. {annotation.doctor.first_name} {annotation.doctor.last_name}
									</span>
									<span class="text-xs">
										{new Date(annotation.created_at).toLocaleDateString('en-US', { 
											year: 'numeric', 
											month: 'short', 
											day: 'numeric',
											hour: '2-digit',
											minute: '2-digit'
										})}
									</span>
								</div>
							</div>
						{/each}
					</div>
				{:else}
					<div class="text-center py-10 rounded-lg border-2 border-dashed border-border bg-muted/20">
						<svg class="w-12 h-12 text-muted-foreground mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
						</svg>
						<h5 class="text-sm font-medium text-muted-foreground mb-1">No annotations yet</h5>
						<p class="text-xs text-muted-foreground/80">Medical notes and observations will appear here</p>
					</div>
				{/if}
			</div>
		</div>
	</div>
</div>
