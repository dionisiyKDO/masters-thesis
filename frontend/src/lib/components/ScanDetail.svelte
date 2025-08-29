<script lang="ts">
	import type { ChestScan } from "$lib/types"

	let { scan }: { scan: ChestScan } = $props();
	const API_BASE = 'http://localhost:8000';

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
	{#if scan.ai_analyses.length > 0}
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
	{#if scan.annotations.length > 0}
		<h4 class="font-semibold ml-2 mb-0">Doctor annotations</h4>
		<div class="grid grid-cols-2 gap-4">
			{#each scan.annotations as annotation}
			<div class="bg-secondary p-4 rounded-md border">
				
				<p class="text-secondary-foreground italic">
					“{annotation.notes}”
				</p>
				<p class="text-xs text-secondary-foreground text-right mt-2">
					– Dr. {annotation.doctor.username},
					{new Date(annotation.created_at).toLocaleDateString()}
				</p>
			</div>
			{/each}
		</div>
	{/if}
</div>
