<script lang="ts">
    import type { AIAnalysis } from "$lib/types";

    interface Props { analysis: AIAnalysis }
	let { analysis }: Props = $props();

	let showModal = $state(false);
</script>

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

	<!-- Heatmap -->
    <div class="text-sm">
		<button
			class="text-primary hover:underline unde transition-colors font-medium disabled:text-primary/50 disabled:no-underline"
			onclick={() => (showModal = true)}
            disabled={!analysis.heatmap_path}
		>
			Show heatmap
		</button>
	</div>

	{#if showModal}
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div onclick={() => (showModal = false)} class="fixed inset-0 bg-black/50 z-40">
			<div
				class="bg-card p-6 rounded-lg border border-border space-y-6 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-2/3 md:w-1/2 lg:w-2/5 z-50"
				onclick={(event) => event.stopPropagation()}
			>
				<h2 class="text-2xl font-bold text-foreground">Heatmap</h2>
				<img
					src={`${analysis.heatmap_path}`}
					alt="Heatmap for {analysis.prediction_label}"
					class="bg-muted/10 rounded-md border-2 border-border w-full min-h-[200px] max-h-[500px] object-contain"
				/>
			</div>
		</div>
	{/if}
</div>