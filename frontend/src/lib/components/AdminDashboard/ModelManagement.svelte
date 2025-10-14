<script lang="ts">
    import type { ModelVersion as Model } from "$lib/types";
    import api from "$lib/api";
    
    let isLoading: boolean = $state(true);
    let models: Model[] | null = $state([]);
    let retrainingStatus: 'idle' | 'running' | 'success' | 'failed' = $state('idle');

    // Fetch functions
    async function fetchModels(): Promise<Model[] | null> {
		try {
			const response = await api.get('/models/');
			if (!response.ok) throw new Error('Failed to fetch users.');
			const data = await response.json();
			return data;
		} catch (err) {
			console.log(err);
			return null;
		}
    }

    // Helper functions
    async function handleToggleActive(modelId: number) {
        if (models != null) {
            console.log(`Toggling active status for model ${modelId}`);
            const model = models.find(u => u.id === modelId);

            const response = await api.patch(`/models/${modelId}/`, {
                is_active: model!.is_active ? false : true,
            });
            if (!response.ok) {
                console.error('Failed to update model');
                return;
            }

            if (model) {
                model.is_active = !model.is_active;
                models = [...models];
            }
        } else {
            console.log('handleToggleActive error: "models" is null');
        }
    }
    
    function startRetraining() {
        console.log('Starting retraining job...');
        retrainingStatus = 'running';
        setTimeout(() => {
            const success = Math.random() > 0.3; // 70% chance of success
            retrainingStatus = success ? 'success' : 'failed';
        }, 2000);
    }
    
    $effect(() => {
        (async () => {
            isLoading = true;
            models = await fetchModels();
            isLoading = false;
        })();
    });
</script>

<div class="space-y-6">
    <!-- Header Section -->
    <div>
        <h2 class="text-2xl font-bold text-card-foreground">Model & CNN Management</h2>
        <p class="text-sm text-muted-foreground mt-1">Manage, activate, and retrain classification models.</p>
    </div>

    <!-- Retraining Section -->
    <div class="px-6 py-4 bg-card rounded-lg shadow-sm border border-border">
        <h3 class="font-semibold text-card-foreground mb-3">Neural Network Retraining</h3>
        <div class="flex items-center gap-4">
            <button
                onclick={startRetraining}
                disabled={retrainingStatus === 'running'}
                class="px-4 py-2 bg-secondary text-secondary-foreground rounded-lg text-sm font-medium hover:bg-secondary/90 transition-colors disabled:opacity-50 disabled:cursor-wait flex items-center gap-2"
            >
                {#if retrainingStatus === 'running'}
                    <svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                    Retraining...
                {:else}
                    <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h5M20 20v-5h-5" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 11a8 8 0 0114.24-5.263M20 13a8 8 0 01-14.24 5.263" /></svg>
                    Initiate Retraining
                {/if}
            </button>

            <!-- Retraining progress -->
            <div class="text-sm">

                <!-- The mock progress bar -->
                <div class="w-48 h-2 bg-muted rounded-full overflow-hidden mt-1">
                    <div class="h-2 bg-primary transition-all duration-500 ease-in-out"
                        style="width: {retrainingStatus === 'running' ? '50%' : retrainingStatus === 'success' ? '100%' : retrainingStatus === 'failed' ? '100%' : '0%'}">
                    </div>
                </div>

                {#if retrainingStatus === 'idle'}
                    <p class="text-muted-foreground">Trigger the pipeline to retrain on new annotated data.</p>
                {:else if retrainingStatus === 'running'}
                    <p class="text-secondary">Pipeline is running. This may take several hours.</p>
                {:else if retrainingStatus === 'success'}
                    <p class="text-primary">Retraining complete. New model version is available for activation.</p>
                {:else if retrainingStatus === 'failed'}
                    <p class="text-destructive">Retraining failed. Check logs for details.</p>
                {/if}

            </div>
        </div>
    </div>

    <!-- Models Table -->
    <div class="bg-card rounded-lg shadow-sm border border-border overflow-x-auto">
        {#if isLoading}
             <div class="flex items-center justify-center p-16"><div class="flex items-center gap-3 text-muted-foreground"><svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Loading models...</div></div>
        {:else}
            <table class="w-full table-fixed text-sm text-left text-foreground">
                <thead class="bg-muted/50 text-xs text-muted-foreground uppercase">
                    <tr>
                        <th class="w-1/5 px-6 py-3">Model</th>
                        <th class="w-2/5 px-6 py-3">Description</th>
                        <th class="w-1/5 px-6 py-3">Metrics</th>
                        <th class="w-1/10 px-6 py-3 text-center">Active</th>
                    </tr>
                </thead>
                <tbody>
                    {#each models as model (model.id)}
                        <tr class="border-b border-border hover:bg-muted/30">
                            <td class="px-6 py-4 font-medium">
                                <div class="flex flex-col gap-1.5">
                                    <span class="font-semibold">{model.model_name}</span>
                                    <span class="text-xs text-muted-foreground">
                                        {model.storage_uri.split('/').pop()}
                                    </span>
                                    <span class="text-xs text-muted-foreground">
                                        {new Date(model.created_at).toLocaleDateString()}
                                    </span>
                                </div>
                            </td>
                            
                            <td class="px-6 py-4 text-muted-foreground">
                                <span>{model.description}</span>
                            </td>

                            <td class="px-6 py-4 text-muted-foreground">
                                <div class="flex flex-col gap-0.5">
                                    {#each Object.entries(model.performance_metrics) as [key, value]}
                                        <div class="flex items-center gap-1">
                                            <span class="font-semibold">{key}:</span>
                                            <span>{value}</span>
                                        </div>
                                    {/each}
                                </div>
                            </td>
                            <td class="px-6 py-4 text-center">
                                <label for="toggle-{model.id}" class="flex items-center justify-center cursor-pointer">
                                    <div class="relative">
                                        <input
                                            type="checkbox"
                                            id="toggle-{model.id}"
                                            class="sr-only"
                                            checked={model.is_active}
                                            onchange={() => handleToggleActive(model.id)} />
                                        <div class="block bg-muted w-10 h-6 rounded-full"></div>
                                        <div class="dot absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition"></div>
                                    </div>
                                </label>
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        {/if}
    </div>
</div>

<style>
    input:checked ~ .dot {
        transform: translateX(100%);
        background-color: var(--primary-foreground);
    }
    input:checked ~ .block {
        background-color: var(--primary);
    }
</style>
