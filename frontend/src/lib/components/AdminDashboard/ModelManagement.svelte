<script lang="ts">
    import type { ModelVersion as Model } from "$lib/types";
    
    let isLoading: boolean = $state(true);
    let models: Model[] = $state([]);
    let retrainingStatus: 'idle' | 'running' | 'success' | 'failed' = $state('idle');

    // Fetch functions
    async function fetchModels(): Promise<Model[]> {
        await new Promise(resolve => setTimeout(resolve, 100));
        return [
            { id: 1, model_name: 'ResNet50-v4', description: 'Latest version trained on augmented dataset.', is_active: true, performance_metrics: { accuracy: 0.94, auc: 0.97 }, storage_uri: "./models/model1.hdf5", created_at: '2024-05-10T10:00:00Z' },
            { id: 2, model_name: 'DenseNet121-v2', description: 'Specialized for subtle cases.', is_active: true, performance_metrics: { accuracy: 0.92, auc: 0.98 }, storage_uri: "./models/model2.hdf5", created_at: '2024-03-22T15:00:00Z' },
            { id: 3, model_name: 'ResNet50-v3', description: 'Previous primary model.', is_active: false, performance_metrics: { accuracy: 0.91, auc: 0.95 }, storage_uri: "./models/model3.hdf5", created_at: '2023-11-15T09:00:00Z' },
        ];
    }

    // Helper functions
    function handleToggleActive(modelId: number) {
        console.log(`Toggling active status for model ${modelId}`);
        const model = models.find(m => m.id === modelId);
        if (model) {
            model.is_active = !model.is_active;
            models = [...models];
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
                        <th class="w-1/5 px-6 py-3">Model Name</th>
                        <th class="w-2/5 px-6 py-3">Description</th>
                        <th class="w-1/5 px-6 py-3">Metrics</th>
                        <th class="w-1/10 px-6 py-3">Is Active</th>
                    </tr>
                </thead>
                <tbody>
                    {#each models as model (model.id)}
                        <tr class="border-b border-border hover:bg-muted/30">
                            <td class="px-6 py-4 font-medium">{model.model_name}</td>
                            <td class="px-6 py-4 text-muted-foreground">{model.description}</td>
                            <td class="px-6 py-4 text-muted-foreground">
                                Acc: {model.performance_metrics.accuracy}, AUC: {model.performance_metrics.auc}
                            </td>
                            <td class="px-6 py-4 text-center">
                                <label for="toggle-{model.id}" class="flex items-center justify-center cursor-pointer">
                                    <div class="relative">
                                        <input type="checkbox" id="toggle-{model.id}" class="sr-only" checked={model.is_active} onchange={() => handleToggleActive(model.id)}>
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
