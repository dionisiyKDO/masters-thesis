<script lang="ts">
    import type { ModelVersion as Model } from "$lib/types";
    import api from "$lib/api";
    
    let isLoading: boolean = $state(true);
    let models: Model[] | null = $state([]);
    let retrainingStatus: 'idle' | 'initializing' | 'running' | 'success' | 'failed' = $state('idle');
    const model_names = ['OwnV4', 'OwnV3', 'OwnV2', 'OwnV1', 'AlexNet', 'VGG16', 'VGG19', 'ResNet50', 'DenseNet121', 'MobileNetV2']

    // Training configuration
    let trainingConfig = $state({
        model_name: 'OwnV3',
        epochs: 2,
        // batch_size: 16,
        learning_rate: 0.0003,
        optimizer: 'adam',
        // early_stopping: true,
    });
    
    let progress: any = $state({});
    let trainingResults: any = $state({});

    $inspect(progress);
    $inspect(trainingResults);

    // #region Fetch funcs
    async function fetchModels(): Promise<Model[] | null> {
        try {
            const response = await api.get('/models/');
            if (!response.ok) throw new Error('Failed to fetch models.');
            const data = await response.json();
            return data;
        } catch (err) {
            console.log(err);
            return null;
        }
    }
    
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

    async function handleDelete(modelId: number) {
        if (models != null) {
            console.log(`Deleting model ${modelId}`);
            const response = await api.delete(`/models/${modelId}/`);
            if (!response.ok) {
                console.error('Failed to delete model');
                return;
            }

            models = models.filter(m => m.id !== modelId);
        } else {
            console.log('handleDelete error: "models" is null');
        }
    }

    async function startRetraining() {
		try {
            console.log('Starting retraining job...');
            retrainingStatus = 'initializing';
            trainingResults = {};
            progress = {};
            
            const response = await api.post('/train/', trainingConfig);
			if (!response.ok) {
                retrainingStatus = 'failed';
                throw new Error('Failed to start training.');
            }

			const data = await response.json();
            setTimeout(poll, 4000) // 4 sec
			return data;
		} catch (err) {
			console.log(err);
			return null;
		}
    }

    async function poll() {
        console.log('Checking status');
        const res = await api.get('/train/progress/');
        progress = await res.json();
        if (progress.status === 'running') {
            retrainingStatus = 'running';
            setTimeout(poll, 1000);
        } else if (progress.status === 'success') {
            retrainingStatus = 'success';
            trainingResults = progress;
            // Refresh models list to show new model
            models = await fetchModels();
        } else if (progress.status === 'failed') {
            retrainingStatus = 'failed';
        }
    }

    // #region MockTrain funcs

    async function mock_startRetraining() {
        console.log('Starting mock retraining job...');
        retrainingStatus = 'running';
        trainingResults = null;
        progress = {};

        // fake initial delay
        await new Promise(r => setTimeout(r, 1000));

        progress = {
            status: 'training',
            epoch: 1,
            total_epochs: trainingConfig.epochs,
            loss: 2.67,
            val_loss: 1.72,
            acc: 0.87,
            val_acc: 0.73,
            auc: 0.94,
            val_auc: 0.61,
            precision: 0.95,
            recall: 0.86,
            f1_score: 0,
            val_precision: 0.73,
            val_recall: 1,
            val_f1_score: 0,
            message: `Epoch 1/${trainingConfig.epochs} completed`,
            elapsed: 165,
            eta: 165
        };

        console.log("Mock progress:", progress);
        setTimeout(() => pollMock(2), 3000);
        return progress;
    }

    async function pollMock(epoch: number) {
        console.log('Mock poll check...');

        // pretend training is progressing
        await new Promise(r => setTimeout(r, 800));

        if (epoch <= trainingConfig.epochs) {
            if (epoch < trainingConfig.epochs) {
                progress = {
                    ...progress,
                    epoch,
                    loss: +(Math.random() * 2).toFixed(3),
                    val_loss: +(Math.random() * 1.5).toFixed(3),
                    acc: +(0.85 + Math.random() * 0.1).toFixed(3),
                    val_acc: +(0.6 + Math.random() * 0.1).toFixed(3),
                    status: 'training',
                    message: `Epoch ${epoch}/${trainingConfig.epochs} completed`,
                    elapsed: 160 + epoch * 40,
                    eta: 160 - epoch * 40
                };
                console.log("Mock epoch update:", progress);
                setTimeout(() => pollMock(epoch + 1), 3000);
            } else {
                // training done
                progress = {
                    status: 'success',
                    epoch: trainingConfig.epochs,
                    total_epochs: trainingConfig.epochs,
                    loss: 0.72,
                    val_loss: 0.98,
                    acc: 0.898,
                    val_acc: 0.611,
                    auc: 0.96,
                    val_auc: 0.86,
                    precision: 0.96,
                    recall: 0.89,
                    f1_score: 0,
                    val_precision: 0.96,
                    val_recall: 0.49,
                    val_f1_score: 0,
                    message: 'Training complete',
                    elapsed: 317,
                    eta: 0
                };
                retrainingStatus = 'success';
                trainingResults = progress;

                // simulate updated model list
                models = [
                    ...(models || []),
                    {
                        id: Math.floor(Math.random() * 1000),
                        model_name: trainingConfig.model_name,
                        is_active: false,
                        created_at: new Date().toISOString(),
                        storage_uri: `s3://models/${trainingConfig.model_name.toLowerCase()}_${Date.now()}`,
                        description: 'Retrained model version',
                        performance_metrics: {
                            accuracy: progress.acc,
                            auc: progress.auc,
                        }
                    }
                ];

                console.log("Mock training complete:", progress);
            }
        }
    }


    // #region Helper funcs
    function formatTime(seconds: number): string {
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        if (hrs > 0) return `${hrs}h ${mins}m ${secs}s`;
        if (mins > 0) return `${mins}m ${secs}s`;
        return `${secs}s`;
    }
    
    function getProgressPercentage(): number {
        if (!progress.epoch || !progress.total_epochs) return 0;
        return (progress.epoch / progress.total_epochs) * 100;
    }

    $effect(() => {
        (async () => {
            isLoading = true;
            models = await fetchModels();
            isLoading = false;
        })();
    });
</script>

<!-- #region HTML------  -->

<div class="space-y-6">
    <!-- Header Section -->
    <div>
        <h2 class="text-2xl font-bold text-card-foreground">CNNs Management</h2>
        <p class="text-sm text-muted-foreground mt-1">Manage, activate, and retrain classification models.</p>
    </div>

    <!-- #region TrainSection -->
    <div class="px-6 py-4 bg-card rounded-lg shadow-sm border border-border space-y-3">
        <div class="flex items-center gap-6">
            <!-- Header & button -->
            <div>
                <h3 class="font-semibold text-card-foreground mb-3">Neural Network Training</h3>
                <button
                    onclick={startRetraining}
                    disabled={retrainingStatus === 'running' || retrainingStatus === 'initializing'}
                    class="px-4 py-2 w-48 justify-center bg-secondary text-secondary-foreground rounded-lg text-sm font-medium hover:bg-secondary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shrink-0"
                >
                    {#if retrainingStatus === 'running' || retrainingStatus === 'initializing'}
                        <svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                        Training...
                    {:else}
                        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h5M20 20v-5h-5" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 11a8 8 0 0114.24-5.263M20 13a8 8 0 01-14.24 5.263" /></svg>
                        Initiate Retraining
                    {/if}
                </button>
            </div>

            <!-- #region ProgressBar -->
            <div class="flex-1">
                {#if retrainingStatus === 'idle'}
                    <p class="text-sm text-muted-foreground">Trigger the pipeline to retrain on new annotated data.</p>
                {:else if retrainingStatus === 'initializing'}
                    <div class="space-y-2">
                        <div class="flex items-center justify-between text-sm">
                            <span class="text-muted-foreground font-medium">Initializing training pipeline for {trainingConfig.epochs} epochs...</span>
                        </div>
                        <div class="w-full h-2 bg-muted rounded-full overflow-hidden">
                            <div class="h-2 bg-secondary animate-pulse" style="width: 5%"></div>
                        </div>
                        <div class="flex items-center justify-between text-xs text-muted-foreground">
                            <span class="text-muted-foreground">Elapsed: TBD</span>
                            <span class="text-muted-foreground">ETA: TBD</span>
                        </div>
                    </div>
                {:else if retrainingStatus === 'running'}
                    <div class="space-y-2">
                        <div class="flex items-center justify-between text-sm">
                            <span class="text-card-foreground font-medium">
                                Epoch {progress.epoch}/{progress.total_epochs}
                            </span>
                            <span class="text-muted-foreground">
                                {getProgressPercentage().toFixed(0)}%
                            </span>
                        </div>
                        
                        <div class="w-full h-2 bg-muted rounded-full overflow-hidden">
                            <div 
                                class="h-2 bg-primary transition-all duration-500 ease-out"
                                style="width: {getProgressPercentage()}%">
                            </div>
                        </div>
                        
                        <div class="flex items-center justify-between text-xs text-muted-foreground">
                            <span>Elapsed: {formatTime(progress.elapsed || 0)}</span>
                            <span>ETA: {formatTime(progress.eta || 0)}</span>
                        </div>
                    </div>
                {:else if retrainingStatus === 'success'}
                    <div class="space-y-2">
                        <div class="flex items-center justify-between text-sm">
                            <span class="text-primary font-medium">Training complete!</span>
                        </div>
                        <div class="w-full h-2 bg-muted rounded-full overflow-hidden">
                            <div class="h-2 bg-primary" style="width: 100%"></div>
                        </div>
                        <div class="flex items-center justify-between text-xs text-muted-foreground">
                            <span class="text-xs text-muted-foreground mt-0.5">New model version is available for activation below.</span>
                        </div>
                    </div>

                {:else if retrainingStatus === 'failed'}
                    <div class="space-y-2">
                        <div class="flex items-center justify-between text-sm">
                            <span class="text-destructive font-medium">Training failed!</span>
                        </div>
                        <div class="w-full h-2 bg-muted rounded-full overflow-hidden">
                            <div class="h-2 bg-destructive" style="width: 100%"></div>
                        </div>
                        <div class="flex items-center justify-between text-xs text-muted-foreground">
                            <span class="text-xs text-muted-foreground mt-0.5">Check server logs for details.</span>
                        </div>
                    </div>
                {/if}
            </div>
        </div>

        <hr class="border-border/50" />

        <!-- #region TrainConfig -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
                <label class="block text-xs font-medium text-muted-foreground ml-2" for="optimizer">Model</label>
                <select
                    id="optimizer"
                    bind:value={trainingConfig.model_name}
                    class="w-full px-3 py-2 text-sm bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-70 disabled:cursor-not-allowed"
                    disabled={retrainingStatus === 'running'|| retrainingStatus === 'initializing'}
                >
                    {#each model_names as name}
                        <option value={name}>{name}</option>
                    {/each}
                </select>
            </div>
            <div>
                <label class="block text-xs font-medium text-muted-foreground ml-2" for="epochs">Epochs</label>
                <input
                    type="number"
                    id="epochs"
                    bind:value={trainingConfig.epochs}
                    min="1"
                    max="200"
                    class="w-full px-3 py-2 text-sm bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-70 disabled:cursor-not-allowed"
                    disabled={retrainingStatus === 'running'|| retrainingStatus === 'initializing'}
                />
            </div>
            <!-- <div>
                <label class="block text-xs font-medium text-muted-foreground ml-2" for="batch_size">Batch Size</label>
                <input
                    type="number"
                    id="batch_size"
                    bind:value={trainingConfig.batch_size}
                    min="1"
                    max="128"
                    class="w-full px-3 py-2 text-sm bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-70 disabled:cursor-not-allowed"
                    disabled={retrainingStatus === 'running'|| retrainingStatus === 'initializing'}
                />
            </div> -->
            <div>
                <label class="block text-xs font-medium text-muted-foreground ml-2" for="lr">Learning Rate</label>
                <input
                    type="number"
                    id="lr"
                    bind:value={trainingConfig.learning_rate}
                    step="0.0001"
                    min="0.00001"
                    max="0.1"
                    class="w-full px-3 py-2 text-sm bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-70 disabled:cursor-not-allowed"
                    disabled={retrainingStatus === 'running'|| retrainingStatus === 'initializing'}
                />
            </div>
            <div>
                <label class="block text-xs font-medium text-muted-foreground ml-2" for="optimizer">Optimizer</label>
                <select
                    id="optimizer"
                    bind:value={trainingConfig.optimizer}
                    class="w-full px-3 py-2 text-sm bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-primary disabled:opacity-70 disabled:cursor-not-allowed"
                    disabled={retrainingStatus === 'running'|| retrainingStatus === 'initializing'}
                >
                    <option value="adam">Adam</option>
                    <option value="adamw">AdamW</option>
                    <option value="sgd">SGD</option>
                </select>
            </div>
            <!-- <div class="flex items-center gap-4 pt-2">
                <label class="flex items-center gap-2 text-sm text-foreground cursor-pointer" for="es">Early Stopping</label>
                <input
                    type="checkbox"
                    id="es"
                    bind:checked={trainingConfig.early_stopping}
                    class="w-4 h-4 rounded border-border text-primary focus:ring-primary disabled:opacity-70 disabled:cursor-not-allowed"
                    disabled={retrainingStatus === 'running'|| retrainingStatus === 'initializing'}
                />
            </div> -->
        </div>

    </div>

    <!-- #region TrainProgress -->
    {#if retrainingStatus === 'running'}
        <!-- Metrics Grid -->
        {#if progress.epoch}
            <div class="bg-card rounded-lg shadow-sm border border-border overflow-hidden px-6 py-5">
                <div class="grid grid-cols-2 md:grid-cols-5 gap-3 pt-1">
                    <div class="bg-muted/30 rounded-lg p-3 border border-border">
                        <div class="text-xs text-muted-foreground mb-1">Loss</div>
                        <div class="text-lg font-semibold text-foreground">{progress.loss?.toFixed(4) || '—'}</div>
                        <div class="text-xs text-muted-foreground">Val: {progress.val_loss?.toFixed(4) || '—'}</div>
                    </div>
                    <div class="bg-muted/30 rounded-lg p-3 border border-border">
                        <div class="text-xs text-muted-foreground mb-1">Accuracy</div>
                        <div class="text-lg font-semibold text-foreground">{(progress.acc * 100)?.toFixed(2) || '—'}%</div>
                        <div class="text-xs text-muted-foreground">Val: {(progress.val_acc * 100)?.toFixed(2) || '—'}%</div>
                    </div>
                    <div class="bg-muted/30 rounded-lg p-3 border border-border">
                        <div class="text-xs text-muted-foreground mb-1">Precision</div>
                        <div class="text-lg font-semibold text-foreground">{(progress.precision * 100)?.toFixed(2) || '—'}%</div>
                        <div class="text-xs text-muted-foreground">Val: {(progress.val_precision * 100)?.toFixed(2) || '—'}%</div>
                    </div>
                    <div class="bg-muted/30 rounded-lg p-3 border border-border">
                        <div class="text-xs text-muted-foreground mb-1">Recall</div>
                        <div class="text-lg font-semibold text-foreground">{(progress.recall * 100)?.toFixed(2) || '—'}%</div>
                        <div class="text-xs text-muted-foreground">Val: {(progress.val_recall * 100)?.toFixed(2) || '—'}%</div>
                    </div>
                    <div class="bg-muted/30 rounded-lg p-3 border border-border">
                        <div class="text-xs text-muted-foreground mb-1">AUC</div>
                        <div class="text-lg font-semibold text-foreground">{progress.auc?.toFixed(4) || '—'}</div>
                        <div class="text-xs text-muted-foreground">Val: {progress.val_auc?.toFixed(4) || '—'}</div>
                    </div>
                </div>
            </div>
        {/if}

    {:else if retrainingStatus === 'success'}
        <!-- Success Results -->
        {#if trainingResults}
            <!-- Final Metrics -->
            <div class="bg-card rounded-lg shadow-sm border border-border overflow-hidden px-6 py-5">
                <div class="grid grid-cols-2 md:grid-cols-5 gap-3 pt-1">
                    <div class="bg-muted/30 rounded-lg p-3 border border-border">
                        <div class="text-xs text-muted-foreground mb-1">Final Accuracy</div>
                        <div class="text-2xl font-bold text-foreground">{(trainingResults.acc * 100)?.toFixed(2)}%</div>
                        <div class="text-xs text-muted-foreground mt-1">Val: {(trainingResults.val_acc * 100)?.toFixed(2)}%</div>
                    </div>
                    <div class="bg-muted/30 rounded-lg p-3 border border-border">
                        <div class="text-xs text-muted-foreground mb-1">Final Loss</div>
                        <div class="text-2xl font-bold text-foreground">{trainingResults.loss?.toFixed(4)}</div>
                        <div class="text-xs text-muted-foreground mt-1">Val: {trainingResults.val_loss?.toFixed(4)}</div>
                    </div>
                    <div class="bg-muted/30 rounded-lg p-4 border border-border">
                        <div class="text-xs text-muted-foreground mb-1">Total Time</div>
                        <div class="text-2xl font-bold text-foreground">{formatTime(trainingResults.elapsed)}</div>
                        <div class="text-xs text-muted-foreground mt-1">{trainingResults.total_epochs} epochs</div>
                    </div>
                    <div class="bg-muted/30 rounded-lg p-4 border border-border">
                        <div class="text-xs text-muted-foreground mb-1">Precision</div>
                        <div class="text-xl font-semibold text-foreground">{(trainingResults.precision * 100)?.toFixed(2)}%</div>
                    </div>
                    <div class="bg-muted/30 rounded-lg p-4 border border-border">
                        <div class="text-xs text-muted-foreground mb-1">Recall</div>
                        <div class="text-xl font-semibold text-foreground">{(trainingResults.recall * 100)?.toFixed(2)}%</div>
                    </div>
                    <div class="bg-muted/30 rounded-lg p-4 border border-border">
                        <div class="text-xs text-muted-foreground mb-1">AUC</div>
                        <div class="text-xl font-semibold text-foreground">{trainingResults.auc?.toFixed(4)}</div>
                    </div>
                    <div class="bg-muted/30 rounded-lg p-4 border border-border">
                        <div class="text-xs text-muted-foreground mb-1">Sensitivity</div>
                        <div class="text-xl font-semibold text-foreground">{(trainingResults.recall * 100)?.toFixed(2)}%</div>
                    </div>
                    <div class="bg-muted/30 rounded-lg p-4 border border-border">
                        <div class="text-xs text-muted-foreground mb-1">Specificity</div>
                        <div class="text-xl font-semibold text-foreground">{( (1 - (1 - trainingResults.recall)) * 100 )?.toFixed(2)}%</div>
                    </div> 
                </div>
            </div>
        {/if}
    {:else if retrainingStatus === 'failed'}
        <div class="bg-card rounded-lg shadow-sm border border-border overflow-hidden px-6 py-5">
            <!-- Error State -->
            <div class="space-y-4">
                <!-- Training Failed message -->
                <div class="flex items-center gap-3 p-4 bg-destructive/10 border border-destructive/30 rounded-lg">
                    <svg class="w-6 h-6 text-destructive flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div>
                        <h4 class="font-semibold text-foreground">Training Failed</h4>
                        <p class="text-sm text-muted-foreground mt-0.5">
                            An error occurred during the training process. Check logs for details.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    {/if}

    <!-- #region Models Table -->
    <div class="bg-card rounded-lg shadow-sm border border-border overflow-x-auto">
        {#if isLoading}
             <div class="flex items-center justify-center p-16"><div class="flex items-center gap-3 text-muted-foreground"><svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Loading models...</div></div>
        {:else}
            <table class="w-full table-fixed text-sm text-left text-foreground">
                <thead class="bg-muted/50 text-xs text-muted-foreground uppercase">
                    <tr>
                        <th class="w-1/5 px-6 py-3">Model</th>
                        <th class="w-3/10 px-6 py-3">Description</th>
                        <th class="w-1/5 px-6 py-3">Metrics</th>
                        <th class="w-1/10 px-6 py-3 text-center">Active</th>
                        <th class="w-1/10 px-6 py-3">Delete</th>
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
                            <td class="px-6 py-4">
                                <button
                                    class="text-destructive hover:underline text-sm"
                                    onclick={() => handleDelete(model.id)}
                                >
                                    Delete
                                </button>
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
