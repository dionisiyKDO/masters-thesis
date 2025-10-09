<script lang="ts">
    let isLoading: boolean = $state(true);
    let stats: PerformanceMetrics = $state({
        agreementRate: { agree: 0, disagree: 0 },
        performanceByVersion: [],
        confusionMatrix: { tn: 0, fp: 0, fn: 0, tp: 0 }
    });

    // Interfaces
    interface AgreementRate {
        agree: number;
        disagree: number;
    }

    interface VersionPerformance {
        version: string;
        agreement: number;
    }

    interface ConfusionMatrix {
        tn: number; // AI: Normal, Doctor: Normal
        fp: number; // AI: Pneumonia, Doctor: Normal
        fn: number; // AI: Normal, Doctor: Pneumonia
        tp: number; // AI: Pneumonia, Doctor: Pneumonia
    }

    interface PerformanceMetrics {
        agreementRate: AgreementRate;
        performanceByVersion: VersionPerformance[];
        confusionMatrix: ConfusionMatrix;
    }

    // Fetch function
    async function fetchStats(): Promise<PerformanceMetrics> {
        await new Promise(resolve => setTimeout(resolve, 100));
        return {
            agreementRate: { agree: 53, disagree: 47 },
            performanceByVersion: [
                { version: 'ResNet50-v2', agreement: 82 },
                { version: 'ResNet50-v3', agreement: 89 },
                { version: 'DenseNet-v1', agreement: 85 },
                { version: 'ResNet50-v4', agreement: 94 },
            ],
            confusionMatrix: {
                tn: 450, // AI: Normal, Doctor: Normal
                fp: 25,  // AI: Pneumonia, Doctor: Normal
                fn: 35,  // AI: Normal, Doctor: Pneumonia
                tp: 280, // AI: Pneumonia, Doctor: Pneumonia
            }
        };
    }
    
    $effect(() => {
        (async () => {
            isLoading = true;
            stats = await fetchStats();
            isLoading = false;
        })();
    });
</script>

<div class="space-y-6">
    <!-- Header -->
    <div>
        <h2 class="text-2xl font-bold text-card-foreground">Performance Statistics</h2>
        <p class="text-sm text-muted-foreground mt-1">Insights into AI model performance against doctor annotations.</p>
    </div>

    {#if isLoading}
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="h-64 bg-card rounded-lg shadow-sm border border-border animate-pulse"></div>
            <div class="h-64 bg-card rounded-lg shadow-sm border border-border animate-pulse"></div>
            <div class="md:col-span-2 bg-card rounded-lg shadow-sm border border-border animate-pulse h-72"></div>
        </div>
    {:else}
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Agreement Rate Card -->
            <div class="bg-card rounded-lg shadow-sm border border-border p-6">
                <h3 class="font-semibold text-card-foreground mb-4">Overall Model-Doctor Agreement</h3>
                <div class="flex items-center justify-center gap-6">
                    <div class="relative w-32 h-32">
                        <svg class="w-full h-full" viewBox="0 0 36 36">
                            <path class="stroke-current text-muted" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke-width="3"></path>
                            <path class="stroke-current text-primary"
                                stroke-dasharray="{stats.agreementRate.agree}, 100"
                                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none" stroke-width="3" stroke-linecap="round"></path>
                        </svg>
                        <div class="absolute inset-0 flex items-center justify-center">
                            <span class="text-3xl font-bold text-card-foreground">{stats.agreementRate.agree}%</span>
                        </div>
                    </div>
                    <div class="space-y-2 text-sm">
                        <div class="flex items-center gap-2"><div class="w-3 h-3 rounded-sm bg-primary"></div><div>Agree</div></div>
                        <div class="flex items-center gap-2"><div class="w-3 h-3 rounded-sm bg-muted"></div><div>Disagree</div></div>
                    </div>
                </div>
            </div>

            <!-- Performance by Version -->
            <div class="bg-card rounded-lg shadow-sm border border-border p-6">
                <h3 class="font-semibold text-card-foreground mb-4">Agreement Rate by Model Version</h3>
                <div class="space-y-3">
                    {#each stats.performanceByVersion as model}
                        <div class="w-full">
                            <div class="flex justify-between text-xs text-muted-foreground mb-1">
                                <span>{model.version}</span>
                                <span>{model.agreement}%</span>
                            </div>
                            <div class="h-2.5 w-full rounded-full bg-muted">
                                <div class="h-2.5 rounded-full bg-primary" style="width: {model.agreement}%"></div>
                            </div>
                        </div>
                    {/each}
                </div>
            </div>

            <!-- Confusion Matrix -->
            <div class="md:col-span-2 bg-card rounded-lg shadow-sm border border-border p-6">
                <h3 class="font-semibold text-card-foreground mb-4">AI-Doctor Confusion Matrix</h3>
                <div class="flex justify-center">
                    <table class="w-full max-w-lg border-collapse">
                        <thead>
                            <tr>
                                <th class="border-b border-r border-border p-2" colspan="2" rowspan="2"></th>
                                <th class="p-2 font-semibold" colspan="2">Doctor's Annotation</th>
                            </tr>
                            <tr>
                                <th class="p-2 font-medium text-muted-foreground bg-muted/30">Normal</th>
                                <th class="p-2 font-medium text-muted-foreground bg-muted/30">Pneumonia</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <th class="p-2 font-semibold text-center origin-center -rotate-90 w-24" rowspan="2">AI Prediction</th>
                                <th class="p-2 font-medium text-muted-foreground bg-muted/30">Normal</th>
                                <td class="p-4 text-center text-lg font-medium bg-primary text-primary-foreground">{stats.confusionMatrix.tn}</td>
                                <td class="p-4 text-center text-lg font-medium bg-muted text-muted-foreground">{stats.confusionMatrix.fn}</td>
                            </tr>
                            <tr>
                                <th class="p-2 font-medium text-muted-foreground bg-muted/30">Pneumonia</th>
                                <td class="p-4 text-center text-lg font-medium bg-muted text-muted-foreground">{stats.confusionMatrix.fp}</td>
                                <td class="p-4 text-center text-lg font-medium bg-primary text-primary-foreground">{stats.confusionMatrix.tp}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    {/if}
</div>
