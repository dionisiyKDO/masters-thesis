<script lang="ts">
    // Fetch functions
    async function fetchDashboardStats() {
        await new Promise(resolve => setTimeout(resolve, 100));
        return {
            totalUsers: 135,
            totalDoctors: 15,
            totalPatients: 120,
            activeModels: 3,
            errorsLast24h: 2,
        };
    }

    async function fetchRecentActivity() {
        await new Promise(resolve => setTimeout(resolve, 100));
        return [
            { id: 1, action: 'ACTIVATED_MODEL', details: "Admin activated model 'ResNet50-v4'", timestamp: new Date(Date.now() - 3600000) },
            { id: 2, action: 'REGISTER_DOCTOR', details: "Dr. Evelyn Reed was registered.", timestamp: new Date(Date.now() - 7200000) },
            { id: 3, action: 'API_ERROR', details: "500 Server Error on /predict endpoint.", timestamp: new Date(Date.now() - 86400000) },
            { id: 4, action: 'USER_LOGIN', details: "Dr. John Carter logged in.", timestamp: new Date(Date.now() - 90000000) },
        ];
    }

    // Helper functions
    function formatDate(dateString: string) {
        const date = new Date(dateString);
        return date.toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' });
    } 

    let statsReq: Promise<any> = fetchDashboardStats();
    let activityReq: Promise<any> = fetchRecentActivity();
</script>

<div class="space-y-6">
    <!-- Quick Stats Section -->
    {#await statsReq}
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {#each Array(4) as _}
                <div class="px-6 py-4 bg-card rounded-lg shadow-sm border border-border animate-pulse">
                    <div class="h-6 bg-muted rounded w-3/4 mb-2"></div>
                    <div class="h-10 bg-muted rounded w-1/2"></div>
                </div>
            {/each}
        </div>
    {:then stats}
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <!-- Total Users Card -->
            <div class="px-6 py-4 bg-card rounded-lg shadow-sm border border-border flex justify-between items-center">
                <div class="flex flex-col items-center">
                    <h3 class="text-sm font-medium text-muted-foreground">Total Users</h3>
                    <p class="text-3xl font-bold text-card-foreground">{stats.totalUsers}</p>
                </div>
                <div>
                    <p class="text-sm text-muted-foreground mt-1">{stats.totalDoctors} Doctors</p>
                    <p class="text-sm text-muted-foreground mt-1">{stats.totalPatients} Patients</p>
                </div>
            </div>

            <!-- Active Models Card -->
            <div class="px-6 py-4 bg-card rounded-lg shadow-sm border border-border">
                <h3 class="text-sm font-medium text-muted-foreground">Active Models</h3>
                <p class="text-3xl font-bold text-card-foreground">{stats.activeModels}</p>
                <p class="text-xs text-muted-foreground mt-1">Ready for prediction</p>
            </div>

            <!-- Errors Card -->
            <div class="px-6 py-4 bg-card rounded-lg shadow-sm border border-border">
                <h3 class="text-sm font-medium text-muted-foreground">Errors (24h)</h3>
                <p class="text-3xl font-bold" class:text-red-500={stats.errorsLast24h > 0} class:text-card-foreground={stats.errorsLast24h === 0}>{stats.errorsLast24h}</p>
                <p class="text-xs text-muted-foreground mt-1">System & API logs</p>
            </div>
        </div>
    {/await}

    <!-- Recent Activity Section -->
    <div class="px-6 py-4 bg-card rounded-lg shadow-sm border border-border">
        <h2 class="text-lg font-semibold text-card-foreground mb-4">Recent Activity</h2>
        {#await activityReq}
            <div class="space-y-3">
                {#each Array(4) as _}
                    <div class="flex items-center gap-4 p-2 bg-background rounded animate-pulse">
                        <div class="w-8 h-8 rounded-full bg-muted"></div>
                        <div class="flex-1 space-y-2">
                            <div class="h-4 bg-muted rounded w-3/4"></div>
                            <div class="h-3 bg-muted rounded w-1/2"></div>
                        </div>
                    </div>
                {/each}
            </div>
        {:then activities}
            <ul class="space-y-2">
                {#each activities as activity}
                    <li class="flex items-center gap-4 p-2 hover:bg-muted/50 rounded-md">
                        <div class="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center {activity.action === 'API_ERROR' ? 'bg-red-100' : 'bg-secondary'}">
                            {#if activity.action === 'API_ERROR'}
                                <svg class="w-5 h-5 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
                            {:else if activity.action === 'REGISTER_DOCTOR'}
                                <svg class="w-5 h-5 text-secondary-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" /></svg>
                            {:else}
                                <svg class="w-5 h-5 text-secondary-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                            {/if}
                        </div>
                        <div class="flex-1 min-w-0">
                            <p class="text-sm text-foreground truncate">{activity.details}</p>
                            <p class="text-xs text-muted-foreground">{formatDate(activity.timestamp)}</p>
                        </div>
                    </li>
                {/each}
            </ul>
        {/await}
    </div>
</div>