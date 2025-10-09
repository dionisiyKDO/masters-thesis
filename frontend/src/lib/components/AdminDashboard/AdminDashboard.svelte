<script lang="ts">
    import PerformanceStatistics from './PerformanceStatistics.svelte';
    import ModelManagement from './ModelManagement.svelte';
    import UserManagement from './UserManagement.svelte';
    import AuditLog from './AuditLog.svelte';
	import Dashboard from './Dashboard.svelte';

    type Tab = 'dashboard' | 'model' | 'audit' | 'users' | 'stats';
    interface Page {  key: Tab; label: string; }

    let currentPage: Tab = $state('dashboard');
    const pages: Page[] = [
        {key: 'dashboard', label: "Dashboard" },
        {key: 'users', label: "User Management" },
        {key: 'model', label: "Model Management" },
        {key: 'audit', label: "Audit log" },
        {key: 'stats', label: "Statistics" },
    ]
</script>

<div class="p-4 sm:p-6 md:p-8 bg-background text-foreground min-h-screen">
    <div class="max-w-7xl mx-auto space-y-6">

        <!-- Header -->
        <div class="flex flex-col">
            <h1 class="text-3xl font-bold text-card-foreground">Admin Panel</h1>
            <p class="text-muted-foreground mt-1">System oversight and management dashboard.</p>
        </div>

        <!-- Navigation -->
        <div class="flex items-center gap-2 border-b border-border">
            {#each pages as {key, label}}
                <button 
                    class="px-4 py-2 text-sm font-medium transition-colors 
                        {currentPage === key 
                            ? 'text-primary border-b-2 border-primary' 
                            : 'text-muted-foreground hover:text-foreground'}"
                    onclick={() => currentPage = key} 
                >
                    {label}
                </button>
            {/each}
        </div>

        <!-- Pages -->
        {#if currentPage === 'dashboard'}
            <Dashboard />
        {:else if currentPage === 'users'}
            <UserManagement />
        {:else if currentPage === 'model'}
            <ModelManagement />
        {:else if currentPage === 'audit'}
            <AuditLog />
        {:else if currentPage === 'stats'}
            <PerformanceStatistics />
        {/if}
        
    </div>
</div>
