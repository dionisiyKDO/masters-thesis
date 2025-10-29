<script lang="ts">
    import type { AuditLog } from "$lib/types";
    import api from "$lib/api";

    let isLoading: boolean = $state(true);
    let searchTerm = $state('');
    let searchDate = $state('');
    let logs: AuditLog[] | null = $state([]);
    let filteredLogs = $derived(
        // logs.filter((log: AuditLog) => 
        //     (log.user.toLowerCase().includes(searchTerm.toLowerCase()) || log.action.toLowerCase().includes(searchTerm.toLowerCase())) &&
        //     (searchDate === '' || new Date(log.created_at).toDateString() === new Date(searchDate).toDateString())
        // )

        logs.filter((log: AuditLog) => {
            let userStr = `${log.user.first_name ?? ''} ${log.user.last_name ?? ''} ${log.user.email ?? ''} ${log.user.role ?? ''}`;
            
            return (
                userStr.toLowerCase().includes(searchTerm.toLowerCase()) ||
                JSON.stringify(log.details).toLowerCase().includes(searchTerm.toLowerCase()) ||
                log.action.toLowerCase().includes(searchTerm.toLowerCase())
            ) && (
                searchDate === '' || new Date(log.created_at).toDateString() === new Date(searchDate).toDateString()
            );
        })
    );
    
    // Fetch functions
    // async function fetchLogs(): Promise<AuditLog[]> {
    //     await new Promise(resolve => setTimeout(resolve, 800));
    //     return [
    //         { id: 1, user: 'admin@app.com', action: 'MODEL_ACTIVATED', details: { model_name: 'ResNet50-v4' }, created_at: new Date(Date.now() - 3600000).toLocaleString() },
    //         { id: 2, user: 'j.carter@clinic.com', action: 'LOGIN_SUCCESS', details: { ip_address: '192.168.1.10' }, created_at: new Date(Date.now() - 4200000).toLocaleString() },
    //         { id: 3, user: 'System', action: 'API_ERROR', details: { endpoint: '/api/predict', status: 500, error: 'Internal Server Error' }, created_at: new Date(Date.now() - 86400000).toLocaleString() },
    //         { id: 4, user: 'admin@app.com', action: 'USER_DEACTIVATED', details: { user_id: 'i9j0k1l2', reason: 'Admin action' }, created_at: new Date(Date.now() - 172800000).toLocaleString() },
    //         { id: 5, user: 'jane.doe@clinic.com', action: 'DIAGNOSIS_ADDED', details: { case_id: 'c123', patient_id: 'p456' }, created_at: new Date(Date.now() - 259200000).toLocaleString() },
    //     ];
    // }
    async function fetchLogs(): Promise<AuditLog[] | null> {
		try {
			const response = await api.get('/auditlogs/');
			if (!response.ok) throw new Error('Failed to fetch users.');
			const data = await response.json();
            console.log(data);
            
			return data;
		} catch (err) {
			console.log(err);
			return null;
		}
	}

    // Helper functions
    function formatDate(dateString: string) {
        const date = new Date(dateString);
        return date.toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' });
    } 
    
    $effect(() => {
        (async () => {
            isLoading = true;
            logs = await fetchLogs();
            isLoading = false;
        })();
    });
</script>

<div class="space-y-6">
    <!-- Header Section -->
    <div>
        <h2 class="text-2xl font-bold text-card-foreground">Audit Log</h2>
        <p class="text-sm text-muted-foreground mt-1">Monitor system activities and troubleshoot errors.</p>
    </div>

    <!-- Filters -->
    <div class="px-4 py-3 bg-card rounded-lg shadow-sm border border-border flex items-center gap-4">
       <input type="text" bind:value={searchTerm} placeholder="Filter by user, action or details..." class="flex-1 bg-background border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50">
       <input type="date" bind:value={searchDate} class="bg-background border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50">
    </div>

    <!-- Logs Table -->
    <div class="bg-card rounded-lg shadow-sm border border-border overflow-x-auto">
        {#if isLoading}
            <div class="flex items-center justify-center p-16"><div class="flex items-center gap-3 text-muted-foreground"><svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Loading logs...</div></div>
        {:else}
            <table class="w-full table-fixed text-sm text-left text-foreground">
                <thead class="bg-muted/50 text-xs text-muted-foreground uppercase">
                    <tr>
                        <th scope="col" class="w-[240px] px-4 py-3">Event</th>
                        <th scope="col" class="px-4 py-3">Details</th>
                    </tr>
                </thead>
                <tbody>
                    {#each filteredLogs as log (log.id)}
                        <tr class="border-b border-border hover:bg-muted/30">
                            <td class="px-4 py-4">
                                <div class="border-l-2 {log.action.includes('ERROR') ? 'border-destructive' : 'border-primary'} pl-3 space-y-1.5">
                                    <div class="text-xs font-semibold {log.action.includes('ERROR') ? 'text-destructive' : 'text-primary'}">
                                        {log.action}
                                    </div>
                                    <div class="text-sm font-medium text-foreground break-all">{log.user.email}</div>
                                    <div class="text-xs text-muted-foreground/70">{formatDate(log.created_at)}</div>
                                </div>
                            </td>
                            <td class="px-4 py-4 text-muted-foreground">
                                <pre class="bg-background/50 p-2 rounded text-xs whitespace-pre-wrap break-all">{JSON.stringify(log.details, null, 2)}</pre>
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        {/if}
    </div>
</div>
