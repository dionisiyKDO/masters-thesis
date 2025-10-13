<script lang="ts">
    import type { User } from "$lib/types";
    import api from "$lib/api";
	import { user } from "$lib/auth";

    let users: User[] | null = $state([]);
    let isLoading = $state(true);
    let searchTerm = $state('');
    let isEditModalOpen = $state(false);
    let selectedUser: User | null = $state(null);

    let filteredUsers = $derived(
        users.filter((user: User) => 
            `${user.first_name} ${user.last_name}`.toLowerCase().includes(searchTerm.toLowerCase()) ||
            user.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
            user.role.toLowerCase().includes(searchTerm.toLowerCase())
        )
    );

    $inspect(users);
    // $inspect(filteredUsers);
    $inspect(selectedUser);
    
    // Fetch functions
    async function fetchUsers(): Promise<User[] | null> {
		try {
			const response = await api.get('/list/users/');
			if (!response.ok) throw new Error('Failed to fetch users.');
			const data = await response.json();
			return data;
		} catch (err) {
			console.log(err);
			return null;
		}
	}

    // Helper functions
    function openEditModal(user: User) {
        selectedUser = { ...user };
        isEditModalOpen = true;
    }

    async function handleSaveChanges() {
        if (users != null) {
            console.log('Saving changes for:', selectedUser);
            
            const response = await api.patch(`/list/users/${selectedUser!.id}/`, {
                first_name: selectedUser!.first_name,
                last_name: selectedUser!.last_name,
                email: selectedUser!.email
            });
            if (!response.ok) {
                console.error('Failed to update user');
                return;
            }

            const index = users.findIndex((u: User) => u.id === selectedUser!.id);
            if (index !== -1) {
                users[index] = selectedUser!;
            }
            isEditModalOpen = false;
            selectedUser = null;
        } else {
            console.log('handleSaveChanges error: "users" is null');
        }
    }

    async function handleDeactivateUser(userId: number) {
        if (users != null) {
            console.log('Deactivating user:', userId);
            const user = users.find(u => u.id === userId);

            const response = await api.patch(`/list/users/${userId}/`, {
                is_active: user!.is_active ? false : true,
            });
            if (!response.ok) {
                console.error('Failed to update user');
                return;
            }

            
            if (user) {
                user.is_active = !user.is_active;
                users = [...users];
            }
        } else {
            console.log('handleDeactivateUser error: "users" is null');
        }
    }

    $effect(() => {
        (async () => {
            isLoading = true;
            users = await fetchUsers();
            isLoading = false;
        })();
    });
</script>

<div class="space-y-6">
    <!-- Header Section -->
    <div>
        <h2 class="text-2xl font-bold text-card-foreground">User Management</h2>
        <p class="text-sm text-muted-foreground mt-1">View, edit, and manage all user accounts.</p>
    </div>

    <!-- Search -->
    <div class="px-4 py-3 bg-card rounded-lg shadow-sm border border-border">
        <input 
            type="text" 
            bind:value={searchTerm}
            placeholder="Search by name, email, or role..."
            class="w-full bg-background border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
        >
    </div>

    <!-- User Table -->
    <div class="bg-card rounded-lg shadow-sm border border-border overflow-x-auto">
        {#if isLoading}
            <div class="flex items-center justify-center p-16">
                <div class="flex items-center gap-3 text-muted-foreground">
                    <svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Loading users...
                </div>
            </div>
        {:else}
            <table class="w-full table-fixed text-sm text-left text-foreground">
                <thead class="bg-muted/50 text-xs text-muted-foreground uppercase">
                    <tr>
                        <th class="w-2/6 min-w-[160px] px-6 py-3">Name</th>
                        <th class="w-2/6 min-w-[200px] px-6 py-3">Email</th>
                        <th class="w-1/12 min-w-[90px] px-6 py-3">Role</th>
                        <th class="w-1/12 min-w-[90px] px-6 py-3">Status</th>
                        <th class="w-1/6 min-w-[120px] px-6 py-3 text-right">Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {#each filteredUsers as user (user.id)}
                        <tr class="border-b border-border hover:bg-muted/30">
                            <td class="px-6 py-4 font-medium">{user.first_name} {user.last_name}</td>
                            <td class="px-6 py-4 text-muted-foreground">{user.email}</td>
                            <td class="px-6 py-4">
                                <span
                                    class="capitalize inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-medium
                                        {user.role === 'doctor'
                                            ? 'bg-primary/20 text-primary ring-1 ring-inset ring-primary/40'
                                            : 'bg-secondary/20 text-secondary-foreground ring-1 ring-inset ring-secondary/40'}"
                                >
                                    {user.role}
                                </span>
                            </td>
                            <td class="px-6 py-4">
                                <span
                                    class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md text-xs font-medium
                                        {user.is_active
                                            ? 'bg-primary/20 text-primary ring-1 ring-inset ring-primary/40'
                                            : 'bg-destructive/20 text-destructive ring-1 ring-inset ring-destructive/40'}"
                                >
                                    {user.is_active ? 'Active' : 'Inactive'}
                                </span>
                            </td>
                            <td class="px-6 py-4 text-right space-x-2">
                                <button onclick={() => handleDeactivateUser(user.id)} class="font-medium {user.is_active ? 'text-red-500 hover:underline' : 'text-green-500 hover:underline'}">
                                    {user.is_active ? 'Deactivate' : 'Activate'}
                                </button>
                                <button onclick={() => openEditModal(user)} class="font-medium text-primary hover:underline">Edit</button>
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        {/if}
    </div>
</div>

<!-- Edit User Modal -->
{#if isEditModalOpen && selectedUser}
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="fixed inset-0 bg-black/60 z-40 flex items-center justify-center p-4" onclick={() => isEditModalOpen = false}>
        <div class="bg-card rounded-lg shadow-xl w-full max-w-md" onclick={(e) => e.stopPropagation()}>
            <div class="p-6 border-b border-border">
                <h3 class="text-lg font-semibold text-card-foreground">Edit User</h3>
                <p class="text-sm text-muted-foreground">Modify the details for {selectedUser.first_name} {selectedUser.last_name}.</p>
            </div>
            <div class="p-6 space-y-4">
                 <div>
                    <label for="firstName" class="block text-sm font-medium text-muted-foreground mb-1">First Name</label>
                    <input type="text" id="firstName" bind:value={selectedUser.first_name} class="input px-3 py-2">
                </div>
                 <div>
                    <label for="lastName" class="block text-sm font-medium text-muted-foreground mb-1">Last Name</label>
                    <input type="text" id="lastName" bind:value={selectedUser.last_name} class="input px-3 py-2">
                </div>
                 <div>
                    <label for="email" class="block text-sm font-medium text-muted-foreground mb-1">Email</label>
                    <input type="email" id="email" bind:value={selectedUser.email} class="input px-3 py-2">
                </div>
            </div>
            <div class="px-6 py-4 bg-muted/30 flex justify-end gap-3 rounded-b-lg">
                <button onclick={() => isEditModalOpen = false} class="px-4 py-2 bg-muted text-muted-foreground rounded-lg text-sm font-medium hover:bg-muted/80 transition-colors">Cancel</button>
                <button onclick={handleSaveChanges} class="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors">Save Changes</button>
            </div>
        </div>
    </div>
{/if}
