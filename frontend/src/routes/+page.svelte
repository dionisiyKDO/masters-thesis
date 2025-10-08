<script lang="ts">
    import { user } from '$lib/auth';
    import PatientDashboard from '$lib/components/PatientDashboard/PatientDashboard.svelte';
    import DoctorDashboard from '$lib/components/DoctorDashboard/DoctorDashboard.svelte';
    import AdminDashboard from '$lib/components/AdminDashboard/AdminDashboard.svelte';
</script>

{#if $user}
    {#if $user.role === 'patient'}
        <PatientDashboard />
    {:else if $user.role === 'doctor'}
        <DoctorDashboard />
    {:else if $user.role === 'admin'}
        <AdminDashboard />
    {:else}
        <p>Invalid user role.</p>
    {/if}
{:else}
    <div class="text-center">
        <h1 class="text-2xl text-foreground">Welcome to the Pneumonia Diagnosis Platform</h1>
        <p class="mt-4 text-base font-normal text-muted-foreground">Please log in to view your dashboard.</p>
        <a href="/login" class="button">
            Go to Login
        </a>
    </div>    
{/if}