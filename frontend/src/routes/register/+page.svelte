<script lang="ts">
	import DoctorFields from '$lib/components/Register/DoctorFields.svelte';
	import PatientFields from '$lib/components/Register/PatientFields.svelte';
	import SharedFields from '$lib/components/Register/SharedFields.svelte';
	import { handleSubmit } from './load';

	import { fade, fly } from 'svelte/transition';
	import { cubicOut } from 'svelte/easing';

	let step = $state(1);
	let role: 'doctor' | 'patient' | '' = $state('');
	let formData: any = $state({});
	$inspect(formData)

	// Animation configuration
	const transitionConfig = {
		duration: 400,
		easing: cubicOut
	};
</script>

<div class="flex flex-1 items-center justify-center p-4">
	<div class="card w-full max-w-md">
		<form onsubmit={handleSubmit} class="relative min-h-[620px]">
			<!-- Step 1: Basic Info -->
			{#if step === 1}
				<div
					class="absolute inset-0 flex flex-col p-8"
					in:fade={{ duration: 300, delay: 150 }}
					out:fade={{ duration: 150 }}
				>
					<div class="mb-8 text-center">
						<h1 class="text-foreground mb-2 text-2xl font-bold">Create Account</h1>
						<p class="text-muted-foreground text-sm">Join our healthcare platform</p>
					</div>

					<div class="flex-1 space-y-6">
						<input class="input" type="text" placeholder="Username" name="username" bind:value={formData.username} required />

						<input class="input" type="password" placeholder="Password" name="password" bind:value={formData.password} required />

						<select class="input" bind:value={role} name="role" required>
							<option value="" disabled>Select your role</option>
							<option value="doctor">Doctor</option>
							<option value="patient">Patient</option>
						</select>
					</div>

					<div class="pt-6">
						<button
							type="button"
							onclick={() => role && (step = 2)}
							disabled={!role}
							class="button w-full"
						>
							Continue
						</button>
					</div>
				</div>
			{/if}

			<!-- Step 2: Role-specific Fields -->
			{#if step === 2}
				<div
					class="absolute inset-0 flex flex-col p-8"
					in:fly={{ x: 300, ...transitionConfig }}
					out:fly={{ x: -300, ...transitionConfig }}
				>
					<div class="mb-8 text-center">
						<h1 class="text-foreground mb-2 text-2xl font-bold">
							{role === 'doctor' ? 'Doctor Profile' : 'Patient Profile'}
						</h1>
						<p class="text-muted-foreground text-sm">Complete your profile information</p>
					</div>

					<div class="flex-1 space-y-6">
						
						<SharedFields bind:formData />

						{#if role === 'doctor'}
							<DoctorFields bind:formData />
						{:else if role === 'patient'}
							<PatientFields bind:formData />
						{/if}
					</div>

					<div class="flex gap-3 pt-6">
						<button type="button" class="button-secondary flex-1" onclick={() => (step = 1)}>Back</button>
						<button type="submit" class="button flex-1"> Create Account </button>
					</div>
				</div>
			{/if}
		</form>
	</div>
</div>
