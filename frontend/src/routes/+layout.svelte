<script lang="ts">
	import favicon from '$lib/assets/favicon.svg';
	import Header from '$lib/Header.svelte';
	import Footer from '$lib/Footer.svelte';
	import '../app.css';

	import { initializeToken } from '$lib/auth';
	
	let { children } = $props();

	initializeToken();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>


<div class="flex flex-col min-h-screen">
	<Header />

	<main class="flex-1 flex mx-auto w-full flex-col p-4 max-w-7xl">
		{@render children?.()}
	</main>

	<Footer />
</div>
