<script>
	import { handleSubmit } from './load';
</script>

<div class="flex flex-col items-center justify-center gap-4">
	<h1 class="text-center text-3xl font-bold">Login</h1>

	<div class="max-w-md min-w-96">
		<form onsubmit={handleSubmit} class="flex flex-col gap-2">
			<input class="input" name="username" type="text" placeholder="username" required />
			<br />
			<input class="input" name="password" type="password" placeholder="password" required />
			<br />
			<button class="button" type="submit">Login</button>
		</form>
	</div>
</div>
