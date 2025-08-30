<script>
import { handleSubmit } from './load';
</script>

<div class="flex-1 flex items-center justify-center p-4">
	<div class="card w-full max-w-md">
		<form onsubmit={handleSubmit} class="flex flex-col min-h-[360px] p-8">
			<div class="text-center mb-8">
				<h1 class="text-2xl font-bold text-foreground mb-1.5">Welcome Back</h1>
				<p class="text-sm text-muted-foreground">Sign in to your account</p>
			</div>

			<div class="flex-1 space-y-5">
				<input
					class="input"
					name="username"
					type="text"
					placeholder="Username"
					required
				/>
				
				<input
					class="input"
					name="password"
					type="password"
					placeholder="Password"
					required
				/>
			</div>

			<div class="pt-6">
				<button class="button w-full" type="submit">
					Sign In
				</button>
			</div>
		</form>
	</div>
</div>