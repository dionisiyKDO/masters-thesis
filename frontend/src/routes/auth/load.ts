import { goto } from '$app/navigation';
import { user, login } from '$lib/auth';

export async function handleSubmit(event: Event) {
	event.preventDefault();

	const form = event.target as HTMLFormElement;
	const username = form.username.value;
	const password = form.password.value;

	login(username, password).then((success) => {
		if (success) {
			goto('/');
		}
	});
}
