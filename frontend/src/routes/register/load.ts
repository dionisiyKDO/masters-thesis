import { goto } from '$app/navigation';
import { login, register } from '$lib/auth';
import type { FormData } from '$lib/types';

export async function handleSubmit(formData: FormData) {
	if (formData.role === 'doctor') {
		formData.patient_profile = undefined;
	} else if (formData.role === 'patient') {
		formData.doctor_profile = undefined;
	}

	register(formData).then((success) => {
		if (success) {
			login(formData.username, formData.password).then((loginSuccess) => {
				if (loginSuccess) goto('/');
			});
		}
	});
}
