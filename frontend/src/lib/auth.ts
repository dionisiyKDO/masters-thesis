import { goto } from '$app/navigation';
import { writable } from 'svelte/store';
import type { LocalUser as User, LoginResponse, FormData } from '$lib/types';

export const user = writable<User | null>(null);

export async function login(username: string, password: string): Promise<boolean> {
	try {
		const isStandalone = window.location.port !== '8000';
		const apiBase = isStandalone ? 'http://localhost:8000' : '';
		const request_link = `${apiBase}/api/auth/token/`

		const response = await fetch(request_link, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ username, password })
		});

		console.log("Making request to: ", request_link);

		if (!response.ok) {
			const data = await response.json();
			const error = data.error || 'Failed to login';
			console.log(error);
			return false;
		}

		const data = (await response.json()) as LoginResponse;

		const receivedAccessToken = data.access;
		const receivedRefreshToken = data.refresh;
		const receivedId = data.id;
		const receivedRole = data.role;

		const usr = {
			id: receivedId,
			username: username,
			role: receivedRole,
			access_token: receivedAccessToken,
			refresh_token: receivedRefreshToken
		};

		localStorage.setItem('access_token', usr.access_token);
		localStorage.setItem('refresh_token', usr.refresh_token);
		localStorage.setItem('username', usr.username);
		localStorage.setItem('role', usr.role);
		localStorage.setItem('user_id', usr.id.toString());

		user.set(usr);

		return true; // Login successful
	} catch (error) {
		console.error('Error during login:', error);
		return false; // Login failed
	}
}

export async function refreshAccessToken(): Promise<string | null> {
	const refreshToken = localStorage.getItem('refresh_token');
	if (!refreshToken) {
		console.error('No refresh token found');
		return null;
	}

	try {
		const isStandalone = window.location.port !== '8000';
		const apiBase = isStandalone ? 'http://localhost:8000' : '';

		const response = await fetch(`${apiBase}/api/auth/token/refresh/`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ refresh: refreshToken })
		});

		if (!response.ok) {
			console.error('Failed to refresh token');
			logout(); // Log out if refresh fails
			return null;
		}

		const data = await response.json();
		console.log('data', data);

		const newAccessToken = data.access;

		localStorage.setItem('access_token', newAccessToken);
		user.update((u) => (u ? { ...u, access_token: newAccessToken } : null));

		return newAccessToken;
	} catch (error) {
		console.error('Error refreshing token:', error);
		logout(); // Log out on error
		return null;
	}
}

export function initializeToken(): void {
	const storedAccessToken = localStorage.getItem('access_token');
	const storedRefreshToken = localStorage.getItem('refresh_token');
	const storedUsername = localStorage.getItem('username');
	const storedUserId = localStorage.getItem('user_id'); // Retrieve the user ID
	const storedRole = localStorage.getItem('role');
	if (storedAccessToken && storedRefreshToken && storedUsername && storedUserId && storedRole) {
		const usr: User = {
            id: parseInt(storedUserId, 10),
            access_token: storedAccessToken,
            refresh_token: storedRefreshToken,
            username: storedUsername,
			role: storedRole
        };
        user.set(usr);
	}
}

export function logout(): void {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('username');
    localStorage.removeItem('role');
    localStorage.removeItem('user_id');
    user.set(null);
    goto('/');
}

export async function register(formData: FormData): Promise<boolean> {
	try {
		const isStandalone = window.location.port !== '8000';
		const apiBase = isStandalone ? 'http://localhost:8000' : '';
		const request_link = `${apiBase}/api/auth/register/`

		const response = await fetch(request_link, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ ...formData })
		});

		console.log("Making request to: ", request_link);

		if (!response.ok) {
			const data = await response.json();
			const error = data.error || 'Failed to register user';
			console.log(error);
			return false;
		}

		return true;
	} catch (error) {
		console.error('Error during login:', error);
		return false;
	}
}