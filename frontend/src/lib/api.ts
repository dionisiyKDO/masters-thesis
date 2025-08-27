import { refreshAccessToken } from '$lib/auth';
import { type MedicalCase } from './types';

const BASE_URL = 'http://localhost:8000/api';

async function fetchWithAuth(url: string, options: RequestInit = {}): Promise<Response> {
    let accessToken = localStorage.getItem('access_token');
    console.log(options);
    
    // Initial request
    let response = await fetch(`${BASE_URL}${url}`, {
        ...options,
        headers: {
            ...options.headers,
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${accessToken}`
        }
    });

    // If token expired (401 Unauthorized), try refreshing it
    if (response.status === 401) {
        console.log('Access token expired. Refreshing...');
        const newAccessToken = await refreshAccessToken();

        if (newAccessToken) {
            // Retry the request with the new token
            response = await fetch(`${BASE_URL}${url}`, {
                ...options,
                headers: {
                    ...options.headers,
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${newAccessToken}`
                }
            });
        } else {
            // Handle refresh failure (e.g., redirect to login)
            console.error('Failed to refresh token. User should be logged out.');
            // The refreshAccessToken function should handle the logout and redirect.
        }
    }

    return response;
}

export default {
    get: (url: string) => fetchWithAuth(url, { method: 'GET' }),
    post: (url:string, data: any) => fetchWithAuth(url, { method: 'POST', body: JSON.stringify(data) }),
    put: (url: string, data: any) => fetchWithAuth(url, { method: 'PUT', body: JSON.stringify(data) }),
    delete: (url: string) => fetchWithAuth(url, { method: 'DELETE' }),
};




