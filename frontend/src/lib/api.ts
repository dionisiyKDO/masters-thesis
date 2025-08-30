import { refreshAccessToken } from '$lib/auth';

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

async function uploadFile(url: string, file: File): Promise<Response> {
    let accessToken = localStorage.getItem('access_token');
    const formData = new FormData();
    formData.append('image', file);

    let response = await fetch(`${BASE_URL}${url}`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${accessToken}`, // don't set Content-Type, browser does it
        },
        body: formData,
    });

    if (response.status === 401) {
        const newAccessToken = await refreshAccessToken();
        if (newAccessToken) {
            response = await fetch(`${BASE_URL}${url}`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${newAccessToken}`,
                },
                body: formData,
            });
        }
    }

    return response;
}

export default {
    get: (url: string) => fetchWithAuth(url, { method: 'GET' }),
    post: (url:string, data: any) => fetchWithAuth(url, { method: 'POST', body: JSON.stringify(data) }),
    put: (url: string, data: any) => fetchWithAuth(url, { method: 'PUT', body: JSON.stringify(data) }),
    delete: (url: string) => fetchWithAuth(url, { method: 'DELETE' }),

    uploadFile,
};
