// src/lib/auth.ts
export interface LocalUser {
	id: number;
    username: string;
    role: string;
    access_token: string;
    refresh_token: string;
}

export type LoginResponse = {
    access: string;
    refresh: string;
    id: number;
    username: string;
    role: string;
    error?: string;
}

// Fetched data types
export interface User {
    id: number
    username: string
    email: string
}
