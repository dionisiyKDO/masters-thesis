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

// API Model Types
export interface SimpleUser {
    id: number;
    username: string;
    email: string;
    first_name: string;
    last_name: string;
    role: string;
}

export interface AIAnalysis {
    id: number;
    prediction_label: 'pneumonia' | 'normal';
    confidence_score: number;
    heatmap_path: string | null;
    heatmap_type: string;
    generated_at: string;
    scan: number;
    model_version: ModelVersion;
}

export interface ModelVersion {
  id: number
  uploaded_by_admin: SimpleUser
  model_name: string
  storage_uri: string
  description: string
  performance_metrics: PerformanceMetrics
  is_active: boolean
  created_at: string
}

export interface PerformanceMetrics {
  accuracy: number
  f1_score: number
  precision: number
}

export interface DoctorAnnotation {
    id: number;
    notes: string;
    created_at: string;
    scan: number;
    doctor: SimpleUser;
}
export interface Ensemble {
    id: number;
    source_analyses: AIAnalysis[];
    method: 'majority_vote' | 'average' | 'weighted';
    combined_prediction_label: 'pneumonia' | 'normal';
    combined_confidence_score: number;
    created_at: string;
    scan: number;
}

export interface ChestScan {
    id: number;
    image_path: string;
    uploaded_at: string;
    case: number;
    ai_analyses: AIAnalysis[];
    annotations: DoctorAnnotation[];
    ensemble_result: Ensemble;
}

export interface MedicalCase {
    id: number;
    title: string;
    description: string;
    status: 'open' | 'closed' | 'monitoring';
    diagnosis_summary: string | null;
    created_at: string;
    updated_at: string;
    patient: SimpleUser;
    primary_doctor: SimpleUser;
}

export interface MedicalCaseDetail extends MedicalCase {
    scans: ChestScan[];
}

// Register form data type
export interface FormData {
	username: string;
	password: string;
	email: string;
	first_name: string;
	last_name: string;
	role: 'patient' | 'doctor';
	doctor_profile: {
		specialization: string;
		license_number: string;
	} | undefined;
	patient_profile: {
		dob: string;
		sex: string;
		medical_record_number: string;
		contact_info: string;
	} | undefined;
}