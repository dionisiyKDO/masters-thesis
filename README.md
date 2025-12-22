# Chest X-ray Diagnostic Platform

A full-stack web platform for automated chest X-ray classification and medical case management, developed as part of a master’s thesis.
Combines deep learning–based pneumonia detection with a secure, role-based web interface for doctors, patients, and administrators.

## Key Features

### Doctors
- Image upload and management for patient X-ray scans
- AI-powered pneumonia predictions using CNN
- Create and manage medical cases and conclusions
- Access historical data per patient

### Patients
- View uploaded scans and diagnostic outcomes
- Access to medical conclusions and diagnoses
- Historical view of all cases

### Administration
- Dedicated admin panel for system management
- User management system
- Error logging and monitoring
- Database seeding
- Neural network management:
  - Model retraining through web interface
  - Training progress visualization
  - Checkpoint selection and management

## Technical Stack

- **Frontend**: Svete5 + TailwindCSS
- **Backend**: Django + DRF
- **Database**: SQLite
- **Auth & Security**: JWT authentication + role-based access control
- **AI Model**: TensorFlow with custom CNN architecture
- **Visualization**: Matplotlib for AI training analytics

## Setup

### Backend

1. Create virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python manage.py migrate
python manage.py seed_db   # optional: populate demo users & data
python manage.py runserver
```

### Frontend

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Run the application:
```bash
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173) in your browser.  
By default, the frontend connects to the Django API at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Requirements

- Python 3.10+
- Node.js 18+ / Bun
- TensorFlow 2.10+
- GPU recommended for model training and retraining

## Project Structure
```
masters-thesis/
├── backend/
│   ├── classifier/              # TensorFlow CNN & data pipeline
│   │   ├── classifier.py        # Classifier definition, training loop
│   │   ├── progress_state.py    # Training progress tracking
│   │   ├── datasets/            # Training / validation datasets
│   │   └── outputs/             # Training metrics, charts, confusion matrix, saved checkpoints, gradcam heatmaps
│   ├── api/                     # App for main Django REST API
│   │   ├── management/commands/seed_db.py     # Custom Django management command (seeding)
│   │   ├── models.py            # Core database models
│   │   ├── serializers.py       # DRF serializers
│   │   ├── permissions.py       # Role-based access control
│   │   ├── views.py             # API endpoints
│   │   ├── audit_mixin.py       # Audit mixin for logging logic
│   │   ├── audit_utils.py       # Audit utils
│   │   └── urls.py              # URL routing
│   ├── users/                   # App for Authentication and roles  (Admin/Doctor/Patient)
│   │   ├── models.py            # User-related database models
│   │   ├── serializers.py       # DRF serializers
│   │   ├── permissions.py       # Role-based access control
│   │   ├── views.py             # API endpoints
│   │   └── urls.py              # URL routing
│   ├── pneumonia_diagnosis/     # Django project settings & entry points
│   │   ├── settings.py          # Project settings
│   │   └── urls.py              # URL routing
│   ├── frontend/                # App for serving frontend/static page requests via Django
│   ├── media/                   # Uploaded scans, heatmaps
│   ├── static/                  # Compiled frontend files (for production)
│   ├── db.sqlite3               # Development database
│   └── manage.py                # Django management CLI
│
├── frontend/
│   ├── src/
│   │   ├── lib/
│   │   │   ├── components/      # Modular Svelte components
│   │   │   │   ├── AdminDashboard/       # Admin UI components
│   │   │   │   ├── DoctorDashboard/      # Doctor UI components
│   │   │   │   ├── PatientDashboard/     # Patient UI components
│   │   │   │   └── Shared UI components  # Reusable UI blocks
│   │   │   ├── api.ts              # REST API client
│   │   │   ├── auth.ts             # Authentication helpers
│   │   │   └── types.ts            # Shared TypeScript types
│   │   ├── routes/              # SvelteKit routes (login, register, dashboard, etc.)
│   │   │   ├── +page.svelte        # Page component
│   │   │   ├── +layout.svelte      # Layout component
│   │   │   ├── load.ts             # Load function for route data
│   │   │   └── +layout.ts          # Layout-level load logic
│   │   └── app.html / app.css   # Root HTML template & global styles
│   ├── svelte.config.js         # SvelteKit configuration
│   ├── vite.config.ts           # Vite bundler configuration
│   └── package.json             # Frontend dependencies & scripts
│
├── pyproject.toml   # Python project configuration
├── .gitignore       # Git ignore rules
└── README.md        # Project overview and documentation
```
