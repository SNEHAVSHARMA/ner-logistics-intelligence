# NER Logistics Intelligence Platform

A demo-ready Smart Logistics and Accessibility Intelligence Platform for the North Eastern Region (NER) of India.

## What is included

This version preserves the original route-planning, AI/ML, weather, accessibility, cost, alerts, analytics, regional-risk and dynamic-recalculation workflow, and adds the operational features requested from the reference project:

- Dashboard
- Route Planner
- Live Map
- Vehicle Tracking / Fleet Status
- Incidents / Incident Reporting
- Field Reports
- Live Alerts
- Analytics
- Regional Risk
- Emergency Help by NER state
- Settings / prototype controls

### Existing core intelligence
- Multiple route alternatives
- Random Forest prototype risk prediction
- Risk categories and risk contributors
- Dynamic route scoring for Fastest / Cheapest / Safest / Balanced priorities
- Weather intelligence
- Accessibility scoring
- Estimated logistics cost calculation
- Route comparison
- AI recommendation with human-readable explanation
- Landslide simulation and dynamic recalculation
- Regional NER risk overview
- Synthetic/demo data clearly labeled as prototype data

### Added operational features
- Fleet cards with vehicle ID, type, capacity, driver and status
- Fleet map with prototype vehicle positions
- Incident map and incident submission form
- Incident severity/status and automatic alert creation
- Field-report submission and recent report list
- Emergency contacts for all eight NER states
- Demo/placeholder labeling for contacts that must be verified before real-world use
- Settings page for prototype-data and feed preferences

## Stack
- Frontend: React + TypeScript + Vite + CSS
- Backend: Python + FastAPI + scikit-learn + pandas + NumPy
- Maps: self-contained visualization; no paid map key required
- ML: Random Forest trained on clearly synthetic prototype data

## Run

### Backend
From the project root, open one terminal:

```bash
cd backend
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Then:
```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

### Frontend
Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Keep both terminals running and open the local Vite URL shown by the frontend terminal.

## Demo

Use:
- Source: Guwahati
- Destination: Itanagar
- Vehicle: Heavy Truck
- Cargo: Medicine / Medical Supplies
- Weight: 5 tonnes
- Priority: Safest

Click **Analyze Routes**, then use **Simulate Landslide** to demonstrate dynamic recalculation.

For the new operational features, open **Vehicles**, **Incidents**, **Field Reports**, **Live Alerts**, and **Emergency Help** from the left navigation.

## API endpoints

Existing:
- GET `/api/health`
- GET `/api/routes`
- GET `/api/weather`
- GET `/api/alerts`
- GET `/api/regions`
- GET `/api/analytics`
- POST `/api/risk/predict`
- POST `/api/cost/calculate`
- POST `/api/routes/analyze`
- POST `/api/routes/recalculate`

Added:
- GET `/api/vehicles`
- GET `/api/incidents`
- POST `/api/incidents`
- GET `/api/field-reports`
- POST `/api/field-reports`
- GET `/api/emergency/{state}`

## Data note

The prototype uses synthetic/demo route, weather, disruption, fleet, incident and ML training data where live datasets/APIs are unavailable. It does not represent real-time road conditions or live GPS tracking. Emergency contact entries marked **demo** must be verified with the relevant state administration before real-world use.
