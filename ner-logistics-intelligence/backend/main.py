
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = FastAPI(title="NER Logistics Intelligence API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Clearly synthetic demo data
# -----------------------------
ROUTES = {
    "Guwahati-Itanagar": [
        {
            "id": "A", "name": "Route A — NH27 / NH15", "distance": 340,
            "time_min": 500, "fuel_efficiency": 4.0, "road": 82,
            "accessibility": 87, "reliability": 89, "terrain": 42,
            "rainfall": 38, "traffic": 34, "historical": 18,
            "toll": 2300, "operations": 6400, "weather_severity": 2,
            "points": [[90,62],[120,76],[148,92],[177,111],[205,132],[231,148],[260,166]]
        },
        {
            "id": "B", "name": "Route B — NH15 / Hill Corridor", "distance": 315,
            "time_min": 465, "fuel_efficiency": 3.7, "road": 64,
            "accessibility": 71, "reliability": 63, "terrain": 72,
            "rainfall": 61, "traffic": 42, "historical": 49,
            "toll": 2500, "operations": 6500, "weather_severity": 4,
            "points": [[90,62],[122,57],[155,68],[187,81],[215,104],[238,126],[260,166]]
        },
        {
            "id": "C", "name": "Route C — NH17 / Alternate Link", "distance": 370,
            "time_min": 550, "fuel_efficiency": 4.2, "road": 75,
            "accessibility": 82, "reliability": 78, "terrain": 55,
            "rainfall": 45, "traffic": 27, "historical": 28,
            "toll": 1900, "operations": 6100, "weather_severity": 3,
            "points": [[90,62],[111,88],[145,115],[173,143],[205,164],[232,181],[260,166]]
        }
    ],
    "Guwahati-Shillong": [
        {
            "id":"A","name":"Route A — NH6 Main Corridor","distance":155,"time_min":230,
            "fuel_efficiency":4.2,"road":88,"accessibility":91,"reliability":90,"terrain":35,
            "rainfall":48,"traffic":45,"historical":17,"toll":900,"operations":3100,"weather_severity":2,
            "points":[[90,62],[116,83],[143,103],[172,119]]
        },
        {
            "id":"B","name":"Route B — Hill Bypass","distance":178,"time_min":255,
            "fuel_efficiency":3.8,"road":72,"accessibility":76,"reliability":70,"terrain":64,
            "rainfall":63,"traffic":30,"historical":34,"toll":750,"operations":3300,"weather_severity":4,
            "points":[[90,62],[103,75],[131,72],[151,94],[172,119]]
        },
        {
            "id":"C","name":"Route C — Scenic Alternate","distance":190,"time_min":285,
            "fuel_efficiency":4.0,"road":78,"accessibility":80,"reliability":79,"terrain":58,
            "rainfall":55,"traffic":25,"historical":25,"toll":600,"operations":3000,"weather_severity":3,
            "points":[[90,62],[107,94],[126,111],[150,127],[172,119]]
        }
    ],
    "Guwahati-Kohima": [
        {
            "id":"A","name":"Route A — NH2 / NH29","distance":330,"time_min":480,"fuel_efficiency":4.0,
            "road":80,"accessibility":84,"reliability":83,"terrain":51,"rainfall":44,"traffic":35,"historical":23,
            "toll":1800,"operations":6200,"weather_severity":2,
            "points":[[90,62],[117,72],[146,89],[178,108],[209,127],[245,137]]
        },
        {
            "id":"B","name":"Route B — Eastern Hill Link","distance":306,"time_min":455,"fuel_efficiency":3.6,
            "road":62,"accessibility":68,"reliability":61,"terrain":76,"rainfall":66,"traffic":38,"historical":52,
            "toll":2100,"operations":6400,"weather_severity":5,
            "points":[[90,62],[115,50],[149,59],[181,76],[214,104],[245,137]]
        },
        {
            "id":"C","name":"Route C — Central Alternate","distance":352,"time_min":520,"fuel_efficiency":4.1,
            "road":76,"accessibility":79,"reliability":76,"terrain":58,"rainfall":50,"traffic":29,"historical":30,
            "toll":1500,"operations":6000,"weather_severity":3,
            "points":[[90,62],[111,84],[140,111],[169,139],[205,154],[245,137]]
        }
    ],
    "Guwahati-Aizawl": [
        {"id":"A","name":"Route A — NH6 / Silchar Corridor","distance":500,"time_min":780,"fuel_efficiency":3.9,"road":74,"accessibility":78,"reliability":77,"terrain":57,"rainfall":62,"traffic":32,"historical":31,"toll":2800,"operations":8200,"weather_severity":4,"points":[[90,62],[113,90],[137,116],[164,141],[190,160],[222,178],[255,188]]},
        {"id":"B","name":"Route B — Southern Hill Route","distance":470,"time_min":760,"fuel_efficiency":3.5,"road":61,"accessibility":66,"reliability":60,"terrain":79,"rainfall":70,"traffic":27,"historical":48,"toll":2600,"operations":8500,"weather_severity":5,"points":[[90,62],[104,73],[134,81],[165,103],[193,133],[222,159],[255,188]]},
        {"id":"C","name":"Route C — Reliable Alternate","distance":530,"time_min":825,"fuel_efficiency":4.1,"road":80,"accessibility":83,"reliability":84,"terrain":48,"rainfall":51,"traffic":24,"historical":22,"toll":2400,"operations":7900,"weather_severity":3,"points":[[90,62],[110,103],[141,135],[175,157],[214,171],[255,188]]}
    ],
    "Guwahati-Agartala": [
        {"id":"A","name":"Route A — NH8 Main Corridor","distance":570,"time_min":840,"fuel_efficiency":4.0,"road":82,"accessibility":86,"reliability":85,"terrain":34,"rainfall":58,"traffic":40,"historical":19,"toll":3000,"operations":8600,"weather_severity":3,"points":[[90,62],[111,95],[139,126],[166,158],[198,189],[226,210],[250,230]]},
        {"id":"B","name":"Route B — Hill Alternate","distance":535,"time_min":815,"fuel_efficiency":3.6,"road":67,"accessibility":72,"reliability":68,"terrain":65,"rainfall":67,"traffic":30,"historical":40,"toll":2700,"operations":8700,"weather_severity":4,"points":[[90,62],[103,78],[129,89],[157,121],[185,151],[216,187],[250,230]]},
        {"id":"C","name":"Route C — Southern Link","distance":610,"time_min":900,"fuel_efficiency":4.2,"road":79,"accessibility":81,"reliability":80,"terrain":45,"rainfall":52,"traffic":22,"historical":24,"toll":2500,"operations":8300,"weather_severity":2,"points":[[90,62],[119,108],[150,148],[181,184],[214,214],[250,230]]}
    ],
    "Guwahati-Gangtok": [
        {"id":"A","name":"Route A — NH27 / NH10","distance":520,"time_min":820,"fuel_efficiency":3.9,"road":78,"accessibility":80,"reliability":79,"terrain":67,"rainfall":54,"traffic":34,"historical":35,"toll":2700,"operations":8200,"weather_severity":3,"points":[[90,62],[105,79],[130,91],[155,102],[180,119],[205,127]]},
        {"id":"B","name":"Route B — Mountain Corridor","distance":490,"time_min":790,"fuel_efficiency":3.5,"road":60,"accessibility":64,"reliability":59,"terrain":86,"rainfall":69,"traffic":28,"historical":55,"toll":2500,"operations":8500,"weather_severity":5,"points":[[90,62],[101,72],[124,73],[148,83],[171,98],[191,115],[205,127]]},
        {"id":"C","name":"Route C — Lower Risk Alternate","distance":545,"time_min":855,"fuel_efficiency":4.1,"road":84,"accessibility":87,"reliability":86,"terrain":52,"rainfall":47,"traffic":25,"historical":20,"toll":2300,"operations":7900,"weather_severity":2,"points":[[90,62],[111,101],[139,130],[166,139],[188,134],[205,127]]}
    ]
}

CITY_COORDS = {
    "Guwahati":[90,62],"Itanagar":[260,166],"Shillong":[172,119],"Kohima":[245,137],
    "Aizawl":[255,188],"Agartala":[250,230],"Gangtok":[205,127]
}

REGIONS = [
    {"name":"Assam","risk":42,"hazards":["Flooding","Heavy rain"],"accessibility":82,"disruptions":7},
    {"name":"Arunachal Pradesh","risk":67,"hazards":["Landslide","Steep terrain"],"accessibility":69,"disruptions":11},
    {"name":"Meghalaya","risk":54,"hazards":["Rainfall","Landslide"],"accessibility":76,"disruptions":8},
    {"name":"Manipur","risk":61,"hazards":["Road blockage","Terrain"],"accessibility":71,"disruptions":9},
    {"name":"Mizoram","risk":64,"hazards":["Landslide","Rainfall"],"accessibility":68,"disruptions":10},
    {"name":"Nagaland","risk":58,"hazards":["Terrain","Road condition"],"accessibility":73,"disruptions":8},
    {"name":"Tripura","risk":39,"hazards":["Flooding","Traffic"],"accessibility":84,"disruptions":5},
    {"name":"Sikkim","risk":63,"hazards":["Landslide","Snow/visibility"],"accessibility":70,"disruptions":9},
]

ALERTS = [
    {"id":1,"location":"Bomdila, Arunachal Pradesh","type":"Landslide","severity":"High","time":"18 min ago","impact":"Risk +14 on hill corridors","route":"Guwahati → Itanagar"},
    {"id":2,"location":"Guwahati outskirts, Assam","type":"Heavy rainfall","severity":"Moderate","time":"32 min ago","impact":"Visibility reduced","route":"Guwahati → Shillong"},
    {"id":3,"location":"Jowai, Meghalaya","type":"Road construction","severity":"Moderate","time":"1 hr ago","impact":"+25 min expected","route":"Guwahati → Shillong"},
    {"id":4,"location":"Dimapur, Nagaland","type":"Traffic congestion","severity":"Moderate","time":"46 min ago","impact":"+35 min expected","route":"Guwahati → Kohima"},
    {"id":5,"location":"Silchar corridor, Assam","type":"Flood","severity":"High","time":"2 hrs ago","impact":"Partial blockage risk","route":"Guwahati → Aizawl"},
]

VEHICLES = {
    "Mini Truck":{"eff":7.5,"max":2},
    "Light Commercial Vehicle":{"eff":6.5,"max":5},
    "Heavy Truck":{"eff":4.0,"max":20},
    "Container Truck":{"eff":3.2,"max":30},
    "Emergency Vehicle":{"eff":8.0,"max":5}
}


# Prototype fleet/incident/field-report data for the operational features.
FLEET = [
    {"id":"NER-MT-014","type":"Mini Truck","capacity_kg":1500,"driver":"B. Sharma","status":"ON TIME","updated":"5 min ago","lat":90,"lng":62},
    {"id":"NER-HT-002","type":"Heavy Truck","capacity_kg":15000,"driver":"R. Lyngdoh","status":"DELAYED","updated":"8 min ago","lat":172,"lng":119},
    {"id":"NER-4X-009","type":"4x4 Vehicle","capacity_kg":800,"driver":"T. Bhutia","status":"AT RISK","updated":"3 min ago","lat":205,"lng":127},
    {"id":"NER-PU-021","type":"Pickup","capacity_kg":1000,"driver":"K. Meitei","status":"ON TIME","updated":"4 min ago","lat":245,"lng":137},
    {"id":"NER-MT-031","type":"Mini Truck","capacity_kg":1500,"driver":"S. Debbarma","status":"DELIVERED","updated":"18 min ago","lat":250,"lng":230},
]

INCIDENTS = [
    {"id":1,"type":"Landslide","description":"Minor landslide debris on hill stretch; single-lane passable.","latitude":25.58,"longitude":91.89,"severity":"High","status":"Resolved","location":"Shillong corridor","reported":"18 min ago"},
    {"id":2,"type":"Flood","description":"Waterlogging reported near low-lying road section.","latitude":26.14,"longitude":91.74,"severity":"Medium","status":"Monitoring","location":"Guwahati outskirts","reported":"32 min ago"},
    {"id":3,"type":"Road blockage","description":"Temporary blockage due to fallen branches.","latitude":25.67,"longitude":94.11,"severity":"Medium","status":"Open","location":"Nagaland corridor","reported":"46 min ago"},
]

FIELD_REPORTS = [
    {"id":"FR-104","reporter":"Field Unit 07","location":"Bomdila","type":"Road condition","severity":"High","description":"Surface damage and debris observed after rainfall.","status":"Submitted","time":"20 min ago"},
    {"id":"FR-103","reporter":"Field Unit 03","location":"Jowai","type":"Traffic","severity":"Medium","description":"Slow-moving traffic near construction zone.","status":"Verified","time":"1 hr ago"},
]

EMERGENCY_CONTACTS = {
    "Assam":[
        {"category":"Emergency services (all-in-one)","name":"National Emergency Number","number":"112","demo":False},
        {"category":"Police","name":"Assam Police Control Room","number":"100","demo":True},
        {"category":"Ambulance","name":"National Ambulance Service","number":"108","demo":False},
        {"category":"Disaster management","name":"Assam State Disaster Management Authority","number":"1070","demo":True},
        {"category":"Highway/road authority","name":"National Highways Helpline","number":"1033","demo":True},
    ],
    "Arunachal Pradesh":[
        {"category":"Emergency services (all-in-one)","name":"National Emergency Number","number":"112","demo":False},
        {"category":"Police","name":"Arunachal Pradesh Police Control Room","number":"100","demo":True},
        {"category":"Ambulance","name":"National Ambulance Service","number":"108","demo":False},
        {"category":"Disaster management","name":"State Disaster Management Authority","number":"1070","demo":True},
    ],
    "Meghalaya":[
        {"category":"Emergency services (all-in-one)","name":"National Emergency Number","number":"112","demo":False},
        {"category":"Police","name":"Meghalaya Police Control Room","number":"100","demo":True},
        {"category":"Ambulance","name":"National Ambulance Service","number":"108","demo":False},
        {"category":"Disaster management","name":"State Disaster Management Authority","number":"1070","demo":True},
    ],
    "Manipur":[{"category":"Emergency services (all-in-one)","name":"National Emergency Number","number":"112","demo":False},{"category":"Police","name":"Manipur Police Control Room","number":"100","demo":True},{"category":"Ambulance","name":"National Ambulance Service","number":"108","demo":False}],
    "Mizoram":[{"category":"Emergency services (all-in-one)","name":"National Emergency Number","number":"112","demo":False},{"category":"Police","name":"Mizoram Police Control Room","number":"100","demo":True},{"category":"Ambulance","name":"National Ambulance Service","number":"108","demo":False}],
    "Nagaland":[{"category":"Emergency services (all-in-one)","name":"National Emergency Number","number":"112","demo":False},{"category":"Police","name":"Nagaland Police Control Room","number":"100","demo":True},{"category":"Ambulance","name":"National Ambulance Service","number":"108","demo":False}],
    "Tripura":[{"category":"Emergency services (all-in-one)","name":"National Emergency Number","number":"112","demo":False},{"category":"Police","name":"Tripura Police Control Room","number":"100","demo":True},{"category":"Ambulance","name":"National Ambulance Service","number":"108","demo":False}],
    "Sikkim":[{"category":"Emergency services (all-in-one)","name":"National Emergency Number","number":"112","demo":False},{"category":"Police","name":"Sikkim Police Control Room","number":"100","demo":True},{"category":"Ambulance","name":"National Ambulance Service","number":"108","demo":False}],
}

# Synthetic ML training data, deliberately reproducible.
rng = np.random.default_rng(42)
n = 900
X = pd.DataFrame({
    "rainfall": rng.uniform(0,100,n),
    "temperature": rng.uniform(8,38,n),
    "weather_severity": rng.integers(0,6,n),
    "road_quality": rng.uniform(20,100,n),
    "road_width": rng.uniform(3,12,n),
    "terrain_slope": rng.uniform(0,90,n),
    "elevation": rng.uniform(50,4000,n),
    "historical_disruptions": rng.uniform(0,100,n),
    "historical_landslides": rng.uniform(0,100,n),
    "historical_floods": rng.uniform(0,100,n),
    "traffic": rng.uniform(0,100,n),
    "accessibility": rng.uniform(20,100,n),
    "cargo_weight": rng.uniform(0.5,30,n)
})
y = (
    0.28*X.rainfall + 2.7*X.weather_severity + 0.18*(100-X.road_quality)
    + 0.22*X.terrain_slope + 0.14*X.historical_disruptions
    + 0.12*X.historical_landslides + 0.10*X.historical_floods
    + 0.10*X.traffic + 0.12*(100-X.accessibility)
    + 0.35*X.cargo_weight + rng.normal(0,4,n)
).clip(0,100)
MODEL_FEATURES = list(X.columns)
MODEL = RandomForestRegressor(n_estimators=120, random_state=42, max_depth=10)
MODEL.fit(X, y)

class AnalyzeRequest(BaseModel):
    source: str
    destination: str
    vehicle_type: str
    cargo_type: str
    cargo_weight: float = Field(ge=0)
    priority: str

class RecalculateRequest(BaseModel):
    routes: list[dict]
    disrupted_route_id: str

class IncidentRequest(BaseModel):
    incident_type: str
    description: str = ""
    latitude: float
    longitude: float
    severity: str

class FieldReportRequest(BaseModel):
    reporter: str
    location: str
    report_type: str
    severity: str
    description: str = ""

def get_routes(source, destination):
    key = f"{source}-{destination}"
    if key in ROUTES:
        return ROUTES[key]
    # A useful fallback: scale the six demo scenarios to unknown city names.
    base = ROUTES["Guwahati-Itanagar"]
    return [dict(r) for r in base]

def risk_prediction(route, weight):
    row = pd.DataFrame([{
        "rainfall": route["rainfall"],
        "temperature": 24,
        "weather_severity": route["weather_severity"],
        "road_quality": route["road"],
        "road_width": max(4, route["road"]/10),
        "terrain_slope": route["terrain"],
        "elevation": 500 + route["terrain"]*20,
        "historical_disruptions": route["historical"],
        "historical_landslides": route["historical"]*0.9,
        "historical_floods": route["rainfall"]*0.65,
        "traffic": route["traffic"],
        "accessibility": route["accessibility"],
        "cargo_weight": weight
    }])
    score = float(np.clip(MODEL.predict(row)[0],0,100))
    contributors = {
        "Heavy rainfall": round(route["rainfall"]*0.20,1),
        "Steep terrain": round(route["terrain"]*0.16,1),
        "Poor road condition": round((100-route["road"])*0.20,1),
        "Historical disruptions": round(route["historical"]*0.12,1),
        "Traffic": round(route["traffic"]*0.08,1)
    }
    # Ensure visible contribution percentages sum to 100 for the UI.
    labels = ["Weather","Road condition","Terrain","Historical disruption","Traffic","Other"]
    vals = [round(route["rainfall"]*0.25,1), round((100-route["road"])*0.20,1),
            round(route["terrain"]*0.18,1), round(route["historical"]*0.15,1),
            round(route["traffic"]*0.10,1), 0]
    total = sum(vals[:5])
    vals[5] = round(max(0,100-total),1)
    return score, contributors, dict(zip(labels,vals))

def risk_category(score):
    if score <= 30: return "Low"
    if score <= 60: return "Moderate"
    if score <= 80: return "High"
    return "Critical"

def cost(route, vehicle, weight):
    v = VEHICLES.get(vehicle, VEHICLES["Heavy Truck"])
    fuel_price = 96
    liters = route["distance"] / v["eff"]
    fuel = liters * fuel_price
    ops = route["operations"] + max(0,weight-5)*180
    total = fuel + route["toll"] + ops
    return {"fuel":round(fuel),"tolls":route["toll"],"operations":round(ops),"total":round(total)}

def normalize(vals, invert=False):
    a=np.array(vals,dtype=float)
    lo,hi=a.min(),a.max()
    if hi-lo < 1e-9: out=np.ones(len(a))*0.5
    else: out=(a-lo)/(hi-lo)
    return 1-out if invert else out

def analyze_routes(req):
    if not req.source or not req.destination:
        raise HTTPException(400,"Source and destination are required.")
    if req.vehicle_type not in VEHICLES:
        raise HTTPException(400,"Unknown vehicle type.")
    if req.cargo_weight > VEHICLES[req.vehicle_type]["max"]:
        raise HTTPException(400,f"{req.vehicle_type} supports up to {VEHICLES[req.vehicle_type]['max']} tonnes in this prototype.")
    routes = get_routes(req.source.strip(), req.destination.strip())
    if req.priority not in ["Fastest","Cheapest","Safest","Balanced"]:
        raise HTTPException(400,"Invalid priority.")

    scored=[]
    for r in routes:
        risk, contrib, breakdown = risk_prediction(r,req.cargo_weight)
        c=cost(r,req.vehicle_type,req.cargo_weight)
        scored.append({**r,"risk":round(risk,1),"risk_category":risk_category(risk),
                       "cost":c,"accessibility":r["accessibility"],"reliability":r["reliability"],
                       "contributors":contrib,"breakdown":breakdown})

    times=normalize([r["time_min"] for r in scored],invert=True)
    costs=normalize([r["cost"]["total"] for r in scored],invert=True)
    risks=normalize([r["risk"] for r in scored],invert=True)
    access=normalize([r["accessibility"] for r in scored])
    rel=normalize([r["reliability"] for r in scored])
    weights={
        "Fastest":[.50,.15,.20,.10,.05],
        "Cheapest":[.20,.50,.15,.10,.05],
        "Safest":[.10,.05,.50,.15,.20],
        "Balanced":[.25,.20,.30,.15,.10]
    }[req.priority]
    for i,r in enumerate(scored):
        utility=weights[0]*times[i]+weights[1]*costs[i]+weights[2]*risks[i]+weights[3]*access[i]+weights[4]*rel[i]
        r["route_score"]=round(utility*100,1)
        r["estimated_time"]=f"{r['time_min']//60}h {r['time_min']%60:02d}m"
        r["weather"]={"rainfall":r["rainfall"],"severity":r["weather_severity"],
                      "condition":"Heavy rain" if r["weather_severity"]>=4 else "Cloudy / intermittent rain"}
        r["points"]=r["points"]
    scored.sort(key=lambda x:x["route_score"],reverse=True)
    rec=scored[0]
    reasons=[]
    if rec["risk"] <= min(r["risk"] for r in scored)+5: reasons.append("low predicted disruption risk")
    if rec["accessibility"] >= max(r["accessibility"] for r in scored)-5: reasons.append("strong road accessibility")
    if rec["reliability"] >= max(r["reliability"] for r in scored)-7: reasons.append("high route reliability")
    reasons.append("an acceptable travel-time and cost trade-off")
    explanation=f"{rec['name']} is recommended because it has {', '.join(reasons)}. "
    faster=min(scored,key=lambda r:r["time_min"])
    if faster["id"]!=rec["id"]:
        explanation+=f"{faster['name']} is faster, but its predicted risk is {faster['risk']} compared with {rec['risk']} on the recommended route."
    return {"recommended":rec,"routes":scored,"explanation":explanation,
            "source":req.source,"destination":req.destination,"priority":req.priority}

@app.get("/api/vehicles")
def vehicles():
    return FLEET

@app.get("/api/incidents")
def incidents():
    return INCIDENTS

@app.post("/api/incidents")
def create_incident(req: IncidentRequest):
    item={"id":max([x["id"] for x in INCIDENTS],default=0)+1,"type":req.incident_type,"description":req.description,"latitude":req.latitude,"longitude":req.longitude,"severity":req.severity,"status":"Open","location":f"{req.latitude:.2f}, {req.longitude:.2f}","reported":"just now"}
    INCIDENTS.insert(0,item)
    ALERTS.insert(0,{"id":100+item["id"],"location":item["location"],"type":item["type"],"severity":item["severity"],"time":"just now","impact":"New field incident submitted","route":"Route impact pending assessment"})
    return item

@app.get("/api/field-reports")
def field_reports():
    return FIELD_REPORTS

@app.post("/api/field-reports")
def create_field_report(req: FieldReportRequest):
    item={"id":f"FR-{100+len(FIELD_REPORTS)+1}","reporter":req.reporter,"location":req.location,"type":req.report_type,"severity":req.severity,"description":req.description,"status":"Submitted","time":"just now"}
    FIELD_REPORTS.insert(0,item)
    return item

@app.get("/api/emergency/{state}")
def emergency(state: str):
    return EMERGENCY_CONTACTS.get(state, EMERGENCY_CONTACTS["Assam"])

@app.get("/api/health")
def health(): return {"status":"ok","ml":"RandomForestRegressor","data":"synthetic-demo"}

@app.get("/api/routes")
def routes(): return {"routes":list(ROUTES.keys())}

@app.get("/api/weather")
def weather():
    return {"regions":[{"location":r["name"],"rainfall":r["risk"]+10,"temperature":24,"condition":"Intermittent rain","severity":round(r["risk"]/25)} for r in REGIONS]}

@app.get("/api/alerts")
def alerts(): return ALERTS

@app.get("/api/regions")
def regions(): return REGIONS

@app.get("/api/analytics")
def analytics():
    return {"average_risk":52,"average_cost":18400,"reliability":79,"disruptions":67,
            "weather_disruptions":38,"routes_analyzed":1248,"cost_savings":14.8,
            "risk_trend":[42,46,44,51,48,56,52],
            "disruption_chart":[7,11,9,13,8,10,6],
            "cost_comparison":[18400,20100,17900,19200,17600],
            "reliability_chart":[89,63,78,85,71]}

@app.post("/api/risk/predict")
def predict(req: AnalyzeRequest):
    routes=get_routes(req.source,req.destination)
    risk, contributors, breakdown=risk_prediction(routes[0],req.cargo_weight)
    return {"risk_score":round(risk,1),"category":risk_category(risk),
            "main_risk_factors":contributors,"breakdown":breakdown,"confidence":round(82+min(12,req.cargo_weight*0.3),1)}

@app.post("/api/cost/calculate")
def calculate_cost(req: AnalyzeRequest):
    return cost(get_routes(req.source,req.destination)[0],req.vehicle_type,req.cargo_weight)

@app.post("/api/routes/analyze")
def analyze(req: AnalyzeRequest):
    return analyze_routes(req)

@app.post("/api/routes/recalculate")
def recalculate(req: RecalculateRequest):
    routes=[dict(r) for r in req.routes]
    for r in routes:
        if r["id"]==req.disrupted_route_id:
            r["risk"]=min(100,round(r["risk"]+26,1))
            r["risk_category"]=risk_category(r["risk"])
            r["reliability"]=max(0,r["reliability"]-18)
    routes.sort(key=lambda x:(x["risk"],-x["reliability"]))
    new=routes[0]
    return {"routes":routes,"recommended":new,
            "message":f"Alternative {new['id']} is recommended because the previous route now has elevated disruption risk.",
            "additional_time":25,"additional_cost":850}
