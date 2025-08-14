# 🚀 Deploying a Simple Machine Learning Model on a Kubernetes Homelab

## 📌 Introduction

This project demonstrates **how to deploy a simple Machine Learning model** into a Kubernetes cluster running in a **homelab** environment.
It’s designed to be beginner-friendly while introducing the main components of **ML infrastructure**: containerization, API serving, Kubernetes deployments, scaling, and monitoring.

We’ll take a **scikit-learn** model (Iris classifier), serve it with **FastAPI**, package it in a Docker image, and deploy it to our cluster with **MetalLB** for external access and **Prometheus** metrics for observability.

---

## 🖥️ Environment

This tutorial was built and tested in the following environment:

* **Kubernetes Cluster:** 3-node cluster in a homelab setup (1 control plane + 2 workers)
* **Cluster Type:** Bare-metal, virtualized via Proxmox VMs
* **Kubernetes Version:** v1.30+
* **Networking:** Calico CNI
* **Load Balancer:** [MetalLB](https://metallb.universe.tf/) for assigning external IPs
* **Ingress / Reverse Proxy:** Caddy (running outside Kubernetes)
* **Monitoring Stack:** Prometheus + Grafana via [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
* **Container Registry:** Docker Hub

---

## 🎯 Learning Goals

By the end of this guide, you will:

* Package an ML model as a Docker image
* Serve it via a production-ready Python web framework (FastAPI)
* Deploy to Kubernetes with **readiness/liveness probes** and **autoscaling**
* Expose it externally with **MetalLB**
* Export metrics for **Prometheus & Grafana**
* Understand the fundamentals of **ML model serving in production**

---

## 1️⃣ Project Structure

```
iris-svc/
├── app.py             # FastAPI app with /predict, /healthz, /metrics
├── train.py           # Script to train and save the ML model
├── requirements.txt   # Python dependencies
├── Dockerfile         # Image build instructions
├── k8s.yaml           # Deployment, Service, and HPA manifests
└── README.md          # This guide
```

---

## 2️⃣ Training the Model

We start with a **very simple model**:
A **Random Forest classifier** trained on the famous [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris).

Why train at build time?

* Keeps the example simple (no need for separate storage or pipelines)
* Produces a **self-contained image** — portable and reproducible

`train.py`:

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

joblib.dump({"model": clf, "target_names": iris.target_names.tolist()}, "model.joblib")
```

---

## 3️⃣ Serving the Model with FastAPI

We use **FastAPI** because:

* It’s fast and async-friendly
* Automatic OpenAPI docs
* Easy integration with **pydantic** for input validation

`app.py`:

```python
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI(title="Iris Classifier", version="0.1.0")
bundle = joblib.load("model.joblib")
model = bundle["model"]
target_names = bundle["target_names"]

PREDICTIONS = Counter("predictions_total", "Number of predictions served")

class Features(BaseModel):
    x: list[float] = Field(..., min_items=4, max_items=4)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/predict")
def predict(f: Features):
    pred = model.predict([f.x])[0]
    PREDICTIONS.inc()
    return {"class_index": int(pred), "class_name": target_names[int(pred)]}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

Endpoints:

* `/healthz` → readiness/liveness probes
* `/predict` → ML inference
* `/metrics` → Prometheus-compatible metrics

---

## 4️⃣ Containerizing the Service

`Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN useradd -m appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
RUN python train.py

COPY app.py .

USER appuser
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build & push:

```bash
docker build -t <dockerhub-username>/iris-svc:0.1.0 .
docker push <dockerhub-username>/iris-svc:0.1.0
```

---

## 5️⃣ Kubernetes Deployment

We create:

* **Namespace**: isolates ML workloads
* **Deployment**: manages Pods, with readiness/liveness probes
* **Service**: type `LoadBalancer` for MetalLB external IP
* **HPA**: scales based on CPU usage

`k8s.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-svc
  namespace: ml
spec:
  replicas: 1
  selector:
    matchLabels:
      app: iris-svc
  template:
    metadata:
      labels:
        app: iris-svc
    spec:
      containers:
        - name: iris-svc
          image: <dockerhub-username>/iris-svc:0.1.0
          ports:
            - name: http
              containerPort: 8000
          readinessProbe:
            httpGet:
              path: /healthz
              port: http
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
---
apiVersion: v1
kind: Service
metadata:
  name: iris-svc
  namespace: ml
spec:
  type: LoadBalancer
  selector:
    app: iris-svc
  ports:
    - name: http
      port: 80
      targetPort: http
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: iris-svc
  namespace: ml
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: iris-svc
  minReplicas: 1
  maxReplicas: 5
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
```

Apply:

```bash
kubectl apply -f k8s.yaml
kubectl -n ml get pods,svc,hpa
```

---

## 6️⃣ Testing the API

Get the external IP:

```bash
kubectl -n ml get svc iris-svc
```

Test:

```bash
curl http://EXTERNAL_IP/healthz

curl -X POST http://EXTERNAL_IP/predict \
  -H 'Content-Type: application/json' \
  -d '{"x":[5.1,3.5,1.4,0.2]}'

curl http://EXTERNAL_IP/metrics
```

---

## 7️⃣ Adding to Prometheus & Grafana

To monitor predictions in Grafana:

* Create a **ServiceMonitor** (if using kube-prometheus-stack)
* Add a panel for the metric `predictions_total`

Example `iris-servicemonitor.yaml`:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: iris-svc
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: iris-svc
  namespaceSelector:
    matchNames: ["ml"]
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

---

## 📚 Key Takeaways

* **Kubernetes** isn’t just for web apps — it’s perfect for ML inference services.
* A good ML service includes:

  * **Health probes** for resilience
  * **Metrics** for observability
  * **Autoscaling** for performance
* **MetalLB** bridges the gap between cloud load balancers and bare-metal clusters.
* **FastAPI** + **scikit-learn** + **Docker** is a great starting stack for MLOps learning.

---

## 🚀 Next Steps

* Replace the static Iris model with a custom-trained model.
* Use **KServe** or **Seldon Core** for advanced ML serving patterns.
* Add **GitOps** with Argo CD for automated deployments.
* Introduce a CI/CD pipeline to rebuild and redeploy when the model changes.
* Experiment with GPU acceleration and batching for larger models.

---

**Author:** *Patrick Bashizi*
**License:** MIT
