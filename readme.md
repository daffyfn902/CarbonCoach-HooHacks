#  CarbonCoach

CarbonCoach is an AI-powered sustainability assistant designed to help individuals track, understand, and reduce their environmental footprint. By leveraging machine learning and generative AI, the platform provides personalized insights and actionable coaching to promote a lower-carbon lifestyle.

---

##  Features

- **Personalized Carbon Tracking**  
  Analyze lifestyle habits to calculate an estimated carbon footprint.

- **AI-Driven Insights**  
  Uses Large Language Models to provide context-aware suggestions for reducing emissions.

- **Predictive Modeling**  
  Utilizes Scikit-Learn to categorize user impact and predict future trends based on behavioral data.

- **Interactive Dashboard**  
  A clean, Flask-based web interface for managing sustainability goals.

---

##  Tech Stack

- **Backend:** Flask 3.1.3  
- **Machine Learning:** Scikit-Learn 1.8.0, Joblib 1.5.3, Pandas 2.3.2, SciPy 1.17.1  
- **Generative AI:** Google GenAI 1.68.0 (Gemini)  
- **Environment Management:** Python-Dotenv 1.2.2  
- **Production Server:** Gunicorn  

---

##  Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.9 or higher  
- A Google Cloud API Key (for Gemini AI features)  

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/daffyfn902/carboncoach.git
cd carboncoach
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Mac/Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add:

```env
GOOGLE_API_KEY=your_api_key_here
FLASK_ENV=development
```

---

##  How to Run

### Development Mode

Run the app locally with hot reloading:

```bash
python app.py
```

The application will be available at:  
http://127.0.0.1:5000

---



## 🤝 Contributors

Evan Khanna
<br>
Nathan Biru

---

