
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# CORS Configuration
CORS(
    app,
    resources={r"/recommend": {"origins": "*"}},
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

# Career and Skills Dataset
data = {
    "Career": [
        "Frontend Developer",
        "Backend Developer",
        "DevOps Engineer",
        "AI Engineer",
        "Data Analyst",
        "Android Developer",
        "iOS Developer",
        "Blockchain Developer",
        "QA Engineer",
        "Software Architect",
        "Cybersecurity Specialist",
        "UX Designer",
        "Game Developer",
        "Technical Writer",
        "MLOps Engineer",
        "Product Manager",
        "Engineering Manager",
        "Developer Relations Specialist",
        "Data Scientist",
        "Graphic Designer",
        "Full Stack Developer"
    ],

    "Skills": [
        "HTML, CSS, JavaScript, React, Vue, Angular, TypeScript, Responsive Design",
        "Node.js, Express, Database Management, RESTful APIs, Authentication, Docker, Kubernetes",
        "Continuous Integration, Continuous Deployment, Infrastructure as Code, Monitoring, Cloud Platforms, Containerization",
        "Machine Learning, Deep Learning, Neural Networks, TensorFlow, PyTorch, Natural Language Processing",
        "Data Cleaning, Data Visualization, Statistical Analysis, SQL, Excel, Python, R",
        "Java, Kotlin, Android SDK, Jetpack, Material Design, Firebase",
        "Swift, Objective-C, iOS SDK, SwiftUI, Core Data, Combine Framework",
        "Smart Contracts, Solidity, Ethereum, Decentralized Applications, Cryptography",
        "Test Planning, Automated Testing, Manual Testing, Selenium, JUnit, Bug Tracking",
        "System Design, Architectural Patterns, Scalability, Microservices, API Design",
        "Network Security, Ethical Hacking, Penetration Testing, Firewalls, Encryption, Compliance Standards",
        "User Research, Wireframing, Prototyping, Interaction Design, Usability Testing, Figma, Sketch",
        "Unity, Unreal Engine, C#, Game Physics, 3D Modeling, Animation",
        "Technical Writing, Documentation, API Writing, Markdown, Content Strategy",
        "Model Deployment, Continuous Integration for ML, Monitoring ML Models, Kubernetes, Docker",
        "Market Research, Product Lifecycle Management, Agile Methodologies, Roadmapping, Stakeholder Communication",
        "Team Leadership, Project Management, Agile Practices, Mentoring, Performance Reviews",
        "Community Engagement, Public Speaking, Content Creation, Developer Advocacy, Social Media Management",
        "Python, Machine Learning, SQL, Statistics, Deep Learning",
        "Photoshop, Illustrator, UI/UX, Typography, Branding",
        "Html, Css, Javascript, Angular, React, GitHub, npm, Tailwind css, node.js, REST apis, sql"
    ]
}

# Convert dataset into DataFrame
df = pd.DataFrame(data)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x.split(', '),
    token_pattern=None
)

X = vectorizer.fit_transform(df["Skills"])

# KNN Model using Cosine Distance
knn = NearestNeighbors(
    n_neighbors=1,
    metric='cosine'
)

knn.fit(X)


# Career Recommendation API
@app.route('/recommend', methods=['POST'])
def recommend_career():

    try:
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON input."}), 400

        user_skills = data.get("skills", [])

        if not user_skills or not isinstance(user_skills, list):
            return jsonify({
                "error": "Invalid input. Expected a list of skills."
            }), 400

        # Normalize user skills
        user_skills = [
            skill.strip().lower()
            for skill in user_skills
            if isinstance(skill, str) and skill.strip()
        ]

        # Get all known skills
        all_known_skills = set()

        for skill_list in df["Skills"]:
            skills = [
                skill.strip().lower()
                for skill in skill_list.split(",")
            ]
            all_known_skills.update(skills)

        # Validate user skills
        valid_skills = [
            skill for skill in user_skills
            if skill in all_known_skills
        ]

        if not valid_skills:
            return jsonify({
                "error": "No matching skills found. Please enter valid skills."
            }), 400

        # Convert user skills into TF-IDF vector
        user_vector = vectorizer.transform(
            [", ".join(valid_skills)]
        )

        # Find nearest career using cosine distance
        distances, indices = knn.kneighbors(user_vector)

        career_index = indices[0][0]

        recommended_career = df.iloc[career_index]["Career"]

        # Identify missing skills
        required_skills = set(
            skill.strip().lower()
            for skill in df.iloc[career_index]["Skills"].split(",")
        )

        missing_skills = sorted(
            required_skills - set(valid_skills)
        )

        return jsonify({
            "career": recommended_career,
            "missing_skills": missing_skills if missing_skills else [
                "None! You are fully qualified!"
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run Flask Application locally
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
