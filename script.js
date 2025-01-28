// Data for projects
const projects = [
  {
    title: "Customer Churn Prediction",
    description: "Developed a machine learning model to predict customer churn with 94% accuracy using XGBoost",
    tags: ["Python", "Scikit-learn", "XGBoost", "Pandas"],
    image: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=2000",
    link: "https://github.com/yourusername/customer-churn-prediction" // Add project link
  },
  {
    title: "Natural Language Processing Pipeline",
    description: "Built an end-to-end NLP pipeline for sentiment analysis on customer reviews",
    tags: ["PyTorch", "BERT", "NLP", "Docker"],
    image: "https://images.unsplash.com/photo-1555421689-491a97ff2040?auto=format&fit=crop&q=80&w=2000",
    link: "https://github.com/yourusername/nlp-pipeline" // Add project link
  },
  {
    title: "Time Series Forecasting",
    description: "Implemented deep learning models for stock price prediction using LSTM networks",
    tags: ["TensorFlow", "LSTM", "Time Series", "Neural Networks"],
    image: "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&q=80&w=2000",
    link: "https://github.com/yourusername/time-series-forecasting" // Add project link
  }
];

// Data for skills
const skills = [
  { category: "Languages", items: ["Python", "R", "SQL", "Julia"] },
  { category: "ML/DL", items: ["TensorFlow", "PyTorch", "Scikit-learn", "Keras"] },
  { category: "Big Data", items: ["Spark", "Hadoop", "MongoDB", "PostgreSQL"] },
  { category: "Tools", items: ["Docker", "Git", "AWS", "Kubernetes"] }
];

// Render projects
const projectContainer = document.getElementById("projects");
projects.forEach(project => {
  const projectCard = document.createElement("div");
  projectCard.className = "bg-white rounded-lg overflow-hidden shadow-lg hover:shadow-xl transition-shadow";

  projectCard.innerHTML = `
    <a href="${project.link}" target="_blank">
      <img src="${project.image}" alt="${project.title}" class="w-full h-48 object-cover" />
      <div class="p-6">
        <h3 class="text-xl font-bold mb-2">${project.title}</h3>
        <p class="text-gray-600 mb-4">${project.description}</p>
        <div class="flex flex-wrap gap-2">
          ${project.tags.map(tag => `<span class="bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full">${tag}</span>`).join("")}
        </div>
      </div>
    </a>
  `;
  projectContainer.appendChild(projectCard);
});

// Render skills
const skillsContainer = document.getElementById("skills");
skills.forEach(skillGroup => {
  const skillCard = document.createElement("div");
  skillCard.className = "bg-white p-6 rounded-lg shadow-md";

  skillCard.innerHTML = `
    <h3 class="text-xl font-bold mb-4">${skillGroup.category}</h3>
    <ul class="space-y-2">
      ${skillGroup.items.map(item => `<li class="text-gray-600">${item}</li>`).join("")}
    </ul>
  `;
  skillsContainer.appendChild(skillCard);
});

// Make "View Projects" button functional
document.querySelector('button').addEventListener('click', () => {
  document.getElementById('projects').scrollIntoView({ behavior: 'smooth' });
});