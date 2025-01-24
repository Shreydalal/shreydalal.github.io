import React from 'react';
import { Brain, Code2, Database, Github, Linkedin, Mail, ScrollText } from 'lucide-react';

function App() {
  const projects = [
    {
      title: "Customer Churn Prediction",
      description: "Developed a machine learning model to predict customer churn with 94% accuracy using XGBoost",
      tags: ["Python", "Scikit-learn", "XGBoost", "Pandas"],
      image: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=2000"
    },
    {
      title: "Natural Language Processing Pipeline",
      description: "Built an end-to-end NLP pipeline for sentiment analysis on customer reviews",
      tags: ["PyTorch", "BERT", "NLP", "Docker"],
      image: "https://images.unsplash.com/photo-1555421689-491a97ff2040?auto=format&fit=crop&q=80&w=2000"
    },
    {
      title: "Time Series Forecasting",
      description: "Implemented deep learning models for stock price prediction using LSTM networks",
      tags: ["TensorFlow", "LSTM", "Time Series", "Neural Networks"],
      image: "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&q=80&w=2000"
    }
  ];

  const skills = [
    { category: "Languages", items: ["Python", "R", "SQL", "Julia"] },
    { category: "ML/DL", items: ["TensorFlow", "PyTorch", "Scikit-learn", "Keras"] },
    { category: "Big Data", items: ["Spark", "Hadoop", "MongoDB", "PostgreSQL"] },
    { category: "Tools", items: ["Docker", "Git", "AWS", "Kubernetes"] }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <header className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white">
        <nav className="container mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <Brain className="w-8 h-8" />
            <span className="text-xl font-bold">DataSci Portfolio</span>
          </div>
          <div className="flex space-x-4">
            <a href="https://github.com" target="_blank" rel="noopener noreferrer">
              <Github className="w-6 h-6 hover:text-blue-200" />
            </a>
            <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer">
              <Linkedin className="w-6 h-6 hover:text-blue-200" />
            </a>
            <a href="mailto:contact@example.com">
              <Mail className="w-6 h-6 hover:text-blue-200" />
            </a>
          </div>
        </nav>
        
        <div className="container mx-auto px-6 py-20">
          <h1 className="text-5xl font-bold mb-4">Data Scientist & ML Engineer</h1>
          <p className="text-xl mb-8">Transforming data into actionable insights through machine learning</p>
          <button className="bg-white text-blue-600 px-6 py-3 rounded-lg font-semibold hover:bg-blue-50 transition-colors">
            View Projects
          </button>
        </div>
      </header>

      {/* Projects Section */}
      <section className="py-20 container mx-auto px-6">
        <h2 className="text-3xl font-bold mb-12 text-center">Featured Projects</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project, index) => (
            <div key={index} className="bg-white rounded-lg overflow-hidden shadow-lg hover:shadow-xl transition-shadow">
              <img src={project.image} alt={project.title} className="w-full h-48 object-cover" />
              <div className="p-6">
                <h3 className="text-xl font-bold mb-2">{project.title}</h3>
                <p className="text-gray-600 mb-4">{project.description}</p>
                <div className="flex flex-wrap gap-2">
                  {project.tags.map((tag, tagIndex) => (
                    <span key={tagIndex} className="bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Skills Section */}
      <section className="bg-gray-100 py-20">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold mb-12 text-center">Technical Skills</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {skills.map((skillGroup, index) => (
              <div key={index} className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-xl font-bold mb-4 flex items-center">
                  {index === 0 && <Code2 className="w-5 h-5 mr-2" />}
                  {index === 1 && <Brain className="w-5 h-5 mr-2" />}
                  {index === 2 && <Database className="w-5 h-5 mr-2" />}
                  {index === 3 && <ScrollText className="w-5 h-5 mr-2" />}
                  {skillGroup.category}
                </h3>
                <ul className="space-y-2">
                  {skillGroup.items.map((item, itemIndex) => (
                    <li key={itemIndex} className="text-gray-600">{item}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8">
        <div className="container mx-auto px-6 text-center">
          <p>© 2024 Data Science Portfolio. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;